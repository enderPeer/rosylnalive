namespace Lrm.Cluster

open System
open System.Buffers.Binary
open System.Collections.Concurrent
open System.Collections.Generic
open System.Net
open System.Net.Sockets
open System.Threading
open System.Threading.Tasks

/// Counters of a ChaosProxy.
type ProxyStats =
    {
        Connections: int64
        Forwarded: int64
        Dropped: int64
        Duplicated: int64
        Severed: int64
    }

[<Sealed; AllowNullLiteral>]
type internal ProxyPair(client: TcpClient, upstream: TcpClient) =
    let mutable closed = 0
    member _.Client = client
    member _.Upstream = upstream
    /// The node id, learned from the Sender of the first frame the node sends (0 until then).
    member val Node = 0UL with get, set

    member _.Close() =
        if Interlocked.Exchange(&closed, 1) = 0 then
            try
                client.Close()
            with _ ->
                ()

            try
                upstream.Close()
            with _ ->
                ()

    member _.IsClosed = Volatile.Read(&closed) = 1

/// A frame-aware TCP chaos proxy (ClusterProtocol contract, "ChaosProxy"): nodes connect to it instead of the
/// coordinator. It parses whole frames in both directions and, per frame, applies the policy's fate (drop,
/// duplicate, delay, reorder) on link (node, 0) or (0, node), drops every frame of a partitioned node, cuts a node's
/// connection at each Sever time, and from a Kill until the next Restart keeps cutting the node off. Frames are
/// forwarded byte for byte; their trailers stay valid. Fault times count from the proxy's creation.
[<Sealed>]
type ChaosProxy(listen: IPEndPoint, target: IPEndPoint, policy: ChaosPolicy, clock: IClock) =
    let listener = new TcpListener(listen)
    do listener.Start()
    let start = clock.NowNs
    let cts = new CancellationTokenSource()
    let pairs = ConcurrentDictionary<ProxyPair, byte>(HashIdentity.Reference)
    let appliedSevers = HashSet<struct (uint64 * int64)>()
    let mutable connections = 0L
    let mutable forwarded = 0L
    let mutable dropped = 0L
    let mutable duplicated = 0L
    let mutable severed = 0L

    let elapsed () = clock.NowNs - start

    /// True when the node is inside a Kill window (killed and not yet restarted) at proxy time `t`.
    let killed (node: uint64) (t: int64) =
        let kills =
            policy.Faults
            |> List.choose (function
                | Fault.Kill(n, at) when n = node && at <= t -> Some at
                | _ -> None)

        match kills with
        | [] -> false
        | _ ->
            let lastKill = List.max kills

            not (
                policy.Faults
                |> List.exists (function
                    | Fault.Restart(n, at) -> n = node && at > lastKill && at <= t
                    | _ -> false)
            )

    let sever (pair: ProxyPair) =
        if not pair.IsClosed then
            Interlocked.Increment(&severed) |> ignore
            pair.Close()

    let checkFaults () =
        let t = elapsed ()

        for pair in List.ofSeq pairs.Keys do
            let node = pair.Node

            if node <> 0UL then
                if killed node t then
                    sever pair
                else
                    for f in policy.Faults do
                        match f with
                        | Fault.Sever(n, at) when n = node && at <= t ->
                            let fresh = lock appliedSevers (fun () -> appliedSevers.Add(struct (n, at)))

                            if fresh then
                                sever pair
                        | _ -> ()

    let pump (pair: ProxyPair) (fromNode: bool) (input: NetworkStream) (output: DelayQueue<byte[]>) =
        Task.Run(fun () ->
            task {
                try
                    let mutable go = true

                    while go && not pair.IsClosed do
                        let! next = FrameStream.readAsync input cts.Token

                        match next with
                        | None -> go <- false
                        | Some bytes ->
                            if fromNode && pair.Node = 0UL then
                                pair.Node <- BinaryPrimitives.ReadUInt64LittleEndian(ReadOnlySpan(bytes, 24, 8))
                                checkFaults ()

                            let node = pair.Node
                            let src, dst = if fromNode then node, 0UL else 0UL, node
                            let now = clock.NowNs
                            let t = now - start

                            let fate =
                                if node <> 0UL && killed node t then
                                    { Copies = [] }
                                else
                                    lock policy (fun () -> policy.Decide(src, dst, t))

                            match fate.Copies with
                            | [] -> Interlocked.Increment(&dropped) |> ignore
                            | copies ->
                                if copies.Length > 1 then
                                    Interlocked.Increment(&duplicated) |> ignore

                                for d in copies do
                                    Interlocked.Increment(&forwarded) |> ignore
                                    output.Enqueue(bytes, now + d)
                with _ ->
                    ()

                pair.Close()
            }
            :> Task)
        |> ignore

    /// The port nodes connect to.
    member _.Port = (listener.LocalEndpoint :?> IPEndPoint).Port

    /// The counters so far.
    member _.Stats =
        {
            Connections = Interlocked.Read(&connections)
            Forwarded = Interlocked.Read(&forwarded)
            Dropped = Interlocked.Read(&dropped)
            Duplicated = Interlocked.Read(&duplicated)
            Severed = Interlocked.Read(&severed)
        }

    /// Starts accepting connections and checking Sever and Kill faults every 20 ms.
    member _.Start() =
        let token = cts.Token

        Task.Run(fun () ->
            task {
                try
                    while not token.IsCancellationRequested do
                        do! Task.Delay(20, token)
                        checkFaults ()
                with _ ->
                    ()
            }
            :> Task)
        |> ignore

        Task.Run(fun () ->
            task {
                try
                    while not token.IsCancellationRequested do
                        let! client = listener.AcceptTcpClientAsync(token)
                        client.NoDelay <- true
                        let upstream = new TcpClient(NoDelay = true)

                        try
                            do! upstream.ConnectAsync(target, token)
                            Interlocked.Increment(&connections) |> ignore
                            let pair = ProxyPair(client, upstream)
                            pairs[pair] <- 0uy
                            let toUpstream = upstream.GetStream()
                            let toClient = client.GetStream()

                            let writer (stream: NetworkStream) =
                                new DelayQueue<byte[]>(
                                    clock,
                                    fun bytes ->
                                        if not pair.IsClosed then
                                            try
                                                stream.Write(bytes, 0, bytes.Length)
                                            with _ ->
                                                pair.Close()
                                )

                            let up = writer toUpstream
                            let down = writer toClient
                            pump pair true (client.GetStream()) up
                            pump pair false (upstream.GetStream()) down

                            Task.Run(fun () ->
                                task {
                                    while not pair.IsClosed && not token.IsCancellationRequested do
                                        do! Task.Delay(50)

                                    (up :> IDisposable).Dispose()
                                    (down :> IDisposable).Dispose()
                                    pairs.TryRemove pair |> ignore
                                    pair.Close()
                                }
                                :> Task)
                            |> ignore
                        with _ ->
                            client.Close()
                            upstream.Close()
                with _ ->
                    ()
            }
            :> Task)
        |> ignore

    interface IDisposable with
        member _.Dispose() =
            cts.Cancel()

            try
                listener.Stop()
            with _ ->
                ()

            for pair in List.ofSeq pairs.Keys do
                pair.Close()
