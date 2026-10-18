namespace Lrm.Cluster

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Net
open System.Net.Sockets
open System.Threading
open System.Threading.Channels
open System.Threading.Tasks

/// A best-effort frame transport for one process (ClusterProtocol contract, "Transports"). Frames can be lost (a peer
/// is unreachable, a connection drops, a queue is full), and the protocol retransmits, so Send never blocks for long
/// and never throws for network errors.
type ITransport =
    inherit IDisposable
    /// Sends one encoded frame to member `dest`.
    abstract Send: dest: uint64 * frame: byte[] -> unit
    /// Starts delivering received frames, decoded and checked, to `receive` (called from any thread).
    abstract Start: receive: (Frame -> unit) -> unit

/// In-process transport: a hub that hands frames between endpoints of one process. Frames are decoded on delivery,
/// so the codec is exercised exactly as over TCP.
[<Sealed>]
type InProcHub() =
    let endpoints = ConcurrentDictionary<uint64, Frame -> unit>()

    /// The transport of member `id`.
    member _.Connect(id: uint64) : ITransport =
        { new ITransport with
            member _.Send(dest: uint64, bytes: byte[]) =
                match endpoints.TryGetValue dest with
                | true, receive ->
                    match FrameCodec.tryDecodeArray bytes with
                    | Ok frame -> receive frame
                    | Error _ -> ()
                | _ -> ()

            member _.Start(receive: Frame -> unit) = endpoints[id] <- receive
            member _.Dispose() = endpoints.TryRemove id |> ignore
        }

/// Items released at a due time on a dedicated thread, in (due time, arrival) order. Used to delay frames.
[<Sealed>]
type DelayQueue<'T>(clock: IClock, deliver: 'T -> unit) =
    let gate = obj ()
    let heap = PriorityQueue<'T, struct (int64 * int64)>()
    let mutable order = 0L
    let mutable stopped = false

    let loop () =
        let mutable running = true

        while running do
            let mutable have = false
            let mutable item = Unchecked.defaultof<'T>
            Monitor.Enter gate

            try
                while not have && not stopped do
                    let mutable next = Unchecked.defaultof<'T>
                    let mutable priority = struct (0L, 0L)

                    if heap.TryPeek(&next, &priority) then
                        let struct (due, _) = priority
                        let wait = due - clock.NowNs

                        if wait <= 0L then
                            item <- heap.Dequeue()
                            have <- true
                        else
                            Monitor.Wait(gate, TimeSpan.FromTicks(max 1L (wait / 100L))) |> ignore
                    else
                        Monitor.Wait gate |> ignore

                if stopped then
                    running <- false
            finally
                Monitor.Exit gate

            if have then
                try
                    deliver item
                with _ ->
                    ()

    let thread =
        Thread(ThreadStart loop, IsBackground = true, Name = "lrm-cluster-delay")

    do thread.Start()

    /// Releases `item` at `dueNs` on the queue's clock.
    member _.Enqueue(item: 'T, dueNs: int64) =
        lock gate (fun () ->
            order <- order + 1L
            heap.Enqueue(item, struct (dueNs, order))
            Monitor.Pulse gate)

    interface IDisposable with
        member _.Dispose() =
            lock gate (fun () ->
                stopped <- true
                Monitor.PulseAll gate)

/// A transport decorator that applies a ChaosPolicy to outgoing frames (drop, duplicate, delay, reorder) and drops
/// every frame while the endpoint is partitioned. Fault times count from the transport's creation.
[<Sealed>]
type ChaosTransport(inner: ITransport, self: uint64, policy: ChaosPolicy, clock: IClock) =
    let start = clock.NowNs

    let queue =
        new DelayQueue<struct (uint64 * byte[])>(clock, (fun (struct (dest, bytes)) -> inner.Send(dest, bytes)))

    interface ITransport with
        member _.Send(dest: uint64, bytes: byte[]) =
            let now = clock.NowNs
            let fate = lock policy (fun () -> policy.Decide(self, dest, now - start))

            for d in fate.Copies do
                queue.Enqueue(struct (dest, bytes), now + d)

        member _.Start(receive: Frame -> unit) =
            inner.Start(fun frame ->
                let cut = lock policy (fun () -> policy.Partitioned(self, clock.NowNs - start))

                if not cut then
                    receive frame)

        member _.Dispose() =
            (queue :> IDisposable).Dispose()
            inner.Dispose()

/// One TCP connection carrying frames both ways: a reader task that decodes frames and a writer task fed by a bounded
/// queue (full means dropped: the protocol retransmits).
[<Sealed; AllowNullLiteral>]
type internal FrameConnection
    (client: TcpClient, onFrame: FrameConnection -> Frame -> unit, onClosed: FrameConnection -> unit) =
    let stream = client.GetStream()

    let outbox =
        Channel.CreateBounded<byte[]>(
            BoundedChannelOptions(4096, FullMode = BoundedChannelFullMode.DropWrite, SingleReader = true)
        )

    let cts = new CancellationTokenSource()
    let mutable closed = 0

    member this.Close() =
        if Interlocked.Exchange(&closed, 1) = 0 then
            cts.Cancel()
            outbox.Writer.TryComplete() |> ignore

            try
                client.Close()
            with _ ->
                ()

            onClosed this

    member _.Send(bytes: byte[]) = outbox.Writer.TryWrite bytes |> ignore

    member this.Run() =
        let token = cts.Token

        Task.Run(fun () ->
            task {
                try
                    while not token.IsCancellationRequested do
                        let! bytes = outbox.Reader.ReadAsync(token)
                        do! stream.WriteAsync(bytes.AsMemory(), token)
                with _ ->
                    ()

                this.Close()
            }
            :> Task)
        |> ignore

        Task.Run(fun () ->
            task {
                try
                    let mutable go = true

                    while go do
                        let! next = FrameStream.readAsync stream token

                        match next with
                        | None -> go <- false
                        | Some(_, frame) -> onFrame this frame
                with _ ->
                    ()

                this.Close()
            }
            :> Task)
        |> ignore

/// The coordinator's side of TCP: listens on an address (127.0.0.1 in tests), accepts node connections, and routes a
/// frame to member `dest` over the connection that last delivered a frame from that member.
[<Sealed>]
type TcpServerTransport(endpoint: IPEndPoint) =
    let listener = new TcpListener(endpoint)
    do listener.Start()
    let peers = ConcurrentDictionary<uint64, FrameConnection>()

    let connections =
        ConcurrentDictionary<FrameConnection, byte>(HashIdentity.Reference)

    let cts = new CancellationTokenSource()

    /// Listens on 127.0.0.1:`port` (0 picks a free port).
    new(port: int) = new TcpServerTransport(IPEndPoint(IPAddress.Loopback, port))

    /// The port the server listens on.
    member _.Port = (listener.LocalEndpoint :?> IPEndPoint).Port

    interface ITransport with
        member _.Start(receive: Frame -> unit) =
            let token = cts.Token

            Task.Run(fun () ->
                task {
                    try
                        while not token.IsCancellationRequested do
                            let! client = listener.AcceptTcpClientAsync(token)
                            client.NoDelay <- true

                            let conn =
                                FrameConnection(
                                    client,
                                    (fun c frame ->
                                        peers[frame.Sender] <- c
                                        receive frame),
                                    (fun c ->
                                        connections.TryRemove c |> ignore

                                        for kv in List.ofSeq peers do
                                            if obj.ReferenceEquals(kv.Value, c) then
                                                peers.TryRemove kv |> ignore)
                                )

                            connections[conn] <- 0uy
                            conn.Run()
                    with _ ->
                        ()
                }
                :> Task)
            |> ignore

        member _.Send(dest: uint64, bytes: byte[]) =
            match peers.TryGetValue dest with
            | true, c -> c.Send bytes
            | _ -> ()

        member _.Dispose() =
            cts.Cancel()

            try
                listener.Stop()
            with _ ->
                ()

            for c in List.ofSeq connections.Keys do
                c.Close()

/// A node's side of TCP: connects to the coordinator (or a ChaosProxy in front of it) and reconnects every 100 ms
/// after a failure. Frames sent while disconnected are dropped.
[<Sealed>]
type TcpClientTransport(host: string, port: int) =
    let cts = new CancellationTokenSource()

    [<VolatileField>]
    let mutable current: FrameConnection = null

    interface ITransport with
        member _.Start(receive: Frame -> unit) =
            let token = cts.Token

            Task.Run(fun () ->
                task {
                    while not token.IsCancellationRequested do
                        let client = new TcpClient(NoDelay = true)

                        try
                            do! client.ConnectAsync(host, port, token)

                            let closed =
                                TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously)

                            let conn =
                                FrameConnection(
                                    client,
                                    (fun _ frame -> receive frame),
                                    (fun _ -> closed.TrySetResult() |> ignore)
                                )

                            current <- conn
                            conn.Run()
                            do! closed.Task
                            current <- null
                        with _ ->
                            // A failed connect leaves the client unused; a closed connection already closed it.
                            client.Dispose()

                        if not token.IsCancellationRequested then
                            try
                                do! Task.Delay(100, token)
                            with _ ->
                                ()
                }
                :> Task)
            |> ignore

        member _.Send(_: uint64, bytes: byte[]) =
            match current with
            | null -> ()
            | c -> c.Send bytes

        member _.Dispose() =
            cts.Cancel()

            match current with
            | null -> ()
            | c -> c.Close()
