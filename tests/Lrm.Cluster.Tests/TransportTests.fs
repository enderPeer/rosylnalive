/// Real-time transports: in-process with chaos, TCP on 127.0.0.1, the ChaosProxy, and the 3-process kill test.
/// Liveness waits are capped at 60 s and the assertions are on digests only, never on timings.
module Lrm.Cluster.Tests.TransportTests

open System
open System.Collections.Concurrent
open System.Net
open System.Threading
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Cluster.Tests.Common
open Lrm.Testing

let private liveness = TimeSpan.FromSeconds 60.0

let private job =
    { FakeJob.standard with
        Lanes = 2
        Steps = 6
    }

let private frameFrom (sender: uint64) (seq: uint64) (payload: byte[]) =
    FrameCodec.encode
        {
            Kind = MessageKind.Notice
            Flags = FrameFlags.Empty
            Epoch = 1UL
            Sender = sender
            Seq = seq
            UnitId = ContentId.zero
            Payload = payload
        }

let private waitFor (what: string) (condition: unit -> bool) =
    let clock = Diagnostics.Stopwatch.StartNew()

    while not (condition ()) && clock.Elapsed < liveness do
        Thread.Sleep 5

    if not (condition ()) then
        failtestf "%s within %O" what liveness

/// Runs a coordinator and `nodes` in-process actor hosts until the run finishes; returns the outcome.
let private runInProcess (nodes: int) (wrap: uint64 -> ITransport -> ITransport) =
    let hub = InProcHub()
    let clock = SystemClock.Instance

    let coordinator =
        Coordinator(
            { CoordinatorConfig.standard with
                MinNodes = nodes
            },
            FakeWorkload job,
            MemoryBlobStore()
        )

    use coordHost = new ActorHost(coordinator, wrap 0UL (hub.Connect 0UL), clock)
    coordHost.Start()

    let hosts =
        [
            for i in 1..nodes do
                let id = uint64 i

                let node =
                    Node(NodeConfig.create id ("n" + Invariant.i32 i) id, FakeExecutor([ 0u; 1u; 2u ]))

                let host = new ActorHost(node, wrap id (hub.Connect id), clock)
                host.Start()
                yield node, host
        ]

    try
        if not (coordinator.Completion.Wait liveness) then
            let hostState (h: ActorHost) =
                sprintf "host %A %O\n" h.Completion.Status h.Completion.Exception

            let states =
                (hostState coordHost + coordinator.Describe())
                :: [ for node, h in hosts -> hostState h + node.Describe() ]

            failtestf "the run did not finish within 60 s\n%s" (String.concat "" states)

        let reports = [ for node, _ in hosts -> node.Stopped.Wait liveness, node ]
        coordinator.Completion.Result, reports
    finally
        for _, h in hosts do
            (h :> IDisposable).Dispose()

let tests =
    testSequenced
    <| testList
        "transport"
        [
            testCase "an in-process cluster with chaos transports matches the reference"
            <| fun () ->
                let clock = SystemClock.Instance

                let outcome, reports =
                    runInProcess 4 (fun id t ->
                        new ChaosTransport(t, id, ChaosPolicy(40UL + id, ChaosConfig.proxy, []), clock) :> ITransport)

                Expect.equal outcome.Status RunStatus.Completed "status"
                Expect.equal outcome.RunDigest (reference job).RunDigest "run digest"
                Expect.equal outcome.ReplicaDigest (reference job).ReplicaDigest "replica digest"

                for stopped, _ in reports do
                    Expect.isTrue stopped "every node was shut down"

            testCase "TCP carries frames both ways, including an 8 MiB payload"
            <| fun () ->
                use server = new TcpServerTransport(0)
                let atServer = ConcurrentQueue<Frame>()
                let atClient = ConcurrentQueue<Frame>()
                (server :> ITransport).Start(atServer.Enqueue)
                use client = new TcpClientTransport("127.0.0.1", server.Port)
                (client :> ITransport).Start(atClient.Enqueue)
                let hello = frameFrom 5UL 1UL "hello"B

                // Frames sent before the connection is up are dropped; keep sending until one arrives.
                waitFor "the server hears the client" (fun () ->
                    (client :> ITransport).Send(0UL, hello)
                    Thread.Sleep 10
                    not atServer.IsEmpty)

                let big = Array.init (8 * 1024 * 1024) (fun i -> byte (i * 31))
                let bigFrame = frameFrom 0UL 2UL big
                (server :> ITransport).Send(5UL, bigFrame)
                waitFor "the client gets the 8 MiB frame" (fun () -> not atClient.IsEmpty)
                let ok, got = atClient.TryDequeue()
                Expect.isTrue ok "received"
                Expect.equal got.Payload big "payload intact"
                let _, first = atServer.TryDequeue()
                Expect.equal first.Payload "hello"B "the server decoded the client's frame"

            testCase "the chaos proxy forwards frames byte for byte and in order"
            <| fun () ->
                use server = new TcpServerTransport(0)
                let atServer = ConcurrentQueue<Frame>()
                (server :> ITransport).Start(atServer.Enqueue)

                use proxy =
                    new ChaosProxy(
                        IPEndPoint(IPAddress.Loopback, 0),
                        IPEndPoint(IPAddress.Loopback, server.Port),
                        ChaosPolicy(1UL, ChaosConfig.none, []),
                        SystemClock.Instance
                    )

                proxy.Start()
                use client = new TcpClientTransport("127.0.0.1", proxy.Port)
                (client :> ITransport).Start ignore

                waitFor "the first frame gets through" (fun () ->
                    (client :> ITransport).Send(0UL, frameFrom 5UL 0UL [||])
                    Thread.Sleep 10
                    not atServer.IsEmpty)

                while atServer.TryDequeue() |> fst do
                    ()

                let frames =
                    [ for i in 1..100 -> frameFrom 5UL (uint64 i) (BitConverter.GetBytes i) ]

                for f in frames do
                    (client :> ITransport).Send(0UL, f)

                waitFor "100 frames arrive" (fun () -> atServer.Count >= 100)
                let got = [ for _ in 1..100 -> snd (atServer.TryDequeue()) ]
                Expect.equal (got |> List.map FrameCodec.encode) frames "byte for byte, in order"
                Expect.equal proxy.Stats.Dropped 0L "nothing dropped without chaos"

            testCase "a Sever fault cuts the node's connection once and the node reconnects"
            <| fun () ->
                use server = new TcpServerTransport(0)
                let atServer = ConcurrentQueue<Frame>()
                (server :> ITransport).Start(atServer.Enqueue)

                // Sever at 0: the cut happens as soon as the proxy learns the node id from its first frame.
                use proxy =
                    new ChaosProxy(
                        IPEndPoint(IPAddress.Loopback, 0),
                        IPEndPoint(IPAddress.Loopback, server.Port),
                        ChaosPolicy(1UL, ChaosConfig.none, [ Fault.Sever(5UL, 0L) ]),
                        SystemClock.Instance
                    )

                proxy.Start()
                use client = new TcpClientTransport("127.0.0.1", proxy.Port)
                (client :> ITransport).Start ignore

                waitFor "frames flow after the cut and the reconnect" (fun () ->
                    (client :> ITransport).Send(0UL, frameFrom 5UL 1UL [||])
                    Thread.Sleep 10
                    proxy.Stats.Severed >= 1L && not atServer.IsEmpty)

                Expect.equal proxy.Stats.Severed 1L "cut once"
                Expect.isGreaterThanOrEqual proxy.Stats.Connections 2L "reconnected"

            testCase "a TCP cluster in one process, through the chaos proxy, matches the reference"
            <| fun () ->
                let clock = SystemClock.Instance

                let coordinator =
                    Coordinator(
                        { CoordinatorConfig.standard with
                            MinNodes = 3
                        },
                        FakeWorkload job,
                        MemoryBlobStore()
                    )

                use server = new TcpServerTransport(0)

                use proxy =
                    new ChaosProxy(
                        IPEndPoint(IPAddress.Loopback, 0),
                        IPEndPoint(IPAddress.Loopback, server.Port),
                        ChaosPolicy(5UL, ChaosConfig.proxy, []),
                        clock
                    )

                proxy.Start()
                use coordHost = new ActorHost(coordinator, server, clock)
                coordHost.Start()

                let hosts =
                    [
                        for i in 1UL .. 3UL do
                            let node =
                                Node(NodeConfig.create i ("n" + Invariant.u64 i) i, FakeExecutor([ 0u; 1u; 2u ]))

                            let host =
                                new ActorHost(node, new TcpClientTransport("127.0.0.1", proxy.Port), clock)

                            host.Start()
                            yield host
                    ]

                try
                    Expect.isTrue (coordinator.Completion.Wait liveness) "finished within 60 s"
                    let o = coordinator.Completion.Result
                    Expect.equal o.Status RunStatus.Completed "status"
                    Expect.equal o.RunDigest (reference job).RunDigest "run digest"
                finally
                    for h in hosts do
                        (h :> IDisposable).Dispose()

            testCase "3 node processes with a kill and a chaos proxy agree with the reference"
            <| fun () ->
                let run = Harness.standardLocalRun 1UL
                let start = Harness.selfStartInfo [ "--cluster-worker" ] Env.workerEnvironment
                let r = Harness.runLocal start run

                let diagnostics =
                    r.Nodes
                    |> List.map (fun n -> sprintf "%s: exit %A, %s" n.Name n.ExitCode n.Errors)

                Expect.isNone r.Failure (sprintf "%A\n%s" r.Failure (String.concat "\n" diagnostics))
                let o = r.Outcome.Value
                Expect.equal o.Status RunStatus.Completed "status"
                Expect.equal o.RunDigest r.Reference.RunDigest "run digest"
                Expect.equal o.ReplicaDigest r.Reference.ReplicaDigest "replica digest"
                Expect.isEmpty o.Divergences "no divergence"

                let killed = r.Nodes |> List.filter _.Killed
                Expect.equal (killed |> List.map _.Name) [ "n2" ] "n2 was killed"
                let survivors = r.Nodes |> List.filter (fun n -> not n.Killed)
                let finals = r.Reference.Chains |> List.map (fun c -> c.Chain, c) |> Map.ofList

                for n in survivors do
                    Expect.equal n.ExitCode (Some 0) (sprintf "%s exited cleanly: %s" n.Name n.Errors)
                    let report = n.Report.Value

                    for c in report.Chains do
                        if c.Head = finals[c.Chain].Head then
                            Expect.equal c.Digest finals[c.Chain].Digest (sprintf "%s chain %A" n.Name c.Chain)

                // Every chain survives at its final head on some live process.
                for KeyValue(chain, f) in finals do
                    let held =
                        survivors
                        |> List.exists (fun n ->
                            n.Report.Value.Chains
                            |> List.exists (fun c -> c.Chain = chain && c.Head = f.Head))

                    Expect.isTrue held (sprintf "chain %A is held at its final head" chain)
        ]
