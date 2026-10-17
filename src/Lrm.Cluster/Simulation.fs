namespace Lrm.Cluster

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.Text
open System.Threading.Tasks
open Lrm.Core

/// One simulated node.
type SimNode =
    {
        NodeId: uint64
        NodeName: string
        NodeCanHost: uint32 list
        /// Virtual time from start (or restart) to Ready.
        ColdStartNs: int64
    }

/// A deterministic simulation of one cluster run of a fake job: the real coordinator and node actors on virtual time,
/// with every frame encoded, sent through a ChaosPolicy and decoded.
type SimConfig =
    {
        /// The chaos seed: link fates, costs and (in Simulation.standard) the fault schedule.
        Seed: uint64
        Job: FakeJob
        Nodes: SimNode list
        Coordinator: CoordinatorConfig
        Chaos: ChaosConfig
        Faults: Fault list
        /// Executor cost of one unit at each stage; the last value repeats for later stages.
        StageCostNs: int64 list
        /// Each work item costs 0 to CostJitterNs extra.
        CostJitterNs: int64
        /// The run fails when virtual time passes this.
        MaxVirtualNs: int64
        /// The run fails after this many events.
        MaxEvents: int64
        /// Where telemetry goes during the run, timed on virtual time; None for no telemetry.
        Sink: ITelemetrySink option
    }

/// Counters of one simulated run.
type SimStats =
    {
        Events: int64
        VirtualNs: int64
        FramesSent: int64
        FramesDelivered: int64
        FramesDropped: int64
        FramesDuplicated: int64
        Kills: int
        Restarts: int
    }

/// The result of one simulated run.
type SimResult =
    {
        Seed: uint64
        /// None when the run did not finish (see Failure).
        Outcome: RunOutcome option
        /// SHA-256 over every event the simulator processed, in order, with its virtual time (Simulation.TraceTag).
        TraceHash: ContentId
        Stats: SimStats
        /// Why the run did not finish: a time or event limit, or an actor exception.
        Failure: string option
        /// The reports of the nodes alive at the end.
        Nodes: NodeReport list
    }

type internal SimEvent =
    | Deliver of dest: uint64 * destIncarnation: int * src: uint64 * srcIncarnation: int * bytes: byte[]
    | Timer of actor: uint64 * incarnation: int * tag: uint64
    | WorkDone of actor: uint64 * incarnation: int * token: uint64 * result: obj
    | KillNode of node: uint64
    | RestartNode of node: uint64

/// The deterministic simulator (ClusterProtocol contract, "Simulation").
[<RequireQualifiedAccess>]
module Simulation =
    /// The trace hash is SHA-256("lrm.cluster.trace/1" || 0x00 || records): one 57-byte record per processed event,
    /// u8 kind (1 deliver, 2 timer, 3 work done, 4 kill, 5 restart), u64 virtual time, u64 actor, u64 other (sender,
    /// tag or token), and 32 bytes (the frame trailer for a delivery, zero otherwise). It covers the schedule, so it
    /// includes virtual times; it is a test fingerprint, never a content address, and wall-clock time never enters it.
    [<Literal>]
    let TraceTag = "lrm.cluster.trace/1"

    /// Nodes 1..n, each able to host every stage of the job.
    let nodes (job: FakeJob) (count: int) (coldStart: int -> int64) =
        [
            for i in 1..count ->
                {
                    NodeId = uint64 i
                    NodeName = "n" + Invariant.i32 i
                    NodeCanHost = [ for s in 0 .. job.Stages - 1 -> uint32 s ]
                    ColdStartNs = coldStart i
                }
        ]

    /// A fault-free configuration: the standard job, `count` nodes ready at 10 ms, a clean LAN, 10 ms per unit.
    let clean (seed: uint64) (count: int) : SimConfig =
        let job = FakeJob.standard

        {
            Seed = seed
            Job = job
            Nodes = nodes job count (fun _ -> Ns.ms 10L)
            Coordinator =
                { CoordinatorConfig.standard with
                    MinNodes = count
                }
            Chaos = ChaosConfig.lan
            Faults = []
            StageCostNs = [ Ns.ms 10L ]
            CostJitterNs = 0L
            MaxVirtualNs = Ns.seconds 600L
            MaxEvents = 5_000_000L
            Sink = None
        }

    /// The chaos configuration of one seed: the standard job on 7 nodes (3 stages, 2 replicas each, and a spare),
    /// cold starts of 300 to 800 ms, standard network chaos, and ChaosPolicy.RandomFaults over the expected run time.
    let standard (seed: uint64) : SimConfig =
        let job = FakeJob.standard
        let count = job.Stages * 2 + 1
        let cold = DetRng(seed, DetRng.streamOf "lrm.cluster.sim.coldstart")
        let colds = [| for _ in 1..count -> Ns.ms (300L + int64 (cold.NextBelow 501u)) |]
        let horizon = Ns.ms 800L + int64 (job.Steps * job.Stages) * Ns.ms 30L
        let ids = [ for i in 1..count -> uint64 i ]

        {
            Seed = seed
            Job = job
            Nodes = nodes job count (fun i -> colds[i - 1])
            Coordinator =
                { CoordinatorConfig.standard with
                    MinNodes = job.Stages * 2
                }
            Chaos = ChaosConfig.standard
            Faults = ChaosPolicy.RandomFaults(seed, ids, horizon)
            StageCostNs = [ Ns.ms 10L; Ns.ms 15L; Ns.ms 8L ]
            CostJitterNs = Ns.ms 5L
            MaxVirtualNs = Ns.seconds 600L
            MaxEvents = 5_000_000L
            Sink = None
        }

    /// Runs one simulation to the end of the run (or a limit).
    let run (config: SimConfig) : SimResult =
        let clock = VirtualClock()
        let queue = PriorityQueue<SimEvent, struct (int64 * int64)>()
        let order = ref 0L

        let schedule (at: int64) (ev: SimEvent) =
            order.Value <- order.Value + 1L
            queue.Enqueue(ev, struct (at, order.Value))

        let policy = ChaosPolicy(config.Seed, config.Chaos, config.Faults)
        let workload = FakeWorkload config.Job
        let coordinator = Coordinator(config.Coordinator, workload, MemoryBlobStore())
        let actors = Dictionary<uint64, IActor>()
        let nodes = Dictionary<uint64, Node>()
        let incarnation = Dictionary<uint64, int>()
        let alive = Dictionary<uint64, bool>()
        let contexts = Dictionary<struct (uint64 * int), IActorContext>()
        let costRngs = Dictionary<struct (uint64 * int), DetRng>()
        let specs = Dictionary<uint64, SimNode>()
        let sent = ref 0L
        let delivered = ref 0L
        let dropped = ref 0L
        let duplicated = ref 0L
        let kills = ref 0
        let restarts = ref 0

        for n in config.Nodes do
            if n.NodeId = 0UL || specs.ContainsKey n.NodeId then
                invalidArg "config" (sprintf "node ids must be distinct and non-zero (%d)" n.NodeId)

            specs[n.NodeId] <- n

        let trace = new Hash.Incremental()
        trace.Append(Encoding.ASCII.GetBytes TraceTag)
        trace.Append([| 0uy |])
        let record = Array.zeroCreate<byte> 57

        let traceEvent (kind: byte) (t: int64) (actor: uint64) (other: uint64) (digest: ContentId) =
            record[0] <- kind
            BinaryPrimitives.WriteInt64LittleEndian(Span(record, 1, 8), t)
            BinaryPrimitives.WriteUInt64LittleEndian(Span(record, 9, 8), actor)
            BinaryPrimitives.WriteUInt64LittleEndian(Span(record, 17, 8), other)
            digest.WriteTo(Span(record, 25, 32))
            trace.Append record

        let isAlive (id: uint64) (inc: int) =
            id = 0UL
            || (alive.TryGetValue id
                |> function
                    | true, a -> a && incarnation[id] = inc
                    | _ -> false)

        let currentIncarnation (id: uint64) =
            match incarnation.TryGetValue id with
            | true, i -> i
            | _ -> 0

        let stageCost (stage: uint32) =
            match config.StageCostNs with
            | [] -> 0L
            | costs -> costs[min (int stage) (costs.Length - 1)]

        let chaosEvent (fault: string) (node: uint64) =
            let t = Telemetry.current ()

            if t.Enabled then
                t.Counter(
                    ClusterNames.Chaos,
                    1.0,
                    TelemetryUnits.Count,
                    [
                        Attr.str "fault" fault
                        Attr.str TelemetryKeys.Node (if node = 0UL then "coord" else "n" + Invariant.u64 node)
                    ]
                )

        let contextFor (id: uint64) (inc: int) : IActorContext =
            let key = struct (id, inc)

            match contexts.TryGetValue key with
            | true, c -> c
            | _ ->
                let c =
                    { new IActorContext with
                        member _.Now = clock.NowNs

                        member _.Send(dest: uint64, frame: Frame) =
                            let bytes = FrameCodec.encode frame
                            let now = clock.NowNs
                            sent.Value <- sent.Value + 1L

                            if dest <> 0UL && not (isAlive dest (currentIncarnation dest)) then
                                // Nobody is listening: the connection is gone.
                                dropped.Value <- dropped.Value + 1L
                            else
                                match (policy.Decide(id, dest, now)).Copies with
                                | [] ->
                                    dropped.Value <- dropped.Value + 1L
                                    chaosEvent "drop" id
                                | copies ->
                                    if copies.Length > 1 then
                                        duplicated.Value <- duplicated.Value + 1L
                                        chaosEvent "duplicate" id

                                    let destInc = currentIncarnation dest

                                    for d in copies do
                                        schedule (now + d) (Deliver(dest, destInc, id, inc, bytes))

                        member _.SetTimer(due: int64, tag: uint64) =
                            schedule (max due clock.NowNs) (Timer(id, inc, tag))

                        member _.Execute(cost: WorkCost, work: unit -> obj, token: uint64) =
                            let result =
                                try
                                    work ()
                                with e ->
                                    box e

                            let now = clock.NowNs

                            let rng =
                                match costRngs.TryGetValue key with
                                | true, r -> r
                                | _ ->
                                    let r =
                                        DetRng(
                                            config.Seed,
                                            DetRng.streamOf (
                                                "lrm.cluster.sim.cost/" + Invariant.u64 id + "/" + Invariant.i32 inc
                                            )
                                        )

                                    costRngs[key] <- r
                                    r

                            let jitter =
                                if config.CostJitterNs > 0L then
                                    int64 (rng.NextU64() % uint64 (config.CostJitterNs + 1L))
                                else
                                    0L

                            let total =
                                (int64 cost.CostUnits * stageCost cost.CostStage + jitter)
                                * int64 (policy.LagFactor(id, now))

                            schedule (now + total) (WorkDone(id, inc, token, result))
                    }

                contexts[key] <- c
                c

        let startNode (spec: SimNode) (inc: int) =
            let nodeId = spec.NodeId

            let executor =
                FakeExecutor(spec.NodeCanHost, (fun u -> policy.Corrupts(nodeId, u)), ignore)

            let node =
                Node(
                    { NodeConfig.create nodeId spec.NodeName (uint64 inc) with
                        ColdStartNs = spec.ColdStartNs
                    },
                    executor
                )

            actors[nodeId] <- node
            nodes[nodeId] <- node
            incarnation[nodeId] <- inc
            alive[nodeId] <- true
            (node :> IActor).Start(contextFor nodeId inc)

        let telemetryScope =
            match config.Sink with
            | Some sink -> Telemetry.install (Telemetry.ofSinkWithClock clock.TelemetryClock sink)
            | None ->
                { new IDisposable with
                    member _.Dispose() = ()
                }

        let mutable failure: string option = None
        let mutable events = 0L

        try
            try
                actors[0UL] <- coordinator
                (coordinator :> IActor).Start(contextFor 0UL 0)

                for spec in config.Nodes do
                    startNode spec 1

                for f in config.Faults do
                    match f with
                    | Fault.Kill(n, t) -> schedule t (KillNode n)
                    | Fault.Restart(n, t) -> schedule t (RestartNode n)
                    | _ -> ()

                let mutable ev = Unchecked.defaultof<SimEvent>
                let mutable prio = struct (0L, 0L)

                while coordinator.Outcome.IsNone && failure.IsNone do
                    if not (queue.TryDequeue(&ev, &prio)) then
                        failure <- Some "the event queue ran dry"
                    else
                        let struct (t, _) = prio

                        if t > config.MaxVirtualNs then
                            failure <- Some(sprintf "virtual time limit of %d ns reached" config.MaxVirtualNs)
                        elif events >= config.MaxEvents then
                            failure <- Some(sprintf "event limit of %d reached" config.MaxEvents)
                        else
                            clock.AdvanceTo t
                            events <- events + 1L

                            match ev with
                            | Deliver(dest, destInc, src, srcInc, bytes) ->
                                if
                                    isAlive dest destInc
                                    && isAlive src srcInc
                                    && not (policy.Partitioned(dest, t))
                                    && not (policy.Partitioned(src, t))
                                then
                                    delivered.Value <- delivered.Value + 1L
                                    traceEvent 1uy t dest src (FrameCodec.trailer bytes)

                                    match FrameCodec.tryDecodeArray bytes with
                                    | Ok frame -> actors[dest].OnFrame(contextFor dest destInc, frame)
                                    | Error e ->
                                        failwithf "simulator frame failed to decode: %s" (FrameError.describe e)
                                else
                                    dropped.Value <- dropped.Value + 1L
                            | Timer(a, inc, tag) ->
                                if isAlive a inc then
                                    traceEvent 2uy t a tag ContentId.zero
                                    actors[a].OnTimer(contextFor a inc, tag)
                            | WorkDone(a, inc, token, result) ->
                                if isAlive a inc then
                                    traceEvent 3uy t a token ContentId.zero
                                    actors[a].OnWorkDone(contextFor a inc, token, result)
                            | KillNode n ->
                                match alive.TryGetValue n with
                                | true, true ->
                                    alive[n] <- false
                                    incarnation[n] <- incarnation[n] + 1
                                    kills.Value <- kills.Value + 1
                                    traceEvent 4uy t n 0UL ContentId.zero
                                    chaosEvent "kill" n
                                | _ -> ()
                            | RestartNode n ->
                                match alive.TryGetValue n with
                                | true, false ->
                                    restarts.Value <- restarts.Value + 1
                                    traceEvent 5uy t n 0UL ContentId.zero
                                    chaosEvent "restart" n
                                    startNode specs[n] incarnation[n]
                                | _ -> ()
            with e ->
                failure <- Some(sprintf "%s: %s" (e.GetType().Name) e.Message)
        finally
            telemetryScope.Dispose()

        let reports =
            [
                for kv in nodes |> Seq.sortBy (fun kv -> kv.Key) do
                    if alive[kv.Key] then
                        yield kv.Value.Report()
            ]

        let traceHash = trace.Finish()
        (trace :> IDisposable).Dispose()

        {
            Seed = config.Seed
            Outcome = coordinator.Outcome
            TraceHash = traceHash
            Stats =
                {
                    Events = events
                    VirtualNs = clock.NowNs
                    FramesSent = sent.Value
                    FramesDelivered = delivered.Value
                    FramesDropped = dropped.Value
                    FramesDuplicated = duplicated.Value
                    Kills = kills.Value
                    Restarts = restarts.Value
                }
            Failure = failure
            Nodes = reports
        }

    /// Runs many simulations in parallel; results come back in the order of the configs.
    let runAll (configs: SimConfig list) : SimResult list =
        let input = Array.ofList configs
        let output = Array.zeroCreate<SimResult> input.Length
        Parallel.For(0, input.Length, (fun i -> output[i] <- run input[i])) |> ignore
        List.ofArray output
