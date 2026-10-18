namespace Lrm.Serving

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.Net
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open Lrm.Core
open Lrm.Telemetry
open Lrm.Cluster
open Lrm.Models.Qwen

/// One node of a harness run.
type HarnessNode =
    {
        Id: uint64
        Name: string
        /// --layers: "all", or the layer ranges of the stages it hosts.
        Layers: string
        /// Start the node only once the coordinator has accepted this many units (0: at the start; a later node is a
        /// spare that joins a running pipeline).
        StartAfterUnits: int
    }

/// A process-level kill of node `Node` (1-based index into the nodes) once the coordinator has accepted `AfterUnits`
/// units, and its restart as a new incarnation with no state. Kills happen in list order.
type HarnessKill =
    {
        Node: int
        AfterUnits: int
        /// Restart it this long after the kill; None keeps it dead.
        RestartAfterMs: int option
    }

/// The shapes of the harness runs (StagePipeline, "The harness").
[<RequireQualifiedAccess>]
type Topology =
    /// Two nodes, one stage each, one replica per stage: a killed node's stage waits for its restart, which replays
    /// the stage's chains (the loss of every replica, without a spare).
    | Split2
    /// Three nodes that host every stage, two replicas per stage: a kill fails over to the other replica with no
    /// catch-up, and a refilled replica replays.
    | Replicated3
    /// Two nodes per stage (two replicas each) and a spare that hosts every stage and joins after the start: both
    /// replicas of the last stage are killed, the spare takes the stage over by replay, and one of the two comes back.
    | BothLost5

[<RequireQualifiedAccess>]
module Topology =
    let name (t: Topology) =
        match t with
        | Topology.Split2 -> "split2"
        | Topology.Replicated3 -> "replicated3"
        | Topology.BothLost5 -> "bothlost5"

    let tryParse (text: string) =
        [ Topology.Split2; Topology.Replicated3; Topology.BothLost5 ]
        |> List.tryFind (fun t -> name t = text)

    /// The topology of a sweep seed: the three in turn.
    let ofSeed (seed: uint64) =
        match seed % 3UL with
        | 1UL -> Topology.Split2
        | 2UL -> Topology.Replicated3
        | _ -> Topology.BothLost5

/// One harness run: a job, its nodes and every fault.
type HarnessScenario =
    {
        Name: string
        Spec: JobSpec
        Topology: Topology
        Nodes: HarnessNode list
        Config: PipelineConfig
        /// The ChaosProxy's seed.
        Seed: uint64
        Chaos: ChaosConfig
        /// Partition and Sever faults of the proxy (times from its start).
        ProxyFaults: Fault list
        Kills: HarnessKill list
        /// Milliseconds each unit sleeps on its node (StageExecutorOptions.CostMs), so faults land mid-run.
        CostMs: int
        Threads: int
        FastRebuild: bool
        /// The whole run must finish within this (the CI liveness limit is 60 s).
        TimeoutMs: int
    }

/// Where the pieces of a harness run execute.
type HarnessMode =
    {
        /// Node processes instead of node actors in this process.
        NodeProcesses: bool
        /// The proxy as a `chaos-proxy` process instead of in this process.
        ProxyProcess: bool
        /// The coordinator as a `coord run` process instead of in this process.
        CoordinatorProcess: bool
        /// Builds a worker process from a worker command (`node ...`, `chaos-proxy ...`, `coord run ...`); needed for
        /// any process.
        Start: (string list -> ProcessStartInfo) option
        /// Write a cluster run log with this title (the coordinator creates it).
        RunLog: string option
        /// Where run logs go (LrmPaths.runs () by default).
        RunsRoot: string option
    }

[<RequireQualifiedAccess>]
module HarnessMode =
    /// Everything in this process: the coordinator, a ChaosProxy on TCP and node actors (fast; the proxy still sees
    /// every frame on a real socket).
    let inProcess =
        {
            NodeProcesses = false
            ProxyProcess = false
            CoordinatorProcess = false
            Start = None
            RunLog = None
            RunsRoot = None
        }

    /// The coordinator here, the proxy and every node as processes started by `start`.
    let processes (start: string list -> ProcessStartInfo) =
        {
            NodeProcesses = true
            ProxyProcess = true
            CoordinatorProcess = false
            Start = Some start
            RunLog = None
            RunsRoot = None
        }

/// What happened to one node incarnation.
type HarnessNodeRun =
    {
        Name: string
        Incarnation: uint64
        Killed: bool
        /// None when the process had to be killed at the end, or for an in-process node.
        ExitCode: int option
        Report: NodeReport option
        Errors: string
        /// The node actor's state at the end (in-process nodes only), for failure messages.
        State: string
    }

/// The result of a harness run.
type HarnessResult =
    {
        Scenario: HarnessScenario
        Outcome: ServingOutcome option
        Failure: string option
        Nodes: HarnessNodeRun list
        Proxy: ProxyStats option
        Kills: int
        Restarts: int
        RunDir: string option
        ElapsedMs: int64
        /// The coordinator's state and its chains' confirmations at the end, for failure messages.
        Diagnostics: string
    }

/// A child process with its output lines.
[<Sealed>]
type internal ChildProcess(psi: ProcessStartInfo, onLine: string -> unit) =
    let p = new Process(StartInfo = psi)
    let output = ConcurrentQueue<string>()
    let errors = ConcurrentQueue<string>()

    do
        p.OutputDataReceived.Add(fun e ->
            if not (isNull e.Data) then
                output.Enqueue e.Data
                onLine e.Data)

        p.ErrorDataReceived.Add(fun e ->
            if not (isNull e.Data) then
                errors.Enqueue e.Data)

        p.Start() |> ignore
        p.BeginOutputReadLine()
        p.BeginErrorReadLine()

    member _.Output = List.ofSeq output
    member _.Errors = String.Join("\n", errors)

    /// True once the process has exited.
    member _.Exited =
        try
            p.HasExited
        with _ ->
            true

    member _.Kill() =
        try
            p.Kill true
        with _ ->
            ()

    member _.CloseInput() =
        try
            p.StandardInput.Close()
        with _ ->
            ()

    /// Waits for the exit (and the last output lines); kills the process after `ms`. The exit code, or None if killed.
    member x.Finish(ms: int) : int option =
        let exited =
            try
                p.WaitForExit ms
            with _ ->
                true

        if not exited then
            x.Kill()

        try
            p.WaitForExit()
        with _ ->
            ()

        if exited then
            try
                Some p.ExitCode
            with _ ->
                None
        else
            None

    interface IDisposable with
        member _.Dispose() = p.Dispose()

/// The serving harness (StagePipeline, "The harness"): one pipeline job on localhost with a coordinator, a ChaosProxy
/// and its nodes, in this process or as processes, with frame chaos (delay, drop, duplicate, reorder), proxy partitions
/// and severs, and process kills and restarts; then the verdict against the oracles.
[<RequireQualifiedAccess>]
module ServingHarness =
    let private weightsCache =
        ConcurrentDictionary<string, Lazy<Qwen3Weights>>(StringComparer.OrdinalIgnoreCase)

    /// Verified weights of a model folder, opened once per process and never disposed (harness runs share them).
    let weights (dir: string) =
        weightsCache.GetOrAdd(dir, fun d -> lazy (Qwen3Weights.Open d)).Value

    let private referenceCache = ConcurrentDictionary<ContentId, Lazy<QwenReference>>()

    /// The oracles of a job on a CPU class, computed once per process.
    let reference (w: Qwen3Weights) (job: ServingJob) (threads: int) =
        referenceCache.GetOrAdd(ServingJob.id job, fun _ -> lazy (QwenReference.cpu w job threads)).Value

    /// The model folder and the job of a spec.
    let job (spec: JobSpec) : string * ServingJob =
        let dir =
            match ServingPaths.modelDir spec.Model with
            | Ok d -> d
            | Error e -> invalidArg "spec" e

        match JobSpec.build (weights dir) spec with
        | Ok j -> dir, j
        | Error e -> invalidArg "spec" e

    let private layersOf (s: Stage) =
        Invariant.format $"{s.FirstLayer}-{s.LastLayer}"

    let private node (id: int) (layers: string) (after: int) =
        {
            Id = uint64 id
            Name = "n" + Invariant.i32 id
            Layers = layers
            StartAfterUnits = after
        }

    /// The nodes and cluster settings of a topology for a job.
    let topology (job: ServingJob) (t: Topology) : HarnessNode list * PipelineConfig =
        let stages = ServingJob.stages job
        let first = layersOf stages.Head
        let last = layersOf (List.last stages)
        let drain = Ns.seconds 10L

        match t with
        | Topology.Split2 when stages.Length = 2 ->
            [ node 1 first 0; node 2 last 0 ],
            {
                Replicas = 1
                MinNodes = 2
                DrainNs = drain
            }
        | Topology.Split2 ->
            [ for s in stages -> node (s.Index + 1) (layersOf s) 0 ],
            {
                Replicas = 1
                MinNodes = stages.Length
                DrainNs = drain
            }
        | Topology.Replicated3 ->
            [ node 1 "all" 0; node 2 "all" 0; node 3 "all" 0 ],
            {
                Replicas = 2
                MinNodes = 3
                DrainNs = drain
            }
        | Topology.BothLost5 ->
            if stages.Length <> 2 then
                invalidArg "t" "bothlost5 needs a two-stage split"

            [ node 1 first 0; node 2 first 0; node 3 last 0; node 4 last 0; node 5 "all" 1 ],
            {
                Replicas = 2
                MinNodes = 4
                DrainNs = drain
            }

    /// The units a job runs when no lane stops early: lanes * steps * stages.
    let units (job: ServingJob) =
        job.Prompts.Length * job.Steps * (job.Boundaries.Length + 1)

    /// A run with no fault but the network's: frame chaos only.
    let clean (spec: JobSpec) (t: Topology) (seed: uint64) : HarnessScenario =
        let _, j = job spec
        let nodes, config = topology j t

        {
            Name = Invariant.format $"{Topology.name t}-clean-{seed}"
            Spec = spec
            Topology = t
            Nodes = nodes
            Config = config
            Seed = seed
            Chaos = ChaosConfig.proxy
            ProxyFaults = []
            Kills = []
            CostMs = 2
            Threads = 1
            FastRebuild = false
            TimeoutMs = 60_000
        }

    /// The seeded chaos run of a topology (StagePipeline, "The harness"): frame chaos (ChaosConfig.proxy), the
    /// topology's kills and restarts at points drawn from DetRng(seed, "lrm.serving.chaos.faults"), and with
    /// probability 1/2 each a proxy partition of one node (0.3 to 1.5 s) and a severed connection. `startMs` shifts the
    /// proxy's fault times past the nodes' start-up.
    let chaos (spec: JobSpec) (t: Topology) (seed: uint64) (startMs: int) : HarnessScenario =
        let _, j = job spec
        let nodes, config = topology j t
        let r = DetRng(seed, DetRng.streamOf ServingTags.Faults)
        let total = units j

        let pct (lo: int) (hi: int) =
            max 1 (total * (lo + int (r.NextBelow(uint32 (hi - lo + 1)))) / 100)

        let ms (lo: int) (hi: int) =
            lo + int (r.NextBelow(uint32 (hi - lo + 1)))

        let kills =
            match t with
            | Topology.Split2 ->
                let n = 1 + int (r.NextBelow(uint32 nodes.Length))

                [
                    {
                        Node = n
                        AfterUnits = pct 15 60
                        RestartAfterMs = Some(ms 200 1200)
                    }
                ]
            | Topology.Replicated3 ->
                let n = 1 + int (r.NextBelow 3u)
                let after = pct 15 60
                let restart = r.NextBelow 2u = 1u
                let delay = ms 200 1500

                [
                    {
                        Node = n
                        AfterUnits = after
                        RestartAfterMs = (if restart then Some delay else None)
                    }
                ]
            | Topology.BothLost5 ->
                let a = pct 15 40
                let b = a + pct 3 15
                let back = if r.NextBelow 2u = 1u then 3 else 4
                let delay = ms 200 800

                [
                    {
                        Node = 3
                        AfterUnits = a
                        RestartAfterMs = (if back = 3 then Some(delay + 400) else None)
                    }
                    {
                        Node = 4
                        AfterUnits = b
                        RestartAfterMs = (if back = 4 then Some delay else None)
                    }
                ]

        // A partition never takes away the second confirmation of a final unit. C1's drain (pickVerifier) counts a
        // member asked in an earlier membership as asked by its id, so a node that is partitioned while it owes the
        // final unit of a chain, expelled and rejoined, is never engaged again as a verifier; with no other replica
        // left the run would end Unverified although every digest is equal (StagePipeline, "Limits"; requested of C1).
        // So the partitioned node is any node of a one-replica split, a stage-0 replica of bothlost5 (the spare can
        // verify stage 0), and in replicated3 only when the killed node comes back. Partitions start early and are
        // short, so they end before the last steps.
        let partition =
            let draw = r.NextBelow 2u = 1u
            let pick = 1 + int (r.NextBelow(uint32 nodes.Length))
            let from = startMs + ms 100 700
            let length = ms 300 1000

            let n, on =
                match t with
                | Topology.Split2 -> pick, draw
                | Topology.Replicated3 -> pick, draw && kills |> List.forall (fun k -> k.RestartAfterMs.IsSome)
                | Topology.BothLost5 -> 1 + (pick % 2), draw

            if on then
                [ Fault.Partition(uint64 n, Ns.ms (int64 from), Ns.ms (int64 (from + length))) ]
            else
                []

        let sever =
            let on = r.NextBelow 2u = 1u
            let n = 1 + int (r.NextBelow(uint32 nodes.Length))
            let at = startMs + ms 100 2000

            if on then
                [ Fault.Sever(uint64 n, Ns.ms (int64 at)) ]
            else
                []

        { clean spec t seed with
            Name = Invariant.format $"{Topology.name t}-seed-{seed}"
            Kills = kills
            ProxyFaults = partition @ sever
        }

    // ---- The pieces of one run ----

    type private CoordinatorSide =
        abstract Port: int
        abstract Accepted: int
        abstract Finished: bool
        abstract Outcome: ServingOutcome option
        abstract Failure: string
        abstract RunDir: string option
        abstract Describe: unit -> string
        abstract Stop: unit -> unit

    type private ProxySide =
        abstract Port: int
        abstract Stats: ProxyStats option
        abstract Stop: unit -> unit

    type private NodeSide =
        abstract Kill: unit -> unit
        abstract Collect: unit -> HarnessNodeRun

    let private startFor (mode: HarnessMode) =
        match mode.Start with
        | Some s -> s
        | None -> invalidArg "mode" "a harness mode with processes needs Start"

    let private waitLine (what: string) (ms: int) (value: unit -> 'a option) (child: ChildProcess) =
        let clock = Stopwatch.StartNew()
        let mutable found = value ()

        while found.IsNone && clock.ElapsedMilliseconds < int64 ms do
            Thread.Sleep 10
            found <- value ()

        match found with
        | Some v -> v
        | None ->
            child.Kill()
            failwithf "%s printed no %s within %d ms: %s" what "start line" ms child.Errors

    /// The coordinator in this process, with the harness's run log (if any) as its ambient facade.
    let private coordinatorHere (job: ServingJob) (config: PipelineConfig) (run: RunContext option) =
        let c = new PipelineCoordinator(job, config, 0)
        let installed = run |> Option.map _.Install()
        c.Start()
        installed |> Option.iter _.Dispose()

        { new CoordinatorSide with
            member _.Port = c.Port
            member _.Accepted = c.Accepted
            member _.Finished = c.IsFinished
            member _.Outcome = c.Outcome
            member _.Failure = c.Failure 0
            member _.RunDir = run |> Option.map _.Directory

            member _.Describe() =
                let chains =
                    match c.Coordinator.Outcome with
                    | Some o ->
                        [
                            for f in o.Chains ->
                                sprintf
                                    "  final (%d, %d) step head %s confirmed by %A, pending %A, unconfirmed %A"
                                    f.Final.Chain.Lane
                                    f.Final.Chain.Stage
                                    f.Final.Head.Value.Short
                                    f.ConfirmedBy
                                    f.PendingBy
                                    f.UnconfirmedBy
                        ]
                    | None -> []

                String.concat "\n" (c.Coordinator.Describe() :: chains)

            member _.Stop() = (c :> IDisposable).Dispose()
        }

    /// The coordinator as a `coord run` process.
    let private coordinatorProcess
        (mode: HarnessMode)
        (dir: string)
        (job: ServingJob)
        (s: HarnessScenario)
        : CoordinatorSide =
        let port = ref 0
        let accepted = ref 0
        let outcome: ServingOutcome option ref = ref None
        let runDir: string option ref = ref None

        let onLine (line: string) =
            if line.StartsWith(ServingCoordinator.ListenPrefix, StringComparison.Ordinal) then
                port.Value <- Invariant.parseI32 (line.Substring ServingCoordinator.ListenPrefix.Length)
            elif line.StartsWith(ServingCoordinator.ProgressPrefix, StringComparison.Ordinal) then
                accepted.Value <- Invariant.parseI32 (line.Substring ServingCoordinator.ProgressPrefix.Length)
            elif line.StartsWith(ServingCoordinator.RunPrefix, StringComparison.Ordinal) then
                runDir.Value <- Some(line.Substring ServingCoordinator.RunPrefix.Length)
            else
                match ServingOutcome.tryParseLine line with
                | Some o -> outcome.Value <- Some o
                | None -> ()

        let args =
            [
                yield! [ "coord"; "run"; "--listen"; "0" ]
                yield! JobSpec.toArgs dir job.Boundaries s.Spec
                yield! [ "--replicas"; Invariant.i32 s.Config.Replicas ]
                yield! [ "--nodes"; Invariant.i32 s.Config.MinNodes ]
                yield! [ "--drain-ms"; Invariant.i64 (s.Config.DrainNs / Ns.PerMs) ]
                yield! [ "--timeout-ms"; Invariant.i32 s.TimeoutMs ]
                yield "--progress"
                match mode.RunLog with
                | Some title -> yield! [ "--run-log"; "--title"; title ]
                | None -> ()
            ]

        let child = new ChildProcess((startFor mode) args, onLine)

        waitLine "the coordinator" 30_000 (fun () -> if port.Value > 0 then Some port.Value else None) child
        |> ignore

        if mode.RunLog.IsSome then
            waitLine "the coordinator" 10_000 (fun () -> runDir.Value) child |> ignore

        { new CoordinatorSide with
            member _.Port = port.Value
            member _.Accepted = accepted.Value
            member _.Finished = outcome.Value.IsSome || child.Exited
            member _.Outcome = outcome.Value
            member _.Failure = sprintf "the coordinator process gave no outcome: %s" child.Errors
            member _.RunDir = runDir.Value
            member _.Describe() = child.Errors

            member _.Stop() =
                child.Finish 15_000 |> ignore
                (child :> IDisposable).Dispose()
        }

    let private proxyHere (target: int) (s: HarnessScenario) : ProxySide =
        let proxy =
            new ChaosProxy(
                IPEndPoint(IPAddress.Loopback, 0),
                IPEndPoint(IPAddress.Loopback, target),
                ChaosPolicy(s.Seed, s.Chaos, s.ProxyFaults),
                SystemClock.Instance
            )

        proxy.Start()

        { new ProxySide with
            member _.Port = proxy.Port
            member _.Stats = Some proxy.Stats
            member _.Stop() = (proxy :> IDisposable).Dispose()
        }

    let private proxyPortLine =
        Regex(@"127\.0\.0\.1:(\d+) ->", RegexOptions.CultureInvariant)

    let private proxyStatsLine =
        Regex(
            @"(\d+) connection\(s\), (\d+) frame\(s\) forwarded, (\d+) dropped, (\d+) duplicated, (\d+) severed",
            RegexOptions.CultureInvariant
        )

    /// The proxy as an `lrm chaos-proxy` process (C1's standalone proxy).
    let private proxyProcess (mode: HarnessMode) (target: int) (s: HarnessScenario) : ProxySide =
        let port = ref 0
        let stats: ProxyStats option ref = ref None

        let onLine (line: string) =
            let m = proxyPortLine.Match line

            if m.Success && port.Value = 0 then
                port.Value <- Invariant.parseI32 m.Groups[1].Value

            let st = proxyStatsLine.Match line

            if st.Success then
                let g (i: int) = Invariant.parseI64 st.Groups[i].Value

                stats.Value <-
                    Some
                        {
                            Connections = g 1
                            Forwarded = g 2
                            Dropped = g 3
                            Duplicated = g 4
                            Severed = g 5
                        }

        let ms (ns: int64) = Invariant.i64 (ns / Ns.PerMs)

        let args =
            [
                yield!
                    [
                        "chaos-proxy"
                        "--listen"
                        "0"
                        "--target"
                        "127.0.0.1:" + Invariant.i32 target
                    ]
                yield! [ "--seed"; Invariant.u64 s.Seed; "--chaos"; "none" ]
                yield! [ "--drop"; Invariant.i32 s.Chaos.DropPermille ]
                yield! [ "--duplicate"; Invariant.i32 s.Chaos.DuplicatePermille ]
                yield! [ "--reorder"; Invariant.i32 s.Chaos.ReorderPermille ]
                yield!
                    [
                        "--min-delay-ms"
                        ms s.Chaos.MinDelayNs
                        "--max-delay-ms"
                        ms s.Chaos.MaxDelayNs
                    ]
                yield! [ "--reorder-delay-ms"; ms s.Chaos.ReorderDelayNs ]
                for f in s.ProxyFaults do
                    match f with
                    | Fault.Partition(n, a, b) -> yield! [ "--partition"; Invariant.u64 n + ":" + ms a + ":" + ms b ]
                    | Fault.Sever(n, a) -> yield! [ "--sever"; Invariant.u64 n + ":" + ms a ]
                    | Fault.Kill(n, a) -> yield! [ "--kill"; Invariant.u64 n + ":" + ms a ]
                    | Fault.Restart(n, a) -> yield! [ "--restart"; Invariant.u64 n + ":" + ms a ]
                    | other -> invalidArg "s" (sprintf "the proxy cannot apply %A" other)
            ]

        let psi = (startFor mode) args
        psi.RedirectStandardInput <- true
        let child = new ChildProcess(psi, onLine)

        waitLine "the proxy" 30_000 (fun () -> if port.Value > 0 then Some port.Value else None) child
        |> ignore

        { new ProxySide with
            member _.Port = port.Value
            member _.Stats = stats.Value

            member _.Stop() =
                child.CloseInput()
                child.Finish 10_000 |> ignore
                (child :> IDisposable).Dispose()
        }

    /// The executor options every node of a run gets.
    let private nodeArgs (dir: string) (job: ServingJob) (s: HarnessScenario) (n: HarnessNode) =
        [
            yield! [ "--model"; dir; "--backend"; s.Spec.Backend ]
            yield!
                [
                    "--split"
                    (if job.Boundaries.IsEmpty then
                         "none"
                     else
                         job.Boundaries |> List.map Invariant.i32 |> String.concat ",")
                ]
            yield! [ "--layers"; n.Layers; "--threads"; Invariant.i32 s.Threads ]
            yield! [ "--max-context"; Invariant.i32 (ServingJob.maxPositions job) ]
            yield!
                [
                    "--chains"
                    Invariant.i32 (2 * job.Prompts.Length * (job.Boundaries.Length + 1) + 2)
                ]
            yield! [ "--cost-ms"; Invariant.i32 s.CostMs ]
            if s.FastRebuild then
                yield "--fast-rebuild"
        ]

    let private nodeOptions (dir: string) (job: ServingJob) (s: HarnessScenario) (n: HarnessNode) port incarnation =
        let parsed =
            ServingNode.parse (
                [
                    "--connect"
                    "127.0.0.1:" + Invariant.i32 port
                    "--id"
                    Invariant.u64 n.Id
                    "--name"
                    n.Name
                    "--incarnation"
                    Invariant.u64 incarnation
                ]
                @ nodeArgs dir job s n
            )

        match parsed with
        | Ok o -> o
        | Error e -> invalidArg "n" e

    /// A node actor in this process.
    let private nodeHere
        (dir: string)
        (job: ServingJob)
        (s: HarnessScenario)
        (n: HarnessNode)
        (port: int)
        (incarnation: uint64)
        (run: (string * string option) option)
        : NodeSide =
        let o = nodeOptions dir job s n port incarnation

        let executor =
            match ServingNode.executor (weights dir) o with
            | Ok e -> e
            | Error e -> invalidArg "n" e

        let runLog =
            run
            |> Option.map (fun (runDir, restartOf) ->
                let id =
                    match restartOf with
                    | Some _ -> RunLogNames.restartedNode (NodeId.parse n.Name) (int (incarnation / 1000UL))
                    | None -> NodeId.parse n.Name

                RunContext.Join(
                    runDir,
                    { JoinOptions.create id with
                        ExecutionClass = Some executor.Class
                        RestartOf = restartOf |> Option.map NodeId.parse
                    }
                ))

        let actor = Node(NodeConfig.create n.Id n.Name incarnation, executor)
        let transport = new TcpClientTransport("127.0.0.1", port)
        let host = new ActorHost(actor, transport, SystemClock.Instance)
        let installed = runLog |> Option.map _.Install()
        host.Start()
        installed |> Option.iter _.Dispose()
        let killed = ref false

        let stop () =
            host.Stop()
            (host :> IDisposable).Dispose()
            (transport :> IDisposable).Dispose()
            executor.Dispose()

        { new NodeSide with
            member _.Kill() =
                killed.Value <- true
                stop ()
                runLog |> Option.iter (fun r -> (r :> IDisposable).Dispose())

            member _.Collect() =
                if not killed.Value then
                    actor.Stopped.Wait(10_000) |> ignore

                let report =
                    if actor.Stopped.IsCompletedSuccessfully then
                        Some actor.Stopped.Result
                    else
                        None

                let errors =
                    if host.Completion.IsFaulted then
                        sprintf "%O" host.Completion.Exception
                    else
                        ""

                let state = actor.Describe()

                if not killed.Value then
                    stop ()

                    runLog
                    |> Option.iter (fun r ->
                        r.Finish(
                            if report.IsSome then
                                Lrm.Telemetry.RunStatus.Ok
                            else
                                Lrm.Telemetry.RunStatus.Failed
                        ))

                ({
                    Name = n.Name
                    Incarnation = incarnation
                    Killed = killed.Value
                    ExitCode = None
                    Report = report
                    Errors = errors
                    State = state
                }
                : HarnessNodeRun)
        }

    /// A node as an `lrm node` process.
    let private nodeProcess
        (mode: HarnessMode)
        (dir: string)
        (job: ServingJob)
        (s: HarnessScenario)
        (n: HarnessNode)
        (port: int)
        (incarnation: uint64)
        (run: (string * string option) option)
        : NodeSide =
        let runArgs =
            match run with
            | None -> []
            | Some(runDir, restartOf) ->
                let id =
                    match restartOf with
                    | Some _ -> (RunLogNames.restartedNode (NodeId.parse n.Name) (int (incarnation / 1000UL))).Value
                    | None -> n.Name

                [
                    yield! [ "--run-dir"; runDir; "--run-node"; id ]
                    match restartOf with
                    | Some r -> yield! [ "--restart-of"; r ]
                    | None -> ()
                ]

        let args =
            [
                yield! [ "node"; "--connect"; "127.0.0.1:" + Invariant.i32 port ]
                yield!
                    [
                        "--id"
                        Invariant.u64 n.Id
                        "--name"
                        n.Name
                        "--incarnation"
                        Invariant.u64 incarnation
                    ]
                yield! nodeArgs dir job s n
                yield! [ "--timeout-ms"; Invariant.i32 (s.TimeoutMs + 15_000) ]
                yield! runArgs
            ]

        let child = new ChildProcess((startFor mode) args, ignore)
        let killed = ref false

        { new NodeSide with
            member _.Kill() =
                killed.Value <- true
                child.Kill()

            member _.Collect() =
                let code = child.Finish 15_000

                let r: HarnessNodeRun =
                    {
                        Name = n.Name
                        Incarnation = incarnation
                        Killed = killed.Value
                        ExitCode = (if killed.Value then None else code)
                        Report = child.Output |> List.tryPick Harness.tryParseReport
                        Errors = child.Errors
                        State = ""
                    }

                (child :> IDisposable).Dispose()
                r
        }

    /// Runs one scenario (StagePipeline, "The harness").
    let run (mode: HarnessMode) (s: HarnessScenario) : HarnessResult =
        let clock = Stopwatch.StartNew()
        let dir, j = job s.Spec

        // The harness's own run log: created here when the coordinator runs here; joined as "harness" otherwise.
        let createdRun =
            match mode.RunLog with
            | Some title when not mode.CoordinatorProcess ->
                let parameters =
                    PipelineRun.parameters
                        dir
                        j
                        s.Config
                        [
                            "source", "Lrm.Serving harness"
                            "scenario", s.Name
                            "topology", Topology.name s.Topology
                            "seed", Invariant.u64 s.Seed
                            "nodes", (s.Nodes |> List.map (fun n -> n.Name + ":" + n.Layers) |> String.concat " ")
                            "processes",
                            (if mode.NodeProcesses then "nodes" else "")
                            + (if mode.ProxyProcess then " proxy" else "")
                            + (if mode.CoordinatorProcess then " coordinator" else "")
                            "kills",
                            (s.Kills
                             |> List.map (fun k ->
                                 Invariant.format $"n{k.Node} after {k.AfterUnits} units"
                                 + (match k.RestartAfterMs with
                                    | Some ms -> Invariant.format $", restarted after {ms} ms"
                                    | None -> ""))
                             |> String.concat "; ")
                            "proxy_faults", (s.ProxyFaults |> List.map (sprintf "%A") |> String.concat "; ")
                        ]

                let run =
                    RunContext.Create(
                        { RunOptions.create "cluster" title with
                            Node = NodeId.parse PipelineRun.CoordinatorNode
                            ExecutionClass = Some j.Class
                            GitSha = ServingPaths.gitSha ()
                            Params = parameters
                            RunsRoot = mode.RunsRoot
                        }
                    )

                Some run
            | _ -> None

        let coordinator =
            if mode.CoordinatorProcess then
                coordinatorProcess mode dir j s
            else
                coordinatorHere j s.Config createdRun

        let runDir = coordinator.RunDir

        let markerRun =
            match createdRun, runDir with
            | Some r, _ -> Some r
            | None, Some d -> Some(RunContext.Join(d, NodeId.parse "harness", Some j.Class, None))
            | None, None -> None

        let marker (fault: string) (node: string) (value: float) =
            match markerRun with
            | Some r ->
                r.Telemetry.Counter(
                    ClusterNames.Chaos,
                    value,
                    TelemetryUnits.Count,
                    [ Attr.str "fault" fault; Attr.str TelemetryKeys.Node node ]
                )
            | None -> ()

        let proxy: ProxySide option ref = ref None
        let started = ConcurrentDictionary<int, NodeSide>()
        let finished = ResizeArray<HarnessNodeRun>()
        let restarts = ref 0
        let kills = ref 0
        // The run-log node id of each node's latest incarnation (a restart joins as <name>-r<attempt>).
        let runNodes = Collections.Generic.Dictionary<int, string>()

        let launch (index: int) (incarnation: uint64) (restart: bool) =
            let n = s.Nodes[index - 1]
            let port = proxy.Value.Value.Port

            let restartOf =
                if restart then
                    Some(
                        match runNodes.TryGetValue index with
                        | true, previous -> previous
                        | _ -> n.Name
                    )
                else
                    None

            runNodes[index] <-
                (if restart then
                     (RunLogNames.restartedNode (NodeId.parse n.Name) (int (incarnation / 1000UL))).Value
                 else
                     n.Name)

            let runInfo = runDir |> Option.map (fun d -> d, restartOf)

            let side =
                if mode.NodeProcesses then
                    nodeProcess mode dir j s n port incarnation runInfo
                else
                    nodeHere dir j s n port incarnation runInfo

            started[index] <- side

        let failure =
            try
                proxy.Value <-
                    Some(
                        if mode.ProxyProcess then
                            proxyProcess mode coordinator.Port s
                        else
                            proxyHere coordinator.Port s
                    )

                for i, n in List.indexed s.Nodes do
                    if n.StartAfterUnits = 0 then
                        launch (i + 1) n.Id false

                let killedAt = Collections.Generic.Dictionary<int, int64>()
                let mutable nextKill = 0

                while not coordinator.Finished && clock.ElapsedMilliseconds < int64 s.TimeoutMs do
                    let accepted = coordinator.Accepted

                    // Late nodes (spares) join once enough units are accepted.
                    for i, n in List.indexed s.Nodes do
                        if
                            n.StartAfterUnits > 0
                            && not (started.ContainsKey(i + 1))
                            && accepted >= n.StartAfterUnits
                        then
                            launch (i + 1) n.Id false

                    if nextKill < s.Kills.Length then
                        let k = s.Kills[nextKill]

                        if accepted >= k.AfterUnits then
                            match started.TryGetValue k.Node with
                            | true, side ->
                                side.Kill()
                                finished.Add(side.Collect())
                                started.TryRemove k.Node |> ignore
                                killedAt[k.Node] <- clock.ElapsedMilliseconds
                                kills.Value <- kills.Value + 1
                                marker "kill" s.Nodes[k.Node - 1].Name 1.0
                                nextKill <- nextKill + 1
                            | _ -> ()

                    for k in s.Kills do
                        match k.RestartAfterMs, killedAt.TryGetValue k.Node with
                        | Some delay, (true, at) when clock.ElapsedMilliseconds >= at + int64 delay ->
                            killedAt.Remove k.Node |> ignore
                            restarts.Value <- restarts.Value + 1
                            let n = s.Nodes[k.Node - 1]
                            // A restart is a new incarnation with no state: id + 1000 * attempt.
                            launch k.Node (n.Id + 1000UL * uint64 restarts.Value) true
                            marker "restart" n.Name 1.0
                        | _ -> ()

                    Thread.Sleep 10

                if coordinator.Outcome.IsSome then
                    None
                elif coordinator.Finished then
                    Some coordinator.Failure
                else
                    Some(sprintf "the run did not finish within %d ms: %s" s.TimeoutMs coordinator.Failure)
            with e ->
                Some(sprintf "harness failed: %O" e)

        // Collect every node (after a Shutdown they exit by themselves), then the proxy and the coordinator.
        for KeyValue(_, side) in List.ofSeq started do
            finished.Add(side.Collect())

        let outcome = coordinator.Outcome
        let stats = proxy.Value |> Option.bind (fun p -> p.Stats)

        proxy.Value |> Option.iter (fun p -> p.Stop())
        let stats = proxy.Value |> Option.bind (fun p -> p.Stats) |> Option.orElse stats

        match stats with
        | Some st ->
            marker "drop" "proxy" (float st.Dropped)
            marker "duplicate" "proxy" (float st.Duplicated)
        | None -> ()

        let diagnostics = coordinator.Describe()
        coordinator.Stop()

        let result =
            {
                Scenario = s
                Outcome = (if outcome.IsSome then outcome else coordinator.Outcome)
                Failure = failure
                Nodes = finished |> Seq.sortBy (fun n -> n.Name, n.Incarnation) |> List.ofSeq
                Proxy = stats
                Kills = kills.Value
                Restarts = restarts.Value
                RunDir = runDir
                ElapsedMs = clock.ElapsedMilliseconds
                Diagnostics = diagnostics
            }

        match markerRun with
        | Some r when createdRun.IsSome ->
            // The creator records the comparison with the oracles and finishes the run.
            let identical =
                match result.Outcome with
                | Some o ->
                    let reference = reference (weights dir) j s.Threads

                    do
                        use _ = r.Install()
                        PipelineRun.recordChecks reference o

                    (ServingOutcome.differences reference o).IsEmpty
                | None -> false

            r.Finish(
                if identical && failure.IsNone then
                    Lrm.Telemetry.RunStatus.Ok
                else
                    Lrm.Telemetry.RunStatus.Failed
            )
        | Some r -> r.Finish Lrm.Telemetry.RunStatus.Ok
        | None -> ()

        result

    /// Why a run is not bit-identical to the oracles, one line per problem; empty when it is: no harness failure, the
    /// outcome's digests and ids equal (ServingOutcome.differences), every surviving node exited cleanly and reports
    /// the reference digest for each chain it holds at its final head, and every chain is held at its final head by a
    /// surviving node.
    let verdict (reference: QwenReference) (r: HarnessResult) : string list =
        let finals =
            reference.Pipeline.Chains |> List.map (fun c -> c.Chain, c) |> Map.ofList

        let live = r.Nodes |> List.filter (fun n -> not n.Killed)

        [
            match r.Failure with
            | Some f -> yield f
            | None -> ()
            match r.Outcome with
            | Some o -> yield! ServingOutcome.differences reference o
            | None -> yield "no outcome"
            for n in live do
                let clean =
                    match n.ExitCode with
                    | Some 0 -> true
                    | None -> n.Report.IsSome
                    | Some _ -> false

                if not clean then
                    yield sprintf "%s (incarnation %d) ended with %A: %s" n.Name n.Incarnation n.ExitCode n.Errors

                match n.Report with
                | Some report ->
                    for c in report.Chains do
                        match finals.TryFind c.Chain with
                        | Some f when c.Head = f.Head && c.Digest <> f.Digest ->
                            yield
                                sprintf
                                    "%s holds chain (%d, %d) at its final head with another digest"
                                    n.Name
                                    c.Chain.Lane
                                    c.Chain.Stage
                        | _ -> ()
                | None -> yield sprintf "%s (incarnation %d) sent no report" n.Name n.Incarnation
            for KeyValue(chain, f) in finals do
                let held =
                    live
                    |> List.exists (fun n ->
                        n.Report
                        |> Option.exists (fun rep ->
                            rep.Chains |> List.exists (fun c -> c.Chain = chain && c.Head = f.Head)))

                if not held then
                    yield sprintf "no surviving node holds chain (%d, %d) at its final head" chain.Lane chain.Stage
        ]

    /// Runs scenarios, `degree` at a time, in order of the list; `onResult` sees each result as it ends.
    let sweep
        (modeOf: HarnessScenario -> HarnessMode)
        (degree: int)
        (onResult: HarnessResult -> unit)
        (scenarios: HarnessScenario list)
        =
        let results = ConcurrentDictionary<int, HarnessResult>()
        let gate = obj ()

        Parallel.ForEach(
            List.indexed scenarios,
            ParallelOptions(MaxDegreeOfParallelism = max 1 degree),
            fun (i, s) ->
                let r = run (modeOf s) s
                results[i] <- r
                lock gate (fun () -> onResult r)
        )
        |> ignore

        [ for i in 0 .. scenarios.Length - 1 -> results[i] ]

    /// One line about a run and its verdict.
    let describe (r: HarnessResult) (problems: string list) =
        let o =
            match r.Outcome with
            | Some o -> ServingOutcome.describe o
            | None -> "no outcome"

        let proxy =
            match r.Proxy with
            | Some p ->
                Invariant.format
                    $"proxy {p.Forwarded} forwarded, {p.Dropped} dropped, {p.Duplicated} duplicated, {p.Severed} severed"
            | None -> "no proxy stats"

        sprintf
            "%s: %s in %d ms; %d kill(s), %d restart(s); %s; %s"
            r.Scenario.Name
            (if problems.IsEmpty then "IDENTICAL" else "DIFFERS")
            r.ElapsedMs
            r.Kills
            r.Restarts
            proxy
            o
