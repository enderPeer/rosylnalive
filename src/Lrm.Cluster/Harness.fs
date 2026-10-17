namespace Lrm.Cluster

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.Net
open System.Reflection
open System.Security.Cryptography
open System.Text.Json
open System.Text.Json.Nodes
open System.Threading
open System.Threading.Tasks
open Lrm.Core

/// Options of a fake-executor node process.
type NodeProcessOptions =
    {
        Host: string
        Port: int
        Id: uint64
        Name: string
        Incarnation: uint64
        /// Milliseconds each unit sleeps on the executing thread, to give the work a cost.
        CostMs: int
        Stages: uint32 list
        /// The process gives up after this long without a Shutdown.
        TimeoutMs: int
    }

/// One multi-process run on localhost: a coordinator and a ChaosProxy in this process, node processes behind them.
type LocalRun =
    {
        Job: FakeJob
        Nodes: int
        Replicas: int
        /// The proxy's chaos seed.
        Seed: uint64
        Chaos: ChaosConfig
        /// Partition, Sever, Kill and Restart faults the proxy applies (times from its start).
        ProxyFaults: Fault list
        /// Kill node process `fst` (1-based) once the coordinator has accepted `snd` units.
        Kill: (int * int) option
        CostMs: int
        /// The whole run must finish within this (the CI liveness limit is 60 s).
        TimeoutMs: int
    }

/// What happened to one node process.
type NodeProcessResult =
    {
        Name: string
        Killed: bool
        /// None when the process had to be killed at the end.
        ExitCode: int option
        Report: NodeReport option
        /// Everything the process wrote to stderr.
        Errors: string
    }

/// The result of a multi-process run.
type LocalRunResult =
    {
        Outcome: RunOutcome option
        Reference: FakeReference
        Nodes: NodeProcessResult list
        Proxy: ProxyStats
        Failure: string option
        ElapsedMs: int64
    }

/// Process-level hosting of the cluster for tests and `lrm chaos`: node processes, the local multi-process run and
/// the standalone chaos proxy.
[<RequireQualifiedAccess>]
module Harness =
    /// The prefix of the one line a node process prints with its report.
    [<Literal>]
    let ReportPrefix = "LRM-NODE-REPORT "

    /// Parses "--key value" pairs; keys may repeat.
    let options (args: string list) : Result<(string * string) list, string> =
        let rec go acc rest =
            match rest with
            | [] -> Ok(List.rev acc)
            | (key: string) :: value :: tail when key.StartsWith("--", StringComparison.Ordinal) && key.Length > 2 ->
                go ((key.Substring 2, value) :: acc) tail
            | bad :: _ -> Error(sprintf "unexpected argument '%s' (expected --key value pairs)" bad)

        go [] args

    let private find (opts: (string * string) list) (key: string) =
        opts |> List.tryFindBack (fun (k, _) -> k = key) |> Option.map snd

    let private all (opts: (string * string) list) (key: string) =
        opts |> List.filter (fun (k, _) -> k = key) |> List.map snd

    let private hostPort (text: string) =
        match text.LastIndexOf ':' with
        | i when i > 0 ->
            match Invariant.tryParseI32 (text.Substring(i + 1)) with
            | Some p when p > 0 && p < 65536 -> Ok(text.Substring(0, i), p)
            | _ -> Error(sprintf "bad port in '%s'" text)
        | _ -> Error(sprintf "expected host:port, got '%s'" text)

    let private stagesOf (text: string) =
        let parts = text.Split(',', StringSplitOptions.RemoveEmptyEntries)
        let parsed = parts |> Array.map Invariant.tryParseI32

        if parsed.Length > 0 && parsed |> Array.forall (fun p -> p.IsSome && p.Value >= 0) then
            Ok [ for p in parsed -> uint32 p.Value ]
        else
            Error(sprintf "bad stage list '%s'" text)

    let private randomU64 () =
        BitConverter.ToUInt64(RandomNumberGenerator.GetBytes 8, 0)

    /// Parses node options: --connect host:port --id N --name nN [--incarnation X] [--cost-ms C] [--stages 0,1,2]
    /// [--timeout-ms T].
    let parseNodeOptions (args: string list) : Result<NodeProcessOptions, string> =
        match options args with
        | Error e -> Error e
        | Ok opts ->
            let get key = find opts key

            match get "connect", get "id" |> Option.bind Invariant.tryParseU64, get "name" with
            | None, _, _ -> Error "--connect host:port is required"
            | _, None, _ -> Error "--id N (1 or more) is required"
            | _, _, None -> Error "--name is required"
            | Some connect, Some id, Some name ->
                match hostPort connect, stagesOf (defaultArg (get "stages") "0,1,2") with
                | Error e, _
                | _, Error e -> Error e
                | Ok(host, port), Ok stages ->
                    if id = 0UL then
                        Error "--id must be at least 1"
                    elif not (SafeName.isValid name) then
                        Error(sprintf "--name '%s' is not a safe name" name)
                    else
                        Ok
                            {
                                Host = host
                                Port = port
                                Id = id
                                Name = name
                                Incarnation =
                                    get "incarnation"
                                    |> Option.bind Invariant.tryParseU64
                                    |> Option.defaultWith randomU64
                                CostMs = get "cost-ms" |> Option.bind Invariant.tryParseI32 |> Option.defaultValue 0
                                Stages = stages
                                TimeoutMs =
                                    get "timeout-ms"
                                    |> Option.bind Invariant.tryParseI32
                                    |> Option.defaultValue 120_000
                            }

    /// The report as one line of JSON.
    let reportJson (r: NodeReport) : string =
        let o = JsonObject()
        o["id"] <- JsonValue.Create(Invariant.u64 r.Id)
        o["name"] <- JsonValue.Create r.Name
        o["incarnation"] <- JsonValue.Create(Invariant.u64 r.Incarnation)
        o["replicaDigest"] <- JsonValue.Create r.ReplicaDigest.Hex
        o["executed"] <- JsonValue.Create r.Executed
        o["replayed"] <- JsonValue.Create r.Replayed
        let chains = JsonArray()

        for h in r.Chains do
            let c = JsonObject()
            c["lane"] <- JsonValue.Create h.Chain.Lane
            c["stage"] <- JsonValue.Create h.Chain.Stage
            c["head"] <- JsonValue.Create h.Head.Value.Hex
            c["digest"] <- JsonValue.Create h.Digest.Hex
            chains.Add c

        o["chains"] <- chains
        o.ToJsonString()

    /// Parses a line written by a node process; None when it is not a report.
    let tryParseReport (line: string) : NodeReport option =
        if isNull line || not (line.StartsWith(ReportPrefix, StringComparison.Ordinal)) then
            None
        else
            try
                use doc = JsonDocument.Parse(line.Substring ReportPrefix.Length)
                let root = doc.RootElement
                let str (e: JsonElement) (key: string) = e.GetProperty(key).GetString()

                let chains =
                    [
                        for c in root.GetProperty("chains").EnumerateArray() ->
                            {
                                Chain =
                                    {
                                        Lane = c.GetProperty("lane").GetUInt32()
                                        Stage = c.GetProperty("stage").GetUInt32()
                                    }
                                Head = UnitId(ContentId.parse (str c "head"))
                                Digest = ContentId.parse (str c "digest")
                            }
                    ]

                Some
                    {
                        Id = Invariant.parseU64 (str root "id")
                        Name = str root "name"
                        Incarnation = Invariant.parseU64 (str root "incarnation")
                        Chains = chains
                        ReplicaDigest = ContentId.parse (str root "replicaDigest")
                        Executed = root.GetProperty("executed").GetInt32()
                        Replayed = root.GetProperty("replayed").GetInt32()
                    }
            with _ ->
                None

    /// Runs a fake-executor node until the coordinator shuts the run down; prints the report line. Exit codes: 0 done,
    /// 1 the node failed, 2 timeout.
    let runNode (o: NodeProcessOptions) : int =
        let delay =
            if o.CostMs > 0 then
                (fun (_: WorkUnit) -> Thread.Sleep o.CostMs)
            else
                ignore

        let executor = FakeExecutor(o.Stages, (fun _ -> false), delay)
        let node = Node(NodeConfig.create o.Id o.Name o.Incarnation, executor)
        use transport = new TcpClientTransport(o.Host, o.Port)
        use host = new ActorHost(node, transport, SystemClock.Instance)
        host.Start()

        let first =
            Task.WhenAny(node.Stopped :> Task, host.Completion, Task.Delay o.TimeoutMs).GetAwaiter().GetResult()

        if obj.ReferenceEquals(first, node.Stopped) then
            Console.Out.WriteLine(ReportPrefix + reportJson node.Stopped.Result)
            Console.Out.Flush()
            0
        elif host.Completion.IsFaulted then
            eprintfn "node %s failed: %O" o.Name host.Completion.Exception
            1
        else
            eprintfn "node %s: no shutdown within %d ms" o.Name o.TimeoutMs
            2

    /// A ProcessStartInfo that runs this process's entry assembly again under the same dotnet host, as
    /// `dotnet <entry dll> <prefix> <args>`, with stdout and stderr redirected and `env` added.
    let selfStartInfo (prefix: string list) (env: (string * string) list) (args: string list) : ProcessStartInfo =
        let host =
            match Environment.ProcessPath with
            | null -> "dotnet"
            | p -> p

        let psi = ProcessStartInfo host
        psi.ArgumentList.Add(Assembly.GetEntryAssembly().Location)

        for a in prefix @ args do
            psi.ArgumentList.Add a

        psi.UseShellExecute <- false
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.CreateNoWindow <- true

        for k, v in env do
            psi.Environment[k] <- v

        psi

    /// The standard local run: the standard job's 3 stages on 3 node processes with 2 replicas each (every node hosts
    /// two stages), mild proxy chaos, node 2 killed after a third of the units, 60 s to finish.
    let standardLocalRun (seed: uint64) : LocalRun =
        let job =
            { FakeJob.standard with
                Lanes = 2
                Steps = 10
            }

        {
            Job = job
            Nodes = 3
            Replicas = 2
            Seed = seed
            Chaos = ChaosConfig.proxy
            ProxyFaults = []
            Kill = Some(2, job.Lanes * job.Steps * job.Stages / 3)
            CostMs = 2
            TimeoutMs = 60_000
        }

    /// Runs a coordinator and a ChaosProxy here and `run.Nodes` node processes started by `start` (given the node
    /// arguments), kills one node part-way through if asked, and collects the outcome and the node reports.
    let runLocal (start: string list -> ProcessStartInfo) (run: LocalRun) : LocalRunResult =
        let clock = Stopwatch.StartNew()
        let reference = FakeReference.run run.Job

        let config =
            { CoordinatorConfig.standard with
                Replicas = run.Replicas
                MinNodes = run.Nodes
            }

        let coordinator = Coordinator(config, FakeWorkload run.Job, MemoryBlobStore())
        use server = new TcpServerTransport(0)
        let policy = ChaosPolicy(run.Seed, run.Chaos, run.ProxyFaults)

        use proxy =
            new ChaosProxy(
                IPEndPoint(IPAddress.Loopback, 0),
                IPEndPoint(IPAddress.Loopback, server.Port),
                policy,
                SystemClock.Instance
            )

        proxy.Start()
        use host = new ActorHost(coordinator, server, SystemClock.Instance)
        host.Start()

        let stages =
            String.Join(",", [ for s in 0 .. run.Job.Stages - 1 -> Invariant.i32 s ])

        let processes =
            [
                for i in 1 .. run.Nodes do
                    let name = "n" + Invariant.i32 i

                    let psi =
                        start
                            [
                                "node"
                                "--connect"
                                "127.0.0.1:" + Invariant.i32 proxy.Port
                                "--id"
                                Invariant.i32 i
                                "--name"
                                name
                                "--incarnation"
                                Invariant.i32 i
                                "--cost-ms"
                                Invariant.i32 run.CostMs
                                "--stages"
                                stages
                                "--timeout-ms"
                                Invariant.i32 (run.TimeoutMs + 10_000)
                            ]

                    let output = ConcurrentQueue<string>()
                    let errors = ConcurrentQueue<string>()
                    let p = new Process(StartInfo = psi)

                    p.OutputDataReceived.Add(fun e ->
                        if not (isNull e.Data) then
                            output.Enqueue e.Data)

                    p.ErrorDataReceived.Add(fun e ->
                        if not (isNull e.Data) then
                            errors.Enqueue e.Data)

                    p.Start() |> ignore
                    p.BeginOutputReadLine()
                    p.BeginErrorReadLine()
                    yield name, p, output, errors
            ]

        let mutable killed: int option = None
        let deadline = int64 run.TimeoutMs

        while not coordinator.Completion.IsCompleted
              && not host.Completion.IsCompleted
              && clock.ElapsedMilliseconds < deadline do
            match run.Kill, killed with
            | Some(index, after), None when coordinator.AcceptedCount >= after ->
                let _, p, _, _ = processes[index - 1]

                try
                    p.Kill true
                with _ ->
                    ()

                killed <- Some index
            | _ -> ()

            Thread.Sleep 10

        let failure =
            if coordinator.Completion.IsCompletedSuccessfully then
                None
            elif host.Completion.IsFaulted then
                Some(sprintf "coordinator failed: %O" host.Completion.Exception)
            else
                Some(sprintf "the run did not finish within %d ms" run.TimeoutMs)

        let outcome =
            if coordinator.Completion.IsCompletedSuccessfully then
                Some coordinator.Completion.Result
            else
                None

        let nodes =
            [
                for i, (name, p, output, errors) in List.indexed processes do
                    let exited = p.WaitForExit 10_000

                    if not exited then
                        try
                            p.Kill true
                        with _ ->
                            ()

                    p.WaitForExit()

                    yield
                        {
                            Name = name
                            Killed = (killed = Some(i + 1))
                            ExitCode = (if exited then Some p.ExitCode else None)
                            Report = output |> Seq.tryPick tryParseReport
                            Errors = String.Join("\n", errors)
                        }

                    p.Dispose()
            ]

        host.Stop()

        {
            Outcome = outcome
            Reference = reference
            Nodes = nodes
            Proxy = proxy.Stats
            Failure = failure
            ElapsedMs = clock.ElapsedMilliseconds
        }

    let private faultOf (kind: string) (text: string) : Result<Fault, string> =
        let parts = text.Split ':'

        let nums =
            parts |> Array.map Invariant.tryParseI64 |> Array.choose id |> Array.map int64

        if nums.Length <> parts.Length || nums.Length = 0 || nums[0] < 1L then
            Error(sprintf "bad --%s '%s'" kind text)
        else
            let node = uint64 nums[0]

            match kind, nums.Length with
            | "partition", 3 -> Ok(Fault.Partition(node, Ns.ms nums[1], Ns.ms nums[2]))
            | "sever", 2 -> Ok(Fault.Sever(node, Ns.ms nums[1]))
            | "kill", 2 -> Ok(Fault.Kill(node, Ns.ms nums[1]))
            | "restart", 2 -> Ok(Fault.Restart(node, Ns.ms nums[1]))
            | _ -> Error(sprintf "bad --%s '%s' (node:ms, or node:fromMs:untilMs for a partition)" kind text)

    /// Parses chaos options: --chaos none|lan|proxy|standard, then optional overrides --drop, --duplicate,
    /// --reorder (permille), --min-delay-ms, --max-delay-ms, --reorder-delay-ms.
    let parseChaos (opts: (string * string) list) (fallback: ChaosConfig) : Result<ChaosConfig, string> =
        let baseConfig =
            match find opts "chaos" with
            | None -> Ok fallback
            | Some "none" -> Ok ChaosConfig.none
            | Some "lan" -> Ok ChaosConfig.lan
            | Some "proxy" -> Ok ChaosConfig.proxy
            | Some "standard" -> Ok ChaosConfig.standard
            | Some other -> Error(sprintf "unknown --chaos '%s' (none, lan, proxy, standard)" other)

        let int key (f: int -> ChaosConfig -> ChaosConfig) (c: Result<ChaosConfig, string>) =
            match c, find opts key with
            | Ok c, Some v ->
                match Invariant.tryParseI32 v with
                | Some n -> Ok(f n c)
                | None -> Error(sprintf "--%s expects an integer" key)
            | c, _ -> c

        baseConfig
        |> int "drop" (fun n c -> { c with DropPermille = n })
        |> int "duplicate" (fun n c -> { c with DuplicatePermille = n })
        |> int "reorder" (fun n c -> { c with ReorderPermille = n })
        |> int "min-delay-ms" (fun n c -> { c with MinDelayNs = Ns.ms (int64 n) })
        |> int "max-delay-ms" (fun n c -> { c with MaxDelayNs = Ns.ms (int64 n) })
        |> int "reorder-delay-ms" (fun n c ->
            { c with
                ReorderDelayNs = Ns.ms (int64 n)
            })
        |> Result.bind ChaosConfig.validate

    /// `lrm chaos-proxy`: --listen PORT --target HOST:PORT [--seed N] [chaos options] [--partition node:fromMs:untilMs]
    /// [--sever node:ms] [--kill node:ms] [--restart node:ms] [--duration-ms N]. Runs until the duration ends, or until
    /// stdin closes or Ctrl+C when no duration is given.
    let runProxy (args: string list) : int =
        let parsed =
            match options args with
            | Error e -> Error e
            | Ok opts ->
                let faults =
                    [ "partition"; "sever"; "kill"; "restart" ]
                    |> List.collect (fun k -> all opts k |> List.map (faultOf k))

                match
                    List.tryPick
                        (function
                        | Error e -> Some e
                        | Ok _ -> None)
                        faults
                with
                | Some e -> Error e
                | None ->
                    match find opts "listen" |> Option.bind Invariant.tryParseI32, find opts "target" with
                    | None, _ -> Error "--listen PORT is required"
                    | _, None -> Error "--target HOST:PORT is required"
                    | Some listen, Some target ->
                        match hostPort target, parseChaos opts ChaosConfig.proxy with
                        | Error e, _
                        | _, Error e -> Error e
                        | Ok(host, port), Ok chaos ->
                            let addresses = Dns.GetHostAddresses host

                            if addresses.Length = 0 then
                                Error(sprintf "cannot resolve '%s'" host)
                            else
                                Ok(
                                    listen,
                                    IPEndPoint(addresses[0], port),
                                    chaos,
                                    find opts "seed" |> Option.bind Invariant.tryParseU64 |> Option.defaultValue 1UL,
                                    faults
                                    |> List.choose (function
                                        | Ok f -> Some f
                                        | Error _ -> None),
                                    find opts "duration-ms" |> Option.bind Invariant.tryParseI32
                                )

        match parsed with
        | Error e ->
            eprintfn "lrm chaos-proxy: %s" e
            2
        | Ok(listen, target, chaos, seed, faults, duration) ->
            let policy = ChaosPolicy(seed, chaos, faults)

            use proxy =
                new ChaosProxy(IPEndPoint(IPAddress.Loopback, listen), target, policy, SystemClock.Instance)

            proxy.Start()
            printfn "lrm chaos-proxy: 127.0.0.1:%d -> %O, seed %d, %d fault(s)" proxy.Port target seed faults.Length
            Console.Out.Flush()

            match duration with
            | Some ms -> Thread.Sleep ms
            | None ->
                use stop = new ManualResetEventSlim false

                Console.CancelKeyPress.Add(fun e ->
                    e.Cancel <- true
                    stop.Set())

                Task.Run(fun () ->
                    try
                        while not (isNull (Console.In.ReadLine())) do
                            ()
                    with _ ->
                        ()

                    stop.Set())
                |> ignore

                stop.Wait()

            let s = proxy.Stats

            printfn
                "lrm chaos-proxy: %d connection(s), %d frame(s) forwarded, %d dropped, %d duplicated, %d severed"
                s.Connections
                s.Forwarded
                s.Dropped
                s.Duplicated
                s.Severed

            0

    /// The worker entry point of a test exe or `lrm chaos`: `node <options>` runs a node process.
    let workerMain (args: string list) : int =
        match args with
        | "node" :: rest ->
            match parseNodeOptions rest with
            | Ok o -> runNode o
            | Error e ->
                eprintfn "cluster worker: %s" e
                2
        | _ ->
            eprintfn "cluster worker: expected 'node <options>'"
            2
