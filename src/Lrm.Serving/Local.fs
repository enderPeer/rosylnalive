namespace Lrm.Serving

open System
open System.Diagnostics
open Lrm.Core
open Lrm.Cluster

/// The options of `lrm coord local`.
type LocalOptions =
    {
        Spec: JobSpec
        /// None: each seed's topology in turn (Topology.ofSeed).
        Topology: Topology option
        Seeds: uint64 list
        Chaos: ChaosConfig
        /// Explicit kills instead of the seed's.
        Kills: HarnessKill list option
        /// Explicit proxy faults instead of the seed's.
        ProxyFaults: Fault list option
        /// Frame chaos only: no kill, partition or sever.
        NoFaults: bool
        NodeProcesses: bool
        ProxyProcess: bool
        CoordinatorProcess: bool
        CostMs: int
        Threads: int
        TimeoutMs: int
        FastRebuild: bool
        RunLog: bool
        Title: string option
        /// Runs at once in a sweep.
        Degree: int
    }

/// `lrm coord local`: the harness from the command line (StagePipeline, "Commands").
[<RequireQualifiedAccess>]
module ServingLocal =
    let private chaosKeys =
        [
            "chaos"
            "drop"
            "duplicate"
            "reorder"
            "min-delay-ms"
            "max-delay-ms"
            "reorder-delay-ms"
        ]

    /// The value options of `lrm coord local`.
    let values =
        JobSpec.values
        @ chaosKeys
        @ [
            "topology"
            "seed"
            "seeds"
            "kill"
            "partition"
            "sever"
            "nodes"
            "proxy"
            "coordinator"
            "cost-ms"
            "threads"
            "timeout-ms"
            "title"
            "parallel"
        ]

    /// The switches of `lrm coord local`.
    let switches = JobSpec.switches @ [ "no-faults"; "fast-rebuild"; "run-log" ]

    let private parseSeeds (text: string) : Result<uint64 list, string> =
        let parts = text.Split(',', StringSplitOptions.RemoveEmptyEntries)

        let one (p: string) =
            match p.Split('-') with
            | [| a |] -> Invariant.tryParseU64 (a.Trim()) |> Option.map List.singleton
            | [| a; b |] ->
                match Invariant.tryParseU64 (a.Trim()), Invariant.tryParseU64 (b.Trim()) with
                | Some a, Some b when a <= b && b - a < 100_000UL -> Some [ a..b ]
                | _ -> None
            | _ -> None

        let parsed = parts |> Array.map one

        if parsed.Length = 0 || parsed |> Array.exists Option.isNone then
            Error(sprintf "bad --seeds '%s' (expected 1-50 or 3,7,9)" text)
        else
            Ok(parsed |> Array.toList |> List.collect Option.get)

    let private ints (text: string) =
        let parts = text.Split ':'
        let parsed = parts |> Array.map Invariant.tryParseI64

        if parsed |> Array.forall Option.isSome then
            Some(parsed |> Array.map Option.get)
        else
            None

    let private kill (text: string) : Result<HarnessKill, string> =
        match ints text with
        | Some [| n; after |] when n >= 1L && after >= 0L ->
            Ok
                {
                    Node = int n
                    AfterUnits = int after
                    RestartAfterMs = None
                }
        | Some [| n; after; restart |] when n >= 1L && after >= 0L && restart >= 0L ->
            Ok
                {
                    Node = int n
                    AfterUnits = int after
                    RestartAfterMs = Some(int restart)
                }
        | _ -> Error(sprintf "bad --kill '%s' (node:afterUnits[:restartAfterMs])" text)

    let private proxyFault (kind: string) (text: string) : Result<Fault, string> =
        match kind, ints text with
        | "partition", Some [| n; a; b |] when n >= 1L && 0L <= a && a < b ->
            Ok(Fault.Partition(uint64 n, Ns.ms a, Ns.ms b))
        | "sever", Some [| n; a |] when n >= 1L && a >= 0L -> Ok(Fault.Sever(uint64 n, Ns.ms a))
        | _ -> Error(sprintf "bad --%s '%s' (node:fromMs:untilMs for a partition, node:ms for a sever)" kind text)

    let private all (f: string -> Result<'a, string>) (xs: string list) : Result<'a list, string> =
        xs
        |> List.fold
            (fun acc x ->
                match acc, f x with
                | Ok ys, Ok y -> Ok(y :: ys)
                | Error e, _
                | _, Error e -> Error e)
            (Ok [])
        |> Result.map List.rev

    let private where (a: CliArgs) (key: string) (fallback: bool) : Result<bool, string> =
        match a.Get key with
        | None -> Ok fallback
        | Some "process" -> Ok true
        | Some "here" -> Ok false
        | Some other -> Error(sprintf "--%s '%s' (expected process or here)" key other)

    /// Parses the options of `lrm coord local`.
    let parse (args: string list) : Result<LocalOptions, string> =
        result {
            let! a = CliArgs.parse values switches args
            let! spec = JobSpec.parse a

            let! topology =
                match a.Get "topology" with
                | None -> Ok None
                | Some t ->
                    match Topology.tryParse t with
                    | Some x -> Ok(Some x)
                    | None -> Error(sprintf "unknown --topology '%s' (split2, replicated3, bothlost5)" t)

            let! seeds =
                match a.Get "seeds", a.Get "seed" with
                | Some s, _ -> parseSeeds s
                | None, Some s -> parseSeeds s
                | None, None -> Ok [ 1UL ]

            let! chaos =
                Harness.parseChaos (a.Values |> List.filter (fun (k, _) -> List.contains k chaosKeys)) ChaosConfig.proxy

            let! kills =
                match a.All "kill" with
                | [] -> Ok None
                | ks -> all kill ks |> Result.map Some

            let! proxyFaults =
                match a.All "partition", a.All "sever" with
                | [], [] -> Ok None
                | ps, ss ->
                    result {
                        let! p = all (proxyFault "partition") ps
                        let! s = all (proxyFault "sever") ss
                        return Some(p @ s)
                    }

            let! nodes = where a "nodes" true
            let! proxy = where a "proxy" true
            let! coordinator = where a "coordinator" false
            let! cost = CliArgs.int a "cost-ms" 2
            let! threads = CliArgs.int a "threads" 1
            let! timeout = CliArgs.int a "timeout-ms" 60_000
            let! degree = CliArgs.int a "parallel" 1

            if coordinator && not nodes then
                return! Error "--coordinator process needs --nodes process"
            else
                return
                    {
                        Spec = spec
                        Topology = topology
                        Seeds = seeds
                        Chaos = chaos
                        Kills = kills
                        ProxyFaults = proxyFaults
                        NoFaults = a.Has "no-faults"
                        NodeProcesses = nodes
                        ProxyProcess = proxy
                        CoordinatorProcess = coordinator
                        CostMs = max 0 cost
                        Threads = max 1 threads
                        TimeoutMs = max 1000 timeout
                        FastRebuild = a.Has "fast-rebuild"
                        RunLog = a.Has "run-log"
                        Title = a.Get "title"
                        Degree = max 1 degree
                    }
        }

    /// The scenario of one seed.
    let scenario (o: LocalOptions) (seed: uint64) : HarnessScenario =
        let t = defaultArg o.Topology (Topology.ofSeed seed)
        let startMs = if o.NodeProcesses then 1000 else 200

        let b =
            if o.NoFaults then
                ServingHarness.clean o.Spec t seed
            else
                ServingHarness.chaos o.Spec t seed startMs

        { b with
            Chaos = o.Chaos
            Kills = (if o.NoFaults then [] else defaultArg o.Kills b.Kills)
            ProxyFaults =
                (if o.NoFaults then
                     []
                 else
                     defaultArg o.ProxyFaults b.ProxyFaults)
            CostMs = o.CostMs
            Threads = o.Threads
            TimeoutMs = o.TimeoutMs
            FastRebuild = o.FastRebuild
        }

    /// Runs the options' scenarios and prints one line per run and a summary. 0 when every run is bit-identical to
    /// the oracles.
    let run (start: string list -> ProcessStartInfo) (o: LocalOptions) : int =
        let dir, job = ServingHarness.job o.Spec
        let reference = ServingHarness.reference (ServingHarness.weights dir) job o.Threads
        let clock = Stopwatch.StartNew()

        printfn
            "lrm coord local: job %s, %d lanes, %d steps, split %A on %s; oracles: run digest %s, logits digest %s"
            reference.JobId.Short
            job.Prompts.Length
            job.Steps
            job.Boundaries
            o.Spec.Backend
            reference.Pipeline.RunDigest.Short
            reference.Single.LogitsDigest.Short

        let mode (s: HarnessScenario) : HarnessMode =
            {
                NodeProcesses = o.NodeProcesses
                ProxyProcess = o.ProxyProcess
                CoordinatorProcess = o.CoordinatorProcess
                Start = Some start
                RunLog =
                    (if o.RunLog then
                         Some(
                             defaultArg
                                 o.Title
                                 (sprintf
                                     "Pipeline on localhost: %s (%s)"
                                     s.Name
                                     (if o.NodeProcesses then "node processes" else "node actors"))
                         )
                     else
                         None)
                RunsRoot = None
            }

        let scenarios = o.Seeds |> List.map (scenario o)
        let bad = ref 0

        let report (r: HarnessResult) =
            let problems = ServingHarness.verdict reference r

            if not problems.IsEmpty then
                bad.Value <- bad.Value + 1

            printfn "  %s" (ServingHarness.describe r problems)

            for p in problems do
                printfn "    %s" p

            match r.RunDir with
            | Some d -> printfn "    run log %s (lrm report %s)" d (IO.Path.GetFileName d)
            | None -> ()

            Console.Out.Flush()

        ServingHarness.sweep mode o.Degree report scenarios |> ignore
        let bad = bad.Value

        printfn
            "lrm coord local: %d of %d run(s) bit-identical to the single-process run, in %d ms"
            (scenarios.Length - bad)
            scenarios.Length
            clock.ElapsedMilliseconds

        if bad = 0 then 0 else 1

    /// `lrm coord local <options>`.
    let main (start: string list -> ProcessStartInfo) (args: string list) : int =
        match parse args with
        | Ok o ->
            try
                run start o
            with :? ArgumentException as e ->
                eprintfn "lrm coord local: %s" e.Message
                2
        | Error e ->
            eprintfn "lrm coord local: %s" e
            2

/// The worker entry point of the serving processes, for `lrm` and test exes (StagePipeline, "Commands").
[<RequireQualifiedAccess>]
module ServingWorker =
    /// `node ...`, `coord run ...` or `chaos-proxy ...`.
    let main (args: string list) : int =
        match args with
        | "node" :: rest -> ServingNode.main rest
        | "coord" :: "run" :: rest -> ServingCoordinator.main rest
        | "chaos-proxy" :: rest -> Harness.runProxy rest
        | _ ->
            eprintfn "serving worker: expected 'node', 'coord run' or 'chaos-proxy' and options"
            2
