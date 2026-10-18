namespace Lrm.Serving

open System
open System.Diagnostics
open System.Threading
open Lrm.Core
open Lrm.Tensor
open Lrm.Telemetry
open Lrm.Cluster
open Lrm.Models.Qwen

/// The cluster settings of a pipeline run (StagePipeline, "The coordinator").
type PipelineConfig =
    {
        /// Replicas per stage: 1 for a two-node split (one node per stage), 2 for failover without catch-up.
        Replicas: int
        /// Members needed before the run starts.
        MinNodes: int
        /// The longest drain after the last unit (C1's DrainNs; 10 s here, because a verifier may replay a chain).
        DrainNs: int64
    }

[<RequireQualifiedAccess>]
module PipelineConfig =
    /// One replica per stage on two nodes, a 10 s drain.
    let split2 =
        {
            Replicas = 1
            MinNodes = 2
            DrainNs = Ns.seconds 10L
        }

    /// C1's coordinator settings for a pipeline run.
    let coordinator (p: PipelineConfig) =
        { CoordinatorConfig.standard with
            Replicas = p.Replicas
            MinNodes = p.MinNodes
            DrainNs = p.DrainNs
        }

/// The run log of a pipeline run, as the coordinator's process creates it (RunLog contract, kind cluster).
[<RequireQualifiedAccess>]
module PipelineRun =
    /// The creator's node id: the coordinator's telemetry name.
    [<Literal>]
    let CoordinatorNode = "coord"

    /// The parameters of a job for the run manifest.
    let parameters (modelDir: string) (job: ServingJob) (p: PipelineConfig) (extra: (string * string) list) =
        [
            "model", ServingPaths.display modelDir
            "model_id", job.Model.Hex
            "job", (ServingJob.id job).Hex
            "backend", ExecutionBackend.name job.Class.Backend
            "split", (job.Boundaries |> List.map Invariant.i32 |> String.concat ",")
            "lanes", Invariant.i32 job.Prompts.Length
            "prompt_tokens", (job.Prompts |> List.map (fun p -> Invariant.i32 p.Length) |> String.concat ",")
            "steps", Invariant.i32 job.Steps
            "stop_at_eos", (if job.StopAtEos then "true" else "false")
            "replicas", Invariant.i32 p.Replicas
            "min_nodes", Invariant.i32 p.MinNodes
            yield! extra
        ]

    /// Creates the run of a pipeline job, with the coordinator as its creating node.
    let create (title: string) (job: ServingJob) (parameters: (string * string) list) =
        RunContext.Create(
            { RunOptions.create "cluster" title with
                Node = NodeId.parse CoordinatorNode
                ExecutionClass = Some job.Class
                GitSha = ServingPaths.gitSha ()
                Params = parameters
            }
        )

    /// Records the comparisons of a finished run with the oracles as golden.compare `exact` checks with a `check` and
    /// no `tap` (the parity report's self-test table), so the run log says whether the pipeline was bit-identical.
    let recordChecks (reference: QwenReference) (o: ServingOutcome) =
        let count (a: 'x list) (b: 'x list) =
            let n = max a.Length b.Length

            [ 0 .. n - 1 ]
            |> List.filter (fun i -> i >= a.Length || i >= b.Length || a[i] <> b[i])
            |> List.length

        let digestCheck (name: string) (mine: ContentId) (theirs: ContentId) =
            Compare.check
                "exact"
                BoundOp.AtMost
                (if mine = theirs then 0.0 else 1.0)
                0.0
                1L
                [
                    Attr.str "check" name
                    Attr.str "kind" "determinism"
                    Attr.str "detail" (mine.Short + " vs " + theirs.Short)
                ]
            |> ignore

        digestCheck "cluster_run_digest_vs_plan_order" o.RunDigest reference.Pipeline.RunDigest
        digestCheck "cluster_replica_digest_vs_plan_order" o.ReplicaDigest reference.Pipeline.ReplicaDigest
        digestCheck "cluster_logits_vs_single_process" o.LogitsDigest reference.Single.LogitsDigest
        let mine = o.Tokens |> List.concat
        let theirs = reference.Single.Tokens |> List.concat

        Compare.check
            "exact"
            BoundOp.AtMost
            (float (count mine theirs))
            0.0
            (int64 (max mine.Length theirs.Length))
            [
                Attr.str "check" "cluster_greedy_ids_vs_single_process"
                Attr.str "kind" "determinism"
            ]
        |> ignore

/// A coordinator of one pipeline job on this process, listening on TCP (StagePipeline, "The coordinator").
[<Sealed>]
type PipelineCoordinator(job: ServingJob, config: PipelineConfig, listenPort: int) =
    let workload = QwenWorkload job

    let coordinator =
        Coordinator(PipelineConfig.coordinator config, workload, MemoryBlobStore())

    let server = new TcpServerTransport(listenPort)
    let host = new ActorHost(coordinator, server, SystemClock.Instance)

    /// The port nodes (or a proxy) connect to.
    member _.Port = server.Port

    member _.Workload = workload
    member _.Coordinator = coordinator

    /// Units accepted so far.
    member _.Accepted = coordinator.AcceptedCount

    /// Starts the actor. The ambient telemetry facade at this call is the coordinator's for the whole run.
    member _.Start() = host.Start()

    /// True once the run finished or the actor failed.
    member _.IsFinished = coordinator.Completion.IsCompleted || host.Completion.IsCompleted

    /// The outcome, once the run finished.
    member _.Outcome: ServingOutcome option =
        if coordinator.Completion.IsCompletedSuccessfully then
            Some(ServingOutcome.ofRun coordinator.Completion.Result workload.Result)
        else
            None

    /// Why the run has no outcome: the actor's exception, or a timeout.
    member _.Failure(timeoutMs: int) : string =
        if host.Completion.IsFaulted then
            sprintf "coordinator failed: %O" host.Completion.Exception
        else
            sprintf "the run did not finish within %d ms\n%s" timeoutMs (coordinator.Describe())

    /// Waits up to `timeoutMs` for the run to finish, calling `tick` every 20 ms with the units accepted so far.
    member c.Wait(timeoutMs: int, tick: int -> unit) : bool =
        let clock = Stopwatch.StartNew()

        while not c.IsFinished && clock.ElapsedMilliseconds < int64 timeoutMs do
            tick coordinator.AcceptedCount
            Thread.Sleep 20

        tick coordinator.AcceptedCount
        coordinator.Completion.IsCompletedSuccessfully

    interface IDisposable with
        member _.Dispose() =
            host.Stop()
            (host :> IDisposable).Dispose()
            (server :> ITransport).Dispose()

/// The options of `lrm coord run`.
type CoordRunOptions =
    {
        Spec: JobSpec
        Listen: int
        Config: PipelineConfig
        TimeoutMs: int
        /// Compute both oracles here and compare (StagePipeline, "The reference").
        Verify: bool
        Threads: int
        RunLog: bool
        Title: string option
        /// Print progress lines (LRM-COORD-PROGRESS) for a harness.
        Progress: bool
    }

/// `lrm coord run`: the coordinator of one pipeline job as its own process.
[<RequireQualifiedAccess>]
module ServingCoordinator =
    /// The line prefixes a coordinator process prints for a harness.
    [<Literal>]
    let ListenPrefix = "LRM-COORD-LISTEN "

    [<Literal>]
    let RunPrefix = "LRM-COORD-RUN "

    [<Literal>]
    let ProgressPrefix = "LRM-COORD-PROGRESS "

    [<Literal>]
    let JobPrefix = "LRM-COORD-JOB "

    /// The value options of `lrm coord run`.
    let values =
        JobSpec.values
        @ [ "listen"; "replicas"; "nodes"; "drain-ms"; "timeout-ms"; "threads"; "title" ]

    /// The switches of `lrm coord run`.
    let switches = JobSpec.switches @ [ "verify"; "run-log"; "progress" ]

    /// Parses the options of `lrm coord run`.
    let parse (args: string list) : Result<CoordRunOptions, string> =
        result {
            let! a = CliArgs.parse values switches args
            let! spec = JobSpec.parse a
            let! listen = CliArgs.int a "listen" 0
            let! replicas = CliArgs.int a "replicas" 1
            let! nodes = CliArgs.int a "nodes" 2
            let! drain = CliArgs.int a "drain-ms" 10_000
            let! timeout = CliArgs.int a "timeout-ms" 600_000
            let! threads = CliArgs.int a "threads" (min 8 Environment.ProcessorCount)

            if
                replicas < 1
                || nodes < 1
                || drain < 0
                || timeout < 1
                || listen < 0
                || listen > 65535
            then
                return! Error "--replicas, --nodes and --timeout-ms must be at least 1, --drain-ms 0 or more"
            else
                return
                    {
                        Spec = spec
                        Listen = listen
                        Config =
                            {
                                Replicas = replicas
                                MinNodes = nodes
                                DrainNs = Ns.ms (int64 drain)
                            }
                        TimeoutMs = timeout
                        Verify = a.Has "verify"
                        Threads = max 1 threads
                        RunLog = a.Has "run-log"
                        Title = a.Get "title"
                        Progress = a.Has "progress"
                    }
        }

    /// Runs the coordinator until the run finishes; prints the outcome line. Exit codes: 0 completed (and identical to
    /// the oracles with --verify), 1 otherwise, 2 bad options.
    let run (o: CoordRunOptions) : int =
        let opened =
            result {
                let! dir = ServingPaths.modelDir o.Spec.Model

                let! weights =
                    try
                        Ok(Qwen3Weights.Open dir)
                    with e ->
                        Error(sprintf "cannot open the model %s: %s" dir e.Message)

                match JobSpec.build weights o.Spec with
                | Ok job -> return dir, weights, job
                | Error e ->
                    weights.Dispose()
                    return! Error e
            }

        match opened with
        | Error e ->
            eprintfn "lrm coord: %s" e
            2
        | Ok(dir, weights, job) ->
            use weights = weights
            let jobId = ServingJob.id job

            let title =
                defaultArg
                    o.Title
                    (sprintf
                        "Pipeline of %s on %s: %d lanes, %d steps, split %A"
                        (IO.Path.GetFileName dir)
                        o.Spec.Backend
                        job.Prompts.Length
                        job.Steps
                        job.Boundaries)

            let runLog =
                if o.RunLog then
                    Some(
                        PipelineRun.create
                            title
                            job
                            (PipelineRun.parameters dir job o.Config [ "source", "lrm coord run" ])
                    )
                else
                    None

            try
                let installed = runLog |> Option.map _.Install()
                use coordinator = new PipelineCoordinator(job, o.Config, o.Listen)
                coordinator.Start()
                installed |> Option.iter _.Dispose()
                printfn "%s%d" ListenPrefix coordinator.Port
                printfn "%s%s" JobPrefix jobId.Hex

                match runLog with
                | Some r -> printfn "%s%s" RunPrefix r.Directory
                | None -> ()

                Console.Out.Flush()
                let last = ref -1

                let finished =
                    coordinator.Wait(
                        o.TimeoutMs,
                        fun n ->
                            if o.Progress && n <> last.Value then
                                last.Value <- n
                                Console.Out.WriteLine(ProgressPrefix + Invariant.i32 n)
                                Console.Out.Flush()
                    )

                match coordinator.Outcome with
                | Some outcome when finished ->
                    printfn "lrm coord: %s" (ServingOutcome.describe outcome)

                    let identical =
                        if o.Verify then
                            let reference = QwenReference.cpu weights job o.Threads

                            match runLog with
                            | Some r ->
                                use _ = r.Install()
                                PipelineRun.recordChecks reference outcome
                            | None -> ()

                            match ServingOutcome.differences reference outcome with
                            | [] ->
                                printfn "lrm coord: bit-identical to the plan-order run and the single-process run"
                                true
                            | ds ->
                                for d in ds do
                                    printfn "lrm coord: DIFFERS: %s" d

                                false
                        else
                            outcome.Status = "completed"

                    Console.Out.WriteLine(ServingOutcome.Prefix + ServingOutcome.toJson outcome)
                    Console.Out.Flush()

                    match runLog with
                    | Some r ->
                        r.Finish(
                            if identical then
                                Lrm.Telemetry.RunStatus.Ok
                            else
                                Lrm.Telemetry.RunStatus.Failed
                        )

                        printfn "lrm coord: run log %s (lrm report %s)" r.Directory (IO.Path.GetFileName r.Directory)
                    | None -> ()

                    if identical && outcome.Status = "completed" then 0 else 1
                | _ ->
                    eprintfn "lrm coord: %s" (coordinator.Failure o.TimeoutMs)
                    runLog |> Option.iter (fun r -> r.Finish Lrm.Telemetry.RunStatus.Failed)
                    1
            finally
                runLog |> Option.iter (fun r -> (r :> IDisposable).Dispose())

    /// `lrm coord run <options>`.
    let main (args: string list) : int =
        match parse args with
        | Ok o -> run o
        | Error e ->
            eprintfn "lrm coord run: %s" e
            2
