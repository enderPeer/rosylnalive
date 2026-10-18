namespace Lrm.Serving

open System
open System.Threading.Tasks
open Lrm.Core
open Lrm.Telemetry
open Lrm.Cluster
open Lrm.Models.Qwen

/// The options of a pipeline node process (`lrm node`, StagePipeline "Commands").
type ServingNodeOptions =
    {
        Host: string
        Port: int
        Id: uint64
        Name: string
        /// Distinguishes restarts of one id; random by default.
        Incarnation: uint64
        /// A model folder or alias (ServingPaths.modelDir).
        Model: string
        Backend: string
        /// The split; None for the midpoint.
        Boundaries: int list option
        /// The layer ranges this node hosts (StageLayout.parseLayers); "all" by default.
        Layers: string
        Threads: int
        /// Positions per chain; None for min(1024, max_position_embeddings).
        MaxContext: int option
        MaxChains: int
        FastRebuild: bool
        CostMs: int
        /// The process gives up after this long without a Shutdown.
        TimeoutMs: int
        /// A run folder to join (RunLog contract), as RunNode.
        RunDir: string option
        /// The node id in the run log; Name by default (a restarted process passes <name>-r<attempt>).
        RunNode: string option
        /// The run-log node id this process restarts (JoinOptions.RestartOf).
        RestartOf: string option
    }

/// A pipeline node: C1's Node actor over a QwenStageExecutor, behind TCP (StagePipeline, "Commands").
[<RequireQualifiedAccess>]
module ServingNode =
    /// The value options of `lrm node`.
    let values =
        [
            "connect"
            "id"
            "name"
            "incarnation"
            "model"
            "backend"
            "split"
            "layers"
            "threads"
            "max-context"
            "chains"
            "cost-ms"
            "timeout-ms"
            "run-dir"
            "run-node"
            "restart-of"
        ]

    /// The switches of `lrm node`.
    let switches = [ "fast-rebuild" ]

    let private randomU64 () =
        BitConverter.ToUInt64(Security.Cryptography.RandomNumberGenerator.GetBytes 8, 0)

    /// Parses `--connect host:port --id N --name nN [--incarnation X] [--model tiny] [--backend cpu-f32] [--split B]
    /// [--layers A-B[,C-D]|all] [--threads T] [--max-context M] [--chains C] [--fast-rebuild] [--cost-ms C]
    /// [--timeout-ms T] [--run-dir DIR [--run-node ID] [--restart-of ID]]`.
    let parse (args: string list) : Result<ServingNodeOptions, string> =
        result {
            let! a = CliArgs.parse values switches args

            let! host, port =
                match a.Get "connect" with
                | Some c -> CliArgs.hostPort c
                | None -> Error "--connect host:port is required"

            let! id =
                match a.Get "id" |> Option.bind Invariant.tryParseU64 with
                | Some id when id >= 1UL -> Ok id
                | _ -> Error "--id N (1 or more) is required"

            let! name =
                match a.Get "name" with
                | Some n when SafeName.isValid n && n <> "coord" -> Ok n
                | Some n -> Error(sprintf "--name '%s' is not a safe name other than coord" n)
                | None -> Error "--name is required"

            let! incarnation =
                match a.Get "incarnation" with
                | None -> Ok(randomU64 ())
                | Some _ -> CliArgs.u64 a "incarnation" 0UL

            let! boundaries =
                match a.Get "split" with
                | None -> Ok None
                | Some t -> StageLayout.parseBoundaries t |> Result.map Some

            let! threads = CliArgs.int a "threads" (min 8 Environment.ProcessorCount)
            let! chains = CliArgs.int a "chains" 16
            let! cost = CliArgs.int a "cost-ms" 0
            let! timeout = CliArgs.int a "timeout-ms" 120_000

            let! maxContext =
                match a.Get "max-context" with
                | None -> Ok None
                | Some _ -> CliArgs.int a "max-context" 0 |> Result.map Some

            let! runNode =
                match a.Get "run-node" with
                | Some n when not (SafeName.isValid n) -> Error(sprintf "--run-node '%s' is not a safe name" n)
                | n -> Ok n

            return
                {
                    Host = host
                    Port = port
                    Id = id
                    Name = name
                    Incarnation = incarnation
                    Model = defaultArg (a.Get "model") "tiny"
                    Backend = defaultArg (a.Get "backend") "cpu-f32"
                    Boundaries = boundaries
                    Layers = defaultArg (a.Get "layers") "all"
                    Threads = threads
                    MaxContext = maxContext
                    MaxChains = chains
                    FastRebuild = a.Has "fast-rebuild"
                    CostMs = cost
                    TimeoutMs = timeout
                    RunDir = a.Get "run-dir"
                    RunNode = runNode
                    RestartOf = a.Get "restart-of"
                }
        }

    /// The executor options of a node over opened weights: the stages --layers covers in the split.
    let executorOptions (c: Qwen3Config) (o: ServingNodeOptions) : Result<StageExecutorOptions, string> =
        result {
            let boundaries = defaultArg o.Boundaries (StageLayout.midpoint c.Layers)
            let! _ = StageLayout.validate c.Layers boundaries
            let! ranges = StageLayout.parseLayers c.Layers o.Layers
            let! stages = StageLayout.hosted (StageLayout.stages c.Layers boundaries) ranges

            let! hosted =
                if stages.IsEmpty then
                    Error(sprintf "--layers %s covers no stage" o.Layers)
                else
                    Ok(stages |> List.map _.Index)

            let defaults = StageExecutorOptions.defaults c

            let options =
                { defaults with
                    Boundaries = boundaries
                    Hosted = hosted
                    MaxContext = defaultArg o.MaxContext defaults.MaxContext
                    MaxChains = o.MaxChains
                    Threads = o.Threads
                    FastRebuild = o.FastRebuild
                    CostMs = o.CostMs
                }

            return! StageExecutorOptions.validate c options
        }

    /// An executor for the node on opened weights.
    let executor (weights: Qwen3Weights) (o: ServingNodeOptions) : Result<IQwenStageExecutor, string> =
        result {
            let! options = executorOptions weights.Config o

            let! backend =
                match QwenStageExecutor.tryBackend o.Backend with
                | Some b -> Ok b
                | None ->
                    Error(
                        sprintf
                            "no serving backend '%s' (registered: %s)"
                            o.Backend
                            (QwenStageExecutor.backends () |> List.map _.Name |> String.concat ", ")
                    )

            return backend.Create weights options
        }

    /// Runs a node until the coordinator shuts the run down and prints C1's report line (Harness.ReportPrefix).
    /// Exit codes: 0 done, 1 the node failed, 2 bad options, 3 timeout.
    let run (o: ServingNodeOptions) : int =
        let opened =
            result {
                let! dir = ServingPaths.modelDir o.Model

                let! weights =
                    try
                        Ok(Qwen3Weights.Open dir)
                    with e ->
                        Error(sprintf "cannot open the model %s: %s" dir e.Message)

                match executor weights o with
                | Ok e -> return weights, e
                | Error e ->
                    weights.Dispose()
                    return! Error e
            }

        match opened with
        | Error e ->
            eprintfn "lrm node %s: %s" o.Name e
            2
        | Ok(weights, executor) ->
            use weights = weights
            use executor = executor

            let runLog =
                o.RunDir
                |> Option.map (fun dir ->
                    let node = NodeId.parse (defaultArg o.RunNode o.Name)

                    RunContext.Join(
                        dir,
                        { JoinOptions.create node with
                            ExecutionClass = Some executor.Class
                            RestartOf = o.RestartOf |> Option.map NodeId.parse
                        }
                    ))

            // The run's facade is ambient on the actor thread and on the executor's workers (both capture it at Start).
            let installed = runLog |> Option.map _.Install()

            let code =
                try
                    let node = Node(NodeConfig.create o.Id o.Name o.Incarnation, executor)
                    use transport = new TcpClientTransport(o.Host, o.Port)
                    use host = new ActorHost(node, transport, SystemClock.Instance)
                    host.Start()

                    eprintfn
                        "lrm node %s: stages %A (layers %s) on %s, class %s"
                        o.Name
                        executor.CanHost
                        o.Layers
                        o.Backend
                        (ExecutionClass.id executor.Class).Value.Short

                    let first =
                        Task
                            .WhenAny(node.Stopped :> Task, host.Completion, Task.Delay o.TimeoutMs)
                            .GetAwaiter()
                            .GetResult()

                    if obj.ReferenceEquals(first, node.Stopped) then
                        Console.Out.WriteLine(Harness.ReportPrefix + Harness.reportJson node.Stopped.Result)
                        Console.Out.Flush()
                        0
                    elif host.Completion.IsFaulted then
                        eprintfn "lrm node %s failed: %O" o.Name host.Completion.Exception
                        1
                    else
                        eprintfn "lrm node %s: no shutdown within %d ms" o.Name o.TimeoutMs
                        3
                finally
                    installed |> Option.iter _.Dispose()

            match runLog with
            | Some r ->
                r.Finish(
                    if code = 0 then
                        Lrm.Telemetry.RunStatus.Ok
                    else
                        Lrm.Telemetry.RunStatus.Failed
                )
            | None -> ()

            code

    /// `lrm node <options>`: parse, then run.
    let main (args: string list) : int =
        match parse args with
        | Ok o -> run o
        | Error e ->
            eprintfn "lrm node: %s" e
            2
