/// The pipeline coordinator (StagePipeline contract, docs/contracts/stage-pipeline.md). Owner: C2a.
///
///     lrm coord run --listen PORT [job options] [--replicas 1] [--nodes 2] [--drain-ms 10000] [--timeout-ms T]
///                   [--verify] [--run-log] [--title TEXT] [--progress]
///         the coordinator of one Qwen pipeline job, waiting for `lrm node` processes (or a chaos proxy) on PORT
///     lrm coord local [job options] [--topology split2|replicated3|bothlost5] [--seed S | --seeds 1-50]
///                     [--kill N:AFTER[:RESTART_MS]] [--partition N:FROM_MS:UNTIL_MS] [--sever N:MS] [--no-faults]
///                     [--chaos none|lan|proxy|standard] [--drop P] [--duplicate P] [--reorder P] ...
///                     [--nodes process|here] [--proxy process|here] [--coordinator here|process]
///                     [--cost-ms 2] [--threads 1] [--timeout-ms 60000] [--fast-rebuild] [--run-log] [--title TEXT]
///                     [--parallel N]
///         the harness on localhost: the coordinator, `lrm chaos-proxy` and `lrm node` processes, with kills,
///         restarts and partitions; compares every run with the single-process run
///
/// Job options: [--model tiny|qwen3-0.6b|DIR] [--backend cpu-f32|cpu-f64] [--split 1] [--steps 16] [--stop-at-eos]
/// and the lanes' prompts in order: --ids 1,2,3 | --ids-from FILE[:TENSOR] | --prompt TEXT | --prompt-set tiny
/// (each repeatable; the tiny set p0, p1, p2 by default).
module Lrm.Cli.Commands.Coord

open Lrm.Cluster
open Lrm.Serving

let private usage () =
    eprintfn "usage: lrm coord run --listen PORT [job options] [--replicas R] [--nodes N] [--verify] [--run-log]"
    eprintfn "       lrm coord local [job options] [--topology T] [--seed S|--seeds A-B] [--kill N:AFTER[:MS]] ..."
    eprintfn "       (see the header of apps/Lrm.Cli/Commands/Coord.fs or docs/contracts/stage-pipeline.md)"
    2

let private run (args: string[]) : int =
    match List.ofArray args with
    | "run" :: rest -> ServingCoordinator.main rest
    | "local" :: rest -> ServingLocal.main (Harness.selfStartInfo [] []) rest
    | _ -> usage ()

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "coord"
            Summary = "Qwen pipeline coordinator: run (one job over TCP), local (the localhost chaos harness)."
            Run = run
        |}
    ]
