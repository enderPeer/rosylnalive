/// A Qwen pipeline node (StagePipeline contract, docs/contracts/stage-pipeline.md). Owner: C2a.
///
///     lrm node --connect HOST:PORT --id N --name nN [--incarnation X] [--model tiny|qwen3-0.6b|DIR]
///              [--backend cpu-f32|cpu-f64] [--split 1] [--layers A-B[,C-D]|all] [--threads T] [--max-context M]
///              [--chains C] [--fast-rebuild] [--cost-ms C] [--timeout-ms T]
///              [--run-dir DIR [--run-node ID] [--restart-of ID]]
///
/// C1's node actor over a QwenStageExecutor: it hosts the stages of the split that --layers covers (CanHost lists
/// only those), joins the coordinator (or a chaos proxy) at HOST:PORT, and prints C1's report line when the run ends.
module Lrm.Cli.Commands.Node

open Lrm.Serving

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "node"
            Summary = "Qwen pipeline node: hosts the stages its --layers cover and joins a coordinator."
            Run = fun args -> ServingNode.main (List.ofArray args)
        |}
    ]
