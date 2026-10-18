/// The model tier (local only, Local\LrmModelCpu): Qwen3-0.6B split at layer 14 (the plan's split) on two node actors
/// behind the chaos proxy, with the stage-1 node killed and restarted, equals the whole model in one process.
module Lrm.Serving.Tests.ModelTests

open Expecto
open Lrm.Core
open Lrm.Serving
open Lrm.Serving.Tests.Common

/// M2a's measured cpu-f32 greedy ids after 785 3974 13876 38835 (ForwardOpsAndQwenModel, "On Qwen3-0.6B").
let private prompt = [| 785; 3974; 13876; 38835 |]

let tests =
    testList
        "model"
        [
            testCase "Qwen3-0.6B, split [14] on two nodes with a kill and a restart, is bit-identical to one process"
            <| fun () ->
                let spec =
                    { JobSpec.tiny with
                        Model = LrmPaths.model "Qwen3-0.6B"
                        Prompts = [ PromptSpec.Ids prompt ]
                        Steps = 3
                    }

                let dir, job = ServingHarness.job spec
                Expect.equal job.Boundaries [ 14 ] "the midpoint of 28 layers"
                let reference = ServingHarness.reference (ServingHarness.weights dir) job 8
                Expect.equal reference.Single.Tokens [ [ 34208; 916; 279 ] ] "M2a's measured ids"
                Expect.equal reference.PipelineResult.LogitsDigest reference.Single.LogitsDigest "two stages = one"

                let s =
                    { ServingHarness.clean spec Topology.Split2 1UL with
                        Name = "qwen3-0.6b-split2"
                        Threads = 8
                        CostMs = 0
                        Kills =
                            [
                                {
                                    Node = 2
                                    AfterUnits = 2
                                    RestartAfterMs = Some 300
                                }
                            ]
                        TimeoutMs = 600_000
                    }

                let r = ServingHarness.run HarnessMode.inProcess s
                expectIdentical reference r
                Expect.isGreaterThanOrEqual r.Outcome.Value.Replays 1 "the restarted node replayed stage 1"
        ]
