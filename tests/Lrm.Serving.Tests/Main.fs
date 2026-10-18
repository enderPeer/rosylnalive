module Lrm.Serving.Tests.Main

open Expecto
open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER. The cpu tier holds the
/// whole tiny-model pipeline, CI's acceptance: the vectors, the executor, the rebuild proofs, 50 proxy seeds and the
/// multi-process harness. The real-time lists run one test at a time (the sweep runs its own seeds in parallel), so
/// their timings do not compete. The model tier runs Qwen3-0.6B split in two (local only).
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                JobTests.tests
                ExecutorTests.tests
                RebuildTests.tests
                testSequenced ClusterTests.tests
                testSequenced ProcessTests.tests
                testSequenced RunLogTests.tests
            ]
        Model = [ None, [ ModelTests.tests ] ]
    }

/// `--serving-worker <node|coord run|chaos-proxy> <options>` runs a serving process for the multi-process tests
/// (started with Env.workerEnvironment) instead of the test runner.
[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "--serving-worker" :: rest ->
        Lrm.Core.Process.init ()
        Lrm.Serving.ServingWorker.main rest
    | _ -> Runner.main argv tiers
