module Lrm.Cluster.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER. Everything is cpu tier:
/// the cluster core needs no GPU, and CI runs the simulation, the seed sweep and the 3-process kill test.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                ProtocolTests.tests
                PartsTests.tests
                CoordinatorTests.tests
                SimulationTests.tests
                TransportTests.tests
                RunLogTests.tests
            ]
    }

/// `--cluster-worker node <options>` runs a node process for the multi-process tests (started with
/// Env.workerEnvironment) instead of the test runner.
[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "--cluster-worker" :: rest ->
        Lrm.Core.Process.init ()
        Lrm.Cluster.Harness.workerMain rest
    | _ -> Runner.main argv tiers
