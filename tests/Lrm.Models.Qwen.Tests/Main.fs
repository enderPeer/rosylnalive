module Lrm.Models.Qwen.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                ConfigTests.tests
                RopeTests.tests
                WeightsTests.tests
                ForwardTests.tests
                DeterminismTests.tests
                SeamTests.tests
                GoldenCheckTests.tests
                TelemetryTests.tests
            ]
        Model = [ None, [ ModelTests.tests; GateCpuTests.tests ] ]
    }

[<EntryPoint>]
let main argv =
    let code = Runner.main argv tiers
    // The parity run logs the acceptance tests created: the tiny model's (cpu tier), the full model's (model tier).
    ParityRun.finish code
    code
