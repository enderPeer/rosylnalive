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
                TelemetryTests.tests
            ]
        Model = [ None, [ ModelTests.tests ] ]
    }

[<EntryPoint>]
let main argv =
    let code = Runner.main argv tiers
    // The parity run log, if a tiny-model acceptance test created one (ParityRun).
    ParityRun.finish code
    code
