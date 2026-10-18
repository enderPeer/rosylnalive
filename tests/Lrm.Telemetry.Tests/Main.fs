module Lrm.Telemetry.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                EventLineTests.tests
                RunLogTests.tests
                MeteringTests.tests
                ValuesTests.tests
                ReportTests.tests
                ChartEdgeTests.tests
            ]
        Bench = [ Bench.tests ]
    }

[<EntryPoint>]
let main argv = Runner.main argv tiers
