module Lrm.Core.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                InvariantTests.tests
                HashTests.tests
                CanonicalTests.tests
                DTypeTests.tests
                NumericsTests.tests
                RngTests.tests
                IdsTests.tests
                TelemetryTests.tests
                ExecutionClassTests.tests
                ProcessTests.tests
            ]
        Bench = [ Bench.tests ]
    }

/// `--probe <name>` runs one child-process probe for ProcessTests instead of the test runner, so the probe sees the
/// process exactly as configured (for example in globalization-invariant mode, where Runner.main would refuse).
[<EntryPoint>]
let main argv =
    match argv with
    | [| "--probe"; name |] -> ProcessTests.runProbe name
    | _ -> Runner.main argv tiers
