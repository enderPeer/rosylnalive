module Lrm.Gpu.Exec.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER. Everything X1a proves
/// runs on CpuDevice, so the cpu tier carries the acceptance; the model tier adds Qwen3-0.6B. GPU groups are X1b's.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = [ ParityTests.tests; PlanTests.tests; SeamTests.tests; TelemetryTests.tests ]
        Model = [ None, [ ModelTests.tests ] ]
    }

[<EntryPoint>]
let main argv =
    let code = Runner.main argv tiers
    // The acceptance run log, if an acceptance test created one (AcceptanceRun).
    AcceptanceRun.finish code
    code
