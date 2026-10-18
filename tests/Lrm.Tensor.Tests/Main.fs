module Lrm.Tensor.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                TensorTests.tests
                SafetensorsTests.tests
                CompareTests.tests
                GoldensTests.tests
                ToleranceTests.tests
                RopeTests.tests
                CpuRefTests.tests
                TinyModelTests.tests
            ]
        Model = [ None, [ ModelTests.tests ] ]
        Bench = [ Bench.tests ]
    }

[<EntryPoint>]
let main argv =
    let code = Runner.main argv tiers
    // The parity run log, if a tiny-model acceptance test created one (ParityRun).
    ParityRun.finish code
    code
