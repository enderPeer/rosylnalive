module Lrm.Tokenizer.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER. The corpus tests need
/// tokenizer.json and are skipped visibly without it; everything else runs in CI.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                PreTokenizerTests.tests
                UnicodeTests.tests
                ToyTests.tests
                CorpusTests.tests
            ]
        Bench = [ Bench.tests ]
    }

[<EntryPoint>]
let main argv =
    let code = Runner.main argv tiers
    // The run log, if a test created one (TokenizerRun).
    TokenizerRun.finish code
    code
