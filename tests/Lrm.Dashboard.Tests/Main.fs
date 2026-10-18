module Lrm.Dashboard.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                MarkdownTests.tests
                CatalogTests.tests
                DashboardTests.tests
                WatchTests.tests
            ]
    }

[<EntryPoint>]
let main argv = Runner.main argv tiers
