module Lrm.Models.Qwen.Tests.Main

open Lrm.Testing

/// Root test lists per tier. The owning workstream (M2) fills them; tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers = TestTiers.empty

[<EntryPoint>]
let main argv = Runner.main argv tiers
