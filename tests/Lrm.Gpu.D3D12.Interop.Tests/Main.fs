module Lrm.Gpu.D3D12.Interop.Tests.Main

open Lrm.Testing

/// Root test lists per tier. The owning workstream (R3) fills them; tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers = TestTiers.empty

[<EntryPoint>]
let main argv = Runner.main argv tiers
