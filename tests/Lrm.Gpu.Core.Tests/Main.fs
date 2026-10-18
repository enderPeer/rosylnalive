module Lrm.Gpu.Core.Tests.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract); tools/test.ps1 picks one via LRM_TEST_TIER. Everything here is cpu
/// tier: the contracts, the catalog, the embedded shader set and RecordingBackend need no GPU. The GPU suites of the
/// selftest kernels belong to the backends (D1, V1) and the conformance harness (E1).
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                HeapLayoutTests.tests
                TensorTableTests.tests
                ArgBlockTests.tests
                CatalogTests.tests
                ShaderTests.tests
                SpirvTests.tests
                BackendTests.tests
            ]
    }

[<EntryPoint>]
let main argv = Runner.main argv tiers
