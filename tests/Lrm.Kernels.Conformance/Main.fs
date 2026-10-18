module Lrm.Kernels.Conformance.Main

open Lrm.Testing

/// Root test lists per tier (TestTiers contract). cpu: the harness's own tests and every case on CpuDevice (CI). gpu:
/// every case on each GPU backend whose provider exists, under that backend's mutex (local).
let tiers () : TestTiers =
    { TestTiers.empty with
        Cpu = [ HarnessTests.tests; Suites.cpuDevice () ]
        Gpu = Suites.gpu ()
    }

[<EntryPoint>]
let main argv =
    let code = Runner.main argv (tiers ())

    for run in Suites.runs do
        run.Finish code

    code
