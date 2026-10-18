module Lrm.Gpu.D3D12.Tests.Main

open Lrm.Testing

/// Root test lists per tier; tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = [ CpuTests.tests ]
        Gpu = [ Backend.D3D12, [ GpuTests.tests ] ]
    }

/// `hello-d3d12 [--index N]`: the hello outside the test runner, holding the d3d12 GPU mutex (TestTiers), so it queues
/// behind other agents' GPU tests. The CLI's `lrm hello-d3d12` calls the same Hello.run.
let private hello (args: string[]) =
    Lrm.Core.Process.init ()

    use _ =
        Locks.acquire (Locks.defaultOptions "Lrm.Gpu.D3D12.Tests hello-d3d12") [ Locks.GpuD3D12 ]

    Lrm.Gpu.D3D12.Hello.run (Common.shaders ()) args

[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "hello-d3d12" :: rest -> hello (Array.ofList rest)
    | _ ->
        let code = Runner.main argv tiers
        // The gpu tier's run log, if it ran.
        Common.finishRun code
        code
