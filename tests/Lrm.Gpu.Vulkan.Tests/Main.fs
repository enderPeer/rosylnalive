module Lrm.Gpu.Vulkan.Tests.Main

open Lrm.Testing

/// Root test lists per tier; tools/test.ps1 picks one via LRM_TEST_TIER. The gpu tier's d3d12 group is the reference
/// run for the cross-backend comparison; it runs before the vulkan group (mutex order).
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = [ CpuTests.tests ]
        Gpu = [ Backend.D3D12, [ GpuTests.d3d12Tests ]; Backend.Vulkan, [ GpuTests.tests ] ]
    }

/// `hello-vulkan [--index N]`: the hello outside the test runner, holding the vulkan GPU mutex (TestTiers), so it
/// queues behind other agents' GPU tests.
let private hello (args: string[]) =
    Lrm.Core.Process.init ()

    use _ =
        Locks.acquire (Locks.defaultOptions "Lrm.Gpu.Vulkan.Tests hello-vulkan") [ Locks.GpuVulkan ]

    Lrm.Gpu.Vulkan.Hello.run (Common.shaders ()) args

[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "hello-vulkan" :: rest -> hello (Array.ofList rest)
    | _ ->
        let code = Runner.main argv tiers
        // The gpu tier's run log, if it ran.
        Common.finishRun code
        code
