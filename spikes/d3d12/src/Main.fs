/// The R5 spike as a test exe (TestTiers): `LRM_TEST_TIER=cpu|gpu dotnet artifacts/bin/WarpSpike/release/WarpSpike.dll`.
/// Before anything else the process is pinned to LRM_WARP_CPUS logical processors (default 4, a windows-latest
/// runner), because WARP runs on the CPU. `warp-info` prints the adapter report and exits (no mutex, no run log).
module Lrm.Spikes.D3D12.Main

open Lrm.Gpu.D3D12.Interop
open Lrm.Testing

let tiers () : TestTiers =
    { TestTiers.empty with
        Cpu = [ CpuTests.tests ]
        Gpu = [ Backend.D3D12, [ GpuTests.tests () ] ]
    }

let private info () =
    let r = WarpAdapter.probe ()
    let a = r.Adapter
    printfn "adapter       %s (vendor %04x, device %04x, LUID %s)" a.Description a.VendorId a.DeviceId (Lrm.Gpu.D3D12.Adapters.luidText a.Luid)
    printfn "umd           %s" a.DriverVersion
    printfn "dxgi index    %d (EnumAdapterByGpuPreference, high performance)" a.Index
    printfn "os            %s (%s)" r.Os r.OsEdition

    for f, v in r.Files do
        printfn "%-13s %s" f v

    match r.Capabilities with
    | Ok c ->
        printfn
            "capabilities  feature level %s, shader model %s, binding tier %d, heap tier %d, root signature 1.%d, waves %d..%d, runtime SDK %s"
            (DeviceCheck.formatFeatureLevel c.MaxFeatureLevel)
            (DeviceCheck.formatShaderModel c.HighestShaderModel)
            (int c.ResourceBindingTier)
            (int c.ResourceHeapTier)
            (int c.HighestRootSignature - 1)
            c.WaveLaneCountMin
            c.WaveLaneCountMax
            (c.RuntimeSdkVersion |> Option.map string |> Option.defaultValue "unknown")
    | Error e -> printfn "capabilities  no device: %s" e

    if r.Accepted then
        printfn "d3d12-warp    ACCEPTED"
        0
    else
        printfn "d3d12-warp    REJECTED: %s" (String.concat "; " r.Rejections)
        3

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()
    let a = Affinity.apply ()
    Common.affinity <- Some a
    printfn "warp spike: %d of %d logical processors (LRM_WARP_CPUS %s, mask %s)" a.Cpus a.MachineCpus a.Requested a.Mask

    match List.ofArray argv with
    | [ "warp-info" ] -> info ()
    | _ ->
        let code = Runner.main argv (tiers ())
        Common.finishRun code

        for run in Lrm.Kernels.Conformance.Suites.runs do
            run.Finish code

        code
