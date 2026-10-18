/// Shared state of the spike's tests: the WARP provider, the affinity applied at start, the gpu tier's run log (kind
/// kernels: node n0 with the d3d12-warp class, and node hw with the hardware adapter's d3d12 class for the hardware
/// comparison) and byte helpers.
module Lrm.Spikes.D3D12.Common

open System
open System.Runtime.InteropServices
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12
open Lrm.Telemetry
open Lrm.Testing

/// The embedded shader set (Lrm.Shaders).
let shaders () = Lrm.Shaders.Shaders.embedded ()

let bytesOf (ws: uint32[]) =
    MemoryMarshal.AsBytes(ReadOnlySpan ws).ToArray()

let wordsOf (bytes: byte[]) =
    MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()

/// The processors this process runs on (Affinity.apply runs in Main, before any device).
let mutable affinity: Affinity.Applied option = None

/// The WARP provider, probed once per process.
let provider = lazy (WarpBackend.WarpProvider())

/// The WARP report (EnumWarpAdapter, capabilities, versions, verdict).
let report () = provider.Value.Report

/// Options: dispatches timed, debug checks on (D1 verifies its encoding against the state rules per submission).
let options () =
    { DeviceOptions.For ExecutionBackend.D3D12Warp with
        TimeDispatches = true
        Debug = true
    }

/// A d3d12-warp backend (D1's backend on WARP, wrapped).
let create () : WarpBackend.WarpBackend =
    match WarpBackend.tryCreate provider.Value (options ()) with
    | Ok b -> b :?> WarpBackend.WarpBackend
    | Error e -> failtestf "no d3d12-warp device: %s" e

/// The d3d12-warp DeviceInfo.
let warpInfo () = WarpAdapter.warpInfo (report ())

/// The class of the embedded set on WARP.
let warpClass () = GpuClass.fill (warpInfo ()) (shaders ())

let private affinityParams () =
    match affinity with
    | Some a ->
        [
            "warp_cpus", Invariant.i32 a.Cpus
            "warp_cpus_requested", a.Requested
            "machine_cpus", Invariant.i32 a.MachineCpus
            "affinity_mask", a.Mask
        ]
    | None -> [ "warp_cpus", "unchanged" ]

/// The parameters every run log of the spike records: the WARP build, the OS, the CPU limit.
let runParams () =
    let r = report ()

    [
        "source", "spikes/d3d12 (R5 WARP spike)"
        "device", r.Adapter.Description
        "driver", r.Adapter.DriverVersion
        "os", r.Os
        "os_edition", r.OsEdition
        yield! r.Files |> List.map (fun (f, v) -> f, v)
        "dxil_flags", ShaderSet.flagsFor ShaderTarget.Dxil (shaders ()) |> String.concat " "
        yield! affinityParams ()
    ]

/// The gpu tier's run log (RunLog contract, kind kernels), created on first use; Main finishes it.
let private run =
    lazy
        (let r =
            RunContext.Create(
                { RunOptions.create "kernels" "WARP spike: device, denormal probe, D1's selftests (R5)" with
                    ExecutionClass = Some(warpClass ())
                    GitSha = Env.commit ()
                    Params = runParams ()
                }
            )

         printfn "warp spike run log: %s" r.Directory
         r)

let runIfCreated () =
    if run.IsValueCreated then Some run.Value else None

/// The node id of the hardware comparison's records in the gpu tier's run log.
let HardwareNode = NodeId.parse "hw"

/// The hardware comparison's node: joined to the run log with the hardware adapter's own d3d12 class (GpuClass.fill
/// over its DeviceInfo and the embedded set, as D1's LoadKernels does). A node has one class (RunLog), so the
/// hardware run's records go here, never to n0, whose class is d3d12-warp.
let mutable private hardwareNode: (DeviceInfo * RunContext) option = None

/// The facade of the hardware node, under `test` = `name`; the node joins the run log on first use.
let hardwareTelemetry (info: DeviceInfo) (name: string) : ITelemetry =
    let node =
        match hardwareNode with
        | Some(i, n) when i = info -> n
        | Some(i, _) -> failwithf "the hardware node has the class of %s, not %s" i.Name info.Name
        | None ->
            let n =
                RunContext.Join(
                    run.Value.Directory,
                    { JoinOptions.create HardwareNode with
                        ExecutionClass = Some(GpuClass.fill info (shaders ()))
                    }
                )

            printfn "warp spike: node %s (%s, d3d12) joined the run log" HardwareNode.Value info.Name
            hardwareNode <- Some(info, n)
            n

    node.Telemetry.Child [ Attr.str "test" name ]

let finishRun (exitCode: int) =
    let status = if exitCode = 0 then RunStatus.Ok else RunStatus.Failed

    match hardwareNode with
    | Some(_, n) -> n.Finish status
    | None -> ()

    match runIfCreated () with
    | Some r -> r.Finish status
    | None -> ()

/// A gpu test whose telemetry goes to the run log, under `test` = its name.
let gpuTest (name: string) (body: unit -> unit) =
    testCase name (fun () ->
        use _ = Telemetry.install (run.Value.Telemetry.Child [ Attr.str "test" name ])
        body ())

/// A gpu test that runs `f` twice and requires identical bytes (TestTiers), with telemetry in the run log.
let gpuTestTwice (name: string) (f: unit -> byte[]) =
    gpuTest name (fun () -> Determinism.runTwice name f |> ignore)

/// Records the WARP device for the test summary. Lrm.Testing knows only d3d12 and vulkan, so WARP is listed under
/// d3d12 (whose mutex its tests hold), with "(d3d12-warp)" in the device name.
let recordDevice () =
    let r = report ()

    Devices.record
        Backend.D3D12
        (r.Adapter.Description + " (d3d12-warp)")
        (sprintf "WARP %s" r.Adapter.DriverVersion)
        (Some (ShaderSet.setHash ShaderTarget.Dxil (shaders ())).Hex)

/// D1's d3d12 enumeration, once per process. Its discovery records (gpu.device.check with backend = d3d12: the d3d12
/// class's verdict on each adapter, WARP rejected) are written once, by the test that asks first.
let d3d12Candidates = lazy (Adapters.enumerate ())

/// The first accepted hardware d3d12 adapter, if any (none on a CI runner).
let hardware () =
    d3d12Candidates.Value |> List.tryFind _.Accepted |> Option.map _.Info
