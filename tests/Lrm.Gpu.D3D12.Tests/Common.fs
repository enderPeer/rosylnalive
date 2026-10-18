/// Shared helpers of the D3D12 backend tests: the backend on the first accepted adapter, the run log of the gpu tier
/// (kind kernels), and byte helpers.
module Lrm.Gpu.D3D12.Tests.Common

open System
open System.Runtime.InteropServices
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12
open Lrm.Telemetry
open Lrm.Testing

/// The embedded shader set (Lrm.Shaders), read once.
let shaders () = Lrm.Shaders.Shaders.embedded ()

/// u32 words as little-endian bytes.
let bytesOf (ws: uint32[]) =
    MemoryMarshal.AsBytes(ReadOnlySpan ws).ToArray()

/// Little-endian bytes as u32 words.
let wordsOf (bytes: byte[]) =
    MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()

/// Every adapter, discovered once per process for the helpers (the device-check test discovers again inside the run
/// log, so its gpu.device.discover span and gpu.device.check and gpu.device.memory records land there).
let private candidates = lazy (Adapters.enumerate ())

/// The first accepted adapter.
let accepted () =
    match candidates.Value |> List.tryFind _.Accepted with
    | Some c -> c.Info
    | None ->
        failtestf "no D3D12 adapter is accepted: %A" (candidates.Value |> List.map (fun c -> c.Info.Name, c.Rejections))

let allCandidates () = candidates.Value

/// The gpu tier's run log (RunLog contract, kind kernels), created on first use; Main finishes it.
let private run =
    lazy
        (let info = accepted ()
         let set = shaders ()

         let r =
             RunContext.Create(
                 { RunOptions.create "kernels" "D3D12 backend: selftests, probe, soak and transfers (D1)" with
                     ExecutionClass = Some(GpuClass.fill info set)
                     GitSha = Env.commit ()
                     Params =
                         [
                             "device", info.Name
                             "driver", info.Driver
                             "dxil_flags", ShaderSet.flagsFor ShaderTarget.Dxil set |> String.concat " "
                             "source", "tests/Lrm.Gpu.D3D12.Tests (gpu tier)"
                         ]
                 }
             )

         printfn "d3d12 backend run log: %s" r.Directory
         r)

/// The run log, if a gpu test created it.
let runIfCreated () =
    if run.IsValueCreated then Some run.Value else None

/// Finishes the run log if a gpu test created it: ok when the exe passed, failed otherwise.
let finishRun (exitCode: int) =
    match runIfCreated () with
    | Some r -> r.Finish(if exitCode = 0 then RunStatus.Ok else RunStatus.Failed)
    | None -> ()

/// A gpu test whose telemetry goes to the run log, under `test` = its name.
let gpuTest (name: string) (body: unit -> unit) =
    testCase name (fun () ->
        use _ = Telemetry.install (run.Value.Telemetry.Child [ Attr.str "test" name ])
        body ())

/// A gpu test that runs `f` twice and requires identical bytes (TestTiers: GPU work runs twice), with telemetry in
/// the run log.
let gpuTestTwice (name: string) (f: unit -> byte[]) =
    gpuTest name (fun () -> Determinism.runTwice name f |> ignore)

/// Options for a backend: dispatches timed, debug checks on (the encoder's state rules are verified per submission).
let options () =
    { DeviceOptions.For ExecutionBackend.D3D12 with
        TimeDispatches = true
        Debug = true
    }

/// A backend on the first accepted adapter.
let create () : D3D12Backend =
    D3D12Backend.Create(accepted (), options ())

/// A backend on the first accepted adapter, with dispatches timed or not (debug checks on).
let createWith (timeDispatches: bool) : D3D12Backend =
    D3D12Backend.Create(
        accepted (),
        { options () with
            TimeDispatches = timeDispatches
        }
    )

/// Records the device for the test summary (device, driver, ShaderSetHash).
let recordDevice () =
    let info = accepted ()

    let driver =
        info.Details
        |> List.tryFind (fun (k, _) -> k = "driver_version")
        |> Option.map (fun (_, v) -> sprintf "%s (UMD %s)" v info.Driver)
        |> Option.defaultValue info.Driver

    Devices.record Backend.D3D12 info.Name driver (Some (ShaderSet.setHash ShaderTarget.Dxil (shaders ())).Hex)
