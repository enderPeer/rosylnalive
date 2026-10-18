/// Shared helpers of the Vulkan backend tests: the backend on the first accepted device, the run log of the gpu tier
/// (kind kernels), and byte helpers.
module Lrm.Gpu.Vulkan.Tests.Common

open System
open System.Runtime.InteropServices
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan
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

/// Every physical device, discovered once per process for the helpers (the device-check test discovers again inside
/// the run log, so its gpu.device.discover span and gpu.device.check and gpu.device.memory records land there).
let private candidates = lazy (PhysicalDevices.enumerate ())

/// The first accepted device.
let accepted () =
    match candidates.Value |> List.tryFind _.Accepted with
    | Some c -> c.Info
    | None ->
        failtestf "no Vulkan device is accepted: %A" (candidates.Value |> List.map (fun c -> c.Info.Name, c.Rejections))

let allCandidates () = candidates.Value

/// A detail of the accepted device (DeviceInfo.Details).
let detail (key: string) =
    (accepted ()).Details
    |> List.tryFind (fun (k, _) -> k = key)
    |> Option.map snd
    |> Option.defaultValue "?"

/// Options for a backend: dispatches timed, debug checks on (the encoder's rules are verified per submission; the
/// validation layer is used when it is installed).
let options () =
    { DeviceOptions.For ExecutionBackend.Vulkan with
        TimeDispatches = true
        Debug = true
    }

/// A backend on the first accepted device.
let create () : VulkanBackend =
    VulkanBackend.Create(accepted (), options ())

/// A backend on the first accepted device, with dispatches timed or not (debug checks on).
let createWith (timeDispatches: bool) : VulkanBackend =
    VulkanBackend.Create(
        accepted (),
        { options () with
            TimeDispatches = timeDispatches
        }
    )

/// A backend whose resource heap is in `placement`.
let createPlaced (placement: HeapPlacement) : VulkanBackend =
    VulkanBackend.Create(accepted (), options (), placement)

/// What a probe backend measures once per process for the run manifest: the fp32 denormal mode and the memory type of
/// each kind of buffer (VulkanHeapRecipe: the selected indices are recorded in the run manifest).
let private measured =
    lazy
        (use b = create ()
         let ib = b :> IGpuBackend
         let layout = HeapLayout.ofPages [ ArenaKind.Scratch, Heap.PageGranularity ]
         let heap = ib.CreateHeap layout
         heap.WriteTable((TensorTable layout).Seal())
         heap.Download(1, 0L, Span(Array.zeroCreate<byte> 64))
         ib.LoadKernels(shaders ()) |> ignore

         let mode =
             b.Fp32Denormals
             |> Option.map Fp32Denormals.name
             |> Option.defaultValue "not measured"

         let types =
             b.MemoryTypeChoices
             |> List.map (fun (k, i, f) -> sprintf "%s=%d(%s)" (BufferKind.name k) i f)
             |> String.concat " "

         mode, types, b.QueueFamily)

/// The gpu tier's run log (RunLog contract, kind kernels), created on first use; Main finishes it.
let private run =
    lazy
        (let info = accepted ()
         let set = shaders ()
         let mode, types, family = measured.Value

         let r =
             RunContext.Create(
                 { RunOptions.create "kernels" "Vulkan backend: selftests, probe, soak and transfers (V1)" with
                     ExecutionClass = Some(GpuClass.fill info set)
                     GitSha = Env.commit ()
                     Params =
                         [
                             "device", info.Name
                             "driver", info.Driver
                             "api", detail "api"
                             "queue_family", Invariant.u32 family
                             "fp32_denormals", mode
                             "memory_types", types
                             "spirv_flags", ShaderSet.flagsFor ShaderTarget.Spirv set |> String.concat " "
                             "source", "tests/Lrm.Gpu.Vulkan.Tests (gpu tier)"
                         ]
                 }
             )

         printfn "vulkan backend run log: %s" r.Directory
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

/// Records the device for the test summary (device, driver, ShaderSetHash).
let recordDevice () =
    let info = accepted ()

    Devices.record
        Backend.Vulkan
        info.Name
        (sprintf "%s (%s, Vulkan %s)" info.Driver (detail "driver_name") (detail "api"))
        (Some (ShaderSet.setHash ShaderTarget.Spirv (shaders ())).Hex)
