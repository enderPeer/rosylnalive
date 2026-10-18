/// Shared helpers of the Vulkan interop tests: the loader, an instance, the first accepted device, the gpu tier's run
/// log (kind kernels) and byte helpers.
module Lrm.Gpu.Vulkan.Interop.Tests.Common

open System
open System.Runtime.InteropServices
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop
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

let sha256Hex (bytes: byte[]) =
    Convert.ToHexString(Security.Cryptography.SHA256.HashData bytes).ToLowerInvariant()

/// The Vulkan loader, opened once per process.
let private loaderOnce = lazy (VulkanLoader.TryOpen())

let loader () =
    match loaderOnce.Value with
    | Some l -> l
    | None -> failtest "no Vulkan loader (vulkan-1.dll) on this machine"

/// A Vulkan 1.4 instance, with VK_EXT_debug_utils when `debugUtils` is true and the loader offers it.
let createInstance (debugUtils: bool) : Instance =
    let l = loader ()

    let debug =
        debugUtils
        && (l.InstanceExtensions()).ContainsKey VkConst.VK_EXT_DEBUG_UTILS_EXTENSION_NAME

    l.CreateInstance(InstanceOptions.create "lrm-vulkan-interop-tests" debug)

/// The first accepted physical device of `instance`, with its report.
let accepted (instance: Instance) : VkPhysicalDevice * DeviceCheck.DeviceReport =
    let candidates =
        instance.PhysicalDevices()
        |> Array.map (fun d -> d, DeviceCheck.check instance d)

    match candidates |> Array.tryFind (fun (_, r) -> r.Accepted) with
    | Some c -> c
    | None ->
        failtestf
            "no Vulkan device is accepted: %A"
            (candidates |> Array.map (fun (_, r) -> r.Capabilities.Identity.Name, r.Reasons))

/// The ExecutionClass of the accepted device running the embedded SPIR-V set.
let executionClass (r: DeviceCheck.DeviceReport) : ExecutionClass =
    let id = r.Capabilities.Identity

    {
        Backend = ExecutionBackend.Vulkan
        DeviceId = GpuClass.deviceId id.VendorId id.DeviceId
        Driver = GpuClass.vulkanDriver id.DriverInfo id.DriverVersion
        ShaderSetHash = Some(ShaderSet.setHash ShaderTarget.Spirv (shaders ()))
        Arch = GpuClass.arch ExecutionBackend.Vulkan
    }

/// The device, read once for the run log.
let private deviceOnce =
    lazy
        (use instance = createInstance false
         let _, r = accepted instance
         r)

/// The gpu tier's run log (RunLog contract, kind kernels), created on first use; Main finishes it.
let private run =
    lazy
        (let r = deviceOnce.Value
         let id = r.Capabilities.Identity

         let ctx =
             RunContext.Create(
                 { RunOptions.create "kernels" "Vulkan interop: device creation and hello-compute (R2)" with
                     ExecutionClass = Some(executionClass r)
                     GitSha = Env.commit ()
                     Params =
                         [
                             "device", id.Name
                             "driver", id.Driver
                             "api", VkVersion.toString id.ApiVersion
                             "vk_header_version", string VkMeta.HeaderVersion
                             "source", "tests/Lrm.Gpu.Vulkan.Interop.Tests (gpu tier)"
                         ]
                 }
             )

         printfn "vulkan interop run log: %s" ctx.Directory
         ctx)

/// Finishes the run log if a gpu test created it: ok when the exe passed, failed otherwise.
let finishRun (exitCode: int) =
    if run.IsValueCreated then
        run.Value.Finish(if exitCode = 0 then RunStatus.Ok else RunStatus.Failed)

/// A gpu test whose telemetry goes to the run log, under `test` = its name.
let gpuTest (name: string) (body: unit -> unit) =
    testCase name (fun () ->
        use _ = Telemetry.install (run.Value.Telemetry.Child [ Attr.str "test" name ])
        body ())

/// Records the device for the test summary (device, driver, ShaderSetHash).
let recordDevice () =
    let r = deviceOnce.Value
    let id = r.Capabilities.Identity

    Devices.record
        Backend.Vulkan
        id.Name
        (sprintf "%s (%s, Vulkan %s)" id.Driver id.DriverName (VkVersion.toString id.ApiVersion))
        (Some (ShaderSet.setHash ShaderTarget.Spirv (shaders ())).Hex)
