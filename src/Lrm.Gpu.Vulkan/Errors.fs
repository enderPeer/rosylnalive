namespace Lrm.Gpu.Vulkan

open System
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop

/// What VK_EXT_device_fault said about a lost device, as text for an exception and a log.
[<RequireQualifiedAccess>]
module FaultText =
    /// One line per fact: the description, each address record and each vendor record, or why there is no report.
    let format (q: DeviceFaultQuery) : string =
        match q with
        | DeviceFaultQuery.NotEnabled -> "device fault: VK_EXT_device_fault is not enabled on this device"
        | DeviceFaultQuery.NoFault r -> sprintf "device fault: the driver described no fault (%A, %d)" r (int r)
        | DeviceFaultQuery.Fault f ->
            [
                yield sprintf "device fault: %s" f.Description

                for t, address, precision in f.Addresses do
                    yield sprintf "  address %A 0x%016x (precision 0x%x)" t address precision

                for d, code, data in f.Vendor do
                    yield sprintf "  vendor %s: code 0x%016x, data 0x%016x" d code data

                if f.VendorBinaryBytes > 0UL then
                    yield sprintf "  vendor binary: %d bytes (not read)" f.VendorBinaryBytes
            ]
            |> String.concat "\n"

/// The device stopped working: vkQueueSubmit2 or the timeline wait returned VK_ERROR_DEVICE_LOST, or a submission did
/// not finish in time. Device loss is not a contract violation (GpuBackendInterface, "Errors"), so it is its own
/// exception. It carries the call that saw it, the VkResult (VK_TIMEOUT for a wait that timed out), and what
/// VK_EXT_device_fault reported right after (vkGetDeviceFaultInfoEXT).
type VulkanDeviceLostException(call: string, result: VkResult, fault: DeviceFaultQuery) =
    inherit
        Exception(
            sprintf "vulkan: the device was lost at %s (%A, %d)\n%s" call result (int result) (FaultText.format fault)
        )

    /// The call after which the loss was seen, for example "vkQueueSubmit2" or "timeline wait for 7".
    member _.Call = call
    /// VK_ERROR_DEVICE_LOST, or VK_TIMEOUT when the device still answers but work did not finish in time.
    member _.Result = result
    /// VK_EXT_device_fault's answer, queried once when the loss was seen.
    member _.Fault = fault

/// Raising GpuContractException with the texts the other backends use (GpuBackendInterface, "Errors").
module internal Contract =
    let fail fmt =
        Printf.kprintf (fun s -> raise (GpuContractException s)) fmt
