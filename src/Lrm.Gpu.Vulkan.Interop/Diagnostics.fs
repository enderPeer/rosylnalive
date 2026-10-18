namespace Lrm.Gpu.Vulkan.Interop

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open Lrm.Core

/// One message from VK_EXT_debug_utils.
type DebugMessage =
    {
        Severity: VkDebugUtilsMessageSeverityFlagBitsEXT
        Types: VkDebugUtilsMessageTypeFlagsEXT
        IdName: string
        IdNumber: int
        Message: string
    }

/// The Vulkan side of a DebugMessenger: its handle and the GC handle that roots its callback delegate. The instance
/// adopts it, so a messenger whose owner was dropped without Dispose is destroyed before vkDestroyInstance; until
/// then the delegate stays rooted whether or not anything references the DebugMessenger.
[<Sealed>]
type internal MessengerLifetime(instance: Instance, handle: VkDebugUtilsMessengerEXT, root: GCHandle) =
    let mutable disposed = 0

    member _.Handle = handle
    member _.IsDisposed = disposed <> 0

    interface IDisposable with
        member this.Dispose() =
            if Threading.Interlocked.Exchange(&disposed, 1) = 0 then
                instance.Release this
                instance.Commands.DestroyDebugUtilsMessengerEXT(instance.Handle, handle)
                // Only now can the driver no longer call the function pointer.
                root.Free()

/// A VkDebugUtilsMessengerEXT whose callback forwards each message to `sink` and to telemetry (a gpu.debug.message
/// event). The callback delegate is rooted by a GCHandle from before vkCreateDebugUtilsMessengerEXT until after
/// vkDestroyDebugUtilsMessengerEXT, so the driver never calls a collected delegate, even when the owner drops this
/// object without disposing it (the VulkanInteropApi contract: callbacks stay rooted). Disposing it, or else disposing
/// its instance, destroys the messenger and frees the root. The callback never throws into the driver: a sink
/// exception is dropped and counted.
[<Sealed>]
type DebugMessenger(instance: Instance, severities: VkDebugUtilsMessageSeverityFlagsEXT, sink: DebugMessage -> unit) =
    let dropped = ref 0

    let callback =
        PFN_vkDebugUtilsMessengerCallbackEXT(fun severity types pData _ ->
            try
                let d =
                    NativePtr.read (NativePtr.ofNativeInt<VkDebugUtilsMessengerCallbackDataEXT> pData)

                let m =
                    {
                        Severity = severity
                        Types = types
                        IdName = VkText.ofPointer d.pMessageIdName
                        IdNumber = d.messageIdNumber
                        Message = VkText.ofPointer d.pMessage
                    }

                Telemetry.event
                    "gpu.debug.message"
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.str "severity" (string severity)
                        Attr.int "id" m.IdNumber
                        Attr.str "id_name" m.IdName
                        Attr.str "message" m.Message
                    ]

                sink m
            with _ ->
                Threading.Interlocked.Increment(&dropped.contents) |> ignore

            VkConst.VK_FALSE)

    let lifetime =
        if not (instance.HasExtension VkConst.VK_EXT_DEBUG_UTILS_EXTENSION_NAME) then
            invalidOp "the instance was created without VK_EXT_debug_utils"

        let root = GCHandle.Alloc callback
        let mutable ci = VkDebugUtilsMessengerCreateInfoEXT.Create()
        ci.messageSeverity <- severities

        ci.messageType <-
            VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            ||| VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            ||| VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT

        ci.pfnUserCallback <- Marshal.GetFunctionPointerForDelegate callback
        let mutable m = VkDebugUtilsMessengerEXT.Null

        try
            instance.Commands.CreateDebugUtilsMessengerEXT(instance.Handle, &ci, &m)
        with _ ->
            root.Free()
            reraise ()

        let l = new MessengerLifetime(instance, m, root)
        instance.Adopt l
        l

    member _.Handle = lifetime.Handle

    /// True once the messenger is destroyed, by Dispose or by disposing its instance.
    member _.IsDisposed = lifetime.IsDisposed

    /// Messages whose sink raised.
    member _.Dropped = dropped.Value

    interface IDisposable with
        member _.Dispose() = (lifetime :> IDisposable).Dispose()

/// VK_EXT_debug_utils object names and labels. Names and labels encode their text per call: for setup and
/// diagnostics, not for every dispatch.
[<RequireQualifiedAccess>]
module DebugUtils =
    /// Names a Vulkan object (vkSetDebugUtilsObjectNameEXT). `handle` is the object's handle as a uint64.
    let setName (device: Device) (objectType: VkObjectType) (handle: uint64) (name: string) =
        let bytes = VkText.utf8z name
        use p = fixed bytes
        let mutable info = VkDebugUtilsObjectNameInfoEXT.Create()
        info.objectType <- objectType
        info.objectHandle <- handle
        info.pObjectName <- NativePtr.toNativeInt p
        device.Commands.SetDebugUtilsObjectNameEXT(device.Handle, &info)

    let private label (name: string) (color: float32[]) (f: VkDebugUtilsLabelEXT -> unit) =
        let bytes = VkText.utf8z name
        use p = fixed bytes
        let mutable l = VkDebugUtilsLabelEXT.Create()
        l.pLabelName <- NativePtr.toNativeInt p
        let c = MemoryMarshal.CreateSpan(&l.color.e0, 4)

        for i in 0 .. min 3 (color.Length - 1) do
            c[i] <- color[i]

        f l

    /// Opens a label region in a command buffer (vkCmdBeginDebugUtilsLabelEXT).
    let beginCommandLabel (device: Device) (cmd: VkCommandBuffer) (name: string) (color: float32[]) =
        label name color (fun l -> device.Commands.CmdBeginDebugUtilsLabelEXT(cmd, &l))

    /// Closes the innermost label region of a command buffer.
    let endCommandLabel (device: Device) (cmd: VkCommandBuffer) =
        device.Commands.CmdEndDebugUtilsLabelEXT(cmd)

    /// Inserts a single label into a command buffer.
    let insertCommandLabel (device: Device) (cmd: VkCommandBuffer) (name: string) (color: float32[]) =
        label name color (fun l -> device.Commands.CmdInsertDebugUtilsLabelEXT(cmd, &l))

    /// Opens a label region on a queue.
    let beginQueueLabel (device: Device) (queue: VkQueue) (name: string) (color: float32[]) =
        label name color (fun l -> device.Commands.QueueBeginDebugUtilsLabelEXT(queue, &l))

    let endQueueLabel (device: Device) (queue: VkQueue) =
        device.Commands.QueueEndDebugUtilsLabelEXT(queue)

/// What VK_EXT_device_fault reports.
type DeviceFaultReport =
    {
        Description: string
        /// (type, reported address, precision).
        Addresses: (VkDeviceFaultAddressTypeKHR * uint64 * uint64) list
        /// (description, vendor fault code, vendor fault data).
        Vendor: (string * uint64 * uint64) list
        VendorBinaryBytes: uint64
    }

/// The outcome of a VK_EXT_device_fault query.
[<RequireQualifiedAccess>]
type DeviceFaultQuery =
    /// The device did not enable VK_EXT_device_fault.
    | NotEnabled
    /// The driver has no fault to describe and says so with this result. On a device that is not lost, NVIDIA 616.92
    /// returns VK_ERROR_UNKNOWN from the count query; the spec makes the query meaningful after VK_ERROR_DEVICE_LOST.
    | NoFault of result: VkResult
    /// The driver described a fault.
    | Fault of DeviceFaultReport

[<RequireQualifiedAccess>]
module DeviceFault =
    /// vkGetDeviceFaultInfoEXT with the two-call idiom: the counts, then the infos. Uses the raw delegate, because a
    /// driver may answer a device that is not lost with an error code, which is an outcome here and not a failure.
    /// A described fault is emitted as a gpu.device.fault event.
    let query (device: Device) : DeviceFaultQuery =
        let f = device.Commands.vkGetDeviceFaultInfoEXT

        if
            isNull f
            || not (List.contains VkConst.VK_EXT_DEVICE_FAULT_EXTENSION_NAME device.EnabledExtensions)
        then
            DeviceFaultQuery.NotEnabled
        else
            let mutable counts = VkDeviceFaultCountsEXT.Create()

            match f.Invoke(device.Handle, NativePtr.toNativeInt &&counts, 0n) with
            | VkResult.VK_SUCCESS
            | VkResult.VK_INCOMPLETE ->
                let addresses =
                    Array.zeroCreate<VkDeviceFaultAddressInfoEXT>(int counts.addressInfoCount)

                let vendor =
                    Array.zeroCreate<VkDeviceFaultVendorInfoEXT>(int counts.vendorInfoCount)

                let vendorBinaryBytes = counts.vendorBinarySize
                // The vendor binary is not read: its size is reported.
                counts.vendorBinarySize <- 0UL
                use pa = fixed addresses
                use pv = fixed vendor
                let mutable info = VkDeviceFaultInfoEXT.Create()
                info.pAddressInfos <- NativePtr.toNativeInt pa
                info.pVendorInfos <- NativePtr.toNativeInt pv

                match f.Invoke(device.Handle, NativePtr.toNativeInt &&counts, NativePtr.toNativeInt &&info) with
                | VkResult.VK_SUCCESS
                | VkResult.VK_INCOMPLETE ->
                    let report =
                        {
                            Description = info.description.AsString
                            Addresses =
                                addresses
                                |> Array.truncate (int counts.addressInfoCount)
                                |> Array.map (fun a -> a.addressType, a.reportedAddress, a.addressPrecision)
                                |> List.ofArray
                            Vendor =
                                vendor
                                |> Array.truncate (int counts.vendorInfoCount)
                                |> Array.map (fun v -> v.description.AsString, v.vendorFaultCode, v.vendorFaultData)
                                |> List.ofArray
                            VendorBinaryBytes = vendorBinaryBytes
                        }

                    Telemetry.event
                        "gpu.device.fault"
                        [
                            Attr.str TelemetryKeys.Backend "vulkan"
                            Attr.str "description" report.Description
                            Attr.int "addresses" report.Addresses.Length
                            Attr.int "vendor_infos" report.Vendor.Length
                            Attr.int64 "vendor_binary_bytes" (int64 report.VendorBinaryBytes)
                        ]

                    DeviceFaultQuery.Fault report
                | r -> DeviceFaultQuery.NoFault r
            | r -> DeviceFaultQuery.NoFault r
