namespace Lrm.Gpu.Vulkan

open System
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop

/// Device discovery for the vulkan backend: R2's DeviceCheck reports, as the DeviceInfo and DeviceCandidate records of
/// GpuBackendInterface.
[<RequireQualifiedAccess>]
module PhysicalDevices =
    /// The Vulkan loader, opened once per process and kept open (instances come and go; the loader stays).
    let private loaderOnce = lazy (VulkanLoader.TryOpen())

    /// The loader, or None when no Vulkan loader is installed.
    let loader () = loaderOnce.Value

    /// GpuBackendInterface: buffer descriptors of at most 16 bytes (Requirements.VulkanDescriptorSize).
    [<Literal>]
    let MaxBufferDescriptorBytes = 16UL

    /// GpuBackendInterface: 128 bytes of push data (Requirements.ArgumentBytes).
    [<Literal>]
    let ArgumentBytes = 128UL

    let private yesNo (b: bool) = if b then "true" else "false"

    /// The heap limits of Lrm.Gpu.Core (Heap.vulkanGeometry's input) from the queried descriptor-heap properties.
    let heapLimits (h: DeviceCheck.DescriptorHeapLimits) : VulkanHeapLimits =
        {
            BufferDescriptorSize = int64 h.BufferDescriptorSize
            BufferDescriptorAlignment = int64 h.BufferDescriptorAlignment
            ImageDescriptorAlignment = int64 h.ImageDescriptorAlignment
            MinResourceHeapReservedRange = int64 h.MinResourceHeapReservedRange
            MaxResourceHeapSize = int64 h.MaxResourceHeapSize
        }

    /// The Driver field of the GPU fill rule: GpuClass.vulkanDriver over driverInfo and driverVersion.
    let driverOf (c: DeviceCheck.Capabilities) =
        GpuClass.vulkanDriver c.Identity.DriverInfo c.Identity.DriverVersion

    /// The largest DEVICE_LOCAL memory heap, in bytes (DeviceInfo.MemoryBytes).
    let deviceMemory (c: DeviceCheck.Capabilities) =
        c.MemoryHeaps
        |> List.filter _.DeviceLocal
        |> List.map (fun h -> int64 h.Size)
        |> function
            | [] -> 0L
            | xs -> List.max xs

    /// The DeviceInfo of a device report. Details, in this order: api, driver_name, driver_info, driver_version,
    /// device_uuid, driver_uuid, device_type, queue_family, timestamp_valid_bits, timestamp_period_ns, then for a device
    /// with VK_EXT_descriptor_heap max_push_data_size, buffer_descriptor_size, resource_heap_alignment,
    /// max_resource_heap_size, min_resource_heap_reserved_range, then min_storage_buffer_offset_alignment,
    /// max_storage_buffer_range, shader_denorm_preserve_float32, shader_denorm_flush_to_zero_float32, device_fault,
    /// memory_budget and memory_types (index:flags, space-separated).
    let info (index: int) (r: DeviceCheck.DeviceReport) : DeviceInfo =
        let c = r.Capabilities
        let id = c.Identity

        let heap =
            match c.DescriptorHeap with
            | Some h ->
                [
                    "max_push_data_size", Invariant.u64 h.MaxPushDataSize
                    "buffer_descriptor_size", Invariant.u64 h.BufferDescriptorSize
                    "resource_heap_alignment", Invariant.u64 h.ResourceHeapAlignment
                    "max_resource_heap_size", Invariant.u64 h.MaxResourceHeapSize
                    "min_resource_heap_reserved_range", Invariant.u64 h.MinResourceHeapReservedRange
                ]
            | None -> []

        {
            Backend = ExecutionBackend.Vulkan
            Index = index
            Name = id.Name
            VendorId = id.VendorId
            PciDeviceId = id.DeviceId
            Driver = driverOf c
            MemoryBytes = deviceMemory c
            Details =
                [
                    "api", VkVersion.toString id.ApiVersion
                    "driver_name", (if id.DriverName = "" then "unknown" else id.DriverName)
                    "driver_info", (if id.DriverInfo = "" then "unknown" else id.DriverInfo)
                    "driver_version", id.Driver
                    "device_uuid", (if id.DeviceUuid = "" then "unknown" else id.DeviceUuid)
                    "driver_uuid", (if id.DriverUuid = "" then "unknown" else id.DriverUuid)
                    "device_type", string id.Type
                    "queue_family",
                    (match c.QueueFamily with
                     | Some f -> Invariant.u32 f
                     | None -> "none")
                    "timestamp_valid_bits", Invariant.u32 c.TimestampValidBits
                    "timestamp_period_ns", Invariant.f64 (float c.TimestampPeriod)
                ]
                @ heap
                @ [
                    "min_storage_buffer_offset_alignment", Invariant.u64 c.MinStorageBufferOffsetAlignment
                    "max_storage_buffer_range", Invariant.u32 c.MaxStorageBufferRange
                    "shader_denorm_preserve_float32", yesNo c.Optional.ShaderDenormPreserveFloat32
                    "shader_denorm_flush_to_zero_float32", yesNo c.Optional.ShaderDenormFlushToZeroFloat32
                    "device_fault", yesNo c.Optional.DeviceFault
                    "memory_budget", yesNo c.Optional.MemoryBudget
                    "memory_types",
                    c.MemoryTypes
                    |> List.map (fun t -> sprintf "%d:%s" t.Index (MemoryTypes.flagsText t.Flags))
                    |> String.concat " "
                ]
        }

    /// The rejections of a device report. The three requirements GpuBackendInterface names use its Requirements texts:
    /// VK_EXT_descriptor_heap, buffer descriptors of at most 16 bytes, and 128 bytes of push data. Every other reason is
    /// DeviceCheck's, so the device check and the backend reject the same devices.
    let rejections (r: DeviceCheck.DeviceReport) : string list =
        let c = r.Capabilities

        let contract =
            [
                match c.DescriptorHeap with
                | None -> Requirements.VulkanDescriptorHeap
                | Some h ->
                    if h.BufferDescriptorSize > MaxBufferDescriptorBytes then
                        Requirements.VulkanDescriptorSize

                    if h.MaxPushDataSize < ArgumentBytes then
                        Requirements.ArgumentBytes
            ]

        let replaced (reason: string) =
            (c.DescriptorHeap.IsNone && reason = "VK_EXT_descriptor_heap not supported")
            || (match c.DescriptorHeap with
                | Some h when h.MaxPushDataSize < ArgumentBytes ->
                    reason.StartsWith("maxPushDataSize", StringComparison.Ordinal)
                | _ -> false)

        contract @ (r.Reasons |> List.filter (replaced >> not))

    /// The candidate of a device report.
    let candidate (index: int) (r: DeviceCheck.DeviceReport) : DeviceCandidate =
        {
            Info = info index r
            Rejections = rejections r
        }

    /// Every physical device, in the loader's order, as candidates; empty without a loader. DeviceCheck reports each
    /// verdict as a gpu.device.check record and its memory as gpu.device.memory gauges, inside a gpu.device.discover
    /// span.
    let enumerate () : DeviceCandidate list =
        match loader () with
        | None -> []
        | Some l ->
            let _, reports = DeviceCheck.checkAll l
            reports |> List.mapi candidate

    /// The deviceUUID that DeviceInfo.Details names, if any.
    let uuidOf (info: DeviceInfo) =
        info.Details
        |> List.tryFind (fun (k, _) -> k = "device_uuid")
        |> Option.map snd
        |> Option.filter (fun v -> v <> "unknown")
