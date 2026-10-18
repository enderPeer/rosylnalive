/// Device acceptance for the LRM Vulkan backend (VulkanHeapRecipe section 1): Vulkan 1.4, VK_EXT_descriptor_heap and
/// the recipe's five features, no fallbacks. Shared by `lrm devices-vulkan`, apps/VkDeviceCheck and V1.
module Lrm.Gpu.Vulkan.Interop.DeviceCheck

#nowarn "9"

open System
open System.Text.Json
open System.Text.Json.Nodes
open Microsoft.FSharp.NativeInterop
open Lrm.Core

/// The acceptance floor for maxPushDataSize. VulkanHeapRecipe pushes a 128-byte ArgBlock, so 128 would do today;
/// the floor stays at 256 (ADR R2-0001): it is the D3D12 root-signature limit (64 DWORDs), so both backends accept
/// the same argument budget and an ArgBlock that grows up to that limit never needs devices re-qualified. The RTX
/// 4080 reports exactly 256, so nothing is rejected that the recipe would accept on this machine.
[<Literal>]
let RequiredPushDataSize = 256UL

/// What VulkanHeapRecipe section 1 asserts at device creation: the 128-byte ArgBlock.
[<Literal>]
let RecipePushDataSize = 128UL

/// Heap slots (HeapLayoutAndTensorTable): slot 0 is the tensor table, 1..63 arena pages.
[<Literal>]
let HeapSlots = 64

/// The largest arena page, which a storage-buffer descriptor must span (maxStorageBufferRange).
[<Literal>]
let PageBytes = 1073741824UL

type DeviceIdentity =
    {
        Name: string
        Type: VkPhysicalDeviceType
        ApiVersion: uint32
        DriverVersion: uint32
        /// The driver version as its vendor writes it: 616.92 for NVIDIA.
        Driver: string
        VendorId: uint32
        DeviceId: uint32
        DriverId: VkDriverId
        DriverName: string
        DriverInfo: string
        /// VkPhysicalDeviceVulkan11Properties.deviceUUID, hex.
        DeviceUuid: string
        /// VkPhysicalDeviceVulkan11Properties.driverUUID, hex.
        DriverUuid: string
    }

/// VkPhysicalDeviceDescriptorHeapPropertiesEXT, in managed form.
type DescriptorHeapLimits =
    {
        SamplerHeapAlignment: uint64
        ResourceHeapAlignment: uint64
        MaxSamplerHeapSize: uint64
        MaxResourceHeapSize: uint64
        MinSamplerHeapReservedRange: uint64
        MinSamplerHeapReservedRangeWithEmbedded: uint64
        MinResourceHeapReservedRange: uint64
        SamplerDescriptorSize: uint64
        ImageDescriptorSize: uint64
        BufferDescriptorSize: uint64
        SamplerDescriptorAlignment: uint64
        ImageDescriptorAlignment: uint64
        BufferDescriptorAlignment: uint64
        MaxPushDataSize: uint64
        ImageCaptureReplayOpaqueDataSize: uint64
        MaxDescriptorHeapEmbeddedSamplers: uint32
        SamplerYcbcrConversionCount: uint32
        SparseDescriptorHeaps: bool
        ProtectedDescriptorHeaps: bool
    }

/// The heap geometry of VulkanHeapRecipe section 3 for a device's limits (KAT-2).
type HeapGeometry =
    {
        SlotBytes: uint64
        SlotArea: uint64
        ReservedRangeOffset: uint64
        ReservedRangeSize: uint64
        HeapRangeSize: uint64
    }

/// The five features the recipe enables, as the device reports them.
type RecipeFeatures =
    {
        ShaderStorageBufferArrayDynamicIndexing: bool
        BufferDeviceAddress: bool
        TimelineSemaphore: bool
        RuntimeDescriptorArray: bool
        Synchronization2: bool
        Maintenance5: bool
        DescriptorHeap: bool
    }

/// Reported, not required.
type OptionalFeatures =
    {
        DescriptorHeapCaptureReplay: bool
        ShaderUntypedPointers: bool
        CooperativeMatrix: bool
        ShaderFloat16: bool
        ShaderInt8: bool
        DeviceFault: bool
        DeviceFaultVendorBinary: bool
        MemoryBudget: bool
        CalibratedTimestamps: bool
        /// VkPhysicalDeviceVulkan12Properties.shaderDenormPreserveFloat32. The SPIR-V blobs declare no denormal
        /// mode (ShaderBuildAndSourceConvention), so this is information, not a requirement.
        ShaderDenormPreserveFloat32: bool
        ShaderDenormFlushToZeroFloat32: bool
    }

type MemoryHeapInfo =
    {
        Index: int
        Size: uint64
        DeviceLocal: bool
        /// VK_EXT_memory_budget, when supported.
        Budget: uint64 option
        Usage: uint64 option
    }

type MemoryTypeInfo =
    {
        Index: int
        Flags: VkMemoryPropertyFlags
        HeapIndex: int
    }

type CooperativeMatrixShape =
    {
        M: uint32
        N: uint32
        K: uint32
        AType: VkComponentTypeKHR
        BType: VkComponentTypeKHR
        CType: VkComponentTypeKHR
        ResultType: VkComponentTypeKHR
        Saturating: bool
        Scope: VkScopeKHR
    }

/// Everything the check observed on one physical device.
type Capabilities =
    {
        Identity: DeviceIdentity
        Extensions: Map<string, uint32>
        /// None without VK_EXT_descriptor_heap.
        DescriptorHeap: DescriptorHeapLimits option
        /// vkGetPhysicalDeviceDescriptorSizeEXT(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).
        StorageBufferDescriptorSize: uint64 option
        Features: RecipeFeatures
        Optional: OptionalFeatures
        /// The recipe's queue family, if the device has one.
        QueueFamily: uint32 option
        TimestampValidBits: uint32
        TimestampPeriod: float32
        MaxStorageBufferRange: uint32
        MinStorageBufferOffsetAlignment: uint64
        MaxComputeSharedMemorySize: uint32
        MaxComputeWorkGroupInvocations: uint32
        MemoryTypes: MemoryTypeInfo list
        MemoryHeaps: MemoryHeapInfo list
        CooperativeMatrixShapes: CooperativeMatrixShape list
        TimeDomains: VkTimeDomainKHR list
    }

type DeviceReport =
    {
        Capabilities: Capabilities
        /// Why the device is rejected; empty when it is accepted.
        Reasons: string list
    }

    member this.Accepted = this.Reasons.IsEmpty

let private isTrue (b: VkBool32) = b <> VkConst.VK_FALSE

let private alignUp (v: uint64) (a: uint64) =
    if a <= 1UL then v else (v + a - 1UL) / a * a

/// VulkanHeapRecipe section 3: 64 slots of bufferDescriptorSize, the reserved range after them aligned to
/// max(64, bufferDescriptorAlignment, imageDescriptorAlignment) and minResourceHeapReservedRange long.
let geometry (h: DescriptorHeapLimits) : HeapGeometry =
    let slotArea = uint64 HeapSlots * h.BufferDescriptorSize

    let offset =
        alignUp slotArea (max 64UL (max h.BufferDescriptorAlignment h.ImageDescriptorAlignment))

    {
        SlotBytes = h.BufferDescriptorSize
        SlotArea = slotArea
        ReservedRangeOffset = offset
        ReservedRangeSize = h.MinResourceHeapReservedRange
        HeapRangeSize = offset + h.MinResourceHeapReservedRange
    }

/// The driver version as its vendor writes it. NVIDIA packs 10.8.8.6 bits (616.92); Intel on Windows packs 18.14
/// bits; everyone else uses VK_MAKE_API_VERSION.
let formatDriverVersion (vendorId: uint32) (v: uint32) =
    match vendorId with
    | 0x10DEu -> sprintf "%d.%02d" ((v >>> 22) &&& 0x3FFu) ((v >>> 14) &&& 0xFFu)
    | 0x8086u when OperatingSystem.IsWindows() -> sprintf "%d.%d" (v >>> 14) (v &&& 0x3FFFu)
    | _ -> VkVersion.toString v

let private hex (bytes: byte[]) =
    Convert.ToHexString(bytes).ToLowerInvariant()

/// The rejection reasons for a device (VulkanHeapRecipe section 1); empty when it is accepted. A pure function of
/// what was observed, so the rules are tested on synthetic capabilities.
let reasons (c: Capabilities) : string list =
    let has = c.Extensions.ContainsKey
    let f = c.Features

    [
        if not (VkVersion.atLeast VkVersion.VK_API_VERSION_1_4 c.Identity.ApiVersion) then
            sprintf "device API version %s < 1.4" (VkVersion.toString c.Identity.ApiVersion)
        match c.DescriptorHeap with
        | None -> "VK_EXT_descriptor_heap not supported"
        | Some _ when not f.DescriptorHeap -> "VK_EXT_descriptor_heap present but descriptorHeap = VK_FALSE"
        | Some h ->
            if h.MaxPushDataSize < RequiredPushDataSize then
                sprintf "maxPushDataSize %d < %d" h.MaxPushDataSize RequiredPushDataSize

            match c.StorageBufferDescriptorSize with
            | Some s when s <> h.BufferDescriptorSize ->
                sprintf
                    "vkGetPhysicalDeviceDescriptorSizeEXT(STORAGE_BUFFER) = %d differs from bufferDescriptorSize %d"
                    s
                    h.BufferDescriptorSize
            | _ -> ()

            let g = geometry h

            if g.HeapRangeSize > h.MaxResourceHeapSize then
                sprintf
                    "the %d-slot heap needs %d bytes > maxResourceHeapSize %d"
                    HeapSlots
                    g.HeapRangeSize
                    h.MaxResourceHeapSize
        if not (has VkConst.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME) then
            "VK_KHR_shader_untyped_pointers not supported (VulkanHeapRecipe section 1)"
        if not f.ShaderStorageBufferArrayDynamicIndexing then
            "shaderStorageBufferArrayDynamicIndexing not supported"
        if not f.BufferDeviceAddress then
            "bufferDeviceAddress not supported"
        if not f.TimelineSemaphore then
            "timelineSemaphore not supported"
        if not f.RuntimeDescriptorArray then
            "runtimeDescriptorArray not supported"
        if not f.Synchronization2 then
            "synchronization2 not supported"
        if not f.Maintenance5 then
            "maintenance5 not supported"
        if uint64 c.MaxStorageBufferRange < PageBytes then
            sprintf "maxStorageBufferRange %d is below a 1 GiB page" c.MaxStorageBufferRange
        if c.QueueFamily.IsNone then
            "no queue family with COMPUTE, without GRAPHICS and with timestamps"
    ]

let private heapLimits (p: VkPhysicalDeviceDescriptorHeapPropertiesEXT) =
    {
        SamplerHeapAlignment = p.samplerHeapAlignment
        ResourceHeapAlignment = p.resourceHeapAlignment
        MaxSamplerHeapSize = p.maxSamplerHeapSize
        MaxResourceHeapSize = p.maxResourceHeapSize
        MinSamplerHeapReservedRange = p.minSamplerHeapReservedRange
        MinSamplerHeapReservedRangeWithEmbedded = p.minSamplerHeapReservedRangeWithEmbedded
        MinResourceHeapReservedRange = p.minResourceHeapReservedRange
        SamplerDescriptorSize = p.samplerDescriptorSize
        ImageDescriptorSize = p.imageDescriptorSize
        BufferDescriptorSize = p.bufferDescriptorSize
        SamplerDescriptorAlignment = p.samplerDescriptorAlignment
        ImageDescriptorAlignment = p.imageDescriptorAlignment
        BufferDescriptorAlignment = p.bufferDescriptorAlignment
        MaxPushDataSize = p.maxPushDataSize
        ImageCaptureReplayOpaqueDataSize = uint64 p.imageCaptureReplayOpaqueDataSize
        MaxDescriptorHeapEmbeddedSamplers = p.maxDescriptorHeapEmbeddedSamplers
        SamplerYcbcrConversionCount = p.samplerYcbcrConversionCount
        SparseDescriptorHeaps = isTrue p.sparseDescriptorHeaps
        ProtectedDescriptorHeaps = isTrue p.protectedDescriptorHeaps
    }

/// Observes one physical device: properties and features over the recipe's chains (only the structs its version and
/// extensions make valid), queues, memory with budgets, cooperative-matrix shapes and time domains.
let observe (instance: Instance) (device: VkPhysicalDevice) : Capabilities =
    let basic = instance.Properties device
    let api = basic.apiVersion
    let exts = instance.DeviceExtensions device
    let is11 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_1 api

    let mutable props = PropertiesChain()
    PropertiesChain.link &props (PropertiesChain.forQuery api exts) |> ignore
    let mutable feats = FeatureChain()
    let featureParts = FeatureChain.forQuery api exts
    FeatureChain.link &feats featureParts |> ignore

    // Properties2 and Features2 are core 1.1; a 1.0 device is judged from its base properties alone.
    if is11 then
        instance.Commands.GetPhysicalDeviceProperties2(device, &props.Properties2)
        instance.Commands.GetPhysicalDeviceFeatures2(device, &feats.Features2)

    let propParts = PropertiesChain.forQuery api exts
    let has = exts.ContainsKey
    let families = instance.QueueFamilies device
    let family = QueueFamily.recipe families
    let limits = basic.limits

    let memory, budget =
        match instance.MemoryBudget device with
        | Some(m, b) -> m, Some b
        | None -> instance.MemoryProperties device, None

    let identity =
        {
            Name = basic.deviceName.AsString
            Type = basic.deviceType
            ApiVersion = api
            DriverVersion = basic.driverVersion
            Driver = formatDriverVersion basic.vendorID basic.driverVersion
            VendorId = basic.vendorID
            DeviceId = basic.deviceID
            DriverId =
                if propParts.Vulkan12 then
                    props.Vulkan12.driverID
                else
                    enum 0
            DriverName =
                if propParts.Vulkan12 then
                    props.Vulkan12.driverName.AsString
                else
                    ""
            DriverInfo =
                if propParts.Vulkan12 then
                    props.Vulkan12.driverInfo.AsString
                else
                    ""
            DeviceUuid =
                if propParts.Vulkan11 then
                    hex (props.Vulkan11.deviceUUID.ToArray())
                else
                    ""
            DriverUuid =
                if propParts.Vulkan11 then
                    hex (props.Vulkan11.driverUUID.ToArray())
                else
                    ""
        }

    let heap =
        if propParts.DescriptorHeap && is11 then
            Some(heapLimits props.DescriptorHeap)
        else
            None

    let v12 = featureParts.Vulkan12 && is11
    let v13 = featureParts.Vulkan13 && is11

    {
        Identity = identity
        Extensions = exts
        DescriptorHeap = heap
        StorageBufferDescriptorSize =
            if heap.IsSome then
                instance.DescriptorSize(device, VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            else
                None
        Features =
            {
                ShaderStorageBufferArrayDynamicIndexing =
                    is11 && isTrue feats.Features2.features.shaderStorageBufferArrayDynamicIndexing
                BufferDeviceAddress = v12 && isTrue feats.Vulkan12.bufferDeviceAddress
                TimelineSemaphore = v12 && isTrue feats.Vulkan12.timelineSemaphore
                RuntimeDescriptorArray = v12 && isTrue feats.Vulkan12.runtimeDescriptorArray
                Synchronization2 = v13 && isTrue feats.Vulkan13.synchronization2
                Maintenance5 = featureParts.Maintenance5 && is11 && isTrue feats.Maintenance5.maintenance5
                DescriptorHeap =
                    featureParts.DescriptorHeap
                    && is11
                    && isTrue feats.DescriptorHeap.descriptorHeap
            }
        Optional =
            {
                DescriptorHeapCaptureReplay =
                    featureParts.DescriptorHeap
                    && is11
                    && isTrue feats.DescriptorHeap.descriptorHeapCaptureReplay
                ShaderUntypedPointers =
                    featureParts.UntypedPointers
                    && is11
                    && isTrue feats.UntypedPointers.shaderUntypedPointers
                CooperativeMatrix =
                    featureParts.CooperativeMatrix
                    && is11
                    && isTrue feats.CooperativeMatrix.cooperativeMatrix
                ShaderFloat16 = v12 && isTrue feats.Vulkan12.shaderFloat16
                ShaderInt8 = v12 && isTrue feats.Vulkan12.shaderInt8
                DeviceFault = featureParts.Fault && is11 && isTrue feats.Fault.deviceFault
                DeviceFaultVendorBinary = featureParts.Fault && is11 && isTrue feats.Fault.deviceFaultVendorBinary
                MemoryBudget = budget.IsSome
                CalibratedTimestamps = has VkConst.VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
                ShaderDenormPreserveFloat32 = propParts.Vulkan12 && isTrue props.Vulkan12.shaderDenormPreserveFloat32
                ShaderDenormFlushToZeroFloat32 =
                    propParts.Vulkan12 && isTrue props.Vulkan12.shaderDenormFlushToZeroFloat32
            }
        QueueFamily = family
        TimestampValidBits =
            match family with
            | Some f -> families[int f].timestampValidBits
            | None -> 0u
        TimestampPeriod = limits.timestampPeriod
        MaxStorageBufferRange = limits.maxStorageBufferRange
        MinStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment
        MaxComputeSharedMemorySize = limits.maxComputeSharedMemorySize
        MaxComputeWorkGroupInvocations = limits.maxComputeWorkGroupInvocations
        MemoryTypes =
            [
                for i in 0 .. int memory.memoryTypeCount - 1 ->
                    let t = memory.memoryTypes.ToArray()[i]

                    {
                        Index = i
                        Flags = t.propertyFlags
                        HeapIndex = int t.heapIndex
                    }
            ]
        MemoryHeaps =
            [
                let heaps = memory.memoryHeaps.ToArray()

                let budgets =
                    budget |> Option.map (fun b -> b.heapBudget.ToArray(), b.heapUsage.ToArray())

                for i in 0 .. int memory.memoryHeapCount - 1 ->
                    {
                        Index = i
                        Size = heaps[i].size
                        DeviceLocal = heaps[i].flags.HasFlag VkMemoryHeapFlagBits.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                        Budget = budgets |> Option.map (fun (b, _) -> b[i])
                        Usage = budgets |> Option.map (fun (_, u) -> u[i])
                    }
            ]
        CooperativeMatrixShapes =
            [
                for p in instance.CooperativeMatrixProperties device ->
                    {
                        M = p.MSize
                        N = p.NSize
                        K = p.KSize
                        AType = p.AType
                        BType = p.BType
                        CType = p.CType
                        ResultType = p.ResultType
                        Saturating = isTrue p.saturatingAccumulation
                        Scope = p.scope
                    }
            ]
        TimeDomains = instance.CalibrateableTimeDomains device |> List.ofArray
    }

let private emit (r: DeviceReport) =
    let c = r.Capabilities
    let id = c.Identity

    let attrs =
        [
            Attr.str TelemetryKeys.Backend "vulkan"
            Attr.str "device" id.Name
            Attr.int64 "vendor_id" (int64 id.VendorId)
            Attr.int64 "device_id" (int64 id.DeviceId)
            Attr.str "driver" id.Driver
            Attr.str "api" (VkVersion.toString id.ApiVersion)
            match c.DescriptorHeap with
            | Some h -> Attr.int64 "max_push_data_size" (int64 h.MaxPushDataSize)
            | None -> ()
            match r.Reasons with
            | [] -> ()
            | rs -> Attr.str "reason" (String.concat "; " rs)
        ]

    Telemetry.check "gpu.device.check" r.Accepted attrs

    for h in c.MemoryHeaps do
        let heapAttrs kind =
            [
                Attr.str TelemetryKeys.Backend "vulkan"
                Attr.str "device" id.Name
                Attr.str "pool" (sprintf "heap%d" h.Index)
                Attr.bool "device_local" h.DeviceLocal
                Attr.str "kind" kind
            ]

        Telemetry.gauge "gpu.device.memory" (float h.Size) TelemetryUnits.Bytes (heapAttrs "size")

        match h.Budget, h.Usage with
        | Some b, Some u ->
            Telemetry.gauge "gpu.device.memory" (float b) TelemetryUnits.Bytes (heapAttrs "budget")
            Telemetry.gauge "gpu.device.memory" (float u) TelemetryUnits.Bytes (heapAttrs "usage")
        | _ -> ()

/// Observes and judges one physical device; the verdict is reported as a gpu.device.check record and every memory
/// heap's size, budget and usage as gpu.device.memory gauges.
let check (instance: Instance) (device: VkPhysicalDevice) : DeviceReport =
    let c = observe instance device

    let r =
        {
            Capabilities = c
            Reasons = reasons c
        }

    emit r
    r

/// Creates a Vulkan 1.4 instance and checks every physical device, inside a gpu.device.discover span. Returns the
/// loader's instance version and the reports.
let checkAll (loader: VulkanLoader) : uint32 * DeviceReport list =
    use span =
        Telemetry.span "gpu.device.discover" [ Attr.str TelemetryKeys.Backend "vulkan" ]

    let instanceVersion = loader.InstanceVersion

    use instance =
        loader.CreateInstance(InstanceOptions.create "lrm-vk-device-check" false)

    let reports =
        instance.PhysicalDevices() |> Array.map (check instance) |> List.ofArray

    span.Add(Attr.int "devices" reports.Length)
    span.Add(Attr.int "accepted" (reports |> List.filter _.Accepted |> List.length))
    instanceVersion, reports

let private deviceTypeName =
    function
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU -> "integrated GPU"
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU -> "discrete GPU"
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU -> "virtual GPU"
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_CPU -> "CPU"
    | _ -> "other"

let private yesNo b = if b then "yes" else "no"

let private mib (b: uint64) = b / 1048576UL

/// The report as text lines, for `lrm devices-vulkan` and VkDeviceCheck.
let format (r: DeviceReport) : string list =
    let c = r.Capabilities
    let id = c.Identity

    [
        sprintf "%s (%s, vendor 0x%04x device 0x%04x)" id.Name (deviceTypeName id.Type) id.VendorId id.DeviceId
        sprintf
            "  Vulkan %s, driver %s (%s %s)"
            (VkVersion.toString id.ApiVersion)
            id.Driver
            id.DriverName
            id.DriverInfo
        if r.Accepted then
            "  verdict: ACCEPTED"
        else
            "  verdict: REJECTED"

            for reason in r.Reasons do
                "    - " + reason
        match c.DescriptorHeap with
        | Some h ->
            let g = geometry h
            "  descriptor heap:"

            sprintf
                "    maxPushDataSize                   %d (floor %d; the recipe pushes %d)"
                h.MaxPushDataSize
                RequiredPushDataSize
                RecipePushDataSize

            sprintf "    maxResourceHeapSize               %d" h.MaxResourceHeapSize
            sprintf "    resourceHeapAlignment             %d" h.ResourceHeapAlignment
            sprintf "    minResourceHeapReservedRange      %d" h.MinResourceHeapReservedRange
            sprintf "    bufferDescriptorSize / align      %d / %d" h.BufferDescriptorSize h.BufferDescriptorAlignment
            sprintf "    imageDescriptorSize / align       %d / %d" h.ImageDescriptorSize h.ImageDescriptorAlignment
            sprintf "    samplerDescriptorSize / align     %d / %d" h.SamplerDescriptorSize h.SamplerDescriptorAlignment

            sprintf
                "    heap geometry                     slots %d B, reserved range at %d, %d B; heapRange %d B"
                g.SlotArea
                g.ReservedRangeOffset
                g.ReservedRangeSize
                g.HeapRangeSize
        | None -> ()
        "  recipe features:"
        let f = c.Features
        sprintf "    shaderStorageBufferArrayDynamicIndexing %s" (yesNo f.ShaderStorageBufferArrayDynamicIndexing)

        sprintf
            "    bufferDeviceAddress %s, timelineSemaphore %s, runtimeDescriptorArray %s"
            (yesNo f.BufferDeviceAddress)
            (yesNo f.TimelineSemaphore)
            (yesNo f.RuntimeDescriptorArray)

        sprintf
            "    synchronization2 %s, maintenance5 %s, descriptorHeap %s"
            (yesNo f.Synchronization2)
            (yesNo f.Maintenance5)
            (yesNo f.DescriptorHeap)

        match c.QueueFamily with
        | Some q ->
            sprintf
                "  queue family %d: compute, timestamps %d bits, timestampPeriod %g ns"
                q
                c.TimestampValidBits
                c.TimestampPeriod
        | None -> "  queue family: none suitable"

        "  memory:"

        for h in c.MemoryHeaps do
            sprintf
                "    heap %d: %d MiB%s%s"
                h.Index
                (mib h.Size)
                (if h.DeviceLocal then " device-local" else "")
                (match h.Budget, h.Usage with
                 | Some b, Some u -> sprintf ", budget %d MiB, usage %d MiB" (mib b) (mib u)
                 | _ -> "")

        for t in c.MemoryTypes do
            sprintf "    type %d: heap %d, %A" t.Index t.HeapIndex t.Flags

        "  optional:"
        let o = c.Optional
        sprintf "    descriptorHeapCaptureReplay       %s" (yesNo o.DescriptorHeapCaptureReplay)
        sprintf "    VK_KHR_shader_untyped_pointers    %s" (yesNo o.ShaderUntypedPointers)

        sprintf
            "    VK_KHR_cooperative_matrix         %s (%d shapes)"
            (yesNo o.CooperativeMatrix)
            c.CooperativeMatrixShapes.Length

        sprintf
            "    VK_EXT_device_fault               %s (vendor binary %s)"
            (yesNo o.DeviceFault)
            (yesNo o.DeviceFaultVendorBinary)

        sprintf "    VK_EXT_memory_budget              %s" (yesNo o.MemoryBudget)

        sprintf
            "    VK_KHR_calibrated_timestamps      %s (%d time domains)"
            (yesNo o.CalibratedTimestamps)
            c.TimeDomains.Length

        sprintf "    shaderDenormPreserveFloat32       %s" (yesNo o.ShaderDenormPreserveFloat32)
        sprintf "    shaderDenormFlushToZeroFloat32    %s" (yesNo o.ShaderDenormFlushToZeroFloat32)
        sprintf "    shaderFloat16                     %s" (yesNo o.ShaderFloat16)
        sprintf "    shaderInt8                        %s" (yesNo o.ShaderInt8)
    ]

/// The reports as JSON, for `lrm devices-vulkan --json`.
let toJson (instanceVersion: uint32) (reports: DeviceReport list) : string =
    let root = JsonObject()
    root["loaderInstanceVersion"] <- JsonValue.Create(VkVersion.toString instanceVersion)
    let arr = JsonArray()

    for r in reports do
        let c = r.Capabilities
        let id = c.Identity
        let o = JsonObject()
        o["name"] <- JsonValue.Create id.Name
        o["type"] <- JsonValue.Create(deviceTypeName id.Type)
        o["vendorId"] <- JsonValue.Create id.VendorId
        o["deviceId"] <- JsonValue.Create id.DeviceId
        o["apiVersion"] <- JsonValue.Create(VkVersion.toString id.ApiVersion)
        o["driver"] <- JsonValue.Create id.Driver
        o["driverName"] <- JsonValue.Create id.DriverName
        o["driverInfo"] <- JsonValue.Create id.DriverInfo
        o["deviceUuid"] <- JsonValue.Create id.DeviceUuid
        o["verdict"] <- JsonValue.Create(if r.Accepted then "ACCEPTED" else "REJECTED")
        let rs = JsonArray()

        for x in r.Reasons do
            rs.Add(JsonValue.Create x)

        o["reasons"] <- rs

        match c.DescriptorHeap with
        | Some h ->
            let g = geometry h
            let d = JsonObject()
            d["maxPushDataSize"] <- JsonValue.Create h.MaxPushDataSize
            d["maxResourceHeapSize"] <- JsonValue.Create h.MaxResourceHeapSize
            d["resourceHeapAlignment"] <- JsonValue.Create h.ResourceHeapAlignment
            d["minResourceHeapReservedRange"] <- JsonValue.Create h.MinResourceHeapReservedRange
            d["bufferDescriptorSize"] <- JsonValue.Create h.BufferDescriptorSize
            d["bufferDescriptorAlignment"] <- JsonValue.Create h.BufferDescriptorAlignment
            d["imageDescriptorAlignment"] <- JsonValue.Create h.ImageDescriptorAlignment
            d["reservedRangeOffset"] <- JsonValue.Create g.ReservedRangeOffset
            d["heapRangeSize"] <- JsonValue.Create g.HeapRangeSize
            o["descriptorHeap"] <- d
        | None -> ()

        let f = JsonObject()

        f["shaderStorageBufferArrayDynamicIndexing"] <-
            JsonValue.Create c.Features.ShaderStorageBufferArrayDynamicIndexing

        f["bufferDeviceAddress"] <- JsonValue.Create c.Features.BufferDeviceAddress
        f["timelineSemaphore"] <- JsonValue.Create c.Features.TimelineSemaphore
        f["runtimeDescriptorArray"] <- JsonValue.Create c.Features.RuntimeDescriptorArray
        f["synchronization2"] <- JsonValue.Create c.Features.Synchronization2
        f["maintenance5"] <- JsonValue.Create c.Features.Maintenance5
        f["descriptorHeap"] <- JsonValue.Create c.Features.DescriptorHeap
        o["features"] <- f
        let opt = JsonObject()
        opt["deviceFault"] <- JsonValue.Create c.Optional.DeviceFault
        opt["memoryBudget"] <- JsonValue.Create c.Optional.MemoryBudget
        opt["cooperativeMatrix"] <- JsonValue.Create c.Optional.CooperativeMatrix
        opt["calibratedTimestamps"] <- JsonValue.Create c.Optional.CalibratedTimestamps
        opt["shaderDenormPreserveFloat32"] <- JsonValue.Create c.Optional.ShaderDenormPreserveFloat32
        opt["shaderUntypedPointers"] <- JsonValue.Create c.Optional.ShaderUntypedPointers
        o["optional"] <- opt

        match c.QueueFamily with
        | Some q -> o["queueFamily"] <- JsonValue.Create q
        | None -> ()

        o["timestampValidBits"] <- JsonValue.Create c.TimestampValidBits
        o["timestampPeriod"] <- JsonValue.Create c.TimestampPeriod
        let heaps = JsonArray()

        for h in c.MemoryHeaps do
            let x = JsonObject()
            x["index"] <- JsonValue.Create h.Index
            x["size"] <- JsonValue.Create h.Size
            x["deviceLocal"] <- JsonValue.Create h.DeviceLocal

            match h.Budget, h.Usage with
            | Some b, Some u ->
                x["budget"] <- JsonValue.Create b
                x["usage"] <- JsonValue.Create u
            | _ -> ()

            heaps.Add x

        o["memoryHeaps"] <- heaps
        let shapes = JsonArray()

        for s in c.CooperativeMatrixShapes do
            shapes.Add(
                JsonValue.Create(
                    sprintf "%dx%dx%d %A*%A+%A->%A %A" s.M s.N s.K s.AType s.BType s.CType s.ResultType s.Scope
                )
            )

        o["cooperativeMatrixShapes"] <- shapes
        arr.Add o

    root["devices"] <- arr
    root.ToJsonString(JsonSerializerOptions(WriteIndented = true))
