/// Device acceptance for the LRM Vulkan backend: Vulkan 1.4 + VK_EXT_descriptor_heap, no fallbacks.
module Lrm.Gpu.Vulkan.DeviceCheck

#nowarn "9"
#nowarn "51"

open Microsoft.FSharp.NativeInterop

/// Push data the runtime passes per dispatch (tensor tokens), mirroring 64 DWORDs of D3D12 root constants.
[<Literal>]
let RequiredPushDataSize = 256UL

type DeviceIdentity =
    {
        Name: string
        Type: VkPhysicalDeviceType
        ApiVersion: uint32
        DriverVersion: uint32
        VendorId: uint32
        DeviceId: uint32
        DriverName: string
        DriverInfo: string
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

/// Reported, not required. Extension-backed entries are true only if the extension is present
/// *and* its feature bit is set.
type OptionalFeatures =
    {
        DescriptorHeapCaptureReplay: bool
        ShaderUntypedPointers: bool
        CooperativeMatrix: bool
        ShaderFloat16: bool
        ShaderInt8: bool
    }

type Capabilities =
    {
        Identity: DeviceIdentity
        DescriptorHeap: DescriptorHeapLimits
        Optional: OptionalFeatures
        Extensions: Map<string, uint32>
    }

type DeviceVerdict =
    | Accepted of Capabilities
    | Rejected of reasons: string list

type DeviceReport =
    {
        Identity: DeviceIdentity
        /// What was observed for the optional items, whatever the verdict.
        Optional: OptionalFeatures
        Verdict: DeviceVerdict
    }

let private isTrue (b: VkBool32) = b <> VkConst.VK_FALSE

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

/// Prepends `s` (a struct on this stack frame) to a pNext chain.
let inline private link (s: nativeptr<'T>) (pNextOfS: byref<nativeint>) (head: byref<nativeint>) =
    pNextOfS <- head
    head <- NativePtr.toNativeInt s

/// Queries one physical device. Structs are only chained when the device's core version or an
/// advertised extension makes them valid, so the query itself never relies on undefined behaviour.
let check (instance: Instance) (device: VkPhysicalDevice) : DeviceReport =
    let mutable props = VkPhysicalDeviceProperties()
    instance.GetPhysicalDeviceProperties.Invoke(device, &&props)
    let api = props.apiVersion
    let exts = instance.DeviceExtensions device
    let has name = exts.ContainsKey name
    let is11 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_1 api
    let is12 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_2 api
    let is13 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_3 api
    let is14 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_4 api
    let hasHeap = has VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME
    let hasUntyped = has VkConst.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME
    let hasCoopMat = has VkConst.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME

    // ---- properties chain ----
    let mutable props2 = VkPhysicalDeviceProperties2()
    let mutable driver = VkPhysicalDeviceDriverProperties()
    let mutable heapProps = VkPhysicalDeviceDescriptorHeapPropertiesEXT()
    props2.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
    driver.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES
    heapProps.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT
    let mutable propsHead = 0n

    if is12 then
        link &&driver &driver.pNext &propsHead

    if hasHeap then
        link &&heapProps &heapProps.pNext &propsHead

    props2.pNext <- propsHead

    // ---- features chain ----
    let mutable feats2 = VkPhysicalDeviceFeatures2()
    let mutable v12 = VkPhysicalDeviceVulkan12Features()
    let mutable v13 = VkPhysicalDeviceVulkan13Features()
    let mutable heapFeats = VkPhysicalDeviceDescriptorHeapFeaturesEXT()
    let mutable untyped = VkPhysicalDeviceShaderUntypedPointersFeaturesKHR()
    let mutable coopMat = VkPhysicalDeviceCooperativeMatrixFeaturesKHR()
    feats2.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
    v12.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
    v13.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
    heapFeats.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT
    untyped.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR
    coopMat.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
    let mutable featsHead = 0n

    if is12 then
        link &&v12 &v12.pNext &featsHead

    if is13 then
        link &&v13 &v13.pNext &featsHead

    if hasHeap then
        link &&heapFeats &heapFeats.pNext &featsHead

    if hasUntyped then
        link &&untyped &untyped.pNext &featsHead

    if hasCoopMat then
        link &&coopMat &coopMat.pNext &featsHead

    feats2.pNext <- featsHead

    // Properties2/Features2 are core 1.1; a 1.0 device is rejected from the base properties alone.
    if is11 then
        instance.GetPhysicalDeviceProperties2.Invoke(device, &&props2)
        instance.GetPhysicalDeviceFeatures2.Invoke(device, &&feats2)

    let identity =
        {
            Name = props.deviceName.AsString
            Type = props.deviceType
            ApiVersion = api
            DriverVersion = props.driverVersion
            VendorId = props.vendorID
            DeviceId = props.deviceID
            DriverName = if is12 then driver.driverName.AsString else ""
            DriverInfo = if is12 then driver.driverInfo.AsString else ""
        }

    let optional =
        {
            DescriptorHeapCaptureReplay = hasHeap && isTrue heapFeats.descriptorHeapCaptureReplay
            ShaderUntypedPointers = hasUntyped && isTrue untyped.shaderUntypedPointers
            CooperativeMatrix = hasCoopMat && isTrue coopMat.cooperativeMatrix
            ShaderFloat16 = is12 && isTrue v12.shaderFloat16
            ShaderInt8 = is12 && isTrue v12.shaderInt8
        }

    let reasons =
        [
            if not is14 then
                sprintf "device API version %s < 1.4" (VkVersion.toString api)
            if not hasHeap then
                "VK_EXT_descriptor_heap not supported"
            elif not (isTrue heapFeats.descriptorHeap) then
                "VK_EXT_descriptor_heap present but descriptorHeap = VK_FALSE"
            elif heapProps.maxPushDataSize < RequiredPushDataSize then
                sprintf "maxPushDataSize %d < %d" heapProps.maxPushDataSize RequiredPushDataSize
            if not (is12 && isTrue v12.bufferDeviceAddress) then
                "bufferDeviceAddress not supported"
            if not (is12 && isTrue v12.timelineSemaphore) then
                "timelineSemaphore not supported"
            if not (is13 && isTrue v13.synchronization2) then
                "synchronization2 not supported"
        ]

    let verdict =
        match reasons with
        | [] ->
            Accepted
                {
                    Identity = identity
                    DescriptorHeap = heapLimits heapProps
                    Optional = optional
                    Extensions = exts
                }
        | rs -> Rejected rs

    {
        Identity = identity
        Optional = optional
        Verdict = verdict
    }

/// Creates a Vulkan 1.4 instance and checks every physical device.
let checkAll (loader: VulkanLoader) : uint32 * DeviceReport list =
    let instanceVersion = loader.InstanceVersion

    use instance =
        loader.CreateInstance("lrm-vk-device-check", VkVersion.VK_API_VERSION_1_4)

    instanceVersion, instance.PhysicalDevices |> Array.map (check instance) |> List.ofArray
