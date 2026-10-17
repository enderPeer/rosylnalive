// Physical-device query structs. Member order and sType values from vk.xml (VK_HEADER_VERSION 365);
// VkPhysicalDeviceLimits / SparseProperties / Features live in VkCore.Generated.fs.
namespace Lrm.Gpu.Vulkan

open System.Runtime.InteropServices

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceProperties =
    val mutable apiVersion: uint32
    val mutable driverVersion: uint32
    val mutable vendorID: uint32
    val mutable deviceID: uint32
    val mutable deviceType: VkPhysicalDeviceType
    val mutable deviceName: FixedBytes256
    val mutable pipelineCacheUUID: FixedBytes16
    val mutable limits: VkPhysicalDeviceLimits
    val mutable sparseProperties: VkPhysicalDeviceSparseProperties

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceProperties2 =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable properties: VkPhysicalDeviceProperties

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceFeatures2 =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable features: VkPhysicalDeviceFeatures

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkConformanceVersion =
    val mutable major: byte
    val mutable minor: byte
    val mutable subminor: byte
    val mutable patch: byte

/// Core 1.2 (was VK_KHR_driver_properties). Extends VkPhysicalDeviceProperties2.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceDriverProperties =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable driverID: VkDriverId
    val mutable driverName: FixedBytes256
    val mutable driverInfo: FixedBytes256
    val mutable conformanceVersion: VkConformanceVersion

/// Core 1.2. Extends VkPhysicalDeviceFeatures2.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceVulkan12Features =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable samplerMirrorClampToEdge: VkBool32
    val mutable drawIndirectCount: VkBool32
    val mutable storageBuffer8BitAccess: VkBool32
    val mutable uniformAndStorageBuffer8BitAccess: VkBool32
    val mutable storagePushConstant8: VkBool32
    val mutable shaderBufferInt64Atomics: VkBool32
    val mutable shaderSharedInt64Atomics: VkBool32
    val mutable shaderFloat16: VkBool32
    val mutable shaderInt8: VkBool32
    val mutable descriptorIndexing: VkBool32
    val mutable shaderInputAttachmentArrayDynamicIndexing: VkBool32
    val mutable shaderUniformTexelBufferArrayDynamicIndexing: VkBool32
    val mutable shaderStorageTexelBufferArrayDynamicIndexing: VkBool32
    val mutable shaderUniformBufferArrayNonUniformIndexing: VkBool32
    val mutable shaderSampledImageArrayNonUniformIndexing: VkBool32
    val mutable shaderStorageBufferArrayNonUniformIndexing: VkBool32
    val mutable shaderStorageImageArrayNonUniformIndexing: VkBool32
    val mutable shaderInputAttachmentArrayNonUniformIndexing: VkBool32
    val mutable shaderUniformTexelBufferArrayNonUniformIndexing: VkBool32
    val mutable shaderStorageTexelBufferArrayNonUniformIndexing: VkBool32
    val mutable descriptorBindingUniformBufferUpdateAfterBind: VkBool32
    val mutable descriptorBindingSampledImageUpdateAfterBind: VkBool32
    val mutable descriptorBindingStorageImageUpdateAfterBind: VkBool32
    val mutable descriptorBindingStorageBufferUpdateAfterBind: VkBool32
    val mutable descriptorBindingUniformTexelBufferUpdateAfterBind: VkBool32
    val mutable descriptorBindingStorageTexelBufferUpdateAfterBind: VkBool32
    val mutable descriptorBindingUpdateUnusedWhilePending: VkBool32
    val mutable descriptorBindingPartiallyBound: VkBool32
    val mutable descriptorBindingVariableDescriptorCount: VkBool32
    val mutable runtimeDescriptorArray: VkBool32
    val mutable samplerFilterMinmax: VkBool32
    val mutable scalarBlockLayout: VkBool32
    val mutable imagelessFramebuffer: VkBool32
    val mutable uniformBufferStandardLayout: VkBool32
    val mutable shaderSubgroupExtendedTypes: VkBool32
    val mutable separateDepthStencilLayouts: VkBool32
    val mutable hostQueryReset: VkBool32
    val mutable timelineSemaphore: VkBool32
    val mutable bufferDeviceAddress: VkBool32
    val mutable bufferDeviceAddressCaptureReplay: VkBool32
    val mutable bufferDeviceAddressMultiDevice: VkBool32
    val mutable vulkanMemoryModel: VkBool32
    val mutable vulkanMemoryModelDeviceScope: VkBool32
    val mutable vulkanMemoryModelAvailabilityVisibilityChains: VkBool32
    val mutable shaderOutputViewportIndex: VkBool32
    val mutable shaderOutputLayer: VkBool32
    val mutable subgroupBroadcastDynamicId: VkBool32

/// Core 1.3. Extends VkPhysicalDeviceFeatures2.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceVulkan13Features =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable robustImageAccess: VkBool32
    val mutable inlineUniformBlock: VkBool32
    val mutable descriptorBindingInlineUniformBlockUpdateAfterBind: VkBool32
    val mutable pipelineCreationCacheControl: VkBool32
    val mutable privateData: VkBool32
    val mutable shaderDemoteToHelperInvocation: VkBool32
    val mutable shaderTerminateInvocation: VkBool32
    val mutable subgroupSizeControl: VkBool32
    val mutable computeFullSubgroups: VkBool32
    val mutable synchronization2: VkBool32
    val mutable textureCompressionASTC_HDR: VkBool32
    val mutable shaderZeroInitializeWorkgroupMemory: VkBool32
    val mutable dynamicRendering: VkBool32
    val mutable shaderIntegerDotProduct: VkBool32
    val mutable maintenance4: VkBool32

// ---- VK_EXT_descriptor_heap (extension 136, spec version 1) ----
// Only the query structs: the extension adds no entry points needed to *query* support.

/// Extends VkPhysicalDeviceFeatures2 / VkDeviceCreateInfo.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceDescriptorHeapFeaturesEXT =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable descriptorHeap: VkBool32
    val mutable descriptorHeapCaptureReplay: VkBool32

/// Extends VkPhysicalDeviceProperties2.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceDescriptorHeapPropertiesEXT =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable samplerHeapAlignment: uint64
    val mutable resourceHeapAlignment: uint64
    val mutable maxSamplerHeapSize: uint64
    val mutable maxResourceHeapSize: uint64
    val mutable minSamplerHeapReservedRange: uint64
    val mutable minSamplerHeapReservedRangeWithEmbedded: uint64
    val mutable minResourceHeapReservedRange: uint64
    val mutable samplerDescriptorSize: uint64
    val mutable imageDescriptorSize: uint64
    val mutable bufferDescriptorSize: uint64
    val mutable samplerDescriptorAlignment: uint64
    val mutable imageDescriptorAlignment: uint64
    val mutable bufferDescriptorAlignment: uint64
    val mutable maxPushDataSize: uint64
    val mutable imageCaptureReplayOpaqueDataSize: unativeint
    val mutable maxDescriptorHeapEmbeddedSamplers: uint32
    val mutable samplerYcbcrConversionCount: uint32
    val mutable sparseDescriptorHeaps: VkBool32
    val mutable protectedDescriptorHeaps: VkBool32

// ---- Optional extensions the device check reports on ----

/// VK_KHR_shader_untyped_pointers. Extends VkPhysicalDeviceFeatures2.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceShaderUntypedPointersFeaturesKHR =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable shaderUntypedPointers: VkBool32

/// VK_KHR_cooperative_matrix. Extends VkPhysicalDeviceFeatures2.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceCooperativeMatrixFeaturesKHR =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable cooperativeMatrix: VkBool32
    val mutable cooperativeMatrixRobustBufferAccess: VkBool32
