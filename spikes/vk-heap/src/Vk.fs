/// Hand-written Vulkan bindings the spike needs beyond Lrm.Gpu.Vulkan.Interop (which is query-only): logical device,
/// memory, buffers, pipelines, command buffers, timeline semaphores, timestamps and VK_EXT_descriptor_heap. Member
/// order follows the C declarations; Layout.fs re-declares each one and the spike checks both agree at run time.
/// Handles: dispatchable ones (VkDevice, VkQueue, VkCommandBuffer) are nativeint, non-dispatchable ones uint64.
module VkHeapSpike.Vk

#nowarn "9"

open System
open System.Runtime.InteropServices
open Lrm.Gpu.Vulkan

// ---- constants (values from the registry reference pages) ----

[<RequireQualifiedAccess>]
module SType =
    let DeviceQueueCreateInfo = 2u
    let DeviceCreateInfo = 3u
    let MemoryAllocateInfo = 5u
    let MappedMemoryRange = 6u
    let SemaphoreCreateInfo = 9u
    let QueryPoolCreateInfo = 11u
    let BufferCreateInfo = 12u
    let ShaderModuleCreateInfo = 16u
    let PipelineShaderStageCreateInfo = 18u
    let ComputePipelineCreateInfo = 29u
    let CommandPoolCreateInfo = 39u
    let CommandBufferAllocateInfo = 40u
    let CommandBufferBeginInfo = 42u
    let PhysicalDeviceVulkan12Features = 51u
    let PhysicalDeviceVulkan13Features = 53u
    let MemoryAllocateFlagsInfo = 1000060000u
    let SemaphoreTypeCreateInfo = 1000207002u
    let SemaphoreWaitInfo = 1000207004u
    let BufferDeviceAddressInfo = 1000244001u
    let MemoryBarrier2 = 1000314000u
    let DependencyInfo = 1000314003u
    let SubmitInfo2 = 1000314004u
    let SemaphoreSubmitInfo = 1000314005u
    let CommandBufferSubmitInfo = 1000314006u
    let PhysicalDeviceMaintenance5Features = 1000470000u
    let PipelineCreateFlags2CreateInfo = 1000470005u
    // VK_EXT_descriptor_heap (extension 136 -> 1000135xxx)
    let ResourceDescriptorInfoEXT = 1000135002u
    let BindHeapInfoEXT = 1000135003u
    let PushDataInfoEXT = 1000135004u
    let DescriptorSetAndBindingMappingEXT = 1000135005u
    let ShaderDescriptorSetAndBindingMappingInfoEXT = 1000135006u
    let PhysicalDeviceDescriptorHeapPropertiesEXT = 1000135008u
    let PhysicalDeviceDescriptorHeapFeaturesEXT = 1000135009u

[<RequireQualifiedAccess>]
module Flags =
    // VkQueueFlagBits
    let QueueGraphics = 0x1u
    let QueueCompute = 0x2u
    let QueueTransfer = 0x4u
    // VkMemoryPropertyFlagBits
    let MemDeviceLocal = 0x1u
    let MemHostVisible = 0x2u
    let MemHostCoherent = 0x4u
    let MemHostCached = 0x8u
    // VkMemoryHeapFlagBits
    let HeapDeviceLocal = 0x1u
    // VkBufferUsageFlagBits
    let BufTransferSrc = 0x1u
    let BufTransferDst = 0x2u
    let BufStorage = 0x20u
    let BufShaderDeviceAddress = 0x20000u
    let BufDescriptorHeapEXT = 0x10000000u
    // VkMemoryAllocateFlagBits
    let AllocDeviceAddress = 0x2u
    // VkPipelineCreateFlagBits2
    let Pipeline2DescriptorHeapEXT = 0x1000000000UL
    // VkShaderStageFlagBits
    let StageCompute = 0x20u
    // VkCommandPoolCreateFlagBits / VkCommandBufferUsageFlagBits
    let PoolTransient = 0x1u
    let CmdOneTimeSubmit = 0x1u
    // VkPipelineStageFlagBits2
    let Stage2None = 0UL
    let Stage2ComputeShader = 0x800UL
    let Stage2Host = 0x4000UL
    let Stage2AllCommands = 0x10000UL
    let Stage2Copy = 0x100000000UL
    // VkAccessFlagBits2
    let Access2TransferRead = 0x800UL
    let Access2TransferWrite = 0x1000UL
    let Access2HostRead = 0x2000UL
    let Access2HostWrite = 0x4000UL
    let Access2ShaderStorageRead = 0x200000000UL
    let Access2ShaderStorageWrite = 0x400000000UL
    let Access2ResourceHeapReadEXT = 0x400000000000000UL
    // VkSpirvResourceTypeFlagBitsEXT
    let SpirvReadOnlyStorageBuffer = 0x40u
    let SpirvReadWriteStorageBuffer = 0x80u
    // VkQueryResultFlagBits
    let Query64 = 0x1u
    let QueryWait = 0x2u

[<RequireQualifiedAccess>]
module Enum =
    let SharingExclusive = 0u
    let DescriptorTypeStorageBuffer = 7u
    let BindPointCompute = 1u
    let CommandBufferLevelPrimary = 0u
    let SemaphoreTypeTimeline = 1u
    let QueryTypeTimestamp = 2u
    let MappingSourceHeapWithConstantOffset = 0u

let VK_WHOLE_SIZE = UInt64.MaxValue

// ---- structs ----

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkExtent3D =
    val mutable width: uint32
    val mutable height: uint32
    val mutable depth: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkQueueFamilyProperties =
    val mutable queueFlags: uint32
    val mutable queueCount: uint32
    val mutable timestampValidBits: uint32
    val mutable minImageTransferGranularity: VkExtent3D

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMemoryType =
    val mutable propertyFlags: uint32
    val mutable heapIndex: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMemoryHeap =
    val mutable size: uint64
    val mutable flags: uint32

/// VkMemoryType[32]
[<Struct; StructLayout(LayoutKind.Sequential, Size = 256)>]
type VkMemoryTypeArray32 =
    val private e0: VkMemoryType

/// VkMemoryHeap[16]
[<Struct; StructLayout(LayoutKind.Sequential, Size = 256)>]
type VkMemoryHeapArray16 =
    val private e0: VkMemoryHeap

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceMemoryProperties =
    val mutable memoryTypeCount: uint32
    val mutable memoryTypes: VkMemoryTypeArray32
    val mutable memoryHeapCount: uint32
    val mutable memoryHeaps: VkMemoryHeapArray16

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDeviceQueueCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable queueFamilyIndex: uint32
    val mutable queueCount: uint32
    val mutable pQueuePriorities: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDeviceCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable queueCreateInfoCount: uint32
    val mutable pQueueCreateInfos: nativeint
    val mutable enabledLayerCount: uint32
    val mutable ppEnabledLayerNames: nativeint
    val mutable enabledExtensionCount: uint32
    val mutable ppEnabledExtensionNames: nativeint
    val mutable pEnabledFeatures: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDeviceMaintenance5Features =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable maintenance5: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkBufferCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable size: uint64
    val mutable usage: uint32
    val mutable sharingMode: uint32
    val mutable queueFamilyIndexCount: uint32
    val mutable pQueueFamilyIndices: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMemoryRequirements =
    val mutable size: uint64
    val mutable alignment: uint64
    val mutable memoryTypeBits: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMemoryAllocateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable allocationSize: uint64
    val mutable memoryTypeIndex: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMemoryAllocateFlagsInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable deviceMask: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkBufferDeviceAddressInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable buffer: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMappedMemoryRange =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable memory: uint64
    val mutable offset: uint64
    val mutable size: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkShaderModuleCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable codeSize: unativeint
    val mutable pCode: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPipelineShaderStageCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable stage: uint32
    val mutable ``module``: uint64
    val mutable pName: nativeint
    val mutable pSpecializationInfo: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkComputePipelineCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable stage: VkPipelineShaderStageCreateInfo
    val mutable layout: uint64
    val mutable basePipelineHandle: uint64
    val mutable basePipelineIndex: int32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPipelineCreateFlags2CreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkCommandPoolCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable queueFamilyIndex: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkCommandBufferAllocateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable commandPool: uint64
    val mutable level: uint32
    val mutable commandBufferCount: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkCommandBufferBeginInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable pInheritanceInfo: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkBufferCopy =
    val mutable srcOffset: uint64
    val mutable dstOffset: uint64
    val mutable size: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkMemoryBarrier2 =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable srcStageMask: uint64
    val mutable srcAccessMask: uint64
    val mutable dstStageMask: uint64
    val mutable dstAccessMask: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDependencyInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable dependencyFlags: uint32
    val mutable memoryBarrierCount: uint32
    val mutable pMemoryBarriers: nativeint
    val mutable bufferMemoryBarrierCount: uint32
    val mutable pBufferMemoryBarriers: nativeint
    val mutable imageMemoryBarrierCount: uint32
    val mutable pImageMemoryBarriers: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkSemaphoreTypeCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable semaphoreType: uint32
    val mutable initialValue: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkSemaphoreCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkSemaphoreSubmitInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable semaphore: uint64
    val mutable value: uint64
    val mutable stageMask: uint64
    val mutable deviceIndex: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkCommandBufferSubmitInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable commandBuffer: nativeint
    val mutable deviceMask: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkSubmitInfo2 =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable waitSemaphoreInfoCount: uint32
    val mutable pWaitSemaphoreInfos: nativeint
    val mutable commandBufferInfoCount: uint32
    val mutable pCommandBufferInfos: nativeint
    val mutable signalSemaphoreInfoCount: uint32
    val mutable pSignalSemaphoreInfos: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkSemaphoreWaitInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable semaphoreCount: uint32
    val mutable pSemaphores: nativeint
    val mutable pValues: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkQueryPoolCreateInfo =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable queryType: uint32
    val mutable queryCount: uint32
    val mutable pipelineStatistics: uint32

// ---- VK_EXT_descriptor_heap ----

/// typedef VkDeviceAddressRangeKHR VkDeviceAddressRangeEXT
[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDeviceAddressRangeEXT =
    val mutable address: uint64
    val mutable size: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkHostAddressRangeEXT =
    val mutable address: nativeint
    val mutable size: unativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkHostAddressRangeConstEXT =
    val mutable address: nativeint
    val mutable size: unativeint

/// union VkResourceDescriptorDataEXT: four pointers, so one pointer-sized slot. Only pAddressRange is used.
[<Struct; StructLayout(LayoutKind.Explicit, Size = 8)>]
type VkResourceDescriptorDataEXT =
    [<FieldOffset(0)>]
    val mutable pAddressRange: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkResourceDescriptorInfoEXT =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable ``type``: uint32
    val mutable data: VkResourceDescriptorDataEXT

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkBindHeapInfoEXT =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable heapRange: VkDeviceAddressRangeEXT
    val mutable reservedRangeOffset: uint64
    val mutable reservedRangeSize: uint64

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPushDataInfoEXT =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable offset: uint32
    val mutable data: VkHostAddressRangeConstEXT

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDescriptorMappingSourceConstantOffsetEXT =
    val mutable heapOffset: uint32
    val mutable heapArrayStride: uint32
    val mutable pEmbeddedSampler: nativeint
    val mutable samplerHeapOffset: uint32
    val mutable samplerHeapArrayStride: uint32

/// union VkDescriptorMappingSourceDataEXT: 56 bytes (its largest member, VkDescriptorMappingSourceIndirectIndexEXT),
/// 8-byte aligned. Only constantOffset is used.
[<Struct; StructLayout(LayoutKind.Explicit, Size = 56)>]
type VkDescriptorMappingSourceDataEXT =
    [<FieldOffset(0)>]
    val mutable constantOffset: VkDescriptorMappingSourceConstantOffsetEXT

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDescriptorSetAndBindingMappingEXT =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable descriptorSet: uint32
    val mutable firstBinding: uint32
    val mutable bindingCount: uint32
    val mutable resourceMask: uint32
    val mutable source: uint32
    val mutable sourceData: VkDescriptorMappingSourceDataEXT

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkShaderDescriptorSetAndBindingMappingInfoEXT =
    val mutable sType: uint32
    val mutable pNext: nativeint
    val mutable mappingCount: uint32
    val mutable pMappings: nativeint

/// Every hand-written struct and the C declaration it must match.
let checkedStructs: (Type * string) list =
    [
        typeof<VkExtent3D>, "VkExtent3D"
        typeof<VkQueueFamilyProperties>, "VkQueueFamilyProperties"
        typeof<VkMemoryType>, "VkMemoryType"
        typeof<VkMemoryHeap>, "VkMemoryHeap"
        typeof<VkPhysicalDeviceMemoryProperties>, "VkPhysicalDeviceMemoryProperties"
        typeof<VkDeviceQueueCreateInfo>, "VkDeviceQueueCreateInfo"
        typeof<VkDeviceCreateInfo>, "VkDeviceCreateInfo"
        typeof<VkPhysicalDeviceMaintenance5Features>, "VkPhysicalDeviceMaintenance5Features"
        typeof<VkBufferCreateInfo>, "VkBufferCreateInfo"
        typeof<VkMemoryRequirements>, "VkMemoryRequirements"
        typeof<VkMemoryAllocateInfo>, "VkMemoryAllocateInfo"
        typeof<VkMemoryAllocateFlagsInfo>, "VkMemoryAllocateFlagsInfo"
        typeof<VkBufferDeviceAddressInfo>, "VkBufferDeviceAddressInfo"
        typeof<VkMappedMemoryRange>, "VkMappedMemoryRange"
        typeof<VkShaderModuleCreateInfo>, "VkShaderModuleCreateInfo"
        typeof<VkPipelineShaderStageCreateInfo>, "VkPipelineShaderStageCreateInfo"
        typeof<VkComputePipelineCreateInfo>, "VkComputePipelineCreateInfo"
        typeof<VkPipelineCreateFlags2CreateInfo>, "VkPipelineCreateFlags2CreateInfo"
        typeof<VkCommandPoolCreateInfo>, "VkCommandPoolCreateInfo"
        typeof<VkCommandBufferAllocateInfo>, "VkCommandBufferAllocateInfo"
        typeof<VkCommandBufferBeginInfo>, "VkCommandBufferBeginInfo"
        typeof<VkBufferCopy>, "VkBufferCopy"
        typeof<VkMemoryBarrier2>, "VkMemoryBarrier2"
        typeof<VkDependencyInfo>, "VkDependencyInfo"
        typeof<VkSemaphoreTypeCreateInfo>, "VkSemaphoreTypeCreateInfo"
        typeof<VkSemaphoreCreateInfo>, "VkSemaphoreCreateInfo"
        typeof<VkSemaphoreSubmitInfo>, "VkSemaphoreSubmitInfo"
        typeof<VkCommandBufferSubmitInfo>, "VkCommandBufferSubmitInfo"
        typeof<VkSubmitInfo2>, "VkSubmitInfo2"
        typeof<VkSemaphoreWaitInfo>, "VkSemaphoreWaitInfo"
        typeof<VkQueryPoolCreateInfo>, "VkQueryPoolCreateInfo"
        typeof<VkDeviceAddressRangeEXT>, "VkDeviceAddressRangeEXT"
        typeof<VkHostAddressRangeEXT>, "VkHostAddressRangeEXT"
        typeof<VkHostAddressRangeConstEXT>, "VkHostAddressRangeConstEXT"
        typeof<VkResourceDescriptorDataEXT>, "VkResourceDescriptorDataEXT"
        typeof<VkResourceDescriptorInfoEXT>, "VkResourceDescriptorInfoEXT"
        typeof<VkBindHeapInfoEXT>, "VkBindHeapInfoEXT"
        typeof<VkPushDataInfoEXT>, "VkPushDataInfoEXT"
        typeof<VkDescriptorMappingSourceConstantOffsetEXT>, "VkDescriptorMappingSourceConstantOffsetEXT"
        typeof<VkDescriptorMappingSourceDataEXT>, "VkDescriptorMappingSourceDataEXT"
        typeof<VkDescriptorSetAndBindingMappingEXT>, "VkDescriptorSetAndBindingMappingEXT"
        typeof<VkShaderDescriptorSetAndBindingMappingInfoEXT>, "VkShaderDescriptorSetAndBindingMappingInfoEXT"
    ]

/// Runs Layout.check over every hand-written struct; empty when all agree.
let layoutMismatches () =
    [
        if not Environment.Is64BitProcess then
            "not a 64-bit process"
        yield! Layout.selfTest ()
        for (t, name) in checkedStructs do
            yield! Layout.check t name
    ]

// ---- PFNs ----
// Every pointer argument is a nativeint to memory the caller keeps alive (a Scratch block or a pinned local).

type W = UnmanagedFunctionPointerAttribute

[<W(CallingConvention.Winapi)>]
type PFN_vkGetPhysicalDeviceQueueFamilyProperties = delegate of nativeint * nativeint * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkGetPhysicalDeviceMemoryProperties = delegate of nativeint * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkGetPhysicalDeviceDescriptorSizeEXT = delegate of nativeint * uint32 -> uint64

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateDevice = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroyDevice = delegate of nativeint * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkGetDeviceQueue = delegate of nativeint * uint32 * uint32 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkDeviceWaitIdle = delegate of nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateBuffer = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroyBuffer = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkGetBufferMemoryRequirements = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkAllocateMemory = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkFreeMemory = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkBindBufferMemory = delegate of nativeint * uint64 * uint64 * uint64 -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkMapMemory = delegate of nativeint * uint64 * uint64 * uint64 * uint32 * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkUnmapMemory = delegate of nativeint * uint64 -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkInvalidateMappedMemoryRanges = delegate of nativeint * uint32 * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkGetBufferDeviceAddress = delegate of nativeint * nativeint -> uint64

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateShaderModule = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroyShaderModule = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateComputePipelines =
    delegate of nativeint * uint64 * uint32 * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroyPipeline = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateCommandPool = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroyCommandPool = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkResetCommandPool = delegate of nativeint * uint64 * uint32 -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkAllocateCommandBuffers = delegate of nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkBeginCommandBuffer = delegate of nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkEndCommandBuffer = delegate of nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdBindPipeline = delegate of nativeint * uint32 * uint64 -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdDispatch = delegate of nativeint * uint32 * uint32 * uint32 -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdCopyBuffer = delegate of nativeint * uint64 * uint64 * uint32 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdPipelineBarrier2 = delegate of nativeint * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdResetQueryPool = delegate of nativeint * uint64 * uint32 * uint32 -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdWriteTimestamp2 = delegate of nativeint * uint64 * uint64 * uint32 -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateQueryPool = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroyQueryPool = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkGetQueryPoolResults =
    delegate of nativeint * uint64 * uint32 * uint32 * unativeint * nativeint * uint64 * uint32 -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkCreateSemaphore = delegate of nativeint * nativeint * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkDestroySemaphore = delegate of nativeint * uint64 * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkQueueSubmit2 = delegate of nativeint * uint32 * nativeint * uint64 -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkWaitSemaphores = delegate of nativeint * nativeint * uint64 -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkWriteResourceDescriptorsEXT = delegate of nativeint * uint32 * nativeint * nativeint -> VkResult

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdBindResourceHeapEXT = delegate of nativeint * nativeint -> unit

[<W(CallingConvention.Winapi)>]
type PFN_vkCmdPushDataEXT = delegate of nativeint * nativeint -> unit

let ok (call: string) (r: VkResult) =
    if r <> VkResult.VK_SUCCESS then
        failwithf "%s failed: %A (%d)" call r (int r)

/// Device-level commands, loaded with vkGetDeviceProcAddr once the device exists.
type DeviceFns(instance: Instance, device: nativeint) =
    let load (name: string) : 'T =
        let b = Text.Encoding.ASCII.GetBytes(name + "\000")
        let h = GCHandle.Alloc(b, GCHandleType.Pinned)

        let fp =
            try
                instance.GetDeviceProcAddr.Invoke(VkDevice device, Microsoft.FSharp.NativeInterop.NativePtr.ofNativeInt (h.AddrOfPinnedObject()))
            finally
                h.Free()

        if fp = 0n then
            failwithf "vkGetDeviceProcAddr returned NULL for %s" name

        Marshal.GetDelegateForFunctionPointer<'T> fp

    member val DestroyDevice: PFN_vkDestroyDevice = load "vkDestroyDevice"
    member val GetDeviceQueue: PFN_vkGetDeviceQueue = load "vkGetDeviceQueue"
    member val DeviceWaitIdle: PFN_vkDeviceWaitIdle = load "vkDeviceWaitIdle"
    member val CreateBuffer: PFN_vkCreateBuffer = load "vkCreateBuffer"
    member val DestroyBuffer: PFN_vkDestroyBuffer = load "vkDestroyBuffer"
    member val GetBufferMemoryRequirements: PFN_vkGetBufferMemoryRequirements = load "vkGetBufferMemoryRequirements"
    member val AllocateMemory: PFN_vkAllocateMemory = load "vkAllocateMemory"
    member val FreeMemory: PFN_vkFreeMemory = load "vkFreeMemory"
    member val BindBufferMemory: PFN_vkBindBufferMemory = load "vkBindBufferMemory"
    member val MapMemory: PFN_vkMapMemory = load "vkMapMemory"
    member val UnmapMemory: PFN_vkUnmapMemory = load "vkUnmapMemory"
    member val InvalidateMappedMemoryRanges: PFN_vkInvalidateMappedMemoryRanges = load "vkInvalidateMappedMemoryRanges"
    member val GetBufferDeviceAddress: PFN_vkGetBufferDeviceAddress = load "vkGetBufferDeviceAddress"
    member val CreateShaderModule: PFN_vkCreateShaderModule = load "vkCreateShaderModule"
    member val DestroyShaderModule: PFN_vkDestroyShaderModule = load "vkDestroyShaderModule"
    member val CreateComputePipelines: PFN_vkCreateComputePipelines = load "vkCreateComputePipelines"
    member val DestroyPipeline: PFN_vkDestroyPipeline = load "vkDestroyPipeline"
    member val CreateCommandPool: PFN_vkCreateCommandPool = load "vkCreateCommandPool"
    member val DestroyCommandPool: PFN_vkDestroyCommandPool = load "vkDestroyCommandPool"
    member val ResetCommandPool: PFN_vkResetCommandPool = load "vkResetCommandPool"
    member val AllocateCommandBuffers: PFN_vkAllocateCommandBuffers = load "vkAllocateCommandBuffers"
    member val BeginCommandBuffer: PFN_vkBeginCommandBuffer = load "vkBeginCommandBuffer"
    member val EndCommandBuffer: PFN_vkEndCommandBuffer = load "vkEndCommandBuffer"
    member val CmdBindPipeline: PFN_vkCmdBindPipeline = load "vkCmdBindPipeline"
    member val CmdDispatch: PFN_vkCmdDispatch = load "vkCmdDispatch"
    member val CmdCopyBuffer: PFN_vkCmdCopyBuffer = load "vkCmdCopyBuffer"
    member val CmdPipelineBarrier2: PFN_vkCmdPipelineBarrier2 = load "vkCmdPipelineBarrier2"
    member val CmdResetQueryPool: PFN_vkCmdResetQueryPool = load "vkCmdResetQueryPool"
    member val CmdWriteTimestamp2: PFN_vkCmdWriteTimestamp2 = load "vkCmdWriteTimestamp2"
    member val CreateQueryPool: PFN_vkCreateQueryPool = load "vkCreateQueryPool"
    member val DestroyQueryPool: PFN_vkDestroyQueryPool = load "vkDestroyQueryPool"
    member val GetQueryPoolResults: PFN_vkGetQueryPoolResults = load "vkGetQueryPoolResults"
    member val CreateSemaphore: PFN_vkCreateSemaphore = load "vkCreateSemaphore"
    member val DestroySemaphore: PFN_vkDestroySemaphore = load "vkDestroySemaphore"
    member val QueueSubmit2: PFN_vkQueueSubmit2 = load "vkQueueSubmit2"
    member val WaitSemaphores: PFN_vkWaitSemaphores = load "vkWaitSemaphores"
    member val WriteResourceDescriptorsEXT: PFN_vkWriteResourceDescriptorsEXT = load "vkWriteResourceDescriptorsEXT"
    member val CmdBindResourceHeapEXT: PFN_vkCmdBindResourceHeapEXT = load "vkCmdBindResourceHeapEXT"
    member val CmdPushDataEXT: PFN_vkCmdPushDataEXT = load "vkCmdPushDataEXT"
