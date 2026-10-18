#r @"C:\Users\end\dev\rosylnalive\gpu\vulkan\app\VkDeviceCheck\bin\Debug\net10.0\Lrm.Gpu.Vulkan.dll"
#nowarn "9"
#nowarn "51"
open Lrm.Gpu.Vulkan
let loader = (VulkanLoader.TryOpen()).Value
let inst = loader.CreateInstance("probe", VkVersion.VK_API_VERSION_1_4)
let pd = inst.PhysicalDevices.[0]
let mutable props = VkPhysicalDeviceProperties()
inst.GetPhysicalDeviceProperties.Invoke(pd, &&props)
let l = props.limits
printfn "maxStorageBufferRange=%d (0x%x)" l.maxStorageBufferRange l.maxStorageBufferRange
printfn "maxPushConstantsSize=%d maxMemoryAllocationCount=%d" l.maxPushConstantsSize l.maxMemoryAllocationCount
printfn "maxComputeSharedMemorySize=%d maxComputeWorkGroupInvocations=%d" l.maxComputeSharedMemorySize l.maxComputeWorkGroupInvocations
printfn "maxComputeWorkGroupCount=%A" l.maxComputeWorkGroupCount
printfn "minStorageBufferOffsetAlignment=%d timestampPeriod=%g ns" l.minStorageBufferOffsetAlignment l.timestampPeriod
(inst :> System.IDisposable).Dispose()
