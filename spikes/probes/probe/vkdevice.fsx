// Read-only probe: queue families, memory heaps, descriptor sizes, and a logical device with descriptorHeap enabled;
// then checks that the VK_EXT_descriptor_heap device commands resolve. Creates and destroys the device only.
#r @"C:\Users\end\dev\rosylnalive\gpu\vulkan\app\VkDeviceCheck\bin\Debug\net10.0\Lrm.Gpu.Vulkan.dll"
#nowarn "9"
#nowarn "51"
open System
open System.Runtime.InteropServices
open System.Text
open Microsoft.FSharp.NativeInterop
open Lrm.Gpu.Vulkan

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_QF = delegate of nativeint * nativeptr<uint32> * nativeint -> unit
[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_Mem = delegate of nativeint * nativeint -> unit
[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_DescSize = delegate of nativeint * int -> uint64
[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_CreateDevice = delegate of nativeint * nativeint * nativeint * nativeptr<nativeint> -> int
[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_DestroyDevice = delegate of nativeint * nativeint -> unit
[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_GetDeviceProcAddr = delegate of nativeint * nativeptr<byte> -> nativeint

let mem (n: int) = let p = Marshal.AllocHGlobal n in (for i in 0 .. n - 1 do Marshal.WriteByte(p, i, 0uy)); p
let loader = (VulkanLoader.TryOpen()).Value
let inst = loader.CreateInstance("probe", VkVersion.VK_API_VERSION_1_4)
let pd = inst.PhysicalDevices.[0].Handle
let iproc<'T when 'T :> Delegate> name = loader.LoadInstanceProcOrFail<'T>(inst.Handle, name)

// queue families
let qf = iproc<PFN_QF> "vkGetPhysicalDeviceQueueFamilyProperties"
let mutable nq = 0u
qf.Invoke(pd, &&nq, 0n)
let qp = mem (int nq * 24)
qf.Invoke(pd, &&nq, qp)
let flagStr (f: int) =
    [ if f &&& 1 <> 0 then "GRAPHICS"
      if f &&& 2 <> 0 then "COMPUTE"
      if f &&& 4 <> 0 then "TRANSFER"
      if f &&& 8 <> 0 then "SPARSE"
      if f &&& 0x20 <> 0 then "VIDEO_DECODE"
      if f &&& 0x40 <> 0 then "VIDEO_ENCODE"
      if f &&& 0x100 <> 0 then "OPTICAL_FLOW" ] |> String.concat "|"
for i in 0 .. int nq - 1 do
    let f = Marshal.ReadInt32(qp, i * 24)
    printfn "queue family %d: %-40s count=%d timestampBits=%d" i (flagStr f) (Marshal.ReadInt32(qp, i * 24 + 4)) (Marshal.ReadInt32(qp, i * 24 + 8))

// memory
let mp = mem 520
(iproc<PFN_Mem> "vkGetPhysicalDeviceMemoryProperties").Invoke(pd, mp)
let nt = Marshal.ReadInt32(mp, 0)
let nh = Marshal.ReadInt32(mp, 260)
for h in 0 .. nh - 1 do
    printfn "memory heap %d: %d MiB flags=0x%x" h (Marshal.ReadInt64(mp, 264 + h * 16) / 1048576L) (Marshal.ReadInt32(mp, 264 + h * 16 + 8))
for t in 0 .. nt - 1 do
    let fl = Marshal.ReadInt32(mp, 4 + t * 8)
    let s =
        [ if fl &&& 1 <> 0 then "DEVICE_LOCAL"
          if fl &&& 2 <> 0 then "HOST_VISIBLE"
          if fl &&& 4 <> 0 then "HOST_COHERENT"
          if fl &&& 8 <> 0 then "HOST_CACHED" ] |> String.concat "|"
    if fl <> 0 then printfn "memory type %d: heap %d %s" t (Marshal.ReadInt32(mp, 4 + t * 8 + 4)) s

// per-type descriptor sizes (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER=6, STORAGE_BUFFER=7, STORAGE_IMAGE=3, SAMPLED_IMAGE=2)
match loader.LoadInstanceProc<PFN_DescSize>(inst.Handle, "vkGetPhysicalDeviceDescriptorSizeEXT") with
| ValueSome f -> printfn "vkGetPhysicalDeviceDescriptorSizeEXT: UBO=%d SSBO=%d storageImage=%d sampledImage=%d" (f.Invoke(pd, 6)) (f.Invoke(pd, 7)) (f.Invoke(pd, 3)) (f.Invoke(pd, 2))
| ValueNone -> printfn "vkGetPhysicalDeviceDescriptorSizeEXT: not exported"

// logical device: Vulkan 1.2/1.3 features + descriptorHeap + untyped pointers + cooperative matrix
let mutable v12 = VkPhysicalDeviceVulkan12Features()
v12.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
v12.bufferDeviceAddress <- 1u
v12.timelineSemaphore <- 1u
v12.shaderFloat16 <- 1u
v12.shaderInt8 <- 1u
v12.storageBuffer8BitAccess <- 1u
v12.vulkanMemoryModel <- 1u
v12.vulkanMemoryModelDeviceScope <- 1u
v12.scalarBlockLayout <- 1u
let mutable v13 = VkPhysicalDeviceVulkan13Features()
v13.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
v13.synchronization2 <- 1u
v13.subgroupSizeControl <- 1u
v13.computeFullSubgroups <- 1u
v13.maintenance4 <- 1u
let mutable heap = VkPhysicalDeviceDescriptorHeapFeaturesEXT()
heap.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT
heap.descriptorHeap <- 1u
let mutable untyped = VkPhysicalDeviceShaderUntypedPointersFeaturesKHR()
untyped.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR
untyped.shaderUntypedPointers <- 1u
let mutable coop = VkPhysicalDeviceCooperativeMatrixFeaturesKHR()
coop.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
coop.cooperativeMatrix <- 1u
v12.pNext <- NativePtr.toNativeInt &&v13
v13.pNext <- NativePtr.toNativeInt &&heap
heap.pNext <- NativePtr.toNativeInt &&untyped
untyped.pNext <- NativePtr.toNativeInt &&coop

let exts = [ "VK_EXT_descriptor_heap"; "VK_KHR_shader_untyped_pointers"; "VK_KHR_cooperative_matrix" ]
let extPtrs = exts |> List.map (fun e -> Marshal.StringToHGlobalAnsi e) |> Array.ofList
let pExts = mem (8 * extPtrs.Length)
extPtrs |> Array.iteri (fun i p -> Marshal.WriteIntPtr(pExts, i * 8, p))

let prio = mem 4
Marshal.WriteInt32(prio, 0, BitConverter.SingleToInt32Bits 1.0f)
let qci = mem 40
Marshal.WriteInt32(qci, 0, 2) // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
Marshal.WriteInt32(qci, 20, 0) // family 0
Marshal.WriteInt32(qci, 24, 1)
Marshal.WriteIntPtr(qci, 32, prio)
let dci = mem 72
Marshal.WriteInt32(dci, 0, 3) // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
Marshal.WriteIntPtr(dci, 8, NativePtr.toNativeInt &&v12)
Marshal.WriteInt32(dci, 20, 1)
Marshal.WriteIntPtr(dci, 24, qci)
Marshal.WriteInt32(dci, 48, extPtrs.Length)
Marshal.WriteIntPtr(dci, 56, pExts)
let mutable device = 0n
let r = (iproc<PFN_CreateDevice> "vkCreateDevice").Invoke(pd, dci, 0n, &&device)
printfn "vkCreateDevice(descriptorHeap+untypedPointers+cooperativeMatrix+BDA+timeline+sync2) = %d" r
if r = 0 then
    let gdpa = iproc<PFN_GetDeviceProcAddr> "vkGetDeviceProcAddr"
    for name in [ "vkWriteResourceDescriptorsEXT"; "vkWriteSamplerDescriptorsEXT"; "vkCmdBindResourceHeapEXT"; "vkCmdBindSamplerHeapEXT"
                  "vkCmdPushDataEXT"; "vkGetImageOpaqueCaptureDataEXT"; "vkCreateComputePipelines"; "vkGetBufferDeviceAddress"
                  "vkQueueSubmit2"; "vkWaitSemaphores"; "vkCmdPipelineBarrier2"; "vkCmdPushConstants"; "vkCmdBindDescriptorSets" ] do
        let b = Encoding.ASCII.GetBytes(name + "\000")
        use p = fixed b
        printfn "  %-34s %s" name (if gdpa.Invoke(device, p) <> 0n then "resolved" else "NULL")
    (iproc<PFN_DestroyDevice> "vkDestroyDevice").Invoke(device, 0n)
(inst :> IDisposable).Dispose()
