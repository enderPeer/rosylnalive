/// Compiled at test time with tail calls on (--tailcalls+, optimized), then scanned for IL tail. prefixes
/// (CpuTests). Each function makes one call in tail position with a stack address. Not part of the test build.
module Lrm.Gpu.Vulkan.Interop.TailCallConsumer

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open Lrm.Gpu.Vulkan.Interop

/// A wrapper taking inref: F# never tail-calls a call with a byref argument.
let pushData (d: VkDeviceCommands) (cmd: VkCommandBuffer) =
    let mutable info = VkPushDataInfoEXT.Create()
    d.CmdPushDataEXT(cmd, &info)

/// A wrapper taking inref, the barrier of every dispatch.
let barrier (d: VkDeviceCommands) (cmd: VkCommandBuffer) =
    let mutable mb = VkMemoryBarrier2.Create()
    let mutable dep = VkDependencyInfo.Create()
    dep.memoryBarrierCount <- 1u
    dep.pMemoryBarriers <- NativePtr.toNativeInt &&mb
    d.CmdPipelineBarrier2(cmd, &dep)

/// A wrapper taking inref and byref.
let createBuffer (d: VkDeviceCommands) (device: VkDevice) =
    let mutable ci = VkBufferCreateInfo.Create()
    let mutable b = VkBuffer.Null
    d.CreateBuffer(device, &ci, &b)

/// A wrapper taking byref only.
let queue (d: VkDeviceCommands) (device: VkDevice) =
    let mutable q = VkQueue.Null
    d.GetDeviceQueue(device, 0u, 0u, &q)

/// Control: the raw delegate with a stack address. This is the hazard Tailcalls=false covers; it must carry tail.
/// here, which proves tail calls were on.
let controlRawDelegate (d: VkDeviceCommands) (cmd: VkCommandBuffer) =
    let mutable info = VkPushDataInfoEXT.Create()
    d.vkCmdPushDataEXT.Invoke(cmd, NativePtr.toNativeInt &&info)

/// Control: a span wrapper with a span over a stack local. A span is a struct, not a byref, so F# may tail-call it;
/// the test reports whether it did.
let controlSpan (d: VkDeviceCommands) (cmd: VkCommandBuffer) (src: VkBuffer) (dst: VkBuffer) =
    let mutable region = VkBufferCopy()
    d.CmdCopyBuffer(cmd, src, dst, MemoryMarshal.CreateReadOnlySpan(&region, 1))
