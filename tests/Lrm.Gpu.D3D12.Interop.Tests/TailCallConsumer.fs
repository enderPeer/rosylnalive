/// A consumer of the interop, compiled by the cpu tier with --tailcalls+ (the Release default) and scanned for IL
/// tail. prefixes (CpuTests, "a consumer built with tail calls gets no tail. prefix at wrapper calls"). It is an
/// embedded resource of the test exe, not part of its build.
///
/// Every function below ends with a wrapper call that receives a stack address: a span over a local, a single
/// element by reference, or NativePtr.toNativeInt &&local through a raw form. None of those calls may carry the
/// prefix (ADR R3-0001). `controlDelegate` makes the same call straight through a cached Fn delegate and must carry
/// it, which shows the compilation had tail calls on, and that the low-level Vtbl.get path is not protected.
module Lrm.Gpu.D3D12.Interop.TailCallConsumer

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open Lrm.Gpu.D3D12.Interop
open Microsoft.FSharp.NativeInterop

let controlDelegate (l: ComPtr<ID3D12GraphicsCommandList>) (r: ComPtr<ID3D12Resource>) =
    let mutable b = Desc.uavBarrier r

    (Vtbl.get<Fn.ID3D12GraphicsCommandList.ResourceBarrier> l.Ptr Slots.ID3D12GraphicsCommandList.ResourceBarrier)
        .Invoke(l.Ptr, 1u, NativePtr.toNativeInt &&b)

let barrierSpan (l: ComPtr<ID3D12GraphicsCommandList>) (r: ComPtr<ID3D12Resource>) =
    let b =
        Desc.transition r D3D12_RESOURCE_STATES.COMMON D3D12_RESOURCE_STATES.COPY_DEST

    l.ResourceBarrier(ReadOnlySpan<D3D12_RESOURCE_BARRIER>(&b))

let barrierOne (l: ComPtr<ID3D12GraphicsCommandList>) (r: ComPtr<ID3D12Resource>) =
    let b = Desc.uavBarrier r
    l.ResourceBarrier &b

let barrierRaw (l: ComPtr<ID3D12GraphicsCommandList>) (r: ComPtr<ID3D12Resource>) =
    let mutable b = Desc.uavBarrier r
    l.ResourceBarrierRaw(1u, NativePtr.toNativeInt &&b)

let barrierGeneric<'L when 'L :> ID3D12GraphicsCommandList> (l: ComPtr<'L>) (r: ComPtr<ID3D12Resource>) =
    let b = Desc.uavBarrier r
    l.ResourceBarrier(ReadOnlySpan<D3D12_RESOURCE_BARRIER>(&b))

let constants (l: ComPtr<ID3D12GraphicsCommandList>) (value: uint32) =
    let c = value
    l.SetComputeRoot32BitConstants(0u, ReadOnlySpan<uint32>(&c), 0u)

let constantsRaw (l: ComPtr<ID3D12GraphicsCommandList>) (value: uint32) =
    let mutable c = value
    l.SetComputeRoot32BitConstantsRaw(0u, 1u, NativePtr.toNativeInt &&c, 0u)

let execute (q: ComPtr<ID3D12CommandQueue>) (l: ComPtr<ID3D12GraphicsCommandList>) =
    q.ExecuteCommandLists(ReadOnlySpan<ComPtr<ID3D12GraphicsCommandList>>(&l))

let executeOne (q: ComPtr<ID3D12CommandQueue>) (l: ComPtr<ID3D12GraphicsCommandList>) = q.ExecuteCommandLists l

let executeRaw (q: ComPtr<ID3D12CommandQueue>) (l: ComPtr<ID3D12GraphicsCommandList>) =
    let mutable p = l.Ptr
    q.ExecuteCommandListsRaw(1u, NativePtr.toNativeInt &&p)

let heaps (l: ComPtr<ID3D12GraphicsCommandList>) (h: ComPtr<ID3D12DescriptorHeap>) =
    l.SetDescriptorHeaps(ReadOnlySpan<ComPtr<ID3D12DescriptorHeap>>(&h))

let clear
    (l: ComPtr<ID3D12GraphicsCommandList>)
    (gpu: D3D12_GPU_DESCRIPTOR_HANDLE)
    (cpu: D3D12_CPU_DESCRIPTOR_HANDLE)
    (r: ComPtr<ID3D12Resource>)
    =
    let v = Desc.rawBufferUav 0UL 0u

    let values =
        MemoryMarshal.Cast<D3D12_UNORDERED_ACCESS_VIEW_DESC, uint32>(MemoryMarshal.CreateReadOnlySpan(&v, 1))

    l.ClearUnorderedAccessViewUint(gpu, cpu, r, values.Slice(0, 4), 0u, 0n)

let uav (dev: ComPtr<ID3D12Device>) (r: ComPtr<ID3D12Resource>) (at: D3D12_CPU_DESCRIPTOR_HANDLE) =
    let mutable d = Desc.rawBufferUav 0UL 16u
    dev.CreateUnorderedAccessView(r, ComPtr<ID3D12Resource>.Null, NativePtr.toNativeInt &&d, at)

let queue (dev: ComPtr<ID3D12Device>) : ComPtr<ID3D12CommandQueue> =
    let d = Desc.commandQueue D3D12_COMMAND_LIST_TYPE.COMPUTE
    dev.CreateCommandQueue(&d)

let allocation (dev: ComPtr<ID3D12Device>) =
    let d = Desc.buffer 4096UL D3D12_RESOURCE_FLAGS.NONE
    dev.GetResourceAllocationInfo(0u, &d)

let unmap (r: ComPtr<ID3D12Resource>) =
    let mutable written = D3D12_RANGE(Begin = 0un, End = 16un)
    r.Unmap(0u, NativePtr.toNativeInt &&written)

let map (r: ComPtr<ID3D12Resource>) =
    let mutable range = D3D12_RANGE(Begin = 0un, End = 0un)
    r.Map(0u, NativePtr.toNativeInt &&range)
