namespace Lrm.Gpu.D3D12.Interop

/// Builders for the descriptions a compute backend fills most, with the values d3dx12.h would use. Pure: they only
/// build structs, so the cpu tier checks their bytes against known answers.
[<RequireQualifiedAccess>]
module Desc =
    /// D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES.
    [<Literal>]
    let AllSubresources = 0xFFFFFFFFu

    /// A command queue of `kind` with normal priority and no flags.
    let commandQueue (kind: D3D12_COMMAND_LIST_TYPE) =
        D3D12_COMMAND_QUEUE_DESC(Type = kind, Priority = 0, Flags = D3D12_COMMAND_QUEUE_FLAGS.NONE, NodeMask = 0u)

    /// Heap properties for a committed resource on a built-in heap type (single adapter: node masks 0).
    let heapProperties (kind: D3D12_HEAP_TYPE) =
        D3D12_HEAP_PROPERTIES(
            Type = kind,
            CPUPageProperty = D3D12_CPU_PAGE_PROPERTY.UNKNOWN,
            MemoryPoolPreference = D3D12_MEMORY_POOL.UNKNOWN,
            CreationNodeMask = 0u,
            VisibleNodeMask = 0u
        )

    /// A buffer of `bytes` bytes.
    let buffer (bytes: uint64) (flags: D3D12_RESOURCE_FLAGS) =
        D3D12_RESOURCE_DESC(
            Dimension = D3D12_RESOURCE_DIMENSION.BUFFER,
            Alignment = 0UL,
            Width = bytes,
            Height = 1u,
            DepthOrArraySize = 1us,
            MipLevels = 1us,
            Format = DXGI_FORMAT.UNKNOWN,
            SampleDesc = DXGI_SAMPLE_DESC(Count = 1u, Quality = 0u),
            Layout = D3D12_TEXTURE_LAYOUT.LAYOUT_ROW_MAJOR,
            Flags = flags
        )

    /// A descriptor heap of `count` descriptors of `kind`.
    let descriptorHeap (kind: D3D12_DESCRIPTOR_HEAP_TYPE) (count: uint32) (shaderVisible: bool) =
        D3D12_DESCRIPTOR_HEAP_DESC(
            Type = kind,
            NumDescriptors = count,
            Flags =
                (if shaderVisible then
                     D3D12_DESCRIPTOR_HEAP_FLAGS.SHADER_VISIBLE
                 else
                     D3D12_DESCRIPTOR_HEAP_FLAGS.NONE),
            NodeMask = 0u
        )

    /// A raw (ByteAddressBuffer) UAV over 32-bit elements [firstElement, firstElement + numElements) of a buffer:
    /// format R32_TYPELESS with the RAW flag, as the HeapLayoutAndTensorTable contract requires.
    let rawBufferUav (firstElement: uint64) (numElements: uint32) =
        D3D12_UNORDERED_ACCESS_VIEW_DESC(
            Format = DXGI_FORMAT.R32_TYPELESS,
            ViewDimension = D3D12_UAV_DIMENSION.BUFFER,
            Buffer =
                D3D12_BUFFER_UAV(
                    FirstElement = firstElement,
                    NumElements = numElements,
                    StructureByteStride = 0u,
                    CounterOffsetInBytes = 0UL,
                    Flags = D3D12_BUFFER_UAV_FLAGS.RAW
                )
        )

    /// A transition barrier on every subresource of `resource`.
    let transition<'R when 'R :> IUnknown>
        (resource: ComPtr<'R>)
        (before: D3D12_RESOURCE_STATES)
        (after: D3D12_RESOURCE_STATES)
        =
        D3D12_RESOURCE_BARRIER(
            Type = D3D12_RESOURCE_BARRIER_TYPE.TRANSITION,
            Flags = D3D12_RESOURCE_BARRIER_FLAGS.NONE,
            Transition =
                D3D12_RESOURCE_TRANSITION_BARRIER(
                    pResource = resource.Ptr,
                    Subresource = AllSubresources,
                    StateBefore = before,
                    StateAfter = after
                )
        )

    /// A UAV barrier on `resource`, or on every UAV access when `resource` is null.
    let uavBarrier<'R when 'R :> IUnknown> (resource: ComPtr<'R>) =
        D3D12_RESOURCE_BARRIER(
            Type = D3D12_RESOURCE_BARRIER_TYPE.UAV,
            Flags = D3D12_RESOURCE_BARRIER_FLAGS.NONE,
            UAV = D3D12_RESOURCE_UAV_BARRIER(pResource = resource.Ptr)
        )

    /// A compute pipeline over DXIL at `bytecode` (`length` bytes), which must stay pinned until creation returns.
    let computePipeline<'S when 'S :> IUnknown> (rootSignature: ComPtr<'S>) (bytecode: nativeint) (length: unativeint) =
        D3D12_COMPUTE_PIPELINE_STATE_DESC(
            pRootSignature = rootSignature.Ptr,
            CS = D3D12_SHADER_BYTECODE(pShaderBytecode = bytecode, BytecodeLength = length),
            NodeMask = 0u,
            CachedPSO = D3D12_CACHED_PIPELINE_STATE(pCachedBlob = 0n, CachedBlobSizeInBytes = 0un),
            Flags = D3D12_PIPELINE_STATE_FLAGS.NONE
        )

    /// A query heap of `count` queries of `kind`.
    let queryHeap (kind: D3D12_QUERY_HEAP_TYPE) (count: uint32) =
        D3D12_QUERY_HEAP_DESC(Type = kind, Count = count, NodeMask = 0u)
