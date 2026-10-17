// R1 spike: the binding convention under test (throwaway; R4a owns the real lrm.hlsli).
//
// - Arguments: one 128-byte ArgBlock of 32 dwords. D3D12 delivers it as 32 root constants at b0; Vulkan as push data
//   written by vkCmdPushDataEXT at offset 0. uint4 d[8] has a 16-byte stride under both the D3D cbuffer packing and
//   DXC's push-constant layout, so dword k is d[k >> 2][k & 3] on both backends. dword0 is lrm_groups_x.
// - Resources: LRM_BUF(slot) is descriptor `slot` of the one resource heap.
//     DXIL:               ResourceDescriptorHeap[slot]          (SM 6.6, CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)
//     SPIR-V, explicit:   g_buf[slot], [[vk::binding(0, 0)]]    (one runtime array at set 0 binding 0)
//     SPIR-V, emulated:   ResourceDescriptorHeap[slot] with -fvk-bind-resource-heap 0 0 (one aliased variable per use)
//   On Vulkan, set 0 binding 0 is mapped onto the heap with HEAP_WITH_CONSTANT_OFFSET, heapOffset 0 and
//   heapArrayStride = bufferDescriptorSize, so slot i is the descriptor at byte i * bufferDescriptorSize.
// - Indices are dynamically uniform (they come from push data or from one table row), so there is no
//   NonUniformResourceIndex.
// - Tensor table: slot 0, 64-byte rows (16 dwords): slot, byteOffset, dtype, rank, dims[4], strides[4], elemCount,
//   flags, debugId, 0. Token 0 is the null row.

struct LrmArgBlock
{
    uint4 d[8];
};

// [[vk::push_constant]] only on SPIR-V: DXIL ignores it, and -WX turns that warning into an error.
#if defined(__spirv__)
[[vk::push_constant]]
#endif
ConstantBuffer<LrmArgBlock> g_lrmArgs : register(b0, space0);

#define LRM_ARG(i) (g_lrmArgs.d[(i) >> 2][(i) & 3])

#if defined(__spirv__) && !defined(LRM_EMULATED_HEAP)
[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);
#define LRM_BUF(slot) g_buf[slot]
#else
#define LRM_BUF(slot) ResourceDescriptorHeap[slot]
#endif

struct LrmTensor
{
    uint slot;
    uint byteOffset;
    uint dtype;
    uint rank;
    uint4 dims;
    uint4 strides;
    uint elemCount;
    uint flags;
    uint debugId;
};

LrmTensor lrm_tensor(uint token)
{
    RWByteAddressBuffer table = LRM_BUF(0);
    uint base = token * 64u;
    uint4 a = table.Load4(base);
    uint4 c = table.Load4(base + 48u);
    LrmTensor t;
    t.slot = a.x;
    t.byteOffset = a.y;
    t.dtype = a.z;
    t.rank = a.w;
    t.dims = table.Load4(base + 16u);
    t.strides = table.Load4(base + 32u);
    t.elemCount = c.x;
    t.flags = c.y;
    t.debugId = c.z;
    return t;
}

// Linear thread index for a grid of lrm_groups_x * y groups of `threads` threads.
uint lrm_linear(uint3 gid, uint3 tid, uint threads)
{
    return (gid.y * LRM_ARG(0) + gid.x) * threads + tid.x;
}
