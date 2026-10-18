// lrm.hlsli: the one header every LRM kernel includes first (ShaderBuildAndSourceConvention contract v1.1,
// docs/contracts/shader-build.md). It gives a kernel its arguments (LRM_ARGS), the bindless heap (LRM_BUF, lrm_buf)
// and the tensor table (lrm_tensor), the same way on both targets:
//   - DXIL (cs_6_6): ResourceDescriptorHeap, with 32 root constants at b0 (rootsig/rootsig.hlsl).
//   - SPIR-V (vulkan1.3): one explicit runtime array at set 0, binding 0, and a push-constant block at offset 0,
//     bound to the resource heap by the Vulkan recipe (VulkanHeapRecipe, docs/decisions/R1-heap-recipe.md).
//
// Every LRM_BUF index is dynamically uniform: a slot comes from a tensor row that a token from LRM_ARGS selects, so
// there is no NonUniformResourceIndex anywhere.
#ifndef LRM_HLSLI
#define LRM_HLSLI

#include "lrm_args.g.hlsli"

// ---- heap and table constants (HeapLayoutAndTensorTable contract) ----------------------------------------------
#define LRM_TABLE_SLOT 0
#define LRM_ROW_BYTES 64
#define LRM_NULL_TOKEN 0

// DType codes (LrmCore contract).
#define LRM_DTYPE_F32 1
#define LRM_DTYPE_BF16 2
#define LRM_DTYPE_F16 3
#define LRM_DTYPE_I32 4
#define LRM_DTYPE_U32 5
#define LRM_DTYPE_U8 6
#define LRM_DTYPE_F64 7
#define LRM_DTYPE_I64 8

// Tensor-row flags.
#define LRM_FLAG_CONTIGUOUS 1u
#define LRM_FLAG_READ_ONLY 2u
#define LRM_FLAG_VIEW 4u

// The f32 quiet NaN that kernels write for undefined results (for example an out-of-range embedding id).
#define LRM_F32_QUIET_NAN_BITS 0x7FC00000u

// ---- bindings (contracts-v1.1, final) ----------------------------------------------------------------------------
// The heap. LRM_BUF(slot) is descriptor `slot` of the one resource heap:
//   - DXIL: ResourceDescriptorHeap[slot] (CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED, rootsig/rootsig.hlsl).
//   - SPIR-V: g_buf[slot], the only resource variable: a runtime array of raw buffers at set 0, binding 0
//     (SpirvBinding in Lrm.Gpu.Core). The Vulkan backend maps set 0, binding 0 onto the heap with
//     VK_EXT_descriptor_heap: HEAP_WITH_CONSTANT_OFFSET, heapOffset 0, heapArrayStride = bufferDescriptorSize, so
//     g_buf[slot] is the descriptor at byte slot * bufferDescriptorSize. No emulation flag decides this binding.
// The arguments. lrm_args is the kernel's generated block (LRM_ARGS(kernel)): the 32 root constants at b0 on DXIL,
// push data at offset 0 on SPIR-V (vkCmdPushDataEXT pushes the whole 128-byte ArgBlock). Argument word i sits at
// byte 4 * i in both layouts; the SPIR-V inspector checks every kernel's offsets, names and types against the
// catalog. [[vk::push_constant]] appears only under __spirv__: DXIL ignores it, and -WX makes that warning an error.
#ifdef __spirv__
[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);
#define LRM_BUF(slot) g_buf[(slot)]
#define LRM_ARGS(kernel) [[vk::push_constant]] ConstantBuffer<lrm_args_##kernel> lrm_args : register(b0, space0);
#else
#define LRM_BUF(slot) ResourceDescriptorHeap[(slot)]
#define LRM_ARGS(kernel) ConstantBuffer<lrm_args_##kernel> lrm_args : register(b0, space0);
#endif

// [numthreads] of a kernel, from the catalog.
#define LRM_NUMTHREADS(kernel) [numthreads(LRM_THREADS_##kernel)]

// The logical group of a linear kernel: gid.y * lrm_groups_x + gid.x. A grid above 65,535 groups is folded into
// rows, so a linear kernel always addresses its work through this and returns when the group has none.
#define LRM_GROUP_LINEAR(gid) ((gid).y * lrm_args.lrm_groups_x + (gid).x)

// The buffer behind a slot.
RWByteAddressBuffer lrm_buf(uint slot)
{
    RWByteAddressBuffer b = LRM_BUF(slot);
    return b;
}

// ---- the tensor table ----------------------------------------------------------------------------------------
// One decoded 64-byte row (16 u32 words, in row order).
struct lrm_tensor_t
{
    uint slot;      // word 0: page slot, 1..63
    uint offset;    // word 1: byte offset in the page (a multiple of 4)
    uint dtype;     // word 2: LRM_DTYPE_*
    uint rank;      // word 3: 1..4
    uint4 dims;     // words 4..7: outermost first, unused dims 1
    uint4 strides;  // words 8..11: in elements, unused strides 0
    uint count;     // word 12: element count
    uint flags;     // word 13: LRM_FLAG_*
    uint debug_id;  // word 14
    uint reserved;  // word 15: 0
};

// The row of a token. Token 0 (null) and unused tokens give all zeros.
lrm_tensor_t lrm_tensor(uint token)
{
    RWByteAddressBuffer table = LRM_BUF(LRM_TABLE_SLOT);
    uint base = token * LRM_ROW_BYTES;
    uint4 w0 = table.Load4(base);
    uint4 w1 = table.Load4(base + 16u);
    uint4 w2 = table.Load4(base + 32u);
    uint4 w3 = table.Load4(base + 48u);
    lrm_tensor_t t;
    t.slot = w0.x;
    t.offset = w0.y;
    t.dtype = w0.z;
    t.rank = w0.w;
    t.dims = w1;
    t.strides = w2;
    t.count = w3.x;
    t.flags = w3.y;
    t.debug_id = w3.z;
    t.reserved = w3.w;
    return t;
}

// ---- element access (index in elements from the tensor's start) ----------------------------------------------
// Byte addresses stay below 2^31: pages are at most 1 GiB and offsets lie inside the page.

float lrm_load_f32(lrm_tensor_t t, uint i)
{
    return asfloat(lrm_buf(t.slot).Load(t.offset + i * 4u));
}

void lrm_store_f32(lrm_tensor_t t, uint i, float v)
{
    lrm_buf(t.slot).Store(t.offset + i * 4u, asuint(v));
}

uint lrm_load_u32(lrm_tensor_t t, uint i)
{
    return lrm_buf(t.slot).Load(t.offset + i * 4u);
}

void lrm_store_u32(lrm_tensor_t t, uint i, uint v)
{
    lrm_buf(t.slot).Store(t.offset + i * 4u, v);
}

int lrm_load_i32(lrm_tensor_t t, uint i)
{
    return asint(lrm_buf(t.slot).Load(t.offset + i * 4u));
}

void lrm_store_i32(lrm_tensor_t t, uint i, int v)
{
    lrm_buf(t.slot).Store(t.offset + i * 4u, asuint(v));
}

// Four consecutive f32 elements i..i+3.
float4 lrm_load4_f32(lrm_tensor_t t, uint i)
{
    return asfloat(lrm_buf(t.slot).Load4(t.offset + i * 4u));
}

void lrm_store4_f32(lrm_tensor_t t, uint i, float4 v)
{
    lrm_buf(t.slot).Store4(t.offset + i * 4u, asuint(v));
}

// bf16 bits widened exactly to f32 (the bits shifted into the high half), without 16-bit types.
float lrm_bf16_widen(uint bits16)
{
    return asfloat(bits16 << 16);
}

// The 16 bits of bf16 element i: the u32 word holding it, then its half (offsets are multiples of 4, so odd i is
// the high half).
uint lrm_load_bf16_bits(lrm_tensor_t t, uint i)
{
    uint w = lrm_buf(t.slot).Load(t.offset + ((i * 2u) & ~3u));
    return (i & 1u) != 0u ? (w >> 16) : (w & 0xFFFFu);
}

// bf16 element i widened to f32.
float lrm_load_bf16(lrm_tensor_t t, uint i)
{
    return lrm_bf16_widen(lrm_load_bf16_bits(t, i));
}

// bf16 elements 2j and 2j+1 as a float2.
float2 lrm_load2_bf16(lrm_tensor_t t, uint j)
{
    uint w = lrm_buf(t.slot).Load(t.offset + j * 4u);
    return float2(asfloat(w << 16), asfloat(w & 0xFFFF0000u));
}

#endif // LRM_HLSLI
