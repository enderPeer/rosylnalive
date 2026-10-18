// spirv-fixture: a second buffer beside g_buf[], as a kernel that declared its own binding would give.
// kernel: scale_f32
// expect: resource
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

[[vk::binding(1, 0)]] RWByteAddressBuffer extra : register(u1, space0);

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    extra.Store(tid.x * 4u, 1u);
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, lrm_args.a);
}
