// spirv-fixture: the heap array at set 1, binding 0, as a header that drifted from the recipe would give.
// kernel: scale_f32
// expect: resource
#include "lrm_args.g.hlsli"

[[vk::binding(0, 1)]] RWByteAddressBuffer g_buf[] : register(u0, space1);
[[vk::push_constant]] ConstantBuffer<lrm_args_scale_f32> lrm_args : register(b0, space0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = (gid.y * lrm_args.lrm_groups_x + gid.x) * 256u + tid.x;
    if (i >= lrm_args.n)
        return;
    uint4 y = g_buf[0].Load4(lrm_args.y * 64u);
    g_buf[y.x].Store(y.y + i * 4u, asuint(lrm_args.a));
}
