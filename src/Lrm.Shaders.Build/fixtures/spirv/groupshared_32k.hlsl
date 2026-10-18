// spirv-fixture: the positive control at the limit: exactly 32 KiB of groupshared memory in two variables.
// kernel: scale_f32
// expect: none
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

groupshared float4 gs_a[1024];
groupshared float4 gs_b[1024];

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    for (uint k = tid.x; k < 1024u; k += 256u)
    {
        gs_a[k] = float4(lrm_args.a, 0.0f, 0.0f, 0.0f);
        gs_b[k] = float4(0.0f, lrm_args.a, 0.0f, 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, gs_a[1023u - tid.x].x + gs_b[tid.x].y);
}
