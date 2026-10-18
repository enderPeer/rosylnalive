// spirv-fixture: InterlockedAdd on a groupshared uint: an integer atomic, which contracts v1 also forbid.
// kernel: scale_f32
// expect: atomics
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

groupshared uint gs_count;

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint previous;
    InterlockedAdd(gs_count, 1u, previous);
    GroupMemoryBarrierWithGroupSync();
    lrm_store_u32(lrm_tensor(lrm_args.y), tid.x, previous + gs_count);
}
