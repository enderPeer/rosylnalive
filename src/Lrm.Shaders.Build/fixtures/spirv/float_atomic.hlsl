// spirv-fixture: InterlockedExchange on a groupshared float becomes OpAtomicExchange %float, a float atomic that
// needs no extra capability, so only the float-atomics rule can see it.
// kernel: scale_f32
// expect: float-atomics
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

groupshared float gs_value[64];

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    float previous;
    InterlockedExchange(gs_value[tid.x & 63u], lrm_args.a, previous);
    GroupMemoryBarrierWithGroupSync();
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, gs_value[tid.x & 63u] + previous);
}
