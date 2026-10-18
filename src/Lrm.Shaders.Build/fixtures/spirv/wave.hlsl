// spirv-fixture: a wave reduction (WaveActiveSum): OpGroupNonUniformFAdd and the GroupNonUniformArithmetic capability.
// kernel: scale_f32
// expect: capabilities, wave-ops
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, WaveActiveSum(lrm_args.a));
}
