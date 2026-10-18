// spirv-fixture: numthreads(128, 1, 1) for scale_f32, whose catalog threads are 256 x 1 x 1.
// kernel: scale_f32
// expect: execution-mode
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, lrm_args.a);
}
