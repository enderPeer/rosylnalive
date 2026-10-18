// spirv-fixture: 64-bit integer arithmetic: the Int64 capability (shaderInt64 is not a recipe feature) and a 64-bit type.
// kernel: scale_f32
// expect: capabilities, scalar-width
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint64_t wide = (uint64_t)lrm_args.n * (uint64_t)(tid.x + 1u);
    lrm_store_u32(lrm_tensor(lrm_args.y), tid.x, (uint)(wide >> 7));
}
