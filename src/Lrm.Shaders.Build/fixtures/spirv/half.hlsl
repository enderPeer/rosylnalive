// spirv-fixture: a 16-bit float (float16_t, with -enable-16bit-types): the Float16 capability and a 16-bit type.
// kernel: scale_f32
// expect: capabilities, 16-bit-types
// flags: -enable-16bit-types
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    float16_t h = (float16_t)lrm_args.a;
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, (float)(h * h));
}
