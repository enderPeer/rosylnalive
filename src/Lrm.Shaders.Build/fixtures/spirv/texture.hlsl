// spirv-fixture: a typed view (RWTexture1D<float>): an image type in UniformConstant storage with its own binding.
// kernel: scale_f32
// expect: capabilities, resource, storage-class, typed-views
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

[[vk::binding(3, 0)]] RWTexture1D<float> image : register(u2, space0);

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    image[tid.x] = lrm_args.a;
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, lrm_args.a);
}
