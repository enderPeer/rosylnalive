// spirv-fixture: a constant buffer beside the push block: a Uniform variable with its own set and binding.
// kernel: scale_f32
// expect: resource, storage-class
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

struct extra_t
{
    float bias;
};
[[vk::binding(2, 0)]] ConstantBuffer<extra_t> extra : register(b1, space0);

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, lrm_args.a + extra.bias);
}
