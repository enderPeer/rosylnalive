// spirv-fixture: NonUniformResourceIndex on a heap index: the ShaderNonUniform capability, which needs
// shaderStorageBufferArrayNonUniformIndexing, a feature the recipe does not enable.
// kernel: scale_f32
// expect: capabilities
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    lrm_tensor_t y = lrm_tensor(lrm_args.y);
    g_buf[NonUniformResourceIndex(y.slot + (tid.x & 1u))].Store(y.offset + tid.x * 4u, asuint(lrm_args.a));
}
