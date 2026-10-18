// spirv-fixture: -denorm preserve on SPIR-V declares DenormPreserve (a capability and an execution mode), which needs
// shaderDenormPreserveFloat32; the RTX 4080 reports false under driver 616.92.
// kernel: scale_f32
// expect: capabilities, execution-mode
// flags: -denorm preserve
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_scale_f32 + tid.x;
    if (i >= lrm_args.n)
        return;
    lrm_store_f32(lrm_tensor(lrm_args.y), i, lrm_load_f32(lrm_tensor(lrm_args.x), i) * lrm_args.a);
}
