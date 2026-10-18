// scale_f32 (selftest): y[i] = a * x[i] for i < n. One IEEE multiply per element, so it is bit-exact against the CPU.
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_scale_f32 + tid.x;
    if (i >= lrm_args.n)
        return;
    lrm_tensor_t x = lrm_tensor(lrm_args.x);
    lrm_tensor_t y = lrm_tensor(lrm_args.y);
    lrm_store_f32(y, i, lrm_load_f32(x, i) * lrm_args.a);
}
