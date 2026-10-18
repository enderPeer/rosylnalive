// spirv-fixture: SV_DispatchThreadID becomes the GlobalInvocationId built-in, which is wrong on folded grids.
// kernel: scale_f32
// expect: built-ins
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= lrm_args.n)
        return;
    lrm_store_f32(lrm_tensor(lrm_args.y), dtid.x, lrm_args.a);
}
