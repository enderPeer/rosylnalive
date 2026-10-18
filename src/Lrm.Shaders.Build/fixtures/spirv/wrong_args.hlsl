// spirv-fixture: inspected as scale_f32 but declares LRM_ARGS(add): the push block's name, its member names and the
// type of word 4 (n of add is u32, a of scale_f32 is f32) differ from the catalog.
// kernel: scale_f32
// expect: push-block
#include "lrm.hlsli"

LRM_ARGS(add)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    lrm_store_u32(lrm_tensor(lrm_args.y), tid.x, lrm_args.n + lrm_args.a + lrm_args.b);
}
