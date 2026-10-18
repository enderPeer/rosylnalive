// spirv-fixture: 40 KiB of groupshared memory. The SPIR-V backend of DXC accepts it (the RTX 4080 allows 48 KiB), so
// only the inspector's 32 KiB sum catches it on this target.
// kernel: scale_f32
// expect: workgroup-size
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

groupshared float4 gs_tile[2560];

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    for (uint k = tid.x; k < 2560u; k += 256u)
        gs_tile[k] = float4(lrm_args.a, 0.0f, 0.0f, 0.0f);
    GroupMemoryBarrierWithGroupSync();
    lrm_store_f32(lrm_tensor(lrm_args.y), tid.x, gs_tile[2559u - tid.x].x);
}
