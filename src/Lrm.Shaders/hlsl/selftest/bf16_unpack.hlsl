// bf16_unpack (selftest): y[i] = widen(x[i]) for i < n, the bf16 bits shifted into the high half of an f32. Exact.
#include "lrm.hlsli"

LRM_ARGS(bf16_unpack)

LRM_NUMTHREADS(bf16_unpack)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_bf16_unpack + tid.x;
    if (i >= lrm_args.n)
        return;
    // Integer ops only, so every bit pattern (NaN payloads and denormals included) moves unchanged.
    lrm_store_u32(lrm_tensor(lrm_args.y), i, lrm_load_bf16_bits(lrm_tensor(lrm_args.x), i) << 16);
}
