// args_echo (selftest): thread i writes argument word i to y[i], for all 32 words; y[0] is lrm_groups_x.
#include "lrm.hlsli"

LRM_ARGS(args_echo)

LRM_NUMTHREADS(args_echo)
void main(uint3 tid : SV_GroupThreadID)
{
    uint w[32] = {
        lrm_args.lrm_groups_x, lrm_args.y, lrm_args.v2, lrm_args.v3,
        lrm_args.v4, lrm_args.v5, lrm_args.v6, lrm_args.v7,
        lrm_args.v8, lrm_args.v9, lrm_args.v10, lrm_args.v11,
        lrm_args.v12, lrm_args.v13, lrm_args.v14, lrm_args.v15,
        lrm_args.v16, lrm_args.v17, lrm_args.v18, lrm_args.v19,
        lrm_args.v20, lrm_args.v21, lrm_args.v22, lrm_args.v23,
        lrm_args.v24, lrm_args.v25, lrm_args.v26, lrm_args.v27,
        lrm_args.v28, lrm_args.v29, lrm_args.v30, lrm_args.v31
    };
    lrm_store_u32(lrm_tensor(lrm_args.y), tid.x, w[tid.x]);
}
