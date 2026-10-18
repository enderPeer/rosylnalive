// groups_echo (selftest): logical group g writes g to y[g] for g < groups. Above 65,535 groups the grid is folded,
// so this checks LRM_GROUP_LINEAR and that the extra groups of a folded grid do nothing.
#include "lrm.hlsli"

LRM_ARGS(groups_echo)

LRM_NUMTHREADS(groups_echo)
void main(uint3 gid : SV_GroupID)
{
    uint g = LRM_GROUP_LINEAR(gid);
    if (g >= lrm_args.groups)
        return;
    lrm_store_u32(lrm_tensor(lrm_args.y), g, g);
}
