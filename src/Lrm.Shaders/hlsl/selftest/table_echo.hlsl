// table_echo (selftest): thread i writes word i of the row of src, as lrm_tensor decodes it, to y[i].
#include "lrm.hlsli"

LRM_ARGS(table_echo)

LRM_NUMTHREADS(table_echo)
void main(uint3 tid : SV_GroupThreadID)
{
    lrm_tensor_t s = lrm_tensor(lrm_args.src);
    uint w[16] = {
        s.slot, s.offset, s.dtype, s.rank,
        s.dims.x, s.dims.y, s.dims.z, s.dims.w,
        s.strides.x, s.strides.y, s.strides.z, s.strides.w,
        s.count, s.flags, s.debug_id, s.reserved
    };
    lrm_store_u32(lrm_tensor(lrm_args.y), tid.x, w[tid.x]);
}
