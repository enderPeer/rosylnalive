// copy_rows (K1, basic): dst[(dst_row0 + r) * dst_stride + c] = src[(src_row0 + r) * src_stride + c] for r < rows and
// c < cols, 4-byte elements of one dtype (f32, u32 or i32). One thread per copied element, element e of the rows * cols
// grid being row e / cols, column e % cols.
//
// Bit-exact in every mode (KernelBounds): the words move as u32, so NaN payloads, signed zeros and denormals pass
// unchanged. src_stride may be 0 (every row reads the same source row) and source rows may overlap; destination rows
// never do (cols <= dst_stride when rows > 1, kernel-abi.md "Tensor access"). Every index stays below the tensors'
// counts, which a page bounds by 2^28 elements, so u32 arithmetic cannot wrap.
#include "lrm.hlsli"

LRM_ARGS(copy_rows)

LRM_NUMTHREADS(copy_rows)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint e = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_copy_rows + tid.x;
    uint cols = lrm_args.cols;
    if (e >= lrm_args.rows * cols)
        return;

    uint r = e / cols;
    uint c = e - r * cols;
    uint v = lrm_load_u32(lrm_tensor(lrm_args.src), (lrm_args.src_row0 + r) * lrm_args.src_stride + c);
    lrm_store_u32(lrm_tensor(lrm_args.dst), (lrm_args.dst_row0 + r) * lrm_args.dst_stride + c, v);
}
