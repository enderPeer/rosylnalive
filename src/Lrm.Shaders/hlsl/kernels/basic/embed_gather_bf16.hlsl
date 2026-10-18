// embed_gather_bf16 (K1, basic): y[r, c] = widen(table[ids[r], c]) for r < rows and c < cols; an id >= vocab gives
// a row of the quiet NaN 0x7FC00000. One thread per output element.
//
// Bit-exact in every fp32 denormal mode (KernelBounds): the bf16 bits are shifted into the high half of a u32 with
// integer operations only, so denormals and NaN payloads move unchanged and no float instruction touches them.
//
// Grid (ceildiv(cols, 256), rows, 1), not linear: gid.x is a block of 256 columns, gid.y the row. Rows are at most
// 65,535 per dispatch (grid y), so the executor splits longer prefills. Indices stay below 2^31: the table holds at
// most 2^29 bf16 elements (one 1 GiB page), and y at most 2^28 f32 elements.
#include "lrm.hlsli"

LRM_ARGS(embed_gather_bf16)

LRM_NUMTHREADS(embed_gather_bf16)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint r = gid.y;
    uint c = gid.x * LRM_THREADS_X_embed_gather_bf16 + tid.x;
    if (r >= lrm_args.rows || c >= lrm_args.cols)
        return;

    uint id = lrm_load_u32(lrm_tensor(lrm_args.ids), r);
    uint bits = LRM_F32_QUIET_NAN_BITS;
    if (id < lrm_args.vocab)
        bits = lrm_load_bf16_bits(lrm_tensor(lrm_args.table), id * lrm_args.cols + c) << 16;

    lrm_store_u32(lrm_tensor(lrm_args.y), r * lrm_args.cols + c, bits);
}
