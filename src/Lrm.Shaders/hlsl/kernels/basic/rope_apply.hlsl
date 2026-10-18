// rope_apply (K1, basic): NeoX rotate-half RoPE of rows = tokens * heads rows of head_dim elements, from the T_spec
// tables (CpuRef.ropeTable over the pinned inv_freq bits), never evaluating sin or cos. Row r has position
// pos = pos0 + r / heads and h = head_dim / 2; for i < h, with c = cos_table[pos, i] and s = sin_table[pos, i]:
//   y[i]     = x[i] * c - x[i + h] * s
//   y[i + h] = x[i + h] * c + x[i] * s
// (for Qwen3-0.6B, head_dim 128: element i pairs with i + 64). One group of 64 threads per row; thread t handles
// i = t, t + 64, ... < h.
//
// Every product and sum is `precise`, so DXIL's fast-math cannot contract a product into an FMA or reorder the terms:
// each operation rounds once, in the order CpuRef's Rows.ropeHead writes it. Element-wise, so batch-invariant by
// construction (BatchInvariance).
//
// In place (y aliases x): the thread that reads x[i] and x[i + h] is the one that writes y[i] and y[i + h].
#include "lrm.hlsli"

LRM_ARGS(rope_apply)

LRM_NUMTHREADS(rope_apply)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint row = LRM_GROUP_LINEAR(gid);
    if (row >= lrm_args.rows)
        return;

    lrm_tensor_t x = lrm_tensor(lrm_args.x);
    lrm_tensor_t ct = lrm_tensor(lrm_args.cos_table);
    lrm_tensor_t st = lrm_tensor(lrm_args.sin_table);
    lrm_tensor_t y = lrm_tensor(lrm_args.y);
    uint hd = lrm_args.head_dim;
    uint h = hd / 2u;
    uint pos = lrm_args.pos0 + row / lrm_args.heads;
    uint xb = row * hd;
    uint tb = pos * h;

    for (uint i = tid.x; i < h; i += LRM_THREADS_X_rope_apply)
    {
        float a = lrm_load_f32(x, xb + i);
        float b = lrm_load_f32(x, xb + i + h);
        float c = lrm_load_f32(ct, tb + i);
        float s = lrm_load_f32(st, tb + i);
        precise float ac = a * c;
        precise float bs = b * s;
        precise float bc = b * c;
        precise float a_s = a * s;
        precise float y0 = ac - bs;
        precise float y1 = bc + a_s;
        lrm_store_f32(y, xb + i, y0);
        lrm_store_f32(y, xb + i + h, y1);
    }
}
