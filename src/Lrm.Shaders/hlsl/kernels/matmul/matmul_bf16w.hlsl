// matmul_bf16w (K2): y[i, j] = sum over p < k of x[i, p] * widen(w[j, p]) for i < m, j < n.
//   x: f32 [m, k] activations; w: bf16 [n, k] weights in the PyTorch Linear layout (row j is output feature j, as
//   Qwen stores it); y: f32 [m, n]. One kernel for decode (m = 1), prefill and lm_head (n = 151,936).
//
// Arithmetic (BatchInvariance, KernelAbiAndCatalog "K2"): every output starts from +0 and accumulates
// acc = mad(x[i, p], widen(w[j, p]), acc) for p = 0, 1, ..., k - 1, one step per p, in increasing order. The order
// depends only on k: tiling over i and j decides which thread computes y[i, j], never how. There is no split-K and no
// reduction across threads or groups, so a row computed alone (m = 1) equals the same row inside any batch. CpuRef
// F32 (Rows.dotBf16) is the same chain with one fused multiply-add per step.
//
// Work split: group (gid.x, gid.y) owns the 64x64 output tile at rows i0 = 64 * gid.y and columns j0 = 64 * gid.x
// (grid ceildiv(n, 64) x ceildiv(m, 64), not linear). Thread (tx, ty) of the 16x16 group owns rows i0 + ty + 16a and
// columns j0 + tx + 16b for a, b in 0..3: a 4x4 register tile. k is walked in slabs of KS = 32: the group stages the
// x slab (64 rows x 32) and the w slab (64 columns x 32, widened to f32) in groupshared memory, and each thread reads
// them as float4 runs of 4 consecutive p. The next slab is loaded into registers while the current one is used.
//
// Out-of-range rows, columns and p are never read: such elements are staged as 0 and never accumulated (the p loop
// stops at k exactly). Rows >= m are not computed at all, so a decode tile (m = 1) does 1/64 of the arithmetic of a
// full one; that skips work, it never changes the steps of a valid output.
//
// Rules: LRM_ARGS, LRM_BUF (through lrm_buf) and lrm_tensor only; no atomics, wave ops or 16-bit types (bf16 is read
// as u32 words); groupshared 2 x 64 x 9 float4 = 18,432 bytes (limit 32 KiB).
#include "lrm.hlsli"

LRM_ARGS(matmul_bf16w)

// The output tile is TILE x TILE; KS is the k slab; a float4 row of a slab holds KS / 4 = 8 runs, padded by one so
// the column reads of a warp fall in distinct banks.
static const uint TILE = 64u;
static const uint KS = 32u;
static const uint RUNS = KS / 4u;
static const uint STRIDE = RUNS + 1u;

// xs[r][q] = x[i0 + r, k0 + 4q .. 4q + 3]; ws[c][q] = widen(w[j0 + c, k0 + 4q .. 4q + 3]).
groupshared float4 xs[TILE][STRIDE];
groupshared float4 ws[TILE][STRIDE];

// Four bf16 elements from two u32 words, widened exactly (element 2e is the low half of word e).
float4 widen4(uint lo, uint hi)
{
    return float4(asfloat(lo << 16), asfloat(lo & 0xFFFF0000u), asfloat(hi << 16), asfloat(hi & 0xFFFF0000u));
}

// x[i, p .. p + 3] for a row i < m; elements at p + e >= k are 0 and are not read.
float4 load_x(lrm_tensor_t x, uint i, uint p, uint k)
{
    uint e0 = i * k + p;
    if (p + 4u <= k)
        return lrm_load4_f32(x, e0);
    float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll] for (uint e = 0u; e < 4u; ++e)
    {
        if (p + e < k)
            v[e] = lrm_load_f32(x, e0 + e);
    }
    return v;
}

// widen(w[j, p .. p + 7]) for a column j < n, as two float4; elements at p + e >= k are 0 and are not read. With k
// even the 8 elements are 4 aligned u32 words (one Load4); otherwise they are read one by one.
void load_w(lrm_tensor_t w, uint j, uint p, uint k, out float4 lo, out float4 hi)
{
    uint e0 = j * k + p;
    if (p + 8u <= k && (k & 1u) == 0u)
    {
        uint4 q = lrm_buf(w.slot).Load4(w.offset + e0 * 2u);
        lo = widen4(q.x, q.y);
        hi = widen4(q.z, q.w);
        return;
    }
    float v[8];
    [unroll] for (uint e = 0u; e < 8u; ++e)
        v[e] = (p + e < k) ? lrm_load_bf16(w, e0 + e) : 0.0f;
    lo = float4(v[0], v[1], v[2], v[3]);
    hi = float4(v[4], v[5], v[6], v[7]);
}

LRM_NUMTHREADS(matmul_bf16w)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint t : SV_GroupIndex)
{
    uint m = lrm_args.m;
    uint n = lrm_args.n;
    uint k = lrm_args.k;
    uint i0 = gid.y * TILE;
    uint j0 = gid.x * TILE;
    // The grid is exact (ceildiv in both dimensions), so every group has work; this guards a caller's grid only.
    if (i0 >= m || j0 >= n)
        return;

    lrm_tensor_t x = lrm_tensor(lrm_args.x);
    lrm_tensor_t w = lrm_tensor(lrm_args.w);
    lrm_tensor_t y = lrm_tensor(lrm_args.y);

    uint tx = tid.x;
    uint ty = tid.y;
    // Valid row slots of this thread: rows i0 + ty + 16a < m for a < live (0..4).
    uint live = (i0 + ty < m) ? min(4u, (m - i0 - ty + 15u) / 16u) : 0u;

    // Staging: thread t loads x runs (row t / 8 and t / 8 + 32, run t % 8) and w runs (column t / 4, runs 2 (t % 4)
    // and 2 (t % 4) + 1).
    uint xr = t >> 3;
    uint xq = t & 7u;
    uint wc = t >> 2;
    uint wq = (t & 3u) * 2u;

    // acc[a][b] is y[i0 + ty + 16a, j0 + tx + 16b]; every output starts from +0.
    float acc[4][4];
    [unroll] for (uint a0 = 0u; a0 < 4u; ++a0)
        [unroll] for (uint b0 = 0u; b0 < 4u; ++b0)
            acc[a0][b0] = 0.0f;

    uint slabs = (k + KS - 1u) / KS;
    float4 sx0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sx1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sw0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sw1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (slabs > 0u)
    {
        if (i0 + xr < m)
            sx0 = load_x(x, i0 + xr, 4u * xq, k);
        if (i0 + xr + 32u < m)
            sx1 = load_x(x, i0 + xr + 32u, 4u * xq, k);
        if (j0 + wc < n)
            load_w(w, j0 + wc, 4u * wq, k, sw0, sw1);
    }

    [loop] for (uint s = 0u; s < slabs; ++s)
    {
        uint k0 = s * KS;
        GroupMemoryBarrierWithGroupSync();
        xs[xr][xq] = sx0;
        xs[xr + 32u][xq] = sx1;
        ws[wc][wq] = sw0;
        ws[wc][wq + 1u] = sw1;
        GroupMemoryBarrierWithGroupSync();

        // Prefetch the next slab into registers while this one is used.
        if (s + 1u < slabs)
        {
            uint kn = k0 + KS;
            sx0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            sx1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            sw0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            sw1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (i0 + xr < m)
                sx0 = load_x(x, i0 + xr, kn + 4u * xq, k);
            if (i0 + xr + 32u < m)
                sx1 = load_x(x, i0 + xr + 32u, kn + 4u * xq, k);
            if (j0 + wc < n)
                load_w(w, j0 + wc, kn + 4u * wq, k, sw0, sw1);
        }

        if (live == 0u)
            continue;

        // The steps of this slab: p = k0 .. k0 + steps - 1, in order, for every output of the thread.
        uint steps = min(KS, k - k0);
        uint runs = steps >> 2;
        [loop] for (uint q = 0u; q < runs; ++q)
        {
            float4 wb0 = ws[tx][q];
            float4 wb1 = ws[tx + 16u][q];
            float4 wb2 = ws[tx + 32u][q];
            float4 wb3 = ws[tx + 48u][q];
            [unroll] for (uint a = 0u; a < 4u; ++a)
            {
                if (a < live)
                {
                    float4 xa = xs[ty + 16u * a][q];
                    // Each output takes p = 4q, 4q + 1, 4q + 2, 4q + 3, in that order.
                    acc[a][0] = mad(xa.x, wb0.x, acc[a][0]);
                    acc[a][0] = mad(xa.y, wb0.y, acc[a][0]);
                    acc[a][0] = mad(xa.z, wb0.z, acc[a][0]);
                    acc[a][0] = mad(xa.w, wb0.w, acc[a][0]);
                    acc[a][1] = mad(xa.x, wb1.x, acc[a][1]);
                    acc[a][1] = mad(xa.y, wb1.y, acc[a][1]);
                    acc[a][1] = mad(xa.z, wb1.z, acc[a][1]);
                    acc[a][1] = mad(xa.w, wb1.w, acc[a][1]);
                    acc[a][2] = mad(xa.x, wb2.x, acc[a][2]);
                    acc[a][2] = mad(xa.y, wb2.y, acc[a][2]);
                    acc[a][2] = mad(xa.z, wb2.z, acc[a][2]);
                    acc[a][2] = mad(xa.w, wb2.w, acc[a][2]);
                    acc[a][3] = mad(xa.x, wb3.x, acc[a][3]);
                    acc[a][3] = mad(xa.y, wb3.y, acc[a][3]);
                    acc[a][3] = mad(xa.z, wb3.z, acc[a][3]);
                    acc[a][3] = mad(xa.w, wb3.w, acc[a][3]);
                }
            }
        }
        // The last slab's remaining 1 to 3 steps (k not a multiple of 4).
        [loop] for (uint p = runs * 4u; p < steps; ++p)
        {
            uint q = p >> 2;
            uint e = p & 3u;
            float wv0 = ws[tx][q][e];
            float wv1 = ws[tx + 16u][q][e];
            float wv2 = ws[tx + 32u][q][e];
            float wv3 = ws[tx + 48u][q][e];
            [unroll] for (uint a = 0u; a < 4u; ++a)
            {
                if (a < live)
                {
                    float xv = xs[ty + 16u * a][q][e];
                    acc[a][0] = mad(xv, wv0, acc[a][0]);
                    acc[a][1] = mad(xv, wv1, acc[a][1]);
                    acc[a][2] = mad(xv, wv2, acc[a][2]);
                    acc[a][3] = mad(xv, wv3, acc[a][3]);
                }
            }
        }
    }

    [unroll] for (uint a = 0u; a < 4u; ++a)
    {
        uint i = i0 + ty + 16u * a;
        if (a < live)
        {
            [unroll] for (uint b = 0u; b < 4u; ++b)
            {
                uint j = j0 + tx + 16u * b;
                if (j < n)
                    lrm_store_f32(y, i * n + j, acc[a][b]);
            }
        }
    }
}
