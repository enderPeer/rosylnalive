// rmsnorm_rows (K1, basic): y[r, :] = widen(w[:]) * (x[r, :] * rsqrt(mean(x[r, :]^2) + eps)), one group of 256 threads
// per row (Qwen3RMSNorm with the weight used as w, not 1 + w; CpuReferenceSemantics).
//
// The sum of squares is fp32 in a fixed order that depends only on cols (BatchInvariance):
//   1. thread t accumulates mad(x, x, s) over its columns t, t + 256, t + 512, ... in increasing order, from +0;
//   2. the 256 partial sums meet in a fixed tree in groupshared: s[t] += s[t + h] for h = 128, 64, ..., 1.
// No atomics and no wave operations, so the order is the same on every device, whichever group runs the row and
// however many rows the dispatch has. The order differs from CpuRef's single sequential sum, so the kernel is judged
// by the relative bound (KernelBounds).
//
// In place (y aliases x): every thread reads and writes only its own columns, and reads each before it writes it.
#include "lrm.hlsli"

LRM_ARGS(rmsnorm_rows)

// The partial sums, one per thread (1 KiB).
groupshared float g_partial[LRM_THREADS_X_rmsnorm_rows];

LRM_NUMTHREADS(rmsnorm_rows)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // The row is the same for every thread of the group, so a group without work returns as a whole, before any
    // barrier.
    uint row = LRM_GROUP_LINEAR(gid);
    if (row >= lrm_args.rows)
        return;

    lrm_tensor_t x = lrm_tensor(lrm_args.x);
    lrm_tensor_t w = lrm_tensor(lrm_args.w);
    lrm_tensor_t y = lrm_tensor(lrm_args.y);
    uint cols = lrm_args.cols;
    uint base = row * cols;
    uint t = tid.x;

    float s = 0.0f;
    for (uint c = t; c < cols; c += LRM_THREADS_X_rmsnorm_rows)
    {
        float v = lrm_load_f32(x, base + c);
        s = mad(v, v, s);
    }
    g_partial[t] = s;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint h = LRM_THREADS_X_rmsnorm_rows / 2u; h > 0u; h >>= 1)
    {
        if (t < h)
            g_partial[t] = g_partial[t] + g_partial[t + h];
        GroupMemoryBarrierWithGroupSync();
    }

    float mean = g_partial[0] / (float)cols;
    float r = rsqrt(mean + lrm_args.eps);

    for (uint c2 = t; c2 < cols; c2 += LRM_THREADS_X_rmsnorm_rows)
    {
        float v = lrm_load_f32(x, base + c2);
        lrm_store_f32(y, base + c2, lrm_load_bf16(w, c2) * (v * r));
    }
}
