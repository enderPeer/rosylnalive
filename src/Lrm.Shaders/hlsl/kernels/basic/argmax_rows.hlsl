// argmax_rows (K1, basic): y[r] = the lowest c with x[r, c] maximal, for r < rows; a row holding any NaN gives
// 0xFFFFFFFF (CpuRef.argmaxRows; the host turns it into the error CpuRef raises). One group of 256 threads per row.
//
//   1. Thread t scans its columns t, t + 256, ... in increasing order and keeps the first maximum it sees (it replaces
//      its candidate only on a strictly greater value), so its candidate is the lowest maximal index among them.
//   2. The 256 candidates meet in a fixed tree in groupshared (h = 128, 64, ..., 1): the greater value wins, and on
//      equal values (+0 and -0 included) the lower index wins, so the result is the lowest index of the row maximum,
//      whatever thread held it.
// The comparisons are exact, so the result does not depend on the order at all (BatchInvariance). NaN is detected
// with an integer test on the bits, never with a float self-comparison, which DXIL's fast-math folds away (ADR
// int-0001); a NaN never reaches a float comparison. The comparisons are float comparisons, so a class that flushes
// fp32 denormals compares them as zero, as KernelBounds' flushed reference does.
#include "lrm.hlsli"

LRM_ARGS(argmax_rows)

// No candidate: a thread whose columns are all beyond cols (cols < 256).
#define ARGMAX_NONE 0xFFFFFFFFu

// One candidate per thread: its value, its index (ARGMAX_NONE for none) and whether it saw a NaN (3 KiB).
groupshared float g_value[LRM_THREADS_X_argmax_rows];
groupshared uint g_index[LRM_THREADS_X_argmax_rows];
groupshared uint g_nan[LRM_THREADS_X_argmax_rows];

LRM_NUMTHREADS(argmax_rows)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // The row is the same for every thread of the group, so a group without work returns as a whole, before any
    // barrier.
    uint row = LRM_GROUP_LINEAR(gid);
    if (row >= lrm_args.rows)
        return;

    lrm_tensor_t x = lrm_tensor(lrm_args.x);
    uint cols = lrm_args.cols;
    uint base = row * cols;
    uint t = tid.x;

    float best = 0.0f;
    uint index = ARGMAX_NONE;
    uint nan = 0u;
    for (uint c = t; c < cols; c += LRM_THREADS_X_argmax_rows)
    {
        uint bits = lrm_load_u32(x, base + c);
        float v = asfloat(bits);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            nan = 1u;
        else if (index == ARGMAX_NONE || v > best)
        {
            best = v;
            index = c;
        }
    }
    g_value[t] = best;
    g_index[t] = index;
    g_nan[t] = nan;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint h = LRM_THREADS_X_argmax_rows / 2u; h > 0u; h >>= 1)
    {
        if (t < h)
        {
            uint ia = g_index[t];
            uint ib = g_index[t + h];
            float va = g_value[t];
            float vb = g_value[t + h];
            bool takeB = ib != ARGMAX_NONE && (ia == ARGMAX_NONE || vb > va || (vb == va && ib < ia));
            if (takeB)
            {
                g_value[t] = vb;
                g_index[t] = ib;
            }
            g_nan[t] = g_nan[t] | g_nan[t + h];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0u)
        lrm_store_u32(lrm_tensor(lrm_args.y), row, g_nan[0] != 0u ? 0xFFFFFFFFu : g_index[0]);
}
