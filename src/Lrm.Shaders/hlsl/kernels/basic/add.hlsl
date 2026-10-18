// add (K1, basic): y[i] = a[i] + b[i] for i < n, one thread per element: one IEEE float32 add, `precise` so that
// DXIL's fast-math flags cannot rewrite it. Bit-exact against CpuRef F32 in the class's measured fp32 denormal mode;
// a NaN result matches any NaN (KernelBounds, fp32 arithmetic: GPUs return 0x7FFFFFFF for a NaN operand).
//
// In place (y aliases a or b): each thread reads its element before it writes it.
#include "lrm.hlsli"

LRM_ARGS(add)

LRM_NUMTHREADS(add)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_add + tid.x;
    if (i >= lrm_args.n)
        return;

    precise float s = lrm_load_f32(lrm_tensor(lrm_args.a), i) + lrm_load_f32(lrm_tensor(lrm_args.b), i);
    lrm_store_f32(lrm_tensor(lrm_args.y), i, s);
}
