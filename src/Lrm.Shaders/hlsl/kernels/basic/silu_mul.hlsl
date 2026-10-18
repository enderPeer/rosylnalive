// silu_mul (K1, basic): y[i] = silu(gate[i]) * up[i] for i < n, silu(g) = g / (1 + exp(-g)), one thread per element.
// The order of operations is CpuRef's (Rows.siluMul): s = g / (1 + exp(-g)), then s * up. exp and the division are
// the GPU's own, so the kernel is judged by the relative bound (KernelBounds). Element-wise, so batch-invariant by
// construction. Large negative gates overflow exp to +inf and give a signed zero, as on the CPU.
//
// In place (y aliases gate or up): each thread reads its element before it writes it.
#include "lrm.hlsli"

LRM_ARGS(silu_mul)

LRM_NUMTHREADS(silu_mul)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_silu_mul + tid.x;
    if (i >= lrm_args.n)
        return;

    float g = lrm_load_f32(lrm_tensor(lrm_args.gate), i);
    float u = lrm_load_f32(lrm_tensor(lrm_args.up), i);
    float s = g / (1.0f + exp(-g));
    lrm_store_f32(lrm_tensor(lrm_args.y), i, s * u);
}
