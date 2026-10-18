// silu_mul (K1, basic): y[i] = silu(gate[i]) * up[i] for i < n, silu(g) = g / (1 + exp(-g)), one thread per element.
// Element-wise, so batch-invariant by construction. Judged by the relative bound (KernelBounds).
//
// CpuRef (Rows.siluMul) computes s = g / (1 + exp(-g)), then s * up. The kernel computes the same s in a form whose
// every step stays in range for every finite gate (ADR K1-0001, decision 6):
//   a = min(|g|, 120), e = exp(-a), in (0, 1]
//   s = g / (1 + e)            for g >= +0
//   s = (-a * e) / (1 + e)     for g <= -0 (g / (1 + exp(-g)) with both sides multiplied by exp(g))
// - The denominator stays in [1, 2]. g / (1 + exp(-g)) itself divides by more than 2^126 for g below -87.34, where
//   the GPU's division is not accurate (the RTX 4080 returned 0), and by +inf below -88.72.
// - exp(-a) is range-reduced (Cody-Waite): n = round(-a * log2 e), r = (-a - n * C1) - n * C2 with C1 = 355/512 (9
//   significant bits, so n * C1 is exact) and C1 + C2 = ln 2, then exp(-a) = 2^n * exp(r) with |r| <= ln 2 / 2. The
//   GPU's exp evaluates exp2(x * log2 e), whose error grows with |x| from rounding that product; on r it stays at the
//   error of a small argument.
// - 2^n (n >= -173) is built from its bits as two normal powers 2^n1 * 2^n2, multiplied in after a * exp(r): that
//   product rounds once and the powers of two scale it exactly unless the result is itself below 2^-126, so a * e
//   keeps its precision even where e alone would be a denormal. With a clamped to 120, a * e < 2^-150 for every gate
//   below -120, which rounds to -0, as silu does in float32 there.
// - Every float operation is `precise`: DXIL's fast-math could otherwise fold C1 + C2 into ln 2, reorder the
//   products, or assume no operand is infinite. (Without it, a gate of -3.4e38 gave NaN on the RTX 4080 with a random
//   up but not with up = 1, as if the driver had formed g * up, which overflows, before dividing.)
// - Non-finite gates as CpuRef: +inf gives +inf, -inf gives NaN (-inf / +inf), a NaN stays a NaN. The NaN test is on
//   the bits, which fast-math cannot fold away.
//
// In place (y aliases gate or up): each thread reads its element before it writes it.
#include "lrm.hlsli"

LRM_ARGS(silu_mul)

// log2 e rounded to float32; it only rounds n, so its own rounding does not matter.
#define SILU_LOG2E 1.44269504f
// ln 2 = C1 + C2: C1 = 355/512 exactly, C2 = ln 2 - C1 rounded to float32 (Cephes expf).
#define SILU_C1 0.693359375f
#define SILU_C2 -2.12194440e-4f
// Beyond this |g|, exp(-|g|) only matters as zero.
#define SILU_A_MAX 120.0f

LRM_NUMTHREADS(silu_mul)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = LRM_GROUP_LINEAR(gid) * LRM_THREADS_X_silu_mul + tid.x;
    if (i >= lrm_args.n)
        return;

    uint gbits = lrm_load_u32(lrm_tensor(lrm_args.gate), i);
    float g = asfloat(gbits);
    float u = lrm_load_f32(lrm_tensor(lrm_args.up), i);

    // a = |g| from the bits, clamped (a NaN gate gives 120 here and is handled below).
    precise float a = min(asfloat(gbits & 0x7FFFFFFFu), SILU_A_MAX);
    precise float x = -a;
    precise float n = round(x * SILU_LOG2E);
    precise float r = (x - n * SILU_C1) - n * SILU_C2;
    precise float er = exp(r);

    // 2^n = 2^n1 * 2^n2 with n1 = floor(n / 2) and n2 = n - n1, both in [-87, 0], so both normal.
    int ni = (int)n;
    int n1 = ni >> 1;
    int n2 = ni - n1;
    float p1 = asfloat((uint)(n1 + 127) << 23);
    float p2 = asfloat((uint)(n2 + 127) << 23);

    precise float e = (er * p1) * p2;
    precise float den = 1.0f + e;
    precise float num = (gbits >> 31) != 0u ? ((x * er) * p1) * p2 : g;
    precise float s = num / den;
    if ((gbits & 0x7FFFFFFFu) > 0x7F800000u)
        s = g;
    else if (gbits == 0xFF800000u)
        s = asfloat(0x7FC00000u);
    precise float y = s * u;
    lrm_store_f32(lrm_tensor(lrm_args.y), i, y);
}
