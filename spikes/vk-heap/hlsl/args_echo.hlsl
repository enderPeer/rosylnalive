// args_echo: writes all 32 ArgBlock dwords to the output tensor, so the host can check that every dword of the
// 128-byte block arrives at its offset. Args: d0 groups_x (1), d1 token out, d2..d31 arbitrary payload.
#include "lrm_spike.hlsli"

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    LrmTensor o = lrm_tensor(LRM_ARG(1));
    RWByteAddressBuffer bo = LRM_BUF(o.slot);
    uint k = tid.x;
    bo.Store(o.byteOffset + k * 4u, g_lrmArgs.d[k >> 2][k & 3u]);
}
