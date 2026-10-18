// bf16_unpack: out[i] = the bf16 bits of in[i] shifted into the high half of an fp32 word. Integer operations only,
// so every bit pattern (NaNs and denormals included) is exact. Args: d0 groups_x, d1 token in (bf16, even byte
// offset), d2 token out (f32), d3 n.
#include "lrm_spike.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = lrm_linear(gid, tid, 256u);
    if (i >= LRM_ARG(3))
        return;
    LrmTensor a = lrm_tensor(LRM_ARG(1));
    LrmTensor o = lrm_tensor(LRM_ARG(2));
    RWByteAddressBuffer ba = LRM_BUF(a.slot);
    RWByteAddressBuffer bo = LRM_BUF(o.slot);
    uint addr = a.byteOffset + i * 2u;
    uint w = ba.Load(addr & ~3u);
    uint bits = (addr & 2u) != 0u ? (w >> 16) : (w & 0xFFFFu);
    bo.Store(o.byteOffset + i * 4u, bits << 16);
}
