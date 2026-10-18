// scale_f32: y[i] = x[i] * a. One fp32 multiply per element, so the result is bit-exact against the CPU for
// normal inputs. Args: d0 groups_x, d1 token x, d2 token y, d3 n, d4 a (fp32 bits).
#include "lrm_spike.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = lrm_linear(gid, tid, 256u);
    if (i >= LRM_ARG(3))
        return;
    LrmTensor x = lrm_tensor(LRM_ARG(1));
    LrmTensor y = lrm_tensor(LRM_ARG(2));
    RWByteAddressBuffer bx = LRM_BUF(x.slot);
    RWByteAddressBuffer by = LRM_BUF(y.slot);
    float v = asfloat(bx.Load(x.byteOffset + i * 4u));
    by.Store(y.byteOffset + i * 4u, asuint(v * asfloat(LRM_ARG(4))));
}
