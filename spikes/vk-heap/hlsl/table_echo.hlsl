// table_echo: copies tensor-table rows [first, first + rows) dword by dword to the output tensor, so the host can
// check that slot 0 is the table and that the descriptor covers all of it.
// Args: d0 groups_x, d1 token out, d2 first row, d3 row count.
#include "lrm_spike.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint j = lrm_linear(gid, tid, 256u);
    if (j >= LRM_ARG(3) * 16u)
        return;
    RWByteAddressBuffer table = LRM_BUF(0);
    uint v = table.Load(LRM_ARG(2) * 64u + j * 4u);
    LrmTensor o = lrm_tensor(LRM_ARG(1));
    RWByteAddressBuffer bo = LRM_BUF(o.slot);
    bo.Store(o.byteOffset + j * 4u, v);
}
