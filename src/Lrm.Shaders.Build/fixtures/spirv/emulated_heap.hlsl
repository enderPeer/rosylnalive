// spirv-fixture: ResourceDescriptorHeap with -fvk-bind-resource-heap, the variant the R1 ADR rejected: DXC emits one
// aliased variable per heap access, all at set 0 binding 0, so "exactly one resource variable" fails.
// kernel: scale_f32
// expect: resource
// flags: -fvk-bind-resource-heap 0 0
#include "lrm_args.g.hlsli"

[[vk::push_constant]] ConstantBuffer<lrm_args_scale_f32> lrm_args : register(b0, space0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint i = (gid.y * lrm_args.lrm_groups_x + gid.x) * 256u + tid.x;
    if (i >= lrm_args.n)
        return;
    RWByteAddressBuffer table = ResourceDescriptorHeap[0];
    uint4 x = table.Load4(lrm_args.x * 64u);
    uint4 y = table.Load4(lrm_args.y * 64u);
    RWByteAddressBuffer bx = ResourceDescriptorHeap[x.x];
    RWByteAddressBuffer by = ResourceDescriptorHeap[y.x];
    by.Store(y.y + i * 4u, asuint(asfloat(bx.Load(x.y + i * 4u)) * lrm_args.a));
}
