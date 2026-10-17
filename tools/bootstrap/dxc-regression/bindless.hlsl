// DXC regression kernel, compiled by tools/bootstrap/dxc-regression.ps1 on every CI run.
// It uses the bindless form of the dev plan: D3D12 indexes ResourceDescriptorHeap directly, and SPIR-V uses one
// explicit runtime array at set 0, binding 0, without NonUniformResourceIndex. Arguments arrive as push data or root
// constants; tensor tokens index a 64-byte row table in slot 0.
struct Args
{
    uint srcToken;
    uint dstToken;
    uint n;
    float scale;
};

#ifdef __spirv__
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);
[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);
#define LRM_BUF(i) g_buf[(i)]
#else
ConstantBuffer<Args> args : register(b0);
#define LRM_BUF(i) ResourceDescriptorHeap[(i)]
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    RWByteAddressBuffer table = LRM_BUF(0);
    uint srcSlot = table.Load(args.srcToken * 64);
    uint srcOffset = table.Load(args.srcToken * 64 + 4);
    uint dstSlot = table.Load(args.dstToken * 64);
    uint dstOffset = table.Load(args.dstToken * 64 + 4);
    if (id.x >= args.n)
        return;
    RWByteAddressBuffer src = LRM_BUF(srcSlot);
    RWByteAddressBuffer dst = LRM_BUF(dstSlot);
    dst.Store(dstOffset + id.x * 4, asuint(asfloat(src.Load(srcOffset + id.x * 4)) * args.scale));
}
