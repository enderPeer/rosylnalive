// Probe, not a kernel: why the ArgBlock is uint4 d[8] and not uint d[32]. Under the D3D cbuffer packing every array
// element starts a new 16-byte register (DXIL reports the cbuffer as 500 bytes, more than 32 root constants), while
// DXC lays the same push constant out with a 4-byte stride on SPIR-V. uint4 d[8] is 128 bytes with stride 16 on both.
struct FlatArgs
{
    uint d[32];
};

#if defined(__spirv__)
[[vk::push_constant]]
#endif
ConstantBuffer<FlatArgs> g_flat : register(b0, space0);

RWByteAddressBuffer g_out : register(u0, space0);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    g_out.Store(tid.x * 4u, g_flat.d[tid.x]);
}
