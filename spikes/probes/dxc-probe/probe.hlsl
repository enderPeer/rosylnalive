struct Args { uint a; uint b; uint c; uint n; };
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= args.n) return;
    RWByteAddressBuffer A = ResourceDescriptorHeap[args.a];
    RWByteAddressBuffer B = ResourceDescriptorHeap[args.b];
    RWByteAddressBuffer C = ResourceDescriptorHeap[args.c];
    uint off = id.x * 4;
    // bf16 widen: weights stored as raw uint16 bits
    uint16_t w = A.Load<uint16_t>(id.x * 2);
    float x = asfloat(uint(w) << 16);
    C.Store<float>(off, x + B.Load<float>(off));
}
