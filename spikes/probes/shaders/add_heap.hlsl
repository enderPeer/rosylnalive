struct Args { uint a; uint b; uint c; uint n; };
ConstantBuffer<Args> args : register(b0);
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  RWByteAddressBuffer A = ResourceDescriptorHeap[args.a];
  RWByteAddressBuffer B = ResourceDescriptorHeap[args.b];
  RWByteAddressBuffer C = ResourceDescriptorHeap[args.c];
  if (id.x >= args.n) return;
  C.Store(id.x*4, asuint(asfloat(A.Load(id.x*4)) + asfloat(B.Load(id.x*4))));
}
