struct Args { uint tokX; uint tokY; uint n; float a; };
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);
[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  RWByteAddressBuffer tbl = ResourceDescriptorHeap[0];
  uint sx = tbl.Load(args.tokX*64); uint ox = tbl.Load(args.tokX*64+4);
  uint sy = tbl.Load(args.tokY*64); uint oy = tbl.Load(args.tokY*64+4);
  if (id.x < args.n) {
    RWByteAddressBuffer bx = ResourceDescriptorHeap[NonUniformResourceIndex(sx)]; RWByteAddressBuffer by = ResourceDescriptorHeap[NonUniformResourceIndex(sy)];
    float v = asfloat(bx.Load(ox + id.x*4));
    by.Store(oy + id.x*4, asuint(v*args.a));
  }
}
