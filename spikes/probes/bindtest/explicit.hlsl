struct Args { uint tokX; uint tokY; uint n; float a; };
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);
#ifdef __spirv__
[[vk::binding(0,0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);
#define LRM_BUF(i) g_buf[NonUniformResourceIndex(i)]
#else
#define LRM_BUF(i) ResourceDescriptorHeap[NonUniformResourceIndex(i)]
#endif
[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  RWByteAddressBuffer tbl = LRM_BUF(0);
  uint sx = tbl.Load(args.tokX*64); uint ox = tbl.Load(args.tokX*64+4);
  uint sy = tbl.Load(args.tokY*64); uint oy = tbl.Load(args.tokY*64+4);
  if (id.x < args.n) {
    RWByteAddressBuffer bx = LRM_BUF(sx); RWByteAddressBuffer by = LRM_BUF(sy);
    float v = asfloat(bx.Load(ox + id.x*4));
    by.Store(oy + id.x*4, asuint(v*args.a));
  }
}
