struct Args { uint a; uint c; uint n; uint pad; };
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);
RWByteAddressBuffer bufs[] : register(u0, space0);
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= args.n) return;
  half h = bufs[args.a].Load<half>(id.x*2);
  bufs[args.c].Store<float>(id.x*4, (float)h * 2.0f);
}
