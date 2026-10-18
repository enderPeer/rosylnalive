struct Args { uint a; uint b; uint c; uint n; };
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);
RWStructuredBuffer<float> bufs[] : register(u0, space0);
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= args.n) return;
  bufs[args.c][id.x] = bufs[args.a][id.x] + bufs[args.b][id.x];
}
