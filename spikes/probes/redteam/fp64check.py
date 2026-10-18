import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding, eager_attention_forward
from transformers import Qwen3Config
torch.manual_seed(0)
n = Qwen3RMSNorm(1024).double()
with torch.no_grad(): n.weight.copy_(1 + 3*torch.rand(1024, dtype=torch.float64))
x = torch.randn(8, 1024, dtype=torch.float64)
hf = n(x)
exact = n.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
print("RMSNorm out dtype", hf.dtype, " relL2(HF fp64, true fp64) =", float((hf-exact).norm()/exact.norm()))
cfg = Qwen3Config(head_dim=128, rope_theta=1e6)
r = Qwen3RotaryEmbedding(cfg).double()
pos = torch.arange(4096)[None]
cos, sin = r(torch.zeros(1, dtype=torch.float64), pos)
inv = 1.0/(1e6 ** (torch.arange(0,128,2,dtype=torch.float64)/128))
f = pos[0,:,None].double()*inv
c64 = torch.cat([f,f],-1).cos()
print("RoPE cos dtype", cos.dtype, " max|cos_HF64 - cos_true64| =", float((cos[0]-c64).abs().max()))
inv32 = 1.0/(1e6 ** (torch.arange(0,128,2,dtype=torch.float)/128))
inv_f64r = (1.0/(1e6 ** (torch.arange(0,128,2,dtype=torch.float64)/128))).float()
print("inv_freq entries where torch fp32 != f32(f64 formula):", int((inv32 != inv_f64r).sum()), "of 64")
