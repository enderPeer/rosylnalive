import sys, time, hashlib, torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
m = sys.argv[1]
torch.manual_seed(0); torch.set_num_threads(8)
with safe_open(m + "/model.safetensors", "pt") as f:
    e = f.get_tensor("model.embed_tokens.weight"); h = f.get_tensor("lm_head.weight")
    print("lm_head == embed_tokens (bitwise):", torch.equal(e.view(torch.int16), h.view(torch.int16)), tuple(e.shape), e.dtype)
    print("q_norm shape:", tuple(f.get_tensor("model.layers.0.self_attn.q_norm.weight").shape))
tok = AutoTokenizer.from_pretrained(m)
ids = tok("public static int Add(int a, int b) => a + b;", return_tensors="pt").input_ids
print("ids:", ids.tolist()[0])
t = time.time()
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    out = model(ids).logits[0, -1]
top = torch.topk(out, 3)
print("top3 next:", [(int(i), tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)])
print("logits sha256[:16]:", hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16], "secs:", round(time.time() - t, 1))
