"""The tiny Qwen3 model behind goldens/tiny/g1 (GoldenSet contract).

A scaled-down Qwen3 in the real checkpoint's config schema (rope_theta, torch_dtype, explicit head_dim): 2 layers,
hidden 64, 4 query heads and 2 KV heads of head_dim 32 (so the query width 128 differs from the hidden size, as in
Qwen3-0.6B), intermediate 128, vocabulary 256, tied embeddings and no lm_head tensor. Every weight is drawn from
Lrm.Core's DetRng, so an F# test can regenerate the file bit for bit:

    z      = (sum of 12 consecutive NextF32 draws) - 6        exact in float64, from DetRng(SEED, streamOf("tiny-g1/<name>"))
    value  = scale * z            for matrices and the embedding
           = scale * (1 + z / 4)  for RMSNorm weights
    weight = Bf16.narrowRne(float32(value))                   float64 -> float32 round to nearest even, then bf16 RNE

Scales: embedding 0.1; q/k/v, gate and up 1/sqrt(fan_in); o_proj and down_proj 2/sqrt(fan_in); RMSNorm weights 1,
except the final norm at 4. These keep the tied-embedding copy effect small, so greedy decoding visits many tokens
with top-2 margins far above any fp32 or GPU error.
"""

from __future__ import annotations

import numpy as np

import common as c

SET, VERSION = "tiny", "g1"
REVISION = "local:tiny-g1"
SEED = 1

H, INTER, VOCAB, LAYERS, HEADS, KV_HEADS, HEAD_DIM, MAX_POS = 64, 128, 256, 2, 4, 2, 32, 512

# Same keys as the real Qwen3-0.6B config.json, which transformers 5.20.1 still loads (rope_theta, torch_dtype).
# eos_token_id is filled in by config(): make_tiny.py sets it to a token that greedy decoding of p0 emits (EOS_RULE).
CONFIG_BASE = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": None,
    "head_dim": HEAD_DIM,
    "hidden_act": "silu",
    "hidden_size": H,
    "initializer_range": 0.02,
    "intermediate_size": INTER,
    "max_position_embeddings": MAX_POS,
    "max_window_layers": LAYERS,
    "model_type": "qwen3",
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": KV_HEADS,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "5.20.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": VOCAB,
}

# Prompts: (name, length, what is dumped). p2 crosses the 64-key block boundary of the GPU attention kernel.
PROMPTS = (("p0", 13, "taps"), ("p1", 1, "taps"), ("p2", 70, "logits"))
GREEDY_PROMPTS = ("p0", "p1")
GREEDY_STEPS = 64
GRAD_PROMPT = "p0"

# Prompt selection: a greedy prompt is the first candidate k = 0, 1, ... whose torch32 top-2 margin is at least
# MIN_MARGIN at every one of the 64 steps (about 1% of the logit spread), so fp32 and GPU errors cannot flip a token.
MIN_MARGIN = 0.05
MAX_CANDIDATES = 2000

# EOS_RULE: eos_token_id is the first token of p0's torch32 greedy output whose first occurrence is at step 8 or
# later and which is not bos_token_id. A decoder that stops at EOS therefore cannot reproduce the 64-step golden.
EOS_MIN_STEP = 8


def weight_specs() -> dict:
    """{tensor name: (shape, kind, scale)} with kind "matrix" or "norm", in the HF checkpoint naming."""
    s = {
        "model.embed_tokens.weight": ([VOCAB, H], "matrix", 0.1),
        "model.norm.weight": ([H], "norm", 4.0),
    }
    q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        s[p + "input_layernorm.weight"] = ([H], "norm", 1.0)
        s[p + "post_attention_layernorm.weight"] = ([H], "norm", 1.0)
        s[p + "self_attn.q_proj.weight"] = ([q, H], "matrix", H**-0.5)
        s[p + "self_attn.k_proj.weight"] = ([kv, H], "matrix", H**-0.5)
        s[p + "self_attn.v_proj.weight"] = ([kv, H], "matrix", H**-0.5)
        s[p + "self_attn.o_proj.weight"] = ([H, q], "matrix", 2.0 * q**-0.5)
        s[p + "self_attn.q_norm.weight"] = ([HEAD_DIM], "norm", 1.0)
        s[p + "self_attn.k_norm.weight"] = ([HEAD_DIM], "norm", 1.0)
        s[p + "mlp.gate_proj.weight"] = ([INTER, H], "matrix", H**-0.5)
        s[p + "mlp.up_proj.weight"] = ([INTER, H], "matrix", H**-0.5)
        s[p + "mlp.down_proj.weight"] = ([H, INTER], "matrix", 2.0 * INTER**-0.5)
    return s


def weights_bf16() -> dict:
    """{name: bf16 bits as uint16 [shape]} drawn as the module docstring says."""
    out = {}
    for name, (shape, kind, scale) in weight_specs().items():
        n = int(np.prod(shape))
        z = c.DetRng(SEED, c.stream_of("tiny-g1/" + name)).normal_ih12(n)
        v = scale * (1.0 + z / 4.0) if kind == "norm" else scale * z
        out[name] = c.bf16_narrow_rne(v.astype(np.float32)).reshape(shape)
    return out


def candidate_ids(name: str, k: int, length: int) -> list[int]:
    """Candidate k of a prompt: NextInt(1, VOCAB) draws from DetRng(SEED, streamOf("tiny-g1/prompt/<name>/<k>")).
    Id 0 (bos_token_id) never appears in a prompt."""
    r = c.DetRng(SEED, c.stream_of(f"tiny-g1/prompt/{name}/{k}"))
    return [r.next_int(1, VOCAB) for _ in range(length)]


def config(eos_token_id: int) -> dict:
    cfg = dict(CONFIG_BASE)
    cfg["eos_token_id"] = int(eos_token_id)
    return cfg


def model_files(eos_token_id: int) -> dict:
    """{file name: bytes} of goldens/tiny/g1/model: config.json, model.safetensors, REVISION and SHA256SUMS."""
    w = weights_bf16()
    files = {
        "config.json": c.json_bytes(config(eos_token_id)),
        "model.safetensors": c.safetensors_bytes({n: c.st_bf16(b) for n, b in w.items()}, {"format": "pt"}),
        "REVISION": (REVISION + "\n").encode("ascii"),
    }
    files["SHA256SUMS"] = c.sha256sums_bytes(files)
    return files
