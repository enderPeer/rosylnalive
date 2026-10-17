# ActivationNames contract

Owner: O1, which defines the names. Status: v1, frozen at the contracts-v1 tag.

Implementers:
- the golden tools (`tools/goldens/common.py`: `activation_names`, `TapRecorder`);
- M2, the Qwen forward on CPU;
- X1, the GPU executor's Taps arena.

Consumers: O2, M2, X1, G2, I1 (`lrm bisect`) and T1 (per-layer reports).

After contracts-v1, names change only through a `contract-change` PR. A new tap is an additive change and keeps every existing name.

## Names

A tap is one named activation of one forward pass. The model-level taps are `embed.out`, `final_norm.out` and `logits`. Every decoder layer `i` (from 0) has 18 taps named `L{i}.<tap>`. All taps of one forward, in canonical order:

| # | Name | What it is, in Qwen3 terms | Shape `[T, ·]` (tiny / Qwen3-0.6B) | HF capture |
| ---: | --- | --- | --- | --- |
| | `embed.out` | Embedding rows of the input ids | H (64 / 1024) | hook `model.embed_tokens` |
| 1 | `L{i}.input_layernorm.out` | RMSNorm of the layer input | H | hook `…input_layernorm` |
| 2 | `L{i}.attn.q_proj.out` | `x @ Wq^T` | nH·hd (128 / 2048) | hook `…self_attn.q_proj` |
| 3 | `L{i}.attn.k_proj.out` | `x @ Wk^T` | nKV·hd (64 / 1024) | hook `…self_attn.k_proj` |
| 4 | `L{i}.attn.v_proj.out` | `x @ Wv^T` | nKV·hd | hook `…self_attn.v_proj` |
| 5 | `L{i}.attn.q_norm.out` | Per-head RMSNorm of q (over head_dim) | nH·hd | hook `…self_attn.q_norm` |
| 6 | `L{i}.attn.k_norm.out` | Per-head RMSNorm of k | nKV·hd | hook `…self_attn.k_norm` |
| 7 | `L{i}.attn.q_rope` | q after NeoX RoPE | nH·hd | wrapper on `apply_rotary_pos_emb`, output 0 |
| 8 | `L{i}.attn.k_rope` | k after RoPE: what the KV cache stores as K | nKV·hd | wrapper on `apply_rotary_pos_emb`, output 1 |
| 9 | `L{i}.attn.v_cache` | What the KV cache stores as V: raw `v_proj.out` | nKV·hd | hook `…self_attn.v_proj` |
| 10 | `L{i}.attn.o_proj.in` | Attention output, heads concatenated | nH·hd | pre-hook `…self_attn.o_proj` |
| 11 | `L{i}.attn.o_proj.out` | `o_proj.in @ Wo^T` | H | hook `…self_attn.o_proj` |
| 12 | `L{i}.resid_mid` | Layer input + `o_proj.out` | H | pre-hook `…post_attention_layernorm` |
| 13 | `L{i}.post_attention_layernorm.out` | RMSNorm of `resid_mid` | H | hook `…post_attention_layernorm` |
| 14 | `L{i}.mlp.gate.out` | `h @ Wgate^T` | I (128 / 3072) | hook `…mlp.gate_proj` |
| 15 | `L{i}.mlp.up.out` | `h @ Wup^T` | I | hook `…mlp.up_proj` |
| 16 | `L{i}.mlp.down.in` | `silu(gate.out) * up.out` | I | pre-hook `…mlp.down_proj` |
| 17 | `L{i}.mlp.down.out` | `down.in @ Wdown^T` | H | hook `…mlp.down_proj` |
| 18 | `L{i}.out` | `resid_mid + mlp.down.out`: the next layer's input | H | hook `model.layers.{i}` |
| | `final_norm.out` | Final RMSNorm of the last layer's output | H | hook `model.norm` |
| | `logits` | `final_norm.out @ embed^T` (tied lm_head) | V (256 / 151,936) | hook `lm_head` |

`…` stands for `model.layers.{i}.` In the Shape column, H is the hidden size, nH and nKV are the query and KV head counts, hd is the head dim, I is the intermediate size and V is the vocabulary size.

**Counts.** A model with L layers has 18·L + 3 taps: 39 for the tiny model, and 507 for Qwen3-0.6B (28 layers). `common.activation_names(L)` gives them in the order above:
- `embed.out`;
- then, for each layer in turn, its 18 taps in the order 1 to 18;
- then `final_norm.out` and `logits`.

**Grammar.** A name is ASCII and case-sensitive. The `L` prefix is uppercase, and the layer number is decimal with no leading zeros.

## Layout

- **Shape.** Every tap is a row-major 2-D tensor `[T, features]`, where T is the number of tokens in the forward. There is no batch dimension: taps are defined for batch 1.
- **HF head order.** Per-head features are laid out head-major, as HF's projections produce them. Head `h` occupies features `[h·hd, (h+1)·hd)`. This holds for every q-, k- and v-shaped tap, including `q_rope`, `k_rope` and `o_proj.in`.
- **Dtype.** A tap has the dtype of the computation that produced it:
  - F32 for torch32, CpuRef F32 and the GPU backends;
  - F64 for torch64p and CpuRef F64.

  Weights are bf16, widened exactly. They are never taps.
- **Positions.** Row `t` is the token at position `p0 + t`, where `p0` is the number of tokens already in the cache, so `p0` is 0 for a prefill. RoPE taps use those positions.

## Semantics

- **Exact definitions.** `resid_mid` and `L{i}.out` are the sums HF computes. `L0`'s input is `embed.out`, and `L{i}`'s input for `i > 0` is `L{i−1}.out`.
- **The KV cache stores exactly the cache taps.** `k_rope` is K after k_norm and RoPE, and `v_cache` is V raw, both fp32 (CpuReferenceSemantics). The golden tools check bit for bit that both equal the transformers `DynamicCache` contents after the forward.
- **Decode steps.** A single-token decode step has T = 1, and its attention taps attend over the cached rows plus the new one. Golden taps are recorded for prefills. Prefill-equals-decode is checked on our side (M2a, X1), not against goldens.
- **Two-stage splits.** A stage that runs layers a..b emits `L{a}.*` to `L{b}.*`. The first stage also emits `embed.out`, and the last also emits `final_norm.out` and `logits`.

## Use in reports and telemetry

The tap name is the `tap` attribute of `golden.compare` checks, with `layer` set to `i`, or −1 for the model-level taps. Reports group by layer and by tap kind, the part after `L{i}.`. ToleranceSpec (G2) keys its bounds by tap kind, so a bound for `attn.q_rope` applies to every layer.

## Vectors

The 39 tiny-model names are listed in canonical order in `goldens/tiny/g1/manifest.json` under `taps.names`. Each `fwd_*.p0` and `fwd_*.p1` file holds all 39, plus `input_ids`.

| Tap (tiny, p0, T = 13) | Shape | torch32 vs torch64p relL2 (e_ref) |
| --- | --- | ---: |
| `embed.out` | [13, 64] | 0 (exact bf16 rows) |
| `L0.attn.q_rope` | [13, 128] | 1.6e-7 |
| `L0.attn.k_rope` | [13, 64] | 1.5e-7 |
| `L1.out` | [13, 64] | 4.9e-7 |
| `logits` | [13, 256] | 5.4e-7 |

The largest e_ref on p0 is 7.4e-7, at `L1.mlp.down.out`. The per-tap e_ref values for every tap and prompt are in the manifest under `measurements`; the numbers here are informative, and the manifest is authoritative.
