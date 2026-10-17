# Goldens

Reference outputs that LRM's own code is checked against. Each set lives in `goldens/<set>/g<N>/` with a `manifest.json` in the `lrm.golden/1` schema (GoldenSet contract, [`docs/contracts/golden-set.md`](../docs/contracts/golden-set.md)). A set never changes in place: any change to its bytes is a new `g<N>`.

| Set | What | Owner | Generator | Committed |
| --- | --- | --- | --- | --- |
| `tiny/g1` | A 2-layer Qwen3 in the real config schema, with torch32 and torch64p forward taps, greedy ids, T_spec, gradients and one AdamW step | O1 | `tools/goldens/make_tiny.py` | everything, 3.8 MiB |
| `tokenizer/g1` | The Qwen3 tokenizer corpus (5,313 cases) with ids from HF tokenizers | O1 | `tools/goldens/make_tokenizer.py` | everything, 0.9 MiB |
| `qwen3-0.6b/g1` | Full-model goldens and tolerances | G2 | `tools/goldens/qwen/` | small files; large ones in `LRM_DATA/golden` |

Committed files are at most 2 MiB each. O1's sets and this README together stay under 5 MiB. Large goldens live under `LRM_DATA/golden/` and are listed in the manifest with `"location": "data"`. Nothing uses Git LFS or release assets.

## The oracles

The ReferenceOracles contract ([`docs/contracts/reference-oracles.md`](../docs/contracts/reference-oracles.md)) defines both oracles:

- **torch32, the spec.** Unpatched transformers 5.20.1 Qwen3 in fp32 with eager attention.
- **torch64p, the truth.** The same model in `.double()`, with exactly three recorded patches:
  1. RMSNorm stays in the input dtype;
  2. softmax runs in `query.dtype`;
  3. RoPE returns T_spec widened to fp64.

  A plain `.double()` run is not fp64: HF runs RMSNorm, softmax and RoPE in fp32 inside it. The self-tests measure this, at about 6e-8 relL2 for one RMSNorm.
- **T_spec.** The RoPE table that torch64p and CpuRef share. It is built from torch's own fp32 `inv_freq` bits, with `freq = f32(f32(pos) * inv_freq)` and `cos, sin = f32(f64 cos, sin)`. The f64 values are correctly rounded, so the table does not depend on any libm.
- **Taps.** Taps are captured by module forward hooks and pre-hooks, plus one wrapper on `modeling_qwen3.apply_rotary_pos_emb` for `q_rope` and `k_rope`. The names follow ActivationNames ([`docs/contracts/activation-names.md`](../docs/contracts/activation-names.md)).

## The tiny set

The tiny set is `goldens/tiny/g1`. Its consumers are O2 (CpuRef per op), M2a (forward and greedy), X1a/X1b and C2a (the tiny-model CI paths), and Phase 2 (gradients and AdamW).

| File | Contents |
| --- | --- |
| `model/config.json` | The real Qwen3-0.6B key set (`rope_theta`, `torch_dtype`, explicit `head_dim`): 2 layers, hidden 64, 4 query and 2 KV heads of head_dim 32, intermediate 128, vocabulary 256, tied embeddings, `eos_token_id` 7 |
| `model/model.safetensors` | 24 bf16 tensors in HF names; no `lm_head` (tied) |
| `model/REVISION`, `model/SHA256SUMS` | `local:tiny-g1`, and `sha256sum` lines for the three other files |
| `fwd_f32.<p>.safetensors` | torch32 `input_ids` (I32) and all 39 taps in F32, for prompts p0 (T=13) and p1 (T=1); only `logits` for p2 (T=70, which crosses a 64-key block) |
| `fwd_f64p.<p>.safetensors` | The same taps from torch64p, in F64 |
| `rope.safetensors` | `inv_freq.bits` (U32 [16]), `t_spec.cos` and `t_spec.sin` (F32 [512, 16]) |
| `greedy.safetensors` | `<p>.ids.<oracle>` (I32 [64]) and `<p>.margin.<oracle>` (F64 [64], top-1 minus top-2 logit), for p0 and p1 |
| `grads_f32.safetensors`, `grads_f64p.safetensors` | `loss` and the gradient of every parameter, for the mean next-token cross-entropy on p0 |
| `adamw_f32.safetensors` | One `torch.optim.AdamW` step (foreach=False) from the bf16 weights with the torch32 gradients. Holds `param/<name>`, `exp_avg/<name>` and `exp_avg_sq/<name>` |
| `manifest.json` | Generator hashes, environment, guards, patches with diffs, tap methods, EOS setting, threads, double run, tie probe, self-tests, prompts, greedy checks, measured e_ref, and every file's size and SHA-256 |

**How the weights are drawn.** The weights come from Lrm.Core's DetRng, so F# can regenerate them bit for bit. The recipe is in `tools/goldens/tiny_model.py` and the manifest.

**How the prompts are chosen.** Each greedy prompt is the first seeded candidate whose top-2 margin is at least 0.05 at every one of the 64 steps.

**Why EOS is token 7.** `eos_token_id` is a token that greedy decoding of p0 first emits at step 9. A decoder that stops at EOS therefore cannot match the golden, because every golden run ignores EOS.

## The tokenizer set

`goldens/tokenizer/g1/cases.jsonl` is the corpus M1 must reproduce exactly. The oracle is HF tokenizers 0.23.3 on the pinned `tokenizer.json` (SHA-256 `aeb13307…`, revision `c1899de2…`).

Each line is one JSON object, with sorted keys and ASCII-only escapes. Every case has `id` (unique) and `cat` (its category), plus the fields for its `kind`:

| `kind` | Fields | Meaning |
| --- | --- | --- |
| `encode` | `text`, `ids` | `ids = Encode(text)`, with no BOS |
| `encode_bytes` | `bytes` (hex), `text`, `ids` | Raw bytes, often invalid UTF-8. `text` is the bytes decoded with one U+FFFD per maximal invalid subpart, and `ids = Encode(text)` |
| `decode` | `ids`, `bytes` (hex), `text` | `bytes` is the concatenated token bytes (DecodeBytes). `text = Decode(ids)`, the bytes decoded with errors=replace |
| `encode_reject` | `text`, `error` | The text holds a lone surrogate, and Encode must raise |

The categories are:
- C# and IL: every snippet, line and three-line window, with LF, CRLF, CR, tab and two-space variants;
- English prose and multilingual lines: de, zh, ja, ko, ru, el, ar, he, hi, th, vi, tr and more, with their NFD forms;
- whitespace: space, tab and newline runs, CRLF, lone CR, and every Unicode White_Space character (U+0085, U+2028 and others) plus look-alikes that are not whitespace;
- digits, including non-ASCII and astral digits;
- contractions in every case variant, and case-fold probes next to an apostrophe (U+017F, U+0130, U+212A and others);
- emoji, astral letters, and letters added in Unicode 15 and 16;
- every one of the 26 added tokens in about 20 contexts, plus ChatML;
- NFC probes, every code point up to U+00FF, invalid UTF-8, lone surrogates and long inputs;
- 395 seeded mixtures, and 495 decode cases, including every single-byte token.

`manifest.json` records:
- counts per kind and category;
- the no-BOS check on every encode;
- `decode(encode(text)) == NFC(text)` for every encode;
- a second opinion from transformers' own `Qwen2Tokenizer`, which agrees on all 4,810 encode cases.

## Regenerating and checking

All tools run in `LRM_DATA/venv-golden` as `python -I`, offline. `tools/goldens/run.ps1` sets this up:

```bash
pwsh tools/goldens/run.ps1 make_tiny.py --check        # regenerate in memory, compare with the committed bytes
pwsh tools/goldens/run.ps1 make_tokenizer.py --check
pwsh tools/goldens/run.ps1 selftest.py                 # the oracle self-tests on their own
python -I tools/goldens/verify_goldens.py              # hashes, sizes, budgets; standard library only
pwsh tools/goldens/run.ps1 make_tiny.py                # rewrite goldens/tiny/g1 (only when making a new g<N>)
```

- **Double run.** Each generator runs its whole computation twice with 8 threads and compares the digests. If they differ, it falls back to one thread.
- **Regeneration.** `--check` regenerates every byte in a fresh process and compares it with the committed tree.
- **Run log.** Each run also writes a run log to `LRM_DATA/runs/<runId>/events.n0.jsonl`. It holds `golden.compare` checks for every self-test and every torch32-vs-torch64p e_ref, so T1's reports can chart them.

## Reading the files

- **safetensors.** Every `.safetensors` file is deterministic: `__metadata__` first, tensors sorted by name, compact JSON, and the header padded with spaces to 8 bytes. This is the same convention as the Qwen3-0.6B checkpoint, whose header the self-test rebuilds byte for byte. The metadata (`lrm.golden`, `oracle`, `prompt`) is informative.
- **Taps.** Taps are row-major `[T, features]` in HF head order.
- **bf16.** The model's bf16 tensors are raw `uint16` bits, widened exactly to f32.
