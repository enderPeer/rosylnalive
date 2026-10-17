# GoldenSet contract

Owner: O1 owns the format, the tiny set and the tokenizer set. G2 owns the `qwen3-0.6b` content. Status: v1, frozen at the contracts-v1 tag.

Consumers:
- O2: `Goldens.open`, CpuRef checks and the safetensors reader;
- M1: the tokenizer corpus;
- M2: the tiny and full-model parity;
- X1, C2 and I1: the tiny and full-model checks on GPUs and clusters;
- G2: writes in this format.

Code: the writers are in `tools/goldens/common.py`, and the stdlib-only verifier is `tools/goldens/verify_goldens.py`.

After contracts-v1, this text changes only through a `contract-change` PR. Golden content never changes in place: any change to a set's bytes makes a new version `g<N+1>`.

## Layout

```
goldens/
  README.md
  <set>/g<N>/manifest.json            schema lrm.golden/1
  <set>/g<N>/<files listed in the manifest>
  <set>/g<N>/model/                   only for a set that ships a model
LRM_DATA/golden/<set>/g<N>/<files>    large files, listed in the manifest with location "data"
```

- **Names.** `<set>` is a lowercase safe name (`tiny`, `tokenizer`, `qwen3-0.6b`). `g<N>` counts from `g1`.
- **Versions coexist.** A new version is a new folder next to the old one. The old one stays until no consumer reads it.
- **What lives in the repo.** Committed files are at most 2 MiB each (2,097,152 bytes). The committed size of a set is capped by its budget: `tiny` 5 MiB, `tokenizer` 2 MiB, `qwen3-0.6b` 10 MiB. O1's two sets plus `goldens/README.md` stay under 5 MiB together; v1 uses 4.70 MiB.
- **What lives in LRM_DATA.** Anything larger goes under `LRM_DATA/golden/` and is never committed. The repo uses no Git LFS and no release assets (AP-7 declined).
- **Line endings.** Text files are LF with a final newline. Binary files are `.safetensors`, which `.gitattributes` marks binary.

## The manifest

`manifest.json` is JSON with sorted keys, 2-space indent, ASCII-only escapes, shortest round-trip floats and a final LF. It contains no timestamp, user name or other run-dependent value, so regeneration reproduces it byte for byte on the same machine and software.

**Required for every set:**

| Field | Contents |
| --- | --- |
| `schema` | `"lrm.golden/1"` |
| `set`, `version` | The folder names, for example `"tiny"` and `"g1"` |
| `kind` | `"tiny-model"`, `"tokenizer-corpus"` or `"full-model"` (G2) |
| `about` | One paragraph: what the set is and who consumes it |
| `generator` | `command` (how to regenerate) and `files`: the SHA-256 of every generator source, in repo paths, with CRLF folded to LF |
| `env` | The environment snapshot (ReferenceOracles contract): versions, venv lock hash and match, HF source hashes, torch config hash, CPU capability, OS, machine, processor, threads, deterministic flag |
| `guards` | The effective guard state: isolation, environment variables, offline mode, hub kernels |
| `model` | `path`, `revision` and `files` (SHA-256 per model file). For the tokenizer set, the tokenizer's location, revision and SHA-256 |
| `threads` | The thread count that produced the files |
| `double_run` | `threads`, `identical`, `digest` and every `attempts` entry |
| `eos` | `ignored: true`, and how each greedy path ignores EOS |
| `files` | One entry per file except the manifest itself, sorted by path: `path` (repo-relative, or relative to `LRM_DATA/golden` for data files), `location` (`"repo"` or `"data"`), `bytes`, `sha256` and `role` |

**Required for model sets (`tiny-model`, `full-model`):**

| Field | Contents |
| --- | --- |
| `oracles` | `torch32` and `torch64p`: role, dtype, attention implementation, load path, and `patches`, the three ReferenceOracles patch records with id, target, kind, source hashes and diff |
| `rope` | `theta`, `head_dim`, `max_pos`, `inv_freq_bits` (hex u32), the T_spec definition, and the T_spec statistics and libm cross-check |
| `taps` | `names` (ActivationNames order), `methods` (hook, prehook or wrap, with module or function), `tap.method: "wrap"`, the wrapper, the shapes rule, and `cache_check` per prompt and oracle |
| `tie_probe` | `lowest_index_on_ties` and `cases` |
| `selftests` | Every self-test record: `passed`, `metric`, `measured`, `bound` |
| `prompts` | `name`, `length`, `ids`, `dumps` (`"taps"` or `"logits"`), plus how each prompt was chosen |
| `greedy` | Steps, the margin rule, and for every prompt and oracle the agreement checks (manual loop = `generate()` = no-cache), the minimum margin and the effective `generate()` config |
| `measurements` | e_ref: relL2(torch32, torch64p) per prompt and tap, and per gradient tensor |

`grads` and `adamw` describe the Phase 2 oracle files when a set has them. G2 adds its own fields, such as candidate pools, calibration and the `tolerances.json` reference. Readers must ignore fields they do not know.

## File formats

### safetensors

Every tensor file is safetensors as written by `common.safetensors_bytes`. The format is deterministic:

- **Header length.** An 8-byte little-endian header length N, with N a multiple of 8.
- **Header JSON.** Compact, ASCII. `"__metadata__"` comes first when present (string to string), then one entry per tensor in code point order of the names: `{"dtype", "shape", "data_offsets"}`. Spaces pad the JSON to N bytes.
- **Data.** The tensor data follows in the same name order, with no gaps and no trailing bytes.
- **Alignment.** In golden files every tensor starts at a multiple of its element size relative to the data start, and the data start is a multiple of 8. The generators check this. Token ids are I64, as torch uses, which also keeps every F64 tensor aligned.
- **Dtypes** are `BF16` (raw u16 bits), `F32`, `F64`, `I32`, `I64`, `U32` and `U8`, little-endian. The HostTensorAndSafetensors reader (O2) must accept all of them.

This is the convention of the Qwen3-0.6B checkpoint itself: the self-test rebuilds its 35,552-byte header byte for byte. Metadata such as `lrm.golden`, `oracle` and `prompt` is informative; the manifest is authoritative.

### Model folders

A set that ships a model has `model/` with exactly these files:

| File | Contents |
| --- | --- |
| `config.json` | The real Qwen3 key set, as in Qwen3-0.6B: `architectures`, `attention_bias`, `attention_dropout`, `bos_token_id`, `eos_token_id`, `head_dim` (explicit), `hidden_act`, `hidden_size`, `initializer_range`, `intermediate_size`, `max_position_embeddings`, `max_window_layers`, `model_type`, `num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`, `rms_norm_eps`, `rope_scaling`, `rope_theta` (top-level), `sliding_window`, `tie_word_embeddings`, `torch_dtype`, `transformers_version`, `use_cache`, `use_sliding_window`, `vocab_size`. Sorted keys, 2-space indent |
| `model.safetensors` | Every parameter in BF16 under its HF name, with `__metadata__ {"format": "pt"}`. There is **no `lm_head.weight`**: `tie_word_embeddings` is true, and the head is the embedding |
| `REVISION` | `local:<set>-<version>` and a newline, for example `local:tiny-g1` |
| `SHA256SUMS` | `sha256sum` binary-mode lines, `<hex> *<name>`, sorted by name, covering every other file in the folder |

**Loader rules.** Qwen3Weights.open (M2) accepts a `local:` revision verified through `SHA256SUMS` only. A Hugging Face revision is verified against the pinned hashes. Qwen3Config.load accepts `rope_theta` at the top level and `rope_parameters.rope_theta`.

### The tokenizer corpus

`cases.jsonl` holds one JSON object per line. Keys are sorted, JSON is compact, and non-ASCII characters are escaped (`\uXXXX`, with surrogate pairs for astral characters), so no raw CR, U+0085, U+2028 or U+2029 appears in the file. The file ends with an LF.

| `kind` | Fields | Expected behaviour |
| --- | --- | --- |
| `encode` | `text`, `ids` | `Encode(text)` = `ids` |
| `encode_bytes` | `bytes` (lowercase hex), `text`, `ids` | `text` is `bytes` decoded as UTF-8 with one U+FFFD per maximal invalid subpart, the Unicode recommended practice that Python `errors="replace"` and .NET `UTF8Encoding` both follow. `Encode(text)` = `ids` |
| `decode` | `ids`, `bytes` (hex), `text` | `DecodeBytes(ids)` = `bytes`, and `Decode(ids)` = `text`, the bytes decoded with errors=replace |
| `encode_reject` | `text`, `error` | `Encode(text)` raises, because the text holds a lone surrogate |

Every case has `id` (`<category>-<4 digits>`, unique) and `cat`. Inputs are unique per kind. M1's cpu-tier test runs every line and emits one `tokenizer.case` check per case.

## The tiny set, `goldens/tiny/g1`

| | |
| --- | --- |
| Model | Qwen3, 2 layers, hidden 64, 4 query heads and 2 KV heads, head_dim 32, intermediate 128, vocabulary 256, max_position_embeddings 512, rope_theta 1e6, rms_norm_eps 1e-6, tied embeddings, bos 0, eos 7 |
| Prompts | p0 (13 tokens) and p1 (1 token) with full taps; p2 (70 tokens, across a 64-key block) with logits only |
| Greedy | 64 steps from p0 and p1 in both oracles, all identical; minimum margins 0.057 (p0) and 0.098 (p1) |
| Files | `fwd_f32.<p>`, `fwd_f64p.<p>`, `rope`, `greedy`, `grads_f32`, `grads_f64p`, `adamw_f32` (all `.safetensors`) and `model/` |

### The weight recipe

Every tensor `name` of shape `S`, with `n` elements, is drawn as follows:

```
rng    = DetRng(seed = 1, stream = DetRng.streamOf("tiny-g1/" + name))
z[k]   = (Σ_{j=0..11} NextF32 draw 12k+j) − 6         the partial sums run in order j = 0..11, exact in float64
value  = scale · z[k]                                   matrices and the embedding
       = scale · (1 + z[k] / 4)                         RMSNorm weights
bits   = Bf16.narrowRne(float32(value))                 float64 → float32 rounds to nearest even, then bf16 RNE
```

The scales are:

| Tensors | scale |
| --- | --- |
| embedding | 0.1 |
| q, k, v, gate and up | 1/√fan_in |
| o_proj and down | 2/√fan_in |
| `model.norm` | 4 |
| other norms | 1 |

The manifest lists each tensor's kind and scale.

### The prompt and EOS rules

- **Prompts.** Candidate `k` of prompt `p` is `NextInt(1, 256)` drawn `length` times from `DetRng(1, streamOf("tiny-g1/prompt/<p>/<k>"))`. A greedy prompt is the first `k` whose torch32 top-2 margin is at least 0.05 at all 64 steps: p0 is candidate 1, and p1 is candidate 3. p2 is candidate 0.
- **EOS.** `eos_token_id` is the first token of p0's greedy output that is first emitted at step 8 or later and is not the BOS id. That is token 7, at step 9. A decoder that stops at EOS cannot reproduce the 64-step golden.

### Files

| File | Tensors |
| --- | --- |
| `fwd_f32.<p>.safetensors`, `fwd_f64p.<p>.safetensors` | `input_ids` (I64 [T]) plus the 39 ActivationNames taps (F32 or F64 `[T, features]`); p2 has only `logits` |
| `rope.safetensors` | `inv_freq.bits` (U32 [16]), `t_spec.cos` and `t_spec.sin` (F32 [512, 16]) |
| `greedy.safetensors` | `<p>.ids.<oracle>` (I64 [64]) and `<p>.margin.<oracle>` (F64 [64]), for p in {p0, p1} and oracle in {torch32, torch64p} |
| `grads_f32.safetensors`, `grads_f64p.safetensors` | `loss` ([1]) and one gradient per HF parameter name, for p0's mean next-token cross-entropy |
| `adamw_f32.safetensors` | `param/<name>`, `exp_avg/<name>` and `exp_avg_sq/<name>` after one AdamW step |

### Vectors

**Model files** (SHA256SUMS):

| File | SHA-256 |
| --- | --- |
| `config.json` | `b4a2dd5f6c4c344f9d1988dc462ff55f0ad54e4bef3a000aef706e086fa97834` |
| `model.safetensors` | `7ade5616bee29bf1008e4a2902cd7aa6b334e3571aa978356d7adfe08e058b2e` |
| `REVISION` | `a9695526b61bb8b86a031c47688d03850c0e5a8a6001ede3c3bcff3a837ed793` |

**Weight recipe.**
- `streamOf("tiny-g1/model.embed_tokens.weight")` = `81bccdbccf056b3c`.
- Its first words under seed 1 are `f488ed5f 5d82fb81 a9e60103 062a29e8`.
- Its first z is −1.31034315…

**First bf16 bits:**
- `model.embed_tokens.weight[0, 0..3]` = `be06 3cb6 3db7 3d27`;
- `model.norm.weight[0..3]` = `4029 4030 4067 4065`;
- `model.layers.0.self_attn.q_proj.weight[0, 0..3]` = `bdfa be8c be8a 3d62`.

**Prompts.**
- p0 = `52 134 173 130 236 105 188 194 155 227 174 203 226`;
- p1 = `151`.

**Other files.** The manifest records the SHA-256 of every other file.

## The tokenizer set, `goldens/tokenizer/g1`

- **Corpus.** 5,313 cases in 21 categories: 4,669 encode, 141 encode_bytes, 495 decode and 8 encode_reject.
- **Oracle.** tokenizers 0.23.3 on `tokenizer.json`, with SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Manifest.** It records the counts, the 26 added tokens with ids and special flags, the real vocabulary size of 151,669, the normalizer, pre-tokenizer and decoder JSON, and these checks:
  - no BOS on every encode;
  - `decode(encode(text)) == NFC(text)` on every encode;
  - every decode equals errors=replace;
  - every reject raised;
  - transformers' `Qwen2Tokenizer` agrees on all 4,810 encode and encode_bytes cases.
- **Digest.** `cases.jsonl` is 964,352 bytes with SHA-256 `33b0dbd14d9138bef7e2edaedfdb9e13eecf5c0bcccba3b21682ff401184fb67`.

## Verification

- **Double run.** Each generator runs twice in-process and compares output digests, at 8 threads with a 1-thread fallback (ReferenceOracles).
- **`--check`.** `make_tiny.py --check` and `make_tokenizer.py --check` regenerate in a fresh process and compare every byte with the committed tree. Both pass for v1 on the generation machine.
- **`verify_goldens.py`.** Uses only the standard library, so any Python works, CI's included. It checks every manifest:
  - schema and folder names;
  - each repo file's size and SHA-256;
  - no unlisted files;
  - the 2 MiB limit;
  - SHA256SUMS and REVISION;
  - the budgets.
- **`Goldens.open`** (O2) verifies the SHA-256 of every file it reads against the manifest before use, and rejects unknown `schema` values.
