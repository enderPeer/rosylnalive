# ReferenceOracles contract

Owner: O1. Status: v1, frozen at the contracts-v1 tag.

Consumers:
- O2: CpuRef F32 and F64, and the comparators;
- G2: the full-model goldens and the tolerances;
- M2: the Qwen model, including RopeConstants;
- Phase 2: gradients and the optimizer.

Code: `tools/goldens/common.py`, an offline tool. Python is never part of the system. Tests: `tools/goldens/selftest.py`, whose results every tiny-golden manifest records.

After contracts-v1, this text and its vectors change only through a `contract-change` PR, with a CHANGELOG entry and updated vectors. The file formats are the GoldenSet contract (`golden-set.md`), and the tap names are ActivationNames (`activation-names.md`).

## The oracles

There are two oracles, both Hugging Face transformers 5.20.1 `Qwen3ForCausalLM` in the pinned venv (`tools/goldens/requirements.lock`), with eager attention.

| Oracle | Role | How it is built | Patches |
| --- | --- | --- | --- |
| **torch32** | The spec: the official reference implementation | `from_pretrained(dir, dtype=float32, attn_implementation="eager")` | none |
| **torch64p** | The truth: exact arithmetic on the spec's own inputs | The same fp32 load, then `.double()` | exactly the three below |

**Why torch64p exists.** A plain `.double()` model is not an fp64 truth. Inside a `.double()` model, HF's Qwen3:
- runs RMSNorm in fp32;
- runs softmax in fp32;
- computes the RoPE tables in fp32 from fp32 angles.

The self-tests measure this. One unpatched fp64 RMSNorm is off from fp64 by relL2 5.5e-8, and one unpatched fp64 attention layer by 7.0e-8. torch64p removes exactly these three fp32 steps. Nothing else changes.

**Load-time assertions.** `common.load_model` fails unless all of these hold:
- the class is `Qwen3ForCausalLM`, and every layer is `Qwen3DecoderLayer`;
- `_attn_implementation` is `eager`;
- `lm_head.weight` shares storage with `model.embed_tokens.weight`, before and after `.double()`;
- the RoPE type is `default`, with `attention_scaling` 1.0;
- the model's `inv_freq` buffer equals the bits that `compute_default_rope_parameters` gives for the same config.

### The three torch64p patches

Patches 1 and 2 rebuild HF's own function from its source with exactly one substitution, so the recorded diff is the whole patch. `Torch64pPatches` is a context manager: it installs the patches and restores the originals on exit. torch32 runs assert that no patch is active.

**1. `rmsnorm-input-dtype`.** `Qwen3RMSNorm.forward` stays in the input dtype.

```diff
     input_dtype = hidden_states.dtype
-    hidden_states = hidden_states.to(torch.float32)
     variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

**2. `softmax-query-dtype`.** `eager_attention_forward` runs softmax in `query.dtype`.

```diff
-    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
+    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query.dtype).to(query.dtype)
```

**3. `rope-t-spec`.** `Qwen3RotaryEmbedding.forward` returns T_spec, defined below, widened to the input dtype. The tables are pinned on the module as `lrm_t_spec`. A position outside the table raises, and so does a module with no table.

For an fp32 input, patches 1 and 2 change nothing. The manifest records each patch by `id`, `target`, the SHA-256 of the original and patched source, and the diff or replacement source.

## T_spec: the RoPE table

T_spec is the one RoPE table that both torch64p and CpuRef use (CpuReferenceSemantics, O2). It is defined for a (theta, head_dim) pair and positions 0 to maxPos-1:

1. **`inv_freq`** is torch's own fp32 vector: `compute_default_rope_parameters(config)`, which is `1.0 / (theta ** (arange(0, head_dim, 2, dtype=float) / head_dim))` in float32. It is pinned as u32 bits. No closed-form F# formula reproduces it: the f64 formula rounded to f32 differs from torch in 24 of the 64 entries for head_dim 128.
2. **`freq[pos, j] = f32(f32(pos) * inv_freq[j])`**: one float32 multiply.
3. **`cos[pos, j]` and `sin[pos, j]`** are `f32(f64 cos(freq))` and `f32(f64 sin(freq))`. Here the f64 value is the correctly rounded cos or sin of the exactly widened angle, and the f32 rounding is round to nearest even.

The table has shape `[maxPos, head_dim/2]`. HF's full row is the half row repeated twice (`cat(row, row)`), which is NeoX rotation.

**The f64 values are correctly rounded.** This keeps T_spec independent of any libm. The tool computes torch's float64 cos and sin. Any entry whose float32 rounding is not certain within ±2 f64 ulps is recomputed with mpmath at 160 bits. The self-test recomputes every tiny entry with mpmath.

**Measured agreement.** In the tiny table and in the full 0..8191 × 64 Qwen table:
- no entry needed mpmath;
- torch's f64, numpy's f64 and the C runtime's `cos`/`sin` all round to T_spec in every entry. The C runtime is Python's `math.cos`, the same UCRT function that .NET's `Math.Cos` calls on Windows.

So CpuRef is expected to match bit for bit on this machine. If it does not, O2 writes an ADR and loads the dumped table, as the plan says.

**What torch32 does instead.** Unpatched torch32 computes cos and sin in float32. Its table differs from T_spec in these entries:

| Table | Entries | cos differs | sin differs |
| --- | ---: | ---: | ---: |
| tiny (head_dim 32, positions 0..511) | 8,192 | 471 | 239 |
| Qwen3-0.6B (head_dim 128, positions 0..8191) | 524,288 | 27,369 | 19,437 |

This difference is part of e_ref, measured honestly.

### Vectors

**`inv_freq` for theta 1,000,000 and head_dim 32** (the tiny model), as u32 bits:
- `3f800000 3ed7e89b 3e361887 3d99940d 3d0186e3 3c5a7bf2 3bb8449c 3b1b690d`
- `3a83126f 39dd1725 393a7753 389d43a4 3804a2b2 375fba4f 36bcb0c1 361f23e4`

Their SHA-256 over u32 little-endian is `7f7aa6f39d2fd0a08c069fbf6d7cf6679288240227d78c053382b884db09abee`.

**`inv_freq` for theta 1,000,000 and head_dim 128** (Qwen3-0.6B):
- The first 8 entries are `3f800000 3f4e4bad 3f263de0 3f05f6ef 3ed7e89b 3eadfcff 3e8c3504 3e61f835`.
- The last 4 are `361f23e4 36003dec 35ceaf79 35a68e4c`.
- The SHA-256 of all 64 entries over u32 little-endian is `138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e`.
- They equal the planning probe `redteam/inv_torch.txt`.

**T_spec entries** (u32 bits):

| head_dim | pos | j | cos | sin |
| ---: | ---: | ---: | --- | --- |
| 32 | 0 | 0 | `3f800000` | `00000000` |
| 32 | 1 | 0 | `3f0a5140` | `3f576aa4` |
| 32 | 1 | 15 | `3f800000` | `361f23e4` |
| 32 | 13 | 3 | `3f0fb02c` | `3f53df36` |
| 32 | 100 | 7 | `3f78d5f0` | `3e708f27` |
| 32 | 511 | 0 | `bef17fe5` | `3f61bbb4` |
| 32 | 511 | 15 | `3f7ffff4` | `3a9ed44f` |
| 128 | 4095 | 0 | `bd871e6c` | `bf7f7136` |
| 128 | 4095 | 63 | `3f7fff27` | `3ba683b4` |
| 128 | 8191 | 1 | `bf7c693c` | `be2add27` |

**Table digests.** Each digest is the SHA-256 of all cos values, then all sin values, as f32 little-endian in row-major order:
- tiny, positions 0..511: `615b420404dd336bbf4509e20dd121daefdcc0cd8039048ca71097b1b6d60801`. This table is committed in `goldens/tiny/g1/rope.safetensors`.
- Qwen3-0.6B, positions 0..8191: `da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf`. G2 dumps this table to `LRM_DATA/golden` and commits the hash.

## Guards

Every golden tool enforces these guards, at import of `common.py` and again inside the libraries:

1. **Isolation.** Python runs as `python -I`, from `LRM_DATA/venv-golden`. `tools/goldens/run.ps1` runs it that way. Tools put their own directory on `sys.path` explicitly.
2. **Environment.** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `USE_HUB_KERNELS=0`, `TOKENIZERS_PARALLELISM=false` and `HF_HUB_DISABLE_TELEMETRY=1`. A missing variable is set, and a conflicting one stops the tool. None of torch, transformers, tokenizers, huggingface_hub or safetensors may be imported before the guards run.
3. **No hub kernels.** The `kernels` package must not be importable, so `use_kernel_forward_from_hub` and `use_kernelized_func` stay no-ops. After import, the tool also checks that:
   - `hub_kernels._kernels_enabled` is false;
   - huggingface_hub reports offline mode;
   - `ALL_ATTENTION_FUNCTIONS` does not register `eager`, so the attention patch cannot be bypassed.
4. **Pinned software.** torch 2.14.1+cpu, transformers 5.20.1, tokenizers 0.23.3 and safetensors 0.8.0. The installed distributions must equal `requirements.lock` exactly, with nothing missing and nothing extra.
5. **Determinism.** `torch.use_deterministic_algorithms(True)` and `torch.manual_seed(0)`. Generation uses 8 threads (see Double runs).

The manifest records the effective guard state, the versions, and the SHA-256 of the HF sources that define the oracle: `modeling_qwen3.py`, `configuration_qwen3.py`, `modeling_rope_utils.py`, `masking_utils.py`, `cache_utils.py` and `generation/utils.py`.

## Self-tests

Each self-test compares an oracle with an independent implementation. The hand-written references are plain numpy in float64, with no torch arithmetic. relL2 is `||a − b|| / ||b||` in float64, with the reference as `b`.

| Self-test | Compared | Bound | Measured (v1) |
| --- | --- | --- | --- |
| `rmsnorm_fp64` | Patched fp64 `Qwen3RMSNorm` vs a hand-written fp64 RMSNorm, at widths 32, 64, 128 and 1024 (rows at 1e-3, 1 and 1e3 scale) | relL2 ≤ **1e-15** | 1.2e-16 |
| `attention_fp64` | Patched fp64 attention (tiny layer 0: q/k/v, per-head q/k norm, T_spec RoPE, causal GQA softmax, o_proj) vs hand-written fp64, at T = 13 and T = 70 | relL2 ≤ **1e-13** | 4.5e-16 |
| `tiny_fp64_forward` | Every torch64p tap of tiny p0 vs a hand-written numpy fp64 forward | relL2 ≤ 1e-13 per tap | 1.6e-15 |
| `rope_t_spec` | The patched rotary output equals T_spec widened, bit for bit; T_spec equals a full mpmath recomputation | exact | exact |
| `rng_kats` | DetRng / Philox4x32-10 and `streamOf` against the LrmCore vectors | exact | exact |
| `bf16_rne` | bf16 narrowing against the LrmCore table and torch's `.to(bfloat16)` on 2^20 random patterns | exact | exact |
| `safetensors` | The writer round-trips through our reader and safetensors 0.8.0, and rebuilds the Qwen3-0.6B header byte for byte (311 entries, data at 35,560) | exact | exact |

**Negative controls.** The RMSNorm and attention self-tests also run without the patches, and the unpatched module must fail the same bound:
- RMSNorm: at least 5.5e-8 unpatched;
- attention: at least 7.0e-8 unpatched.

This shows that the patches are what make torch64p fp64.

**When they run.** `make_tiny.py` runs every self-test before it generates anything, and stops if one fails. `selftest.py` runs them on their own.

## Taps

Taps follow ActivationNames. They are captured during a batch-1 prefill with `use_cache=True`, through:
- **module forward hooks** (`hook`: the module's output);
- **forward pre-hooks** (`prehook`: the module's first positional input);
- **exactly one function wrapper.** It replaces `modeling_qwen3.apply_rotary_pos_emb` for the duration of one forward and records `q_rope` and `k_rope`. A pre-hook on each `self_attn` tells the wrapper which layer is calling. The manifest says `tap.method: wrap`.

**Checks on every forward.**
- The wrapper runs once per layer.
- The tap set equals ActivationNames for the layer count.
- The `logits` tap equals the model output.
- `k_rope` and `v_cache` equal the `DynamicCache` keys and values bit for bit, for every layer. The manifest records this as `cache_check`.

## Greedy decoding and EOS

**EOS is ignored in every golden run.**

- **Manual loop (the canonical schedule).** One prefill with the cache, then single-token decode steps. Each step takes `torch.argmax` of the last row, and the margin is the top-1 minus top-2 logit in float64. The loop never checks EOS.
- **`generate()`.** It gets a fresh `GenerationConfig(do_sample=False, max_new_tokens=64, eos_token_id=None)`. In transformers 5.20.1, `_prepare_generation_config` fills every `None` field of the passed config from `model.generation_config`, which carries the checkpoint's `eos_token_id`. So `eos_token_id=None` alone does not disable EOS. The tool therefore swaps `model.generation_config` for a neutral `GenerationConfig()` during the call and restores it afterwards. It asserts that the effective config has `eos_token_id` None, `do_sample` false and no `min_new_tokens`, and records it. A control run without the swap shows the leak: on the tiny model it stops after 10 tokens, at EOS (`docs/decisions/O1-0001-eos-in-generate.md`).
- **`min_new_tokens`** is never used, because it masks the EOS logits.
- **No-cache recompute.** Each step runs a full forward over the whole sequence with `use_cache=False`.
- **Agreement.** The manual loop, `generate()` and the no-cache recompute must give the same ids over all steps. On the tiny model, torch32 and torch64p must also give the same ids.

## The tie probe

CpuRef's argmax returns the lowest index on ties. The probe checks that `torch.argmax` does the same, so goldens and CpuRef agree on ties. It plants two equal maxima at seeded positions:
- in rows of length 2, 7, 256, 4097 and 151,936;
- in float32 and float64;
- 4 times each, 40 cases in all, at the generation thread count.

Every case must return `min(a, b)`. The manifest records `tie_probe` and each greedy step's margin, and the tiny prompts are chosen with margins of at least 0.05, so no golden step is a tie.

## Double runs

Each generator runs its whole computation twice with 8 threads, hashes every output file, and compares the digests. If they differ, it repeats at 1 thread and requires that pair to agree. If 1 thread also disagrees, generation stops. The manifest records the threads used, the digest and every attempt.

Regeneration in a fresh process (`--check`) must reproduce every committed byte. On the generation machine (AVX2, MKL, 8 threads), torch32 is bit-reproducible run to run, so both checks pass in v1.

## Gradients and the optimizer (Phase 2 oracles)

- **Loss.** `mean over t < T−1 of cross_entropy(logits[t], ids[t+1])`, computed with `F.cross_entropy` in the model's own dtype. HF's built-in `loss_function` upcasts logits to float32, which would put a float32 step inside torch64p, so it is not used.
- **Gradients.** These are `param.grad` after one backward of that loss, for every named parameter. The tied embedding gets both the lookup and the lm_head contributions.
- **AdamW.** One `torch.optim.AdamW` step with `foreach=False`, `fused=False`, `amsgrad=False` and `maximize=False`, from zero state. The master weights are float32, the bf16 weights widened. Hyperparameters: lr 1e-3, betas (0.9, 0.95), eps 1e-8, weight_decay 0.1. The golden records each parameter after the step, `exp_avg` and `exp_avg_sq`.

## Telemetry

The tools emit TelemetryFacade records to `LRM_DATA/runs/<runId>/events.n0.jsonl`, with the natural RunLog mapping and keys `v, t, k, n, d, x, u, node, seq, a`. T1's RunLog contract is authoritative once merged. The records are:
- one `golden.compare` Check per self-test, with `metric`, `measured`, `bound` and `check`;
- one `golden.compare` per torch32-vs-torch64p e_ref measurement, with `pair`, `tap`, `prompt`, `layer` and `kind: measurement`, or `tensor` and `kind: gradient` for gradients;
- `goldens.*` spans and gauges for phases, sizes, margins and case counts.

Timestamps go only to the run log, never into goldens.

## Using common.py (G2 and later tools)

| Function | Purpose |
| --- | --- |
| `setup_torch(threads)`, `check_effective_guards()`, `env_snapshot(threads)` | Guards and the environment record |
| `load_model(dir, "torch32" \| "torch64p", t_spec_tables)` | The oracles, with the load-time assertions |
| `Torch64pPatches()` | The patches, as a context manager; `.record()` returns the manifest entries |
| `torch_inv_freq_bits(config)`, `t_spec(bits, max_pos)`, `t_spec_crosscheck(...)` | RoPE |
| `forward_taps(model, ids)` | ActivationNames taps and the cache check |
| `greedy_manual`, `greedy_generate`, `greedy_no_cache`, `greedy_generate_without_neutral_config` | Greedy and the EOS control |
| `tie_probe()`, `double_run(build)` | Tie probe and double runs |
| `cross_entropy_grads(model, ids)`, `adamw_step(params, grads)` | Phase 2 oracles |
| `safetensors_bytes`, `read_safetensors`, `json_bytes`, `file_entry`, `sha256sums_bytes`, `compare_tree` | GoldenSet writers |
| `DetRng`, `stream_of`, `bf16_narrow_rne` | Lrm.Core-compatible random numbers and bf16 |
| `RunLog` | Telemetry |

G2's tools live in `tools/goldens/qwen/`. They import `common.py` from the parent folder and never modify it; a change to it is an O1 change.
