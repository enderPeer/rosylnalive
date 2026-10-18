# qwen3-0.6b: full-model goldens and tolerances

The Gate 1 reference for Qwen3-0.6B: what torch32 (the spec) and torch64p (the truth) compute on a fixed set of prompts, and the tolerances every model-level check uses. Owner: G2. Generator: [`tools/goldens/qwen/make_qwen.py`](../../tools/goldens/qwen/README.md). Format: the GoldenSet contract (`docs/contracts/golden-set.md`, schema `lrm.golden/1`, kind `full-model`); oracles: the ReferenceOracles contract. Decisions: [ADR G2-0001](../../docs/decisions/G2-0001-tolerance-calibration.md) (the numbers) and [ADR G2-0002](../../docs/decisions/G2-0002-qwen-golden-layout.md) (prompts and layout).

Consumers: M2b (gate1-cpu), X1b (GPU taps), I1 (per-backend Gate 1, `lrm bisect`, long prompts), I2 (the cluster split), I3 (`lrm gate1`), T1 (the bounds in reports).

## Where the files are

`g1/` holds the small files; everything large is in `LRM_DATA/golden/qwen3-0.6b/g1/`, listed in `g1/manifest.json` with `"location": "data"`, its size and SHA-256. `Goldens.Open` verifies each file before use. The model is the pinned checkpoint `LRM_DATA/models/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` (bf16, tied embeddings; the checkpoint's `lm_head.weight` is byte-identical to `model.embed_tokens.weight`).

**Committed, in `g1/`:**

| File | Contents |
| --- | --- |
| `manifest.json` | Generator hashes, environment, guards, model and tokenizer hashes, the torch64p patches, RoPE facts, tap methods, EOS rules, double runs, tie probe, self-tests, corpus, pool, calibration and gate rules, every prompt with its ids, every greedy check, the measurements, and every file |
| `tolerances.json` | The ToleranceSpec numbers (`lrm.tolerances/1`): seven pairs, bounds per tap kind with their e_ref, and `greedy` with τ_logit, the margin rule and the teacher-forced threshold |
| `calibration.json` | e_ref per calibration prompt and tap: relL2(torch32, torch64p), and for logits also max \|Δlogit\| and the largest per-row KL; the same for a bf16 model (Phase 2 calibration only) |
| `pool.json` | The 200 candidates: category, source, tokens, text SHA-256, smallest torch32 margin, prefix at the margin rule, eligibility |
| `pool.safetensors` | `greedy_ids` (I64 [200, 64]), `margins` (F64 [200, 64]) and `lengths` (I64 [200]), in `pool.json` order |
| `greedy.safetensors` | Per gate prompt and oracle: `<prompt>.ids.<oracle>` (I64 [64]) and `<prompt>.margin.<oracle>` (F64 [64]) |

**In `LRM_DATA/golden/qwen3-0.6b/g1/`:**

| File | Tensors |
| --- | --- |
| `rope.safetensors` | `inv_freq.bits` (U32 [64], head_dim 128), `t_spec.cos` and `t_spec.sin` (F32 [8192, 64]) |
| `fwd_f32.<prompt>`, `fwd_f64p.<prompt>` | `input_ids` (I64 [T]) and the prefill `logits` (F32 or F64 [T, 151936]); for the two taps prompts all 507 ActivationNames taps instead |
| `tf_f32.<prompt>`, `tf_f64p.<prompt>` | Teacher forcing over `input_ids` = prompt + torch32 greedy ids [0:63] (I64 [T+63]): per row `argmax` (I64), `margin` (F64), `lse` (F64), `top64.ids` (I64 [T+63, 64]) and `top64.values` (the oracle's dtype) |
| `long_f32.<prompt>`, `long_f64p.<prompt>` | `input_ids`, `final_norm.out` ([T, 1024], every row), `logits.rows` (I64 [R]) and `logits` ([R, 151936]) at those rows, and the per-row `argmax`, `margin`, `lse` and `top64.*` for every row |

Every file ends in `.safetensors`. Rows of the `tf_*` files: row T−1+s is greedy step s. Top-64 is ordered by value and then index, with ties resolved exactly; argmax is the lowest index of the maximum, as CpuRef's.

## The prompts

All ids are `tokenizers`' `encode(text, add_special_tokens=False).ids` on the pinned `tokenizer.json` (TokenizerApi: no BOS). The manifest's `prompts` lists each gate and long prompt with its ids, category, source and text SHA-256.

| Prompt | Category | Tokens | Dumps | Gated steps | Smallest margin, torch32 | torch64p | float32 ulps (torch64p) | EOS at step | Logits e_ref |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `csharp-002` | csharp | 34 | taps | 64 | 0.0666 | 0.0666 | 58,588 |  | 1.52e-06 |
| `csharp-044` | csharp | 406 | logits | 64 | 0.0904 | 0.0904 | 47,411 |  | 1.12e-06 |
| `license-001` | license | 154 | logits | 64 | 0.0745 | 0.0745 | 39,063 |  | 1.36e-06 |
| `license-008` | license | 510 | logits | 64 | 0.2131 | 0.2131 | 111,708 |  | 2.25e-06 |
| `design-001` | design | 242 | logits | 64 | 0.0578 | 0.0578 | 30,279 |  | 1.18e-06 |
| `design-009` | design | 467 | logits | 64 | 0.1235 | 0.1235 | 64,774 |  | 1.48e-06 |
| `multilingual-002` | multilingual | 61 | taps | 64 | 0.2191 | 0.2191 | 114,862 |  | 1.45e-06 |
| `multilingual-023` | multilingual | 152 | logits | 64 | 0.0802 | 0.0802 | 43,329 |  | 1.42e-06 |
| `chatml-000` | chatml | 248 | logits | 64 | 0.1035 | 0.1035 | 54,281 | 47 | 1.78e-06 |
| `chatml-027` | chatml | 708 | logits | 64 | 0.0826 | 0.0826 | 43,320 |  | 1.28e-06 |
| `long-design` | design | 1037 | long | n/a | 0.0000 (rows) | | | | 1.07e-06 |
| `long-csharp` | csharp | 3636 | long | n/a | 0.0016 (rows) | | | | 1.22e-06 |

The smallest margin is in float64 over the 64 decision rows; the float32 column is that margin in float32 ulps of the top logit (`generate()` takes its argmax over float32). Every gate prompt's 64 torch32 ids hold at least 21 distinct tokens. `chatml-000` emits EOS (`<|im_end|>`) at step 47, which the goldens ignore; with the checkpoint's own generation config, `generate()` stops there (ADR O1-0001). Logits e_ref is relL2(torch32, torch64p) of the prefill logits; the long prompts' smallest margin is over their rows, which are not gated.

- **Gate prompts** come from the pool: at most 900 tokens, a torch32 top-2 margin of at least 0.05 at all 64 greedy steps and at least 8 distinct tokens among the 64 greedy ids, confirmed by torch64p; per category the first such candidate and the longest. All ten pass the full rule, so the prefix fallback is not used.
- **Taps** are dumped for the two shortest gate prompts of different categories: `csharp-002` (34 tokens) and `multilingual-002` (61 tokens).
- **Long prompts** (logits only, single node) are fixed concatenations: `long-design` (design-doc paragraphs) and `long-csharp` (C# snippets).
- **Calibration prompts** (`calibration.json`): `csharp-000`, `csharp-012`, `license-000`, `license-002`, `design-000`, `design-001`, `multilingual-001`, `multilingual-000`, `chatml-002`, `chatml-000`, `design-013` and `chatml-027` (17 to 835 tokens), fixed by category and length before any model ran. Three gate prompts are among them (`design-001`, `chatml-000`, `chatml-027`); the other seven gate prompts and both long prompts are holdouts.

The pool (`pool.json`):

| Category | Candidates | Tokens (min to max) | Eligible | Median smallest margin |
| --- | ---: | --- | ---: | ---: |
| csharp | 48 | 10 to 406 | 15 | 0.0416 |
| license | 32 | 14 to 627 | 8 | 0.0226 |
| design | 32 | 30 to 835 | 7 | 0.0202 |
| multilingual | 32 | 17 to 242 | 14 | 0.0371 |
| chatml | 56 | 20 to 708 | 23 | 0.0305 |

Eligible: at most 900 tokens, a margin of at least 0.05 at all 64 steps, and at least 8 distinct tokens among the 64 greedy ids (67 of 200).

## How Gate 1 uses the set

The pair names and bounds come from `tolerances.json` through `ToleranceSpec.load` (the tap kind is the part of a tap name after `L{i}.`).

- **Logits.** A candidate's prefill logits [T, 151,936] against `fwd_*.<prompt>`: relL2, max |Δlogit| and the largest per-row KL(P_reference ‖ P_candidate), with the pair's `logits` bounds.
- **Taps.** For the two taps prompts, relL2 per tap with the bound of its tap kind.
- **Greedy.** The canonical schedule (one prefill, then single-token decode steps, EOS never checked) gives 64 ids equal to `<prompt>.ids.<oracle>` in `greedy.safetensors` over the prompt's gated steps (all 64 for every g1 prompt).
- **Teacher forcing.** A forward over each `tf_*` file's `input_ids`: the candidate's argmax equals the reference's at every row whose reference `margin` is at least `greedy.teacher_forced_min_margin` (0.02).
- **Long prompts.** `final_norm.out` for every row, `logits` at `logits.rows`, and the teacher-forced argmax over every row with a margin of at least 0.02.
- **CpuRef F64** stays within relL2 1e-10 of torch64p on every tap and on the logits.

## Tolerances

| Pair | Rule | relL2, per-layer and model taps | Logits relL2 | Logits max \|Δlogit\| | Logits KL |
| --- | --- | --- | ---: | ---: | ---: |
| `cpu64-vs-torch64p` | fixed | 1e-10 | 1e-10 | none | none |
| `cpu32-vs-torch32` | k 3; floors 3e-05, 0.002, 1e-07 | 3.00e-05 to 5.91e-05 | 3.00e-05 | 2.00e-03 | 1.00e-07 |
| `cpu32-vs-torch64p` | k 2; floors 3e-05, 0.002, 1e-07 | 3.00e-05 to 3.94e-05 | 3.00e-05 | 2.00e-03 | 1.00e-07 |
| `gpu-vs-cpu32` | k 10; floors 0.0001, 0.01, 1e-06 | 1.00e-04 to 1.97e-04 | 1.00e-04 | 1.00e-02 | 1.00e-06 |
| `gpu-vs-torch32` | k 10; floors 0.0001, 0.01, 1e-06 | 1.00e-04 to 1.97e-04 | 1.00e-04 | 1.00e-02 | 1.00e-06 |
| `gpu-vs-torch64p` | k 9; floors 0.0001, 0.01, 1e-06 | 1.00e-04 to 1.77e-04 | 1.00e-04 | 1.00e-02 | 1.00e-06 |
| `vulkan-vs-d3d12` | fixed | 0.0001 | 0.0001 | 0.01 | none |

- e_ref per tap kind is the largest relL2(torch32, torch64p) over the 12 calibration prompts and 28 layers (ADR G2-0001); every calibrated bound is clamp(k·e_ref, floor, 10·floor) and carries its e_ref.
- The CPU32 relL2 floor is 3e-5, not the plan's 1e-5: CpuRef F32 measured up to 1.4e-5 on the residual stream, where torch32 itself is about 2e-6 from the truth. All other k and floors are the plan's.
- τ_logit = 0.01 (the largest logits maxAbs bound, from the GPU floor), so the gate margin rule is max(0.05, 4·τ_logit) = 0.05 and the teacher-forced threshold 2·τ_logit = 0.02.

## Measured

e_ref per tap kind (the largest over the 12 calibration prompts and 28 layers) and the bounds it gives:

| Tap kind | Metric | e_ref | Where (prompt, layer) | cpu32-vs-torch32 | cpu32-vs-torch64p | gpu-vs-torch32 | gpu-vs-torch64p |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `embed.out` | relL2 | 0.00e+00 | csharp-000 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `input_layernorm.out` | relL2 | 5.80e-06 | csharp-012, L14 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.q_proj.out` | relL2 | 2.18e-06 | csharp-012, L14 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.k_proj.out` | relL2 | 1.96e-05 | csharp-012, L14 | 5.89e-05 | 3.93e-05 | 1.96e-04 | 1.77e-04 |
| `attn.v_proj.out` | relL2 | 1.97e-05 | csharp-012, L14 | 5.91e-05 | 3.94e-05 | 1.97e-04 | 1.77e-04 |
| `attn.q_norm.out` | relL2 | 2.20e-06 | multilingual-001, L12 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.k_norm.out` | relL2 | 2.20e-06 | csharp-012, L14 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.q_rope` | relL2 | 2.20e-06 | multilingual-001, L12 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.k_rope` | relL2 | 2.20e-06 | csharp-012, L14 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.v_cache` | relL2 | 1.97e-05 | csharp-012, L14 | 5.91e-05 | 3.94e-05 | 1.97e-04 | 1.77e-04 |
| `attn.o_proj.in` | relL2 | 9.27e-06 | csharp-012, L14 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `attn.o_proj.out` | relL2 | 1.48e-05 | csharp-012, L14 | 4.45e-05 | 3.00e-05 | 1.48e-04 | 1.33e-04 |
| `resid_mid` | relL2 | 1.51e-06 | license-002, L27 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `post_attention_layernorm.out` | relL2 | 4.57e-06 | csharp-012, L13 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `mlp.gate.out` | relL2 | 1.96e-06 | multilingual-001, L11 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `mlp.up.out` | relL2 | 3.08e-06 | multilingual-001, L12 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `mlp.down.in` | relL2 | 6.28e-06 | csharp-012, L11 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `mlp.down.out` | relL2 | 5.45e-06 | csharp-012, L11 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `out` | relL2 | 2.97e-06 | multilingual-001, L27 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `final_norm.out` | relL2 | 2.73e-06 | license-000 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `logits` | relL2 | 2.12e-06 | license-000 | 3.00e-05 | 3.00e-05 | 1.00e-04 | 1.00e-04 |
| `logits` | maxAbs | 1.70e-04 | chatml-000 | 2.00e-03 | 2.00e-03 | 1.00e-02 | 1.00e-02 |
| `logits` | kl | 5.52e-10 | license-000 | 1.00e-07 | 1.00e-07 | 1.00e-06 | 1.00e-06 |

**CpuRef under these tolerances** (`tools/goldens/qwen/cpuref_probe.fsx`: O2's CpuRef ops composed into the Qwen3 forward as `tests/Lrm.Tensor.Tests/TinyModel.fs` composes them for the tiny model; report in [`reports/cpuref-probe/`](reports/cpuref-probe/parity.md)): all 3,162 comparisons pass, the worst at 0.42 of its bound (CpuRef F32 `L14.attn.o_proj.out` of `csharp-002`, relL2 1.85e-5 against a bound of 4.45e-5). CpuRef F32 and F64 give the oracles' 64 greedy ids on every gate prompt and their argmax on every teacher-forced row with a margin of at least 0.02; CpuRef F64 is within 2.2e-14 of torch64p on every tap. This is G2's calibration evidence, not the gate1-cpu acceptance, which M2b runs with M2a's forward.

**bf16** (calibration only): a bf16 model's logits are 1.7e-2 to 3.0e-2 (relL2) from torch64p on the calibration prompts (`calibration.json`).

## Regenerating and checking

```bash
python -I tools/goldens/verify_goldens.py                                   # committed files: hashes, sizes, budgets
python -I tools/goldens/verify_goldens.py --data C:/Users/end/dev/lrm-data   # and the LRM_DATA files
dotnet fsi tools/goldens/qwen/check_set.fsx                                 # Goldens.Open, VerifyAll, ToleranceSpec.load, T_spec = CpuRef
dotnet fsi tools/goldens/qwen/cpuref_probe.fsx --long                       # CpuRef F32 and F64 under these tolerances
pwsh tools/goldens/run.ps1 qwen/make_qwen.py --check                        # regenerate (about 1 h 50 min) and compare
```

`--check` passes only on the generation machine with the same `LRM_DATA` (GoldenSet, Verification): the manifest records the machine, and torch's kernels depend on its code path. On the generation machine it passed for g1 (run `20261018-095500-e277b974`, 1 h 48 min): every committed byte and every data file's SHA-256 regenerated identically ([`reports/make-qwen-check/goldens.md`](reports/make-qwen-check/goldens.md)).

## Vectors

| What | Value |
| --- | --- |
| `inv_freq` bits for head_dim 128, SHA-256 over u32 little-endian (the ReferenceOracles vector) | `138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e`; the 64 entries are in the manifest under `rope.inv_freq_bits` |
| T_spec 0..8191: SHA-256 of all cos then all sin values, f32 little-endian (the ReferenceOracles vector) | `da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf` |
| `rope.safetensors` | 4,194,896 bytes, SHA-256 `fb6dd54a6bfc1ac217b2899c349ff0146ece80cdf5bba94ea045e5fab6f26c8c` |
| `csharp-002` prompt ids (34) | `262 1667 762 2033 284 2535 2943 78533 25797 11 20251 568 49684 3576 317 262 2033 22834 19098 7188 15872 543 262 470 2535 2033 12614 88136 40583 568 49684 3576 317 92` |
| `csharp-002` greedy ids, steps 0..11, both oracles | `4710 3983 358 2776 537 2704 421 419 374 279 1850 1616` |
| `multilingual-002` greedy ids, steps 0..11, both oracles | `4710 27473 419 1119 6364 382 785 19415 13719 678 5975 11` |
| `tolerances.json` | 24,269 bytes, SHA-256 `e7c8c2cb8c35b54c21e3d128183b4183d7a210d65b1dcbcaa4edadd2f3412f99` |

The manifest lists the size and SHA-256 of every other file.

## Reports

In [`reports/`](reports/): the canonical run with the parity and goldens templates (`make-qwen/`), its `--check` regeneration with the goldens template (`make-qwen-check/`), and the CpuRef probe with the parity template (`cpuref-probe/`).
