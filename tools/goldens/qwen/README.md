# Qwen3-0.6B goldens and tolerances (G2)

The tools that build `goldens/qwen3-0.6b/g1`: the full-model goldens for Gate 1 and the ToleranceSpec numbers. They build on O1's `tools/goldens/common.py` (ReferenceOracles), which they import and never modify. What the set holds, file by file, is in [`goldens/qwen3-0.6b/README.md`](../../../goldens/qwen3-0.6b/README.md).

| File | What it does |
| --- | --- |
| `make_qwen.py` | The generator: every stage below, double runs, the manifest, the run log |
| `qwen_corpus.py` | The corpora, the seeded pool, the calibration subset and the long prompts (no model) |
| `qwen_oracles.py` | Model checks, a bf16 calibration model, prefill and tap helpers, row summaries, logits metrics |
| `qwen_tolerances.py` | e_ref per tap kind, the calibration rule and `tolerances.json` |
| `test_qwen.py` | Self-tests of the logic above that need no forward pass (contract vectors, pool, selection) |
| `check_set.fsx` | Reads the set through LRM's F# readers: `Goldens.Open`, `VerifyAll`, `ToleranceSpec.load` (and that `vulkan-vs-d3d12` bounds maxAbs on logits only), CpuRef's RoPE table against the dumped T_spec, and each prompt's files by `role` and `dumps` |
| `cpuref_probe.fsx` | CpuRef F32 and F64 on the gate prompts under the calibrated tolerances: the evidence for the floors (ADR G2-0001) |
| `corpus/` | The committed corpora (below) |

## Running

All Python runs in `LRM_DATA/venv-golden` as `python -I`, offline, through O1's runner; the F# scripts need a Release build (`dotnet build Lrm.slnx -c Release`).

```bash
pwsh tools/goldens/run.ps1 qwen/test_qwen.py           # tool self-tests, about a minute
pwsh tools/goldens/run.ps1 qwen/make_qwen.py           # regenerate goldens/qwen3-0.6b/g1 (and LRM_DATA/golden/qwen3-0.6b/g1 if empty)
pwsh tools/goldens/run.ps1 qwen/make_qwen.py --check   # regenerate in memory; compare every committed byte and data hash
pwsh tools/goldens/run.ps1 qwen/make_qwen.py --quick --out <dir>/goldens/qwen3-0.6b/g1 --data-out <dir>/golden/qwen3-0.6b/g1 [--force]
dotnet fsi tools/goldens/qwen/check_set.fsx            # the set through Goldens.Open, VerifyAll and ToleranceSpec.load
dotnet fsi tools/goldens/qwen/cpuref_probe.fsx --long  # CpuRef against the goldens; writes a parity run log
python -I tools/goldens/verify_goldens.py --data C:/Users/end/dev/lrm-data   # hashes and budgets, data files included
```

- **Cost.** The canonical run takes 1 h 50 min to 2 h at 8 threads on the development machine (i9-13900K, AVX2, MKL, shared with other work): pool 40 to 48 min, calibration 10 to 11 min, selection 2 min, gate 52 to 56 min, long prompts 4 min. Into an empty data folder it writes 6.3 GiB under `LRM_DATA/golden`. The pool's double run and the gate stage's no-cache recomputes dominate. `--quick` (2 candidates per category, 3 calibration prompts, short long prompts) takes about 20 minutes and never writes the canonical folders.
- **The shared data folder.** Every worktree on the machine reads `LRM_DATA/golden/qwen3-0.6b/g1`, and golden content never changes in place (GoldenSet). So a run whose `--data-out` already holds files writes no data. It hashes every data file it produces and compares it with the folder's file. If all are byte-identical, it writes the committed files (`goldens.data_unchanged` in the run log). Otherwise it stops with the list of missing, extra and changed files and writes nothing. `--force` replaces the folder through a rename aside, so a file another process holds open leaves the old folder intact. Use it for a smoke folder or a new version, never for a published set; a changed set is `g<N+1>`.
- **Threads.** Every stage runs at 8 threads and falls back to 1 thread if its double run differs.
- **`--check`** is a same-machine, same-`LRM_DATA` check, like O1's: the manifest records the machine (`env`), and torch's fp32 and fp64 kernels depend on the CPU's code path. It passed for g1 on the generation machine (`goldens/qwen3-0.6b/reports/make-qwen-check/`).

## The stages

1. **model.** `REVISION` equals the toolchain pin `c1899de2…`, every `SHA256SUMS` line matches its file, the toolchain's model and tokenizer hashes equal those lines, the config ties the embeddings, and the checkpoint's `lm_head.weight` holds exactly the bytes of `model.embed_tokens.weight`.
2. **self-tests.** O1's oracle self-tests (`tools/goldens/selftest.py`, on the tiny model), and `qwen_fp64_forward`: every torch64p tap of a 16-token prompt on the real model against a hand-written numpy fp64 forward over weights read straight from `model.safetensors`, relL2 at most 1e-13 per tap. Then the argmax tie probe.
3. **rope.** torch's `inv_freq` bits for head_dim 128 (64 entries) and T_spec for positions 0..8191. Both digests must equal the ReferenceOracles vectors, every entry is recomputed with mpmath at 160 bits, and torch's, numpy's and the C runtime's f64 cos and sin must round to T_spec everywhere; torch32's own fp32 table must differ in exactly the contract's 27,369 cos and 19,437 sin entries.
4. **pool.** torch32 64-step greedy (the manual loop, EOS ignored) on all 200 candidates, with the top-2 margin of every step.
5. **calibration.** On the 12 calibration prompts: every tap in torch32, torch64p and bf16. e_ref = relL2(torch32, torch64p) per tap, with torch64p as the reference, and for logits also max |Δlogit| and the largest per-row KL(P_torch64p ‖ P_torch32). bf16 is recorded for Phase 2 only. `tolerances.json` follows: per tap kind the largest e_ref over prompts and layers, bound = clamp(k·e_ref, floor, 10·floor).
6. **selection.** τ_logit is the largest logits maxAbs bound of any pair, and the margin rule is max(0.05, 4·τ_logit). Gate prompts are pool candidates of at most 900 tokens whose torch32 margin meets the rule at all 64 steps and whose 64 greedy ids hold at least 8 distinct tokens: per category, the first such candidate and the longest, each confirmed by torch64p (same ids, same margin rule). At least 8 are needed, with English, C# and ChatML among them; otherwise the prefix fallback fills in (candidates whose first low-margin step comes at step 16 or later, gated on that prefix).
7. **gate.** Per gate prompt and oracle: the prefill logits (every tap for the two shortest gate prompts of different categories), the manual greedy loop, `generate()` through `common.greedy_generate`, a no-cache recompute, and the teacher-forced summaries of prompt + torch32 greedy ids [0:63].
8. **long.** Two prompts of 1,037 and 3,636 tokens: `final_norm.out` for every row, full logits at the last row of every 64-token block and the last 64 rows, and the per-row summaries for every row.

**Double runs.** The pool and calibration stages run through `common.double_run` (their files are small). The gate and long stages hash every file in two complete runs and keep the second; the agreement checks (`generate()`, the no-cache recompute, the controls) run only in the first. A stage passes when the digests agree at 8 threads, or at 1 thread after a difference.

## The corpora

| File | Contents | Source |
| --- | --- | --- |
| `corpus/csharp.txt` | 49 C# snippets, each after a `//// snippet: <name>` line | written for this repo |
| `corpus/apache-2.0.txt` | the Apache License 2.0 | the `LICENSE` file of Qwen3-0.6B at the pinned revision, byte for byte (SHA-256 `832dd9e0…`, its `SHA256SUMS` line) |
| `corpus/design-doc.md` | the LRM design document | `docs/LRM-design.md` at commit `f480fef`, byte for byte; only prose blocks are used (no headings, no tables) |
| `corpus/multilingual.txt` | 59 lines in 25 languages, `<code>\t<line>` | written for this repo |
| `corpus/chatml.json` | 14 Qwen3 ChatML wrappers, with and without the empty thinking block | written for this repo |

**The pool.** Candidate *k* of a category draws from `DetRng(0x6732, streamOf("qwen3-0.6b-g1/pool/<category>/<k>"))` until a text of 8 to 900 tokens that is not already in the pool comes up. The quotas are csharp 48, license 32, design 32, multilingual 32 and chatml 56. C# draws one snippet, two to four consecutive snippets, or a window of lines; license and design draw one to seven consecutive paragraphs; multilingual draws one to six distinct lines; ChatML draws a wrapper and fills its slot from another category. Token ids are `tokenizers`' `encode(text, add_special_tokens=False).ids` (TokenizerApi: no BOS).

**The calibration subset** is chosen before any model runs: per category, the first candidate below 160 tokens and the first at or above 160, then the two longest candidates of the pool.

## Telemetry and reports

Each run writes a `parity` run log under `LRM_DATA/runs` (`common.RunLog`), which `lrm report <runId>` renders with the parity template and `--template goldens` with the goldens template:

- `golden.compare` for every self-test, every calibration e_ref (pair `torch32-vs-torch64p`, `kind: measurement`, with `tap`, `prompt` and `layer`), the bf16 distances (pair `bf16-vs-torch64p`), and the gate and long prompts' holdout e_ref of the logits;
- `goldens.greedy.min_margin`, `goldens.greedy.min_margin_f32_ulps` and the `goldens.greedy.*` and `goldens.teacher_forced.*` checks per gate prompt and oracle;
- `goldens.pool.min_margin` per candidate and `goldens.pool.eligible` per category; `goldens.tolerance.bound` per pair, tap kind and metric; `goldens.gate.margin_rule`; `goldens.wall_seconds` per stage (no template slot yet);
- `goldens.file_bytes`, `goldens.set_bytes` (with the 10 MiB budget), `goldens.data_bytes`, `goldens.double_run_identical` per stage, `goldens.regen_identical` with `--check`, `goldens.data_unchanged` when the data folder already held files; `goldens.*` spans per stage.

The rendered reports of the canonical run, its `--check` and the CpuRef probe are committed under `goldens/qwen3-0.6b/reports/`.
