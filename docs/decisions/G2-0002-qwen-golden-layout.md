# ADR G2-0002: The Qwen3-0.6B golden set: prompt pool, gate selection and data layout

- **Status:** accepted, 2026-10-18.
- **Owner:** G2.
- **Scope:** `goldens/qwen3-0.6b/g1` and `LRM_DATA/golden/qwen3-0.6b/g1`, built by `tools/goldens/qwen/make_qwen.py`. It binds M2b (gate1-cpu), X1b, I1, I2 and I3, which read these files. The format rules are O1's GoldenSet contract; the numbers are ADR G2-0001.

## Context

The plan (section 6) asks for a seeded pool of at least 200 prompts from committed corpora, torch32 margins for all of them, calibration on a fixed subset of 12, at least 8 gate prompts of at most 900 tokens with a top-2 margin of at least max(0.05, 4·τ_logit) over 64 steps (English, C# and ChatML among them, or a prefix fallback with teacher-forced argmax), long prompts of about 1,000 to at most 4,000 tokens with logit checks only, and per-prompt top-64, logsumexp and teacher-forced argmax. Committed files are at most 10 MiB together and 2 MiB each; everything large lives in `LRM_DATA/golden`.

Several choices are left open: how the pool is drawn, which 12 prompts calibrate, which passing candidates become gate prompts, which two prompts get full taps, what "teacher-forced" runs over, and what a long prompt stores when its full logits are several GB.

## Decision

1. **The pool** is 200 candidates in fixed category quotas, in pool order csharp (48), license (32), design (32), multilingual (32) and chatml (56). Candidate *k* of a category draws from `DetRng(0x6732, streamOf("qwen3-0.6b-g1/pool/<category>/<k>"))` until a text of 8..900 tokens not already in the pool comes up (`qwen_corpus.py`). Ids are `tokenizers`' `encode(text, add_special_tokens=False).ids`, so they are what TokenizerApi's Encode must give.
2. **The calibration subset** is fixed before any model runs, by category and length only: per category, the first candidate below 160 tokens and the first at or above, then the two longest candidates of the pool. It spans 17 to 835 tokens and every category.
3. **Gate prompts.** A candidate is eligible when it has at most 900 tokens, its torch32 margin meets the rule at all 64 steps, and its 64 greedy ids hold at least 8 distinct tokens. The last condition is G2's: the first canonical run picked `csharp-000`, whose continuation is 64 newlines, which passes the margin rule but tests one repeated choice. Per category in pool order, the first eligible candidate and the longest one are taken (then the next by k if torch64p does not confirm one): at most 10 prompts, and at least 8 with English, C# and ChatML among them. torch64p confirms a prompt when its manual loop gives the same 64 ids and the same margin rule holds. The prefix fallback is implemented (first low-margin step at step 16 or later, gated on that prefix), and used only when the full rule leaves fewer than 8 or misses a required category.
4. **Taps** are dumped for the two shortest gate prompts of different categories: all 507 ActivationNames taps in both oracles. Every other gate prompt stores `input_ids` and the prefill `logits` [T, 151,936].
5. **Teacher forcing** runs over `prompt + torch32 greedy ids[0:63]`, so row T−1+s of the forward is greedy step s. `tf_<oracle>.<prompt>` stores, for every row, the argmax (lowest index on ties), the top-2 margin in float64, logsumexp in float64 and the top-64 ids and values, ordered by value and then index with ties resolved exactly. Gate 1's teacher-forced check compares a candidate's argmax with the reference's at every row whose reference margin is at least 2·τ_logit.
6. **Long prompts** are fixed concatenations, not pool draws: `long-design` (design-doc paragraphs from the start, 1,037 tokens) and `long-csharp` (C# snippets in file order, 3,636 tokens). Each stores `final_norm.out` for every row, the full logits only at the last row of every 64-token block and the last 64 rows (`logits.rows`), and the per-row summaries of item 5 for every row. Full logits of `long-csharp` would be 6.6 GB in the two oracles; the stored rows still cover every 64-key block boundary and the end of the prompt, and `final_norm.out` with the top-64 summaries covers every row.
7. **Double runs.** The pool and calibration stages run through `common.double_run`. The gate and long stages hash every file in two complete runs and keep the second; the agreement checks (`generate()`, the no-cache recompute, the EOS control) run in the first only, because they produce no file. A stage falls back to 1 thread if its digests differ at 8.
8. **`--check` is a same-machine check**, as O1 decided for the tiny set: the manifest records the machine (`env`), and both oracles' bits depend on the CPU code path. The manifest records the model by its `LRM_DATA`-relative path, never an absolute one.

## Evidence

- The pool, the subset and the long prompts are deterministic: `pwsh tools/goldens/run.ps1 qwen/test_qwen.py` builds them twice and compares, and checks quotas, lengths, distinct texts and the tokenizer's ids.
- The F# readers accept the set: `dotnet fsi tools/goldens/qwen/check_set.fsx` runs `Goldens.Open`, `VerifyAll` over every repo and data file, `ToleranceSpec.load`, and compares CpuRef's RoPE table with the dumped T_spec bit for bit.
- The canonical run's selection, double runs and wall times are in its report, `goldens/qwen3-0.6b/reports/make-qwen/` (run `20261018-072732-1cc39110`): every stage identical at 8 threads.
- **Across processes.** The first canonical run (run `20261018-053140-09d91adb`, before the distinct-token rule replaced `csharp-000` with `csharp-002`) and the final one share 43 files, and all 43 are byte-identical: `pool.safetensors`, `calibration.json`, `rope.safetensors`, both long prompts' files, and the `fwd_*` and `tf_*` files of the nine shared gate prompts. The pool, calibration and long stages have the same double-run digests in both.

## Consequences

- **Consumers read prompts from the manifest.** `prompts` lists each gate and long prompt with its ids, role, category, `dumps` (`taps`, `logits` or `long`) and `gated_steps`. `greedy.safetensors` holds `<prompt>.ids.<oracle>` and `<prompt>.margin.<oracle>`.
- **A long-prompt check compares the stored rows.** A candidate computes the whole prefill and compares `final_norm.out` on every row and `logits` on `logits.rows`, with the logits bounds of ToleranceSpec.
- **Changing any of this is a new version.** A different pool, subset, rule or layout makes `g2`; `g1` stays until no consumer reads it.

## Alternatives

- **Full logits for long prompts:** 6.6 GB for `long-csharp` alone, and a 4,000-row comparison per check. The stored rows plus `final_norm.out` give the same coverage of positions in 268 MB, 4% of the size.
- **Gate prompts by best margin:** picks the most predictable texts (license boilerplate) and few long ones. The first and the longest eligible candidate per category keep both short and long prompts in every category.
- **Calibrating on the gate prompts:** couples the tolerances to the selection that uses them. The fixed subset is chosen without any model output; three of the ten g1 gate prompts happen to be in it too (`design-001`, `chatml-000`, `chatml-027`), and the other seven and both long prompts are holdouts, whose e_ref the manifest records under `measurements.holdout`.
