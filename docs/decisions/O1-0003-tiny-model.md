# O1-0003: The tiny model's shape, weights and prompts

Status: accepted (O1, 2026-10-18). Affects: GoldenSet (`goldens/tiny/g1`), and the tiny-model CI paths of O2, M2a, X1a/X1b and C2a.

## Context

The tiny model is what CI checks on every PR, since the real weights stay local. It must:
- exercise every code path of the real Qwen3-0.6B: GQA, per-head q/k norms, an explicit head_dim whose query width differs from the hidden size, tied embeddings with no lm_head tensor, bf16 weights, and the real config schema;
- be small enough that all of its goldens fit in 5 MiB, together with the tokenizer corpus;
- give greedy decoding that is a real test: many distinct tokens, and top-2 margins far above fp32 and GPU error.

## Decision

**Shape.**
- 2 layers, so a two-stage split has one layer per stage.
- Hidden 64; 4 query heads and 2 KV heads of head_dim 32, so the query width is 128 and differs from the hidden size; intermediate 128; vocabulary 256.
- max_position_embeddings 512 and rope_theta 1e6.
- Every feature width is a multiple of 64, like the real model's. T is arbitrary.

**Weights.**
- Drawn from Lrm.Core's DetRng with an exact Irwin–Hall normal, rounded to float32, then to bf16. F# can regenerate them without torch (recipe in `golden-set.md`).
- Scales: embedding 0.1, residual-branch outputs (o_proj and down) 2/√fan_in, final norm 4. At HF-like scales the tied embedding made greedy decoding copy the last token forever; these scales give 31 distinct tokens in p0's 64 greedy steps.

**Prompts.**
- p0 has 13 tokens and p1 has 1 (decode-shaped), both with full taps.
- p2 has 70 tokens and logits only. It crosses the 64-key block boundary of the GPU attention kernel, and the greedy runs from p0 reach position 76.
- A greedy prompt is the first seeded candidate with a top-2 margin of at least 0.05 at all 64 steps. 0.05 is the floor of the plan's gate-prompt rule, max(0.05, 4·τ_logit), and about 1% of the logit spread.

**EOS.** `eos_token_id` is a token that p0's greedy output emits (O1-0001).

**Phase 2 oracles.** Gradients in torch32 and torch64p, and one AdamW step, are committed now. They cost 2.7 MiB, which still fits the budget, and Phase 2 can start without a new golden version.

## Consequences

- **Budget.** The tiny set is 3.76 MiB, the tokenizer set 0.93 MiB and the README 6 KB: 4.70 MiB in all, under 5 MiB. A larger tiny model, or more taps prompts, would not fit. A new version would have to drop the AdamW moments first.
- **Margins.** The minimum margins are 0.057 (p0) and 0.098 (p1). The logits have a standard deviation of about 4, so a GPU logit error at the GPU floor (max |Δlogit| 1e-2) stays below every margin.
- **What the tiny model does not exercise.** It does not reach the real model's long-context positions. G2's full-model goldens and T_spec 0..8191 cover those.
