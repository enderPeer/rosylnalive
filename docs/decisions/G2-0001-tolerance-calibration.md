# ADR G2-0001: Tolerance calibration for Qwen3-0.6B, and a higher CPU32 relL2 floor

- **Status:** proposed, 2026-10-18. G2 has calibrated the numbers. Two of them depart from a value fixed upstream: the CPU32 relL2 floor (item 3) differs from the plan, and the `vulkan-vs-d3d12` maxAbs bound (item 5) differs from the contract's fixed-bound line. The ADR becomes accepted when the user and the integrator sign off on both at the plan's "tolerances reviewed" gate (wave 2), and the integrator records them in `tolerance-spec.md` and the dev plan.
- **Owner:** G2.
- **Scope:** the ToleranceSpec numbers in `goldens/qwen3-0.6b/g1/tolerances.json` (the types, schema and loader are O2's, `docs/contracts/tolerance-spec.md`). It binds every model-level check: M2b (gate1-cpu), X1b, I1, I2 and I3, and T1's parity reports.

## Context

The plan (section 5) and the ToleranceSpec contract fix the rule and give starting values:

- e_ref = relL2(torch32, torch64p), with torch64p as the reference; bound = clamp(k·e_ref, floor, 10·floor);
- CPU32: k = 2 against torch64p and 3 against torch32; floors relL2 1e-5, max |Δlogit| 2e-3, KL 1e-7;
- GPU: k = 9 against torch64p and 10 against torch32; floors 1e-4, 1e-2 and 1e-6;
- fixed: CPU64 against torch64p 1e-10 on the full model; Vulkan against D3D12 relL2 1e-4 and max |Δlogit| 1e-2;
- `gpu-vs-cpu32` "is calibrated by G2 under the same rule, or left out";
- gate prompts need a margin of at least max(0.05, 4·τ_logit); the teacher-forced check applies where the margin is at least 2·τ_logit.

The contract says G2 calibrates the final values and records each one in the file. Open points: how e_ref per tensor becomes one number per tap kind, which metrics get bounds, what τ_logit is, and whether the starting floors hold for the implementations Gate 1 checks.

## Decision

1. **e_ref per tap kind** is the largest relL2(torch32, torch64p) over the 12 calibration prompts and, for per-layer taps, over the 28 layers. Each bound carries its e_ref; `tolerances.json` names the prompt and layer it came from (`calibration.where`), and `calibration.json` holds every measured value.
2. **Metrics.** relL2 is bounded for all 21 tap kinds. max |Δ| and KL are bounded for `logits` only, which is what their floors (max |Δlogit|, KL of the logits) are for.
3. **The CPU32 relL2 floor is 3e-5**, not 1e-5, for both CPU32 pairs; its ceiling is therefore 3e-4. Every other k and floor keeps its starting value. The reason is the CpuRef probe (Evidence): CpuRef F32, whose sequential FMA order is the spec of every fp32 kernel, is farther from both oracles on the residual stream than the 1e-5 floor allows, while torch32 itself is unusually accurate there, so k·e_ref cannot cover it. With 3e-5, CpuRef F32's worst comparison is at 0.42 of its bound. **The trade-off:** under the 1e-5 floor only one tap kind fails, `out` (`L27.out` of `multilingual-002`, at 1.21 times its bound against torch64p and 1.02 times against torch32). The next closest kinds would sit at 0.64 (`mlp.down.out`), 0.57 (`mlp.down.in`) and 0.51 (`logits`) of a 1e-5-floor bound. But the `lrm.tolerances/1` rule holds one floor per pair and metric, so the higher floor loosens every CPU32 relL2 bound that sits at the floor by 3 times, the logits' included: there CpuRef F32 measures at most 5.1e-6 and e_ref is 2.1e-6, yet the bound is 3e-5. Per-kind floors would keep 1e-5 everywhere except `out` and need an O2 schema change (Alternatives). This item departs from the plan's value and waits for sign-off (Status).
4. **`gpu-vs-cpu32` is calibrated** with `gpu-vs-torch32`'s rule (k = 10, GPU floors): CpuRef F32 is an fp32 implementation of the same spec at the same precision as torch32, and X1b and I1 compare GPU taps with it.
5. **Fixed pairs.** `cpu64-vs-torch64p`: relL2 1e-10 for every tap (`default`). `vulkan-vs-d3d12`: relL2 1e-4 for every tap and max |Δlogit| 1e-2 for `logits`; the plan's 1e-2 is a logits bound, and activations reach 10^3 or more in Qwen3's residual stream. This is narrower than the contract's fixed-bound line ("relL2 ≤ 1e-4 and maxAbs ≤ 1e-2"), whose sample file puts maxAbs in the pair's `default`, so it applies to every tap there. With g1's file, `spec.TryBound(vulkan-vs-d3d12, <any kind but logits>, maxAbs)` returns None on purpose: X1b and I1 compare activations across the backends by relL2 only. The integrator's note in `tolerance-spec.md` has to land before X1b and I1 are written, and this item waits for sign-off (Status).
6. **τ_logit** is the largest logits maxAbs bound over every pair, because Gate 1 compares greedy ids for each of them. A candidate whose logits are within τ_logit of the reference moves a top-2 margin by at most 2·τ_logit. With these numbers τ_logit = 0.01 (the GPU floor), the gate margin rule is max(0.05, 0.04) = **0.05**, and the teacher-forced threshold is **0.02**. All three are in `tolerances.json` under `greedy`.

## Evidence

Machine: i9-13900K (AVX2, MKL), 8 threads, torch 2.14.1+cpu, transformers 5.20.1; the run ids and reports are below.

**e_ref** (`make_qwen.py`, 12 calibration prompts of 17 to 835 tokens; largest value per tap kind; runs `20261018-072732-1cc39110` and `20261018-120457-85a6c443` wrote the same `calibration.json` byte for byte):

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

The largest per-layer values are at layer 14 of `csharp-012`: `k_proj`, `v_proj` (and `v_cache`, the same tensor) at 2.0e-5 and `o_proj.out` at 1.5e-5. Only there is k·e_ref above the floor (for `o_proj.out`, in every pair but `cpu32-vs-torch64p`); every other bound is its floor. So the bounds that k·e_ref sets rest on a single prompt, `csharp-012`, in layers 12 to 14. On any other calibration prompt the largest value is 5 to 6 times smaller for `k_proj` and `v_proj` (3.6e-6 and 3.3e-6, `chatml-002` L8) and 3 times smaller for `o_proj.out` (4.5e-6, `license-000` L27).

**Holdouts.** torch32's logits are within 2.1e-6 (relL2), 1.7e-4 (max \|Δlogit\|) and 5.5e-10 (KL) of torch64p on every calibration prompt. The nine holdouts are the seven gate prompts outside the subset and both long prompts (`measurements.holdout` in the manifest). Their logits reach 1.06 times the calibrated e_ref in relL2 (`license-008`: 2.25e-6 against 2.12e-6), 0.87 times in max \|Δlogit\| (`multilingual-023`) and 0.95 times in KL (`license-008`). Per tap, the two taps prompts reach at most 0.96 of their kind's e_ref (`L14.attn.k_rope` and `L14.attn.k_norm.out` of `multilingual-002`, run `20261018-072732-1cc39110`); both are holdouts, of 34 and 61 tokens. So the holdouts land near the calibrated maximum, and one logits value just above it. They are not well below it. The bounds still cover them by a wide margin, because every logits bound, and the bound of every per-layer tap kind except `k_proj.out`, `v_proj.out`, `v_cache` and `o_proj.out`, is a floor: the CPU32 logits relL2 bound of 3e-5 is 14 times e_ref, and it would be 4.7 times e_ref at the plan's 1e-5. A later version could take e_ref over the calibration prompts and the holdouts together (g2).

**CpuRef against the goldens** (`cpuref_probe.fsx`, O2's CpuRef ops composed into the Qwen3 forward as `tests/Lrm.Tensor.Tests/TinyModel.fs` does for the tiny model; two taps prompts and every gate prompt):

Run `20261018-091611-7cb55719`, 38 minutes at 8 threads; every comparison passes, and the worst is at 0.42 of its bound. The two taps prompts give the per-layer rows; every gate prompt gives logits, greedy and teacher-forced rows; the long prompts give `final_norm.out` and the stored logit rows.

| Pair | Metric | Taps | Checks | Failed | Largest measured | Worst ratio to bound | Where |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu32-vs-torch32 | greedy_match | logits | 10 | 0 | | all steps matched | |
| cpu32-vs-torch32 | kl | logits | 12 | 0 | 1.85e-09 | 0.019 | chatml-000 logits |
| cpu32-vs-torch32 | maxAbs | logits | 12 | 0 | 3.16e-04 | 0.158 | multilingual-002 logits |
| cpu32-vs-torch32 | relL2 | final_norm.out | 4 | 0 | 4.63e-06 | 0.154 | multilingual-002 final_norm.out |
| cpu32-vs-torch32 | relL2 | logits | 12 | 0 | 4.84e-06 | 0.161 | long-csharp logits |
| cpu32-vs-torch32 | relL2 | per-layer taps | 1008 | 0 | 1.85e-05 | 0.416 | csharp-002 L14.attn.o_proj.out |
| cpu32-vs-torch32 | teacher-forced mismatches | logits | 12 | 0 | 0 | | |
| cpu32-vs-torch64p | kl | logits | 12 | 0 | 2.28e-09 | 0.023 | multilingual-002 logits |
| cpu32-vs-torch64p | maxAbs | logits | 12 | 0 | 3.87e-04 | 0.193 | multilingual-002 logits |
| cpu32-vs-torch64p | relL2 | final_norm.out | 2 | 0 | 5.00e-06 | 0.167 | multilingual-002 final_norm.out |
| cpu32-vs-torch64p | relL2 | logits | 12 | 0 | 5.10e-06 | 0.170 | long-csharp logits |
| cpu32-vs-torch64p | relL2 | per-layer taps | 1008 | 0 | 1.21e-05 | 0.404 | multilingual-002 L27.out |
| cpu64-vs-torch64p | greedy_match | logits | 10 | 0 | | all steps matched | |
| cpu64-vs-torch64p | relL2 | final_norm.out | 2 | 0 | 8.25e-15 | 0.000 | multilingual-002 final_norm.out |
| cpu64-vs-torch64p | relL2 | logits | 10 | 0 | 9.80e-15 | 0.000 | license-008 logits |
| cpu64-vs-torch64p | relL2 | per-layer taps | 1008 | 0 | 2.16e-14 | 0.000 | multilingual-002 L13.attn.v_cache |
| cpu64-vs-torch64p | teacher-forced mismatches | logits | 10 | 0 | 0 | | |

Under the plan's 1e-5 floor, two of these comparisons fail: `L27.out` of `multilingual-002`, at 1.21e-5 against torch64p and 1.02e-5 against torch32, where torch32's own e_ref is 2.1e-6. Tap by tap, CpuRef F32's distance to torch64p is about 2.2 times torch32's own (the median over the two taps prompts), up to 7 times on the residual stream (`resid_mid`, `out`, `mlp.down.out`) and 17 times on an `input_layernorm.out` that reads it: a few dimensions of the residual stream carry activations of 10^3 and more, which torch32's blocked MKL sums keep within about one rounding, while a sequential length-3,072 FMA sum does not. The logits stay at 3.0e-6 to 5.1e-6, max \|Δlogit\| at most 3.9e-4 and KL at most 2.3e-9, all at most 0.2 of their bounds; CpuRef F64 is within 2.2e-14 of torch64p on every tap.

**The first probe**, on a smoke set (`--quick`: 3 calibration prompts, 4 gate prompts) with the starting floors, found the problem: CpuRef F32 failed exactly one tap, `L27.out` of `csharp-000`, at relL2 1.40e-5 against torch32 and 1.29e-5 against torch64p, where torch32's own e_ref is 1.78e-6. There the ratio of CpuRef F32's distance from torch64p to torch32's, tap by tap, reached 54 (`L4.resid_mid` of `license-001`: 1.9e-6 against torch32's 3.6e-8). Logits, greedy ids, teacher-forced rows and CpuRef F64 passed with wide margins, as on the final set.

## Consequences

- **M2b** checks CpuRef F32 against both oracles and CpuRef F64 against torch64p with these numbers. If M2a's forward composes the same CpuRef ops in the same order, its results equal the probe's bit for bit, and the probe's ratios are what gate1-cpu will see.
- **X1b and I1** check GPU taps with the GPU pairs. Every GPU relL2 bound is at least the 1e-4 floor, about 5 times CpuRef F32's largest distance (1.85e-5); a GPU kernel in sequential `mad` order (K2) is expected near CpuRef.
- **The contract text** keeps the plan's starting values as the starting values. The integrator records the calibrated CPU32 floor and the logits-only `vulkan-vs-d3d12` maxAbs there and in the dev plan once both are signed off (requested in G2's PR #19).
- **Reopening.** A change to CpuRef's reduction order, a new gate prompt set, or a different machine class re-runs the probe; any new number is a new golden version `g2`, never an edit of `g1`.

## Alternatives

- **Raise the CPU32 k instead** (to about 10): it ties the bound to torch32's e_ref, which is a poor proxy exactly where CpuRef is weakest. On the residual stream torch32 is up to 17 times more accurate than a sequential sum (54 times on the smoke set), so k would have to grow far beyond the plan's CPU/GPU split, and every other tap kind would loosen with it.
- **Per-kind floors.** These would be the tighter choice: 3e-5 for `out` and the plan's 1e-5 for every other kind. CpuRef F32 would then be at most 0.64 of a 1e-5-floor bound (`mlp.down.out`) and 0.40 of `out`'s. The `lrm.tolerances/1` rule holds one floor per metric and pair, so this needs an O2 contract change: for example, a rule `floors` entry keyed by tap kind, with the loader's exact check unchanged. It is proposed for g2. If the user and the integrator prefer it over item 3 at the review gate, O2 changes the schema first and G2 writes g2 with it. g1 stays as it is, because golden content never changes in place.
- **Keep the 1e-5 floor:** gate1-cpu would fail on a correct CpuRef.
- **Leave `gpu-vs-cpu32` out:** X1b's bisect against CpuRef F32 would have no bound.
