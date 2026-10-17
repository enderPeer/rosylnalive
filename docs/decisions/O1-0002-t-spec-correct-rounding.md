# O1-0002: T_spec uses correctly rounded f64 cos and sin

Status: accepted (O1, 2026-10-18). Affects: ReferenceOracles (T_spec), CpuReferenceSemantics (O2 `ropeTable`), M2 RopeConstants and G2's table dump.

## Context

The plan defines T_spec as follows:
- `freq = f32(f32(pos) · inv_freq)`, with torch's fp32 `inv_freq` bits as input;
- `cos` and `sin` = `f32(f64 cos/sin(freq))`.

"f64 cos" is not unique. libm implementations may differ in the last f64 place:
- torch's vectorized double cos is SLEEF, with up to 1.0 ulp error;
- numpy has its own implementation;
- the C runtime is the one .NET's `Math.Cos` calls (UCRT on Windows, glibc on Linux).

A last-place difference changes the f32 result only when the true value is within about one f64 ulp of an f32 rounding midpoint. That is rare, but it is not impossible over a table of 524,288 entries.

## Decision

T_spec's f64 cos and sin are the **correctly rounded** values of the exactly widened f32 angle, so the table is defined without reference to any library.

The tool computes them in two steps:
1. It takes torch's float64 cos and sin.
2. It recomputes with mpmath at 160 bits every entry whose float32 rounding is not constant within ±2 f64 ulps of torch's value. mpmath is already in the pinned venv.

The tiny self-test recomputes every entry with mpmath and requires the two-step result to match exactly.

## Measured

| Table | Entries | Needed mpmath | torch f64 differs | numpy f64 differs | C runtime differs | torch32 fp32 cos/sin differs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny: theta 1e6, head_dim 32, positions 0..511 | 8,192 × 2 | 0 | 0 | 0 | 0 | 471 / 239 |
| Qwen3-0.6B: theta 1e6, head_dim 128, positions 0..8191 | 524,288 × 2 | 0 | 0 | 0 | 0 | 27,369 / 19,437 |

## Consequences

- CpuRef's `ropeTable(invFreqBits, maxPos)` with .NET `Math.Cos` and `Math.Sin` is expected to equal T_spec bit for bit on Windows x64, because the C runtime agrees in every entry.
- If an entry ever differs on some platform, the plan's fallback applies: an O2 ADR, and CpuRef loads the dumped table. The golden is not changed.
- The torch32 deviation is real. Unpatched torch32 uses fp32 cos and sin, which differ from T_spec in about 5% of the entries. It belongs in e_ref (torch32 vs torch64p), not in any CpuRef bound.
