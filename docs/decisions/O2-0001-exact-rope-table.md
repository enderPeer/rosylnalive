# O2-0001: CpuRef computes T_spec in exact arithmetic, not with libm

Status: accepted (O2, 2026-10-18). Affects: CpuReferenceSemantics (`ropeTable`), M2 RopeConstants, X1 (the Rope arena), C2 (stage nodes build their own tables), and O1-0002's fallback.

## Context

T_spec (ReferenceOracles; O1-0002) defines the RoPE tables:
- `freq = f32(f32(pos) · inv_freq)`;
- `cos` and `sin` = `f32(RN64(cos freq))` and `f32(RN64(sin freq))`, where RN64 is the **correctly rounded** float64.

CpuRef's table must equal it bit for bit. O1-0002 expected `Math.Cos` and `Math.Sin` to do this on Windows x64, because the C runtime agreed in every entry O1 measured. If an entry ever differed on some platform, the plan's fallback was an ADR plus loading the dumped table.

Two things make a libm table a weak foundation here:
1. **The agreement is not guaranteed.** The C runtime's `cos` is not documented as correctly rounded. UCRT, glibc, a different .NET runtime or a different CPU path may differ in the last float64 place. An entry within about one float64 ulp of a float32 rounding midpoint then flips. That is rare, but every node of a cluster builds its own Rope arena.
2. **The table is a constant of the model, not a result of the execution class.** The same (inv_freq bits, maxPos) must give the same bytes on every node. Otherwise two replicas of a stage disagree, and the cluster's bit-identity and replay rules break across hosts. ExecutionClass scopes results, not model constants.

Loading the dumped table instead would make a 4 MiB data file, the G2 dump for positions 0..8191, a dependency of every node and every maxPos. It would also still leave `Math.Cos` in the definition of anything the dump does not cover.

## Decision

`ExactTrig.cosSin` computes RN64(cos x) and RN64(sin x) of a float32 x in exact fixed-point integer arithmetic (`System.Numerics.BigInteger`). `RopeTable.create` and `CpuRef.ropeTable` use it for every entry, and nothing else.

- **π** comes from Machin's formula, `π = 16·atan(1/5) − 4·atan(1/239)`, with 32 guard bits, cached per precision.
- **Reduction.** x is converted to fixed point exactly. It is reduced by π/2 with 64 guard bits beyond the integer part of x/(π/2), so the reduced argument r, with |r| ≤ π/4, is exact to within one unit of 2^−Q.
- **Series.** sin r and cos r are Taylor series in Q-bit fixed point, starting with Q = 128. The quadrant comes from the reduction multiple mod 4.
- **Certified rounding.** The accumulated error is far below 2^16 units of 2^−Q (2^−112 for Q = 128). A float64 is accepted only when both ends of that interval round to it. Otherwise Q doubles: 256, 512 and so on.
- **Tiny arguments.** For |x| < 2^−27, the results are exactly x and 1, because the higher terms are below half a float64 ulp.
- **Rounding to float32** is `float32 d`, round to nearest even.

## Measured

| Check | Result |
| --- | --- |
| Tiny table (head_dim 32, positions 0..511) vs `goldens/tiny/g1/rope.safetensors` | bit-equal; digest `615b4204…b6d60801` |
| Qwen3-0.6B table (head_dim 128, positions 0..8191) | digest `da820d28…8f6ebfaf`, the ReferenceOracles value |
| ReferenceOracles entry vectors (head_dim 32 and 128) | all equal |
| `cos 1`, `sin 1` | `3fe14a280fb5068c`, `3feaed548f090cee`, as mpmath at 200 bits |
| A random property: within one float64 ulp of `Math.Cos`/`Math.Sin`, even and odd in x | holds |
| Entries a `Math.Cos`/`Math.Sin` table would get wrong on this machine (Qwen table) | 0, as O1 measured |
| Time for the Qwen table, 32 threads (bench tier) | 0.2 s |
| Thread counts | identical tables at 1 and 32 threads |

## Consequences

- **The fallback of O1-0002 is never needed.** There is no table to load, and no platform dependence. The dumped tables remain the cross-check, and the cpu tier compares the tiny one bit for bit.
- **The table is the same on every machine, runtime and thread count.** M2's RopeConstants hold only the pinned `inv_freq` bits, and any maxPos up to 2^24 works.
- **The GPU kernels** (`rope_apply`) read this table and never evaluate sin or cos (KernelAbiAndCatalog). So a GPU run's RoPE inputs are identical on every node.
- **`exp` and `log` stay libm.** The other CpuRef functions still come from the .NET runtime, and ExecutionClass scopes their bits through `Driver` and `Arch` (CpuReferenceSemantics, rule 6). Only the table, a model constant, is made exact.
