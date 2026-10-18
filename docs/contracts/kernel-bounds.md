# KernelBounds contract

Owner: O2. Status: v1, frozen at the contracts-v1 tag. Consumers: E1 (the conformance harness takes every bound from here), K1, K2 and K3 (the kernels are judged by it), X1 and I1 (kernel reports). Code: `src/Lrm.Tensor/KernelBounds.fs`. Tests: `tests/Lrm.Tensor.Tests/ToleranceTests.fs` (cpu tier).

After contracts-v1, this text, `KernelBounds.fs` and its vectors change only through a `contract-change` PR with a CHANGELOG entry. **The factor and the floor change only with evidence**: a conformance run that shows why. Model-level tolerances (taps and logits against the oracles) are ToleranceSpec (`tolerance-spec.md`), not this contract.

## The rule

Every catalog kernel (KernelAbiAndCatalog) has exactly one bound, of one of two kinds:

| Kind | A case passes when |
| --- | --- |
| **BitExact** | The output's bytes equal CpuRef F32's, element by element. NaN bit patterns count as bits |
| **Relative** | `relL2(gpu, CpuRef64) ≤ max(Floor, Factor · relL2(CpuRef32, CpuRef64))`, with **Factor = 2** and **Floor = 1e-6**, and neither side is NaN |

**Why the bound is relative to the reference's own error.** A fixed 1e-6 sits below the float32 rounding floor of long reductions: at k = 2048 to 3072, the matmul sizes of Qwen3-0.6B, CpuRef F32's own distance from CpuRef F64 can exceed 1e-6. So each case is measured against what float32 can do on that case. A kernel passes when it stays within twice the error of the float32 reference, which itself follows the exact sequential definition (CpuReferenceSemantics).

| Kernel | Kind | Why |
| --- | --- | --- |
| `scale_f32` | BitExact | one float32 multiply per element |
| `table_echo` | BitExact | copies table words |
| `args_echo` | BitExact | copies argument words |
| `groups_echo` | BitExact | writes group indices |
| `bf16_unpack` | BitExact | bf16 widening is exact |
| `embed_gather_bf16` | BitExact | gathers rows and widens bf16 exactly; an out-of-range id gives the quiet-NaN row `7fc00000` |
| `rmsnorm_rows` | Relative | a reduction plus rsqrt |
| `rope_apply` | Relative | products and a sum per element, from the pinned table |
| `silu_mul` | Relative | `exp` |
| `add` | BitExact | one float32 add per element |
| `copy_rows` | BitExact | copies 4-byte elements |
| `argmax_rows` | BitExact | the index is exact: ties take the lowest index, and a row holding a NaN gives `0xFFFFFFFF` |
| `matmul_bf16w` | Relative | a length-k reduction |
| `attention_gqa_causal` | Relative | online softmax over 64-key blocks |

`KernelBounds.all` lists these 14 kernels in catalog order. A kernel without an entry has no bound: `KernelBounds.find` raises `ArgumentException`, and the harness refuses the kernel. A kernel added to the catalog gets its entry in the same PR.

## Measuring a case

E1's harness does the following for every case:

1. **Inputs.** It runs CpuRef F32 and CpuRef F64 on the case's inputs, through the CpuRef op the kernel implements (CpuReferenceSemantics). F64 receives the same values as F32: float32 inputs widened exactly, and bf16 weights widened. So the two runs differ only in arithmetic precision.
2. **Bit-exact kernels.** It calls `KernelBounds.checkExact kernel elementSize gpu cpuRef32`, or `KernelBounds.check` with the tensors. The measurement is the number of differing elements; it passes at 0.
3. **Relative kernels.** It calls `KernelBounds.checkRelative kernel gpu cpuRef32 cpuRef64`, with all three widened to float64, or `KernelBounds.check` with `Some cpuRef64`. The result carries:
   - `ReferenceError` = relL2(CpuRef32, CpuRef64);
   - `Bound` = `max(1e-6, 2 · ReferenceError)`;
   - `Measured` = relL2(gpu, CpuRef64).

   NaN in any of them fails the case.
4. **Logging.** It emits `conformance.case` with `KernelBounds.attrs check`: `kernel`, `bound_kind` (`exact` or `relative`), `measured`, `bound`, `ref_error` and `elements`, next to its own case attributes. **The measured reference error is always logged**, so reports show how close each kernel sits to the float32 floor.

CpuDevice runs the same CpuRef row kernels, so on CpuDevice every kernel equals CpuRef F32 bit for bit. That is stricter than its bound, and ConformanceHarness requires it.

## Vectors

| Input | Result |
| --- | --- |
| `relativeBound 1e-9` | 1e-6 (the floor) |
| `relativeBound 3e-6` | 6e-6 |
| `relativeBound NaN` | NaN, so the case fails |
| Relative case where the GPU output equals CpuRef32 | passes: measured = reference error ≤ 2 · reference error |
| The same case with one element off by 1.0 | fails |
| A NaN in the GPU output | fails |
| `add`: outputs `(1, 2, −3, 4)` against `(1, 2, 3, 4)` | measured 1, fails |
| `embed_gather_bf16`: NaN `7fc00000` against NaN `7fc00000` | passes: bits are compared |
| `checkRelative "add"`, `checkExact "matmul_bf16w"`, `find "softmax_rows"` | raise `ArgumentException` |

**A measured example.** A matmul_bf16w case with m = 17, k = 256, n = 96, uniform inputs and bf16 weights: the float32 reference error is between 0 and 1e-6, and CpuRef F32 passes its own bound.

## Verification

The cpu tier of `Lrm.Tensor.Tests` checks:
- the 14 kernels, each listed once;
- that the bit-exact set is exactly the nine kernels above;
- the vectors;
- the attribute keys that `conformance.case` carries;
- the measured matmul example through `KernelBounds.check`.

E1's harness checks that every catalog kernel has an entry.
