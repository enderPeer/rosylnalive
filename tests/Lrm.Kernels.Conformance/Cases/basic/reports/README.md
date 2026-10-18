# K1 conformance reports

Rendered with `lrm report <runId> --out <folder>` (T1's templates) from the run logs of the conformance exe. Each full run judges every discovered case: E1's selftest and harness cases and K1's cases in `Cases/basic`. The `d3d12` and `cpu-device` runs are at commit `36498a4` (the `silu_mul` rewrite, `precise` argmax compares and the `silu_mul` bands).

| Folder | Run | Backend and class | Templates | Result |
| --- | --- | --- | --- | --- |
| [`d3d12/`](d3d12/kernels.md) | `20261018-134424-7c697dd0` | d3d12, NVIDIA GeForce RTX 4080 `pci:10de:2704`, UMD 32.0.16.1692 (driver 616.92), DXIL ShaderSetHash `75b8dd81…`, class `150eae47dabb`, fp32 denormals `preserve` | [kernels](d3d12/kernels.md), [inference](d3d12/inference.md) (memory per arena and metering) | 133 of 133 cases pass; 6 cases of the missing `matmul_bf16w` and `attention_gqa_causal` skipped |
| [`cpu-device/`](cpu-device/kernels.md) | `20261018-134516-7cc059c7` | cpu-device (cpu tier) | [kernels](cpu-device/kernels.md) | 133 of 133 cases pass, bit-exact to CpuRef F32 |
| [`d3d12-silu-before/`](d3d12-silu-before/kernels.md) | `20261018-132802-935b6c08` | d3d12 as above, but with the first `silu_mul` body (commit `2d02c87`, DXIL ShaderSetHash `9a7fe626…`, class `95b5c71f37ff`), against the 18 `silu_mul` cases of `36498a4` | [kernels](d3d12-silu-before/kernels.md) | 4 of 18 cases fail: the bands of large negative gates (ADR K1-0001, decision 6 and Evidence) |
| [`d3d12-mutants/`](d3d12-mutants/kernels.md) | `20261018-125623-babc5f61` | d3d12, built from `74b14a7` with one bug in each of the seven kernels (ADR K1-0001, "Mutation check"), so DXIL ShaderSetHash `9c10642e…` | [kernels](d3d12-mutants/kernels.md) | 51 cases fail, each mutated kernel in at least one; the failing-cases table gives each failure's measured value, bound and detail |

The cpu tier does not build K1's six GPU-only cases (ADR [K1-0001](../../../../../docs/decisions/K1-0001-basic-kernels.md), decision 10), and runs E1's matmul and attention cases on their emulators; so both full runs have 133 cases, not the same 133. The ADR has the per-kernel numbers (largest measured error, its reference error, batch splits, slowest dispatch) and a mutation check.

**Records the kernels template does not show yet** (request to the integrator in the K1 pull request): for passing cases, `conformance.case`'s `measured`, `bound`, `ref_error`, `bound_kind`, `time_ns`, `time_source`, `batch_splits` and `batch_matched`. The template lists those only for failing cases. So the margin of a passing kernel to its bound (for example `silu_mul` at 0.025 to 0.072 of it, by band of gates) is in the run log and the ADR, not in a chart.

**Timing.** The dispatch times are GPU timestamps of the faster of two runs, taken between the harness's uploads and checks, so the GPU sits at low clocks much of the time; they show the 100 ms budget holds, not the kernels' throughput.
