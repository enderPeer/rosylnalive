# K1 conformance reports

Rendered with `lrm report <runId> --out <folder>` (T1's templates) from the run logs of the conformance exe, at commit `74b14a7` (the seven basic kernels and their cases). Each run judges every discovered case: E1's selftest and harness cases and K1's cases in `Cases/basic`.

| Folder | Run | Backend and class | Templates | Result |
| --- | --- | --- | --- | --- |
| [`d3d12/`](d3d12/kernels.md) | `20261018-125923-5c5d258f` | d3d12, NVIDIA GeForce RTX 4080 `pci:10de:2704`, UMD 32.0.16.1692 (driver 616.92), DXIL ShaderSetHash `9a7fe626…`, class `95b5c71f37ff`, fp32 denormals `preserve` | [kernels](d3d12/kernels.md), [inference](d3d12/inference.md) (memory per arena and metering) | 126 of 126 cases pass; 6 cases of the missing `matmul_bf16w` and `attention_gqa_causal` skipped |
| [`cpu-device/`](cpu-device/kernels.md) | `20261018-130019-da5f259d` | cpu-device (cpu tier) | [kernels](cpu-device/kernels.md) | 126 of 126 cases pass, bit-exact to CpuRef F32 |

The cpu tier does not build K1's six GPU-only cases (ADR [K1-0001](../../../../../docs/decisions/K1-0001-basic-kernels.md), decision 10), and runs E1's matmul and attention cases on their emulators; so both runs have 126 cases, not the same 126. The ADR has the per-kernel numbers (largest measured error, its reference error, batch splits, slowest dispatch) and a mutation check.

**Records the kernels template does not show yet** (request to the integrator in the K1 pull request): for passing cases, `conformance.case`'s `measured`, `bound`, `ref_error`, `bound_kind`, `time_ns`, `time_source`, `batch_splits` and `batch_matched`. The template lists those only for failing cases, so the margin of a passing kernel to its bound (for example `silu_mul` at 0.058 of it) is in the run log and the ADR, not in a chart.

**Timing.** The dispatch times are GPU timestamps of the faster of two runs, taken between the harness's uploads and checks, so the GPU sits at low clocks much of the time; they show the 100 ms budget holds, not the kernels' throughput.
