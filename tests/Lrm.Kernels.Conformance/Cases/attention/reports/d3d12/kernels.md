# Kernels: Conformance on d3d12


**PASS.** All 64 conformance cases pass. Every dispatch is under the 100 ms budget.

- Conformance cases: 64, failed: 0
- GPU dispatches: 254, total GPU time 44.69 ms

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

| Kernel | Family | d3d12 |
| --- | --- | --- |
| attention_gqa_causal | attention | ✓ 36/36 |
| scale_f32 | elementwise | ✓ 11/11 |
| bf16_unpack | memory | ✓ 5/5 |
| args_echo | other | ✓ 2/2 |
| groups_echo | other | ✓ 6/6 |
| table_echo | other | ✓ 4/4 |

### Failing cases

_No rows._

## fp32 denormals

The mode each shader backend measured with the `scale_f32` denormal probe right after loading its kernels (GpuBackendInterface). It is fixed within an execution class, and it decides what bit-exact means for a kernel that does fp32 arithmetic (KernelAbiAndCatalog, kernel rules).

| Backend | Node | Blob | fp32 denormals | Bit-exact fp32 arithmetic means |
| --- | --- | --- | --- | --- |
| d3d12 | n0 |  | preserve | equal to CpuRef as it is |

The probe's words (`scale_f32`, a = 0.5): each input, what preserve and flush give, and what each measurement returned.

| Input | Preserve | Flush | d3d12 n0 |
| --- | --- | --- | --- |
| `3f800000` | 3f000000 | 3f000000 | 3f000000 |
| `01000000` | 00800000 | 00800000 | 00800000 |
| `00800000` | 00400000 | 00000000 | 00400000 |
| `80800000` | 80400000 | 80000000 | 80400000 |
| `00000002` | 00000001 | 00000000 | 00000001 |
| `807ffffe` | 803fffff | 80000000 | 803fffff |
| `00400000` | 00200000 | 00000000 | 00200000 |

## GPU time

![GPU time by kernel](charts/kernels-gpu_by_kernel.svg)

![GPU time per layer](charts/kernels-layer_stack.svg)

## Timeline

![Kernel timeline](charts/kernels-timeline.svg)

![Kernel timeline, embedding and layers 0 to 1](charts/kernels-timeline_zoom.svg)

![Dispatch durations](charts/kernels-dispatch_hist.svg)

### Slowest dispatches

| Kernel | Backend | Layer | Step | Duration | Of budget |
| --- | --- | ---: | ---: | ---: | ---: |
| attention_gqa_causal | d3d12 |  |  | 9.88 ms | 9.9% |
| attention_gqa_causal | d3d12 |  |  | 9.88 ms | 9.9% |
| attention_gqa_causal | d3d12 |  |  | 3.44 ms | 3.4% |
| attention_gqa_causal | d3d12 |  |  | 2.68 ms | 2.7% |
| attention_gqa_causal | d3d12 |  |  | 2.65 ms | 2.7% |
| attention_gqa_causal | d3d12 |  |  | 2.64 ms | 2.6% |
| attention_gqa_causal | d3d12 |  |  | 960.51 µs | 1.0% |
| attention_gqa_causal | d3d12 |  |  | 693.25 µs | 0.7% |
| attention_gqa_causal | d3d12 |  |  | 624.64 µs | 0.6% |
| attention_gqa_causal | d3d12 |  |  | 622.59 µs | 0.6% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-134200-f1727848` |
| Title | Conformance on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T13:42:00.876Z |
| Ended | 2026-10-18T13:42:09.389Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders a9ac4697732f |
| Class id | `35e521522c3b` |
| Driver | 32.0.16.1692 |
| Commit | `7d1ff2354129ffe2a01a05e8773f0868135c795b` |
| Nodes | 1 |
| Records | 3,941 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| cases | 88 |
| device | NVIDIA GeForce RTX 4080 |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders a9ac4697732f | 3,941 | ok | 0 |

### Log problems

None.
