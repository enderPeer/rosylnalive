# Kernels: Conformance on d3d12


**PASS.** All 126 conformance cases pass. Every dispatch is under the 100 ms budget.

- Conformance cases: 126, failed: 0
- GPU dispatches: 547, total GPU time 37.55 ms

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

| Kernel | Family | d3d12 |
| --- | --- | --- |
| rmsnorm_rows | norm | ✓ 22/22 |
| rope_apply | rope | ✓ 15/15 |
| add | elementwise | ✓ 14/14 |
| scale_f32 | elementwise | ✓ 11/11 |
| silu_mul | elementwise | ✓ 11/11 |
| bf16_unpack | memory | ✓ 5/5 |
| copy_rows | memory | ✓ 14/14 |
| embed_gather_bf16 | memory | ✓ 10/10 |
| argmax_rows | reduce | ✓ 12/12 |
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
| silu_mul | d3d12 |  |  | 7.24 ms | 7.2% |
| silu_mul | d3d12 |  |  | 5.92 ms | 5.9% |
| add | d3d12 |  |  | 1.79 ms | 1.8% |
| rmsnorm_rows | d3d12 |  |  | 1.70 ms | 1.7% |
| rmsnorm_rows | d3d12 |  |  | 1.70 ms | 1.7% |
| argmax_rows | d3d12 |  |  | 1.43 ms | 1.4% |
| argmax_rows | d3d12 |  |  | 1.43 ms | 1.4% |
| copy_rows | d3d12 |  |  | 1.33 ms | 1.3% |
| copy_rows | d3d12 |  |  | 1.24 ms | 1.2% |
| add | d3d12 |  |  | 711.68 µs | 0.7% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-125923-5c5d258f` |
| Title | Conformance on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T12:59:23.645Z |
| Ended | 2026-10-18T13:00:13.396Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 9a7fe626e4d3 |
| Class id | `95b5c71f37ff` |
| Driver | 32.0.16.1692 |
| Commit | `74b14a7394a307c7ffab5e7108b8b11955140c67` |
| Nodes | 1 |
| Records | 7,395 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| cases | 132 |
| device | NVIDIA GeForce RTX 4080 |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 9a7fe626e4d3 | 7,395 | ok | 0 |

### Log problems

None.
