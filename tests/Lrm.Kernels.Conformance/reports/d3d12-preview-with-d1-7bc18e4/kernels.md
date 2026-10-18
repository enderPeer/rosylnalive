# Kernels: Conformance on d3d12


**PASS.** All 27 conformance cases pass. Every dispatch is under the 100 ms budget.

- Conformance cases: 27, failed: 0
- GPU dispatches: 105, total GPU time 600.32 µs

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

| Kernel | Family | d3d12 |
| --- | --- | --- |
| scale_f32 | elementwise | ✓ 10/10 |
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
| groups_echo | d3d12 |  |  | 66.82 µs | 0.1% |
| groups_echo | d3d12 |  |  | 65.38 µs | 0.1% |
| groups_echo | d3d12 |  |  | 34.59 µs | 0.0% |
| groups_echo | d3d12 |  |  | 33.98 µs | 0.0% |
| groups_echo | d3d12 |  |  | 33.92 µs | 0.0% |
| groups_echo | d3d12 |  |  | 33.44 µs | 0.0% |
| groups_echo | d3d12 |  |  | 33.38 µs | 0.0% |
| groups_echo | d3d12 |  |  | 33.34 µs | 0.0% |
| groups_echo | d3d12 |  |  | 33.22 µs | 0.0% |
| groups_echo | d3d12 |  |  | 18.30 µs | 0.0% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-051908-a788ae25` |
| Title | Conformance on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T05:19:08.162Z |
| Ended | 2026-10-18T05:19:08.697Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b |
| Class id | `9469669b5935` |
| Driver | 32.0.16.1692 |
| Commit | `d551eabba992ff7bd040b298fc9f53404700bcfe` |
| Nodes | 1 |
| Records | 1,032 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| cases | 49 |
| device | NVIDIA GeForce RTX 4080 |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b | 1,032 | ok | 0 |

### Log problems

None.
