# Kernels: Conformance on d3d12


**PASS.** All 73 conformance cases pass. Every dispatch is under the 100 ms budget.

- Conformance cases: 73, failed: 0
- GPU dispatches: 302, total GPU time 319.62 ms

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

| Kernel | Family | d3d12 |
| --- | --- | --- |
| matmul_bf16w | matmul | ✓ 45/45 |
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
| matmul_bf16w | d3d12 |  |  | 25.02 ms | 25.0% |
| matmul_bf16w | d3d12 |  |  | 23.49 ms | 23.5% |
| matmul_bf16w | d3d12 |  |  | 23.41 ms | 23.4% |
| matmul_bf16w | d3d12 |  |  | 22.22 ms | 22.2% |
| matmul_bf16w | d3d12 |  |  | 19.51 ms | 19.5% |
| matmul_bf16w | d3d12 |  |  | 17.48 ms | 17.5% |
| matmul_bf16w | d3d12 |  |  | 16.52 ms | 16.5% |
| matmul_bf16w | d3d12 |  |  | 11.06 ms | 11.1% |
| matmul_bf16w | d3d12 |  |  | 10.85 ms | 10.8% |
| matmul_bf16w | d3d12 |  |  | 10.16 ms | 10.2% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-140156-4152f402` |
| Title | Conformance on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T14:01:56.176Z |
| Ended | 2026-10-18T14:03:17.173Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 |
| Class id | `4b1614e079f0` |
| Driver | 32.0.16.1692 |
| Commit | `e1f776aec4a8b527cbf32d208e984db6cf2bf398` |
| Nodes | 1 |
| Records | 4,114 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| cases | 95 |
| device | NVIDIA GeForce RTX 4080 |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 | 4,114 | ok | 0 |

### Log problems

None.
