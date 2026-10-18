# Kernels: Conformance on d3d12


**PASS.** All 133 conformance cases pass. Every dispatch is under the 100 ms budget.

- Conformance cases: 133, failed: 0
- GPU dispatches: 582, total GPU time 67.93 ms

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

| Kernel | Family | d3d12 |
| --- | --- | --- |
| rmsnorm_rows | norm | ✓ 22/22 |
| rope_apply | rope | ✓ 15/15 |
| add | elementwise | ✓ 14/14 |
| scale_f32 | elementwise | ✓ 11/11 |
| silu_mul | elementwise | ✓ 18/18 |
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
| add | d3d12 |  |  | 13.49 ms | 13.5% |
| add | d3d12 |  |  | 13.41 ms | 13.4% |
| silu_mul | d3d12 |  |  | 13.32 ms | 13.3% |
| silu_mul | d3d12 |  |  | 5.74 ms | 5.7% |
| silu_mul | d3d12 |  |  | 4.32 ms | 4.3% |
| copy_rows | d3d12 |  |  | 1.32 ms | 1.3% |
| copy_rows | d3d12 |  |  | 1.30 ms | 1.3% |
| rmsnorm_rows | d3d12 |  |  | 1.12 ms | 1.1% |
| rmsnorm_rows | d3d12 |  |  | 1.11 ms | 1.1% |
| argmax_rows | d3d12 |  |  | 931.84 µs | 0.9% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-134424-7c697dd0` |
| Title | Conformance on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T13:44:24.117Z |
| Ended | 2026-10-18T13:45:10.148Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 75b8dd8195db |
| Class id | `150eae47dabb` |
| Driver | 32.0.16.1692 |
| Commit | `36498a419c7fa7a937f4172361793d2f5b769822` |
| Nodes | 1 |
| Records | 7,941 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| cases | 139 |
| device | NVIDIA GeForce RTX 4080 |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 75b8dd8195db | 7,941 | ok | 0 |

### Log problems

None.
