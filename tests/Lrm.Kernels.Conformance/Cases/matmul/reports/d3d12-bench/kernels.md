# Kernels: matmul_bf16w throughput on d3d12


No conformance cases were recorded. **4 dispatches exceed the 100 ms budget.**

- Conformance cases: 0, failed: 0
- GPU dispatches: 1,353, total GPU time 2.80 s

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

_No rows._

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
| matmul_bf16w | d3d12 |  |  | 322.11 ms | 322.1% |
| matmul_bf16w | d3d12 |  |  | 183.13 ms | 183.1% |
| matmul_bf16w | d3d12 |  |  | 180.59 ms | 180.6% |
| matmul_bf16w | d3d12 |  |  | 167.99 ms | 168.0% |
| matmul_bf16w | d3d12 |  |  | 84.98 ms | 85.0% |
| matmul_bf16w | d3d12 |  |  | 83.69 ms | 83.7% |
| matmul_bf16w | d3d12 |  |  | 43.49 ms | 43.5% |
| matmul_bf16w | d3d12 |  |  | 43.43 ms | 43.4% |
| matmul_bf16w | d3d12 |  |  | 25.72 ms | 25.7% |
| matmul_bf16w | d3d12 |  |  | 25.10 ms | 25.1% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-140617-1ce83fea` |
| Title | matmul_bf16w throughput on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T14:06:17.133Z |
| Ended | 2026-10-18T14:07:39.156Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 |
| Class id | `4b1614e079f0` |
| Driver | 32.0.16.1692 |
| Commit | `3f84e1547f7c12e41fb1417cd0ef1233eb9947af` |
| Nodes | 1 |
| Records | 2,082 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| device | NVIDIA GeForce RTX 4080 |
| reps | 5 |
| source | tests/Lrm.Kernels.Conformance/Cases/matmul/MatmulBench.fs (K2) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 | 2,082 | ok | 0 |

### Log problems

None.
