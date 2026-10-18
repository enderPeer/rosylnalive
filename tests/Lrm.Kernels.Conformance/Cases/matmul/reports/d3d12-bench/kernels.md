# Kernels: matmul_bf16w throughput on d3d12


No conformance cases were recorded. Every dispatch is under the 100 ms budget.

- Conformance cases: 0, failed: 0
- GPU dispatches: 441, total GPU time 373.36 ms

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
| matmul_bf16w | d3d12 |  |  | 21.82 ms | 21.8% |
| matmul_bf16w | d3d12 |  |  | 21.80 ms | 21.8% |
| matmul_bf16w | d3d12 |  |  | 21.70 ms | 21.7% |
| matmul_bf16w | d3d12 |  |  | 21.46 ms | 21.5% |
| matmul_bf16w | d3d12 |  |  | 21.39 ms | 21.4% |
| matmul_bf16w | d3d12 |  |  | 21.30 ms | 21.3% |
| matmul_bf16w | d3d12 |  |  | 10.81 ms | 10.8% |
| matmul_bf16w | d3d12 |  |  | 10.80 ms | 10.8% |
| matmul_bf16w | d3d12 |  |  | 10.79 ms | 10.8% |
| matmul_bf16w | d3d12 |  |  | 10.73 ms | 10.7% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-132542-33830713` |
| Title | matmul_bf16w throughput on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T13:25:42.403Z |
| Ended | 2026-10-18T13:25:43.972Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 |
| Class id | `4b1614e079f0` |
| Driver | 32.0.16.1692 |
| Commit | `558815bd640c4dee35eea5bfc49ee9d65ab113ac` |
| Nodes | 1 |
| Records | 1,026 |
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
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 2d513d812671 | 1,026 | ok | 0 |

### Log problems

None.
