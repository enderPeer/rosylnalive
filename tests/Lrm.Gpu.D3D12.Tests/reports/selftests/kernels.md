# Kernels: D3D12 backend: selftests, probe, soak and transfers (D1)


No conformance cases were recorded. Every dispatch is under the 100 ms budget.

- Conformance cases: 0, failed: 0
- GPU dispatches: 20,083, total GPU time 34.65 ms

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

![Kernel timeline, step 4](charts/kernels-timeline.svg)

![Kernel timeline, embedding and layers 0 to 1, step 4](charts/kernels-timeline_zoom.svg)

![Dispatch durations](charts/kernels-dispatch_hist.svg)

### Slowest dispatches

| Kernel | Backend | Layer | Step | Duration | Of budget |
| --- | --- | ---: | ---: | ---: | ---: |
| groups_echo | d3d12 |  |  | 66.34 µs | 0.1% |
| groups_echo | d3d12 |  | 1 | 65.54 µs | 0.1% |
| groups_echo | d3d12 |  | 2 | 65.54 µs | 0.1% |
| groups_echo | d3d12 |  | 1 | 65.54 µs | 0.1% |
| groups_echo | d3d12 |  | 2 | 65.54 µs | 0.1% |
| groups_echo | d3d12 |  | 4 | 65.54 µs | 0.1% |
| groups_echo | d3d12 |  | 1 | 65.54 µs | 0.1% |
| scale_f32 | d3d12 |  |  | 4.10 µs | 0.0% |
| scale_f32 | d3d12 |  | 1 | 4.10 µs | 0.0% |
| scale_f32 | d3d12 |  |  | 3.39 µs | 0.0% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-051025-85c086f7` |
| Title | D3D12 backend: selftests, probe, soak and transfers (D1) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T05:10:25.004Z |
| Ended | 2026-10-18T05:10:35.200Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b |
| Class id | `9469669b5935` |
| Driver | 32.0.16.1692 |
| Commit | `8823dc74bdd530d864c21011bb68dc8fd8efe951` |
| Nodes | 1 |
| Records | 21,279 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| driver | 32.0.16.1692 |
| dxil_flags | -T cs_6_6 -E main -HV 2021 -O3 -WX -denorm preserve |
| source | tests/Lrm.Gpu.D3D12.Tests (gpu tier) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b | 21,279 | ok | 0 |

### Log problems

None.
