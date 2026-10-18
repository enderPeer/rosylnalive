# Kernels: D3D12 fp32 denormal probe on scale_f32 (int-0001)


No conformance cases were recorded. Every dispatch is under the 100 ms budget.

- Conformance cases: 0, failed: 0
- GPU dispatches: 4, total GPU time 12.29 µs

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

_No rows._

### Failing cases

_No rows._

## fp32 denormals

The mode each shader backend measured with the `scale_f32` denormal probe right after loading its kernels (GpuBackendInterface). It is fixed within an execution class, and it decides what bit-exact means for a kernel that does fp32 arithmetic (KernelAbiAndCatalog, kernel rules).

| Backend | Node | Blob | fp32 denormals | Bit-exact fp32 arithmetic means |
| --- | --- | --- | --- | --- |
| d3d12 | n0 | control, no -denorm | flush | equal to CpuRef with every fp32 denormal input and result set to zero of the same sign |
| d3d12 | n0 | shader set | preserve | equal to CpuRef as it is |

The probe's words (`scale_f32`, a = 0.5): each input, what preserve and flush give, and what each measurement returned.

| Input | Preserve | Flush | d3d12 n0, control, no -denorm | d3d12 n0, shader set |
| --- | --- | --- | --- | --- |
| `3f800000` | 3f000000 | 3f000000 | 3f000000 | 3f000000 |
| `01000000` | 00800000 | 00800000 | 00800000 | 00800000 |
| `00800000` | 00400000 | 00000000 | 00000000 | 00400000 |
| `80800000` | 80400000 | 80000000 | 80000000 | 80400000 |
| `00000002` | 00000001 | 00000000 | 00000000 | 00000001 |
| `807ffffe` | 803fffff | 80000000 | 80000000 | 803fffff |
| `00400000` | 00200000 | 00000000 | 00000000 | 00200000 |

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
| scale_f32 | d3d12 |  |  | 4.10 µs | 0.0% |
| scale_f32 | d3d12 |  |  | 3.07 µs | 0.0% |
| scale_f32 | d3d12 |  |  | 3.07 µs | 0.0% |
| scale_f32 | d3d12 |  |  | 2.05 µs | 0.0% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-042410-b20091fc` |
| Title | D3D12 fp32 denormal probe on scale_f32 (int-0001) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T04:24:10.913Z |
| Ended | 2026-10-18T04:24:11.714Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b |
| Class id | `9469669b5935` |
| Driver | 32.0.16.1692 |
| Commit | `872a967ccc0235d1d666ec5e255923715b8dbd26` |
| Nodes | 1 |
| Records | 8 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| driver | 616.92 |
| dxil_flags | -T cs_6_6 -E main -HV 2021 -O3 -WX -denorm preserve |
| probe | scale_f32, a = 0.5, Fp32Denormals.probeInputs (GpuBackendInterface) |
| source | tests/Lrm.Gpu.D3D12.Interop.Tests (gpu tier) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b | 8 | ok | 0 |

### Log problems

None.
