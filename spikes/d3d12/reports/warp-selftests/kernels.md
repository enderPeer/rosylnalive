# Kernels: WARP spike: device, denormal probe, D1's selftests (R5)


No conformance cases were recorded. Every dispatch is under the 100 ms budget.

- Conformance cases: 0, failed: 0
- GPU dispatches: 83, total GPU time 264.56 ms

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
| d3d12-warp | n0 |  | preserve | equal to CpuRef as it is |
| d3d12-warp | n0 | control, no -denorm | flush | equal to CpuRef with every fp32 denormal input and result set to zero of the same sign |

The probe's words (`scale_f32`, a = 0.5): each input, what preserve and flush give, and what each measurement returned.

| Input | Preserve | Flush | d3d12 n0 | d3d12-warp n0 | d3d12-warp n0, control, no -denorm |
| --- | --- | --- | --- | --- | --- |
| `3f800000` | 3f000000 | 3f000000 | 3f000000 | 3f000000 | 3f000000 |
| `01000000` | 00800000 | 00800000 | 00800000 | 00800000 | 00800000 |
| `00800000` | 00400000 | 00000000 | 00400000 | 00400000 | 00000000 |
| `80800000` | 80400000 | 80000000 | 80400000 | 80400000 | 80000000 |
| `00000002` | 00000001 | 00000000 | 00000001 | 00000001 | 00000000 |
| `807ffffe` | 803fffff | 80000000 | 803fffff | 803fffff | 80000000 |
| `00400000` | 00200000 | 00000000 | 00200000 | 00200000 | 00000000 |

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
| scale_f32 | d3d12-warp |  |  | 66.46 ms | 66.5% |
| scale_f32 | d3d12-warp |  |  | 65.75 ms | 65.7% |
| scale_f32 | d3d12-warp |  |  | 16.34 ms | 16.3% |
| groups_echo | d3d12-warp |  | 4 | 15.61 ms | 15.6% |
| scale_f32 | d3d12-warp |  |  | 14.50 ms | 14.5% |
| groups_echo | d3d12-warp |  | 1 | 13.04 ms | 13.0% |
| groups_echo | d3d12-warp |  | 2 | 12.94 ms | 12.9% |
| groups_echo | d3d12-warp |  | 1 | 12.82 ms | 12.8% |
| groups_echo | d3d12-warp |  | 2 | 12.69 ms | 12.7% |
| scale_f32 | d3d12-warp |  |  | 5.91 ms | 5.9% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-064937-ce781a73` |
| Title | WARP spike: device, denormal probe, D1's selftests (R5) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T06:49:37.466Z |
| Ended | 2026-10-18T06:49:40.807Z |
| Execution class | d3d12-warp pci:1414:008c, 10.0.26100.9711, win-x64+avx2+fma shaders b12cabbf300b |
| Class id | `ab6cc56163bb` |
| Driver | 10.0.26100.9711 |
| Commit | `ddd32a703854a7501e606526c0e715cc5b47d777` |
| Nodes | 1 |
| Records | 785 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| D3D12Core.dll | 10.0.26100.9549 |
| affinity_mask | f |
| d3d10warp.dll | 10.0.26100.9711 |
| d3d12.dll | 10.0.26100.9549 |
| device | Microsoft Basic Render Driver |
| driver | 10.0.26100.9711 |
| dxgi.dll | 10.0.26100.9711 |
| dxil_flags | -T cs_6_6 -E main -HV 2021 -O3 -WX -denorm preserve |
| machine_cpus | 32 |
| os | 10.0.26200.9711 |
| os_edition | Professional 25H2 |
| source | spikes/d3d12 (R5 WARP spike) |
| warp_cpus | 4 |
| warp_cpus_requested | 4 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12-warp pci:1414:008c, 10.0.26100.9711, win-x64+avx2+fma shaders b12cabbf300b | 785 | ok | 0 |

### Log problems

None.
