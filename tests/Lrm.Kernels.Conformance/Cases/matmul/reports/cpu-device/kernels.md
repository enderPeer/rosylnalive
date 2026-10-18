# Kernels: Conformance on CpuDevice (every case, bit-exact to CpuRef F32)


**PASS.** All 83 conformance cases pass. Every dispatch is under the 100 ms budget.

- Conformance cases: 83, failed: 0
- GPU dispatches: 341, total GPU time 399.62 ms

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

| Kernel | Family | cpu-device |
| --- | --- | --- |
| matmul_bf16w | matmul | ✓ 33/33 |
| attention_gqa_causal | attention | ✓ 2/2 |
| rmsnorm_rows | norm | ✓ 6/6 |
| rope_apply | rope | ✓ 2/2 |
| add | elementwise | ✓ 4/4 |
| scale_f32 | elementwise | ✓ 11/11 |
| silu_mul | elementwise | ✓ 2/2 |
| bf16_unpack | memory | ✓ 5/5 |
| copy_rows | memory | ✓ 3/3 |
| embed_gather_bf16 | memory | ✓ 1/1 |
| argmax_rows | reduce | ✓ 2/2 |
| args_echo | other | ✓ 2/2 |
| groups_echo | other | ✓ 6/6 |
| table_echo | other | ✓ 4/4 |

### Failing cases

_No rows._

## fp32 denormals

The mode each shader backend measured with the `scale_f32` denormal probe right after loading its kernels (GpuBackendInterface). It is fixed within an execution class, and it decides what bit-exact means for a kernel that does fp32 arithmetic (KernelAbiAndCatalog, kernel rules).

| Backend | Node | Blob | fp32 denormals | Bit-exact fp32 arithmetic means |
| --- | --- | --- | --- | --- |
| cpu-device | n0 |  | preserve | equal to CpuRef as it is |

The probe's words (`scale_f32`, a = 0.5): each input, what preserve and flush give, and what each measurement returned.

| Input | Preserve | Flush | cpu-device n0 |
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
| matmul_bf16w | cpu-device |  |  | 43.23 ms | 43.2% |
| matmul_bf16w | cpu-device |  |  | 42.58 ms | 42.6% |
| matmul_bf16w | cpu-device |  |  | 39.91 ms | 39.9% |
| matmul_bf16w | cpu-device |  |  | 35.05 ms | 35.1% |
| matmul_bf16w | cpu-device |  |  | 27.52 ms | 27.5% |
| matmul_bf16w | cpu-device |  |  | 24.93 ms | 24.9% |
| matmul_bf16w | cpu-device |  |  | 21.59 ms | 21.6% |
| matmul_bf16w | cpu-device |  |  | 21.49 ms | 21.5% |
| matmul_bf16w | cpu-device |  |  | 13.11 ms | 13.1% |
| matmul_bf16w | cpu-device |  |  | 11.07 ms | 11.1% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-140141-3dc48e1e` |
| Title | Conformance on CpuDevice (every case, bit-exact to CpuRef F32) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T14:01:41.627Z |
| Ended | 2026-10-18T14:01:44.522Z |
| Execution class | cpu-device cpu, .NET 10.0.13, win-x64+avx2+fma |
| Class id | `00f093f75ec2` |
| Driver | .NET 10.0.13 |
| Commit | `e1f776aec4a8b527cbf32d208e984db6cf2bf398` |
| Nodes | 1 |
| Records | 2,900 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | cpu-device |
| cases | 83 |
| device | CPU device (emulated kernel catalog) |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | cpu-device cpu, .NET 10.0.13, win-x64+avx2+fma | 2,900 | ok | 0 |

### Log problems

None.
