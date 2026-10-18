# Kernels: attention_gqa_causal bench on d3d12 (cold GPU)


No conformance cases were recorded. **9 dispatches exceed the 100 ms budget.**

- Conformance cases: 0, failed: 0
- GPU dispatches: 24, total GPU time 2.27 s

## Conformance matrix

Passing cases out of all cases, per kernel and backend.

_No rows._

### Failing cases

_No rows._

## fp32 denormals

The mode each shader backend measured with the `scale_f32` denormal probe right after loading its kernels (GpuBackendInterface). It is fixed within an execution class, and it decides what bit-exact means for a kernel that does fp32 arithmetic (KernelAbiAndCatalog, kernel rules).

_No rows._

The probe's words (`scale_f32`, a = 0.5): each input, what preserve and flush give, and what each measurement returned.

_No rows._

## GPU time

![GPU time by kernel](charts/kernels-gpu_by_kernel.svg)

![GPU time per layer](charts/kernels-layer_stack.svg)

## Timeline

![Kernel timeline, step 12](charts/kernels-timeline.svg)

![Kernel timeline, embedding and layers 0 to 1, step 12](charts/kernels-timeline_zoom.svg)

![Dispatch durations](charts/kernels-dispatch_hist.svg)

### Slowest dispatches

| Kernel | Backend | Layer | Step | Duration | Of budget |
| --- | --- | ---: | ---: | ---: | ---: |
| attention_gqa_causal | d3d12 |  | 12 | 349.39 ms | 349.4% |
| attention_gqa_causal | d3d12 |  | 12 | 331.44 ms | 331.4% |
| attention_gqa_causal | d3d12 |  | 12 | 222.89 ms | 222.9% |
| attention_gqa_causal | d3d12 |  | 7 | 218.66 ms | 218.7% |
| attention_gqa_causal | d3d12 |  | 7 | 217.75 ms | 217.8% |
| attention_gqa_causal | d3d12 |  | 7 | 192.60 ms | 192.6% |
| attention_gqa_causal | d3d12 |  | 6 | 122.63 ms | 122.6% |
| attention_gqa_causal | d3d12 |  | 6 | 103.24 ms | 103.2% |
| attention_gqa_causal | d3d12 |  | 6 | 103.15 ms | 103.2% |
| attention_gqa_causal | d3d12 |  | 5 | 61.95 ms | 62.0% |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-134306-e0f3eaaa` |
| Title | attention_gqa_causal bench on d3d12 (cold GPU) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T13:43:06.106Z |
| Ended | 2026-10-18T13:44:08.666Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders a9ac4697732f |
| Class id | `35e521522c3b` |
| Driver | 32.0.16.1692 |
| Commit | `7d1ff2354129ffe2a01a05e8773f0868135c795b` |
| Nodes | 1 |
| Records | 72 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| device | NVIDIA GeForce RTX 4080 |
| mode | cold: one dispatch after 2500 ms of idle GPU |
| reps | 3 |
| source | tests/Lrm.Kernels.Conformance/Cases/attention/bench/attention-bench.fsx |
| step 10 | prefill_t1024: rows 1024, past 0, heads 16:8, head_dim 128 |
| step 12 | prefill_t4096: rows 4096, past 0, heads 16:8, head_dim 128 |
| step 3 | decode_p4095: rows 1, past 4095, heads 16:8, head_dim 128 |
| step 4 | chunk_t128_past3968: rows 128, past 3968, heads 16:8, head_dim 128 |
| step 5 | chunk_t256_past3840: rows 256, past 3840, heads 16:8, head_dim 128 |
| step 6 | chunk_t512_past3584: rows 512, past 3584, heads 16:8, head_dim 128 |
| step 7 | chunk_t1024_past3072: rows 1024, past 3072, heads 16:8, head_dim 128 |
| step 9 | prefill_t900: rows 900, past 0, heads 16:8, head_dim 128 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders a9ac4697732f | 72 | ok | 0 |

### Log problems

None.
