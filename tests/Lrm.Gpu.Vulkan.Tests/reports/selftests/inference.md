# Inference: Vulkan backend: selftests, probe, soak and transfers (V1)


| Prefill | Decode | First token | FLOPs per token | Achieved |
| ---: | ---: | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a |

## Speed

![Decode step latency](charts/inference-decode_latency.svg)

![Decode latency distribution](charts/inference-latency_hist.svg)

_No rows._

## Layers

![Layer time per token](charts/inference-layer_time.svg)

## Memory

![Peak memory per arena](charts/inference-arena_peak.svg)

| Node | Arena | Peak | Last | Samples |
| --- | --- | ---: | ---: | ---: |
| n0 | weights | 1.40 GiB | 128.00 KiB | 21 |
| n0 | activations | 1.00 GiB | 320.00 KiB | 29 |
| n0 | scratch | 576.00 KiB | 576.00 KiB | 25 |
| n0 | table | 1.00 MiB | 1.00 MiB | 35 |

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n0 | (none) |  | 2.06 s | 56.60 ms | 0 | 5.66 GiB | 0 | 23,982 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-155431-a38f1165` |
| Title | Vulkan backend: selftests, probe, soak and transfers (V1) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T15:54:31.497Z |
| Ended | 2026-10-18T15:54:36.382Z |
| Execution class | vulkan pci:10de:2704, 616.92, win-x64 shaders a1a15c99c3fb |
| Class id | `9c41399f5fa9` |
| Driver | 616.92 |
| Commit | `d6493b708775ebc8648102db781198eed5465b88` |
| Nodes | 1 |
| Records | 27,665 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| api | 1.4.351 |
| device | NVIDIA GeForce RTX 4080 |
| driver | 616.92 |
| fp32_denormals | flush |
| memory_types | page=1(device_local) resource_heap=4(device_local\|host_visible\|host_coherent) upload_staging=2(host_visible\|host_coherent) readback_staging=3(host_visible\|host_coherent\|host_cached) |
| queue_family | 2 |
| source | tests/Lrm.Gpu.Vulkan.Tests (gpu tier) |
| spirv_flags | -T cs_6_6 -E main -HV 2021 -O3 -WX -spirv -fspv-target-env=vulkan1.3 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | vulkan pci:10de:2704, 616.92, win-x64 shaders a1a15c99c3fb | 27,665 | ok | 0 |

### Log problems

None.
