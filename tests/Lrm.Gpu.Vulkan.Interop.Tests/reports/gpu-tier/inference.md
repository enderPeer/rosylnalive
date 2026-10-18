# Inference: Vulkan interop: device creation and hello-compute (R2)


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

_No rows._

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n0 | (none) |  | 1.57 s | 2.98 µs | 0 | 1.50 KiB | 0 | 18 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-144112-f93fa5d9` |
| Title | Vulkan interop: device creation and hello-compute (R2) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T14:41:12.819Z |
| Ended | 2026-10-18T14:41:13.521Z |
| Execution class | vulkan pci:10de:2704, 616.92, win-x64 shaders a1a15c99c3fb |
| Class id | `9c41399f5fa9` |
| Driver | 616.92 |
| Commit | `2b77c602cc52f6c883949b2613763a01a3723352` |
| Nodes | 1 |
| Records | 102 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| api | 1.4.351 |
| device | NVIDIA GeForce RTX 4080 |
| driver | 616.92 |
| source | tests/Lrm.Gpu.Vulkan.Interop.Tests (gpu tier) |
| vk_header_version | 365 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | vulkan pci:10de:2704, 616.92, win-x64 shaders a1a15c99c3fb | 102 | ok | 0 |

### Log problems

None.
