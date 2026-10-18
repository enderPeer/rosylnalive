# Inference: D3D12 backend: selftests, probe, soak and transfers (D1)


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
| n0 | weights | 1.40 GiB | 128.00 KiB | 12 |
| n0 | activations | 1.00 GiB | 320.00 KiB | 20 |
| n0 | scratch | 576.00 KiB | 576.00 KiB | 12 |
| n0 | table | 1.00 MiB | 1.00 MiB | 22 |

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n0 | (none) |  | 2.78 s | 64.62 ms | 0 | 5.63 GiB | 0 | 23,291 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-055325-c51ed242` |
| Title | D3D12 backend: selftests, probe, soak and transfers (D1) |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T05:53:25.737Z |
| Ended | 2026-10-18T05:53:39.028Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b |
| Class id | `9469669b5935` |
| Driver | 32.0.16.1692 |
| Commit | `0e5e4b923dc59e6e671916fba5528037bd672d83` |
| Nodes | 1 |
| Records | 26,020 |
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
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders b12cabbf300b | 26,020 | ok | 0 |

### Log problems

None.
