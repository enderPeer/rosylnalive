# Inference: Conformance on d3d12


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
| n0 | scratch | 942.25 MiB | 942.25 MiB | 1 |
| n0 | table | 1.00 MiB | 1.00 MiB | 1 |

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n0 | (none) |  | 4.62 s | 67.93 ms | 0 | 8.98 GiB | 0 | 7,212 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-134424-7c697dd0` |
| Title | Conformance on d3d12 |
| Kind | kernels |
| Status | ok |
| Started | 2026-10-18T13:44:24.117Z |
| Ended | 2026-10-18T13:45:10.148Z |
| Execution class | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 75b8dd8195db |
| Class id | `150eae47dabb` |
| Driver | 32.0.16.1692 |
| Commit | `36498a419c7fa7a937f4172361793d2f5b769822` |
| Nodes | 1 |
| Records | 7,941 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | d3d12 |
| cases | 139 |
| device | NVIDIA GeForce RTX 4080 |
| source | tests/Lrm.Kernels.Conformance (ConformanceHarness) |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | d3d12 pci:10de:2704, 32.0.16.1692, win-x64 shaders 75b8dd8195db | 7,941 | ok | 0 |

### Log problems

None.
