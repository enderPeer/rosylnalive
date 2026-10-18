# Inference: Tiny model: greedy decoding of p0, 64 steps (M2a), on cpu-f64


| Prefill | Decode | First token | FLOPs per token | Achieved |
| ---: | ---: | ---: | ---: | ---: |
| 2,472 tok/s | 534.8 tok/s | 5.26 ms | 270.30 KFLOP | 166.93 MFLOP/s |

## Speed

![Decode step latency](charts/inference-decode_latency.svg)

![Decode latency distribution](charts/inference-latency_hist.svg)

| Phase | Steps | Tokens | Time | Rate | Median step |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 1 | 13 | 5.26 ms | 2,472 tok/s | 5.26 ms |
| decode | 63 | 63 | 117.80 ms | 534.8 tok/s | 1.81 ms |

## Layers

![Layer time per token](charts/inference-layer_time.svg)

## Memory

![Peak memory per arena](charts/inference-arena_peak.svg)

_No rows._

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n0 | forward | 0 | 123.06 ms | 0 ns | 20.54M | 18.97 MiB | 76 | 7,488 |
| n0 | generate |  | 1.76 ms | 0 ns | 0 | 0 B | 0 | 1 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-060459-d3304488` |
| Title | Tiny model: greedy decoding of p0, 64 steps (M2a), on cpu-f64 |
| Kind | inference |
| Status | ok |
| Started | 2026-10-18T06:04:59.391Z |
| Ended | 2026-10-18T06:04:59.518Z |
| Execution class | cpu-f64 cpu, .NET 10.0.13, win-x64+avx2+fma |
| Class id | `f705487c3bfc` |
| Driver | .NET 10.0.13 |
| Commit | `2b52580cbf2cc1d6451435d6bb4c46f047a13cc6` |
| Nodes | 1 |
| Records | 7,490 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | cpu-f64 |
| backends | cpu-f32,cpu-f64 |
| command | lrm generate |
| model | goldens/tiny/g1/model |
| model_sha256 | 7ade5616bee29bf1008e4a2902cd7aa6b334e3571aa978356d7adfe08e058b2e |
| prompt | goldens/tiny/g1/fwd_f32.p0.safetensors |
| prompt_tokens | 13 |
| revision | local:tiny-g1 |
| steps | 64 |
| stop_at_eos | false |
| threads | 8 |
| warmup_steps | 2 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | cpu-f64 cpu, .NET 10.0.13, win-x64+avx2+fma | 7,490 | ok | 0 |

### Log problems

None.
