# Inference: Tiny model: greedy decoding of p0, 64 steps (M2a), on cpu-f32


| Prefill | Decode | First token | FLOPs per token | Achieved |
| ---: | ---: | ---: | ---: | ---: |
| 1,135 tok/s | 536.3 tok/s | 11.45 ms | 270.30 KFLOP | 159.35 MFLOP/s |

## Speed

![Decode step latency](charts/inference-decode_latency.svg)

![Decode latency distribution](charts/inference-latency_hist.svg)

| Phase | Steps | Tokens | Time | Rate | Median step |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 1 | 13 | 11.45 ms | 1,135 tok/s | 11.45 ms |
| decode | 63 | 63 | 117.47 ms | 536.3 tok/s | 1.80 ms |

## Layers

![Layer time per token](charts/inference-layer_time.svg)

## Memory

![Peak memory per arena](charts/inference-arena_peak.svg)

_No rows._

## Metering

Resource use attributed to node, component and stage (layers summed). Wall time is span self time, so nested spans are not counted twice.

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n0 | forward | 0 | 128.92 ms | 0 ns | 20.54M | 18.97 MiB | 76 | 7,488 |
| n0 | generate |  | 2.34 ms | 0 ns | 0 | 0 B | 0 | 1 |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-060459-1cc52a2e` |
| Title | Tiny model: greedy decoding of p0, 64 steps (M2a), on cpu-f32 |
| Kind | inference |
| Status | ok |
| Started | 2026-10-18T06:04:59.187Z |
| Ended | 2026-10-18T06:04:59.353Z |
| Execution class | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma |
| Class id | `7cc3fe8cde13` |
| Driver | .NET 10.0.13 |
| Commit | `2b52580cbf2cc1d6451435d6bb4c46f047a13cc6` |
| Nodes | 1 |
| Records | 7,490 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | cpu-f32 |
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
| n0 | creator | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 7,490 | ok | 0 |

### Log problems

None.
