# Parity: Tiny model on five node processes: both stage-1 replicas killed, the spare n5 takes over, n4 restarted (C2a)


**PASS.** All 4 oracle self-tests pass.

- Comparisons: 0, failed: 0
- Pairs: none; cases: 0
- Closest to its bound: (no pair) (self-test) cluster_greedy_ids_vs_single_process exact: 0.000 of the bound

## Error by layer

Every per-layer tap's rel_l2 against the golden, summarized per layer over all cases and taps. A pair passes when every dot and whisker stays under its bound line.

![rel_l2 by layer](charts/parity-rel_l2_by_layer.svg)

![Distance to the bound, every comparison](charts/parity-ratio_histogram.svg)

## Per pair

## Per case

_No rows._

## Oracle self-tests

4 of 4 pass.

| Self-test | Metric | Measured | Bound | Ratio | Result |
| --- | --- | ---: | ---: | ---: | --- |
| cluster_run_digest_vs_plan_order | exact | 0 | 0 | 0.000 | ✓ pass |
| cluster_replica_digest_vs_plan_order | exact | 0 | 0 | 0.000 | ✓ pass |
| cluster_logits_vs_single_process | exact | 0 | 0 | 0.000 | ✓ pass |
| cluster_greedy_ids_vs_single_process | exact | 0 | 0 | 0.000 | ✓ pass |

## Closest to the bound

| Pair | Case | Tap | Metric | Measured | Bound | Ratio | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| (no pair) | (self-test) | cluster_greedy_ids_vs_single_process | exact | 0 | 0 | 0.000 | ✓ pass |
| (no pair) | (self-test) | cluster_logits_vs_single_process | exact | 0 | 0 | 0.000 | ✓ pass |
| (no pair) | (self-test) | cluster_replica_digest_vs_plan_order | exact | 0 | 0 | 0.000 | ✓ pass |
| (no pair) | (self-test) | cluster_run_digest_vs_plan_order | exact | 0 | 0 | 0.000 | ✓ pass |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-154120-1dd7d965` |
| Title | Tiny model on five node processes: both stage-1 replicas killed, the spare n5 takes over, n4 restarted (C2a) |
| Kind | cluster |
| Status | ok |
| Started | 2026-10-18T15:41:20.595Z |
| Ended | 2026-10-18T15:41:23.792Z |
| Execution class | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma |
| Class id | `7cc3fe8cde13` |
| Driver | .NET 10.0.13 |
| Commit | `39a5f1bdf7a78497af1acb842b7c95c22da27e46` |
| Nodes | 7 |
| Records | 13,481 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| backend | cpu-f32 |
| job | da6f4c6d6e6f4e0907cb835885b1c2b56f37d5cedd0e7dbf55f4c9fd318139eb |
| kills | n3 after 20 units; n4 after 30 units, restarted after 300 ms |
| lanes | 3 |
| min_nodes | 4 |
| model | goldens/tiny/g1/model |
| model_id | ae4c888d8089683fbd1eaebdc8e394aece8494d94f069ac8da34ae9dbe8ab797 |
| nodes | n1:0-0 n2:0-0 n3:1-1 n4:1-1 n5:all |
| processes | nodes proxy |
| prompt_tokens | 13,1,70 |
| proxy_faults | partition n1 1200-1700 ms |
| replicas | 2 |
| scenario | bothlost5-seed-103 |
| seed | 103 |
| source | Lrm.Serving harness |
| split | 1 |
| steps | 16 |
| stop_at_eos | false |
| topology | bothlost5 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| coord | creator | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 1,039 | ok | 0 |
| n1 | joined | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 2,692 | ok | 0 |
| n2 | joined | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 2,688 | ok | 0 |
| n3 | joined | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 330 | no | 0 |
| n4 | joined | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 663 | no | 0 |
| n4-r1 | joined | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 3,186 | ok | 0 |
| n5 | joined | cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma | 2,883 | ok | 0 |

### Log problems

- n3: line 332: the last line is cut off (no final LF)
- n3: no run.end trailer (the node is still running, or it stopped without finishing)
- n4: line 665: the last line is cut off (no final LF)
- n4: no run.end trailer (the node is still running, or it stopped without finishing)
