# Cluster: Two-stage pipeline, 2 replicas per stage and a spare, one node killed (synthetic)

> **Synthetic data.** This report was rendered from a run log made by a synthetic generator (`lrm report --synthetic`), so the visuals can be reviewed before the real runs exist. None of its numbers are measurements.

**PASS.** 94 work units, 1 reissued after 1 node loss; no divergence, and every replica digest matches.

| Work units | Reissued | Nodes lost | Divergences | Replica digests equal |
| ---: | ---: | ---: | ---: | ---: |
| 94 | 1 | 1 | 0 | 2/2 |

## Pipeline

Green units were accepted, grey ones are matching duplicates from the other replica, amber ones were reissued after a node loss, and red ones were cut off when their node died.

![Stage Gantt chart](charts/cluster-gantt.svg)

![End-to-end step latency](charts/cluster-step_latency.svg)

## Nodes

![Busy time per node by unit outcome](charts/cluster-busy.svg)

| Node | Stage | Backend | Units | Reissued | Lost | Busy | Bytes sent |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| s0a | 0 | vulkan | 10 | 0 | 1 | 117.44 ms | 2.03 MiB |
| s0b | 0 | vulkan | 24 | 0 | 0 | 250.45 ms | 2.09 MiB |
| spare | 0 | vulkan | 12 | 1 | 0 | 152.03 ms | 48.00 KiB |
| s1a | 1 | d3d12 | 24 | 0 | 0 | 295.88 ms | 1.50 KiB |
| s1b | 1 | d3d12 | 24 | 0 | 0 | 299.07 ms | 1.50 KiB |

### Cluster events

| Time | Node | Event | Detail |
| ---: | --- | --- | --- |
| 295.82 ms | coord | cluster.node_lost | target=s0a reason=heartbeat timeout |
| 295.82 ms | coord | cluster.reissue | unit=7db4c2db05c7 from=s0a to=spare |

## Metering

| Node | Component | Stage | Wall (self) | GPU | FLOPs | Bytes moved | Tokens | Records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coord | coordinator |  | 590.26 ms | 0 ns | 0 | 0 B | 0 | 24 |
| coord | coordinator | 0 | 0 ns | 0 ns | 0 | 2.09 KiB | 0 | 24 |
| s0a | stage | 0 | 117.44 ms | 0 ns | 0 | 2.03 MiB | 0 | 19 |
| s0b | stage | 0 | 250.45 ms | 0 ns | 0 | 2.09 MiB | 0 | 48 |
| s1a | stage | 1 | 295.88 ms | 0 ns | 0 | 1.50 KiB | 0 | 48 |
| s1b | stage | 1 | 299.07 ms | 0 ns | 0 | 1.50 KiB | 0 | 48 |
| spare | stage | 0 | 152.03 ms | 0 ns | 0 | 48.00 KiB | 0 | 24 |

## Run

| Field | Value |
| --- | --- |
| Run | `synthetic-cluster` |
| Title | Two-stage pipeline, 2 replicas per stage and a spare, one node killed (synthetic) |
| Kind | cluster |
| Status | ok |
| Started | 2026-10-18T09:00:00.000Z |
| Ended | 2026-10-18T09:00:00.622Z |
| Execution class | not recorded |
| Class id | n/a |
| Driver | n/a |
| Commit | not recorded |
| Nodes | 6 |
| Records | 239 |
| Synthetic | yes |

| Parameter | Value |
| --- | --- |
| generator | lrm report --synthetic cluster |
| model | Qwen3-0.6B |
| replicas | 2 |
| revision | c1899de289a04d12100db370d81485cdf75e47ca |
| seed | 7 |
| spares | 1 |
| split | 14 |
| steps | 24 |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| coord | creator | not recorded | 52 | ok | 0 |
| s0a | joined | vulkan pci:10de:2704, 616.92, win-x64 shaders 87f057325043 | 19 | no | 0 |
| s0b | joined | vulkan pci:10de:2704, 616.92, win-x64 shaders 87f057325043 | 48 | ok | 0 |
| s1a | joined | d3d12 pci:10de:2704, 616.92, win-x64 shaders 87f057325043 | 48 | ok | 0 |
| s1b | joined | d3d12 pci:10de:2704, 616.92, win-x64 shaders 87f057325043 | 48 | ok | 0 |
| spare | joined | vulkan pci:10de:2704, 616.92, win-x64 shaders 87f057325043 | 24 | ok | 0 |

### Log problems

- s0a: no run.end trailer (the node is still running, or it stopped without finishing)
