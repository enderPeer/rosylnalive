# D1 gpu-tier reports

`lrm report` rendered these files from the gpu tier's run log (kind `kernels`) of `tests/Lrm.Gpu.D3D12.Tests`. Both reports name the run and the commit in their Run table.

- [`kernels.md`](kernels.md) is the default template. It shows the fp32 denormal probe's bit table, GPU time by kernel, the timeline of one selftest submission and the dispatch histogram.
- [`inference.md`](inference.md) was rendered with `--template inference`. It shows peak memory per arena and the metering.

## What the reports do not show yet

No T1 template reads the verdicts below, so they exist only as records in the run log (`events.n0.jsonl`). Request 2 of the D1 pull request asks the integrator for template slots. Until those slots land, the conformance matrix in `kernels.md` stays empty: the selftests are not `conformance.case` checks, which belong to E1's harness.

| Record | Kind | What it holds |
| --- | --- | --- |
| `gpu.selftest` | check | One per selftest case and run. The negative control's cases fail by design and carry `negative_control = true` |
| `gpu.soak` | check | One per soak run (`mode` = `timed_dispatches`, `submission_timer`, `no_timers`): host, GPU, frame and private-memory flatness, and managed memory per dispatch against its bound |
| `gpu.heap.descriptor_writes` | check | The resource heap's 64 and the probe heap's 64 descriptor writes |
| `gpu.device.check`, `gpu.device.memory`, `gpu.device.discover` | check, gauge, span | Adapter discovery and verdicts |
| `gpu.transfer`, `gpu.transfer.rate` | span, gauge | Every staged transfer, and the 1.5 GB round trip's rates |
| `gpu.buffer_memory` | gauge | Staging, copy-scratch and probe-heap memory outside the arenas |

The `gpu.probe` spans and every other span, GPU and meter record are counted in the metering table of `inference.md`.
