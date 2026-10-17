# Sample reports

Rendered by `lrm report --synthetic all --out docs/reports/samples` from synthetic run logs, so the report visuals can be reviewed before real runs exist. Every number here is made up; each report says so in its banner. The files are regenerated, never edited: a cpu-tier test fails when they differ from a fresh render.

| Report | Phase | Shows | Charts |
| --- | ---: | --- | ---: |
| [parity](parity.md) | 1 | Agreement with the goldens, tap by tap and layer by layer | 4 |
| [inference](inference.md) | 1 | Prefill and decode speed, per-layer time, memory per arena and metering | 4 |
| [kernels](kernels.md) | 1 | The conformance matrix and GPU time by kernel, layer and dispatch | 5 |
| [cluster](cluster.md) | 1 | The stage Gantt chart, node loss and reissue, step latency and per-node load | 3 |
| [gate1](gate1.md) | 1 | Every Gate 1 check by group and backend, and logits agreement per gate prompt | 2 |
| [training](training.md) | 2 | Loss per curriculum stage, gradient and activation statistics per layer | 5 |
| [routing](routing.md) | 3 | Roslyn expert usage and router entropy per layer | 4 |
| [thinking](thinking.md) | 3 | Roslyn calls made from inside the model, as a trace with outcomes | 3 |
