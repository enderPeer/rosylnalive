{{! matmul: K2's matmul_bf16w bench report, filled by report.fsx from gpu.matmul.throughput and gpu.matmul.bandwidth gauges, gpu.matmul.batch_invariance, gpu.matmul.cpuref_exact and gpu.matmul.probe checks and gpu.fp32_denormals events (MatmulBench.fs). Rendered with T1's template engine and charts until the kernels template charts these records. }}
# matmul_bf16w: {{value run.title}}

{{value run.banner}}
{{value verdict}}

- Throughput points: {{value points}}, the fastest {{value peak}} ({{value peak_at}})
- lm_head decode (m = 1): {{value lm_head_decode}}
- Slowest dispatch: {{value slowest}}, {{value slowest_budget}} of the 100 ms budget
- fp32 denormals: {{value denormals}}

## Throughput

Each point is the faster of {{value reps}} back-to-back dispatches after one warm-up, from the backend's `gpu.dispatch` timers. FLOPs are the FLOP registry's 2mnk; traffic is its minimum (x and y in fp32, w in bf16, each element once).

{{chart throughput}}

{{chart bandwidth}}

{{chart time}}

{{table points}}

## Batch invariance and CpuRef

Per shape, one m = 512 call, one m = 37 call and single-row calls (m = 1) over the same activation rows and weight. The rows alone must equal the rows of both batched calls bit for bit (BatchInvariance). The CpuRef column compares the single rows with CpuRef F32 (Rows.dotBf16); it is informational, because the conformance bound is KernelBounds.

{{table invariance}}

## Arithmetic probes

Two dot products whose result tells the arithmetic apart: whether a `mad` step is fused (one rounding, as CpuRef's FusedMultiplyAdd), and whether the steps run in increasing p. Informational: the bound is KernelBounds.

{{table probes}}

## Run

{{table run.summary}}

{{table run.params}}

### Log problems

{{value run.problems}}
