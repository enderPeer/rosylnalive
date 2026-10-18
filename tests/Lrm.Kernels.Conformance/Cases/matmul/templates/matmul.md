{{! matmul: K2's matmul_bf16w bench report, filled by report.fsx from gpu.matmul.throughput and gpu.matmul.bandwidth gauges (warm time_ns, cold_ns, idle_ms, x1_dispatch), gpu.matmul.batch_invariance, gpu.matmul.cpuref_exact and gpu.matmul.probe checks and gpu.fp32_denormals events (MatmulBench.fs). Rendered with T1's template engine and charts until the kernels template charts these records. }}
# matmul_bf16w: {{value run.title}}

{{value run.banner}}
{{value verdict}}

- Throughput points: {{value points}}, the fastest {{value peak}} ({{value peak_at}})
- lm_head decode (m = 1): {{value lm_head_decode}}
- Slowest dispatch X1 makes: {{value slowest}}, {{value slowest_budget}} of the 100 ms budget
- lm_head at X1's row cap: {{value lm_head_cap}}
- fp32 denormals: {{value denormals}}

## Throughput

Each point is the fastest of {{value reps}} back-to-back dispatches after one warm-up, from the backend's `gpu.dispatch` timers: the GPU is warm. FLOPs and traffic are the FLOP registry's cost of the matmul (`OpCost.ofOp`): 2mnk, and every element of x and y in fp32 and of w in bf16 once.

{{chart throughput}}

{{chart bandwidth}}

{{chart time}}

## Cold dispatches

A conformance case, or X1's first dispatch after host work, starts while the GPU idles at low clocks. So after the warm sweep, the points nearest the budget (lm_head at every m, the projections at m = 4,000) are also timed as single dispatches, each alone after {{value idle}} idle. Whether the clocks have dropped by then varies, so each point takes {{value cold_samples}} of them and keeps the slowest. The budget is judged on the slower of warm and cold, and only for what X1 dispatches: the projections up to 4,000 rows and lm_head up to its row cap. lm_head is measured past the cap to show why the cap is there.

{{chart lm_head_cold}}

{{chart cold}}

## All points

Warm is the fastest of the back-to-back dispatches; cold is the slowest and the fastest of the dispatches after an idle gap. "Of budget" is the slower of warm and the slowest cold dispatch.

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
