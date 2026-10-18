# ADR X1-0001: The executor's plan is the forward's own trace, with liveness-aliased pools and a strict match

- **Status:** proposed (X1a, 2026-10-18), with PR #31.
- **Owner:** X1.
- **Scope:** GpuExecutor (`docs/contracts/gpu-executor.md`), `src/Lrm.Gpu.Exec`, `apps/Lrm.Cli/Commands/GpuBackends.fs`. It binds X1b (the same code on Vulkan and D3D12), C2b (GPU stages), I1 and I2 (plan sizes for gate and chaos runs) and T1 (what the reports chart).

## Context

ForwardOpsAndQwenModel fixes the seam: one forward over `IForwardOps<'t>`, a static op order, a plan given at creation (`OpsOptions`), and handle lifetimes a sealed tensor table can serve (ADR M2-0001). HeapLayoutAndTensorTable fixes one heap of 64 slots and one table of 16,383 rows, written once. KernelAbiAndCatalog fixes the kernels, their flat-token access and the 100 ms budget; ADR K2-0001 caps lm_head at 64 rows per dispatch, and ADR K3-0001 caps attention at 2^20 query-key pairs. CpuDevice (ADR E1-0001) enforces barriers between dependent dispatches and starts memory as `7fbadbad` words. The executor has to turn the seam into dispatches without restating the model, keep the GPU memory of the full model small enough for I2's replicas on one 16 GB device, and fail loudly where a static plan cannot serve a call.

## Decision

1. **Pass one traces Forward.runStage itself.** A shapes-only `IForwardOps` records every call of one forward per planned stage and head variant (`all`, and `last` for the head stage when T > 1), at the plan's largest T, from position 0, with the plan's taps. The executor never restates the layer order, the tap names or the residual adds: a change to the forward changes the plan with it.
2. **Results live in a per-stage pool, placed by liveness.** Each result gets its capacity `[T, cols]` at the first offset where no result live at the same time sits (256-byte alignment, op order). The stage's result lives on; the head stage's result goes to the Logits arena. Every result is a table view of its pool. For Qwen3-0.6B at 900 rows the pool is 42.2 MiB, where one buffer per op would be about 2.8 GB.
3. **Pass two matches every call against the static order.** The op, its parameters and the origin of every input must be the plan's next op; a departure, an EndForward with another handle or too few ops, and a read of a result after the op that reuses its bytes raise `ForwardException`. Aliasing is therefore never silent: code that drives the ops differently from the forward fails at the first difference, and the error names the op.
4. **One command list per forward, submitted at EndForward.** A barrier goes before every dispatch whose byte ranges conflict with a dispatch since the last barrier, so independent dispatches (q, k and v; the chunks of one op) can overlap. Readbacks submit and wait first. A failed forward's list is dropped.
5. **Taps are copies into the Taps arena when the tap is recorded** (`ICommandList.Copy`), one snapshot per (stage, tap), because liveness reuses the bytes of the tapped result.
6. **Long work is chunked by budgets, not by fixed sizes.** Matmul by multiply-adds per dispatch (default: lm_head at 64 rows), attention by query-key pairs (default 2^20), both greedy and deterministic; rows are batch-invariant, so the split is free to change.
7. **New caches are zeros.** NewCache writes zeros with host uploads: the seam promises a zero tensor, and CpuDevice's poisoned memory would otherwise show through a full-cache `Read`.
8. **Telemetry.** A GPU timer around every dispatch gives `gpu.dispatch` with `kernel`, `layer`, `step` and `phase`, which the kernels template charts; the arena gauges are emitted again at every forward from position 0, so `lrm generate`, which creates its runner before it installs the run's telemetry, still has them in the run.
9. **The registry.** `cpu-device` and `d3d12` are `GpuBackendFactory` entries: a fresh backend per runner, owned by it. A backend whose loaded shader set lacks a kernel the executor dispatches is refused at creation (and disposed). A registry runner reports that, and a missing device, as `GpuBackendUnavailableException`, a `ForwardException`, so the commands print one line. `vulkan` waits for V1's provider: a plan change, since X1a's plan row lists it; X1b registers it.
10. **The All head's logits are capped, not the plan.** The `all` head's result holds `LogitsRows` = min(T, `GpuExecOptions.MaxLogitsRows`, the rows one 1 GiB page holds: 1,766 for Qwen3-0.6B), and BeginForward refuses an All-head forward of more rows; the `last` head (greedy prefill) and decode steps run at every T. So a long prompt never fails creation for its logits alone, and a plan that only decodes greedily sets `MaxLogitsRows` 1 and keeps no `[T, vocab]` logits.
11. **Only the planned stages' weights.** The plan allocates and uploads the weights the traces of its stages read (the embedding with an embedding stage or the tied head). A pipeline node that plans stage 1 of Qwen3-0.6B split at 14 holds 717 MiB of weights, not 1,137 MiB.

## Evidence

Development machine, 32 threads, .NET 10.0.13; `tools/test.ps1` and the test exe directly.
- **Bit identity on CpuDevice** (cpu tier, `Lrm.Gpu.Exec.Tests`): tiny p0 and p1 with all 39 taps and 15 decode steps; p2's prefill, decode and two chunkings; the last-row head; two stages with and without the host; dispatches split into a few rows; 1 and 8 CpuDevice threads: all bit-identical to each other and to CpuOps F32, caches included (`KvCache.digest`). Greedy ids of p0 and p1 equal the torch32 golden. Every tap of p0 and p1 is within the GPU floors against torch32.
- **Planner properties** over 100 random tiny-model plans (single stages of a split and capped logits rows among them), and Qwen3-0.6B's plans from its config: the vectors of the contract (weights 1,192,165,376 bytes in 2 pages, 1,416 table rows, 549 dispatches for a 900-row prefill; stage 1 of a split at 14 alone 751,697,920 bytes of weights; at 2,048 rows the All head at 1,766 rows).
- **The logits cap and single stages on CpuDevice** (cpu tier): with `MaxLogitsRows` 2, an All head over 3 rows is refused before any dispatch, and the last-row head over 4 rows, a decode step and the All head over 2 rows with its logits tap equal cpu-f32 bit for bit; each stage of the tiny model's split planned alone holds only its weights and equals cpu-f32's stage on the host's residual.
- **Qwen3-0.6B on CpuDevice** (model tier): a 4-token prefill and 3 decode steps bit-identical to CpuOps F32 in 9 s.
- **CpuDevice's checks** held on every run: no missing barrier, no read of unwritten memory, no kernel rule broken.
- **A preview on D3D12** (informative, for X1b; not on this branch's tree). On a scratch merge of this branch with `main` (K1 to K3's kernels), a throwaway gpu-tier test under the D3D12 mutex ran the unchanged executor on the RTX 4080 (`pci:10de:2704`, driver 32.0.16.1692, DXIL ShaderSetHash `dd7e4d94…`): the tiny model's p0 and p2, prefill with every tap and 15 decode steps, were bit-identical run to run, gave the same 16 greedy ids as cpu-f32, and every logits row and tap was within relL2 9.5e-7 of cpu-f32; Qwen3-0.6B gave the same 16 greedy ids as cpu-f32 after `785 3974 13876 38835`, 16 steps in 0.24 s after a 0.63 s create.

## Consequences

- **X1b** runs the same code on D3D12 and Vulkan; what changes is the class (bounds instead of bit identity against cpu-f32) and the measured dispatch times against the budgets on idle clocks.
- **C2b** gets per-stage pools and per-stage taps; a stage's result can feed the next stage on the same ops without the host.
- **I1 and I2** size plans with `lrm exec-plan` (`--stage` for a pipeline node, `--max-logits-rows` for greedy-only plans). Long prompts run with HeadRows.Last at any T; long-prompt logit runs that want every row take the prompt as a chunked prefill of at most 1,766 rows per forward for Qwen3-0.6B, which is bit-identical to one prefill (BatchInvariance).
- **M2 (request):** the registry's runners (`lrm generate`, `lrm forward`) create their ops from `OpsOptions`, which says nothing about the head, so the registry plans the All head at up to 1,766 rows (1 GiB at long prompts) that `lrm generate` never reads. A head hint in `OpsOptions` would let the registry set `MaxLogitsRows` 1 for greedy runs.
- **T1 (request):** the inference template's `layer_time` reads `model.layer` spans, which time the host's recording on a GPU executor; charting per-layer `gpu.dispatch` time there would show the device's time per layer.
- **What would reopen this:** a forward whose op order depends on values (the trace would no longer describe every forward), a backend that cannot overlap dispatches without explicit barriers being honoured, or a kernel catalog change that lets a kernel take row offsets (chunk views would go).

## Alternatives

- **One buffer per op position, no aliasing** (ADR M2-0001's sketch). Simpler and needs no FirstClobber, but Qwen3-0.6B's activations at 900 rows would take about 2.8 GB per stage, which I2's replicas on one device cannot afford.
- **A hand-written plan of the Qwen layer.** It restates the order the forward already defines, which is the divergence the seam exists to prevent.
- **Matching only op kinds.** Cheaper, but an input from another place would read bytes the plan has reused, silently.
- **A command list per op or per layer.** It would time each layer on the host, but adds a submission and a wait per layer on the GPU, and the timers already give per-layer device time.
- **Taps as the result's own handle** (as CpuOps does). It would pin every tapped result in the pool, which with all 507 taps of Qwen3-0.6B is the unaliased plan again.
- **The All head's logits split over several tensors or pages** (lm_head already runs in 64-row dispatches). It would lift the 1,766-row limit of HeadRows.All, but a handle is one table token: Read, ReadRows, Argmax and Tap would have to walk the pieces, and the seam's single-tensor result would be gone. A chunked prefill gives the same rows bit for bit, so the cap is the smaller change.
