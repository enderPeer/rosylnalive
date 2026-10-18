# GpuExecutor contract

Owner: X1 (X1a wrote v1; X1b runs it on Vulkan and D3D12). Status: v1, proposed with PR #31; it changes only through a `contract-change` PR once it is tagged. Consumers: X1b (the same executor on the GPUs), C2b (the two-stage pipeline on GPU stages), I1 to I3 (gate runs, `lrm bisect`), T1 (the inference and kernels templates chart its records), M2 (the registry in `Commands/Backends.fs`). Code: `src/Lrm.Gpu.Exec/` (`Options.fs`, `Trace.fs`, `Plan.fs`, `GpuForwardOps.fs`) and `apps/Lrm.Cli/Commands/GpuBackends.fs`. Tests: `tests/Lrm.Gpu.Exec.Tests/` (cpu tier on CpuDevice; model tier on Qwen3-0.6B). Decision record: `docs/decisions/X1-0001-static-plan.md`.

The executor implements `IForwardOps<GpuTensor>` (ForwardOpsAndQwenModel) on any `IGpuBackend` (GpuBackendInterface) with the kernel catalog (KernelAbiAndCatalog), one heap and one tensor table written once (HeapLayoutAndTensorTable). It references no concrete backend and no CpuDevice (ProjectGraph, `exec-never-backend`). Bit-identity claims below hold within one ExecutionClass; on CpuDevice, whose kernels are CpuRef F32's, the executor is bit-identical to CpuOps F32.

## The pieces

| Type or module | What it is |
| --- | --- |
| `GpuForwardOps.create`, `createWith` | The constructors: plan, kernels, heap, table, weights, T_spec |
| `GpuExecOptions` | Dispatch timers, the attention and matmul budgets, backend ownership |
| `Chunking` | How long work is split so every dispatch stays inside the 100 ms budget |
| `StaticOrder.trace`, `TracedForward` | Pass one's trace: Forward.runStage recorded over shapes |
| `Liveness`, `PlanVariant`, `PlanOp` | One stage variant's static order, the liveness of its results and where each lives |
| `ExecPlan`, `StagePlan` | The whole plan: layout, sealed table, tokens |
| `GpuOps`, `GpuTensor` | Pass two: the ops and their handle |
| `GpuBackendFactory.create` | An `OpsFactory` (the registry entry) for a GPU backend |

## Creating the executor

`GpuForwardOps.create backend shaders weights options` (and `createWith exec backend shaders weights options`, which returns the `GpuOps` with its plan) does, in order:

1. **Pass one**, `ExecPlan.build weights.Model options exec`: refuses with `ArgumentException` an `OpsOptions` that `OpsOptions.validate` refuses, a `GpuExecOptions` whose budgets are below 1, a plan whose T exceeds 65,535 rows (the grid y of one `embed_gather_bf16` dispatch), and any tensor larger than one 1 GiB page (for example the logits of HeadRows.All at more than 1,766 rows of Qwen3-0.6B). A table beyond 16,383 rows or a layout beyond 63 pages raises `GpuContractException` from R4a's checks.
2. **Kernels.** `backend.LoadKernels shaders`. Every kernel the executor dispatches (`ExecPlan.kernels`: `embed_gather_bf16`, `rmsnorm_rows`, `rope_apply`, `silu_mul`, `add`, `copy_rows`, `argmax_rows`, `matmul_bf16w`, `attention_gqa_causal`) must be in the loaded set; otherwise `GpuContractException` names the missing ones.
3. **The class**, `backend.ExecutionClass`, is the ops' `ExecutionClass` (known once kernels are loaded).
4. **The heap and the table.** `backend.CreateHeap plan.Layout`, then `WriteTable plan.Table`, exactly once.
5. **The weights**, streamed from the memory-mapped checkpoint (`Qwen3Weights.Span`, no host copy) with `IGpuHeap.Upload` in chunks of at most 64 MiB; a bf16 tensor of odd length gets its last word padded with zero. The tied lm_head is the embedding's tensor (one copy).
6. **T_spec**, CpuRef's RoPE table for positions 0 .. MaxContext − 1 (`RopeConstants.table`), as two f32 tensors `[MaxContext, head_dim / 2]`.

With `exec.OwnsBackend` (the default, and always for registry entries) the ops own the backend: `Dispose` disposes it, and a failed create disposes it before the exception leaves. Creation runs inside a `gpu.exec.create` span with `component` = `executor`.

## Options

| Field | Default | Meaning |
| --- | --- | --- |
| `TimeDispatches` | true | A GPU timer around every dispatch, reported at `Wait` as `gpu.dispatch` (below), while the ambient telemetry is enabled. Create the backend without `DeviceOptions.TimeDispatches`, or each dispatch is reported twice |
| `AttentionPairsPerDispatch` | 2^20 | The most query-key pairs per head one attention dispatch covers (ADR K3-0001, decision 8) |
| `MatMulMacsPerDispatch` | 64 · 151,936 · 1,024 | The most multiply-adds (rows · n · k) one matmul dispatch covers: Qwen3-0.6B's lm_head at 64 rows (ADR K2-0001) |
| `OwnsBackend` | true | Dispose disposes the backend |

## Pass one: the plan

**The static order.** `StaticOrder.trace model stage T head taps` runs `Forward.runStage` itself over `TraceOps`, an `IForwardOps` of shapes that computes nothing, under the no-op telemetry facade. It records every call: each compute op (Embed, RmsNorm, MatMul, Rope, KvWrite, Attention, SiluMul, Add, LastRow) with its parameters (weight name, eps, heads, attention shape) and where each input comes from (`ValueRef.Value p`, the result of compute op p; `Input`, a handle from outside the forward; `Cache layer`), and each Tap. The executor therefore never restates the model's layer order: the plan is the forward's own.

**Variants.** Each planned stage is traced from position 0 at the plan's largest T, `min(MaxRows, MaxContext)`, with the plan's tap filter: the variant `all` with `HeadRows.All`; for the stage with the head and T above 1 also `last` with `HeadRows.Last`, whose order adds LastRow before lm_head. A forward of `Rows` rows uses `last` exactly when its stage has the head, its head is `Last` and `Rows` > 1. The op order depends on nothing else (ForwardOpsAndQwenModel, "A static order"), and positions and row counts are only the scalars of the dispatches.

**Liveness.** Each call has a sequence number in its trace, taps included. A result's last use is the latest call that reads it (an op input or a tap); the stage's result (what the forward returns) lives on.

**Memory.**
- **Activations.** Each planned stage has its own pool in the Activations arena. The variant's results are placed in op order, first fit at 256-byte alignment among the results still live, each at its capacity `[T, cols]` (`[1, cols]` after LastRow). So two results live at the same time never share a byte, an op's result never overlaps its inputs, and the pool is the largest set of results live together, not the whole forward. The pool holds the largest of the stage's variants. Qwen3-0.6B at 900 rows needs 42.2 MiB.
- **Logits.** The result of the stage with the head lives at the start of that stage's logits pool in the Logits arena, which also holds `argmax_rows`' output, u32 `[max(T, MaxContext)]`.
- **Taps.** One snapshot per tap the plan records and the stage produces, f32 `[T, cols]`, in the Taps arena, per stage.
- **KV cache.** Per planned stage and layer, `2 · Caches` tensors f32 `[MaxContext, KvDim]` in the KvCache arena.
- **Scratch.** The ids of a forward, u32 `[T]`, and the upload buffer, f32 `[T, hidden]`.
- **Weights** (bf16, read only) and **Rope** (T_spec, read only).

`HeapPlanner` places every allocation (pages of at most 1 GiB, never straddled) and builds the layout; arenas without allocations have no pages (Taps when the plan records none).

**FirstClobber.** For each result, the sequence number of the first later op whose result overlaps it in the pool (none for the stage's result). A handle of a result is readable until the forward records that op.

**The table.** Rows are added in a fixed order, so the same model, OpsOptions and GpuExecOptions give the same layout and the same `SealedTable.Id`:
1. the weights in canonical order, under their safetensors names, `ReadOnly`;
2. `rope.cos`, `rope.sin` (`ReadOnly`); `scratch.ids`, `scratch.upload`; `logits.argmax`;
3. per stage `s<i>`, in OpsOptions.Stages order: the cache tensors `s<i>.kv.L<layer>.<j>`; the pool `s<i>.pool` and the logits pool `s<i>.logits`; every result of every variant as a view of its pool (`TensorFlags.View`), named `s<i>.<variant>.<op>.<Op>[.<weight>]`, with dims `[rows, cols]` at capacity; the chunk views of every split matmul (below), `….c<n>.x` and `….c<n>.y`; the tap snapshots `s<i>.tap.<name>`.

## Pass two: running a forward

`GpuOps` keeps every rule of the seam (ForwardOpsAndQwenModel, "Lifetimes and the plan") exactly as CpuOps does, with the same exceptions. On top of them:

- **BeginForward** refuses a call outside the plan with `ForwardException` before any op: an unplanned stage, rows outside 1 .. MaxRows, positions beyond MaxContext, taps the plan's filter does not cover. It ends the previous result and taps of the stage, picks the variant, and opens one command list for the forward.
- **The static order is enforced.** Each compute op must be the variant's next op, with the same parameters and the planned inputs (a result of this forward by its op, an outside handle, or a cache tensor of the planned layer). Anything else raises `ForwardException` ("departs from the static plan"), as does EndForward with another handle than the planned result or before every op ran. Forward.runStage meets this by construction; other code that drives the ops must call them in the same order.
- **Reused bytes.** A handle of a result of the current forward whose FirstClobber op has been recorded raises `ForwardException` ("overwritten after its last planned use") wherever it is used. Forward.runStage never does it; a caller reading intermediates mid-forward can.
- **Ops to kernels.** Every op records typed ArgBlocks (`Kernels.g.fs`); kernels take every size from their fields, and the tokens are capacities:

  | Op | Kernel | Fields |
  | --- | --- | --- |
  | `Embed ids` | `embed_gather_bf16` | ids uploaded into `scratch.ids` when the op is recorded; rows T, cols hidden, vocab. An id outside the vocabulary raises `ArgumentOutOfRangeException` first |
  | `RmsNorm(x, w, eps)` | `rmsnorm_rows` | rows = T · (cols / len(w)), cols = len(w), eps rounded to f32 |
  | `MatMul(x, w)` | `matmul_bf16w` | m = rows of the chunk, n and k from w's shape |
  | `Rope(x, heads, pos0)` | `rope_apply` | rows = T · heads, head_dim, pos0; the Rope arena's tables |
  | `KvWrite(cache, x, pos0)` | `copy_rows` | rows T into cache rows pos0 .., strides KvDim |
  | `Attention(q, k, v, shape, past)` | `attention_gqa_causal` | per chunk: rows, q_row0, past + q_row0, the heads, head_dim, scale = `CpuRef.attentionScale` |
  | `SiluMul(g, u)`, `Add(a, b)` | `silu_mul`, `add` | n = rows · cols |
  | `LastRow x` | `copy_rows` | one row from row T − 1 |
  | `Argmax x` | `argmax_rows` | into `logits.argmax`; then the list is submitted, waited for and the rows read back |
  | `Tap(name, x)` | `ICommandList.Copy` | x's rows into the stage's snapshot of the tap |

- **Barriers.** Before each dispatch, a global `Barrier` when it reads bytes a dispatch since the last barrier writes, or writes bytes one reads or writes (byte ranges of the rows each dispatch touches). Disjoint chunks of one tensor need none. A `Copy` is a full barrier on both sides (GpuBackendInterface), so it resets the tracking, and so does a new command list.
- **Chunking.** `Chunking.matmul budget m n k`: dispatches of `matmulRows` rows each, `budget / (n · k)` rounded down to a multiple of the kernel's 64-row tile when at least 64, and at least 1. Chunk c ≥ 1 reads and writes the plan's chunk views, which start c · rows rows into x and y. `Chunking.attention budget rows past`: in order from q_row0 = 0, each dispatch takes the most rows whose pairs `n · (past + q_row0) + n · (n + 1) / 2` stay within the budget, and at least one. Rows are batch-invariant (BatchInvariance), so a split changes no bit.
- **Submission.** EndForward with the result submits the list and waits; the result and the taps then stay valid until the stage's next BeginForward. EndForward None drops the list unsubmitted. `Read`, `ReadRows` and `Argmax` submit and wait for whatever is recorded first (also inside a forward), then download.
- **NewCache** takes a free tensor of the stage's and layer's pool, or raises `ForwardException` when that layer's `2 · Caches` tensors are all live, and writes `rows · KvDim` zero floats into it with uploads. `ReleaseCache` returns it. The limit is per layer (CpuOps counts per stage), which is the same for every cache `KvCache.create` makes: one K and one V per layer.
- **Upload** takes an f32 `[rows, cols]` host tensor of at most `T · hidden` elements into `scratch.upload` (`ArgumentException` for another dtype or rank, `ForwardException` when it does not fit); it lives until the next forward ends, and a second upload ends the first.
- **Dispose** disposes the backend (with `OwnsBackend`); every call afterwards raises `ObjectDisposedException`, except `ReleaseCache`, which does nothing.

## Telemetry

Through the ambient facade, with the reserved names (TelemetryFacade). Telemetry never changes a value: a cpu-tier test runs the same schedule with the no-op facade, an in-memory sink, and with the timers off, and compares every bit.

| Record | Kind | When | Attributes | Charted by |
| --- | --- | --- | --- | --- |
| `gpu.dispatch` | GpuInterval | every dispatch, at the submission's `Wait` (with `TimeDispatches`) | `kernel`; `layer` for the ops of a decoder layer; `step` (the forwards of the stage since its last forward from position 0: the greedy step of a Greedy.run); `phase` (`prefill` or `decode`, `Forward.phaseOf`); `backend` (from the backend); `component` = `forward` and `stage` from the forward's scope | kernels: `gpu_by_kernel`, `layer_stack`, `timeline`, `timeline_zoom`, `dispatch_hist`, `dispatches`, `gpu_total`, `slowest`; inference: the GPU column of `meters` |
| `mem.arena` | Gauge, bytes | at `CreateHeap` (the backend's), and again at every BeginForward from position 0, so a run whose ops were created before its telemetry was installed still has them | `backend`, `arena` (`weights`, `rope`, `kv_cache`, `activations`, `scratch`, `logits`, `taps`, and `table` for slot 0) | inference: `arena_peak`, `arenas` |
| `gpu.heap.slots` | Gauge | with `mem.arena` | `backend` | |
| `gpu.exec.create` | Span | one `create` | `component` = `executor`, `backend`, and when it ends `tensors`, `pages`, `heap_bytes`, `stages`, `max_rows` | inference: `meters` |

The forward's own records (`model.forward`, `model.layer`, `model.flops`, `model.tokens`, ForwardOpsAndQwenModel) and the backend's (`gpu.bytes_moved`, `gpu.dispatches`, `gpu.wait`, GpuBackendInterface) come with them. Note that `model.layer` spans time the host's recording of a layer: the GPU work runs at the forward's submission, inside its `model.forward` span, and per layer it is the `gpu.dispatch` records with that `layer`.

## The registry and the commands

`apps/Lrm.Cli/Commands/GpuBackends.fs` registers, through `Backends.register` and after cpu-f32 and cpu-f64:

| Name | Backend |
| --- | --- |
| `cpu-device` | a CpuDevice (E1) with the runner's threads, preserve, CpuRef's NaN and barrier checks on |
| `d3d12` | Direct3D 12 (D1), the first accepted adapter of `Backend.tryCreate` |

Each is `GpuBackendFactory.create backend summary exec shaders device`: every runner creates a fresh backend for its options and owns it. `vulkan` is registered by X1b once Lrm.Gpu.Vulkan (V1) has a provider. So `lrm forward` and `lrm generate` take `--backend cpu-device` (and `d3d12` once the shader set holds K1 to K3's kernels).

`lrm exec-plan [--model DIR|tiny|qwen3-0.6b] [--max-rows N] [--max-context N] [--taps none|all|NAMES] [--caches N] [--split LAYER]` prints the plan for a model's `config.json` without weights or a device: the arenas and pages, the table rows, and per stage and variant the ops, the pool and the dispatches of a prefill of the largest T and of a decode step.

## Vectors

Computed independently in Python from the rules above; `PlanTests` checks them.

**Rows per matmul dispatch** at the default budget (n × k: rows): lm_head 151,936 × 1,024: 64; q_proj 2,048 × 1,024: 4,736; k_proj and v_proj 1,024 × 1,024: 9,472; o_proj 1,024 × 2,048: 4,736; gate_proj and up_proj 3,072 × 1,024: 3,136; down_proj 1,024 × 3,072: 3,136. lm_head over 900 rows takes 15 dispatches.

**Attention dispatches** at 2^20 pairs, as (q_row0, rows):

| Rows, past | Dispatches |
| --- | --- |
| 900, 0 | (0, 900) |
| 1,447, 0 | (0, 1447) |
| 1,448, 0 | (0, 1447), (1447, 1) |
| 1, 4,095 | (0, 1) |
| 4,096, 0 | (0, 1447), (1447, 600), (2047, 460), (2507, 388), (2895, 341), (3236, 309), (3545, 284), (3829, 264), (4093, 3) |

**The static order of the tiny model** (2 layers): `all` has 41 compute ops (Embed, 19 per layer, the final norm, lm_head), `last` 42; the trace records the 39 ActivationNames taps in canonical order. Layer 1's ops are the contract table of ForwardOpsAndQwenModel, "The forward", in order.

**Qwen3-0.6B from its config** (no weights), MaxRows 900, MaxContext 1,024, `Caches` 2, no taps:

| Arena | Bytes | Pages |
| --- | ---: | ---: |
| weights | 1,192,165,376 (pages of 1,072,562,176 and 119,603,200) | 2 |
| rope | 524,288 | 1 |
| kv_cache | 469,762,048 | 1 |
| activations | 44,236,800 | 1 |
| scratch | 3,735,552 | 1 |
| logits | 547,028,992 | 1 |

The table has 1,416 rows: 310 weights, 2 rope, 2 scratch, 1 argmax, 112 cache tensors, 2 pools, 959 results (479 of `all`, 480 of `last`) and 28 chunk views (lm_head of `all`, 14 x and 14 y). A prefill of 900 rows makes 549 dispatches on `all` (211 matmul) and 536 on `last`; a decode step makes 535. The activations pool (44,236,800 bytes) is the planner's own figure, checked against the plan, not computed independently.

## Acceptance (measured, informative)

On the development machine (32 threads, .NET 10.0.13, CpuDevice preserve with barrier checks), class `cpu-device cpu, .NET 10.0.13, win-x64+avx2+fma`:

| Check | Result |
| --- | --- |
| Tiny p0 and p1: prefill with all 39 taps, then 15 decode steps; logits, taps, caches and greedy ids against CpuOps F32 | bit-identical |
| Tiny greedy decoding, 16 steps of p0 and p1 | equal to cpu-f32 and to the torch32 golden |
| Tiny p2 (70 tokens): prefill = decode = 64 + 6 = 10 + 31 + 1 + 28, logits, caches and taps | bit-identical, and equal to cpu-f32 |
| The last-row head against the last row of the full head (p2) | bit-identical |
| Two stages through the host and without it, against one stage and against cpu-f32 (p0, 16 steps) | bit-identical |
| Matmul and attention split into dispatches of a few rows (p2) | bit-identical |
| CpuDevice 1 thread against 8 (p2) | bit-identical |
| cpu-device against torch32 (pair gpu-vs-torch32, GPU floors), all taps of p0 and p1 | every check passes |
| Qwen3-0.6B: a 4-token prefill and 3 decode steps (model tier) | bit-identical to cpu-f32; ids `34208 916 279` after `785 3974 13876 38835` |

## Verification

`tests/Lrm.Gpu.Exec.Tests`, cpu tier:
- **Acceptance** (`ParityTests`): every check of the table above, each a `golden.compare` `exact` self-test in the acceptance run (`AcceptanceRun`; kept under `LRM_DATA/runs` with `LRM_WRITE_EXEC_RUN=1`).
- **The plan** (`PlanTests`): the tiny model's static order; a property over random plans of the tiny model (positions, rows, one or two stages, tap filters, caches, budgets): ops per variant, homes, alignment, views and their offsets, last uses and FirstClobber, results live together sharing no byte, results never overlapping their inputs, chunk views for every split matmul, tap snapshots, the cache pool, every table row valid, and the same layout and table id when built again; the chunking vectors; Qwen3-0.6B's plan against the vectors; the refusals.
- **The seam** (`SeamTests`): calls outside the plan refused before any op; ops outside a forward and cache ops inside one; the lifetimes of intermediates, results, taps, failed forwards and uploads; two stages with and without the host; the cache limit and zeroed caches; departures from the static order; reading reused bytes; a missing kernel refused and the backend disposed; Dispose; the registry's cpu-device runner against cpu-f32; no reference to a concrete backend.
- **Telemetry** (`TelemetryTests`): a real run log with a `gpu.dispatch` per dispatch carrying every kernel, both layers, every step, both phases and the backend, 19 dispatches per layer of a decode step, `mem.arena` for every arena with its lowercase name, `gpu.exec.create`, the model spans; the arenas of ops created before the run; telemetry changing no value.

Model tier: Qwen3-0.6B (`ModelTests`). The GPU tiers are X1b's.
