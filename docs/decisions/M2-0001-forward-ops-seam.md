# ADR M2-0001: One Qwen forward over a handle-generic op seam, with the stage boundary in the class's dtype

- **Status:** accepted, 2026-10-18.
- **Owner:** M2 (M2a).
- **Scope:** ForwardOpsAndQwenModel (`docs/contracts/qwen-model.md`). It binds X1 (GpuForwardOps), C2 (the stage executor) and M2b, and touches the ProjectGraph (Lrm.Models.Qwen now references Lrm.Telemetry).

## Context

The Qwen3 forward must run, bit-identically within each class, on:
- CpuRef F32 and F64 (the oracle chain);
- CpuDevice (X1a: bit-exact to CpuOps F32);
- Vulkan and D3D12 (X1b);
- split across pipeline stages on different nodes (C2, I2).

The GPU executor records a two-pass static plan and copies taps into a Taps arena at record time (GpuExecutor contract). The cluster rebuilds caches by exact replay and checks a fast rebuild against it. One definition of the forward has to serve all of these without reading values back mid-forward, and without each backend re-deriving the layer order. That order is where a tap name or a residual add silently diverges.

## Decision

1. **One forward, an op seam.**
   - The forward (`Forward.runStage`) is written once over `IForwardOps<'t>`, whose members are the CpuRef ops plus `NewCache`, `KvWrite`, `LastRow`, `Argmax`, `Tap`, `Read` and `Upload`.
   - `'t` is the backend's handle: a host tensor for CpuOps, a token for a GPU. It is a type parameter, not an interface or a box, so CpuOps runs on CpuRef's tensors with no wrapper.
   - The registry and the CLI see the same model through `IQwenRunner`, which hides `'t`.
2. **A static order.**
   - The forward calls the ops in one order fixed by the model, stage, T, pos0 and the tap filter. The order is the contract's table: 19 calls per layer.
   - No op but `Argmax` and `Read` returns host data, so a GPU can record the whole forward as a plan.
   - Weights are passed by safetensors name, already aliased.
3. **Taps are snapshots.** `Tap` returns a handle that later ops never change: a GPU copies into its Taps arena when the op is recorded, and CpuOps returns the tensor itself. The forward asks only for the taps the filter wants.
4. **The cache lives in the ops.**
   - `NewCache` allocates each layer's K and V in the ops' dtype, which is the KvCache arena on a GPU. `KvCache<'t>` holds the handles and `Length`.
   - K is stored after norm and RoPE, and V raw.
   - The cache is fp32 on every fp32 class and fp64 on cpu-f64, as CpuReferenceSemantics requires; an fp32 cache in F64 mode would break the 1e-12 chain.
   - `KvCache.digest` is a fixed encoding (contract vectors K1 to K4), which C2 uses to compare a fast rebuild with exact replay.
5. **The stage boundary carries the residual stream in the ops' dtype.** That is fp32 on every fp32 class, the plan's fp32 residual, and fp64 on cpu-f64. A stage's input is ids (layer 0) or that residual; its output is the residual or the logits. Composition through the host (`Read` then `Upload`) is exact.
6. **The final norm runs on every row**, and only the head takes the last row in greedy decoding. This follows the FLOP registry's forward (`OpCost.forwardOps`). The metering then equals the registry exactly in every class, and `final_norm.out` is always `[T, hidden]`. The bits are the same either way (BatchInvariance).
7. **Metering through the registry.**
   - The forward meters each op with `Metering.op "model"`, so CpuRef, CpuDevice and the GPUs meter the same work identically.
   - It emits `model.forward` spans (phase, step, tokens, stage) and `model.layer` spans (layer, phase), which the inference template already reads.
   - The registry lives in Lrm.Telemetry, so Lrm.Models.Qwen references it. It is an additive edge, allowed by every layering rule, because Telemetry references only Core.
8. **Loading.** `Qwen3Weights.Open` verifies every SHA256SUMS file and requires a pin for a Hugging Face commit. It treats lm_head as the embedding when it is missing or equal by SHA-256 (checked once). The forward then reads one tensor, and a GPU uploads it once.
9. **A plan at creation, and handle lifetimes on the seam.** The GPU executor seals one tensor table of 16,383 rows before its first dispatch (HeapLayoutAndTensorTable), so the seam cannot assume dynamic allocation. A forward of Qwen3-0.6B makes about 540 handles (19 per layer × 28, plus 5), so 64 greedy steps would make about 34,000.
   - **The plan.** Ops are created with `OpsOptions`: the stages they run, the rows of one forward, the positions, the taps a forward may record, and the caches of each stage alive at once. A GPU sizes its arenas and seals its table from these once. A call outside the plan raises `ForwardException` before any op runs.
   - **The bracket.** `runStage` announces each forward with `BeginForward(ForwardCall)`, whose fields are exactly what the static order depends on. It ends the forward with `EndForward(Some result)`, or `EndForward None` when an op raised.
   - **Lifetimes.** Intermediates end with their forward. The result and the taps last until the next forward of the same stage index, so a stage's result can feed the next stage directly and a greedy step's logits are read before the next step. An upload lasts until the next forward ends. Cache tensors last until `ReleaseCache`: `KvCache` is `IDisposable`, and `Greedy.run` and the runner release their caches.
   - **Ownership.** The ops and `IQwenRunner` are `IDisposable`; the runner owns its ops, and the commands `use` it.
   - **Reads.** `ReadRows` reads back only the rows asked for. `KvCache.digest` reads a cache's valid rows, not the whole `[MaxPos, cols]` tensor (about 235 MB for Qwen3-0.6B at the 1,024 positions of I2's chaos runs).
   - **Enforced on the CPU.** CpuOps enforces the plan and the lifetimes exactly, although host memory would not need it. Everything that runs on CpuOps (Forward, Greedy, the runner, the commands and every test) therefore keeps to what a static GPU plan can serve. A cpu-tier test proves each rule.

## Evidence

The evidence below was measured before decision 9. Decision 9 changed no op and no bit, and the same tests pass with CpuOps enforcing the lifetimes: logits, taps, caches and greedy ids are identical.

Measured on the development machine: 32 threads, .NET 10.0.13, class `cpu-f32 cpu, .NET 10.0.13, win-x64+avx2+fma`. Commands: `tools/test.ps1 -Tier cpu` and `-Tier model` (the Lrm.Models.Qwen.Tests exe), and `lrm generate` and `lrm forward` on the tiny model.
- **The tiny model**, on CpuOps over the seam:
  - cpu-f64 against torch64p, worst relL2 1.5e-15 over all 39 taps;
  - cpu-f32 against torch32, 8.1e-7, and against torch64p, 8.8e-7;
  - 64 greedy steps identical to both oracles.

  These are the numbers O2 measured for CpuRef's own chain (`cpu-reference.md`), so the seam adds no rounding.
- **Bit for bit, on the tiny model, in both modes:** prefill = decode = chunked prefill, for logits, caches and all 39 taps; two stages through the host = one stage over 17 steps; 1 = 7 = 32 threads.
- **The static order:** two prompts of the same length record the same 42 op calls.
- **Metering:** a 4-step greedy run meters exactly the registry's 3,508,960 FLOPs, which Python computed independently from the registry's formulas.
- **Qwen3-0.6B:**
  - lm_head.weight's SHA-256 equals the embedding's, so the checkpoint's 311 tensors are read as 310;
  - a 4-token prefill equals 4 decode steps in both modes;
  - cpu-f32 decodes at about 8 tokens/s on 8 threads (informative).

## Consequences

- **X1** implements `IForwardOps<GpuTensor>` in `GpuForwardOps`, following the contract's member table and "Lifetimes and the plan":
  - pass one of its static plan runs the forward over a recording ops for each planned stage at `MaxRows`; pass two places every token and seals the table once;
  - `BeginForward` picks the call's recorded plan, and each intermediate's token is reused by its place in the static order;
  - the result and the taps of each planned stage have fixed slots, `Tap` copies into the Taps arena, and the Taps arena is sized from `OpsOptions.Taps`;
  - `NewCache` takes a tensor from a KvCache pool of `Caches` sets per planned stage, and `ReleaseCache` returns it;
  - `Read`, `ReadRows` and `Argmax` are the only readbacks.

  X1 registers its factories with `Backends.register` and reuses `Forward`, `Greedy` and `KvCache` unchanged. X1a's acceptance (tiny model bit-identical to CpuOps F32 over a prefill and 16 decode steps) can compare `IQwenRunner` results or `KvCache.digest` directly.
- **C2** builds a stage executor on `Forward.runStage` with one `KvCache` per stage. It passes the residual as the ops' `Read` and `Upload` (fp32 on fp32 classes), replays by rewinding `Length`, and compares rebuilds with `KvCache.digest`.
- **M2b** adds full-model parity against G2's goldens with the same checks and the same ParityRun pattern, and wires the Qwen tokenizer (M1) into `lrm forward` and `lrm generate --prompt`. Until then, text prompts work only on byte-level models.
- **Reopening.** A new op kind (a fused kernel, a bf16 cache) is an additive member and a contract change. Changing the per-layer order changes every backend's plan and needs a contract change with evidence. So does loosening a lifetime, because a GPU's plan depends on each one.

## Alternatives

- **A tensor-graph IR that backends compile.** It is more general, but it is a second definition of the model to keep equal to CpuRef, and nothing in Phase 1 needs graph rewrites. The static op order gives X1 its plan without it.
- **One forward per backend.** This is the divergence the seam exists to prevent: tap names, residual order and the cache layout would be re-derived each time.
- **A boxed handle (an `ITensor` interface).** It would cost an allocation and a cast per op on the CPU, and would let a backend receive a handle of another backend's type. The type parameter rules out both, and `IQwenRunner` hides it where only results matter.
- **The final norm on the last row only in greedy decoding.** It saves `(T − 1)·hidden` work per prefill, but the metering would then differ from the registry's forward, which the reports and the cross-backend metering rely on.
- **A cache outside the ops** (host arrays that a GPU would mirror). It would force a readback or an upload per step, and double the cache memory on the GPU.
- **Dynamic handles with no lifetimes** (the first draft of this seam). Every op returned a fresh handle, caches were made per run, and taps lived as long as the caller kept them. That is free on the host but impossible under a table sealed once: the executor would have to guess where a forward begins and ends, or Greedy and the runner would need a rewrite for the GPU.
- **An explicit `Release` for every handle.** It is as exact as the bracket, but it puts the burden on every caller, and one missed release leaks a table row for good. The bracket frees intermediates in one place, and the per-stage rule covers what callers keep.
- **A `Plan` member that callers invoke before the first forward.** It is the same information as `OpsOptions`, in a second place that could be called twice or late. Passing it at creation lets the factory seal the table before it returns.
