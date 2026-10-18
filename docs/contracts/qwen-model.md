# ForwardOpsAndQwenModel contract

Owner: M2 (M2a wrote v1; M2b builds on it). Status: v1, proposed with PR #15; it changes only through a `contract-change` PR once it is tagged. Consumers: X1 (`GpuForwardOps` implements `IForwardOps`; `Cmd/GpuBackends.fs` registers into the registry), C2 (`QwenStageExecutor` runs `Forward.runStage` per stage and checks caches with `KvCache.digest`), M2b (full-model parity), I1 to I3 (gate runs, `lrm bisect`), T1 (the inference and parity reports chart the records below). Code: `src/Lrm.Models.Qwen/` and `apps/Lrm.Cli/Commands/{Backends,Forward,Generate}.fs`. Tests: `tests/Lrm.Models.Qwen.Tests/` (cpu tier on the tiny model; model tier on Qwen3-0.6B). Decision record: `docs/decisions/M2-0001-forward-ops-seam.md`.

The model consumes CpuReferenceSemantics (`cpu-reference.md`: every op and the KV-cache dtype), HostTensorAndSafetensors (`host-tensor.md`: the mapped checkpoint), ComparatorsAndGoldenReader (`comparators.md`), ActivationNames (`activation-names.md`: the taps), GoldenSet (`golden-set.md`: model folders), TelemetryFacade (`telemetry-facade.md`) and ExecutionClass (`execution-class.md`). Bit-identity claims below hold within one ExecutionClass only.

## The pieces

| Type or module | What it is |
| --- | --- |
| `Qwen3Config` | The dimensions of a dense Qwen3 decoder, from `config.json` |
| `Qwen3Weights` | A verified model folder: config, mapped `model.safetensors`, REVISION and SHA256SUMS |
| `Qwen3Params`, `Qwen3Model` | The tensor names the forward reads (aliases resolved) and the config: what the forward runs on |
| `RopeConstants` | The pinned RoPE `inv_freq` bits, keyed by (rope_theta, head_dim) |
| `IForwardOps<'t>`, `OpsOptions`, `ForwardCall` | The op seam, the plan its ops are created with, and one forward call as the ops see it. CpuOps implements the seam on CpuRef; X1 implements it on the GPU |
| `Forward` | The forward, written once over the seam: `runStage`, `run`, `runStages` |
| `KvCache<'t>`, `Stage` | One stage's KV cache, and a contiguous layer range |
| `Greedy` | Greedy decoding on the canonical schedule |
| `CpuOps<'T>` | cpu-f32 (`'T = float32`) and cpu-f64 (`'T = float`) |
| `IQwenRunner`, `OpsFactory`, `CpuBackends` | A model bound to one backend with the handle type hidden, and the registry entries |
| `GoldenCheck` | The comparisons of `lrm forward --compare` and `lrm generate --expect` |

## Config

`Qwen3Config.load path` (also `parse json`, `parseBytes source bytes`) reads a Hugging Face `config.json` and raises `Qwen3ConfigException` (a `FormatException`) for anything the forward cannot run. Every rule below is one refusal.

1. **Well-formed.** Valid JSON, an object, and no key twice at the top level or inside `rope_parameters`. System.Text.Json would keep the last duplicate silently.
2. **The model.** `model_type` is `qwen3`, and `architectures` includes `Qwen3ForCausalLM`. `hidden_act`, when present, is `silu`.
3. **Explicit head_dim.** `head_dim` is required, at least 1 and even (NeoX RoPE rotates halves). Qwen3-0.6B has 16 heads of 128 over a hidden size of 1024, so head_dim is never `hidden_size / heads`.
4. **Integers.** `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size` and `max_position_embeddings` are required and at least 1. `num_attention_heads` is a multiple of `num_key_value_heads`.
5. **rms_norm_eps** is required, positive and finite. It is kept as float64; each mode rounds it once to its own type (CpuReferenceSemantics).
6. **rope_theta, either spelling.** `rope_theta` at the top level, `rope_parameters.rope_theta` (transformers 5), or both when they are bit-equal. Neither is a refusal. `rope_parameters` may hold only `rope_type` (which must be `default`) and `rope_theta`: any other field or type changes the table, which the pinned T_spec does not cover.
7. **No other attention.** `rope_scaling` is absent or null. `attention_bias` is not true. `use_sliding_window` is not true. `layer_types`, when present, holds only `full_attention`. `sliding_window` itself is ignored, as transformers ignores it without `use_sliding_window`.
8. **Ids.** `tie_word_embeddings` is false when absent, as in transformers. `bos_token_id` is one id or null. `eos_token_id` is one id, a list of ids or null, kept as `EosTokenIds`.

Other keys are ignored. `Qwen3Config.dims` gives the FLOP registry's `TransformerDims` (RunLog contract).

## Loading: Qwen3Weights.Open

`Qwen3Weights.Open(dir, ?options)` opens a model folder (GoldenSet, "Model folders") and verifies it before anything is used. Structural problems raise `Qwen3WeightsException`; a hash mismatch raises Lrm.Tensor's `IntegrityException` with the file and the expected and actual values.

1. **SHA256SUMS** is required. Its lines are `<64 lowercase hex> *<name>` (binary mode) or `<hex>  <name>` (text mode), with plain file names, each once. It must list `REVISION`, `config.json` and `model.safetensors`.
2. **Every listed file is hashed** before use, and a listed file that is missing is a refusal. `config.json` and `REVISION` are parsed from the very bytes that were hashed.
3. **REVISION** is one line: `local:<name>` (letters, digits, `-`, `_`, `.`), or a 40-digit lowercase commit.
   - **`local:` revisions are verified through SHA256SUMS only** (the tiny model is `local:tiny-g1`).
   - **A commit must be pinned.** `Qwen3WeightsOptions.Pins` (default `Qwen3Pins.all`) lists the accepted commits with the SHA-256 of some files. Each pinned file must be listed in SHA256SUMS with the pinned hash. An unpinned commit is a refusal. `Qwen3Pins.qwen3_0_6b` pins revision `c1899de289a04d12100db370d81485cdf75e47ca`, its `config.json`, `model.safetensors` and `tokenizer.json`; a cpu-tier test checks it against `tools/bootstrap/toolchain.json`.
4. **model.safetensors** is opened with `SafetensorsFile.OpenVerified`: the SHA-256 is computed over the mapped bytes the reader then serves, and the file stays open read-only with `FileShare.Read` until `Dispose`.
5. **The tensors match the config exactly.** Every name of `Qwen3Params.shapes config` is present, BF16, with that shape. Any other tensor, apart from `lm_head.weight`, is a refusal.
6. **lm_head.** One tensor serves the head when the embeddings are tied:

   | lm_head.weight | tie_word_embeddings | Result (`LmHead`) |
   | --- | --- | --- |
   | missing | true | `Missing`: the head reads `model.embed_tokens.weight` (the tiny model, Qwen3-0.6B-Base) |
   | present, SHA-256 equal to the embedding's | true | `AliasVerified`: checked once at open; the head reads the embedding, and lm_head is never copied (Qwen3-0.6B) |
   | present, different | true | refused |
   | present | false | `Separate`: its own tensor, shape `[vocab, hidden]` |
   | missing | false | refused |

**Reading tensors.** `Bf16 name` returns a `HostTensor<uint16>` copied out of the mapping once and shared by every caller; `Span name` returns the mapped bytes without a copy (valid until `Dispose`), for a GPU to stream. Both resolve `lm_head.weight` to the embedding when the head is tied, and both refuse a name the forward does not read. `Names` lists the tensors the forward reads, in canonical order (310 for Qwen3-0.6B, 24 for the tiny model). `Sums` holds every verified SHA256SUMS line, and `Sha256` the checkpoint's.

## RoPE: RopeConstants

torch's float32 `inv_freq` differs from every closed-form F# formula in some entries (ReferenceOracles), so the bits are data. `RopeConstants.entries` holds them by (rope_theta, head_dim), with theta matched bit for bit:

| rope_theta | head_dim | Model | Words | SHA-256 of the words, u32 little-endian |
| --- | ---: | --- | ---: | --- |
| 1e6 | 32 | tiny | 16 | `7f7aa6f39d2fd0a08c069fbf6d7cf6679288240227d78c053382b884db09abee` |
| 1e6 | 128 | Qwen3-0.6B | 64 | `138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e` |

- `tryFind theta headDim` returns a copy or None. `invFreqBits` raises `KeyNotFoundException` for a pair without bits, naming the pinned ones. `forConfig` looks up a config.
- `table exec config maxPos` is CpuRef's `RopeTable.create`: T_spec, computed in exact integer arithmetic (ADR O2-0001), so the same bytes on every machine, runtime and thread count. A table of N positions is the prefix of every larger one.
- A new model size needs its bits dumped by the golden tools (ReferenceOracles, T_spec) and added here.

## The op seam: IForwardOps<'t>

`'t` is the backend's activation handle: `HostTensor<float32>` or `HostTensor<float>` for CpuOps, a tensor token for a GPU. Every handle is a row-major `[rows, cols]` tensor in the ops' `DType`. The ops are `IDisposable`.

| Member | Semantics (the CpuRef op it is, CpuReferenceSemantics) |
| --- | --- |
| `ExecutionClass` | The class whose bits these ops produce |
| `DType` | F32, or F64 for cpu-f64. Activations, caches and the stage residual are in it |
| `MaxContext` | The positions the RoPE table covers; a cache holds at most this many |
| `Rows x`, `Cols x` | The rows and the columns of a handle |
| `BeginForward call` | Starts a forward (`ForwardCall`: `Stage`, `Rows`, `Pos0`, `Head`, `Taps`). Raises `ForwardException` for a call outside the plan or inside another forward. A GPU picks or records the call's plan here |
| `EndForward result` | Ends it: `Some` the result when the forward completed, `None` when an op raised |
| `Embed ids` | `embed`: `[ids, hidden]`. An id outside the vocabulary raises `ArgumentOutOfRangeException` |
| `RmsNorm(x, weight, eps)` | `rmsNorm`, over rows of the weight's length, so `[T, heads·hd]` with an `hd` weight is the per-head q_norm and k_norm. `eps` is rounded once to `DType` |
| `MatMul(x, weight)` | `matmulBf16W`: `x [T, k]` times `w [n, k]` transposed |
| `Rope(x, heads, pos0)` | `ropeApply` with the pinned table; row t is at position pos0 + t |
| `NewCache(stage, layer, rows)` | A zero tensor `[rows, KvHeads·HeadDim]` in `DType` for one layer of a planned stage: its K or V (a GPU takes it from its KvCache arena) |
| `ReleaseCache x` | Returns a cache tensor to the ops; the handle is invalid afterwards |
| `KvWrite(cache, x, pos0)` | `copyRows`: x's rows into cache rows pos0 .. pos0 + T − 1, bit for bit |
| `Attention(q, k, v, shape, past)` | `attention` with `attentionScale head_dim`, over cache rows 0 .. past + T − 1 |
| `SiluMul(gate, up)`, `Add(a, b)` | `siluMul`, `add` |
| `LastRow x` | The last row as `[1, cols]` |
| `Argmax x` | `argmaxRows`: per row the lowest maximal index, or `0xFFFFFFFF` for a row holding a NaN. Reads back |
| `Tap(name, x)` | A snapshot of x that later ops never change, valid as long as the forward's result (below). A GPU copies it into its Taps arena when the op is recorded; CpuOps returns x itself, because CpuRef results are fresh arrays and the forward never taps a cache. Called only for a tap the call's filter wants, which the plan's filter covers |
| `Read x` | x on the host, `[rows, cols]` in `DType`: a copy the caller owns. Reads back |
| `ReadRows(x, row0, count)` | Rows row0 .. row0 + count − 1 of x on the host, count ≥ 1. Reads back only those rows |
| `Upload x` | A handle holding a host tensor `[rows, cols]` in `DType` (a stage boundary's residual). Another dtype or rank raises `ArgumentException` |

**Rules a backend relies on.**
- **No hidden readback.** Every op but `Argmax`, `Read` and `ReadRows` returns its handle at once. The forward never reads a value back in the middle of a forward.
- **A static order.** The forward calls the ops in one order that depends only on the model and the `ForwardCall` (stage, T, pos0, head rows, tap filter), never on values (a cpu-tier test records two prompts and compares). A backend can record a plan.
- **Weights by name.** Ops are created over one verified `Qwen3Weights` and resolve names themselves. The forward passes `Qwen3Params` names, so a tied lm_head arrives as the embedding's name.

### Lifetimes and the plan

A GPU executor seals one tensor table before its first dispatch (HeapLayoutAndTensorTable: 16,383 rows, written once). The seam therefore fixes, before any forward, everything the ops will be asked for, and when each handle ends.

**The plan.** The ops are created with `OpsOptions`, which bound every call; a GPU sizes and seals its heap and table from them. `OpsOptions.validate` refuses a plan with `ArgumentException`, and a call outside it raises `ForwardException` before any op runs.

| Field | Bounds | `OpsOptions.defaults` |
| --- | --- | --- |
| `Threads` | CPU threads (the bits are the same at every count) | 8, or fewer cores |
| `MaxContext` | Positions: the RoPE table, and each cache's rows | `max_position_embeddings` |
| `MaxRows` | The rows T of one forward | `max_position_embeddings` |
| `Stages` | The stages these ops run, with distinct indices | the whole model as stage 0 |
| `Taps` | The taps any forward may record (`TapFilter.Covers`); ActivationNames names only | `Off` |
| `Caches` | The `KvCache`s of each planned stage alive at once | 1 |

**Handle lifetimes.** `Forward.runStage` brackets each forward with `BeginForward` and `EndForward`.
- **Where ops run.** The compute ops (`Embed` to `LastRow`, and `Tap`) run only inside a forward. `NewCache`, `ReleaseCache` and `Upload` run only outside one. `Rows`, `Cols`, `Argmax`, `Read` and `ReadRows` work on any live handle at any time.
- **Intermediates**, the handles the ops return inside a forward, are valid until its `EndForward`.
- **The result and the taps** of a forward (the handle `EndForward` receives, and the handles `Tap` returned) stay valid until the next `BeginForward` of a stage with the same `Index` on these ops. So a stage's result can be the next stage's input without the host, the taps of every stage of a `runStages` call can be read after it, and a greedy step's logits are read before the next step begins.
- **A failed forward** ends with `EndForward None`; every handle it made, taps included, is invalid.
- **An upload** stays valid until the next forward ends; a second `Upload` before then invalidates it.
- **A cache tensor** stays valid until `ReleaseCache`. `KvCache.Dispose` releases a cache's tensors; `Greedy.run` and `IQwenRunner.Forward` release theirs before they return.
- **Dispose** releases everything; the ops are unusable after it. One ops instance serves one caller at a time.

**CpuOps enforces all of it**, although host memory would not need it: a call outside the plan, an op outside a forward, a handle past its lifetime, or more caches than planned raises `ForwardException`, and anything after `Dispose` raises `ObjectDisposedException`. Code that runs on CpuOps (Forward, Greedy, the runner, the commands and the tests) therefore keeps to what a static GPU plan can serve. With these rules a GPU recycles each intermediate's token by its place in the static order, keeps the result and taps of each planned stage in fixed slots, and pools `Stages × Caches` cache sets, so its table never grows after it is sealed.

## The forward

`Forward.runStage ops model stage cache pos0 input options` runs one stage over T rows at positions pos0 .. pos0 + T − 1.

**Per layer i**, in this order, each op's result tapped under its ActivationNames name when the filter wants it:

| # | Call | Tap `L{i}.` |
| ---: | --- | --- |
| 1 | `RmsNorm(x, input_layernorm)` | `input_layernorm.out` |
| 2 to 4 | `MatMul(h, q_proj)`, `MatMul(h, k_proj)`, `MatMul(h, v_proj)` | `attn.q_proj.out`, `attn.k_proj.out`, `attn.v_proj.out` |
| 5, 6 | `RmsNorm(q, q_norm)`, `RmsNorm(k, k_norm)` | `attn.q_norm.out`, `attn.k_norm.out` |
| 7, 8 | `Rope(qn, heads, pos0)`, `Rope(kn, kv_heads, pos0)` | `attn.q_rope`, `attn.k_rope` |
| 9, 10 | `KvWrite(K_i, k_rope, pos0)`, `KvWrite(V_i, v, pos0)` | `attn.v_cache` (the v_proj output, which the cache stores raw) |
| 11 | `Attention(q_rope, K_i, V_i, shape, pos0)` | `attn.o_proj.in` |
| 12 | `MatMul(o, o_proj)` | `attn.o_proj.out` |
| 13 | `Add(x, o_proj.out)` | `resid_mid` |
| 14 | `RmsNorm(resid_mid, post_attention_layernorm)` | `post_attention_layernorm.out` |
| 15, 16 | `MatMul(h2, gate_proj)`, `MatMul(h2, up_proj)` | `mlp.gate.out`, `mlp.up.out` |
| 17 | `SiluMul(gate, up)` | `mlp.down.in` |
| 18 | `MatMul(d, down_proj)` | `mlp.down.out` |
| 19 | `Add(resid_mid, down.out)` | `out` |

**The ends.**
- **Input.** The stage that starts at layer 0 takes `StageInput.Ids` and runs `Embed` first (tap `embed.out`). Every later stage takes `StageInput.Hidden`, the residual stream `[T, hidden]` in `DType`. The wrong kind, or a residual whose width is not `hidden`, raises `ForwardException` before any op (a `[T, 2·hidden]` residual would otherwise pass the first RmsNorm as two per-head norms).
- **Head.** The stage that ends at the last layer runs `RmsNorm(x, model.norm)` on every row (tap `final_norm.out`, `[T, hidden]`), then, for `HeadRows.Last` with T > 1, `LastRow`, then `MatMul(·, LmHead)` (tap `logits`). It returns the logits: `[T, vocab]` for `HeadRows.All`, `[1, vocab]` for `HeadRows.Last`. The final norm runs on every row because the FLOP registry's forward does. Other stages return the residual after their last layer.
- **Taps** go to `options.Taps` (a `TapStore` with a `TapFilter`: `Off`, `All` or `Names`), in forward order, as snapshots. A stage emits only its own taps (ActivationNames, "Two-stage splits"). A `Names` filter with a name that is not an ActivationNames tap of the model (`TapFilter.unknown`) raises `ForwardException`, so a misspelt tap is never silently not recorded.

**Positions and the cache.**
- T ≥ 1, `0 ≤ pos0 ≤ cache.Length` and `pos0 + T ≤ cache.MaxPos`; otherwise `ForwardException`, before any op runs.
- The stage writes cache rows pos0 .. pos0 + T − 1 of each of its layers and sets `cache.Length` to pos0 + T. A pos0 below `Length` rewrites from there: a replay.
- A call refused before its first op (by these checks or by the plan in `BeginForward`) changes nothing. Once the ops run, `Length` is pos0 until the forward completes, so a forward whose op raises leaves it at pos0.
- The cache must hold exactly the stage's layers, and must not be disposed.

`Forward.run` is the whole model as one stage. `Forward.runStages ops model stages caches pos0 ids transfer options` runs stages that cover the model in order, each with its own cache, passing the residual through `transfer` at each boundary; `Forward.hostRoundTrip ops` is `Upload ∘ Read`, what a pipeline node does. It checks every cache before the first stage runs. If a stage or a transfer raises, every cache's `Length` is pos0: the rows from pos0 on are then valid in none of them, and the pipeline is consistent for a replay from pos0.

**Phase.** `Forward.phaseOf rows pos0` is `decode` for a single-token call after position 0, else `prefill`. A one-token prompt at position 0 is a prefill; a chunk after position 0 is a prefill.

## The KV cache

- **Layout.** `KvCache<'t>` holds, per layer of one stage, K after k_norm and RoPE and V raw, each `[MaxPos, KvHeads·HeadDim]`, row p for position p. Rows 0 .. `Length` − 1 are valid. Rows beyond may hold anything and are never read.
- **Dtype.** The cache is in the ops' `DType` (CpuReferenceSemantics): fp32 for cpu-f32 and every GPU, fp64 for cpu-f64. An fp32 cache in cpu-f64 would break the 1e-12 chain to torch64p.
- **Creation and release.** `KvCache.create ops model stage maxPos`, for a planned stage, with maxPos in 1 .. `ops.MaxContext`. The cache holds tensors of its ops; `Dispose` returns them (`ReleaseCache`), after which the cache refuses use (`ObjectDisposedException`, and `ForwardException` from a forward). A `KvCache` built over tensors the caller owns releases nothing on `Dispose`.
- **Length.** The forward sets it. A caller sets it directly only to rewind, or after filling rows another way, such as restoring a snapshot. It must lie in 0 .. MaxPos.
- **Digest.** `KvCache.digest ops cache` is the SHA-256 of:
  - the ASCII tag `lrm.kvcache/1` and one 0 byte;
  - FirstLayer, LayerCount, Length and the DType code (LrmCore), each as u32 little-endian;
  - then for each layer in order, K rows 0 .. Length − 1 and then V rows 0 .. Length − 1, as their little-endian element bytes.

  Within one class, two caches have the same digest exactly when their valid rows are bit-identical. C2 compares a fast rebuild with exact replay by digest. The digest reads back only the valid rows (`ReadRows`), never a whole `[MaxPos, cols]` tensor.

## Stages

`Stage` is `{ Index; FirstLayer; LastLayer }` (inclusive). `Embeds` holds for the stage that starts at layer 0, and `Stage.hasHead config stage` for the one that ends at the last layer.
- `Stage.whole config` is the model as stage 0.
- `Stage.split config boundaries` gives the stages that start at layer 0 and at each boundary. Boundaries increase strictly inside 1 .. layers − 1, and anything else raises `ArgumentException`. The plan's split of Qwen3-0.6B is `split c [ 14 ]`: layers 0 to 13 with the embedding, and 14 to 27 with the head.
- **The boundary carries the residual stream in the ops' DType**: fp32 on every fp32 class (cpu-f32, cpu-device, the GPUs), which is the plan's fp32 residual, and fp64 on cpu-f64.
- **Composition is exact.** Within one class, stages composed through `hostRoundTrip` give the same logits, taps and cache rows as one stage, and the residual at boundary b is the `L{b−1}.out` tap.

## Greedy decoding

`Greedy.run ops model prompt options` follows the canonical schedule (ReferenceOracles):
1. One prefill of the prompt with `HeadRows.Last`.
2. Then single-token decode steps, each feeding back the previous step's argmax (lowest index on ties).

It returns `GreedyResult { Prompt; Ids; Margins; StoppedAtEos }`.
- **Steps.** `options.Steps` ≥ 1 ids are generated: the prefill gives the first, each decode step one more. The cache has exactly `Greedy.positions prompt steps` = prompt + steps − 1 positions, which must fit `ops.MaxContext`; it is released when the run ends. The ops' plan must cover a forward of the prompt's rows on the whole model.
- **EOS.** `StopAtEos = true` stops after the first id in the config's `EosTokenIds` and keeps it as the last id. **Parity runs pass false** (`GreedyOptions.parity`): the goldens ignore EOS. On the tiny model, p0 stops after 10 ids, the tenth being EOS 7, which is the golden's first 10.
- **NaN.** A step whose logits hold a NaN (`Argmax` gives `0xFFFFFFFF`) raises `ForwardException` naming the step.
- **Margins.** With `Margins = true`, each step's top-1 minus top-2 logit, in float64 from `DType` (a full logits readback on a GPU).

## CpuOps and the registry

- **CpuOps.** `CpuOps.f32 weights options` and `CpuOps.f64 weights options` are cpu-f32 and cpu-f64, created with the plan (`OpsOptions`, "Lifetimes and the plan").
  - The class is `ExecutionClass.forCpu`.
  - Every op is CpuRef's, so the bits are the same at every thread count.
  - The RoPE table covers 0 .. MaxContext − 1.
  - `Read` and `ReadRows` return copies. `Upload` refuses another dtype.
  - It enforces the plan and the lifetimes exactly.
- **IQwenRunner.** A model bound to one backend with the handle type hidden. It has Backend, ExecutionClass, DType, Model, MaxContext, `Forward(ids, taps, head)` (a prefill into a fresh cache that it releases, results on the host) and `Greedy(prompt, options)`. It is `IDisposable` and owns its ops: `Dispose` releases them (a GPU's weights and heap), so callers `use` it. `QwenRunner.ofOps backend model ops` builds one over any ops.
- **OpsFactory** is `{ Name; Summary; Create: Qwen3Weights -> OpsOptions -> IQwenRunner }`; the options are the runner's plan, and the caller disposes the runner. `CpuBackends.cpuF32` and `cpuF64` are named by their ExecutionBackend names.
- **The registry** is `apps/Lrm.Cli/Commands/Backends.fs`. It starts with cpu-f32 and cpu-f64. `Backends.register factory` adds one and refuses a name twice; GpuBackends.fs (X1) registers cpu-device, vulkan and d3d12 there. `Backends.all ()` and `tryFind name` read it.

## The commands

| Command | What it does |
| --- | --- |
| `lrm backends` | Lists the registry |
| `lrm forward` | A prefill from position 0 on one backend: prints the logits' shape and top-5 and the argmax per row. `--taps` takes `none`, `all` or ActivationNames names; any other name is refused (exit 2). `--out FILE` writes `input_ids` (I64 `[T]`), the taps and `logits` in the golden `fwd_*` layout, with `lrm.*` metadata. `--compare FILE` measures every written tensor the file also holds (relL2, plus max \|Δ\| and KL for logits) as `golden.compare` checks and exits 1 when one fails (`GoldenCheck.tensors`). A tensor whose shape differs from the golden's is a failed check, never skipped; with `--require-taps`, so is every tensor of the file but `input_ids` that the run did not produce. A file whose `input_ids` differ from the prompt is refused (exit 2) unless `--allow-other-ids` is given |
| `lrm generate` | Greedy decoding on one or more backends (`--backend cpu-f32,cpu-f64`); prints the ids, the text for a byte-level model, the margins and the tokens/s of prefill and decode (`--warmup N` runs N untimed, unrecorded steps first). `--expect FILE:TENSOR` compares the ids generated with the tensor's first ids (a `greedy_match` check, `GoldenCheck.greedy`): `--steps 16` against a 64-id golden compares 16, an EOS stop compares the ids up to it, and `--steps` beyond the golden's length is refused (exit 2). It exits 1 on a mismatch |

**Shared options.**
- `--model DIR`: `tiny` is `goldens/tiny/g1/model`, and `qwen3-0.6b` (the default) is `LRM_DATA/models/Qwen3-0.6B`.
- `--backend NAME`.
- The prompt, given once: `--ids 1,2,3`, `--ids-from FILE[:TENSOR]` (an integer tensor, `input_ids` by default) or `--prompt TEXT`.
- `--threads N` (8 by default).
- `--run-log`: a run under `LRM_DATA/runs`, of kind `inference`, or `parity` for `lrm forward --compare`; `--title TEXT` names it. Each backend gets its own run with its own ExecutionClass (its parameters name the `backend`), so an inference report's headline rates are one class's; `lrm generate` on several backends writes one run each, titled `..., on <backend>`.

**Text prompts.** `--prompt` is UTF-8 bytes for a byte-level model: a vocabulary of 256 and no `tokenizer.json`, which is the tiny model. Any other model needs the Qwen tokenizer (TokenizerApi, M1), which this version does not use yet; such a model takes `--ids` or `--ids-from`.

**Bounds for `--compare`.**
- **The pair** comes from the backend and the file's dtype: an F32 file is torch32 and an F64 file torch64p, so cpu-f64 on F64 is `cpu64-vs-torch64p` and cpu-f32 on F32 is `cpu32-vs-torch32`. `--pair` overrides it.
- **The bounds** are, in order of precedence: `--bound` (relL2 for every tap); `--tolerances FILE` (ToleranceSpec); the pair's fixed bound or floor. Those are cpu64 relL2 1e-10 (the full-model bound; pass `--bound 1e-12` on the tiny model); cpu32 relL2 1e-5, max \|Δlogit\| 2e-3 and KL 1e-7; GPU 1e-4, 1e-2 and 1e-6.

## Telemetry

The forward emits these through the ambient facade (TelemetryFacade; reserved prefix `model.*`). Telemetry never changes a value: a cpu-tier test runs greedy decoding with the no-op facade, a strict in-memory sink and a run log, and compares the ids and margins bit for bit.

| Record | Kind | Attributes | Charted by |
| --- | --- | --- | --- |
| `model.generate` | span | `component` = `generate`, `prompt_tokens`, `steps`, `stop_at_eos` | inference: `meters` |
| `model.forward` | span, one per runStage | `component` = `forward`, `phase` (`prefill` or `decode`), `step` (greedy step, 0 the prefill), `tokens` (T), `pos0`, `stage`, `first_layer`, `last_layer`, `head_rows` (on the head stage) | inference: `prefill_rate`, `decode_rate`, `ttft`, `decode_latency`, `latency_hist`, `phases` |
| `model.layer` | span, a child of its model.forward | `layer`, `phase`, `tokens`; `component` = `forward` and `stage` from the scope | inference: `layer_time` |
| `model.flops`, `model.bytes_moved` | meter (`flop`, `B`) | `op` (the registry name), `layer` inside a layer, `component`, `stage` | inference: `flops_per_token`, `achieved`, `meters` |
| `model.tokens` | meter (`token`) | T, once per forward of the stage with the head | inference: `flops_per_token`, `meters` |

- **Metering.** Every op is metered by the FLOP registry (`Metering.op "model"`, RunLog contract, "FLOP registry"), so per forward the meters equal `OpCost.forward dims T pos0 headRows`, with the argmax metered by Greedy (`component` `forward`, the stage's `stage`). CpuRef, CpuDevice and the GPUs therefore meter the same work identically. Bytes follow the registry's fp32-activation convention in every class.
- **Attribution.** The forward sets `component` = `forward` and `stage` with `Telemetry.scope`, and `layer` inside each layer, so CpuRef's `cpuref.*` spans and the meters attribute per layer and stage.
- **Comparisons** by the tests and `lrm forward --compare` and `lrm generate --expect` are `golden.compare` checks with ComparatorsAndGoldenReader's keys: `pair`, `backend` (the registry name, which the gate1 template groups by), `prompt`, `tap`, `layer` (−1 for model-level taps and `greedy.ids`), and `kind` = `chain`; `greedy_match` also carries `steps`. A comparison that measured nothing (a shape mismatch, a required tap the run did not produce) is relL2 = NaN, which never passes and which the parity report sorts first, with the reason in `detail`.
- **The bit-identity proofs** of the tests (prefill = decode, stages, threads) are `golden.compare` `exact` checks with a `check` and no `tap`, which the parity report lists in its self-test table.
- **Greedy margins** of the tests are `goldens.greedy.min_margin` gauges with `prompt` and `oracle` = the backend name, which the parity report charts.

## Vectors

**RoPE** (ReferenceOracles vectors; the tables are CpuRef's):
- tiny, positions 0..511: digest `615b420404dd336bbf4509e20dd121daefdcc0cd8039048ca71097b1b6d60801`, equal to `goldens/tiny/g1/rope.safetensors` bit for bit;
- Qwen3-0.6B, positions 0..8191: digest `da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf`;
- the entry vectors (`(1, 0)` → cos `3f0a5140`, sin `3f576aa4`, and the rest of the ReferenceOracles table).

**Pins.** `Qwen3Pins.qwen3_0_6b`:

| File | SHA-256 |
| --- | --- |
| `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

**KvCache.digest**, computed independently with Python's `hashlib` and `struct`. The cache has FirstLayer 3, two layers, MaxPos 4 and Cols 2:
- layer 3: K rows (1, 2), (3, 4); V rows (−1, −2), (−3, −4);
- layer 4: K rows (0.5, 0.25), (0, −0); V rows (8, 16), (32, 64);
- rows 2 and 3 hold 99, which the digest ignores.

| Vector | DType | Length | Digest |
| --- | --- | ---: | --- |
| K1 | F32 | 2 | `fe32a706cc6ab5e9638beaa88e0a2b4c9a589195986c4c493bd959b5f5abf893` |
| K2 | F64 | 2 | `a8b5f2a491a1feb61f56e5d9d83be920b69961bc2062350a2127ad8cb386b3c0` |
| K3 | F32 | 1 | `6091fb9e00a1d9c26335f9bab1c10d1e157ab531de22bdada3a144d5f9de401f` |
| K4 | F32, the tiny model's empty cache (FirstLayer 0, two layers) | 0 | `6512ddd1f2a5f8be07b058ced48aa376662e2564eddaecffbae06c89a9591f9b` |

**Decision tables.**

| Input | Result |
| --- | --- |
| `Stage.split` Qwen3-0.6B `[14]` | (0, 0..13), (1, 14..27) |
| `Stage.split` Qwen3-0.6B `[7; 14; 21]` | four stages of 7 layers |
| `Stage.split` with 0, 28 or a repeated boundary | `ArgumentException` |
| `phaseOf` (13, 0); (1, 0); (1, 13); (6, 64) | prefill; prefill; decode; prefill |
| Ops per forward of T > 1 rows on the tiny model, head on the last row | 42: embed, 19 per layer, last row, final norm, lm_head |

**FLOPs**, computed independently from the registry's formulas. Greedy decoding of tiny p0 for 4 steps (rows/past (13, 0), (1, 13), (1, 14), (1, 15), head rows 1) meters `model.flops` = 3,508,960: 2,756,290 + 249,842 + 250,890 + 251,938.

**Acceptance on the tiny model** (`goldens/tiny/g1`; measured, informative):

| Check | Bound | Measured |
| --- | --- | --- |
| cpu-f64 vs torch64p, all 39 taps (p0, p1) | relL2 ≤ 1e-12 | worst 1.5e-15 (`L1.mlp.down.out`) |
| cpu-f64 vs torch64p, logits (p2, 70 tokens) | relL2 ≤ 1e-12 | 1.2e-15 |
| cpu-f32 vs torch32, all 39 taps (p0, p1) | relL2 ≤ 1e-5 | worst 8.1e-7 |
| cpu-f32 vs torch64p, all 39 taps (p0, p1) | relL2 ≤ 1e-5 | worst 8.8e-7 |
| cpu-f32 logits (p0, p1, p2) | max \|Δ\| ≤ 2e-3, KL ≤ 1e-7 | 1.2e-5, 8.0e-12 |
| Greedy, 16 and 64 steps (p0, p1) | identical | identical in both modes; min margins 0.0567 (p0), 0.0978 (p1) |
| Prefill = decode = chunked prefill (p2: 70 rows, 64 + 6, 10 + 31 + 1 + 28) | bit-exact | logits, caches and every tap, both modes |
| Two stages through the host = one stage (p0, prefill + 16 steps) | bit-exact | logits and caches, both modes |
| 1 = 7 = 32 threads (p2) | bit-exact | logits and every tap, both modes |

**On Qwen3-0.6B** (model tier, measured):
- it opens at the pinned revision with 10 SHA256SUMS files verified and lm_head `AliasVerified`;
- a 4-token prefill equals 4 decode steps bit for bit, in both modes;
- greedy decoding after ids `785 3974 13876 38835` on cpu-f32 gives `34208 916 279`.

## Verification

The cpu tier of `Lrm.Models.Qwen.Tests` covers:
- **Config:** every field of Qwen3-0.6B's and the tiny config; both rope_theta spellings; eos lists; 18 refusals, one per rule; the registry dims.
- **RoPE:** the bit digests; the tiny bits against `rope.safetensors`, and the Qwen bits against the planning probe `spikes/probes/redteam/inv_torch.txt`; both table digests; the entry vectors; the refusal; copies.
- **Loading:** the tiny folder and its manifest hash; shared tensors and the lm_head alias. Refusals on tampered copies: a flipped byte, a changed config, an unlisted, missing or malformed SHA256SUMS line, a missing SHA256SUMS, a bad or unpinned REVISION, a pin that disagrees. lm_head equal, different, untied and missing; extra, missing, misshaped and F32 tensors; the pin against `toolchain.json`.
- **The tiny acceptance** above, each check a `golden.compare` record in the parity run log (`ParityRun`, nodes n0 cpu-f64 and n1 cpu-f32). The run goes to a temporary folder that the exe deletes, or, with `LRM_WRITE_PARITY_RUN=1`, under `LRM_DATA/runs` for `lrm report`. Also the tap order, shapes and dtypes; the tap filter; StopAtEos on p0; and the runner and both factories against the generic forward, bit for bit.
- **The seam** (`SeamTests`): the bracket of every forward, and a refused call reaching no op; an intermediate ending with its forward, and the result and taps with the stage's next forward; two stages on one ops; ops outside a forward and cache ops inside one; each bound of the plan; caches released by `Dispose`, Greedy and the runner (two runs in one planned cache); uploads; `Dispose`; `ReadRows`; `runStages` rolling every cache back to pos0.
- **The comparisons** (`GoldenCheckTests`): every tap of p0 passing; a shape mismatch failing; a last-row head against the golden's last row; required taps; `greedy_match` of 16 ids against the 64-id golden, a mismatch, and too many ids.
- **Determinism:**
  - prefill = decode = chunked prefill;
  - the last-row head;
  - two stages, with the stages' taps and the boundary residual;
  - Stage.split;
  - thread counts;
  - the caches against the k_rope and v_cache taps;
  - a replay after a rewind;
  - the KvCache vectors K1 to K4;
  - the static op order;
  - telemetry changing no value;
  - the forward's refusals, a residual of another width and an unknown tap name among them;
  - a NaN stopping greedy decoding.
- **Telemetry:** the spans and their attributes; the meters against `OpCost.forward`; stages and tokens; a real run log that the inference template renders with data in every chart the forward feeds.

The model tier opens Qwen3-0.6B and runs the short forwards above. Full-model parity against G2's goldens is M2b's.
