# ConformanceHarness contract

Owner: E1. Status: v1, a draft until the next contracts tag (wave 2; README, "Freeze tags"). Consumers: K1, K2 and K3 (they write cases), D1, V1 and R5 (they run the harness on their backends), X1 (it runs on CpuDevice), I1 and I3 (the conformance matrix and the `conformance` step of `lrm gate1`), T1 (the kernels report). Code: the harness in `tests/Lrm.Kernels.Conformance/Harness/` (`Case.fs`, `Poison.fs`, `References.fs`, `Session.fs`, `Discovery.fs`), its suites in `Suites.fs`, the selftest cases in `Cases/selftest/`, the harness's own cases for K1 to K3 in `Cases/harness/`; CpuDevice in `src/Lrm.Gpu.CpuDevice/` (`KernelRules.fs`, `Emulators.fs`, `CpuDevice.fs`). Tests: `tests/Lrm.Kernels.Conformance/HarnessTests.fs` and the CpuDevice suite every run executes (cpu tier), `tests/Lrm.Gpu.CpuDevice.Tests` (cpu tier).

The harness judges every catalog kernel (KernelAbiAndCatalog) on every backend the same way. A case gives a kernel, its tensors and its scalar arguments; the harness derives everything else: the work's sizes, the reference, the bytes to poison and compare, the batch-invariance splits and the bound. It runs against any `IGpuBackend` (GpuBackendInterface): CpuDevice in CI, D3D12 and Vulkan locally once D1 and V1 exist.

## Cases: KernelCase and CaseIo

```fsharp
type CaseData = Input of bytes: byte[] | Output of elements: int64 | Meta of elements: int64
type CaseTensor = { Field: string; DType: DType; Data: CaseData; Dims: int64 list option }
type CaseIo = { Tensors: CaseTensor list; InPlace: (string * string) list; NullMeta: string list }
type KernelCase = { Kernel: string; Name: string; Io: CaseIo; Scalars: (string * uint32) list; Doc: string }
```

- **One tensor per token field.** `Tensors` binds each token field of the kernel to a tensor: an `Input` (its little-endian element bytes) for a read field, an `Output` (its element count) for a written field, a `Meta` tensor (only its table row is read) or the null token (`NullMeta`) for a meta field. `InPlace` passes a written field the token of a read field, and only where the catalog declares the alias. Every token field is bound exactly once.
- **Scalars.** Every u32, i32 and f32 field has exactly one value, given as its argument word (an f32 as its raw bits, so NaN payloads, signed zeros and denormals pass unchanged). Names are the catalog's.
- **Counts are capacities.** A tensor may hold more elements than the dispatch touches (KernelAbiAndCatalog, tensor access). `Dims` (default `[elements]`) may be anything whose product is the count: kernels never read the dims of a data token.
- **Written tensors** have 4-byte elements, and a kernel writes exactly one field; both hold for the whole v1 catalog.
- **Validation.** `Session.Create` raises `ArgumentException` naming the case and the rule for: an unknown kernel, a kernel without a KernelBounds entry, a name that is empty or holds `/`, a field that is not a token field, data of the wrong kind for the field's access, a dtype the field does not accept, input bytes that are not a positive multiple of the element size, an output of no elements, dims that do not give the count, an unbound or doubly bound field, an undeclared alias, a missing, repeated or unknown scalar, and two cases with one id (`<kernel>/<name>`).
- **Builders.** `Case.create kernel name |> Case.f32 "x" xs |> Case.output "y" n |> Case.u32 "n" n |> Case.f32Arg "a" a`; also `f32Bits`, `bf16`, `u32s`, `i32s`, `inputBytes`, `outputOf`, `outputU32`, `meta`, `nullMeta`, `inPlace`, `dims`, `i32`, `f32ArgBits` and `doc`. `Case.gpuOnly` marks a case too large to emulate in CI (an lm_head-sized matmul with many rows, say): CpuDevice never plans it and its suite skips it with the reason; GPU backends run it. `CaseGen` gives deterministic data (DetRng seed `0x4C524D2D45314341`, stream `DetRng.streamOf("conformance." + name)`).

## Where cases live

Each family writes its cases under `tests/Lrm.Kernels.Conformance/Cases/<family>/`, owned by the family: `basic` (K1), `matmul` (K2), `attention` (K3), `selftest` and `harness` (E1). A case file is a module `Lrm.Kernels.Conformance.Cases.<Name>` with a public value `cases: KernelCase list`. The project compiles `Cases/**/*.fs` by glob, after the harness, and `Discovery.cases` collects every such module's list in ordinal order of the module names, so a family adds a file and nothing else. Case files depend only on the harness, never on each other.

## A session

`Session.Create(backend, kernels, cases, options)` takes a backend whose kernels are loaded (`kernels` is what `LoadKernels` returned, so a caller can open a run log for the class first) and whose heap does not exist; `Session.Load` loads the embedded shader set first. It then:

1. **Plans** every case and the probe into one heap and one table, which are written once (HeapLayoutAndTensorTable): each tensor gets a region in the Scratch arena (HeapPlanner) of 16 guard words, its elements in whole words, and 16 guard words; its row starts after the first guard. An input that no written field is passed gets the `ReadOnly` flag. Names are `<kernel>/<case>/<field>`.
2. **Adds the split views** (below) as `TensorTable.AddView` rows, one per (tensor, element offset, length).
3. Creates the heap, writes the table, and **measures the fp32 denormal mode** with the probe of GpuBackendInterface (`scale_f32`, a = 0.5, `Fp32Denormals.probeInputs`), run twice with poisoned outputs: it fails when a word keeps its poison or the runs differ, and otherwise records the mode with `Fp32Denormals.recordProbe` (`gpu.fp32_denormals`). The mode decides every reference below.

`session.Contains case` is false when the backend's kernel set lacks the case's kernel (a GPU backend before K1 to K3 add the source); the suites skip such cases.

## Judging a case

`session.Run case` performs these steps and returns a `CaseOutcome`. A case passes when no step fails.

1. **The reference.** CpuRef F32 (and for relative kernels CpuRef F64, inputs widened exactly) on the case's inputs, through the CpuRef op the kernel implements, in the class's measured mode (below). The touched elements of the written field are its write footprint (CpuDevice's `KernelRules.footprint`, from the u32 fields alone).

   | Kernel | Reference |
   | --- | --- |
   | `scale_f32`, `add` | one float32 multiply or add per element |
   | `table_echo` | the 16 words of the planned row of `src` (zeros for the null token) |
   | `args_echo` | the 32 dispatched words, word 0 the physical x |
   | `groups_echo` | y[g] = g |
   | `bf16_unpack` | `Bf16.widenBits` |
   | `embed_gather_bf16` | `CpuRef.embedGather` (an id >= vocab gives `7fc00000`) |
   | `rmsnorm_rows` | `CpuRef.rmsNorm` (F64: eps widened from its f32 word) |
   | `rope_apply` | `Rows.ropeHead` per row at position pos0 + row / heads, from the case's tables (CpuRef.ropeApply's row kernel) |
   | `silu_mul` | `CpuRef.siluMul` |
   | `copy_rows` | the source words at their strides |
   | `argmax_rows` | `CpuRef.argmaxRows` |
   | `matmul_bf16w` | `CpuRef.matmulBf16W` |
   | `attention_gqa_causal` | `CpuRef.attention` on q rows q_row0 .. q_row0 + rows - 1 with the first past + rows cache rows (F64: scale widened from its f32 word) |

2. **Poisoned runs** (the rule of VulkanHeapRecipe section 8 and ADR R1-heap-recipe pitfall 15). Before each run the harness uploads every tensor's whole region: inputs with their data between poisoned guards, outputs poisoned throughout.
   - **Poison words** are float32 signalling NaNs: for a random word r, `(r & 0x80000000) | 0x7F800000 | m` with m = r's bits 0..21, or 1 when those are 0. No arithmetic produces them.
   - **Seeds differ per configuration, case and run:** `Poison.seed configuration case run` = `Canonical.hash "lrm.conformance.poison/1"` over the three strings, where the configuration is the class id (`ExecutionClass.id`, hex), the case its id and the run `run1/<field>`, `run2/<field>` or `split<i>/<field>`. The words are DetRng (Philox key = the seed's bytes 0..7, stream = bytes 8..15, little-endian) drawn in order.
   - **A poison word never equals its expected word:** where the reference's word of a written element equals the poison, the poison's lowest mantissa bit is flipped (mantissa 2 instead of one that would become 0), so it stays a signalling NaN.
   - **In place**, the written field's region holds the input, so the input is its poison; such a case must have at least one touched element whose reference differs from its input, else it fails ("cannot detect a missing write").
3. **Two runs, each checked** after download: every guard word still equals its poison (else "written outside the tensor"); every word of every region that is not a touched element of the written field still equals what was uploaded (else "an input changed" or "written outside the written elements"); and no touched word still equals its poison (else "still hold their poison", gating or not).
4. **Bit-identical twice.** The touched words of run 1 and run 2 are equal.
5. **The bound** (KernelBounds), on run 1: `checkExact kernel 4` against CpuRef F32's words for bit-exact kernels (any NaN matches any NaN for `scale_f32` and `add`), `checkRelative` against CpuRef F32 and F64 for relative kernels.
6. **Strict on CpuDevice.** On a class whose backend is `cpu-device`, run 1 must also equal CpuRef F32's words bit for bit (stricter than the bound, BatchInvariance proof obligation), with the device's arithmetic model applied: a `+ftz` class flushes the fp32 results of relative kernels, and a `+nan-canonical` class turns every NaN result of an fp32 arithmetic kernel into `7fffffff`.
7. **Batch invariance** (BatchInvariance). Each split runs alone, poisoned like a run, and must write exactly its touched elements (the same region checks) with the same bits as run 1 at the same positions. Splits restrict the case to rows [r0, r1): the first row alone, the last row alone and, with at least four rows, rows [rows / 3, 2 rows / 3); a case with one row has none. For `groups_echo` they are the prefixes of 1 and groups / 2 groups. Views keep the split's outputs where the whole case puts them:

   | Kernel | A row | A split restricted to rows [r0, r1) |
   | --- | --- | --- |
   | `scale_f32`, `bf16_unpack`, `silu_mul`, `add` | an element | n = r1 - r0; every data field a view at element r0 (`bf16_unpack`: r0 even) |
   | `embed_gather_bf16` | an id | rows = r1 - r0; `ids` a view at r0, `y` at r0 * cols |
   | `rmsnorm_rows` | a row | rows = r1 - r0; `x` and `y` views at r0 * cols |
   | `argmax_rows` | a row | rows = r1 - r0; `x` a view at r0 * cols, `y` at r0 |
   | `rope_apply` | a token (heads rows) | rows = (r1 - r0) * heads, pos0 + r0; `x` and `y` views at r0 * heads * head_dim |
   | `copy_rows` | a row | rows = r1 - r0, src_row0 + r0, dst_row0 + r0 |
   | `matmul_bf16w` | a row of x and y | m = r1 - r0; `x` a view at r0 * k (none when k = 0), `y` at r0 * n |
   | `attention_gqa_causal` | a query row | rows = r1 - r0, q_row0 + r0, past + r0: a single row is a decode step, a range a chunked prefill |
   | `groups_echo` | a group | groups = r1 (prefixes) |
   | `table_echo`, `args_echo` | none | no splits |

8. **The budget.** The dispatch time is the faster of the two runs, per dispatch, from the backend's `gpu.dispatch` timers (`DeviceOptions.TimeDispatches`, which the suites set), or host time around Submit and Wait when the backend reports none (`time_source` `gpu` or `host`). On a shader backend (`ExecutionBackend.runsShaders`) it must be below 100 ms (`HarnessOptions.BudgetNs`); on CpuDevice it is measured and reported but does not gate, because emulation time says nothing about a GPU.
9. **Telemetry.** One `conformance.case` Check per case, passed when every step passed, with `KernelBounds.attrs` (`kernel`, `bound_kind`, `measured`, `bound`, `ref_error`, `elements`) and: `case`, `backend` (ExecutionBackend name), `detail` (the failures, or what passed), `xclass`, `fp32_denormals`, `strict`, `twice_identical`, `batch_splits`, `batch_matched`, `within_class_only` (elements a mixed class leaves out), `time_ns`, `time_source`, `budget_ns` and `budget_gating`. The suites wrap each case in its own run log scope, and the backend's `gpu.dispatch` records carry the kernel.

**The negative control.** `session.Control case` runs the same steps with every dispatch removed and emits nothing. It must fail for every case: the poison is left, or (in place) the result is never computed. The suites assert it for every case on every backend.

## fp32 denormals and NaN

The mode is the class's (ADR int-0001; KernelAbiAndCatalog, kernel rules), measured by the probe. The references follow it for the kernels that compute on fp32 values (`scale_f32`, `add`, `argmax_rows` and the relative kernels); the kernels that only move bits compare every bit in every mode.

| Mode | Reference |
| --- | --- |
| preserve | CpuRef as it is (CpuDevice; D3D12 on the RTX 4080, driver 616.92) |
| flush | every fp32 (and bf16) denormal input and f32 scalar argument set to zero of the same sign; for `scale_f32` and `add` the results too (`Fp32Denormals.flushBits`); for `argmax_rows` the argmax of the flushed inputs; relative results are covered by KernelBounds' allowance |
| mixed | `scale_f32` and `add`: elements whose input or result is a denormal are left out; `argmax_rows`: rows holding a denormal are left out; a relative case with any denormal input is compared within the class only; run-twice, the region checks and batch invariance still hold |

**NaN.** Both GPUs return `7fffffff` for any NaN operand of fp32 arithmetic (R1 pitfall 10), CpuRef propagates the payload: KernelBounds compares NaN results of `scale_f32` and `add` by class. Bit-moving kernels keep payloads (`embed_gather_bf16`'s `7fc00000`, `bf16_unpack`, `copy_rows`, the echoes), and a relative case fails on any NaN.

## CpuDevice

`CpuDeviceBackend` (provider `CpuDevice.provider`, `CpuDevice.create options deviceOptions`) implements IGpuBackend, IGpuHeap, ICommandList and IKernelSet on host memory.

- **Class.** `ExecutionClass.forCpu cpu-device` at any time. A model (below) appends `+ftz` and/or `+nan-canonical` to `Arch`, so a model never shares a class, and no bit-identity claim, with the default device.
- **Kernel set.** Every catalog kernel has an emulator, sources or not (`Contains` is true for all 14); `MirrorShaderSet` reports only the kernels with a blob, as a shader backend does. `ShaderSetHash` is `Canonical.hash "lrm.cpudevice.kernels/1"` over the emulator version (u32 1), the catalog hash, the denormal model's name and the NaN model. `LoadKernels` refuses a set built for another catalog.
- **Memory.** Each page and the table buffer is a host array allocated on first use and filled with `UninitializedWord` (default `7fbadbad`), because device memory starts undefined; nothing may rely on zeros.
- **Decoding.** Submit runs the commands in order. Each dispatch decodes its 128-byte block (`ArgBlock.OfBytes`) and every token's row from the table image in slot 0 (`TensorRow.OfBytes`, then `TensorRow.validate`), and checks the image against the sealed table.
- **Emulation.** One emulator per kernel runs each physical thread group: linear kernels derive the logical group as `gid.y * lrm_groups_x + gid.x`, so the extra groups of a folded grid find no work; reductions are CpuRef F32's row kernels (`Rows.dot`, `dotBf16`, `rmsNorm`, `ropeHead`, `siluMul`, `add`, `tryArgmax`, `attention`), so the device equals CpuRef F32 bit for bit; bit-moving kernels use integer operations. Read tokens are reached only through read-only spans. `Threads` workers run groups in parallel; groups write disjoint outputs, so the bits do not depend on the count.
- **Rules it enforces**, each with `GpuContractException`: the shared checks (`Dispatch.prepare`: catalog id, word 0, words past the last field, token bounds, null and unassigned tokens, dtypes, `Contiguous`, `ReadOnly`, aliasing, the grid, its limits and folding; `Dispatch.checkCopy`; `Dispatch.checkTransfer`: page slots, multiples of 4, inside the page); the kernel set; the once-only `CreateHeap`, `LoadKernels` and `WriteTable`, and the table's layout; a dispatch before them; single-use command lists from their own backend; timers (names, nesting, open at Submit); unknown submissions; and the kernel-specific rules below. After `Dispose` every call raises `ObjectDisposedException`.
- **Minimum element counts and argument rules** (kernel-abi.md, "Tensor access"), checked when the dispatch is recorded: the argument rules first (`rmsnorm_rows` and `argmax_rows`: cols >= 1; `rope_apply`: heads >= 1 and divides rows, head_dim even and >= 2; `copy_rows`: one dtype, cols <= dst_stride when rows > 1; `attention_gqa_causal`: kv_heads >= 1 and divides q_heads, head_dim >= 1), then every data token's count against the end of its footprint, computed in int64 with overflow raising. The message names the kernel, the field, the tensor, both counts and the rule.
- **Barriers** (`CheckBarriers`, default on). A dispatch that reads bytes an earlier dispatch of the same command list wrote, or writes bytes an earlier one read or wrote, with no `Barrier` or `Copy` between them, raises; the bytes are the footprints (whole words), so disjoint writes into one tensor, interleaved strided rows and shared reads need no barrier. A command-list boundary counts as a barrier.
- **Arithmetic models** (for tests of class-dependent rules; never the default). `Fp32Denormals = Flush` models a flushing GPU class: the kernels that compute on fp32 values see fp32 and bf16 denormal inputs and f32 scalar arguments as zero of the same sign, and their fp32 results are flushed. `CanonicalNaN` turns every NaN result of an fp32 arithmetic kernel into `7fffffff`. `Mixed` is refused.
- **Debug.** With `DeviceOptions.Debug`, `Trace` lists every dispatch run: the kernel, the 32 decoded words and the physical grid.
- **Telemetry.** As every backend (GpuBackendInterface): `mem.arena`, `gpu.heap.slots`, `gpu.bytes_moved`, `gpu.dispatches`, `gpu.wait`, and with `TimeDispatches` `gpu.dispatch` from host time around each emulated dispatch, all with `backend` `cpu-device`.

## Backends and runs

- **cpu tier (CI):** the harness's own tests, then every case not marked `GpuOnly` on CpuDevice (at most 4 threads), then the negative control.
- **gpu tier (local):** for each GPU backend whose assembly (`Lrm.Gpu.D3D12`, `Lrm.Gpu.Vulkan`) exposes a provider, every case on its first accepted device, under that backend's mutex (TestTiers), then the negative control; `Devices.record` names the device, driver and ShaderSetHash. `Discovery.provider` finds a public static property or field of type `IGpuBackendProvider` (an F# module value) or a public provider type with a parameterless constructor, so D1 and V1 add their provider and nothing else. Until then the gpu tier has no conformance tests.
- **Run logs.** Each backend's suite writes a run log of kind `kernels` (RunLog) with the backend's class; `lrm report <runId>` renders the kernels report: the matrix by kernel and backend, the failing cases with measured, bound and detail, the denormal mode and the probe's words, and GPU time per kernel from `gpu.dispatch`. A rendered CpuDevice run is committed under `tests/Lrm.Kernels.Conformance/reports/`.

## Vectors

Computed independently in Python (hashlib, and a Philox4x32-10 written from the Random123 paper that reproduces Random123's known answer `6627e8d5 e169c58d bc57ac4c 9b00dbd8` at counter 0, key 0); `HarnessTests.fs` checks them.

| Id | Input | Result |
| --- | --- | --- |
| P1 | `Poison.seed "cfg" "scale_f32/n1" "run1/y"` | `d0234eb5b214ca17e194f78018a6ed6db94d9979e8baee26c6eb10d42bdbbd08` |
| P2 | its first four poison words | `7fbe4924 ff867e52 ff8961ba 7f8e1b58` |
| P3 | `Poison.avoiding p p` for p = `7f800001` | `7f800002`, a signalling NaN |
| F1 | the footprint of vector A2 (KernelAbiAndCatalog) | x 37,888, w 155,582,464, y 5,621,632 elements |
| S1 | the splits of a case with 10 rows | rows 0 alone, 9 alone, 3 to 5 |

The CpuDevice tests also decode the ABI vectors: A1 and A4 run with exactly the vectors' words and grids, A2 and A3 decode to their records, footprints and folded grids, and `table_echo` returns rows T1 and T2 of HeapLayoutAndTensorTable from the table image.

## Verification

- `Lrm.Kernels.Conformance` (cpu tier): every selftest case and the harness's cases for K1 to K3 pass on CpuDevice, bit-exact to CpuRef F32, and every one fails its negative control; faulty backends around CpuDevice are caught one fault at a time (a dropped dispatch, one wrong bit, a run that differs, writes before and past a tensor, a write inside the capacity past n, a changed input, a batch-variant kernel that is otherwise within its bound, a dispatch over 100 ms on a shader class, and none of those on the unbroken wrapper); a flushing, a NaN-canonical and a combined CpuDevice pass every case against references in their mode; the references in each mode (probe words, a flushed denormal add, argmax ties after flushing, mixed exclusions); vectors P1 to P3; case validation; discovery.
- `Lrm.Gpu.CpuDevice.Tests` (cpu tier): every interface rule; every shared dispatch rule; each kernel's minimum counts (each data field one element short raises, exactly enough and more run) and argument rules, and overflow; the barrier rule (read after write, write after read, write after write, disjoint and interleaved writes, Copy, list boundaries); each emulator against CpuRef F32's tensor ops; grid folding on the physical grid; thread-count invariance; the models; the ABI vectors; no tail calls.

## Change log

- **v1** (E1, wave 2): first version.
