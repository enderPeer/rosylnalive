# KernelAbiAndCatalog contract

Owner: R4 (R4a wrote v1; R4b v1.1). Status: v1.1: the SPIR-V inspector of ShaderBuildAndSourceConvention v1.1 now enforces the push-block layout and the kernel rules on every SPIR-V blob. It is additive for the catalog and every existing kernel, and it tightens the kernel rules (below) for new ones. Consumers: the GPU backends (D1, V1, R5), the CPU device and conformance harness (E1), the kernels (K1 to K3) and the executor (X1). Code: `src/Lrm.Gpu.Core/ArgBlock.fs`, `Catalog.fs`, `Dispatch.fs`, `CatalogGen.fs`, the source of truth `src/Lrm.Gpu.Core/Catalog/catalog.json`, and the generated `src/Lrm.Gpu.Core/Kernels.g.fs` and `src/Lrm.Shaders/include/lrm_args.g.hlsli`. Tests: `tests/Lrm.Gpu.Core.Tests/ArgBlockTests.fs`, `CatalogTests.fs`, `BackendTests.fs`.

A kernel is called with one **128-byte argument block**. The **kernel catalog** fixes, for every kernel, the meaning of each word of that block, its thread-group size and its grid. Hosts, backends, emulators and HLSL all derive from the same catalog, so the ABI cannot drift between them.

## The argument block

`ArgBlock` is 32 little-endian u32 words (128 bytes; the RTX 4080's `maxPushDataSize` is 256, and the design caps arguments at 128).

| Word | Content |
| ---: | --- |
| 0 | `lrm_groups_x`: the number of thread groups in x of the **physical** dispatch. `ICommandList.Dispatch` writes it; callers pass 0 |
| 1..F | the kernel's F catalog fields, in catalog order: field *i* (from 0) is word *i* + 1 |
| F+1..31 | 0 |

- **Field types.** Every field is one word. `token`: a tensor token (u32). `u32`: unsigned. `i32`: two's complement. `f32`: the raw IEEE-754 bits, so negative zero, denormals and NaN payloads pass unchanged (`ArgBlock.f32Bits`).
- **D3D12.** The block is the 32 root constants at `b0` of the canonical root signature (ShaderBuildAndSourceConvention): `SetComputeRoot32BitConstants(0, 32, block, 0)`.
- **Vulkan.** The block is push data at offset 0, 128 bytes (`vkCmdPushDataEXT`, VulkanHeapRecipe). In SPIR-V it is the push-constant block `lrm_args_<kernel>`: member *i* at offset 4*i*, named `lrm_groups_x` or after field *i* − 1, and typed u32, i32 or f32 as the field is. The SPIR-V inspector checks the offsets, names and types against the catalog for every blob (ShaderBuildAndSourceConvention v1.1, rule `push-block`). The recipe's spike read the same bytes through `uint4 d[8]`; LRM keeps the typed blocks, which were run on the RTX 4080 through the spike (ShaderBuildAndSourceConvention, "The argument block type"). VulkanHeapRecipe v1 section 7 still names `uint4 d[8]`; R4b has requested a contract-change of R1 so that it names these blocks, and contracts-v1.1 waits for it.
- **HLSL.** `LRM_ARGS(kernel)` declares `lrm_args` of type `lrm_args_<kernel>` (generated): `uint lrm_groups_x`, then the fields under their names, `uint` for tokens and u32, `int` for i32, `float` for f32. Scalar members sit at 4-byte offsets in both the cbuffer and the push-constant layout.
- **F#.** `Kernels.<Kernel>.encode` builds the block from a typed record `Kernels.<Kernel>Args` (fields in PascalCase) with word 0 left 0; `decode` reads it back; `grid` gives the logical grid; `dispatch commands args` records the dispatch.

## Grids and folding

A catalog grid is three formulas `(x, y, z)` over the kernel's **u32 fields**: integer literals of at least 1, field names, `+`, `*` and `ceildiv(a, b)`, which is the ceiling of a / b. Grammar: `expr = term {'+' term}`, `term = atom {'*' atom}`, `atom = integer | field | 'ceildiv(' expr ',' expr ')' | '(' expr ')'`. Formulas are evaluated in int64 with overflow checks (`GridExpr.eval`, `KernelSpec.gridOf`). The canonical text (`GridExpr.print`) is what the catalog stores and hashes.

The backend computes the logical grid **(X, Y, Z)** from the block itself, so a caller cannot pass a wrong grid. `Grid.physical` then maps it to a dispatch:

1. Every dimension must be at least 1 (an empty grid is a contract violation; callers skip empty work), and Y and Z at most 65,535.
2. X at most 65,535: dispatched as (X, Y, Z).
3. X above 65,535 is allowed only for a **linear** kernel, one whose catalog y and z formulas are the literal `1`. It is **folded** into y' = ceil(X / 65535) rows of x' = ceil(X / y') groups, dispatched as (x', y', 1). X must be at most 65,535 squared.
4. Word 0 is the physical x.

A linear kernel addresses its work through its **logical group** `LRM_GROUP_LINEAR(gid)` = `gid.y * lrm_groups_x + gid.x`. A folded grid has x' * y' - X extra groups at the end, fewer than y'. **Every kernel returns early when its group or thread has no work**, judged from its own arguments (for example `i >= n`). Kernels take only `SV_GroupID`, `SV_GroupThreadID` and `SV_GroupIndex`; `SV_DispatchThreadID` is forbidden because it is wrong on folded grids. A non-linear kernel uses `gid` directly and never folds.

## Dispatch rules

`Dispatch.prepare table kernel block` is the single implementation of these rules. Every backend calls it (or reproduces it exactly), so a violation raises the same `GpuContractException` on every backend, CpuDevice included.

1. The kernel id is in the catalog, and the backend's loaded shader set has the kernel.
2. Word 0 is 0, and every word after the last field is 0.
3. **Tokens.** Every token is below 16,384. A `meta` token is the null token or an assigned row. Any other token is an assigned (non-null) row whose dtype the field lists, with the `Contiguous` flag, and without `ReadOnly` when the kernel writes it.
4. **Aliasing.** When two data tokens overlap in memory and one of them is written, the written field must list the other in its catalog `alias`, and the two must span exactly the same bytes (same slot, offset and extent). Two written tokens never overlap. A declared alias means "in place": the kernel reads each element before it writes it.
5. **Grid.** The catalog grid, folded as above; word 0 of the dispatched block is the physical x.

The kernel-specific argument rules (element counts and the other rules below) are not part of `Dispatch.prepare`: E1's CpuDevice checks them in each kernel's emulator, and GPU kernels assume them.

## Tensor access: flat tokens, sizes from the fields

**Every v1 kernel treats each data token as a flat, contiguous array of its dtype.** Element *i* of a token is at byte `offset + i * size` of the row's slot (a bf16 element is half *i* mod 2 of the u32 word at `offset + 4 * floor(i / 2)`), for 0 <= *i* < `count`, the row's element count (word 12). The shape of the work comes **only from the kernel's u32 fields** (`n`, `rows`, `cols`, `m`, `k`, `q_row0`, `past`, ...). A kernel **never reads the rank, dims or strides of a data token**; only a `meta` token's row is read as a row (`table_echo`).

- **Bracketed shapes are layouts, not dims.** In the kernel tables below, a shape such as `Input [rows, cols]` or `Queries [tokens, q_heads * head_dim]` says how the kernel indexes the flat array (row-major, element `r * cols + c`). It is not a statement about the token's dims.
- **A token's count is a capacity.** A token may hold more elements than one dispatch uses. The table is sealed once at startup (HeapLayoutAndTensorTable), so the executor (X1) creates max-size tokens (for example activations for the longest prefill chunk, the KV cache for `max_pos`) and passes the actual sizes of each dispatch (decode with T = 1, prefill, chunked prefill) in the fields. A row's dims describe the tensor for hosts, tools and debugging; kernels do not depend on them.
- **Contiguous.** Rule 3 above requires the `Contiguous` flag on every data token, so a token's extent is exactly its `count` elements.

**Minimum element counts.** For each dispatch, every data token must hold at least the elements the kernel touches, computed in int64 from the fields; `x >= e` means that token's `count` is at least `e`. The other argument rules complete each kernel's preconditions. E1's CpuDevice checks both before it emulates a kernel and raises `GpuContractException` on a violation; GPU kernels assume them and never check. The grid rule already makes every grid dimension at least 1 (for example `n >= 1` for `scale_f32`, `rows * cols >= 1` for `copy_rows`).

| Kernel | Minimum element counts | Other argument rules |
| --- | --- | --- |
| `scale_f32` | x, y >= n | |
| `table_echo` | y >= 16 | |
| `args_echo` | y >= 32 | |
| `groups_echo` | y >= groups | |
| `bf16_unpack` | x, y >= n | |
| `embed_gather_bf16` | ids >= rows; table >= vocab * cols; y >= rows * cols | |
| `rmsnorm_rows` | x, y >= rows * cols; w >= cols | cols >= 1 |
| `rope_apply` | x, y >= rows * head_dim; cos_table, sin_table >= (pos0 + rows / heads) * (head_dim / 2) | heads >= 1 and divides rows; head_dim even and at least 2 |
| `silu_mul` | gate, up, y >= n | |
| `add` | a, b, y >= n | |
| `copy_rows` | src >= (src_row0 + rows - 1) * src_stride + cols; dst >= (dst_row0 + rows - 1) * dst_stride + cols | src and dst have the same dtype; cols <= dst_stride when rows > 1 (destination rows never overlap) |
| `argmax_rows` | x >= rows * cols; y >= rows | cols >= 1 |
| `matmul_bf16w` | x >= m * k; w >= n * k; y >= m * n | |
| `attention_gqa_causal` | q, y >= (q_row0 + rows) * q_heads * head_dim; k, v >= (past + rows) * kv_heads * head_dim | kv_heads >= 1 and divides q_heads; head_dim >= 1 |

## Kernel rules

Every kernel in the catalog obeys these rules. DXC, the source lint and the SPIR-V inspector (ShaderBuildAndSourceConvention) and E1's conformance harness check them.

- **No float atomics**; in v1, no atomics at all. **No wave or quad operations** (the wave size differs between vendors). **No counters, no typed views, no 16-bit types.** Resources only through `LRM_BUF`, `lrm_buf` and `lrm_tensor`. In SPIR-V these are the inspector's rules `float-atomics`, `atomics`, `wave-ops`, `typed-views`, `16-bit-types` and `resource`.
- **32-bit scalars only (v1.1).** No 8-bit or 64-bit integer or float types (`double`, `int64_t`, `uint64_t` and their vectors): the Vulkan recipe enables neither `shaderInt64` nor `shaderFloat64`. The source lint refuses the 64-bit type names, and the inspector's rule `scalar-width` refuses the widths in SPIR-V.
- **Logical addressing only (v1.1).** No buffer device addresses or physical pointers (for example `vk::RawBufferLoad`): memory is reached only through the heap. In SPIR-V this is the `module` rule (`OpMemoryModel Logical GLSL450`) and the `capabilities` rule.
- **Storage (v1.1).** Module-scope variables are only the heap array, the argument block, the group and thread ids, groupshared arrays and `static` (Private) data; locals live in functions. No cbuffers, images, outputs or other storage. In SPIR-V this is the `storage-class` rule, with `built-ins` for the inputs: only `SV_GroupID`, `SV_GroupThreadID` and `SV_GroupIndex`, and no `WorkgroupSize` built-in (`execution-mode`).
- **One body for both targets.** No conditional compilation and no compiler macros such as `__spirv__` (the source lint), so DXIL and SPIR-V compile the same code.
- **groupshared at most 32 KiB**, enforced by DXC's validator for `cs_6_6`. Because both targets compile one body, the SPIR-V blob declares the same groupshared variables, and the SPIR-V inspector also sums the Workgroup-storage variables of every SPIR-V blob against 32 KiB (rule `workgroup-size`). The RTX 4080 would allow 48 KiB, so the device would not catch it.
- **fp32 denormals are class-dependent.** The CPU reference keeps denormals. The DXIL blobs declare `fp32-denorm-mode=preserve`; the SPIR-V blobs declare no float-controls mode, so a Vulkan device may flush. Each backend measures its mode with the denormal probe and records it (GpuBackendInterface); within one ExecutionClass it is fixed, so run-to-run bit identity is unaffected. For a kernel that does fp32 arithmetic, "bit-exact" against the CPU means equal to the CPU result **computed in the class's measured mode**: for `preserve`, CpuRef as it is; for `flush`, CpuRef with every fp32 denormal input and result replaced by zero of the same sign; for `mixed`, elements whose input or result is a denormal are compared within the class only. Kernels that only move bits (`bf16_unpack`, `embed_gather_bf16`, `copy_rows` and the echo selftests) are bit-exact in every mode. E1's harness and KernelBounds (O2) apply this rule; toleranced bounds must cover a flushed denormal, an absolute error below 2^-126 per flushed value.
- **Every dispatch finishes in under 100 ms** on the reference GPU (the faster of two runs, ConformanceHarness), far from TDR limits. Callers split long work: the executor chunks prefill attention with `q_row0` and `past`.
- **x at most 65,535**, through folding as above.
- **Determinism.** A kernel's output is a function of its inputs and arguments only. Thread groups write disjoint outputs and never race. Reductions follow BatchInvariance.
- `meta` tokens read only the table row, never the tensor's bytes.

## The catalog

`src/Lrm.Gpu.Core/Catalog/catalog.json` (schema `lrm.catalog/1`) is the only source. `Lrm.Shaders.Build gen-catalog` writes:
- `src/Lrm.Gpu.Core/Kernels.g.fs`: `Kernels.all` (the catalog as data; **`KernelId` i is `Kernels.all[i]`**), `Kernels.catalog`, `Kernels.CatalogHashHex`, and per kernel the typed record and module.
- `src/Lrm.Shaders/include/lrm_args.g.hlsli`: per kernel `struct lrm_args_<name>`, `LRM_THREADS_<name>` (the numthreads triple), `LRM_THREADS_X_<name>`, `LRM_THREADS_Y_<name>`, `LRM_THREADS_Z_<name>` and `LRM_LINEAR_<name>`.

Both files are byte-deterministic (UTF-8, LF) and **diff-clean**: the cpu tier (`CatalogTests`), the shader build and CI's shader step fail when they differ from a fresh generation. The parser is strict: unknown properties, unknown types or dtypes, bad formulas and docs that are not one line of printable ASCII are errors. `Catalog.problems` enforces:
- kernel and field names are snake_case (a-z, 0-9 and single `_`), at most 48 characters, unique (also in PascalCase), and do not start with `lrm`; field names are not HLSL reserved words or common intrinsics;
- the source path is `<family directory>/<name>.hlsl`: `hlsl/selftest` (R4), `hlsl/kernels/basic` (K1), `hlsl/kernels/matmul` (K2), `hlsl/kernels/attention` (K3);
- threads: x and y at most 1024, z at most 64, x * y * z at most 1024;
- 1 to 31 fields; every kernel writes at least one token; `meta` tokens list no dtypes and no alias; other tokens list their dtypes; only written tokens alias, and only read tokens of the same kernel;
- grid formulas use only u32 fields, with literals of at least 1.

**Catalog hash.** `Catalog.hash` is `Canonical.hash "lrm.catalog/1"` over the LrmCore canonical encoding of: the version (u32), then the kernels as a sequence, each with its name, family name and source (strings), its threads x, y and z (u32), the canonical grid texts x, y and z (strings), and its fields as a sequence, each with its name (string), its kind code (u8: token 1, u32 2, i32 3, f32 4) and, for a token, its access code (u8: read 1, write 2, readwrite 3, meta 4), its dtype codes (a sequence of u8) and its aliases (a sequence of strings). **Docs are not hashed**, so editing a doc changes no ShaderSetHash.

**Missing sources.** The catalog lists every Phase 1 kernel from the start. The shader build compiles the kernels whose source exists and lists the rest as `missing` in the manifest. A backend's kernel set reports `Contains = false` for them, and dispatching one raises. K1 to K3 add their sources at the catalog path and change nothing else.

**Changes.** After contracts-v1 the catalog changes only through a `contract-change` PR with a CHANGELOG entry (OwnershipAndContractChange). An additive change (a new kernel, appended at the end so existing ids keep their numbers) gets same-day review. Changing an existing entry's fields, threads or grid is breaking and merges together with its consumers' fixes. Either way, regenerate; the catalog hash and both ShaderSetHash values change.

## The catalog at contracts-v1

The tables are rendered from `catalog.json`; `CatalogTests` checks the names, the order, the threads and the grids.

| Id | Kernel | Family | Threads | Grid (x, y, z) | Linear |
| ---: | --- | --- | --- | --- | --- |
| 0 | `scale_f32` | S (R4) | 256x1x1 | `ceildiv(n, 256)`, `1`, `1` | yes |
| 1 | `table_echo` | S (R4) | 16x1x1 | `1`, `1`, `1` | yes |
| 2 | `args_echo` | S (R4) | 32x1x1 | `1`, `1`, `1` | yes |
| 3 | `groups_echo` | S (R4) | 1x1x1 | `groups`, `1`, `1` | yes |
| 4 | `bf16_unpack` | S (R4) | 256x1x1 | `ceildiv(n, 256)`, `1`, `1` | yes |
| 5 | `embed_gather_bf16` | K1 | 256x1x1 | `ceildiv(cols, 256)`, `rows`, `1` | no |
| 6 | `rmsnorm_rows` | K1 | 256x1x1 | `rows`, `1`, `1` | yes |
| 7 | `rope_apply` | K1 | 64x1x1 | `rows`, `1`, `1` | yes |
| 8 | `silu_mul` | K1 | 256x1x1 | `ceildiv(n, 256)`, `1`, `1` | yes |
| 9 | `add` | K1 | 256x1x1 | `ceildiv(n, 256)`, `1`, `1` | yes |
| 10 | `copy_rows` | K1 | 256x1x1 | `ceildiv(rows * cols, 256)`, `1`, `1` | yes |
| 11 | `argmax_rows` | K1 | 256x1x1 | `rows`, `1`, `1` | yes |
| 12 | `matmul_bf16w` | K2 | 16x16x1 | `ceildiv(n, 64)`, `ceildiv(m, 64)`, `1` | no |
| 13 | `attention_gqa_causal` | K3 | 64x1x1 | `rows`, `q_heads`, `1` | no |

### 0. `scale_f32`

y[i] = a * x[i] for i < n, one thread per element. Bit-exact.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `x` | token, read, f32 | Input, at least n elements. |
| 2 | `y` | token, write, f32; may alias `x` | Output, at least n elements. |
| 3 | `n` | u32 | Element count. |
| 4 | `a` | f32 | The scale factor. |

### 1. `table_echo`

Writes the 16 words of the tensor row of src, as lrm_tensor decodes them, to y[0..15].

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `src` | token, meta, any dtype | The token whose row is echoed; may be the null token. |
| 2 | `y` | token, write, u32 | Output, at least 16 elements. |

### 2. `args_echo`

Writes all 32 argument words, lrm_groups_x included, to y[0..31].

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `y` | token, write, u32 | Output, at least 32 elements. |
| 2..31 | `v2` .. `v31` | u32 | Echoed to y[2] .. y[31]. |

### 3. `groups_echo`

Logical group g < groups writes g to y[g]; tests grid folding above 65,535 groups.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `y` | token, write, u32 | Output, at least groups elements. |
| 2 | `groups` | u32 | Logical group count. |

### 4. `bf16_unpack`

y[i] = widen(x[i]) for i < n: bf16 bits shifted into the high half of an f32. Bit-exact.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `x` | token, read, bf16 | Input, at least n elements. |
| 2 | `y` | token, write, f32 | Output, at least n elements. |
| 3 | `n` | u32 | Element count. |

### 5. `embed_gather_bf16`

y[r, c] = widen(table[ids[r], c]) for r < rows, c < cols; an id >= vocab gives a row of quiet NaN (0x7FC00000). Bit-exact.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `ids` | token, read, u32 | Token ids [rows]. |
| 2 | `table` | token, read, bf16 | Embedding table [vocab, cols]. |
| 3 | `y` | token, write, f32 | Output [rows, cols]. |
| 4 | `rows` | u32 | Number of ids. |
| 5 | `cols` | u32 | Hidden size. |
| 6 | `vocab` | u32 | Rows of the table; larger ids are out of range. |

### 6. `rmsnorm_rows`

y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * widen(w[:]), one group per row; the reduction order depends only on cols.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `x` | token, read, f32 | Input [rows, cols]. |
| 2 | `w` | token, read, bf16 | Weight [cols], used as w, not 1 + w. |
| 3 | `y` | token, write, f32; may alias `x` | Output [rows, cols]. |
| 4 | `rows` | u32 | Rows (tokens, or tokens * heads for q_norm and k_norm). |
| 5 | `cols` | u32 | Row length. |
| 6 | `eps` | f32 | rms_norm_eps. |

### 7. `rope_apply`

NeoX rotate-half RoPE from tables, never evaluating sin or cos: row r has position pos0 + r / heads; for i < head_dim / 2, y[i] = x[i] * c - x[i + h] * s and y[i + h] = x[i + h] * c + x[i] * s.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `x` | token, read, f32 | Input [rows, head_dim], rows = tokens * heads. |
| 2 | `cos_table` | token, read, f32 | cos table [max_pos, head_dim / 2]. |
| 3 | `sin_table` | token, read, f32 | sin table [max_pos, head_dim / 2]. |
| 4 | `y` | token, write, f32; may alias `x` | Output [rows, head_dim]. |
| 5 | `rows` | u32 | tokens * heads. |
| 6 | `heads` | u32 | Heads per token. |
| 7 | `head_dim` | u32 | Head size; even. |
| 8 | `pos0` | u32 | Position of the first token. |

### 8. `silu_mul`

y[i] = silu(gate[i]) * up[i] for i < n, silu(g) = g / (1 + exp(-g)).

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `gate` | token, read, f32 | Gate projection, at least n elements. |
| 2 | `up` | token, read, f32 | Up projection, at least n elements. |
| 3 | `y` | token, write, f32; may alias `gate`, `up` | Output, at least n elements. |
| 4 | `n` | u32 | Element count. |

### 9. `add`

y[i] = a[i] + b[i] for i < n. Bit-exact.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `a` | token, read, f32 | First addend, at least n elements. |
| 2 | `b` | token, read, f32 | Second addend, at least n elements. |
| 3 | `y` | token, write, f32; may alias `a`, `b` | Sum, at least n elements. |
| 4 | `n` | u32 | Element count. |

### 10. `copy_rows`

dst[(dst_row0 + r) * dst_stride + c] = src[(src_row0 + r) * src_stride + c] for r < rows, c < cols, 4-byte elements of one dtype. Bit-exact.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `src` | token, read, f32, u32, i32 | Source rows. |
| 2 | `dst` | token, write, f32, u32, i32 | Destination rows; same dtype as src. |
| 3 | `rows` | u32 | Rows to copy. |
| 4 | `cols` | u32 | Elements per row. |
| 5 | `src_row0` | u32 | First source row. |
| 6 | `dst_row0` | u32 | First destination row. |
| 7 | `src_stride` | u32 | Elements between source rows. |
| 8 | `dst_stride` | u32 | Elements between destination rows. |

### 11. `argmax_rows`

y[r] = the lowest c with x[r, c] maximal, for r < rows; a row holding any NaN gives 0xFFFFFFFF. The index is bit-exact.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `x` | token, read, f32 | Input [rows, cols], such as logits. |
| 2 | `y` | token, write, u32 | Indices [rows]. |
| 3 | `rows` | u32 | Rows. |
| 4 | `cols` | u32 | Row length. |

### 12. `matmul_bf16w`

y[i, j] = sum over p < k of x[i, p] * widen(w[j, p]), 64x64 output tiles, p accumulated sequentially in increasing order with mad; one kernel for decode, prefill and lm_head.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `x` | token, read, f32 | Activations [m, k]. |
| 2 | `w` | token, read, bf16 | Weight [n, k] (PyTorch Linear layout). |
| 3 | `y` | token, write, f32 | Output [m, n]. |
| 4 | `m` | u32 | Rows of x and y. |
| 5 | `n` | u32 | Rows of w, columns of y. |
| 6 | `k` | u32 | The reduction length. |

### 13. `attention_gqa_causal`

Causal GQA attention, one group per (query row, q head), fixed 64-key blocks with online softmax, the causal mask as a loop bound: query row i reads q row q_row0 + i, has position past + i and attends keys 0..past + i.

| Word | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 1 | `q` | token, read, f32 | Queries [tokens, q_heads * head_dim], after norm and RoPE. |
| 2 | `k` | token, read, f32 | Key cache [max_pos, kv_heads * head_dim], after norm and RoPE. |
| 3 | `v` | token, read, f32 | Value cache [max_pos, kv_heads * head_dim]. |
| 4 | `y` | token, write, f32 | Output [tokens, q_heads * head_dim]; row q_row0 + i. |
| 5 | `rows` | u32 | Query rows in this dispatch. |
| 6 | `q_row0` | u32 | First q and y row of this dispatch. |
| 7 | `past` | u32 | Position of the first query row. |
| 8 | `q_heads` | u32 | Query heads. |
| 9 | `kv_heads` | u32 | Key and value heads; divides q_heads. |
| 10 | `head_dim` | u32 | Head size. |
| 11 | `scale` | f32 | Score scale, 1 / sqrt(head_dim). |

### Notes for the kernel owners

- **Selftests (R4).** Backends and E1 run them first. `scale_f32` and `bf16_unpack` are bit-exact against the CPU (`scale_f32` in the class's denormal mode, kernel rules); `scale_f32` is also the denormal probe (GpuBackendInterface). `table_echo` checks the table decode, the null row included. `args_echo` checks that all 32 words arrive, word 0 included. `groups_echo` with `groups` above 65,535 checks folding, and that the extra groups write nothing.
- **K1.** `embed_gather_bf16`, `add`, `copy_rows` and the `argmax_rows` index are bit-exact (KernelBounds; `add` and `argmax_rows` in the class's denormal mode). Every K1 to K3 kernel reads its tokens as flat arrays and takes its sizes from its fields (tensor access, above). `rope_apply` is table driven and never evaluates sin or cos; the tables are CpuRef's `ropeTable` (pinned `inv_freq`). `argmax_rows` returns the lowest index of the maximum, and `0xFFFFFFFF` for a row that holds a NaN, which the host turns into an error as CpuRef raises.
- **K2.** `matmul_bf16w` uses the PyTorch Linear layout: `w` is `[n, k]`. Each output accumulates p = 0..k-1 sequentially in increasing order with `mad`, whatever m is (BatchInvariance), so M = 1 rows equal the same rows at M = 37 and M = 512.
- **K3.** `attention_gqa_causal`: query row i of the dispatch reads q row `q_row0 + i`, writes y row `q_row0 + i`, has position `past + i` and attends key positions 0 to `past + i`. Query head h uses kv head `h / (q_heads / kv_heads)`. Key blocks of 64 start at position 0, whatever `past` and `q_row0` are, so decode, prefill and chunked prefill agree.

## Vectors

Independently computed with Python `struct`; the cpu tier checks them (`ArgBlockTests.fs`). Each is the block as dispatched, with word 0 written. Words are little-endian hex; trailing zero words are elided.

| Id | Call | Words from 0 | Grid |
| --- | --- | --- | --- |
| A1 | `scale_f32 {x = t5; y = t6; n = 1000; a = 0.5}` | `04000000 05000000 06000000 e8030000 0000003f`, then 27 zero words | (4, 1, 1) |
| A2 | `matmul_bf16w {x = t1; w = t2; y = t3; m = 37; n = 151936; k = 1024}` | `46090000 01000000 02000000 03000000 25000000 80510200 00040000`, then 25 zero words | (2374, 1, 1) |
| A3 | `add {a = t1; b = t2; y = t3; n = 8192 * 3072}` | `00c00000 01000000 02000000 03000000 00008001`, then 27 zero words | 98,304 groups folded to (49152, 2, 1) |
| A4 | `attention_gqa_causal {q, k, v, y = t1..t4; rows = 7; q_row0 = 0; past = 0; q_heads = 16; kv_heads = 8; head_dim = 128; scale = f32(1 / sqrt 128)}` | `07000000 01000000 02000000 03000000 04000000 07000000 00000000 00000000 10000000 08000000 80000000 f304b53d`, then 20 zero words | (7, 16, 1) |

Folding edge cases: X = 65,535 gives (65535, 1, 1); X = 65,536 gives (32768, 2, 1); X = 70,000 gives (35000, 2, 1).

**Catalog hash at contracts-v1:** `2b17293091faa39ed7267c35c6226ebf1f57e263c63629a257b268acfce1cbfe`, recomputed independently in Python from `catalog.json`.

## Verification

The cpu tier checks: vectors A1 to A4; f32 bit preservation; the byte round trip; a property that folded grids cover the logical grid with fewer than y' extra groups; the grid edge cases; the frozen kernel list, threads and grids; the catalog hash, and that docs do not change it; that the generated files are diff-clean and parse back to `Kernels.catalog`; the parser's strictness; every catalog rule; field words; the set of linear kernels; typed encode and decode; and every dispatch rule, through RecordingBackend. E1 decodes real blocks in CpuDevice and checks the minimum element counts and argument rules. The SPIR-V inspector checks the push offsets, names and types and the groupshared size of every SPIR-V blob (ShaderBuildAndSourceConvention v1.1).

## Change log

- **v1** (R4a, PR #10): first version.
- **v1.1** (R4b, PR #12): the SPIR-V push block is named and typed as the catalog, and the SPIR-V inspector enforces it, the 32 KiB groupshared limit and the kernel rules on every SPIR-V blob. The kernel rules gain 32-bit scalars only, logical addressing only and the storage limits; no existing kernel breaks them. The block, the catalog and its hash are unchanged.
