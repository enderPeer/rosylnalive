# HeapLayoutAndTensorTable contract

Owner: R4 (R4a). Status: v1, frozen at the contracts-v1 tag. Consumers: the GPU backends (D1, V1, R5), the CPU device and conformance harness (E1), the kernels (K1 to K3) and the executor (X1). Code: `src/Lrm.Gpu.Core/HeapLayout.fs`, `src/Lrm.Gpu.Core/TensorTable.fs`. Tests: `tests/Lrm.Gpu.Core.Tests/HeapLayoutTests.fs`, `TensorTableTests.fs`.

Every GPU backend has **one resource heap of 64 descriptor slots, written once at startup**. Slot 0 holds the tensor table; slots 1 to 63 hold pages, the buffers of the memory arenas. Kernels never see a descriptor index directly: they receive 32-bit tensor tokens, look the token's row up in the table, and get the slot and byte offset from the row. One mechanism serves every kernel on both APIs.

## The heap

| Number | Value | Where |
| --- | --- | --- |
| Slots | 64, all written once when the heap is created | `Heap.SlotCount` |
| Table slot | 0 | `Heap.TableSlot` |
| Table | 16,384 rows of 64 bytes = 1 MiB | `Heap.TableRows`, `Heap.RowBytes`, `Heap.TableBytes` |
| Page | at most 1 GiB, a multiple of 64 KiB | `Heap.MaxPageBytes`, `Heap.PageGranularity` |
| Allocation alignment | 256 bytes (HeapPlanner) | `Heap.AllocationAlignment` |

- **Slot mapping, Vulkan.** The resource heap holds the 64 descriptors at stride 16: slot *i* is at byte `16·i` (`Heap.vulkanDescriptorOffset`). The driver's reserved range follows them, 64-byte aligned, at byte 1024 (`Heap.vulkanReservedRangeOffset`); the heap is `1024 + reservedRangeBytes` bytes (`Heap.vulkanHeapBytes`). The reserved range is never written. A device whose buffer descriptors do not fit in 16 bytes is rejected. R1's ADR (VulkanHeapRecipe) records the remaining mapping values; V1 implements them.
- **Slot mapping, D3D12.** Slot *i* is descriptor index *i* (`Heap.d3d12DescriptorIndex`) of one shader-visible CBV/SRV/UAV heap of exactly 64 descriptors. Every page is a raw UAV: `DXGI_FORMAT_R32_TYPELESS`, `D3D12_BUFFER_UAV_FLAG_RAW`, `FirstElement` 0 and `NumElements` = bytes / 4. The table in slot 0 is a raw UAV of 1 MiB.
- **Unused slots** (after the last page) hold null descriptors, written once like the others. Kernels never touch them: no table row names them.
- **Why 1 GiB.** A byte address inside a page is below 2³⁰, so `offset + 4·index` stays below 2³¹ for every in-page element and kernels compute addresses in u32 without overflow. Large arenas use several pages.

## Arenas

Pages belong to arenas. The kinds, in contract order, with frozen codes:

| Code | Kind | Name | Use |
| ---: | --- | --- | --- |
| 1 | `Weights` | `weights` | bf16 model weights, read only |
| 2 | `Rope` | `rope` | RoPE cos and sin tables, read only |
| 3 | `KvCache` | `kv_cache` | the KV cache |
| 4 | `Activations` | `activations` | activations of the forward pass |
| 5 | `Scratch` | `scratch` | temporaries inside a stage |
| 6 | `Logits` | `logits` | logits and argmax results |
| 7 | `Taps` | `taps` | tap snapshots, copied at record time (GpuExecutor) |
| 8 | `Master` | `master` | reserved (Phase 2): fp32 master weights |
| 9 | `Grad` | `grad` | reserved (Phase 2): gradients |
| 10 | `AdamM` | `adam_m` | reserved (Phase 2): AdamW first moments |
| 11 | `AdamV` | `adam_v` | reserved (Phase 2): AdamW second moments |
| 12 | `RoslynIn` | `roslyn_in` | reserved (Phase 3): queries the GPU writes for Roslyn |
| 13 | `RoslynOut` | `roslyn_out` | reserved (Phase 3): Roslyn's answers |

- **Reserved kinds** exist in the enum with their codes, so later phases add no renumbering. A v1 layout never holds them: `HeapLayout.ofPages` and `HeapPlanner.Allocate` reject them ("reserved until contracts-v2"). contracts-v2 lifts this.
- **Names** are the `arena` attribute of the `mem.arena` telemetry gauge (TelemetryFacade). The table is reported as arena `table`.

## HeapLayout

A `HeapLayout` is an immutable list of pages. Slot 0 is the table; the pages fill slots 1 to N in order; slots N+1 to 63 are unused.

`HeapLayout.ofPages [(kind, bytes); ...]` accepts a page list when:
1. there are at most 63 pages;
2. each size is a multiple of 64 KiB, from 64 KiB to 1 GiB;
3. no kind is reserved;
4. kinds appear in contract order (each arena's pages are contiguous).

Each page gets `Slot` (1..63), `Arena`, `Index` (its position within the arena, from 0) and `Bytes`. `SlotOf(arena, index)`, `Page(slot)`, `ArenaBytes(arena)` and `TotalBytes` (table included) query it. Two layouts are equal when their page lists are equal.

### HeapPlanner (pass one of a static plan)

`HeapPlanner(pageBytes)` (default 1 GiB) places allocations without touching a device:
- `Allocate(arena, bytes)` puts the allocation at the end of the arena's last page, aligned to 256 bytes, when it fits there; otherwise at offset 0 of a new page. An allocation never straddles two pages, so a tensor is at most one page. It returns a `Placement` `{Arena; Page; Offset; Bytes}`.
- `Build()` turns the fill levels into a layout: every arena's pages, in contract order, each rounded up to 64 KiB. After `Build`, `Allocate` raises.
- The same allocation sequence always gives the same placements and the same layout.

## The tensor table

### TensorToken

A token is a u32 row index. **Token 0 is the null token**; real tokens are 1 to 16,383. Kernels receive tokens in their argument block (KernelAbiAndCatalog).

### TensorRow (64 bytes, 16 little-endian u32 words)

| Word | Field | Rule |
| ---: | --- | --- |
| 0 | slot | 1..63, a page of the layout |
| 1 | byte offset | inside the page; a multiple of 4 and of the element size |
| 2 | dtype | the LrmCore DType code (F32 1, BF16 2, F16 3, I32 4, U32 5, U8 6, F64 7, I64 8) |
| 3 | rank | 1..4 |
| 4..7 | dims[0..3] | outermost first; each used dim ≥ 1; unused dims are 1 |
| 8..11 | strides[0..3] | in elements; unused strides are 0 |
| 12 | element count | the product of the dims, at most 2³² − 1 |
| 13 | flags | see below; bits 3..31 are 0 |
| 14 | debug id | `TensorRow.debugIdOf name`: the first 4 bytes of SHA-256(UTF-8 name), little-endian; 0 for no name. Never read for computation |
| 15 | reserved | 0 |

| Flag | Bit | Meaning |
| --- | ---: | --- |
| `Contiguous` | 1 | The strides are the packed row-major strides of the dims. Set exactly when that holds. Every v1 kernel requires it on its data tokens |
| `ReadOnly` | 2 | Kernels never write the tensor (weights, RoPE tables). A write token on it is a contract violation |
| `View` | 4 | The row aliases bytes that another row owns |

- **Extent.** A row spans `1 + Σ (dims[i] − 1)·strides[i]` elements from its offset; in bytes, that times the element size, **rounded up to 4**, because kernels load whole u32 words (an odd-length bf16 tensor owns its last word). The extent lies inside the page.
- **The null row** (token 0) and every unused row are 64 zero bytes. dtype code 0 is not a DType, so a zero row never decodes as valid.
- `TensorRow.validate layout row` checks every rule above; `TensorRow.create` builds a contiguous row, `TensorRow.strided` a row with explicit strides.

### TensorTable (pass two) and SealedTable

- `TensorTable(layout)` adds rows under unique, non-empty names and returns tokens 1, 2, 3, ... in order. Every row must validate against the layout. `Add(name, placement, dtype, dims, flags)` takes a planner placement and checks that it holds the tensor; `AddView(name, source, dims, elementOffset)` adds a contiguous view that lies inside the source's extent, with `View` and the source's `ReadOnly`.
- `Seal()` freezes the table, exactly once. After it, `Add` and `Seal` raise.
- `SealedTable.Image` is the 1 MiB image: row *t* at byte `64·t`, little-endian, unused rows zero. `SealedTable.Id` is its SHA-256. `Row(token)` returns the zero row for the null token and unused tokens, and raises for a token ≥ 16,384.
- **The table is written once.** `IGpuHeap.WriteTable` writes the sealed image into slot 0 exactly once, before the first dispatch (GpuBackendInterface). Nothing else writes slot 0.

## Kernel access (lrm.hlsli)

`lrm_tensor(token)` loads the four `uint4` words of row `token` from slot 0 at byte `64·token` and returns them as `lrm_tensor_t`, in row order. `lrm_load_f32`, `lrm_store_f32`, `lrm_load_u32`, `lrm_load_bf16` and the other accessors take an element index from the tensor's start. A kernel reads only a data token's slot and offset: it indexes the token as a flat array and takes every size from its arguments, never from the row's rank, dims or strides. The element count is a capacity, which E1 checks against the arguments (KernelAbiAndCatalog, tensor access). Because the slot comes from the table and the token from the argument block, **every descriptor index is dynamically uniform**; there is no `NonUniformResourceIndex` (ShaderBuildAndSourceConvention).

## Vectors

Independently computed with Python `struct` and `hashlib`; the cpu tier checks them (`TensorTableTests.fs`).

**T1, the Qwen3-0.6B embedding:** bf16 `[151936, 1024]`, slot 1 at byte 256, `ReadOnly`, name `model.embed_tokens.weight`.

| Words | Hex (little-endian words) |
| --- | --- |
| 0..3 slot, offset, dtype, rank | `01000000 00010000 02000000 02000000` |
| 4..7 dims | `80510200 00040000 01000000 01000000` |
| 8..11 strides | `00040000 01000000 00000000 00000000` |
| 12..15 count, flags, debug id, reserved | `00004609 03000000 fcd443ac 00000000` |

Count 155,582,464; flags `Contiguous | ReadOnly` = 3; debug id `0xac43d4fc`.

**T2, a rank-3 view:** f32 `[7, 16, 128]`, slot 4 at byte 4096, `View`, name `L0.attn.q_rope`.

| Words | Hex |
| --- | --- |
| 0..3 | `04000000 00100000 01000000 03000000` |
| 4..7 | `07000000 10000000 80000000 01000000` |
| 8..11 | `00080000 80000000 01000000 00000000` |
| 12..15 | `00380000 05000000 d4fd8611 00000000` |

Count 14,336; flags `Contiguous | View` = 5; debug id `0x1186fdd4`.

**The null row:** 64 zero bytes. **The table image:** row 0 zero, row 1 at bytes 64..127, row 2 at 128..191, the rest zero.

## Verification

The cpu tier checks: frozen arena codes and names; the fixed numbers; both slot mappings; every `ofPages` rejection; the planner's placements, alignment, page rounding and contract order; the reserved arenas; vectors T1 and T2; every row rule; the bf16 extent rounding; table tokens, sealing, the image layout and the 16,383-row capacity; placements and views. E1's CpuDevice enforces the same rules at run time, and D1 and V1 check heap writes equal the slot count.
