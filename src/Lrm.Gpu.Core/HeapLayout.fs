namespace Lrm.Gpu.Core

open System
open System.Collections.Generic

/// The memory arenas of the resource heap (HeapLayoutAndTensorTable contract), in contract order. The codes are frozen.
/// Master, Grad, AdamM, AdamV, RoslynIn and RoslynOut are reserved for Phase 2 and 3: a v1 layout never holds them.
[<RequireQualifiedAccess>]
type ArenaKind =
    /// Model weights (bf16), read only.
    | Weights
    /// RoPE cos and sin tables, read only.
    | Rope
    /// The KV cache.
    | KvCache
    /// Activations of the forward pass.
    | Activations
    /// Temporaries inside one stage.
    | Scratch
    /// Logits and argmax results.
    | Logits
    /// Tap snapshots, copied at record time.
    | Taps
    /// Reserved (Phase 2): fp32 master weights.
    | Master
    /// Reserved (Phase 2): gradients.
    | Grad
    /// Reserved (Phase 2): AdamW first moments.
    | AdamM
    /// Reserved (Phase 2): AdamW second moments.
    | AdamV
    /// Reserved (Phase 3): Roslyn queries written by the GPU.
    | RoslynIn
    /// Reserved (Phase 3): Roslyn answers written by the CPU.
    | RoslynOut

[<RequireQualifiedAccess>]
module ArenaKind =
    /// Every kind, in contract (code) order.
    let all =
        [
            ArenaKind.Weights
            ArenaKind.Rope
            ArenaKind.KvCache
            ArenaKind.Activations
            ArenaKind.Scratch
            ArenaKind.Logits
            ArenaKind.Taps
            ArenaKind.Master
            ArenaKind.Grad
            ArenaKind.AdamM
            ArenaKind.AdamV
            ArenaKind.RoslynIn
            ArenaKind.RoslynOut
        ]

    /// The frozen code, 1 to 13 in contract order. 0 is never an arena.
    let code (kind: ArenaKind) : int =
        match kind with
        | ArenaKind.Weights -> 1
        | ArenaKind.Rope -> 2
        | ArenaKind.KvCache -> 3
        | ArenaKind.Activations -> 4
        | ArenaKind.Scratch -> 5
        | ArenaKind.Logits -> 6
        | ArenaKind.Taps -> 7
        | ArenaKind.Master -> 8
        | ArenaKind.Grad -> 9
        | ArenaKind.AdamM -> 10
        | ArenaKind.AdamV -> 11
        | ArenaKind.RoslynIn -> 12
        | ArenaKind.RoslynOut -> 13

    /// The kind with this code, or None.
    let tryOfCode (c: int) =
        all |> List.tryFind (fun k -> code k = c)

    /// The name used in telemetry (the `arena` attribute of mem.arena), reports and logs.
    let name (kind: ArenaKind) =
        match kind with
        | ArenaKind.Weights -> "weights"
        | ArenaKind.Rope -> "rope"
        | ArenaKind.KvCache -> "kv_cache"
        | ArenaKind.Activations -> "activations"
        | ArenaKind.Scratch -> "scratch"
        | ArenaKind.Logits -> "logits"
        | ArenaKind.Taps -> "taps"
        | ArenaKind.Master -> "master"
        | ArenaKind.Grad -> "grad"
        | ArenaKind.AdamM -> "adam_m"
        | ArenaKind.AdamV -> "adam_v"
        | ArenaKind.RoslynIn -> "roslyn_in"
        | ArenaKind.RoslynOut -> "roslyn_out"

    /// Parses a name (ordinal match).
    let tryParse (text: string) =
        all
        |> List.tryFind (fun k -> String.Equals(name k, text, StringComparison.Ordinal))

    /// True for the kinds reserved for later phases (Master, Grad, AdamM, AdamV, RoslynIn, RoslynOut).
    let isReserved (kind: ArenaKind) = code kind >= 8

/// One page of the heap: a buffer behind one descriptor slot, holding part of one arena.
type HeapPage =
    {
        /// The descriptor slot, 1 to 63 (slot 0 is the tensor table).
        Slot: int
        Arena: ArenaKind
        /// The page's index within its arena, from 0.
        Index: int
        /// The page size: a multiple of 64 KiB, at most 1 GiB.
        Bytes: int64
    }

/// The device limits the Vulkan heap geometry depends on (VulkanHeapRecipe, section 3): from
/// VkPhysicalDeviceDescriptorHeapPropertiesEXT, queried by the Vulkan backend, never hard-coded.
type VulkanHeapLimits =
    {
        /// bufferDescriptorSize (16 on the RTX 4080 under driver 616.92).
        BufferDescriptorSize: int64
        /// bufferDescriptorAlignment (8).
        BufferDescriptorAlignment: int64
        /// imageDescriptorAlignment (32).
        ImageDescriptorAlignment: int64
        /// minResourceHeapReservedRange (96,768).
        MinResourceHeapReservedRange: int64
        /// maxResourceHeapSize (33,554,432).
        MaxResourceHeapSize: int64
    }

/// Where the 64 slots and the driver's reserved range sit in the Vulkan resource heap (VulkanHeapRecipe, section 3),
/// in bytes from heapRange.address.
type VulkanHeapGeometry =
    {
        /// Bytes from one slot to the next: bufferDescriptorSize, also the mapping's heapArrayStride (SpirvBinding).
        SlotStride: int64
        /// The 64 slots: 64 * SlotStride.
        SlotAreaBytes: int64
        /// reservedRangeOffset: the slot area aligned up to max(64, bufferDescriptorAlignment, imageDescriptorAlignment).
        ReservedRangeOffset: int64
        /// reservedRangeSize: minResourceHeapReservedRange.
        ReservedRangeBytes: int64
        /// heapRange.size: ReservedRangeOffset + ReservedRangeBytes, at most maxResourceHeapSize.
        HeapBytes: int64
    }

/// The fixed numbers of the heap (HeapLayoutAndTensorTable contract).
[<RequireQualifiedAccess>]
module Heap =
    /// Descriptor slots in the heap, all written once at startup.
    [<Literal>]
    let SlotCount = 64

    /// The slot that holds the tensor table.
    [<Literal>]
    let TableSlot = 0

    /// Rows in the tensor table. Token 0 is the null token.
    [<Literal>]
    let TableRows = 16384

    /// Bytes per tensor-table row (16 u32).
    [<Literal>]
    let RowBytes = 64

    /// Bytes of the tensor table (16,384 rows of 64 bytes, 1 MiB).
    [<Literal>]
    let TableBytes = 1048576L

    /// The largest page: 1 GiB. Byte addresses inside a page therefore fit in 30 bits, so kernels compute them in u32
    /// without overflow.
    [<Literal>]
    let MaxPageBytes = 1073741824L

    /// Page sizes are multiples of 64 KiB.
    [<Literal>]
    let PageGranularity = 65536L

    /// HeapPlanner aligns every allocation to 256 bytes.
    [<Literal>]
    let AllocationAlignment = 256L

    /// Vulkan: bytes per descriptor in the resource heap on the RTX 4080 (bufferDescriptorSize 16): slot i sits at
    /// byte 16 * i. The general rule is vulkanGeometry (slot i at i * bufferDescriptorSize).
    [<Literal>]
    let VulkanDescriptorStride = 16

    /// Vulkan: bytes of the 64 descriptors at the start of the resource heap.
    [<Literal>]
    let VulkanDescriptorBytes = 1024L

    /// Vulkan: the alignment of the reserved range, which follows the descriptors.
    [<Literal>]
    let VulkanReservedAlignment = 64L

    /// Vulkan: the byte offset of slot `slot` in the resource heap.
    let vulkanDescriptorOffset (slot: int) =
        if slot < 0 || slot >= SlotCount then
            Fail.contract "slot %d is outside 0..%d" slot (SlotCount - 1)

        int64 slot * int64 VulkanDescriptorStride

    /// Vulkan: where the driver's reserved range starts for 16-byte descriptors whose alignments are at most 64 (the
    /// RTX 4080: 1,024). The general rule is vulkanGeometry.
    let vulkanReservedRangeOffset =
        (VulkanDescriptorBytes + VulkanReservedAlignment - 1L) / VulkanReservedAlignment
        * VulkanReservedAlignment

    /// Vulkan: the size of the resource heap for a device that needs `reservedRangeBytes` of reserved range.
    let vulkanHeapBytes (reservedRangeBytes: int64) =
        if reservedRangeBytes < 0L then
            Fail.contract "negative reserved range %d" reservedRangeBytes

        vulkanReservedRangeOffset + reservedRangeBytes

    /// Rounds `bytes` up to a multiple of `alignment` (a power of two).
    let alignUp (alignment: int64) (bytes: int64) =
        (bytes + alignment - 1L) &&& ~~~(alignment - 1L)

    /// Vulkan: the limits of the RTX 4080 under driver 616.92, as VulkanHeapRecipe records them (its KAT-2). The
    /// stride-16 helpers above are this device's geometry.
    let rtx4080VulkanLimits =
        {
            BufferDescriptorSize = 16L
            BufferDescriptorAlignment = 8L
            ImageDescriptorAlignment = 32L
            MinResourceHeapReservedRange = 96768L
            MaxResourceHeapSize = 33554432L
        }

    let private isPowerOfTwo (v: int64) = v > 0L && v &&& (v - 1L) = 0L

    /// Vulkan: the heap geometry of VulkanHeapRecipe section 3 for a device's limits: slot i at byte
    /// i * bufferDescriptorSize, then the reserved range at the slot area aligned up to max(64,
    /// bufferDescriptorAlignment, imageDescriptorAlignment). Raises GpuContractException for limits that break the
    /// recipe: a non-positive size, an alignment that is not a power of two, a descriptor size that is not a multiple
    /// of its alignment, a negative reserved range, or a heap above maxResourceHeapSize.
    let vulkanGeometry (limits: VulkanHeapLimits) : VulkanHeapGeometry =
        let l = limits

        if l.BufferDescriptorSize <= 0L then
            Fail.contract "bufferDescriptorSize %d must be positive" l.BufferDescriptorSize

        if
            not (isPowerOfTwo l.BufferDescriptorAlignment)
            || not (isPowerOfTwo l.ImageDescriptorAlignment)
        then
            Fail.contract
                "descriptor alignments %d and %d must be powers of two"
                l.BufferDescriptorAlignment
                l.ImageDescriptorAlignment

        if l.BufferDescriptorSize % l.BufferDescriptorAlignment <> 0L then
            Fail.contract
                "bufferDescriptorSize %d is not a multiple of bufferDescriptorAlignment %d"
                l.BufferDescriptorSize
                l.BufferDescriptorAlignment

        if l.MinResourceHeapReservedRange < 0L then
            Fail.contract "minResourceHeapReservedRange %d is negative" l.MinResourceHeapReservedRange

        let slotArea = int64 SlotCount * l.BufferDescriptorSize

        let reservedOffset =
            alignUp (max 64L (max l.BufferDescriptorAlignment l.ImageDescriptorAlignment)) slotArea

        let heapBytes = reservedOffset + l.MinResourceHeapReservedRange

        if heapBytes > l.MaxResourceHeapSize then
            Fail.contract
                "the resource heap needs %d bytes, more than maxResourceHeapSize %d"
                heapBytes
                l.MaxResourceHeapSize

        {
            SlotStride = l.BufferDescriptorSize
            SlotAreaBytes = slotArea
            ReservedRangeOffset = reservedOffset
            ReservedRangeBytes = l.MinResourceHeapReservedRange
            HeapBytes = heapBytes
        }

    /// Vulkan: the byte offset of slot `slot` in a heap of geometry `g`.
    let vulkanSlotOffset (g: VulkanHeapGeometry) (slot: int) =
        if slot < 0 || slot >= SlotCount then
            Fail.contract "slot %d is outside 0..%d" slot (SlotCount - 1)

        int64 slot * g.SlotStride

    /// D3D12: the descriptor index of slot `slot` in the shader-visible CBV/SRV/UAV heap (a raw R32_TYPELESS UAV).
    let d3d12DescriptorIndex (slot: int) =
        if slot < 0 || slot >= SlotCount then
            Fail.contract "slot %d is outside 0..%d" slot (SlotCount - 1)

        slot

/// Where the slots of a heap point: slot 0 is the tensor table, slots 1 to N are pages in arena order. The remaining
/// slots hold null descriptors on D3D12 and, because Vulkan has no null descriptors without robustness2, a small
/// dummy page's descriptor on Vulkan (VulkanHeapRecipe); no table row names them. A layout is immutable and is
/// written to the heap once.
[<Sealed>]
type HeapLayout internal (pages: HeapPage[]) =
    let bySlot = pages |> Array.map (fun p -> p.Slot, p) |> dict

    /// The pages, in slot order.
    member _.Pages: HeapPage list = List.ofArray pages

    /// The number of pages (slots 1 to PageCount).
    member _.PageCount = pages.Length

    /// The page in `slot`, or None for slot 0 and for unused slots.
    member _.TryPage(slot: int) =
        match bySlot.TryGetValue slot with
        | true, p -> Some p
        | _ -> None

    /// The page in `slot`. Raises GpuContractException for slot 0 and unused slots.
    member x.Page(slot: int) =
        match x.TryPage slot with
        | Some p -> p
        | None ->
            Fail.contract
                "slot %d holds no page (the layout has %d pages in slots 1..%d)"
                slot
                pages.Length
                pages.Length

    /// The slot of page `index` of `arena`.
    member _.SlotOf(arena: ArenaKind, index: int) =
        match pages |> Array.tryFind (fun p -> p.Arena = arena && p.Index = index) with
        | Some p -> p.Slot
        | None -> Fail.contract "the layout has no page %d in arena %s" index (ArenaKind.name arena)

    /// Bytes of all pages of `arena`.
    member _.ArenaBytes(arena: ArenaKind) =
        pages |> Array.sumBy (fun p -> if p.Arena = arena then p.Bytes else 0L)

    /// The arenas that have pages, in contract order.
    member _.Arenas = pages |> Array.map _.Arena |> Array.distinct |> List.ofArray

    /// Bytes of the table and every page.
    member _.TotalBytes = Heap.TableBytes + (pages |> Array.sumBy _.Bytes)

    override _.Equals(other: obj) =
        match other with
        | :? HeapLayout as o -> o.Pages = List.ofArray pages
        | _ -> false

    override _.GetHashCode() = hash (List.ofArray pages)

    override _.ToString() =
        pages
        |> Array.map (fun p -> sprintf "%d:%s/%d:%d" p.Slot (ArenaKind.name p.Arena) p.Index p.Bytes)
        |> String.concat " "
        |> sprintf "HeapLayout(table; %s)"

[<RequireQualifiedAccess>]
module HeapLayout =
    /// Checks a page list and builds the layout: pages in arena order (each arena's pages contiguous), sizes multiples
    /// of 64 KiB from 64 KiB to 1 GiB, at most 63 pages, and no reserved arena (contracts-v1).
    let tryOfPages (pages: (ArenaKind * int64) list) : Result<HeapLayout, string> =
        let problems = ResizeArray<string>()

        if pages.Length > Heap.SlotCount - 1 then
            problems.Add(sprintf "%d pages do not fit in slots 1..%d" pages.Length (Heap.SlotCount - 1))

        pages
        |> List.iteri (fun i (arena, bytes) ->
            if ArenaKind.isReserved arena then
                problems.Add(sprintf "page %d: arena %s is reserved until contracts-v2" i (ArenaKind.name arena))

            if bytes <= 0L || bytes > Heap.MaxPageBytes then
                problems.Add(sprintf "page %d: %d bytes is outside 64 KiB..1 GiB" i bytes)
            elif bytes % Heap.PageGranularity <> 0L then
                problems.Add(sprintf "page %d: %d bytes is not a multiple of 64 KiB" i bytes))

        pages
        |> List.pairwise
        |> List.iteri (fun i ((a, _), (b, _)) ->
            if ArenaKind.code b < ArenaKind.code a then
                problems.Add(
                    sprintf
                        "page %d: arena %s comes after %s (contract order)"
                        (i + 1)
                        (ArenaKind.name b)
                        (ArenaKind.name a)
                ))

        if problems.Count > 0 then
            Error(String.concat "; " problems)
        else
            let counters = Dictionary<ArenaKind, int>()

            pages
            |> List.mapi (fun i (arena, bytes) ->
                let index =
                    match counters.TryGetValue arena with
                    | true, n -> n
                    | _ -> 0

                counters[arena] <- index + 1

                {
                    Slot = i + 1
                    Arena = arena
                    Index = index
                    Bytes = bytes
                })
            |> Array.ofList
            |> fun a -> Ok(new HeapLayout(a))

    /// tryOfPages, raising GpuContractException on a bad page list.
    let ofPages (pages: (ArenaKind * int64) list) =
        match tryOfPages pages with
        | Ok l -> l
        | Error e -> Fail.contract "invalid heap layout: %s" e

    /// A layout with no pages: only the tensor table.
    let empty = ofPages []

/// Where an allocation landed: page `Page` of `Arena`, at byte `Offset` of that page.
type Placement =
    {
        Arena: ArenaKind
        /// The page's index within its arena.
        Page: int
        /// Byte offset inside the page, a multiple of 256.
        Offset: int64
        /// The requested size.
        Bytes: int64
    }

/// Pass one of a static plan: places allocations into arena pages (next fit, 256-byte aligned, never straddling a
/// page), then Build turns the fill levels into a HeapLayout whose pages are rounded up to 64 KiB. Deterministic: the
/// same allocation sequence gives the same placements and layout.
[<Sealed>]
type HeapPlanner(pageBytes: int64) =
    do
        if
            pageBytes < Heap.PageGranularity
            || pageBytes > Heap.MaxPageBytes
            || pageBytes % Heap.PageGranularity <> 0L
        then
            Fail.contract "page size %d must be a multiple of 64 KiB from 64 KiB to 1 GiB" pageBytes

    let fills = Dictionary<ArenaKind, ResizeArray<int64>>()
    let mutable built = false

    /// A planner with 1 GiB pages.
    new() = HeapPlanner(Heap.MaxPageBytes)

    /// The largest page this planner makes.
    member _.PageBytes = pageBytes

    /// Places `bytes` in `arena`: at the end of the arena's last page when it fits there, otherwise at the start of a
    /// new page. Raises GpuContractException for a reserved arena, a size outside 1..PageBytes, or after Build.
    member _.Allocate(arena: ArenaKind, bytes: int64) : Placement =
        if built then
            Fail.contract "HeapPlanner.Allocate after Build"

        if ArenaKind.isReserved arena then
            Fail.contract "arena %s is reserved until contracts-v2" (ArenaKind.name arena)

        if bytes <= 0L || bytes > pageBytes then
            Fail.contract
                "allocation of %d bytes in %s must be 1..%d bytes (one page)"
                bytes
                (ArenaKind.name arena)
                pageBytes

        let pages =
            match fills.TryGetValue arena with
            | true, p -> p
            | _ ->
                let p = ResizeArray<int64>()
                fills[arena] <- p
                p

        let start =
            if pages.Count = 0 then
                None
            else
                let offset = Heap.alignUp Heap.AllocationAlignment pages[pages.Count - 1]

                if offset + bytes <= pageBytes then Some offset else None

        match start with
        | Some offset ->
            pages[pages.Count - 1] <- offset + bytes

            {
                Arena = arena
                Page = pages.Count - 1
                Offset = offset
                Bytes = bytes
            }
        | None ->
            pages.Add bytes

            {
                Arena = arena
                Page = pages.Count - 1
                Offset = 0L
                Bytes = bytes
            }

    /// Bytes used so far in `arena`, summed over its pages.
    member _.Used(arena: ArenaKind) =
        match fills.TryGetValue arena with
        | true, p -> Seq.sum p
        | _ -> 0L

    /// The layout: every arena's pages in contract order, each rounded up to 64 KiB. Allocate is closed afterwards.
    member _.Build() : HeapLayout =
        built <- true

        ArenaKind.all
        |> List.collect (fun arena ->
            match fills.TryGetValue arena with
            | true, p ->
                p
                |> Seq.map (fun used -> arena, Heap.alignUp Heap.PageGranularity used)
                |> List.ofSeq
            | _ -> [])
        |> HeapLayout.ofPages
