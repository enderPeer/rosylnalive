namespace Lrm.Gpu.Vulkan

#nowarn "9"
#nowarn "51"

open System
open System.IO
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop
open Microsoft.FSharp.NativeInterop

/// Where the resource heap lives (VulkanHeapRecipe section 2).
[<RequireQualifiedAccess>]
type HeapPlacement =
    /// The recipe: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory (the BAR), written in place through its persistent
    /// mapping. Falls back to Copied on a device without such a memory type.
    | InPlace
    /// The recipe's fallback, forced: DEVICE_LOCAL memory, the descriptors written to host memory and copied in with one
    /// vkCmdCopyBuffer, then a COPY to RESOURCE_HEAP_READ barrier.
    | Copied

/// The resource heap of one backend (HeapLayoutAndTensorTable, VulkanHeapRecipe sections 2 to 4): the 1 MiB table and
/// every page as DEVICE_LOCAL buffers with device addresses, a 64 KiB dummy page behind the unused slots (there are no
/// null descriptors without robustness2), and one resource heap of geometry `geometry` whose 64 slots are written
/// once, in one vkWriteResourceDescriptorsEXT call: slot i at byte i * bufferDescriptorSize from heapRange.address. The
/// reserved range at the end is never written or read.
[<Sealed>]
type internal VulkanHeap
    (ctx: DeviceContext, layout: HeapLayout, label: string, geometry: VulkanHeapGeometry, placement: HeapPlacement) =
    let c = ctx.Commands
    let dev = ctx.Handle
    let caps = ctx.Capabilities

    let limits =
        match caps.DescriptorHeap with
        | Some h -> h
        | None -> invalidOp "vulkan: the device has no VK_EXT_descriptor_heap limits"

    let placement =
        match placement with
        | HeapPlacement.InPlace when not (MemoryTypes.hasHeapInPlace ctx.MemoryTypes) -> HeapPlacement.Copied
        | p -> p

    let buffers: NativeBuffer option[] = Array.create Heap.SlotCount None
    let mutable dummy: NativeBuffer option = None
    let mutable heapBuffer: NativeBuffer option = None
    let mutable range: HeapRange option = None
    let mutable descriptorWrites = 0
    let mutable destinations: (int64 * int64)[] = [||]
    let mutable table: SealedTable option = None
    let mutable disposed = false

    let release () =
        for i in 0 .. buffers.Length - 1 do
            buffers[i] |> Option.iter ctx.DestroyBuffer
            buffers[i] <- None

        dummy |> Option.iter ctx.DestroyBuffer
        dummy <- None
        heapBuffer |> Option.iter ctx.DestroyBuffer
        heapBuffer <- None

    let alignUp (a: uint64) (v: uint64) =
        if a <= 1UL then v else (v + a - 1UL) / a * a

    do
        try
            let page (bytes: int64) (name: string) =
                let b = ctx.CreateBuffer(BufferKind.Page, bytes, name)
                let align = max 1UL caps.MinStorageBufferOffsetAlignment

                if b.Address % align <> 0UL then
                    ctx.DestroyBuffer b

                    invalidOp (
                        sprintf
                            "vulkan: %s's device address 0x%x is not a multiple of minStorageBufferOffsetAlignment %d"
                            name
                            b.Address
                            align
                    )

                b

            buffers[Heap.TableSlot] <- Some(page Heap.TableBytes (label + ".table"))

            for p in layout.Pages do
                buffers[p.Slot] <-
                    Some(page p.Bytes (sprintf "%s.page%d.%s%d" label p.Slot (ArenaKind.name p.Arena) p.Index))

            if layout.PageCount < Heap.SlotCount - 1 then
                dummy <- Some(page Heap.PageGranularity (label + ".dummy"))

            let kind =
                match placement with
                | HeapPlacement.InPlace -> BufferKind.ResourceHeap
                | HeapPlacement.Copied -> BufferKind.ResourceHeapCopied

            let alignment = max 1UL limits.ResourceHeapAlignment

            // Slack of one alignment, so heapRange.address can be aligned up inside the buffer (recipe section 3).
            let hb =
                ctx.CreateBuffer(kind, geometry.HeapBytes + int64 alignment, label + ".resource_heap")

            heapBuffer <- Some hb
            let address = alignUp alignment hb.Address
            let offset = int64 (address - hb.Address)

            range <-
                Some
                    {
                        Address = address
                        Bytes = uint64 geometry.HeapBytes
                        ReservedOffset = uint64 geometry.ReservedRangeOffset
                        ReservedBytes = uint64 geometry.ReservedRangeBytes
                    }

            let slotBuffer (slot: int) =
                match buffers[slot] with
                | Some b -> b
                | None -> dummy.Value

            let ranges =
                Array.init Heap.SlotCount (fun i ->
                    let b = slotBuffer i
                    let mutable r = VkDeviceAddressRangeEXT()
                    r.address <- b.Address
                    r.size <- uint64 b.Bytes
                    r)

            use pRanges = fixed ranges

            let infos =
                Array.init Heap.SlotCount (fun i ->
                    let mutable x = VkResourceDescriptorInfoEXT.Create()
                    x.``type`` <- VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    x.data.pAddressRange <- NativePtr.toNativeInt (NativePtr.add pRanges i)
                    x)

            // The descriptors go straight into the mapped heap, or into host staging for the copy path.
            let staging =
                match placement with
                | HeapPlacement.InPlace -> None
                | HeapPlacement.Copied ->
                    Some(ctx.CreateBuffer(BufferKind.Upload, geometry.SlotAreaBytes, label + ".descriptor_staging"))

            try
                let hostBase =
                    match staging with
                    | Some s -> s.Mapped
                    | None -> hb.Mapped + nativeint offset

                destinations <-
                    Array.init Heap.SlotCount (fun i -> Heap.vulkanSlotOffset geometry i, geometry.SlotStride)

                let ds =
                    destinations
                    |> Array.map (fun (at, n) ->
                        let mutable h = VkHostAddressRangeEXT()
                        h.address <- hostBase + nativeint at
                        h.size <- unativeint n
                        h)

                c.WriteResourceDescriptorsEXT(dev, ReadOnlySpan infos, ReadOnlySpan ds)
                descriptorWrites <- Heap.SlotCount

                match staging with
                | Some s ->
                    let bindings =
                        {
                            Heap = None
                            Pipeline = fun _ -> invalidOp "no pipelines"
                            Buffer =
                                function
                                | Target.HeapStaging -> s
                                | Target.HeapBuffer -> hb
                                | t -> invalidOp (sprintf "%O is not part of the heap fill" t)
                        }

                    ctx.Execute (Some bindings) 0 (Plan.heapUpload offset geometry.SlotAreaBytes) None
                    |> ignore

                    ctx.WaitIdle()
                | None -> ()
            finally
                staging |> Option.iter ctx.DestroyBuffer
        with _ ->
            (try
                ctx.ResetFrames()
             with _ ->
                 ())

            release ()
            reraise ()

    member _.Layout = layout
    member _.Label = label
    member _.Geometry = geometry
    member _.Placement = placement

    /// Descriptors written when the heap was created: always Heap.SlotCount (64), in one call, never again.
    member _.DescriptorWrites = descriptorWrites

    /// Every byte range of the heap that a descriptor write targeted, as (offset from heapRange.address, bytes): the 64
    /// slots, all inside the slot area, none in the reserved range.
    member _.DescriptorDestinations = destinations

    /// heapRange: the address and size that vkCmdBindResourceHeapEXT binds, with the reserved range.
    member _.Range = range.Value

    /// The device address and size of the buffer behind each slot (the dummy page for unused slots).
    member _.SlotRange(slot: int) : uint64 * int64 =
        match buffers[slot] with
        | Some b -> b.Address, b.Bytes
        | None -> dummy.Value.Address, dummy.Value.Bytes

    /// The buffer behind a slot, or None for unused slots.
    member _.SlotBuffer(slot: int) = buffers[slot]

    /// Bytes of the resource heap buffer (heapRange.size plus one alignment of slack).
    member _.ResourceHeapBytes = heapBuffer |> Option.map _.Bytes |> Option.defaultValue 0L

    /// Bytes of the dummy page behind the unused slots (0 when all 63 page slots hold pages).
    member _.DummyPageBytes = dummy |> Option.map _.Bytes |> Option.defaultValue 0L

    /// Bytes of device memory the heap holds outside its arenas: the resource heap buffer and the dummy page.
    member x.OverheadBytes = x.ResourceHeapBytes + x.DummyPageBytes

    /// The memory type of the resource heap buffer.
    member _.HeapMemoryType = heapBuffer.Value.MemoryType

    /// The slot area's current bytes (64 * bufferDescriptorSize from heapRange.address), read through the mapping of an
    /// in-place heap; None for a copied heap. The reserved range is never read.
    member _.SlotAreaBytes() : byte[] option =
        match placement, heapBuffer, range with
        | HeapPlacement.InPlace, Some hb, Some r ->
            let out = Array.zeroCreate<byte>(int geometry.SlotAreaBytes)
            Marshal.Copy(hb.Mapped + nativeint (r.Address - hb.Address), out, 0, out.Length)
            Some out
        | _ -> None

    member _.Table = table

    member _.TableOrFail =
        match table with
        | Some t -> t
        | None -> Contract.fail "WriteTable has not run"

    /// The bindings of a command buffer on this heap.
    member _.Bindings(pipeline: KernelId -> VkPipeline) : Bindings =
        {
            Heap = range
            Pipeline = pipeline
            Buffer =
                fun t ->
                    match t with
                    | Target.Page s ->
                        match buffers[s] with
                        | Some b -> b
                        | None -> invalidOp (sprintf "slot %d holds no page" s)
                    | _ -> invalidOp (sprintf "%O is not a heap buffer" t)
        }

    /// Writes `bytes` bytes to `slot` at `offset` (slot 0 included: WriteTable uses it), after all submitted work.
    /// `fill dest at n` fills the staging memory `dest` (a native pointer of `n` bytes) with source bytes `at..at+n`.
    member x.Write(slot: int, offset: int64, bytes: int64, fill: nativeint -> int64 -> int -> unit) =
        ctx.WaitIdle()

        if bytes > 0L then
            let staging = ctx.UploadStaging(Transfer.stagingBytes bytes) |> Array.ofList
            let bindings = x.Bindings(fun _ -> invalidOp "no pipelines")

            Transfer.chunks bytes
            |> List.iteri (fun k (at, n) ->
                let i = k % 2
                let s = staging[i]
                ctx.WaitFor s.LastValue
                fill s.Mapped at (int n)
                let f = ctx.Execute (Some bindings) 0 (Plan.upload slot (offset + at) i n) None
                s.LastValue <- f.Value)

            ctx.WaitIdle()

    /// Reads `bytes` bytes of `slot` at `offset`, after all submitted work. `drain src at n` takes staging memory `src`
    /// holding source bytes `at..at+n`. The next chunk's copy runs while the previous one drains.
    member x.Read(slot: int, offset: int64, bytes: int64, drain: nativeint -> int64 -> int -> unit) =
        ctx.WaitIdle()

        if bytes > 0L then
            let staging = ctx.ReadbackStaging(Transfer.stagingBytes bytes) |> Array.ofList
            let bindings = x.Bindings(fun _ -> invalidOp "no pipelines")
            let chunks = Transfer.chunks bytes |> Array.ofList

            let submit k =
                let at, n = chunks[k]
                let i = k % 2
                let f = ctx.Execute (Some bindings) 0 (Plan.download slot (offset + at) i n) None
                staging[i].LastValue <- f.Value

            submit 0

            for k in 0 .. chunks.Length - 1 do
                if k + 1 < chunks.Length then
                    submit (k + 1)

                let at, n = chunks[k]
                let s = staging[k % 2]
                ctx.WaitFor s.LastValue
                drain s.Mapped at (int n)

    /// Writes host bytes (pinned for the call).
    member x.WriteSpan(slot: int, offset: int64, data: ReadOnlySpan<byte>) =
        if data.Length > 0 then
            use p = fixed data
            let src = NativePtr.toNativeInt p

            x.Write(
                slot,
                offset,
                int64 data.Length,
                fun dest at n ->
                    Buffer.MemoryCopy(
                        NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte>(src + nativeint at)),
                        NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> dest),
                        int64 n,
                        int64 n
                    )
            )
        else
            ctx.WaitIdle()

    /// Writes `bytes` bytes read from a stream straight into the staging memory.
    member x.WriteStream(slot: int, offset: int64, source: Stream, bytes: int64) =
        x.Write(
            slot,
            offset,
            bytes,
            fun dest _ n -> source.ReadExactly(Span<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> dest), n))
        )

    /// Reads into host memory (pinned for the call).
    member x.ReadSpan(slot: int, offset: int64, destination: Span<byte>) =
        if destination.Length > 0 then
            use p = fixed destination
            let dst = NativePtr.toNativeInt p

            x.Read(
                slot,
                offset,
                int64 destination.Length,
                fun src at n ->
                    Buffer.MemoryCopy(
                        NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> src),
                        NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte>(dst + nativeint at)),
                        int64 n,
                        int64 n
                    )
            )
        else
            ctx.WaitIdle()

    /// WriteTable's rules (GpuBackendInterface): once, against this heap's layout. Then the 1 MiB image goes to slot 0.
    member x.WriteSealed(t: SealedTable) =
        if table.IsSome then
            Contract.fail "the tensor table is written once"

        if not (obj.ReferenceEquals(t.Layout, layout) || t.Layout.Equals layout) then
            Contract.fail "the table was checked against a different layout"

        x.WriteSpan(Heap.TableSlot, 0L, t.Image.Span)
        table <- Some t

    /// Waits for every command buffer that bound a heap and resets it (VulkanHeapRecipe section 4), then destroys the
    /// heap's buffers.
    member _.Dispose() =
        if not disposed then
            disposed <- true

            try
                ctx.ResetFrames()
            finally
                release ()
