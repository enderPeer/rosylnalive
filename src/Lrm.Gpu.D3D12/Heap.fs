namespace Lrm.Gpu.D3D12

#nowarn "9"
#nowarn "51"

open System
open System.IO
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12.Interop
open Microsoft.FSharp.NativeInterop

/// Host transfers through the staging buffers (GpuBackendInterface: Upload, UploadFrom and Download wait for all
/// submitted work, then transfer, in chunks of at most 64 MiB).
[<RequireQualifiedAccess>]
module Transfer =
    /// The largest chunk, which is also the size of each of the two upload and two readback staging buffers.
    [<Literal>]
    let ChunkBytes = 67108864L

    /// The staging size for a transfer of `bytes`: the chunk size, or less for a small transfer (64 KiB steps).
    let stagingBytes (bytes: int64) =
        min ChunkBytes (max Heap.PageGranularity (Heap.alignUp Heap.PageGranularity bytes))

    /// The chunks of a transfer: (offset from the start, size) pairs of at most ChunkBytes.
    let chunks (bytes: int64) =
        [
            let mutable at = 0L

            while at < bytes do
                let n = min ChunkBytes (bytes - at)
                yield at, n
                at <- at + n
        ]

/// The resource heap of one backend (HeapLayoutAndTensorTable): the 1 MiB table and every page as committed DEFAULT
/// buffers in COMMON, and one shader-visible CBV/SRV/UAV heap of exactly 64 descriptors, written once: slot 0 the
/// table and slots 1..N the pages as raw UAVs (R32_TYPELESS, RAW, NumElements = bytes / 4), slots N+1..63 null
/// descriptors.
[<Sealed>]
type internal D3D12Heap(ctx: DeviceContext, layout: HeapLayout, label: string) =
    let device = ctx.Device

    let resources: ComPtr<ID3D12Resource>[] =
        Array.create Heap.SlotCount ComPtr<ID3D12Resource>.Null

    let mutable descriptors = ComPtr<ID3D12DescriptorHeap>.Null
    let mutable descriptorWrites = 0
    let mutable table: SealedTable option = None
    let mutable disposed = false

    do
        try
            resources[Heap.TableSlot] <-
                ctx.CreateBuffer(D3D12_HEAP_TYPE.DEFAULT, Heap.TableBytes, sprintf "%s.table" label)

            for p in layout.Pages do
                resources[p.Slot] <-
                    ctx.CreateBuffer(
                        D3D12_HEAP_TYPE.DEFAULT,
                        p.Bytes,
                        sprintf "%s.page%d.%s%d" label p.Slot (ArenaKind.name p.Arena) p.Index
                    )

            let hdesc =
                Desc.descriptorHeap D3D12_DESCRIPTOR_HEAP_TYPE.CBV_SRV_UAV (uint32 Heap.SlotCount) true

            descriptors <- device.CreateDescriptorHeap(&hdesc)
            descriptors.SetName(label + ".descriptors")

            let increment =
                device.GetDescriptorHandleIncrementSize D3D12_DESCRIPTOR_HEAP_TYPE.CBV_SRV_UAV

            let start = descriptors.GetCPUDescriptorHandleForHeapStart()

            for slot in 0 .. Heap.SlotCount - 1 do
                let at =
                    D3D12_CPU_DESCRIPTOR_HANDLE(
                        ptr = start.ptr + unativeint (Heap.d3d12DescriptorIndex slot) * unativeint increment
                    )

                let bytes =
                    if slot = Heap.TableSlot then
                        Heap.TableBytes
                    else
                        match layout.TryPage slot with
                        | Some p -> p.Bytes
                        | None -> 0L

                // A null descriptor still names a format and dimension; its other fields are ignored.
                let mutable uav = Desc.rawBufferUav 0UL (uint32 (max 1L (bytes / 4L)))

                device.CreateUnorderedAccessView(
                    resources[slot],
                    ComPtr<ID3D12Resource>.Null,
                    NativePtr.toNativeInt &&uav,
                    at
                )

                descriptorWrites <- descriptorWrites + 1
        with _ ->
            for i in 0 .. resources.Length - 1 do
                ComPtr.release &resources[i]

            ComPtr.release &descriptors
            reraise ()

    member _.Layout = layout
    member _.Label = label

    /// Descriptors written when the heap was created: always Heap.SlotCount (64), never again.
    member _.DescriptorWrites = descriptorWrites

    /// The resource behind a slot (ComPtr.Null for unused slots).
    member _.Resource(slot: int) = resources[slot]

    member _.Table = table

    member _.TableOrFail =
        match table with
        | Some t -> t
        | None -> Contract.fail "WriteTable has not run"

    /// The bindings of a command list on this heap.
    member _.Bindings(rootSignature: ComPtr<ID3D12RootSignature>, pipeline: KernelId -> ComPtr<ID3D12PipelineState>) =
        {
            DescriptorHeap = descriptors
            RootSignature = rootSignature
            Pipeline = pipeline
            Resource =
                fun t ->
                    match t with
                    | Target.Page s -> resources[s]
                    | _ -> invalidOp (sprintf "%O is not a heap resource" t)
        }

    /// Writes `bytes` bytes to `slot` at `offset` (slot 0 included: WriteTable uses it), after all submitted work.
    /// `fill dest at` fills the staging memory `dest` (a native pointer of `n` bytes) with source bytes `at..at+n`.
    member x.Write(slot: int, offset: int64, bytes: int64, fill: nativeint -> int64 -> int -> unit) =
        ctx.WaitIdle()

        if bytes > 0L then
            let staging = ctx.UploadStaging(Transfer.stagingBytes bytes) |> Array.ofList

            let bindings =
                x.Bindings(ComPtr<ID3D12RootSignature>.Null, fun _ -> invalidOp "no pipelines")

            Transfer.chunks bytes
            |> List.iteri (fun k (at, n) ->
                let i = k % 2
                let s = staging[i]
                ctx.WaitFor s.LastFence
                fill s.Mapped at (int n)
                let f = ctx.Execute (Some bindings) 0 (Plan.upload slot (offset + at) i n) None
                s.LastFence <- f.FenceValue)

            ctx.WaitIdle()

    /// Reads `bytes` bytes of `slot` at `offset`, after all submitted work. `drain src at n` takes staging memory
    /// `src` holding source bytes `at..at+n`. The next chunk's copy runs while the previous one drains.
    member x.Read(slot: int, offset: int64, bytes: int64, drain: nativeint -> int64 -> int -> unit) =
        ctx.WaitIdle()

        if bytes > 0L then
            let staging = ctx.ReadbackStaging(Transfer.stagingBytes bytes) |> Array.ofList

            let bindings =
                x.Bindings(ComPtr<ID3D12RootSignature>.Null, fun _ -> invalidOp "no pipelines")

            let chunks = Transfer.chunks bytes |> Array.ofList

            let submit k =
                let at, n = chunks[k]
                let i = k % 2
                let f = ctx.Execute (Some bindings) 0 (Plan.download slot (offset + at) i n) None
                staging[i].LastFence <- f.FenceValue

            submit 0

            for k in 0 .. chunks.Length - 1 do
                if k + 1 < chunks.Length then
                    submit (k + 1)

                let at, n = chunks[k]
                let s = staging[k % 2]
                ctx.WaitFor s.LastFence
                let mutable range = D3D12_RANGE(Begin = 0un, End = unativeint n)
                let resource = s.Resource
                let p = resource.Map(0u, NativePtr.toNativeInt &&range)

                try
                    drain p at (int n)
                finally
                    let mutable written = D3D12_RANGE(Begin = 0un, End = 0un)
                    resource.Unmap(0u, NativePtr.toNativeInt &&written)

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

    member _.Dispose() =
        if not disposed then
            disposed <- true

            for i in 0 .. resources.Length - 1 do
                ComPtr.release &resources[i]

            ComPtr.release &descriptors
