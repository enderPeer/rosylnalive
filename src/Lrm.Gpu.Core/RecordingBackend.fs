namespace Lrm.Gpu.Core

open System
open System.Collections.Generic
open System.IO
open Lrm.Core

/// A command as RecordingBackend recorded it.
[<RequireQualifiedAccess>]
type RecordedCommand =
    | Dispatch of PreparedDispatch
    | Barrier
    | Copy of source: BufferRange * destination: BufferRange
    | BeginTimer of timer: TimerId * name: string * attrs: Attr list
    | EndTimer of timer: TimerId

/// One submitted command list.
type RecordedSubmission =
    {
        Submission: Submission
        Commands: RecordedCommand list
    }

/// An IGpuBackend for unit tests and planners (GpuBackendInterface contract). It enforces every interface rule the
/// real backends enforce and keeps page memory on the host, so Upload, Download and Copy work byte for byte, but it
/// never runs a kernel: dispatches are checked (Dispatch.prepare) and recorded. Timers get synthetic timestamps
/// (1,000 ns per command), so their results are deterministic.
[<Sealed>]
type RecordingBackend(info: DeviceInfo, options: DeviceOptions) =
    let mutable heap: RecordingHeap option = None
    let mutable kernels: (ShaderSet * IKernelSet) option = None
    let submissions = ResizeArray<RecordedSubmission>()
    let completions = Dictionary<uint64, Completion>()
    let mutable disposed = false

    let checkLive () =
        if disposed then
            raise (ObjectDisposedException "RecordingBackend")

    /// The default device: a d3d12-like recording device with PCI ids 0 and driver "recording".
    static member DefaultDevice =
        {
            Backend = ExecutionBackend.D3D12
            Index = 0
            Name = "Recording backend"
            VendorId = 0u
            PciDeviceId = 0u
            Driver = "recording"
            MemoryBytes = 16L <<< 30
            Details = [ "kind", "recording"; "timing", "synthetic" ]
        }

    /// A recording backend on the default device.
    new() = new RecordingBackend(RecordingBackend.DefaultDevice, DeviceOptions.For ExecutionBackend.D3D12)

    /// Every submission so far, in order.
    member _.Submissions: RecordedSubmission list = List.ofSeq submissions

    /// The heap, once CreateHeap has run.
    member internal _.Heap = heap

    member internal _.HeapOrFail =
        match heap with
        | Some h -> h
        | None -> Fail.contract "CreateHeap has not run"

    member internal _.KernelsOrFail =
        match kernels with
        | Some(_, k) -> k
        | None -> Fail.contract "LoadKernels has not run"

    member internal _.Backend = info.Backend
    member internal _.Options = options

    member internal _.Complete(commands: RecordedCommand list) : Submission =
        let s = Submission(uint64 (submissions.Count + 1))
        submissions.Add { Submission = s; Commands = commands }
        let h = heap

        // Execute copies in order; dispatches are only recorded. Timers get synthetic timestamps.
        let mutable clock = 0L
        let opened = Dictionary<int, string * Attr list * int64>()
        let timers = SortedDictionary<int, TimerResult>()
        let dispatchTimers = ResizeArray<TimerResult>()
        let mutable dispatches = 0

        for c in commands do
            clock <- clock + 1000L

            match c with
            | RecordedCommand.Dispatch d ->
                dispatches <- dispatches + 1

                if options.TimeDispatches then
                    dispatchTimers.Add
                        {
                            Name = GpuTelemetry.Dispatch
                            Attrs = [ Attr.str TelemetryKeys.Kernel d.Spec.Name ]
                            GpuStartNs = clock
                            DurationNs = 1000L
                        }
            | RecordedCommand.Copy(src, dst) ->
                match h with
                | Some h -> h.CopyNow(src, dst)
                | None -> ()
            | RecordedCommand.BeginTimer(TimerId id, name, attrs) -> opened[id] <- (name, attrs, clock)
            | RecordedCommand.EndTimer(TimerId id) ->
                let name, attrs, start = opened[id]

                timers[id] <-
                    {
                        Name = name
                        Attrs = attrs
                        GpuStartNs = start
                        DurationNs = clock - start
                    }
            | RecordedCommand.Barrier -> ()

        GpuTelemetry.dispatches info.Backend dispatches

        completions[s.Value] <-
            {
                Submission = s
                Dispatches = dispatches
                // Timers in the order they were opened, then the per-dispatch timings.
                Timers = List.ofSeq timers.Values @ List.ofSeq dispatchTimers
            }

        s

    interface IGpuBackend with
        member _.Info = info

        member _.ExecutionClass =
            checkLive ()

            match kernels with
            | Some(set, _) when ExecutionBackend.runsShaders info.Backend -> GpuClass.fill info set
            | _ when not (ExecutionBackend.runsShaders info.Backend) -> ExecutionClass.forCpu info.Backend
            | _ -> Fail.contract "the execution class of a shader backend is known after LoadKernels"

        member this.CreateHeap(layout: HeapLayout) =
            checkLive ()

            if heap.IsSome then
                Fail.contract "CreateHeap runs once per backend"

            let h = RecordingHeap(this, layout)
            heap <- Some h
            GpuTelemetry.heapCreated info.Backend layout
            h :> IGpuHeap

        member _.LoadKernels(shaders: ShaderSet) =
            checkLive ()

            if kernels.IsSome then
                Fail.contract "LoadKernels runs once per backend"

            if shaders.CatalogHash <> Catalog.hash Kernels.catalog then
                Fail.contract
                    "the shader set was built for catalog %s, this build has %s"
                    shaders.CatalogHash.Short
                    (Catalog.hash Kernels.catalog).Short

            let target =
                ShaderTarget.forBackend info.Backend |> Option.defaultValue ShaderTarget.Dxil

            let names = ShaderSet.blobsFor target shaders |> List.map _.Kernel |> Set.ofList

            let hash = ShaderSet.setHash target shaders

            let set =
                { new IKernelSet with
                    member _.ShaderSetHash = hash

                    member _.Contains(k: KernelId) =
                        k.Value >= 0
                        && k.Value < Kernels.all.Length
                        && names.Contains Kernels.all[k.Value].Name
                }

            kernels <- Some(shaders, set)
            set

        member this.Begin() =
            checkLive ()
            RecordingCommandList(this) :> ICommandList

        member this.Submit(commands: ICommandList) =
            checkLive ()

            match commands with
            | :? RecordingCommandList as c when obj.ReferenceEquals(c.Owner, this) -> this.Complete(c.Close())
            | _ -> Fail.contract "Submit takes a command list from this backend's Begin"

        member _.Wait(s: Submission) =
            checkLive ()

            use _ =
                Telemetry.span GpuTelemetry.Wait [ Attr.str TelemetryKeys.Backend (ExecutionBackend.name info.Backend) ]

            match completions.TryGetValue s.Value with
            | true, c ->
                GpuTelemetry.timers info.Backend c.Timers
                c
            | _ -> Fail.contract "submission %d was not made by this backend" s.Value

        member _.Dispose() = disposed <- true

/// The heap of a RecordingBackend: host memory per page, allocated on first write.
and [<Sealed>] internal RecordingHeap(owner: RecordingBackend, layout: HeapLayout) =
    let pages = Dictionary<int, byte[]>()
    let mutable table: SealedTable option = None

    let page (slot: int) =
        match pages.TryGetValue slot with
        | true, p -> p
        | _ ->
            let p = Array.zeroCreate<byte>(int (layout.Page slot).Bytes)
            pages[slot] <- p
            p

    /// The bytes of a page slot (allocated on first use), or the table image for slot 0.
    member _.Bytes(slot: int) : byte[] =
        if slot = Heap.TableSlot then
            match table with
            | Some t -> t.Image.ToArray()
            | None -> Array.zeroCreate (int Heap.TableBytes)
        else
            page slot

    member _.TableOrFail =
        match table with
        | Some t -> t
        | None -> Fail.contract "WriteTable has not run"

    member _.CopyNow(src: BufferRange, dst: BufferRange) =
        let s = page src.Slot
        let d = page dst.Slot
        Buffer.BlockCopy(s, int src.Offset, d, int dst.Offset, int src.Bytes)
        GpuTelemetry.bytesMoved owner.Backend "copy" src.Bytes

    interface IGpuHeap with
        member _.Layout = layout
        member _.Table = table

        member _.WriteTable(t: SealedTable) =
            if table.IsSome then
                Fail.contract "the tensor table is written once"

            if not (obj.ReferenceEquals(t.Layout, layout) || t.Layout.Equals layout) then
                Fail.contract "the table was checked against a different layout"

            table <- Some t
            GpuTelemetry.bytesMoved owner.Backend "upload" Heap.TableBytes

        member _.Upload(slot: int, offset: int64, data: ReadOnlySpan<byte>) =
            Dispatch.checkTransfer layout false slot offset (int64 data.Length)
            data.CopyTo(Span(page slot, int offset, data.Length))
            GpuTelemetry.bytesMoved owner.Backend "upload" (int64 data.Length)

        member _.UploadFrom(slot: int, offset: int64, source: Stream, length: int64) =
            Dispatch.checkTransfer layout false slot offset length
            let p = page slot
            let mutable done' = 0L
            let chunk = 64 <<< 20

            while done' < length do
                let n = int (min (int64 chunk) (length - done'))
                source.ReadExactly(p, int (offset + done'), n)
                done' <- done' + int64 n

            GpuTelemetry.bytesMoved owner.Backend "upload" length

        member x.Download(slot: int, offset: int64, destination: Span<byte>) =
            Dispatch.checkTransfer layout true slot offset (int64 destination.Length)

            if slot <> Heap.TableSlot && not (pages.ContainsKey slot) then
                destination.Clear()
            else
                ReadOnlySpan(x.Bytes slot, int offset, destination.Length).CopyTo destination

            GpuTelemetry.bytesMoved owner.Backend "download" (int64 destination.Length)

/// A command list of a RecordingBackend.
and [<Sealed>] internal RecordingCommandList(owner: RecordingBackend) =
    let commands = ResizeArray<RecordedCommand>()
    let openTimers = Stack<int>()
    let mutable nextTimer = 0
    let mutable closed = false

    let checkOpen () =
        if closed then
            Fail.contract "the command list was already submitted"

    member _.Owner = owner

    member _.Close() =
        checkOpen ()

        if openTimers.Count > 0 then
            Fail.contract "%d timer(s) are still open at Submit" openTimers.Count

        closed <- true
        List.ofSeq commands

    interface ICommandList with
        member _.Dispatch(kernel: KernelId, args: ArgBlock) =
            checkOpen ()
            let heap = owner.HeapOrFail
            let set = owner.KernelsOrFail
            let prepared = Dispatch.prepare heap.TableOrFail kernel args

            if not (set.Contains kernel) then
                Fail.contract "kernel %s is not in the loaded shader set" prepared.Spec.Name

            commands.Add(RecordedCommand.Dispatch prepared)

        member _.Barrier() =
            checkOpen ()
            commands.Add RecordedCommand.Barrier

        member _.Copy(source: BufferRange, destination: BufferRange) =
            checkOpen ()
            Dispatch.checkCopy (owner.HeapOrFail :> IGpuHeap).Layout source destination
            commands.Add(RecordedCommand.Copy(source, destination))

        member _.BeginTimer(name: string, attrs: Attr list) =
            checkOpen ()

            if not (TelemetryNames.isValid name) then
                Fail.contract "timer name '%s' is not a valid telemetry name" name

            let id = nextTimer
            nextTimer <- nextTimer + 1
            openTimers.Push id
            commands.Add(RecordedCommand.BeginTimer(TimerId id, name, attrs))
            TimerId id

        member _.EndTimer(TimerId id) =
            checkOpen ()

            if openTimers.Count = 0 || openTimers.Peek() <> id then
                Fail.contract "timer %d is not the most recently opened timer" id

            openTimers.Pop() |> ignore
            commands.Add(RecordedCommand.EndTimer(TimerId id))

/// Helpers for tests that use RecordingBackend.
[<RequireQualifiedAccess>]
module Recording =
    /// The bytes of a page (or of the table, slot 0) of a recording backend's heap.
    let pageBytes (backend: RecordingBackend) (slot: int) =
        match backend.Heap with
        | Some h -> Array.copy (h.Bytes slot)
        | None -> Fail.contract "CreateHeap has not run"

    /// Every dispatch of every submission, in order.
    let dispatches (backend: RecordingBackend) =
        backend.Submissions
        |> List.collect (fun s ->
            s.Commands
            |> List.choose (function
                | RecordedCommand.Dispatch d -> Some d
                | _ -> None))

    /// A provider that enumerates one recording device, for registry tests.
    let provider (device: DeviceInfo) =
        { new IGpuBackendProvider with
            member _.Backend = device.Backend
            member _.Enumerate() = [ { Info = device; Rejections = [] } ]

            member _.Create(d, o) =
                new RecordingBackend(d, o) :> IGpuBackend
        }
