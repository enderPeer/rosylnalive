namespace Lrm.Gpu.CpuDevice

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Runtime.ExceptionServices
open System.Runtime.InteropServices
open System.Threading.Tasks
open Lrm.Core
open Lrm.Gpu.Core

/// How a CpuDevice behaves (ConformanceHarness contract, "CpuDevice").
type CpuDeviceOptions =
    {
        /// Threads that run a dispatch's thread groups, at least 1. Groups write disjoint outputs (kernel rules), so the
        /// bits are the same for every count.
        Threads: int
        /// Preserve (the default) emulates CpuRef F32 exactly. Flush is a model of a flushing GPU class: the kernels that
        /// compute on fp32 values see every fp32 and bf16 denormal input and fp32 scalar argument as zero of the same
        /// sign, and their fp32 results are flushed the same way. Mixed is refused.
        Fp32Denormals: Fp32DenormalMode
        /// A model of the GPUs' NaN rule (ADR R1-heap-recipe pitfall 10): every NaN result of an fp32 arithmetic kernel
        /// becomes 0x7FFFFFFF. False (the default) keeps CpuRef's payload propagation.
        CanonicalNaN: bool
        /// Raise GpuContractException when a dispatch reads or writes bytes that an earlier dispatch of the same command
        /// list wrote (or writes bytes it read) with no Barrier or Copy between them (GpuBackendInterface, Barrier).
        CheckBarriers: bool
        /// Report Contains = false for kernels without a blob in the shader set, as a shader backend does. False (the
        /// default): every catalog kernel has an emulator, sources or not.
        MirrorShaderSet: bool
        /// The word every byte of a page or table buffer holds before anything writes it. Real device memory starts
        /// undefined, so nothing may rely on zeros.
        UninitializedWord: uint32
    }

    /// One thread, preserve, CpuRef's NaN, barrier checks on, every kernel, uninitialized memory 0x7FBADBAD.
    static member Default =
        {
            Threads = 1
            Fp32Denormals = Fp32DenormalMode.Preserve
            CanonicalNaN = false
            CheckBarriers = true
            MirrorShaderSet = false
            UninitializedWord = 0x7FBADBADu
        }

/// One dispatch as CpuDevice ran it, kept when DeviceOptions.Debug is set: the kernel, the 32 argument words decoded
/// from the 128 bytes of the block (word 0 is the physical x) and the physical grid it ran over.
type ExecutedDispatch =
    {
        Kernel: string
        Words: uint32[]
        Grid: PhysicalGrid
    }

/// A command as a CpuDevice command list holds it until Submit runs it.
[<RequireQualifiedAccess>]
type internal CpuCommand =
    | Dispatch of prepared: PreparedDispatch * footprint: FieldFootprint list
    | Barrier
    | Copy of source: BufferRange * destination: BufferRange
    | BeginTimer of timer: int * name: string * attrs: Attr list
    | EndTimer of timer: int

/// One access a recorded dispatch makes, in bytes of a slot, for the barrier checks.
type internal ByteAccess =
    {
        Index: int
        Kernel: string
        Field: string
        Slot: int
        Bytes: ElementSet
        Writes: bool
    }

[<RequireQualifiedAccess>]
module internal ByteSets =
    /// The bytes of an element set of a token at `offset` with elements of `size` bytes, in whole u32 words (kernels
    /// load whole words, so a bf16 element owns its word for hazard purposes).
    let ofElements (offset: int64) (size: int) (s: ElementSet) =
        if s.IsEmpty then
            ElementSet.Range(offset, offset)
        elif size = 4 then
            {
                Start = offset + s.Start * 4L
                Stride = s.Stride * 4L
                Length = s.Length * 4L
                Count = s.Count
            }
        else
            let lo = (offset + s.Start * int64 size) &&& ~~~3L
            let hi = Heap.alignUp 4L (offset + s.End * int64 size)
            ElementSet.Range(lo, hi)

    /// True when two sets share an element.
    let overlaps (a: ElementSet) (b: ElementSet) =
        if a.IsEmpty || b.IsEmpty || a.End <= b.Start || b.End <= a.Start then
            false
        else
            let small, big = if a.Count <= b.Count then a, b else b, a
            small.Runs |> Seq.exists (fun struct (lo, hi) -> big.Overlaps(lo, hi))

/// The host memory of a CpuDevice heap: the table buffer (slot 0, 1 MiB) and one array per page, each allocated on first
/// use and filled with the uninitialized word. After Release every access raises ObjectDisposedException, so freed
/// memory is never quietly allocated again.
[<Sealed>]
type internal DeviceMemory(layout: HeapLayout, uninitialized: uint32) =
    let buffers: byte[][] = Array.zeroCreate Heap.SlotCount
    let mutable released = false

    let checkLive () =
        if released then
            raise (ObjectDisposedException "CpuDeviceBackend")

    let fresh (bytes: int64) =
        let b = GC.AllocateUninitializedArray<byte>(int bytes)
        MemoryMarshal.Cast<byte, uint32>(Span b).Fill uninitialized
        b

    /// The buffer of a slot: the table (slot 0) or a page of the layout.
    member _.Buffer(slot: int) : byte[] =
        checkLive ()

        match buffers[slot] with
        | null ->
            let size =
                if slot = Heap.TableSlot then
                    Heap.TableBytes
                else
                    (layout.Page slot).Bytes

            let b = fresh size
            buffers[slot] <- b
            b
        | b -> b

    /// The buffer of a slot when something has touched it, else null.
    member _.TryBuffer(slot: int) : byte[] =
        checkLive ()
        buffers[slot]

    /// The word an untouched buffer holds.
    member _.Uninitialized = uninitialized

    /// Frees every buffer.
    member _.Release() =
        released <- true
        Array.fill buffers 0 buffers.Length null

/// Host transfers and copies on top of the shared checks (Dispatch.checkTransfer, Dispatch.checkCopy), whose
/// `offset + bytes > size` wraps for an offset near Int64.MaxValue: once those checks pass (offset and size
/// non-negative), the end is compared as `offset > size - bytes`, which cannot overflow, so such a range raises
/// GpuContractException here instead of an ArgumentOutOfRangeException from the copy.
[<RequireQualifiedAccess>]
module internal Ranges =
    let transfer (layout: HeapLayout) (allowTable: bool) (slot: int) (offset: int64) (bytes: int64) =
        Dispatch.checkTransfer layout allowTable slot offset bytes

        let size =
            if slot = Heap.TableSlot then
                Heap.TableBytes
            else
                (layout.Page slot).Bytes

        if offset > size - bytes then
            Fail.contract
                "transfer in slot %d: %d bytes at offset %d are past the %d-byte buffer"
                slot
                bytes
                offset
                size

    let copy (layout: HeapLayout) (source: BufferRange) (destination: BufferRange) =
        Dispatch.checkCopy layout source destination

        for what, r in [ "source", source; "destination", destination ] do
            let page = (layout.Page r.Slot).Bytes

            if r.Offset > page - r.Bytes then
                Fail.contract "copy %s: %d bytes at offset %d are past the %d-byte page" what r.Bytes r.Offset page

[<RequireQualifiedAccess>]
module internal Fp32Model =
    let private isDenormal32 (bits: uint32) =
        bits &&& 0x7F800000u = 0u && bits &&& 0x007FFFFFu <> 0u

    let private isDenormal16 (bits: uint16) =
        bits &&& 0x7F80us = 0us && bits &&& 0x007Fus <> 0us

    /// A copy of a token's elements with every fp32 (or bf16) denormal replaced by zero of the same sign.
    let flushedCopy (v: TokenView) : TokenView =
        let size = DType.size v.DType

        let bytes =
            Array.zeroCreate<byte>(Heap.alignUp 4L (int64 v.Count * int64 size) |> int)

        Span<byte>(v.Bytes, v.Offset, v.Count * size).CopyTo(Span bytes)

        match v.DType with
        | DType.F32 ->
            let w = MemoryMarshal.Cast<byte, uint32>(Span bytes)

            for i in 0 .. v.Count - 1 do
                if isDenormal32 w[i] then
                    w[i] <- w[i] &&& 0x80000000u
        | DType.BF16 ->
            let h = MemoryMarshal.Cast<byte, uint16>(Span(bytes, 0, v.Count * 2))

            for i in 0 .. v.Count - 1 do
                if isDenormal16 h[i] then
                    h[i] <- h[i] &&& 0x8000us
        | _ -> ()

        {
            Bytes = bytes
            Offset = 0
            Count = v.Count
            DType = v.DType
        }

    /// The block with every f32 field's bits flushed.
    let flushArgs (k: KernelSpec) (args: ArgBlock) =
        let w = args.ToWords()

        k.Fields
        |> List.iteri (fun i f ->
            if f.Kind = FieldKind.F32 then
                w[i + 1] <- Fp32Denormals.flushBits w[i + 1])

        ArgBlock.OfWords w

    /// Applies `f` to every u32 word of an element set of an f32 token.
    let mapWords (v: TokenView) (s: ElementSet) (f: uint32 -> uint32) =
        let w = v.U32

        for struct (lo, hi) in s.Runs do
            for i in int lo .. int hi - 1 do
                w[i] <- f w[i]

    /// 0x7FFFFFFF for a NaN, the bits otherwise.
    let canonicalNaN (bits: uint32) =
        if bits &&& 0x7F800000u = 0x7F800000u && bits &&& 0x007FFFFFu <> 0u then
            0x7FFFFFFFu
        else
            bits

/// The CPU emulation of the kernel catalog behind the GpuBackendInterface (ConformanceHarness contract, "CpuDevice").
/// It holds every page in host memory, decodes the real 128-byte argument blocks and the real tensor rows from the
/// table image in slot 0, and runs one emulator per catalog kernel over the physical grid, bit-exact to CpuRef F32. It
/// enforces every rule of the GPU contracts with GpuContractException: the shared checks (Dispatch.prepare,
/// checkCopy, checkTransfer), the once-only rules, the kernel-specific minimum element counts and argument rules
/// (KernelRules), and barriers between dependent dispatches. Submit runs the commands in order on the calling thread
/// (and CpuDeviceOptions.Threads workers); Wait returns the completion and reports the timers.
[<Sealed>]
type CpuDeviceBackend(info: DeviceInfo, options: DeviceOptions, cpu: CpuDeviceOptions) =
    do
        if info.Backend <> ExecutionBackend.CpuDevice then
            invalidArg "info" (sprintf "a CpuDevice is backend cpu-device, not %s" (ExecutionBackend.name info.Backend))

        if cpu.Threads < 1 then
            invalidArg "cpu" (sprintf "%d threads" cpu.Threads)

        if cpu.Fp32Denormals = Fp32DenormalMode.Mixed then
            invalidArg "cpu" "a CpuDevice models preserve or flush; mixed has no single behaviour"

    let backend = ExecutionBackend.CpuDevice
    let clock = Stopwatch.StartNew()
    let mutable heap: CpuHeap option = None
    let mutable kernels: IKernelSet option = None
    let completions = Dictionary<uint64, Completion>()
    let trace = ResizeArray<ExecutedDispatch>()
    let mutable submitted = 0UL
    let mutable disposed = false

    let executionClass =
        let c = ExecutionClass.forCpu backend

        let suffix =
            (if cpu.Fp32Denormals = Fp32DenormalMode.Flush then
                 "+ftz"
             else
                 "")
            + (if cpu.CanonicalNaN then "+nan-canonical" else "")

        { c with Arch = c.Arch + suffix }

    let checkLive () =
        if disposed then
            raise (ObjectDisposedException "CpuDeviceBackend")

    /// Raises ObjectDisposedException after Dispose: every call of the backend, its heap, its kernel set and its
    /// command lists starts here (ConformanceHarness contract, "CpuDevice"), so a use after Dispose fails on CpuDevice
    /// as it would touch freed memory on a GPU.
    member internal _.CheckLive() = checkLive ()

    /// The version of the emulators, part of the kernel set's hash.
    static member EmulatorVersion = 1

    /// The default device: one CPU device, driver = the .NET runtime.
    static member DefaultDevice =
        {
            Backend = ExecutionBackend.CpuDevice
            Index = 0
            Name = "CPU device (emulated kernel catalog)"
            VendorId = 0u
            PciDeviceId = 0u
            Driver = RuntimeInformation.FrameworkDescription
            MemoryBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes
            Details = [ "kind", "emulator"; "arch", ExecutionClass.hostArch () ]
        }

    /// A CpuDevice on the default device with default options.
    new() =
        new CpuDeviceBackend(
            CpuDeviceBackend.DefaultDevice,
            DeviceOptions.For ExecutionBackend.CpuDevice,
            CpuDeviceOptions.Default
        )

    /// A CpuDevice on the default device.
    new(cpu: CpuDeviceOptions) =
        new CpuDeviceBackend(CpuDeviceBackend.DefaultDevice, DeviceOptions.For ExecutionBackend.CpuDevice, cpu)

    member _.Options = cpu

    /// With DeviceOptions.Debug: every dispatch run so far, in order (the debug layer of a CpuDevice).
    member _.Trace: ExecutedDispatch list = List.ofSeq trace

    member internal _.Backend = backend

    member internal _.Now =
        clock.Elapsed.Ticks * (1_000_000_000L / TimeSpan.TicksPerSecond)

    member internal _.HeapOrFail: CpuHeap =
        match heap with
        | Some h -> h
        | None -> Fail.contract "CreateHeap has not run"

    member internal _.KernelsOrFail =
        match kernels with
        | Some k -> k
        | None -> Fail.contract "LoadKernels has not run"

    /// Runs one dispatch: decodes the block and the rows from the heap's bytes, applies the arithmetic model, runs the
    /// emulator over the physical grid.
    member internal this.RunDispatch(d: PreparedDispatch, footprint: FieldFootprint list) =
        let h: CpuHeap = this.HeapOrFail
        let mem: DeviceMemory = h.Memory
        let k = d.Spec
        // The real block, as the root constants or push data carry it: 128 little-endian bytes.
        let args = ArgBlock.OfBytes(ReadOnlySpan(d.Args.ToBytes()))
        let table = mem.Buffer Heap.TableSlot

        if options.Debug then
            trace.Add
                {
                    Kernel = k.Name
                    Words = args.ToWords()
                    Grid = d.Physical
                }

        let rowOf (token: TensorToken) =
            if token.Value >= uint32 Heap.TableRows then
                Fail.contract "%s: token %d is outside the table" k.Name token.Value

            let row =
                TensorRow.OfBytes(ReadOnlySpan(table, int token.Value * Heap.RowBytes, Heap.RowBytes))

            row

        let views = Dictionary<string, TokenView>()
        let metas = Dictionary<string, TensorRow>()

        for u in d.Tokens do
            let row = rowOf u.Token

            if row <> u.Row then
                Fail.contract
                    "%s.%s: the table in slot 0 differs from the sealed table at token %d"
                    k.Name
                    u.Field
                    u.Token.Value

            if u.Access = Access.Meta then
                metas[u.Field] <- row
            else
                match TensorRow.validate h.Layout row with
                | Error e -> Fail.contract "%s.%s: the row of token %d is invalid: %s" k.Name u.Field u.Token.Value e
                | Ok() -> ()

                views[u.Field] <-
                    {
                        Bytes = mem.Buffer row.Slot
                        Offset = int row.ByteOffset
                        Count = int row.ElemCount
                        DType = row.DType.Value
                    }

        let arithmetic = Emulators.arithmetic.Contains k.Name
        let flush = arithmetic && cpu.Fp32Denormals = Fp32DenormalMode.Flush

        let args, readViews =
            if flush then
                let r = Dictionary<string, TokenView>(views)

                for f in footprint do
                    if Access.reads f.Access then
                        let v = views[f.Field]

                        if v.DType = DType.F32 || v.DType = DType.BF16 then
                            r[f.Field] <- Fp32Model.flushedCopy v

                Fp32Model.flushArgs k args, r
            else
                args, views

        let readAccess = footprint |> List.map (fun f -> f.Field, f.Access) |> Map.ofList

        let ctx =
            {
                Spec = k
                Args = args
                Grid = d.Physical
                View =
                    fun field ->
                        match readAccess.TryFind field with
                        | Some a when not (Access.writes a) -> readViews[field]
                        | _ -> views[field]
                MetaRow = fun field -> metas[field]
            }

        let emulate = Emulators.byName[k.Name]
        let p = d.Physical
        let total = int64 p.X * int64 p.Y * int64 p.Z

        let group (index: int64) =
            let gx = int (index % int64 p.X)
            let gy = int ((index / int64 p.X) % int64 p.Y)
            let gz = int (index / (int64 p.X * int64 p.Y))
            emulate ctx gx gy gz

        if cpu.Threads <= 1 || total <= 1L then
            for g in 0L .. total - 1L do
                group g
        else
            try
                Parallel.For(0L, total, ParallelOptions(MaxDegreeOfParallelism = cpu.Threads), (fun g -> group g))
                |> ignore
            with :? AggregateException as e when e.InnerExceptions.Count >= 1 ->
                // As on one thread: the first exception itself, not wrapped.
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw()

        // The arithmetic model on the results: flushed denormals, then canonical NaN.
        if arithmetic && (flush || cpu.CanonicalNaN) then
            for f in footprint do
                if Access.writes f.Access && views[f.Field].DType = DType.F32 then
                    let v = views[f.Field]

                    if flush then
                        Fp32Model.mapWords v f.Elements Fp32Denormals.flushBits

                    if cpu.CanonicalNaN then
                        Fp32Model.mapWords v f.Elements Fp32Model.canonicalNaN

    member internal this.Run(commands: CpuCommand list) : Submission =
        submitted <- submitted + 1UL
        let s = Submission submitted
        let opened = Dictionary<int, string * Attr list * int64>()
        let timers = SortedDictionary<int, TimerResult>()
        let dispatchTimers = ResizeArray<TimerResult>()
        let mutable dispatches = 0

        for c in commands do
            match c with
            | CpuCommand.Dispatch(d, fp) ->
                dispatches <- dispatches + 1
                let start = this.Now
                this.RunDispatch(d, fp)

                if options.TimeDispatches then
                    dispatchTimers.Add
                        {
                            Name = GpuTelemetry.Dispatch
                            Attrs = [ Attr.str TelemetryKeys.Kernel d.Spec.Name ]
                            GpuStartNs = start
                            DurationNs = max 1L (this.Now - start)
                        }
            | CpuCommand.Copy(src, dst) -> this.HeapOrFail.CopyNow(src, dst)
            | CpuCommand.Barrier -> ()
            | CpuCommand.BeginTimer(id, name, attrs) -> opened[id] <- (name, attrs, this.Now)
            | CpuCommand.EndTimer id ->
                let name, attrs, start = opened[id]

                timers[id] <-
                    {
                        Name = name
                        Attrs = attrs
                        GpuStartNs = start
                        DurationNs = this.Now - start
                    }

        GpuTelemetry.dispatches backend dispatches

        completions[s.Value] <-
            {
                Submission = s
                Dispatches = dispatches
                Timers = List.ofSeq timers.Values @ List.ofSeq dispatchTimers
            }

        s

    interface IGpuBackend with
        member _.Info =
            checkLive ()
            info

        member _.ExecutionClass =
            checkLive ()
            executionClass

        member this.CreateHeap(layout: HeapLayout) =
            checkLive ()

            if heap.IsSome then
                Fail.contract "CreateHeap runs once per backend"

            let h = new CpuHeap(this, layout, DeviceMemory(layout, cpu.UninitializedWord))
            heap <- Some h
            GpuTelemetry.heapCreated backend layout
            h :> IGpuHeap

        member _.LoadKernels(shaders: ShaderSet) =
            checkLive ()

            if kernels.IsSome then
                Fail.contract "LoadKernels runs once per backend"

            let catalogHash = Catalog.hash Kernels.catalog

            if shaders.CatalogHash <> catalogHash then
                Fail.contract
                    "the shader set was built for catalog %s, this build has %s"
                    shaders.CatalogHash.Short
                    catalogHash.Short

            let blobs = shaders.Blobs |> List.map _.Kernel |> Set.ofList

            let has (k: KernelId) =
                k.Value >= 0
                && k.Value < Kernels.all.Length
                && Emulators.byName.ContainsKey Kernels.all[k.Value].Name
                && (not cpu.MirrorShaderSet || blobs.Contains Kernels.all[k.Value].Name)

            let hash =
                Canonical.hash "lrm.cpudevice.kernels/1" (fun w ->
                    w.U32(uint32 CpuDeviceBackend.EmulatorVersion)
                    w.Id catalogHash
                    w.String(Fp32Denormals.name cpu.Fp32Denormals)
                    w.Bool cpu.CanonicalNaN)

            let set =
                { new IKernelSet with
                    member _.ShaderSetHash =
                        checkLive ()
                        hash

                    member _.Contains(k: KernelId) =
                        checkLive ()
                        has k
                }

            kernels <- Some set
            set

        member this.Begin() =
            checkLive ()
            CpuCommandList(this) :> ICommandList

        member this.Submit(commands: ICommandList) =
            checkLive ()

            match commands with
            | :? CpuCommandList as c when obj.ReferenceEquals(c.Owner, this) -> this.Run(c.Close())
            | _ -> Fail.contract "Submit takes a command list from this backend's Begin"

        member _.Wait(s: Submission) =
            checkLive ()

            use _ =
                Telemetry.span GpuTelemetry.Wait [ Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend) ]

            match completions.TryGetValue s.Value with
            | true, c ->
                GpuTelemetry.timers backend c.Timers
                c
            | _ -> Fail.contract "submission %d was not made by this backend" s.Value

        member _.Dispose() =
            if not disposed then
                disposed <- true
                heap |> Option.iter (fun (h: CpuHeap) -> h.Memory.Release())

/// The heap of a CpuDevice: the table and pages in host memory.
and [<Sealed>] internal CpuHeap(owner: CpuDeviceBackend, layout: HeapLayout, memory: DeviceMemory) =
    let mutable table: SealedTable option = None

    member _.Memory: DeviceMemory = memory
    member _.Layout: HeapLayout = layout

    member _.TableOrFail: SealedTable =
        match table with
        | Some t -> t
        | None -> Fail.contract "WriteTable has not run: the tensor table is written once, before the first dispatch"

    member _.CopyNow(src: BufferRange, dst: BufferRange) : unit =
        let s = memory.Buffer src.Slot
        let d = memory.Buffer dst.Slot
        Buffer.BlockCopy(s, int src.Offset, d, int dst.Offset, int src.Bytes)
        GpuTelemetry.bytesMoved owner.Backend "copy" src.Bytes

    interface IGpuHeap with
        member _.Layout =
            owner.CheckLive()
            layout

        member _.Table =
            owner.CheckLive()
            table

        member _.WriteTable(t: SealedTable) =
            owner.CheckLive()

            if table.IsSome then
                Fail.contract "the tensor table is written once"

            if not (obj.ReferenceEquals(t.Layout, layout) || t.Layout.Equals layout) then
                Fail.contract "the table was checked against a different layout"

            t.Image.Span.CopyTo(Span(memory.Buffer Heap.TableSlot))
            table <- Some t
            GpuTelemetry.bytesMoved owner.Backend "upload" Heap.TableBytes

        member _.Upload(slot: int, offset: int64, data: ReadOnlySpan<byte>) =
            owner.CheckLive()
            Ranges.transfer layout false slot offset (int64 data.Length)
            data.CopyTo(Span(memory.Buffer slot, int offset, data.Length))
            GpuTelemetry.bytesMoved owner.Backend "upload" (int64 data.Length)

        member _.UploadFrom(slot: int, offset: int64, source: Stream, length: int64) =
            owner.CheckLive()
            Ranges.transfer layout false slot offset length

            if isNull source then
                nullArg "source"

            let page = memory.Buffer slot
            let chunk = 64L <<< 20
            let mutable done' = 0L

            while done' < length do
                let n = int (min chunk (length - done'))
                source.ReadExactly(page, int (offset + done'), n)
                done' <- done' + int64 n

            GpuTelemetry.bytesMoved owner.Backend "upload" length

        member _.Download(slot: int, offset: int64, destination: Span<byte>) =
            owner.CheckLive()
            Ranges.transfer layout true slot offset (int64 destination.Length)

            match memory.TryBuffer slot with
            | null ->
                // Never touched: the uninitialized words, without allocating a page that may be 1 GiB.
                MemoryMarshal.Cast<byte, uint32>(destination).Fill memory.Uninitialized
            | b -> ReadOnlySpan(b, int offset, destination.Length).CopyTo destination

            GpuTelemetry.bytesMoved owner.Backend "download" (int64 destination.Length)

/// A command list of a CpuDevice. Every dispatch is checked when it is recorded: Dispatch.prepare, the kernel set, the
/// kernel's minimum element counts and argument rules, and (CheckBarriers) the barrier rule against every earlier
/// dispatch since the last Barrier or Copy.
and [<Sealed>] internal CpuCommandList(owner: CpuDeviceBackend) =
    let commands = ResizeArray<CpuCommand>()
    let openTimers = Stack<int>()
    let pending = ResizeArray<ByteAccess>()
    let mutable dispatches = 0
    let mutable nextTimer = 0
    let mutable closed = false

    let checkOpen () =
        owner.CheckLive()

        if closed then
            Fail.contract "the command list was already submitted"

    member _.Owner = owner

    member _.Close() =
        checkOpen ()

        if openTimers.Count > 0 then
            Fail.contract "%d timer(s) are still open at Submit" openTimers.Count

        closed <- true
        List.ofSeq commands

    member private _.CheckBarriers(k: KernelSpec, accesses: ByteAccess list) =
        for a in accesses do
            for p in pending do
                if p.Slot = a.Slot && (a.Writes || p.Writes) && ByteSets.overlaps a.Bytes p.Bytes then
                    let what =
                        match p.Writes, a.Writes with
                        | true, false -> "reads bytes that"
                        | false, true -> "writes bytes that"
                        | _ -> "writes bytes that"

                    Fail.contract
                        "%s.%s %s dispatch %d (%s.%s) %s in slot %d with no barrier between them: dependent dispatches need ICommandList.Barrier"
                        k.Name
                        a.Field
                        what
                        p.Index
                        p.Kernel
                        p.Field
                        (if p.Writes then "wrote" else "read")
                        a.Slot

        pending.AddRange accesses

    interface ICommandList with
        member this.Dispatch(kernel: KernelId, args: ArgBlock) =
            checkOpen ()
            let heap = owner.HeapOrFail
            let set = owner.KernelsOrFail
            let prepared = Dispatch.prepare heap.TableOrFail kernel args

            if not (set.Contains kernel) then
                Fail.contract "kernel %s is not in the loaded shader set" prepared.Spec.Name

            let byField = prepared.Tokens |> List.map (fun u -> u.Field, u) |> Map.ofList

            let footprint =
                KernelRules.check
                    prepared.Spec
                    prepared.Args
                    (fun f -> int64 byField[f].Row.ElemCount)
                    (fun f -> byField.TryFind f |> Option.bind _.Row.DType)
                    (fun f -> heap.TableOrFail.Name byField[f].Token)

            if owner.Options.CheckBarriers then
                let accesses =
                    footprint
                    |> List.filter (fun f -> not f.Elements.IsEmpty)
                    |> List.map (fun f ->
                        let u = byField[f.Field]

                        {
                            Index = dispatches
                            Kernel = prepared.Spec.Name
                            Field = f.Field
                            Slot = u.Slot
                            Bytes = ByteSets.ofElements u.Start (DType.size u.Row.DType.Value) f.Elements
                            Writes = Access.writes f.Access
                        })

                this.CheckBarriers(prepared.Spec, accesses)

            dispatches <- dispatches + 1
            commands.Add(CpuCommand.Dispatch(prepared, footprint))

        member _.Barrier() =
            checkOpen ()
            pending.Clear()
            commands.Add CpuCommand.Barrier

        member _.Copy(source: BufferRange, destination: BufferRange) =
            checkOpen ()
            Ranges.copy owner.HeapOrFail.Layout source destination
            // Copy is a full barrier on both sides.
            pending.Clear()
            commands.Add(CpuCommand.Copy(source, destination))

        member _.BeginTimer(name: string, attrs: Attr list) =
            checkOpen ()

            if not (TelemetryNames.isValid name) then
                Fail.contract "timer name '%s' is not a valid telemetry name" name

            let id = nextTimer
            nextTimer <- nextTimer + 1
            openTimers.Push id
            commands.Add(CpuCommand.BeginTimer(id, name, attrs))
            TimerId id

        member _.EndTimer(TimerId id) =
            checkOpen ()

            if openTimers.Count = 0 || openTimers.Peek() <> id then
                Fail.contract "timer %d is not the most recently opened timer" id

            openTimers.Pop() |> ignore
            commands.Add(CpuCommand.EndTimer id)

/// The CpuDevice provider and helpers (GpuBackendInterface: apps and tests register providers; libraries never do).
[<RequireQualifiedAccess>]
module CpuDevice =
    /// A provider of one CpuDevice with the given options.
    let providerWith (cpu: CpuDeviceOptions) =
        { new IGpuBackendProvider with
            member _.Backend = ExecutionBackend.CpuDevice

            member _.Enumerate() =
                [
                    {
                        Info = CpuDeviceBackend.DefaultDevice
                        Rejections = []
                    }
                ]

            member _.Create(device, o) =
                new CpuDeviceBackend(device, o, cpu) :> IGpuBackend
        }

    /// The provider of the default CpuDevice.
    let provider = providerWith CpuDeviceOptions.Default

    /// A new CpuDevice backend.
    let create (cpu: CpuDeviceOptions) (options: DeviceOptions) =
        new CpuDeviceBackend(CpuDeviceBackend.DefaultDevice, options, cpu) :> IGpuBackend
