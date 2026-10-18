namespace Lrm.Gpu.D3D12

open System
open System.Collections.Generic
open System.IO
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12.Interop

/// One submitted command list: its encoding, its fence value and, once it has completed, its Completion.
type internal SubmissionRecord =
    {
        Number: uint64
        Encoded: Encoded
        mutable Fence: uint64
        mutable Completion: Completion option
        /// True once Wait has emitted the timers as telemetry (once per submission).
        mutable Emitted: bool
    }

/// The d3d12 backend (GpuBackendInterface) on in-box Direct3D 12 at shader model 6.6: a COMPUTE queue and fence,
/// committed pages with per-page state tracking, one shader-visible heap of 64 raw UAVs written once, a pipeline per
/// DXIL blob on the canonical root signature, timestamp queries for timers and dispatches, GetDeviceRemovedReason
/// after every submission, and DRED breadcrumbs when the runtime offers them. Right after LoadKernels it runs the fp32
/// denormal probe and records gpu.fp32_denormals. Calls come from one thread at a time.
[<Sealed>]
type D3D12Backend private (info: DeviceInfo, options: DeviceOptions, ctx: DeviceContext, debugLayer: bool) =
    let backend = ExecutionBackend.D3D12
    let backendAttr = Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend)
    let mutable heap: D3D12Heap option = None
    let mutable kernels: D3D12Kernels option = None
    let mutable klass: ExecutionClass option = None
    let mutable denormals: (Fp32DenormalMode * uint32[]) option = None
    let records = Dictionary<uint64, SubmissionRecord>()
    let mutable submitted = 0UL
    let mutable probeRuns = 0
    let mutable disposed = false

    let checkLive () =
        if disposed then
            raise (ObjectDisposedException "D3D12Backend")

    let result (ticks: uint64[]) (s: TimerSlot) : TimerResult =
        let start = ctx.TicksToNs ticks[s.Begin]

        {
            Name = s.Name
            Attrs = s.Attrs
            GpuStartNs = start
            DurationNs = max 0L (ctx.TicksToNs ticks[s.End] - start)
        }

    /// Reads a completed submission's timestamps into its Completion (run when its frame retires).
    let materialize (r: SubmissionRecord) (frame: Frame) =
        let ticks = frame.ReadStamps r.Encoded.Queries

        r.Completion <-
            Some
                {
                    Submission = Submission r.Number
                    Dispatches = r.Encoded.Dispatches
                    Timers = (r.Encoded.Timers @ r.Encoded.DispatchTimers) |> List.map (result ticks)
                }

    /// Devices this process created through the backend. The debug layer may only be enabled before the first one:
    /// enabling it while a device exists removes that device.
    static let mutable devicesCreated = 0

    /// How long a wait may block before the backend treats the device as lost.
    static member WaitTimeout = TimeSpan.FromSeconds 30.0

    /// Creates a backend on an accepted adapter (the provider's Create). DRED auto-breadcrumbs and page-fault
    /// reporting are forced on for the device when the runtime offers them; DeviceOptions.Debug enables the debug
    /// layer when it is installed.
    static member Create(device: DeviceInfo, options: DeviceOptions) : D3D12Backend =
        if device.Backend <> ExecutionBackend.D3D12 then
            invalidArg "device" (sprintf "%s is not a d3d12 device" (ExecutionBackend.name device.Backend))

        use factory = DxgiApi.createFactory<IDXGIFactory6> 0u

        use adapter: ComPtr<IDXGIAdapter1> =
            factory.EnumAdapterByGpuPreference(uint32 device.Index, DXGI_GPU_PREFERENCE.HIGH_PERFORMANCE)

        let desc = adapter.GetDesc1()

        let luid =
            (int64 desc.AdapterLuid.HighPart <<< 32) ||| int64 desc.AdapterLuid.LowPart

        match Adapters.luidOf device with
        | Some expected when expected <> luid ->
            invalidOp (
                sprintf
                    "adapter %d is LUID %s, not the enumerated %s (%s); enumerate again"
                    device.Index
                    (Adapters.luidText luid)
                    (Adapters.luidText expected)
                    device.Name
            )
        | _ -> ()

        let debugLayer =
            options.Debug
            && devicesCreated = 0
            && (match D3D12Api.tryGetDebugInterface<ID3D12Debug>() with
                | ValueSome d ->
                    use d = d
                    d.EnableDebugLayer()
                    true
                | ValueNone -> false)

        let dev, dred =
            Dred.withEnabled (fun () ->
                D3D12Api.createDevice<IDXGIAdapter1, ID3D12Device> adapter DeviceCheck.CreationFeatureLevel)

        devicesCreated <- devicesCreated + 1

        let ctx =
            try
                new DeviceContext(dev, dred, D3D12Backend.WaitTimeout)
            with _ ->
                let mutable d = dev
                ComPtr.release &d
                reraise ()

        new D3D12Backend(device, options, ctx, debugLayer)

    /// The fp32 denormal mode the probe measured right after LoadKernels (GpuBackendInterface), and the probe's
    /// seven result words. None before LoadKernels, or when the shader set has no scale_f32.
    member _.Fp32Denormals: Fp32DenormalMode option = denormals |> Option.map fst

    /// The probe's result words (Fp32Denormals.probeInputs scaled by 0.5 on this device).
    member _.ProbeResults: uint32[] option = denormals |> Option.map snd

    /// True when DRED was forced on for this backend's device.
    member _.DredEnabled = ctx.DredEnabled

    /// True when DeviceOptions.Debug asked for the debug layer and it is installed.
    member _.DebugLayer = debugLayer

    member _.Options = options

    /// Descriptors the heap wrote when it was created (64; 0 before CreateHeap).
    member _.DescriptorWrites =
        match heap with
        | Some h -> h.DescriptorWrites
        | None -> 0

    /// Command allocators and lists created so far. It stays flat in a soak once the in-flight depth stops growing.
    member _.FramesCreated = ctx.FramesCreated

    /// The queue's timestamp frequency (ticks per second).
    member _.TimestampFrequency = ctx.TimestampFrequency

    /// GetDeviceRemovedReason: S_OK (0) while the device works.
    member _.DeviceRemovedReason =
        let device = ctx.Device
        device.GetDeviceRemovedReason()

    /// The resource behind a heap slot, borrowed (never release it): ComPtr.Null for unused slots and before
    /// CreateHeap. For diagnostics and tests that check the heap's resources directly.
    member _.SlotResource(slot: int) =
        match heap with
        | Some h when slot >= 0 && slot < Heap.SlotCount -> h.Resource slot
        | _ -> ComPtr<ID3D12Resource>.Null

    /// DRED's report for this backend's device now (breadcrumbs and page faults are available after a removal).
    member _.Dred() : DredReport = Dred.report ctx.DredEnabled ctx.Device

    member internal _.HeapOrFail: D3D12Heap =
        match heap with
        | Some h -> h
        | None -> Contract.fail "CreateHeap has not run"

    member internal _.KernelsOrFail: D3D12Kernels =
        match kernels with
        | Some k -> k
        | None -> Contract.fail "LoadKernels has not run"

    member internal _.CheckLive() = checkLive ()

    member internal _.Backend = backend

    /// The fp32 denormal probe on a private heap (slot 0 a table with tokens x and y, slot 1 a 64 KiB page): scale_f32
    /// with a = Fp32Denormals.ProbeScale over Fp32Denormals.probeInputs, the outputs poisoned first. Records
    /// gpu.fp32_denormals with the words and the dispatch's gpu.dispatch interval (case fp32_denormal_probe).
    member private _.RunProbe(k: D3D12Kernels) : (Fp32DenormalMode * uint32[]) option =
        if not (k.Contains Kernels.ScaleF32.Id) then
            None
        else
            probeRuns <- probeRuns + 1
            let run = probeRuns
            use _ = Telemetry.scope [ Attr.str TelemetryKeys.Case "fp32_denormal_probe" ]
            let layout = HeapLayout.ofPages [ ArenaKind.Scratch, Heap.PageGranularity ]
            let h = new D3D12Heap(ctx, layout, "lrm.d3d12.probe")

            try
                let inputs = Fp32Denormals.probeInputs
                let n = inputs.Length
                let yOffset = 256
                let table = TensorTable layout

                let row name offset flags =
                    TensorRow.create 1 offset DType.F32 [ int64 n ] flags (TensorRow.debugIdOf name)

                let x = table.Add("probe.x", row "probe.x" 0L TensorFlags.ReadOnly)
                let y = table.Add("probe.y", row "probe.y" (int64 yOffset) TensorFlags.None)
                let sealedTable = table.Seal()
                h.WriteSealed sealedTable
                let poison i = 0x7FA00000u + uint32 (run * 16 + i)
                let page = Array.zeroCreate<uint32>(yOffset / 4 + n)

                for i in 0 .. n - 1 do
                    page[i] <- inputs[i]
                    page[yOffset / 4 + i] <- poison i

                let pageBytes = MemoryMarshal.AsBytes(ReadOnlySpan page)
                h.WriteSpan(1, 0L, pageBytes)

                let prepared =
                    Dispatch.prepare
                        sealedTable
                        Kernels.ScaleF32.Id
                        (Kernels.ScaleF32.encode
                            {
                                X = x
                                Y = y
                                N = uint32 n
                                A = Fp32Denormals.ProbeScale
                            })

                let enc = Plan.encode true [ Command.Dispatch prepared ]
                let stamps = ref [||]

                let frame =
                    ctx.Execute
                        (Some(h.Bindings(k.RootSignature, k.Pipeline)))
                        enc.Queries
                        enc.Ops
                        (Some(fun f -> stamps.Value <- f.ReadStamps enc.Queries))

                ctx.WaitFor frame.FenceValue
                let out = Array.zeroCreate<uint32> n
                h.ReadSpan(1, int64 yOffset, MemoryMarshal.AsBytes(Span out))
                GpuTelemetry.bytesMoved backend "upload" (Heap.TableBytes + int64 pageBytes.Length)
                GpuTelemetry.bytesMoved backend "download" (int64 (4 * n))

                for i in 0 .. n - 1 do
                    if out[i] = poison i then
                        invalidOp (
                            sprintf
                                "the fp32 denormal probe's word %d still holds its poison: scale_f32 wrote nothing"
                                i
                        )

                let mode = Fp32Denormals.recordProbe backend out

                for t in enc.DispatchTimers do
                    let r = result stamps.Value t
                    Telemetry.gpuInterval r.Name r.GpuStartNs r.DurationNs (backendAttr :: r.Attrs)

                Some(mode, out)
            finally
                h.Dispose()

    member internal x.SubmitCommands(commands: Command list) : Submission =
        let enc = Plan.encode options.TimeDispatches commands

        if options.Debug then
            let slots =
                commands
                |> List.choose (function
                    | Command.Dispatch d -> Some(d.Tokens |> List.map _.Slot)
                    | _ -> None)

            match Plan.verify slots enc.Ops with
            | Ok() -> ()
            | Error e -> invalidOp ("d3d12: the encoder broke a state rule: " + e)

        let usesScratch =
            enc.Ops
            |> List.exists (function
                | NativeOp.Copy(Target.Scratch, _, _, _, _)
                | NativeOp.Copy(_, _, Target.Scratch, _, _) -> true
                | _ -> false)

        if usesScratch then
            ctx.EnsureScratch()

        let bindings =
            heap
            |> Option.map (fun h ->
                match kernels with
                | Some k -> h.Bindings(k.RootSignature, k.Pipeline)
                | None -> h.Bindings(ComPtr<ID3D12RootSignature>.Null, fun _ -> invalidOp "LoadKernels has not run"))

        let r =
            {
                Number = submitted + 1UL
                Encoded = enc
                Fence = 0UL
                Completion = None
                Emitted = false
            }

        let frame = ctx.Execute bindings enc.Queries enc.Ops (Some(materialize r))
        r.Fence <- frame.FenceValue
        submitted <- r.Number
        records[r.Number] <- r
        GpuTelemetry.dispatches backend enc.Dispatches

        for bytes in enc.Copies do
            GpuTelemetry.bytesMoved backend "copy" bytes

        Submission r.Number

    interface IGpuBackend with
        member _.Info = info

        member _.ExecutionClass =
            checkLive ()

            match klass with
            | Some c -> c
            | None -> Contract.fail "the execution class of a shader backend is known after LoadKernels"

        member this.CreateHeap(layout: HeapLayout) =
            checkLive ()

            if heap.IsSome then
                Contract.fail "CreateHeap runs once per backend"

            let h = new D3D12Heap(ctx, layout, "lrm.d3d12.heap")
            heap <- Some h
            GpuTelemetry.heapCreated backend layout
            new D3D12HeapView(this, h) :> IGpuHeap

        member this.LoadKernels(shaders: ShaderSet) =
            checkLive ()

            if kernels.IsSome then
                Contract.fail "LoadKernels runs once per backend"

            if shaders.CatalogHash <> Catalog.hash Kernels.catalog then
                Contract.fail
                    "the shader set was built for catalog %s, this build has %s"
                    shaders.CatalogHash.Short
                    (Catalog.hash Kernels.catalog).Short

            let k =
                use span = Telemetry.span "gpu.load_kernels" [ backendAttr ]
                let k = new D3D12Kernels(ctx, shaders)
                span.Add(Attr.int "kernels" k.Count)
                k

            kernels <- Some k
            klass <- Some(GpuClass.fill info shaders)
            denormals <- this.RunProbe k
            k :> IKernelSet

        member this.Begin() =
            checkLive ()
            new D3D12CommandList(this) :> ICommandList

        member this.Submit(commands: ICommandList) =
            checkLive ()

            match commands with
            | :? D3D12CommandList as c when obj.ReferenceEquals(c.Owner, this) -> this.SubmitCommands(c.Close())
            | _ -> Contract.fail "Submit takes a command list from this backend's Begin"

        member _.Wait(s: Submission) =
            checkLive ()

            match records.TryGetValue s.Value with
            | true, r ->
                use _ = Telemetry.span GpuTelemetry.Wait [ backendAttr ]

                if r.Completion.IsNone then
                    ctx.WaitFor r.Fence

                match r.Completion with
                | Some c ->
                    if not r.Emitted then
                        r.Emitted <- true
                        GpuTelemetry.timers backend c.Timers

                    c
                | None -> invalidOp (sprintf "submission %d completed without its completion record" s.Value)
            | _ -> Contract.fail "submission %d was not made by this backend" s.Value

        member _.Dispose() =
            if not disposed then
                disposed <- true

                try
                    try
                        ctx.WaitIdle()
                    with _ ->
                        ()
                finally
                    kernels |> Option.iter _.Dispose()
                    heap |> Option.iter _.Dispose()
                    kernels <- None
                    heap <- None
                    ctx.Dispose()

/// The IGpuHeap of a D3D12Backend: the shared transfer rules (Dispatch.checkTransfer), then the staged copies, then
/// gpu.bytes_moved.
and [<Sealed>] internal D3D12HeapView(owner: D3D12Backend, h: D3D12Heap) =
    let backendAttr =
        Attr.str TelemetryKeys.Backend (ExecutionBackend.name owner.Backend)

    let meter (direction: string) (bytes: int64) =
        GpuTelemetry.bytesMoved owner.Backend direction bytes

    /// A gpu.transfer span: the host time of one transfer, waits for earlier work included.
    let span (direction: string) (slot: int) (bytes: int64) =
        Telemetry.span
            "gpu.transfer"
            [
                backendAttr
                Attr.str "direction" direction
                Attr.int "slot" slot
                Attr.int64 "bytes" bytes
            ]

    interface IGpuHeap with
        member _.Layout = h.Layout
        member _.Table = h.Table

        member _.WriteTable(t: SealedTable) =
            owner.CheckLive()
            use _ = span "upload" Heap.TableSlot Heap.TableBytes
            h.WriteSealed t
            meter "upload" Heap.TableBytes

        member _.Upload(slot: int, offset: int64, data: ReadOnlySpan<byte>) =
            owner.CheckLive()
            Dispatch.checkTransfer h.Layout false slot offset (int64 data.Length)
            use _ = span "upload" slot (int64 data.Length)
            h.WriteSpan(slot, offset, data)
            meter "upload" (int64 data.Length)

        member _.UploadFrom(slot: int, offset: int64, source: Stream, length: int64) =
            owner.CheckLive()
            Dispatch.checkTransfer h.Layout false slot offset length
            use _ = span "upload" slot length
            h.WriteStream(slot, offset, source, length)
            meter "upload" length

        member _.Download(slot: int, offset: int64, destination: Span<byte>) =
            owner.CheckLive()
            Dispatch.checkTransfer h.Layout true slot offset (int64 destination.Length)
            use _ = span "download" slot (int64 destination.Length)
            h.ReadSpan(slot, offset, destination)
            meter "download" (int64 destination.Length)

/// A command list of a D3D12Backend: commands are checked when recorded (the same checks, in the same order, with the
/// same texts as RecordingBackend) and encoded at Submit.
and [<Sealed>] D3D12CommandList internal (owner: D3D12Backend) =
    let commands = ResizeArray<Command>()
    let openTimers = Stack<int>()
    let mutable nextTimer = 0
    let mutable closed = false

    let checkOpen () =
        if closed then
            Contract.fail "the command list was already submitted"

    member internal _.Owner = owner

    /// Commands recorded so far (for tests of the encoder against a real list).
    member internal _.Commands = List.ofSeq commands

    member internal _.Close() =
        checkOpen ()

        if openTimers.Count > 0 then
            Contract.fail "%d timer(s) are still open at Submit" openTimers.Count

        closed <- true
        List.ofSeq commands

    interface ICommandList with
        member _.Dispatch(kernel: KernelId, args: ArgBlock) =
            checkOpen ()
            let heap = owner.HeapOrFail
            let set = owner.KernelsOrFail
            let prepared = Dispatch.prepare heap.TableOrFail kernel args

            if not (set.Contains kernel) then
                Contract.fail "kernel %s is not in the loaded shader set" prepared.Spec.Name

            commands.Add(Command.Dispatch prepared)

        member _.Barrier() =
            checkOpen ()
            commands.Add Command.Barrier

        member _.Copy(source: BufferRange, destination: BufferRange) =
            checkOpen ()
            Dispatch.checkCopy owner.HeapOrFail.Layout source destination
            commands.Add(Command.Copy(source, destination))

        member _.BeginTimer(name: string, attrs: Attr list) =
            checkOpen ()

            if not (TelemetryNames.isValid name) then
                Contract.fail "timer name '%s' is not a valid telemetry name" name

            let id = nextTimer
            nextTimer <- nextTimer + 1
            openTimers.Push id
            commands.Add(Command.BeginTimer(id, name, attrs))
            TimerId id

        member _.EndTimer(TimerId id) =
            checkOpen ()

            if openTimers.Count = 0 || openTimers.Peek() <> id then
                Contract.fail "timer %d is not the most recently opened timer" id

            openTimers.Pop() |> ignore
            commands.Add(Command.EndTimer id)

/// The d3d12 backend's provider (GpuBackendInterface: each backend project exposes one; apps and tests register it).
[<Sealed>]
type D3D12Provider() =
    interface IGpuBackendProvider with
        member _.Backend = ExecutionBackend.D3D12
        member _.Enumerate() = Adapters.enumerate ()

        member _.Create(device: DeviceInfo, options: DeviceOptions) =
            D3D12Backend.Create(device, options) :> IGpuBackend

/// The d3d12 backend's entry points.
[<RequireQualifiedAccess>]
module D3D12 =
    /// The provider to register with Backend.register (apps and tests do; libraries never do).
    let provider: IGpuBackendProvider = D3D12Provider()

    /// A backend on the first accepted adapter (or `options.Index`), through a private registry holding only this
    /// provider.
    let tryCreate (options: DeviceOptions) : Result<IGpuBackend, string> =
        let registry = BackendRegistry()
        registry.Register provider

        registry.TryCreate
            { options with
                Backend = ExecutionBackend.D3D12
            }
