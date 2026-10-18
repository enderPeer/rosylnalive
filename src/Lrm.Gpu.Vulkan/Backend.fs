namespace Lrm.Gpu.Vulkan

open System
open System.Collections.Generic
open System.IO
open System.Runtime.InteropServices
open Lrm.Gpu.Vulkan.Interop
open Lrm.Core
open Lrm.Gpu.Core

/// A submitted command list until its frame retires: its timeline value and what reading its timestamps needs. The
/// native ops are not kept: they were recorded into the frame's command buffer at Submit.
type internal PendingSubmission =
    {
        Value: uint64
        Queries: int
        Timers: TimerSlot list
        DispatchTimers: TimerSlot list
    }

/// The timer results of a retired submission that had timers, kept so that every Wait returns the same Completion
/// (GpuBackendInterface): the timers as TimerResults, and the gpu.dispatch timings as columns (the kernel's shared
/// attribute list, start and duration: 24 bytes per timed dispatch), from which Results rebuilds an equal list.
[<Sealed>]
type internal RetiredTimers(timers: TimerResult list, dispatchAttrs: Attr list[], starts: int64[], durations: int64[]) =
    /// True once Wait has emitted the timers as telemetry (once per submission).
    member val Emitted = false with get, set

    /// Timers in the order they were opened, then the gpu.dispatch timings (Completion.Timers).
    member _.Results: TimerResult list =
        if starts.Length = 0 then
            timers
        else
            timers
            @ List.init starts.Length (fun i ->
                {
                    Name = GpuTelemetry.Dispatch
                    Attrs = dispatchAttrs[i]
                    GpuStartNs = starts[i]
                    DurationNs = durations[i]
                })

/// The vulkan backend (GpuBackendInterface) on Vulkan 1.4 with VK_EXT_descriptor_heap, following VulkanHeapRecipe v1.1:
/// the recipe's device on its compute-only queue family, a timeline semaphore per backend, DEVICE_LOCAL pages of at
/// most 1 GiB, one resource heap written once, a heap pipeline per SPIR-V blob with a NULL layout, the 128-byte ArgBlock
/// as push data, synchronization2 barriers, ALL_COMMANDS timestamps for timers and dispatches, VK_EXT_device_fault
/// diagnostics on device loss and VK_EXT_debug_utils object names. Right after LoadKernels it runs the fp32 denormal
/// probe and records gpu.fp32_denormals. Calls come from one thread at a time.
[<Sealed>]
type VulkanBackend
    private (info: DeviceInfo, options: DeviceOptions, ctx: DeviceContext, placement: HeapPlacement, validation: bool) =
    let backend = ExecutionBackend.Vulkan
    let backendAttr = Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend)

    let limits =
        match ctx.Capabilities.DescriptorHeap with
        | Some h -> PhysicalDevices.heapLimits h
        | None -> invalidOp "vulkan: the device has no VK_EXT_descriptor_heap limits"

    /// The heap geometry of the queried limits (Heap.vulkanGeometry, VulkanHeapRecipe section 3).
    let geometry = Heap.vulkanGeometry limits
    let mutable heap: VulkanHeap option = None
    let mutable kernels: VulkanKernels option = None
    let mutable klass: ExecutionClass option = None
    let mutable denormals: (Fp32DenormalMode * uint32[]) option = None
    // What Wait needs, and nothing more (a long run makes millions of submissions): the dispatch count of every
    // submission (index number - 1, 4 bytes each); the timeline value and timer slots of each one until its frame
    // retires; and from then on the timer results of the submissions that have timers. GpuBackendInterface keeps every
    // completion waitable, so those last two stay for the backend's lifetime.
    let dispatchCounts = ResizeArray<int>()
    let pending = Dictionary<uint64, PendingSubmission>()
    let retiredTimers = Dictionary<uint64, RetiredTimers>()
    let mutable submitted = 0UL
    let mutable probeRuns = 0
    let mutable probeDescriptorWrites = 0
    let mutable disposed = false

    let checkLive () =
        if disposed then
            raise (ObjectDisposedException "VulkanBackend")

    let result (ticks: uint64[]) (s: TimerSlot) : TimerResult =
        {
            Name = s.Name
            Attrs = s.Attrs
            GpuStartNs = ctx.TicksToNs ticks[s.Begin]
            DurationNs = max 0L (ctx.TicksToNs(ticks[s.End] - ticks[s.Begin]))
        }

    /// Runs when submission `number`'s frame retires: drops its pending entry and keeps its timer results, if it has
    /// timers. A frame whose Submit raised after vkQueueSubmit2 left no entry, or one with another value, and is ignored.
    let retire (number: uint64) (frame: Frame) =
        match pending.TryGetValue number with
        | true, p when p.Value = frame.Value ->
            pending.Remove number |> ignore

            if p.Queries > 0 then
                let ticks = frame.ReadStamps(ctx.Commands, ctx.Handle, p.Queries, ctx.TimestampMask)
                let dispatches = Array.ofList p.DispatchTimers

                retiredTimers[number] <-
                    RetiredTimers(
                        p.Timers |> List.map (result ticks),
                        dispatches |> Array.map _.Attrs,
                        dispatches |> Array.map (fun s -> ctx.TicksToNs ticks[s.Begin]),
                        dispatches
                        |> Array.map (fun s -> max 0L (ctx.TicksToNs(ticks[s.End] - ticks[s.Begin])))
                    )
        | _ -> ()

    /// How long a wait may block before the backend treats the device as lost.
    static member WaitTimeout = TimeSpan.FromSeconds 30.0

    /// The validation layer DeviceOptions.Debug enables when it is installed (AP-3; not installed on the development
    /// machine).
    static member ValidationLayer = "VK_LAYER_KHRONOS_validation"

    /// Creates a backend on an accepted device (the provider's Create), with the heap in `placement`.
    static member Create(device: DeviceInfo, options: DeviceOptions, placement: HeapPlacement) : VulkanBackend =
        if device.Backend <> ExecutionBackend.Vulkan then
            invalidArg "device" (sprintf "%s is not a vulkan device" (ExecutionBackend.name device.Backend))

        let loader =
            match PhysicalDevices.loader () with
            | Some l -> l
            | None -> raise (DllNotFoundException "vulkan: no Vulkan loader (vulkan-1.dll) is installed")

        let debugUtils =
            (loader.InstanceExtensions()).ContainsKey VkConst.VK_EXT_DEBUG_UTILS_EXTENSION_NAME

        let validation =
            options.Debug
            && (loader.InstanceLayers()).ContainsKey VulkanBackend.ValidationLayer

        let instance =
            loader.CreateInstance
                { InstanceOptions.create "lrm" debugUtils with
                    Layers = if validation then [ VulkanBackend.ValidationLayer ] else []
                }

        VulkanObjects.created ()

        try
            let physicals = instance.PhysicalDevices()

            if device.Index < 0 || device.Index >= physicals.Length then
                invalidOp (sprintf "device %d is not there (%d devices); enumerate again" device.Index physicals.Length)

            let physical = physicals[device.Index]
            let caps = DeviceCheck.observe instance physical

            match PhysicalDevices.uuidOf device with
            | Some expected when expected <> caps.Identity.DeviceUuid ->
                invalidOp (
                    sprintf
                        "device %d is %s (uuid %s), not the enumerated %s (uuid %s); enumerate again"
                        device.Index
                        caps.Identity.Name
                        caps.Identity.DeviceUuid
                        device.Name
                        expected
                )
            | _ -> ()

            match DeviceCheck.reasons caps with
            | [] -> ()
            | rs -> invalidOp (sprintf "vulkan device %d is rejected: %s" device.Index (String.concat "; " rs))

            let messenger =
                if validation && debugUtils then
                    Some(
                        new DebugMessenger(
                            instance,
                            VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                            ||| VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                            ignore
                        )
                    )
                else
                    None

            let dev =
                Lrm.Gpu.Vulkan.Interop.Device.create instance physical Lrm.Gpu.Vulkan.Interop.DeviceOptions.diagnostics

            VulkanObjects.created ()

            let ctx =
                try
                    new DeviceContext(instance, dev, caps, messenger, VulkanBackend.WaitTimeout)
                with _ ->
                    (dev :> IDisposable).Dispose()
                    VulkanObjects.destroyed ()
                    messenger |> Option.iter (fun m -> (m :> IDisposable).Dispose())
                    reraise ()

            new VulkanBackend(device, options, ctx, placement, validation)
        with _ ->
            (instance :> IDisposable).Dispose()
            VulkanObjects.destroyed ()
            reraise ()

    /// Creates a backend on an accepted device (the provider's Create), with the recipe's heap placement.
    static member Create(device: DeviceInfo, options: DeviceOptions) : VulkanBackend =
        VulkanBackend.Create(device, options, HeapPlacement.InPlace)

    /// The fp32 denormal mode the probe measured right after LoadKernels (GpuBackendInterface), or None before
    /// LoadKernels or when the shader set has no scale_f32.
    member _.Fp32Denormals: Fp32DenormalMode option = denormals |> Option.map fst

    /// The probe's result words (Fp32Denormals.probeInputs scaled by 0.5 on this device).
    member _.ProbeResults: uint32[] option = denormals |> Option.map snd

    member _.Options = options

    /// True when DeviceOptions.Debug asked for the validation layer and it is installed.
    member _.ValidationEnabled = validation

    /// True when VK_EXT_debug_utils is enabled, so every object carries a debug name.
    member _.DebugNames = ctx.DebugUtils

    /// True when VK_EXT_device_fault is enabled on the device (its report goes into VulkanDeviceLostException).
    member _.DeviceFaultEnabled =
        List.contains VkConst.VK_EXT_DEVICE_FAULT_EXTENSION_NAME ctx.Device.EnabledExtensions

    /// The device extensions enabled: VK_EXT_descriptor_heap, plus VK_EXT_device_fault and VK_EXT_memory_budget when
    /// supported.
    member _.EnabledExtensions = ctx.Device.EnabledExtensions

    /// The recipe's queue family (compute, no graphics, timestamps).
    member _.QueueFamily = ctx.Device.QueueFamily

    /// The heap geometry of the queried limits.
    member _.Geometry = geometry

    /// Where the heap lives once CreateHeap has run (InPlace, or Copied on a device without a BAR memory type).
    member _.HeapPlacement =
        match heap with
        | Some h -> h.Placement
        | None -> placement

    /// The memory type chosen for each kind of buffer created so far (by flags): (kind, index, flags).
    member _.MemoryTypeChoices: (BufferKind * int * string) list =
        ctx.MemoryChoices
        |> List.map (fun (k, i) -> k, i, MemoryTypes.flagsText ctx.MemoryTypes[i])

    /// Descriptors the resource heap wrote when CreateHeap created it (64; 0 before CreateHeap), and never again.
    member _.HeapDescriptorWrites =
        match heap with
        | Some h -> h.DescriptorWrites
        | None -> 0

    /// Descriptors the fp32 denormal probe's transient heap wrote at LoadKernels (64; 0 before LoadKernels or without
    /// scale_f32).
    member _.ProbeDescriptorWrites = probeDescriptorWrites

    /// Every descriptor this backend has written, on every resource heap it created.
    member x.DescriptorWrites = x.HeapDescriptorWrites + probeDescriptorWrites

    /// The byte ranges of the resource heap that its descriptor writes targeted, as (offset from heapRange.address,
    /// bytes); empty before CreateHeap.
    member _.DescriptorDestinations =
        match heap with
        | Some h -> List.ofArray h.DescriptorDestinations
        | None -> []

    /// heapRange.address, heapRange.size, reservedRangeOffset and reservedRangeSize of the resource heap.
    member _.HeapRange: (uint64 * uint64 * uint64 * uint64) option =
        heap
        |> Option.map (fun h -> h.Range.Address, h.Range.Bytes, h.Range.ReservedOffset, h.Range.ReservedBytes)

    /// The slot area's bytes of an in-place resource heap (the 64 descriptors), read through its mapping; the reserved
    /// range is never read.
    member _.SlotAreaBytes() : byte[] option = heap |> Option.bind _.SlotAreaBytes()

    /// The device address and size behind a slot of the resource heap (the dummy page for unused slots).
    member _.SlotRange(slot: int) : (uint64 * int64) option =
        heap
        |> Option.filter (fun _ -> slot >= 0 && slot < Heap.SlotCount)
        |> Option.map (fun h -> h.SlotRange slot)

    /// The SPIR-V inspector's report of every loaded blob.
    member _.SpirvReports: SpirvReport list =
        match kernels with
        | Some k -> k.Reports
        | None -> []

    /// Submissions whose frame has not retired yet.
    member _.PendingSubmissions = pending.Count

    /// Command pools created so far. It stays flat in a soak once the in-flight depth stops growing.
    member _.FramesCreated = ctx.FramesCreated

    /// timestampPeriod (ns per tick).
    member _.TimestampPeriod = ctx.TimestampPeriod

    /// VK_EXT_device_fault's report now (meaningful after VK_ERROR_DEVICE_LOST).
    member _.DeviceFault() : DeviceFaultQuery = ctx.Fault()

    member internal _.HeapOrFail: VulkanHeap =
        match heap with
        | Some h -> h
        | None -> Contract.fail "CreateHeap has not run"

    member internal _.KernelsOrFail: VulkanKernels =
        match kernels with
        | Some k -> k
        | None -> Contract.fail "LoadKernels has not run"

    member internal _.CheckLive() = checkLive ()

    member internal _.Backend = backend

    /// The fp32 denormal probe on a transient private heap (slot 0 a table with tokens x and y, slot 1 a 64 KiB page,
    /// its own resource heap with its own 64 descriptors, counted in ProbeDescriptorWrites): scale_f32 with
    /// a = Fp32Denormals.ProbeScale over Fp32Denormals.probeInputs, the outputs poisoned first. Records
    /// gpu.fp32_denormals with the words, the dispatch in gpu.dispatches and, with TimeDispatches, its gpu.dispatch
    /// interval (case fp32_denormal_probe). Its private transfers are not IGpuHeap transfers, so they are not
    /// gpu.bytes_moved: the gpu.probe span carries them (`upload_bytes`, `download_bytes`) with the heap's
    /// `descriptor_writes`, and gpu.buffer_memory its memory. The transient heap is released after every command buffer
    /// that bound it is complete and reset (VulkanHeapRecipe section 4).
    member private _.RunProbe(k: VulkanKernels) : (Fp32DenormalMode * uint32[]) option =
        if not (k.Contains Kernels.ScaleF32.Id) then
            None
        else
            probeRuns <- probeRuns + 1
            let run = probeRuns
            use _ = Telemetry.scope [ Attr.str TelemetryKeys.Case "fp32_denormal_probe" ]
            use span = Telemetry.span "gpu.probe" [ backendAttr ]
            let layout = HeapLayout.ofPages [ ArenaKind.Scratch, Heap.PageGranularity ]
            let h = new VulkanHeap(ctx, layout, "lrm.vulkan.probe", geometry, placement)
            probeDescriptorWrites <- probeDescriptorWrites + h.DescriptorWrites
            BufferMemory.report "probe_heap" (layout.TotalBytes + h.OverheadBytes)
            span.Add(Attr.int "descriptor_writes" h.DescriptorWrites)

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

                let enc = Plan.encode options.TimeDispatches [ Command.Dispatch prepared ]
                let stamps = ref [||]

                let frame =
                    ctx.Execute
                        (Some(h.Bindings k.Pipeline))
                        enc.Queries
                        enc.Ops
                        (Some(fun f ->
                            stamps.Value <- f.ReadStamps(ctx.Commands, ctx.Handle, enc.Queries, ctx.TimestampMask)))

                GpuTelemetry.dispatches backend enc.Dispatches
                ctx.WaitFor frame.Value
                let out = Array.zeroCreate<uint32> n
                h.ReadSpan(1, int64 yOffset, MemoryMarshal.AsBytes(Span out))
                span.Add(Attr.int64 "upload_bytes" (Heap.TableBytes + int64 pageBytes.Length))
                span.Add(Attr.int64 "download_bytes" (int64 (4 * n)))

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
                BufferMemory.report "probe_heap" 0L

    member internal _.SubmitCommands(commands: Command list) : Submission =
        let enc = Plan.encode options.TimeDispatches commands

        if options.Debug then
            match Plan.verify enc.Dispatches enc.Ops with
            | Ok() -> ()
            | Error e -> invalidOp ("vulkan: the encoder broke a recording rule: " + e)

        let bindings =
            heap
            |> Option.map (fun h ->
                match kernels with
                | Some k -> h.Bindings k.Pipeline
                | None -> h.Bindings(fun _ -> invalidOp "LoadKernels has not run"))

        let number = submitted + 1UL
        let frame = ctx.Execute bindings enc.Queries enc.Ops (Some(retire number))

        pending[number] <-
            {
                Value = frame.Value
                Queries = enc.Queries
                Timers = enc.Timers
                DispatchTimers = enc.DispatchTimers
            }

        dispatchCounts.Add enc.Dispatches
        submitted <- number
        GpuTelemetry.dispatches backend enc.Dispatches

        for bytes in enc.Copies do
            GpuTelemetry.bytesMoved backend "copy" bytes

        Submission number

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

            let h = new VulkanHeap(ctx, layout, "lrm.vulkan.heap", geometry, placement)
            heap <- Some h
            GpuTelemetry.heapCreated backend layout
            BufferMemory.report "resource_heap" h.ResourceHeapBytes
            BufferMemory.report "dummy_page" h.DummyPageBytes
            new VulkanHeapView(this, h) :> IGpuHeap

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
                let k = new VulkanKernels(ctx, shaders, uint32 geometry.SlotStride)
                span.Add(Attr.int "kernels" k.Count)
                k

            kernels <- Some k
            klass <- Some(GpuClass.fill info shaders)
            denormals <- this.RunProbe k
            k :> IKernelSet

        member this.Begin() =
            checkLive ()
            new VulkanCommandList(this) :> ICommandList

        member this.Submit(commands: ICommandList) =
            checkLive ()

            match commands with
            | :? VulkanCommandList as c when obj.ReferenceEquals(c.Owner, this) -> this.SubmitCommands(c.Close())
            | _ -> Contract.fail "Submit takes a command list from this backend's Begin"

        member _.Wait(s: Submission) =
            checkLive ()

            let number = s.Value

            if number = 0UL || number > submitted then
                Contract.fail "submission %d was not made by this backend" number

            use _ = Telemetry.span GpuTelemetry.Wait [ backendAttr ]

            match pending.TryGetValue number with
            | true, p ->
                ctx.WaitFor p.Value

                if pending.ContainsKey number then
                    invalidOp (sprintf "submission %d completed but its frame did not retire" number)
            | _ -> ()

            let dispatches = dispatchCounts[int (number - 1UL)]

            match retiredTimers.TryGetValue number with
            | true, t ->
                let timers = t.Results

                if not t.Emitted then
                    t.Emitted <- true
                    GpuTelemetry.timers backend timers

                {
                    Submission = s
                    Dispatches = dispatches
                    Timers = timers
                }
            | _ ->
                {
                    Submission = s
                    Dispatches = dispatches
                    Timers = []
                }

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
                    // The heap waits for and resets every command buffer that bound it before it is destroyed.
                    (try
                        heap |> Option.iter _.Dispose()
                     with _ ->
                         ())

                    kernels <- None
                    heap <- None
                    ctx.Dispose()

/// The IGpuHeap of a VulkanBackend: the shared transfer rules (Dispatch.checkTransfer), then the staged copies, then
/// gpu.bytes_moved.
and [<Sealed>] internal VulkanHeapView(owner: VulkanBackend, h: VulkanHeap) =
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

/// A command list of a VulkanBackend: commands are checked when recorded (the same checks, in the same order, with the
/// same texts as RecordingBackend and D3D12) and encoded at Submit.
and [<Sealed>] VulkanCommandList internal (owner: VulkanBackend) =
    let commands = ResizeArray<Command>()
    let openTimers = Stack<int>()
    let mutable nextTimer = 0
    let mutable closed = false

    // A list from before the backend's Dispose raises ObjectDisposedException, as the backend's own calls do, before any
    // contract check (whose state Dispose has released).
    let checkOpen () =
        owner.CheckLive()

        if closed then
            Contract.fail "the command list was already submitted"

    member internal _.Owner = owner

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

/// The vulkan backend's provider (GpuBackendInterface: each backend project exposes one; apps and tests register it).
[<Sealed>]
type VulkanProvider() =
    interface IGpuBackendProvider with
        member _.Backend = ExecutionBackend.Vulkan
        member _.Enumerate() = PhysicalDevices.enumerate ()

        member _.Create(device: DeviceInfo, options: DeviceOptions) =
            VulkanBackend.Create(device, options) :> IGpuBackend

/// The vulkan backend's entry points.
[<RequireQualifiedAccess>]
module Vulkan =
    /// The provider to register with Backend.register (apps and tests do; libraries never do).
    let provider: IGpuBackendProvider = VulkanProvider()

    /// A backend on the first accepted device (or `options.Index`), through a private registry holding only this
    /// provider.
    let tryCreate (options: DeviceOptions) : Result<IGpuBackend, string> =
        let registry = BackendRegistry()
        registry.Register provider

        registry.TryCreate
            { options with
                Backend = ExecutionBackend.Vulkan
            }
