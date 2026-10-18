namespace Lrm.Gpu.Core

open System
open System.Collections.Generic
open System.IO
open Lrm.Core

/// What a backend reports about one device (GpuBackendInterface contract).
type DeviceInfo =
    {
        Backend: ExecutionBackend
        /// Index among this backend's devices, in enumeration order.
        Index: int
        /// Marketing name, for example "NVIDIA GeForce RTX 4080".
        Name: string
        /// PCI vendor id (0x10de NVIDIA, 0x1414 Microsoft WARP).
        VendorId: uint32
        /// PCI device id.
        PciDeviceId: uint32
        /// The driver string of the GPU fill rule (GpuClass.d3d12Driver, GpuClass.vulkanDriver).
        Driver: string
        /// Dedicated device memory in bytes.
        MemoryBytes: int64
        /// Further facts for logs and manifests (API version, feature tiers, limits), in a stable order.
        Details: (string * string) list
    }

/// One device found by enumeration, with the reasons it was rejected (empty when accepted).
type DeviceCandidate =
    {
        Info: DeviceInfo
        Rejections: string list
    }

    member c.Accepted = c.Rejections.IsEmpty

/// How to create a backend.
type DeviceOptions =
    {
        Backend: ExecutionBackend
        /// A device index from enumeration; None takes the first accepted device.
        Index: int option
        /// Enable the API's debug or validation layer when it is installed.
        Debug: bool
        /// Bracket every dispatch with timestamps and report it as a gpu.dispatch GpuInterval.
        TimeDispatches: bool
    }

    static member For(backend: ExecutionBackend) =
        {
            Backend = backend
            Index = None
            Debug = false
            TimeDispatches = false
        }

/// A byte range inside one page slot. Offsets and sizes are multiples of 4.
[<Struct>]
type BufferRange =
    {
        Slot: int
        Offset: int64
        Bytes: int64
    }

/// A timer opened by ICommandList.BeginTimer.
[<Struct>]
type TimerId =
    | TimerId of int

    member x.Value =
        let (TimerId v) = x
        v

/// A GPU-timed region, available after Wait.
type TimerResult =
    {
        Name: string
        Attrs: Attr list
        /// Start in the queue's timestamp domain, in ns.
        GpuStartNs: int64
        DurationNs: int64
    }

/// A submitted command list: its fence value, increasing from 1 per backend.
[<Struct>]
type Submission =
    | Submission of uint64

    member x.Value =
        let (Submission v) = x
        v

/// The outcome of one submission.
type Completion =
    {
        Submission: Submission
        Dispatches: int
        /// Timers in the order they were opened, then gpu.dispatch timings when DeviceOptions.TimeDispatches is set.
        Timers: TimerResult list
    }

/// The kernels a backend loaded from a ShaderSet.
type IKernelSet =
    /// The ShaderSetHash of the backend's target.
    abstract ShaderSetHash: ContentId
    /// True when the set has a blob for the kernel.
    abstract Contains: kernel: KernelId -> bool

/// The resource heap: 64 descriptor slots, written once (HeapLayoutAndTensorTable contract).
type IGpuHeap =
    abstract Layout: HeapLayout
    /// Writes the sealed tensor table into slot 0. Exactly once, before the first Submit.
    abstract WriteTable: table: SealedTable -> unit
    /// The table written by WriteTable, if any.
    abstract Table: SealedTable option
    /// Copies bytes into a page (slot 1..63). Waits for all submitted work first and returns when the copy is done.
    abstract Upload: slot: int * offset: int64 * data: ReadOnlySpan<byte> -> unit
    /// Upload from a stream, `length` bytes, staged in chunks of at most 64 MiB.
    abstract UploadFrom: slot: int * offset: int64 * source: Stream * length: int64 -> unit
    /// Copies bytes out of a page. Waits for all submitted work first.
    abstract Download: slot: int * offset: int64 * destination: Span<byte> -> unit

/// Commands recorded for one submission. A list is used once: Submit closes it.
type ICommandList =
    /// Dispatches a catalog kernel. The backend computes the grid from the catalog formula, folds it when needed,
    /// writes argument word 0 (lrm_groups_x) and checks the block against the table (Dispatch.prepare).
    abstract Dispatch: kernel: KernelId * args: ArgBlock -> unit
    /// A global barrier: every earlier command finishes its writes before any later command starts.
    abstract Barrier: unit -> unit
    /// Copies bytes between pages, state-correct on every backend. It acts as a full barrier on both sides.
    abstract Copy: source: BufferRange * destination: BufferRange -> unit
    /// Opens a GPU-timed region; `name` is a telemetry name (gpu.*).
    abstract BeginTimer: name: string * attrs: Attr list -> TimerId
    /// Closes a timer. Timers close in reverse order of opening.
    abstract EndTimer: timer: TimerId -> unit

/// One GPU device (GpuBackendInterface contract). Calls come from one thread at a time.
type IGpuBackend =
    inherit IDisposable
    abstract Info: DeviceInfo
    /// The execution class (GpuClass.fill). Shader backends know it once LoadKernels has run.
    abstract ExecutionClass: ExecutionClass
    /// Creates the heap and writes all 64 descriptors. Exactly once per backend.
    abstract CreateHeap: layout: HeapLayout -> IGpuHeap
    /// Creates pipelines for the backend's target of a shader set. Exactly once per backend.
    abstract LoadKernels: shaders: ShaderSet -> IKernelSet
    /// A new, empty command list.
    abstract Begin: unit -> ICommandList
    /// Submits a command list and returns its fence value.
    abstract Submit: commands: ICommandList -> Submission
    /// Blocks until a submission has completed.
    abstract Wait: submission: Submission -> Completion

/// Enumerates and creates the devices of one backend. Apps and tests register providers; libraries never do.
type IGpuBackendProvider =
    abstract Backend: ExecutionBackend
    /// Every device of this backend, accepted or not, in a stable order.
    abstract Enumerate: unit -> DeviceCandidate list
    /// Creates a backend on an accepted device.
    abstract Create: device: DeviceInfo * options: DeviceOptions -> IGpuBackend

/// The device requirements, as the reasons backends report (GpuBackendInterface contract).
[<RequireQualifiedAccess>]
module Requirements =
    /// Vulkan: the device must expose VK_EXT_descriptor_heap.
    [<Literal>]
    let VulkanDescriptorHeap = "requires VK_EXT_descriptor_heap"

    /// Vulkan: the 64 descriptors sit at stride 16, so a buffer descriptor must fit in 16 bytes.
    [<Literal>]
    let VulkanDescriptorSize = "requires buffer descriptors of at most 16 bytes"

    /// D3D12: shader model 6.6 (ResourceDescriptorHeap indexing).
    [<Literal>]
    let D3D12ShaderModel66 = "requires shader model 6.6"

    /// D3D12: resource binding tier 3.
    [<Literal>]
    let D3D12BindingTier3 = "requires resource binding tier 3"

    /// Every backend: 128 bytes of arguments (Vulkan maxPushDataSize, D3D12 root constants).
    [<Literal>]
    let ArgumentBytes = "requires 128 bytes of kernel arguments"

/// A registry of backend providers. Backend (the module) wraps one process-wide instance.
[<Sealed>]
type BackendRegistry() =
    let providers = Dictionary<ExecutionBackend, IGpuBackendProvider>()

    /// Registers a provider; a second provider for the same backend replaces the first.
    member _.Register(p: IGpuBackendProvider) =
        if isNull (box p) then
            nullArg "p"

        lock providers (fun () -> providers[p.Backend] <- p)

    /// The registered backends, in ExecutionBackend order.
    member _.Backends =
        lock providers (fun () -> ExecutionBackend.all |> List.filter providers.ContainsKey)

    /// Every device of every registered backend, in ExecutionBackend order.
    member x.Enumerate() : DeviceCandidate list =
        let ps =
            lock providers (fun () ->
                ExecutionBackend.all
                |> List.choose (fun b ->
                    match providers.TryGetValue b with
                    | true, p -> Some p
                    | _ -> None))

        ps |> List.collect (fun p -> p.Enumerate())

    /// Creates a backend: the device `Index` (which must be accepted), else the first accepted device.
    member _.TryCreate(o: DeviceOptions) : Result<IGpuBackend, string> =
        let provider =
            lock providers (fun () ->
                match providers.TryGetValue o.Backend with
                | true, p -> Some p
                | _ -> None)

        match provider with
        | None -> Error(sprintf "no provider is registered for %s" (ExecutionBackend.name o.Backend))
        | Some p ->
            let candidates = p.Enumerate()

            let chosen =
                match o.Index with
                | Some i -> candidates |> List.tryFind (fun c -> c.Info.Index = i)
                | None -> candidates |> List.tryFind _.Accepted

            match chosen with
            | Some c when c.Accepted ->
                try
                    Ok(p.Create(c.Info, o))
                with e ->
                    Error(
                        sprintf
                            "%s device %d (%s): %s"
                            (ExecutionBackend.name o.Backend)
                            c.Info.Index
                            c.Info.Name
                            e.Message
                    )
            | Some c ->
                Error(
                    sprintf
                        "%s device %d (%s) is rejected: %s"
                        (ExecutionBackend.name o.Backend)
                        c.Info.Index
                        c.Info.Name
                        (String.concat "; " c.Rejections)
                )
            | None ->
                let all =
                    candidates
                    |> List.map (fun c ->
                        sprintf "%d %s: %s" c.Info.Index c.Info.Name (String.concat "; " c.Rejections))

                Error(
                    sprintf
                        "no accepted %s device%s%s"
                        (ExecutionBackend.name o.Backend)
                        (match o.Index with
                         | Some i -> sprintf " with index %d" i
                         | None -> "")
                        (if all.IsEmpty then
                             ""
                         else
                             " (" + String.concat ", " all + ")")
                )

/// The factories of the GpuBackendInterface contract, over one process-wide registry.
[<RequireQualifiedAccess>]
module Backend =
    let private registry = BackendRegistry()

    /// The process-wide registry.
    let defaultRegistry () = registry

    /// Registers a provider (apps and tests do this; libraries never wire backends).
    let register (p: IGpuBackendProvider) = registry.Register p

    /// Every device of every registered backend, accepted or rejected.
    let enumerate () = registry.Enumerate()

    /// Creates a backend on an accepted device.
    let tryCreate (o: DeviceOptions) = registry.TryCreate o

/// The GPU fill rule of the ExecutionClass contract (R4a).
[<RequireQualifiedAccess>]
module GpuClass =
    /// DeviceId: "pci:<vendor>:<device>", each four lowercase hex digits, for example "pci:10de:2704".
    let deviceId (vendorId: uint32) (pciDeviceId: uint32) =
        sprintf "pci:%04x:%04x" vendorId pciDeviceId

    /// D3D12 Driver: the user-mode driver version from IDXGIAdapter::CheckInterfaceSupport (a LARGE_INTEGER) as
    /// four decimal parts, high word first, for example "32.0.16.1692".
    let d3d12Driver (umdVersion: int64) =
        let v = uint64 umdVersion

        sprintf
            "%d.%d.%d.%d"
            ((v >>> 48) &&& 0xFFFFUL)
            ((v >>> 32) &&& 0xFFFFUL)
            ((v >>> 16) &&& 0xFFFFUL)
            (v &&& 0xFFFFUL)

    /// Vulkan Driver: VkPhysicalDeviceDriverProperties.driverInfo trimmed (NVIDIA: "616.92"), or the raw
    /// driverVersion as "0x" and 8 hex digits when driverInfo is empty or not printable ASCII.
    let vulkanDriver (driverInfo: string) (driverVersion: uint32) =
        let info = if isNull driverInfo then "" else driverInfo.Trim()

        if
            info.Length > 0
            && info.Length <= 256
            && info |> Seq.forall (fun c -> c >= ' ' && c <= '~')
        then
            info
        else
            sprintf "0x%08x" driverVersion

    /// Arch for a GPU class: the host OS and process architecture, for example "win-x64". CPU instruction-set flags
    /// are added only for d3d12-warp, whose rasterizer runs on the CPU.
    let arch (backend: ExecutionBackend) =
        let full = ExecutionClass.hostArch ()

        if backend = ExecutionBackend.D3D12Warp then
            full
        else
            match full.IndexOf '+' with
            | -1 -> full
            | i -> full.Substring(0, i)

    /// The class of a shader backend on `device` running `shaders`: DeviceId from the PCI ids, Driver from the device,
    /// ShaderSetHash of the backend's target, Arch from `arch`.
    let fill (device: DeviceInfo) (shaders: ShaderSet) : ExecutionClass =
        match ShaderTarget.forBackend device.Backend with
        | None -> invalidArg "device" (sprintf "%s does not run shaders" (ExecutionBackend.name device.Backend))
        | Some target ->
            {
                Backend = device.Backend
                DeviceId = deviceId device.VendorId device.PciDeviceId
                Driver = device.Driver
                ShaderSetHash = Some(ShaderSet.setHash target shaders)
                Arch = arch device.Backend
            }

/// The reserved telemetry names the GPU stack emits (TelemetryFacade contract, gpu.* and mem.arena).
[<RequireQualifiedAccess>]
module GpuTelemetry =
    /// GpuInterval per dispatch when DeviceOptions.TimeDispatches is set; attrs backend, kernel.
    [<Literal>]
    let Dispatch = "gpu.dispatch"

    /// Counter per submission: the number of dispatches; attrs backend.
    [<Literal>]
    let Dispatches = "gpu.dispatches"

    /// Meter in bytes for Upload, Download and Copy; attrs backend, direction (upload, download, copy).
    [<Literal>]
    let BytesMoved = "gpu.bytes_moved"

    /// Span: host time blocked in Wait; attrs backend.
    [<Literal>]
    let Wait = "gpu.wait"

    /// Gauge in bytes per arena at CreateHeap; attrs backend, arena.
    [<Literal>]
    let MemArena = TelemetryNames.MemArena

    /// Gauge: slots that hold a page or the table, at CreateHeap; attrs backend.
    [<Literal>]
    let HeapSlots = "gpu.heap.slots"

    /// Span per kernel compile in the shader build; attrs kernel, target.
    [<Literal>]
    let ShaderCompile = "gpu.shader.compile"

    /// Gauge: bytes of one compiled blob; attrs kernel, target.
    [<Literal>]
    let ShaderBytes = "gpu.shader.bytes"

    let private backendAttr (backend: ExecutionBackend) =
        Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend)

    /// What every backend's CreateHeap emits: mem.arena per arena with pages (and the table as arena "table"), then
    /// gpu.heap.slots.
    let heapCreated (backend: ExecutionBackend) (layout: HeapLayout) =
        let b = backendAttr backend

        Telemetry.gauge
            MemArena
            (float Heap.TableBytes)
            TelemetryUnits.Bytes
            [ b; Attr.str TelemetryKeys.Arena "table" ]

        for arena in layout.Arenas do
            Telemetry.gauge
                MemArena
                (float (layout.ArenaBytes arena))
                TelemetryUnits.Bytes
                [ b; Attr.str TelemetryKeys.Arena (ArenaKind.name arena) ]

        Telemetry.gauge HeapSlots (float (1 + layout.PageCount)) TelemetryUnits.Count [ b ]

    /// gpu.bytes_moved for one transfer.
    let bytesMoved (backend: ExecutionBackend) (direction: string) (bytes: int64) =
        Telemetry.meter
            BytesMoved
            (float bytes)
            TelemetryUnits.Bytes
            [ backendAttr backend; Attr.str "direction" direction ]

    /// gpu.dispatches for one submission.
    let dispatches (backend: ExecutionBackend) (count: int) =
        Telemetry.counter Dispatches (float count) TelemetryUnits.Count [ backendAttr backend ]

    /// One GpuInterval per timer result.
    let timers (backend: ExecutionBackend) (results: TimerResult list) =
        for r in results do
            Telemetry.gpuInterval r.Name r.GpuStartNs r.DurationNs (backendAttr backend :: r.Attrs)

    /// Event, once per backend after LoadKernels: the measured fp32 denormal mode; attrs backend, mode.
    [<Literal>]
    let Fp32Denormals = "gpu.fp32_denormals"

/// How a device treats fp32 denormals in kernel arithmetic, as measured by the scale_f32 probe
/// (GpuBackendInterface contract). It is a property of the ExecutionClass: fixed for one device, driver and blob set.
[<RequireQualifiedAccess>]
type Fp32DenormalMode =
    /// Denormal inputs and results are kept, as on the CPU reference.
    | Preserve
    /// Denormal inputs and results become zero of the same sign.
    | Flush
    /// Neither model explains the probe (for example inputs flushed but results kept).
    | Mixed

/// The fp32 denormal probe (GpuBackendInterface contract): scale_f32 with a = ProbeScale over probeInputs. Backends
/// run it right after LoadKernels, classify the result bits and record the mode; it never rejects a device.
[<RequireQualifiedAccess>]
module Fp32Denormals =
    /// The scale factor of the probe dispatch.
    [<Literal>]
    let ProbeScale = 0.5f

    /// The probe's input bits: normal controls, normal inputs with denormal results, and denormal inputs.
    let probeInputs: uint32[] =
        [|
            0x3F800000u // 1.0 -> 0.5 (control)
            0x01000000u // 2^-125 -> the smallest normal (control)
            0x00800000u // the smallest normal -> a denormal result
            0x80800000u // its negative
            0x00000002u // a denormal input -> the smallest denormal
            0x807FFFFEu // a negative denormal input
            0x00400000u // a denormal input, 2^-127
        |]

    let name (mode: Fp32DenormalMode) =
        match mode with
        | Fp32DenormalMode.Preserve -> "preserve"
        | Fp32DenormalMode.Flush -> "flush"
        | Fp32DenormalMode.Mixed -> "mixed"

    let private isDenormal (bits: uint32) =
        bits &&& 0x7F800000u = 0u && bits &&& 0x007FFFFFu <> 0u

    /// A denormal becomes zero of the same sign; other values are unchanged.
    let flushBits (bits: uint32) =
        if isDenormal bits then bits &&& 0x80000000u else bits

    let private scale (bits: uint32) =
        BitConverter.SingleToUInt32Bits(BitConverter.UInt32BitsToSingle bits * ProbeScale)

    /// The result bits the probe must give in `mode` (Preserve or Flush; Mixed has no single answer).
    let expected (mode: Fp32DenormalMode) : uint32[] =
        match mode with
        | Fp32DenormalMode.Preserve -> probeInputs |> Array.map scale
        | Fp32DenormalMode.Flush -> probeInputs |> Array.map (flushBits >> scale >> flushBits)
        | Fp32DenormalMode.Mixed -> invalidArg "mode" "mixed has no single expected result"

    /// The mode the probe's result bits show; raises GpuContractException when the controls are wrong (the probe
    /// itself failed) or the length differs.
    let classify (results: uint32[]) : Fp32DenormalMode =
        if isNull results || results.Length <> probeInputs.Length then
            Fail.contract "the fp32 denormal probe needs %d results" probeInputs.Length

        let preserve = expected Fp32DenormalMode.Preserve

        if results[0] <> preserve[0] || results[1] <> preserve[1] then
            Fail.contract "the fp32 denormal probe's normal controls are wrong: scale_f32 itself failed"

        if results = preserve then
            Fp32DenormalMode.Preserve
        elif results = expected Fp32DenormalMode.Flush then
            Fp32DenormalMode.Flush
        else
            Fp32DenormalMode.Mixed

    /// Emits gpu.fp32_denormals for a backend's measured mode.
    let record (backend: ExecutionBackend) (mode: Fp32DenormalMode) =
        Telemetry.event
            GpuTelemetry.Fp32Denormals
            [
                Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend)
                Attr.str "mode" (name mode)
            ]
