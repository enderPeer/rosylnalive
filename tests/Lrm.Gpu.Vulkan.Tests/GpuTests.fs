/// gpu tier (vulkan mutex; the D3D12 reference run under the d3d12 mutex): the backend on the real device. The device
/// check; the GpuBackendInterface rules, with the same exceptions and texts as RecordingBackend; calls after Dispose;
/// every descriptor write (the resource heap's 64 and the probe heap's 64), the reserved range never written, the slot
/// area unchanged; the fp32 denormal probe and its telemetry; the five selftests bit-identical twice, against the CPU
/// (KernelBounds, CpuRef) and against D3D12 where the classes agree; the copied-heap fallback; state-correct copies; a
/// whole 1 GiB page; timers; a round trip of the 1.5 GB Qwen3-0.6B weights file; 10,000-dispatch soaks timed, with a
/// timer per submission and without timers, managed memory included; Vulkan object counts; and hello-vulkan.
/// Everything is written to the run log (kind kernels) that `lrm report` renders.
module Lrm.Gpu.Vulkan.Tests.GpuTests

open System
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices
open System.Security.Cryptography
open Expecto
open Lrm.Gpu.Vulkan.Interop
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan
open Lrm.Gpu.Vulkan.Tests.Common
open Lrm.Tensor
open Lrm.Testing

let private range slot offset bytes : BufferRange =
    {
        Slot = slot
        Offset = offset
        Bytes = bytes
    }

let private tokenOf (table: SealedTable) (name: string) =
    match table.TryFind name with
    | Some t -> t
    | None -> failtestf "no tensor %s" name

/// Random normal float32 words in [1, 2) * 2^e for small e, so products stay normal.
let private normals (seed: string) (n: int) =
    let rng = DetRng(SelfTests.Seed, DetRng.streamOf seed)
    Array.init n (fun _ -> (0x3F800000u + ((rng.NextU32() % 8u) <<< 23)) ||| (rng.NextU32() >>> 9))

let private f32 (w: uint32) = BitConverter.UInt32BitsToSingle w
let private bits (x: float32) = BitConverter.SingleToUInt32Bits x

let private isDenormal (w: uint32) =
    w &&& 0x7F800000u = 0u && w &&& 0x007FFFFFu <> 0u

// ---- GpuBackendInterface rules on hardware, compared with RecordingBackend ---------------------------------------

/// Runs the same sequence of calls, valid and invalid, and returns each step's outcome: "ok", or the exception type and
/// text. Vulkan must give exactly RecordingBackend's outcomes (GpuBackendInterface, "Errors"); D1 runs the same script.
let private contractScript (b: IGpuBackend) (foreign: IGpuBackend) : (string * string) list =
    let steps = ResizeArray<string * string>()

    let step (name: string) (f: unit -> unit) =
        let outcome =
            try
                f ()
                "ok"
            with
            | :? GpuContractException as e -> "GpuContractException: " + e.Message
            | :? ObjectDisposedException -> "ObjectDisposedException"
            | e -> e.GetType().Name + ": " + e.Message

        steps.Add(name, outcome)

    let layout, table = SelfTests.Suite.Plan()
    let other = HeapLayout.ofPages [ ArenaKind.Scratch, Heap.PageGranularity ]
    let otherTable = (TensorTable other).Seal()
    let tok = tokenOf table
    let mutable heap: IGpuHeap option = None
    let h () = heap.Value
    let pageBytes = (layout.Page 1).Bytes

    let scale x y n =
        Kernels.ScaleF32.encode { X = x; Y = y; N = n; A = 1.0f }

    let scaleArgs = scale (tok "scale.x") (tok "scale.y") 16u

    let dispatch (args: ArgBlock) () =
        b.Begin().Dispatch(Kernels.ScaleF32.Id, args)

    step "ExecutionClass before LoadKernels" (fun () -> b.ExecutionClass |> ignore)
    step "Dispatch before CreateHeap" (dispatch scaleArgs)
    step "Copy before CreateHeap" (fun () -> b.Begin().Copy(range 1 0L 4L, range 2 0L 4L))
    step "CreateHeap" (fun () -> heap <- Some(b.CreateHeap layout))
    step "CreateHeap twice" (fun () -> b.CreateHeap layout |> ignore)
    step "Dispatch before LoadKernels" (dispatch scaleArgs)

    step "LoadKernels of another catalog" (fun () ->
        b.LoadKernels
            { shaders () with
                CatalogHash = Hash.sha256Utf8 "another catalog"
            }
        |> ignore)

    step "LoadKernels" (fun () -> b.LoadKernels(shaders ()) |> ignore)
    step "LoadKernels twice" (fun () -> b.LoadKernels(shaders ()) |> ignore)
    step "ExecutionClass after LoadKernels" (fun () -> b.ExecutionClass |> ignore)
    step "Dispatch before WriteTable" (dispatch scaleArgs)
    step "WriteTable of another layout's table" (fun () -> h().WriteTable otherTable)
    step "Upload to the table slot" (fun () -> h().Upload(0, 0L, ReadOnlySpan(Array.zeroCreate<byte> 4)))
    step "WriteTable" (fun () -> h().WriteTable table)
    step "WriteTable twice" (fun () -> h().WriteTable table)

    step "Download the table" (fun () ->
        let out = Array.zeroCreate<byte> 128
        h().Download(0, 0L, Span out)

        if out[64..] <> (table.Row(TensorToken 1u)).ToBytes() then
            failwith "row 1 differs")

    step "Upload at an unaligned offset" (fun () -> h().Upload(1, 2L, ReadOnlySpan(Array.zeroCreate<byte> 4)))
    step "Upload of an unaligned size" (fun () -> h().Upload(1, 0L, ReadOnlySpan(Array.zeroCreate<byte> 3)))
    step "Upload past the page" (fun () -> h().Upload(1, pageBytes - 4L, ReadOnlySpan(Array.zeroCreate<byte> 8)))
    step "Upload to an unused slot" (fun () -> h().Upload(60, 0L, ReadOnlySpan(Array.zeroCreate<byte> 4)))
    step "Download from slot 64" (fun () -> h().Download(64, 0L, Span(Array.zeroCreate<byte> 4)))

    step "UploadFrom past the page" (fun () ->
        use s = new MemoryStream(Array.zeroCreate<byte> 16)
        h().UploadFrom(1, pageBytes - 8L, s, 16L))

    step "Upload and Download round trip" (fun () ->
        let data = Array.init 4096 (fun i -> byte (i * 7 + 3))
        h().Upload(2, 1024L, ReadOnlySpan data)
        let back = Array.zeroCreate<byte> data.Length
        h().Download(2, 1024L, Span back)

        if back <> data then
            failwith "bytes differ")

    step "UploadFrom round trip" (fun () ->
        let data = Array.init 8192 (fun i -> byte (i * 13 + 1))
        use s = new MemoryStream(data)
        h().UploadFrom(3, 4096L, s, int64 data.Length)
        let back = Array.zeroCreate<byte> data.Length
        h().Download(3, 4096L, Span back)

        if back <> data then
            failwith "bytes differ")

    step "Dispatch with word 0 set" (dispatch (scaleArgs.WithGroupsX 1u))
    step "Dispatch with a word past the last field" (dispatch (scaleArgs.With(5, 1u)))
    step "Dispatch with the null token" (dispatch (scale TensorToken.Null (tok "scale.y") 16u))
    step "Dispatch with an unassigned token" (dispatch (scale (TensorToken 999u) (tok "scale.y") 16u))
    step "Dispatch with a token above 16,383" (dispatch (scale (TensorToken 20000u) (tok "scale.y") 16u))
    step "Dispatch with a wrong dtype" (dispatch (scale (tok "bf16.x") (tok "scale.y") 16u))
    step "Dispatch writing a read-only tensor" (dispatch (scale (tok "scale.y") (tok "scale.x") 16u))
    step "Dispatch with an empty grid" (dispatch (scale (tok "scale.x") (tok "scale.y") 0u))

    step "Dispatch of a kernel the set lacks" (fun () ->
        Kernels.RmsnormRows.dispatch
            (b.Begin())
            {
                X = tok "scale.x"
                W = tok "bf16.x"
                Y = tok "scale.y"
                Rows = 1u
                Cols = 4u
                Eps = 1e-6f
            })

    step "Dispatch of kernel id 99" (fun () -> b.Begin().Dispatch(KernelId 99, ArgBlock.Zero))
    step "Copy from the table slot" (fun () -> b.Begin().Copy(range 0 0L 4L, range 1 0L 4L))
    step "Copy within a page, overlapping" (fun () -> b.Begin().Copy(range 1 0L 8L, range 1 4L 8L))
    step "Copy of different sizes" (fun () -> b.Begin().Copy(range 1 0L 8L, range 2 0L 4L))
    step "Copy at an unaligned offset" (fun () -> b.Begin().Copy(range 1 2L 4L, range 2 0L 4L))
    step "Copy past the page" (fun () -> b.Begin().Copy(range 1 (pageBytes - 4L) 8L, range 2 0L 8L))
    step "Timer with an invalid name" (fun () -> b.Begin().BeginTimer("Not A Name", []) |> ignore)

    step "EndTimer out of order" (fun () ->
        let c = b.Begin()
        let t0 = c.BeginTimer("gpu.outer", [])
        c.BeginTimer("gpu.inner", []) |> ignore
        c.EndTimer t0)

    step "Submit with a timer open" (fun () ->
        let c = b.Begin()
        c.BeginTimer("gpu.open", []) |> ignore
        b.Submit c |> ignore)

    step "Submit twice" (fun () ->
        let c = b.Begin()
        b.Submit c |> ignore
        b.Submit c |> ignore)

    step "Record after Submit" (fun () ->
        let c = b.Begin()
        b.Submit c |> ignore
        c.Barrier())

    step "Submit a list from another backend" (fun () -> b.Submit(foreign.Begin()) |> ignore)
    step "Wait for a submission never made" (fun () -> b.Wait(Submission 999UL) |> ignore)

    step "Submit and Wait twice" (fun () ->
        let s = b.Submit(b.Begin())

        if b.Wait s <> b.Wait s then
            failwith "two waits differ")

    b.Dispose()
    step "Begin after Dispose" (fun () -> b.Begin() |> ignore)
    List.ofSeq steps

let private interfaceRules () =
    use foreign = new RecordingBackend()
    let expected = contractScript (new RecordingBackend()) foreign
    let actual = contractScript (create ()) foreign

    for (name, e), (_, a) in List.zip expected actual do
        printfn "  %-42s %s" name a
        Expect.equal a e (sprintf "step '%s'" name)

    // The script exercises errors, not only successes.
    let raised =
        expected |> List.filter (fun (_, o) -> o.StartsWith "GpuContractException")

    Expect.isGreaterThan raised.Length 30 "most steps raise the contract exception"

/// Past the 48-step script, where the backends need not agree: after Dispose every call raises
/// ObjectDisposedException, also on a command list begun before it; a second Dispose does nothing. A shader set whose
/// SPIR-V blob breaks the inspector's rules is an ArgumentException, not a contract violation, and leaves LoadKernels
/// free to run with the right set.
let private disposedAndBadSpirv () =
    use b = create ()
    let ib = b :> IGpuBackend
    let layout, table = SelfTests.Suite.Plan()
    let heap = ib.CreateHeap layout
    heap.WriteTable table
    let set = shaders ()
    let before = VulkanObjects.count ()

    let withScale (f: byte[] -> byte[]) =
        { set with
            Blobs =
                set.Blobs
                |> List.map (fun blob ->
                    if blob.Target = ShaderTarget.Spirv && blob.Kernel = "scale_f32" then
                        { blob with Bytes = f blob.Bytes }
                    else
                        blob)
        }

    // The inspector refuses a zeroed magic number; V1's guard refuses a module without its final OpFunctionEnd, which
    // the inspector accepts and on which the driver hung (never handed to the driver now).
    for what, bad in
        [
            "a zeroed magic number (the inspector)",
            withScale (fun b ->
                let c = Array.copy b
                c[0] <- 0uy
                c)
            "a module truncated by its OpFunctionEnd (V1's guard)", withScale (fun b -> b[.. b.Length - 5])
        ] do
        Expect.throwsT<ArgumentException> (fun () -> ib.LoadKernels bad |> ignore) what

        Expect.equal
            (VulkanObjects.count ())
            before
            (what + ": the failed LoadKernels destroyed every pipeline it made")

    ib.LoadKernels set |> ignore
    let tok = tokenOf table
    let c = ib.Begin()
    let t = c.BeginTimer("gpu.v1.open", [])
    let s = ib.Submit(ib.Begin())
    ib.Wait s |> ignore
    ib.Dispose()

    let disposed (name: string) (f: unit -> unit) =
        Expect.throwsT<ObjectDisposedException> f name

    disposed "Dispatch on a list begun before Dispose" (fun () ->
        Kernels.ScaleF32.dispatch
            c
            {
                X = tok "scale.x"
                Y = tok "scale.y"
                N = 16u
                A = 1.0f
            })

    disposed "Barrier on it" (fun () -> c.Barrier())
    disposed "Copy on it" (fun () -> c.Copy(range 1 0L 4L, range 2 0L 4L))
    disposed "BeginTimer on it" (fun () -> c.BeginTimer("gpu.v1.late", []) |> ignore)
    disposed "EndTimer on it" (fun () -> c.EndTimer t)
    disposed "Submit it" (fun () -> ib.Submit c |> ignore)
    disposed "Wait" (fun () -> ib.Wait s |> ignore)
    disposed "Begin" (fun () -> ib.Begin() |> ignore)
    disposed "ExecutionClass" (fun () -> ib.ExecutionClass |> ignore)
    disposed "CreateHeap" (fun () -> ib.CreateHeap layout |> ignore)
    disposed "LoadKernels" (fun () -> ib.LoadKernels set |> ignore)
    disposed "WriteTable" (fun () -> heap.WriteTable table)
    disposed "Upload" (fun () -> heap.Upload(1, 0L, ReadOnlySpan(Array.zeroCreate<byte> 4)))
    disposed "Download" (fun () -> heap.Download(1, 0L, Span(Array.zeroCreate<byte> 4)))
    ib.Dispose()

// ---- device, heap, probe ---------------------------------------------------------------------------------------

let private deviceCheck () =
    recordDevice ()
    // Discovery inside the run log: gpu.device.discover, gpu.device.check and gpu.device.memory (DeviceCheck).
    let all = PhysicalDevices.enumerate ()
    Expect.equal (all |> List.map _.Info) (allCandidates () |> List.map _.Info) "discovery is stable"
    let info = accepted ()
    Expect.equal info.Backend ExecutionBackend.Vulkan "backend"

    for c in all do
        if c.Info.Name.Contains "RTX 4080" then
            Expect.isTrue c.Accepted "the RTX 4080 is accepted"

    use b = create ()

    Expect.contains
        b.EnabledExtensions
        VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME
        "VK_EXT_descriptor_heap is enabled"

    Expect.isTrue b.DeviceFaultEnabled "VK_EXT_device_fault is enabled where the device supports it"

    Expect.equal
        (Some b.QueueFamily)
        (Invariant.tryParseI32 (detail "queue_family") |> Option.map uint32)
        "the recipe's queue family"

    Expect.isGreaterThan b.TimestampPeriod 0.0 "timestamp period"

    printfn
        "vulkan: %s (%s), driver %s, Vulkan %s, queue family %d, %s ns per tick, extensions %s, debug names %s, validation %s"
        info.Name
        (GpuClass.deviceId info.VendorId info.PciDeviceId)
        info.Driver
        (detail "api")
        b.QueueFamily
        (Invariant.f64 b.TimestampPeriod)
        (String.concat "," b.EnabledExtensions)
        (if b.DebugNames then "on" else "unavailable")
        (if b.ValidationEnabled then "on" else "not installed")

    // Before any loss the driver has nothing to describe: the query says so and never raises (R2: NVIDIA 616.92
    // answers a healthy device with VK_ERROR_UNKNOWN).
    let fault = b.DeviceFault()
    printfn "device fault on a healthy device: %s" (FaultText.format fault)

    match fault with
    | DeviceFaultQuery.Fault f -> failtestf "a healthy device reports a fault: %s" f.Description
    | DeviceFaultQuery.NotEnabled
    | DeviceFaultQuery.NoFault _ -> ()

    printfn "memory types: %s" (detail "memory_types")

/// Every descriptor write the backend makes, on every resource heap: the resource heap's 64 at CreateHeap (in one call,
/// each into its slot, none into the reserved range), the denormal probe's transient heap's 64 at LoadKernels
/// (released when the probe ends), and none afterwards, through WriteTable, a dispatch, a copy, an upload and a
/// download; the slot area keeps its bytes and the heap never moves.
let private heapWrites () =
    use b = create ()
    let ib = b :> IGpuBackend
    let layout, table = SelfTests.Suite.Plan()
    Expect.equal b.DescriptorWrites 0 "nothing before CreateHeap"
    let heap = ib.CreateHeap layout
    Expect.equal b.HeapDescriptorWrites Heap.SlotCount "the resource heap writes its 64 descriptors at CreateHeap"
    Expect.equal b.ProbeDescriptorWrites 0 "no probe before LoadKernels"
    let g = b.Geometry
    let address, size, reservedOffset, reservedBytes = b.HeapRange.Value
    let alignment = uint64 (Invariant.parseU64 (detail "resource_heap_alignment"))
    Expect.equal (address % alignment) 0UL "heapRange.address is aligned to resourceHeapAlignment"
    Expect.equal size (uint64 g.HeapBytes) "heapRange.size"
    Expect.equal reservedOffset (uint64 g.ReservedRangeOffset) "reservedRangeOffset"
    Expect.equal reservedBytes (uint64 g.ReservedRangeBytes) "reservedRangeSize"
    let destinations = b.DescriptorDestinations

    Expect.equal
        destinations
        [ for i in 0 .. Heap.SlotCount - 1 -> int64 i * g.SlotStride, g.SlotStride ]
        "slot i is written at byte i * bufferDescriptorSize"

    let intoReserved =
        destinations
        |> List.filter (fun (at, n) -> at + n > int64 reservedOffset && at < int64 (reservedOffset + reservedBytes))

    Expect.isEmpty intoReserved "no descriptor write touches the reserved range"

    let align =
        uint64 (Invariant.parseU64 (detail "min_storage_buffer_offset_alignment"))

    let dummy = b.SlotRange(Heap.SlotCount - 1) |> Option.get

    for slot in 0 .. Heap.SlotCount - 1 do
        let a, n = b.SlotRange slot |> Option.get
        Expect.equal (a % align) 0UL (sprintf "slot %d's address is aligned" slot)

        if slot = 0 then
            Expect.equal n Heap.TableBytes "slot 0 is the 1 MiB table"
        elif slot <= layout.PageCount then
            Expect.equal n (layout.Page slot).Bytes (sprintf "slot %d is its page" slot)
            Expect.notEqual (a, n) dummy (sprintf "slot %d is not the dummy page" slot)
        else
            Expect.equal (a, n) dummy (sprintf "slot %d holds the dummy page" slot)

    let slotArea0 = b.SlotAreaBytes() |> Option.get
    Expect.equal slotArea0.Length (int g.SlotAreaBytes) "the slot area"
    Expect.isTrue (slotArea0 |> Array.exists ((<>) 0uy)) "the slot area holds descriptors"
    heap.WriteTable table
    Expect.equal b.DescriptorWrites Heap.SlotCount "WriteTable writes no descriptor"
    ib.LoadKernels(shaders ()) |> ignore
    Expect.equal b.HeapDescriptorWrites Heap.SlotCount "LoadKernels writes nothing on the resource heap"
    Expect.equal b.ProbeDescriptorWrites Heap.SlotCount "the probe's transient heap writes its 64 descriptors"
    let tok = tokenOf table
    let c = ib.Begin()

    Kernels.ScaleF32.dispatch
        c
        {
            X = tok "scale.x"
            Y = tok "scale.y"
            N = uint32 SelfTests.ScaleN
            A = 1.0f
        }

    c.Copy(range 1 0L 64L, range 2 0L 64L)
    ib.Wait(ib.Submit c) |> ignore
    heap.Upload(3, 0L, ReadOnlySpan(Array.zeroCreate<byte> 64))
    heap.Download(3, 0L, Span(Array.zeroCreate<byte> 64))
    let total = b.DescriptorWrites
    Expect.equal total (2 * Heap.SlotCount) "64 + 64, and none from dispatches, copies or transfers"
    let slotArea1 = b.SlotAreaBytes() |> Option.get
    Expect.equal slotArea1 slotArea0 "the slot area keeps its bytes"
    Expect.equal b.HeapRange (Some(address, size, reservedOffset, reservedBytes)) "the heap never moves"

    printfn
        "heap: %s, heapRange 0x%x + %d, slots of %d bytes in %d, reserved range %d + %d, memory types %A"
        (string b.HeapPlacement)
        address
        size
        g.SlotStride
        g.SlotAreaBytes
        reservedOffset
        reservedBytes
        b.MemoryTypeChoices

    Telemetry.check
        "gpu.heap.descriptor_writes"
        (b.HeapDescriptorWrites = Heap.SlotCount
         && b.ProbeDescriptorWrites = Heap.SlotCount
         && total = 2 * Heap.SlotCount)
        [
            Attr.int "slots" Heap.SlotCount
            Attr.int "resource_heap" b.HeapDescriptorWrites
            Attr.int "probe_heap" b.ProbeDescriptorWrites
            Attr.int "total" total
        ]

    Telemetry.check
        "gpu.heap.reserved_range"
        (intoReserved.IsEmpty && slotArea1 = slotArea0)
        [
            Attr.int64 "heap_bytes" (int64 size)
            Attr.int64 "slot_area_bytes" g.SlotAreaBytes
            Attr.int64 "reserved_offset" (int64 reservedOffset)
            Attr.int64 "reserved_bytes" (int64 reservedBytes)
            Attr.int "writes_into_reserved" intoReserved.Length
            Attr.bool "slot_area_unchanged" (slotArea1 = slotArea0)
        ]

/// The probe's words and mode on a fresh backend.
let private probe () : byte[] =
    use b = create ()
    (b :> IGpuBackend).LoadKernels(shaders ()) |> ignore

    match b.Fp32Denormals, b.ProbeResults with
    | Some mode, Some words ->
        printfn "fp32 denormal probe (vulkan): %s, %s" (Fp32Denormals.name mode) (Fp32Denormals.wordsText words)

        if mode <> Fp32DenormalMode.Flush then
            // R1's spike and ADR int-0001 expect flush (the SPIR-V blobs declare no denormal mode); any other mode is a
            // finding to report, not a rejection.
            printfn "FINDING: the fp32 denormal mode is %s, not flush" (Fp32Denormals.name mode)

        Expect.notEqual mode Fp32DenormalMode.Mixed "one model explains the probe"
        Expect.equal words (Fp32Denormals.expected mode) "the words of the measured mode"
        bytesOf words
    | _ -> failtest "the probe did not run"

/// The probe's telemetry, in an InMemorySink (which also checks every name), with dispatches untimed and timed. It
/// follows the GpuBackendInterface table: a gpu.dispatch interval only with TimeDispatches; gpu.dispatches counts its
/// dispatch either way; its private transfers are not IGpuHeap transfers, so no gpu.bytes_moved, and the gpu.probe
/// span carries them with the transient heap's 64 descriptor writes; gpu.buffer_memory shows that heap, then 0.
let private probeTelemetry () =
    for timed in [ false; true ] do
        let sink = InMemorySink()
        let mutable overhead = 0L

        do
            use _ = Telemetry.install (Telemetry.ofSink sink)
            use b = createWith timed
            (b :> IGpuBackend).LoadKernels(shaders ()) |> ignore
            Expect.isSome b.Fp32Denormals "the probe ran"
            let alignment = Invariant.parseI64 (detail "resource_heap_alignment")
            // The transient heap's resource heap buffer (heapRange plus one alignment) and its dummy page.
            overhead <- b.Geometry.HeapBytes + alignment + Heap.PageGranularity

        let named name =
            sink.Records |> List.filter (fun r -> r.Name = name)

        let attr (key: string) (r: TelemetryRecord) =
            r.Attrs |> List.tryFind (fun a -> a.Key = key) |> Option.map _.Value

        let label = if timed then "timed" else "untimed"
        let intervals = named GpuTelemetry.Dispatch

        Expect.equal intervals.Length (if timed then 1 else 0) (label + ": gpu.dispatch only with TimeDispatches")

        for r in intervals do
            Expect.equal (attr TelemetryKeys.Case r) (Some(AttrValue.Str "fp32_denormal_probe")) "the probe's case"

        Expect.equal
            (named GpuTelemetry.Dispatches |> List.sumBy _.Value)
            1.0
            (label + ": gpu.dispatches counts the probe's dispatch")

        Expect.isEmpty (named GpuTelemetry.BytesMoved) (label + ": no gpu.bytes_moved for the private transfers")
        let span = named "gpu.probe" |> List.exactlyOne
        // The 1 MiB table, and the page's 64 input and padding words plus 7 output words; the 7 results back.
        Expect.equal (attr "upload_bytes" span) (Some(AttrValue.Int(Heap.TableBytes + 284L))) "upload_bytes"
        Expect.equal (attr "download_bytes" span) (Some(AttrValue.Int 28L)) "download_bytes"
        Expect.equal (attr "descriptor_writes" span) (Some(AttrValue.Int(int64 Heap.SlotCount))) "descriptor_writes"

        let probeHeap =
            named "gpu.buffer_memory"
            |> List.filter (fun r -> attr "buffer" r = Some(AttrValue.Str "probe_heap"))
            |> List.map _.Value

        Expect.equal
            probeHeap
            [ float (Heap.TableBytes + Heap.PageGranularity + overhead); 0.0 ]
            "the transient heap, then 0"

        Expect.equal (named GpuTelemetry.Fp32Denormals).Length 1 "one gpu.fp32_denormals"
        Expect.isNonEmpty (named "gpu.memory_type") "the memory types chosen are recorded"

        printfn
            "probe telemetry (%s): %d gpu.dispatch, %g gpu.dispatches, %d gpu.bytes_moved, gpu.probe %A"
            label
            intervals.Length
            (named GpuTelemetry.Dispatches |> List.sumBy _.Value)
            (named GpuTelemetry.BytesMoved).Length
            (span.Attrs |> List.map (fun a -> a.Key, a.Value))

// ---- the selftests ----------------------------------------------------------------------------------------------

/// Cross-checks a selftest run with KernelBounds against the CPU expectation (CpuRef semantics in the mode).
let private checkWithKernelBounds (r: SelftestRun) =
    for c in r.Cases do
        let k =
            KernelBounds.checkExact c.Kernel 4 (ReadOnlySpan c.Output) (ReadOnlySpan c.Expected)

        Expect.isTrue k.Passed (sprintf "%s %s: KernelBounds counts %g differing elements" c.Kernel c.Case k.Measured)
        Expect.isTrue c.Passed (sprintf "%s %s: %s" c.Kernel c.Case c.Detail)

let private selftestsOn (create: unit -> VulkanBackend) : byte[] =
    use b = create ()
    let ib = b :> IGpuBackend
    let suite = SelfTests.Suite.Create ib
    ib.LoadKernels(shaders ()) |> ignore
    let mode = b.Fp32Denormals |> Option.defaultWith (fun () -> failtest "no probe")
    let r1 = suite.Run(1, mode)
    let r2 = suite.Run(2, mode)
    checkWithKernelBounds r1
    checkWithKernelBounds r2
    Expect.equal r1.Bytes r2.Bytes "two runs on one backend give the same bits"
    Expect.equal r1.Completion.Dispatches 9 "nine dispatches"

    for c in r1.Cases do
        printfn
            "selftest %-12s %-14s %s, %d elements%s"
            c.Kernel
            c.Case
            (if c.Passed then "bit-exact" else "FAILED")
            c.Elements
            (if c.Detail = "" then "" else " (" + c.Detail + ")")

    let gpu =
        r2.Completion.Timers |> List.filter (fun t -> t.Name = GpuTelemetry.Dispatch)

    Expect.equal gpu.Length 9 "every dispatch is timed"

    for t in gpu do
        Expect.isLessThan t.DurationNs 100_000_000L "every dispatch is under 100 ms"

    r1.Bytes

let private selftests () = selftestsOn create

/// The recipe's fallback heap placement (DEVICE_LOCAL, filled by a copy) runs the same selftests with the same bits.
let private copiedHeap () : byte[] =
    let bytes = selftestsOn (fun () -> createPlaced HeapPlacement.Copied)

    do
        use b = createPlaced HeapPlacement.Copied
        let ib = b :> IGpuBackend

        ib.CreateHeap(HeapLayout.ofPages [ ArenaKind.Scratch, Heap.PageGranularity ])
        |> ignore

        Expect.equal b.HeapPlacement HeapPlacement.Copied "the heap took the copy path"
        Expect.isNone (b.SlotAreaBytes()) "a copied heap is not mapped"

        Expect.exists
            b.MemoryTypeChoices
            (fun (k, _, _) -> k = BufferKind.ResourceHeapCopied)
            "the copied heap's memory type is recorded"

    let inPlace = selftestsOn create
    Expect.equal bytes inPlace "the copied heap gives the in-place heap's bits"
    bytes

let private negativeControl () =
    use b = create ()
    let ib = b :> IGpuBackend
    let suite = SelfTests.Suite.Create ib
    ib.LoadKernels(shaders ()) |> ignore
    let r = suite.Run(3, b.Fp32Denormals.Value, dispatch = false)

    for c in r.Cases do
        Expect.isFalse c.Passed (sprintf "%s %s fails without its dispatch" c.Kernel c.Case)
        Expect.equal c.NotWritten c.Elements (sprintf "%s %s: every element is unwritten" c.Kernel c.Case)

// ---- across backends: the D3D12 reference ------------------------------------------------------------------------

/// D3D12's selftest run, from the d3d12 group of this exe (it runs first, under the d3d12 mutex).
let mutable private d3d12Reference: (Fp32DenormalMode * Lrm.Gpu.D3D12.SelftestRun * ExecutionClass) option =
    None

/// The D3D12 reference: D1's selftests on D3D12 run 1, kept for the Vulkan comparison. Its telemetry is not in the
/// Vulkan run log (whose class is vulkan); D1's own gpu tier logs D3D12.
let private d3d12Selftests () =
    let candidates = Lrm.Gpu.D3D12.Adapters.enumerate ()

    match candidates |> List.tryFind _.Accepted with
    | None -> skiptestf "no accepted D3D12 adapter: %A" (candidates |> List.map (fun c -> c.Info.Name, c.Rejections))
    | Some c ->
        use b =
            Lrm.Gpu.D3D12.D3D12Backend.Create(
                c.Info,
                { DeviceOptions.For ExecutionBackend.D3D12 with
                    TimeDispatches = true
                }
            )

        let ib = b :> IGpuBackend
        let suite = Lrm.Gpu.D3D12.SelfTests.Suite.Create ib
        ib.LoadKernels(shaders ()) |> ignore
        let mode = b.Fp32Denormals.Value
        let r = suite.Run(1, mode)

        for x in r.Cases do
            Expect.isTrue x.Passed (sprintf "d3d12 %s %s: %s" x.Kernel x.Case x.Detail)

        printfn "d3d12 reference: %s, %s, %d cases" c.Info.Name (Fp32Denormals.name mode) r.Cases.Length
        d3d12Reference <- Some(mode, r, ib.ExecutionClass)

/// Vulkan's selftest words equal D3D12's wherever the two classes agree: every bit of the bit-moving selftests
/// (table_echo, args_echo, groups_echo, bf16_unpack), and every scale_f32 element when the two denormal modes are the
/// same, or else every element whose input and CPU result are normal (KernelAbiAndCatalog: bit-exact claims on fp32
/// denormals are class-dependent). The two backends' inputs, layouts and tables are the same bytes (cpu tier).
let private acrossBackends () =
    match d3d12Reference with
    | None ->
        let backends = Env.tryGet "LRM_BACKENDS" |> Option.defaultValue "all"
        skiptestf "the d3d12 reference did not run (LRM_BACKENDS=%s or no D3D12 adapter)" backends
    | Some(dMode, dRun, dClass) ->
        use b = create ()
        let ib = b :> IGpuBackend
        let suite = SelfTests.Suite.Create ib
        ib.LoadKernels(shaders ()) |> ignore
        let vMode = b.Fp32Denormals.Value
        let vRun = suite.Run(1, vMode)
        let vClass = ib.ExecutionClass
        Expect.isFalse (ExecutionClass.same vClass dClass) "two classes"

        Expect.equal
            (List.map (fun (c: SelftestCase) -> c.Kernel, c.Case) vRun.Cases)
            (dRun.Cases |> List.map (fun c -> c.Kernel, c.Case))
            "the same cases"

        let mutable compared = 0
        let mutable excluded = 0
        let mutable differing = 0

        for v, d in List.zip vRun.Cases dRun.Cases do
            let vw = wordsOf v.Output
            let dw = wordsOf d.Output
            Expect.equal vw.Length dw.Length (sprintf "%s %s: the same element count" v.Kernel v.Case)

            let inputs =
                match v.Case with
                | "x_to_y" -> Some SelfTests.scaleInputs
                | "in_place" -> Some SelfTests.inPlaceInputs
                | _ -> None

            let mutable caseExcluded = 0
            let mutable caseDiffering = 0

            for i in 0 .. vw.Length - 1 do
                let agree =
                    match inputs with
                    | Some xs when vMode <> dMode ->
                        not (isDenormal xs[i])
                        && not (isDenormal (SelfTests.scaleBits SelfTests.ScaleA xs[i]))
                    | _ -> true

                if agree then
                    compared <- compared + 1

                    if vw[i] <> dw[i] then
                        differing <- differing + 1
                        caseDiffering <- caseDiffering + 1
                else
                    excluded <- excluded + 1
                    caseExcluded <- caseExcluded + 1

            printfn
                "  %-12s %-14s %d elements, %d compared, %d excluded (denormal input or result), %d differ"
                v.Kernel
                v.Case
                vw.Length
                (vw.Length - caseExcluded)
                caseExcluded
                caseDiffering

            Telemetry.check
                "gpu.selftest.across_backends"
                (caseDiffering = 0)
                [
                    Attr.str TelemetryKeys.Kernel v.Kernel
                    Attr.str TelemetryKeys.Case v.Case
                    Attr.str "reference" "d3d12"
                    Attr.str "mode" (Fp32Denormals.name vMode)
                    Attr.str "reference_mode" (Fp32Denormals.name dMode)
                    Attr.int "elements" vw.Length
                    Attr.int "excluded" caseExcluded
                    Attr.int "differing" caseDiffering
                ]

        printfn
            "vulkan (%s) against d3d12 (%s): %d elements compared, %d excluded, %d differ"
            (Fp32Denormals.name vMode)
            (Fp32Denormals.name dMode)
            compared
            excluded
            differing

        Expect.equal differing 0 "Vulkan's selftest words equal D3D12's where the classes agree"

        if vMode <> dMode then
            Expect.isGreaterThan excluded 0 "the modes differ, so some scale_f32 elements are class-only"

// ---- copies, views, timers --------------------------------------------------------------------------------------

/// dispatch -> copy to another page -> dispatch -> copy within a page -> dispatch -> copy within a page, in one
/// command list, every intermediate checked against the CPU.
let private copies () : byte[] =
    let pageBytes = Heap.PageGranularity
    let n = 1024
    let bytes = int64 (4 * n)

    let layout =
        HeapLayout.ofPages [ ArenaKind.Activations, pageBytes; ArenaKind.Scratch, pageBytes ]

    let t = TensorTable layout

    let add name slot (offset: int64) flags =
        t.Add(name, TensorRow.create slot offset DType.F32 [ int64 n ] flags (TensorRow.debugIdOf name))

    let a = add "a" 1 0L TensorFlags.ReadOnly
    let tb = add "b" 1 8192L TensorFlags.None
    let tc = add "c" 2 0L TensorFlags.None
    let td = add "d" 2 8192L TensorFlags.None
    let te = add "e" 2 16384L TensorFlags.None
    let tf = add "f" 1 16384L TensorFlags.None
    let tg = add "g" 1 24576L TensorFlags.None
    let table = t.Seal()
    use backend = create ()
    let b = backend :> IGpuBackend
    let heap = b.CreateHeap layout
    heap.WriteTable table
    b.LoadKernels(shaders ()) |> ignore
    let x = normals "d1.copies" n
    heap.Upload(1, 0L, ReadOnlySpan(bytesOf x))

    for slot, offset in [ 1, 8192L; 2, 0L; 2, 8192L; 2, 16384L; 1, 16384L; 1, 24576L ] do
        heap.Upload(slot, offset, ReadOnlySpan(bytesOf (Array.create n 0x7FA5A5A5u)))

    let c = b.Begin()

    let scale x y a =
        Kernels.ScaleF32.dispatch c { X = x; Y = y; N = uint32 n; A = a }

    scale a tb 2.0f
    c.Copy(range 1 8192L bytes, range 2 0L bytes)
    scale tc td 3.0f
    c.Copy(range 2 8192L bytes, range 2 16384L bytes)
    scale te tf 0.5f
    c.Copy(range 1 16384L bytes, range 1 24576L bytes)
    b.Wait(b.Submit c) |> ignore

    let read slot offset =
        let out = Array.zeroCreate<byte>(int bytes)
        heap.Download(slot, offset, Span out)
        wordsOf out

    let eb = x |> Array.map (fun w -> bits (f32 w * 2.0f))
    let ed = eb |> Array.map (fun w -> bits (f32 w * 3.0f))
    let ef = ed |> Array.map (fun w -> bits (f32 w * 0.5f))

    for name, slot, offset, expected in
        [
            "b", 1, 8192L, eb
            "c (copied across pages)", 2, 0L, eb
            "d", 2, 8192L, ed
            "e (copied within a page)", 2, 16384L, ed
            "f", 1, 16384L, ef
            "g (copied within a page)", 1, 24576L, ef
        ] do
        Expect.equal (read slot offset) expected name

    ignore tg
    Array.concat [ bytesOf (read 1 24576L); bytesOf (read 2 16384L) ]

/// A whole 1 GiB page is one storage-buffer descriptor: scale_f32 on tokens at its start, past 512 MiB and at its end.
let private wholePage () : byte[] =
    let layout = HeapLayout.ofPages [ ArenaKind.Activations, Heap.MaxPageBytes ]
    let n = 4096
    let t = TensorTable layout

    let offsets = [ 0L; (512L <<< 20) + 65536L; Heap.MaxPageBytes - 65536L ]

    let pairs =
        offsets
        |> List.mapi (fun i o ->
            let x =
                t.Add(sprintf "x%d" i, TensorRow.create 1 o DType.F32 [ int64 n ] TensorFlags.ReadOnly 0u)

            let y =
                t.Add(sprintf "y%d" i, TensorRow.create 1 (o + 32768L) DType.F32 [ int64 n ] TensorFlags.None 0u)

            o, x, y)

    let table = t.Seal()
    use backend = create ()
    let b = backend :> IGpuBackend
    let heap = b.CreateHeap layout
    heap.WriteTable table
    b.LoadKernels(shaders ()) |> ignore
    let input = normals "d1.whole_page" n
    let c = b.Begin()

    for o, x, y in pairs do
        heap.Upload(1, o, ReadOnlySpan(bytesOf input))
        heap.Upload(1, o + 32768L, ReadOnlySpan(bytesOf (Array.create n 0x7FA5A5A5u)))

        Kernels.ScaleF32.dispatch
            c
            {
                X = x
                Y = y
                N = uint32 n
                A = -2.0f
            }

    b.Wait(b.Submit c) |> ignore
    let expected = input |> Array.map (fun w -> bits (f32 w * -2.0f))

    [|
        for o, _, _ in pairs do
            let out = Array.zeroCreate<byte>(4 * n)
            heap.Download(1, o + 32768L, Span out)
            Expect.equal (wordsOf out) expected (sprintf "the token at byte %d of the 1 GiB page" o)
            yield! out
    |]

let private timers () =
    use backend = create ()
    let b = backend :> IGpuBackend
    let suite = SelfTests.Suite.Create b
    b.LoadKernels(shaders ()) |> ignore
    let tok = tokenOf suite.Table
    let c = b.Begin()
    let outer = c.BeginTimer("gpu.v1.outer", [ Attr.str TelemetryKeys.Case "timers" ])

    Kernels.GroupsEcho.dispatch
        c
        {
            Y = tok "groups_echo.y.folded"
            Groups = uint32 SelfTests.FoldedGroups
        }

    let inner = c.BeginTimer("gpu.v1.inner", [])

    Kernels.ScaleF32.dispatch
        c
        {
            X = tok "scale.x"
            Y = tok "scale.y"
            N = uint32 SelfTests.ScaleN
            A = 1.0f
        }

    c.EndTimer inner
    c.EndTimer outer
    let s1 = b.Submit c
    let s2 = b.Submit(b.Begin())
    let s3 = b.Submit(b.Begin())
    Expect.equal [ s1.Value; s2.Value; s3.Value ] [ 1UL; 2UL; 3UL ] "submission values 1, 2, 3"
    let w = b.Wait s1
    Expect.equal (b.Wait s1) w "waiting again returns the same completion"
    Expect.equal w.Dispatches 2 "two dispatches"

    Expect.equal
        (w.Timers |> List.map _.Name)
        [ "gpu.v1.outer"; "gpu.v1.inner"; "gpu.dispatch"; "gpu.dispatch" ]
        "order"

    let o = w.Timers[0]
    let i = w.Timers[1]
    Expect.isGreaterThan o.DurationNs 0L "the outer timer has a duration"
    Expect.isGreaterThanOrEqual i.GpuStartNs o.GpuStartNs "inner starts inside outer"
    Expect.isLessThanOrEqual (i.GpuStartNs + i.DurationNs) (o.GpuStartNs + o.DurationNs) "and ends inside"
    Expect.equal (b.Wait s3).Dispatches 0 "an empty list"
    Expect.equal (b.Wait s2).Timers [] "no timers"
    Expect.equal backend.PendingSubmissions 0 "every submission has retired"
    Expect.equal (b.Wait s1) w "a retired submission still returns the same completion"

// ---- the 1.5 GB round trip and the soak -------------------------------------------------------------------------

/// The Qwen3-0.6B weights file through two Weights pages and back, hashed on the way out and checked against
/// SHA256SUMS: UploadFrom from the file stream, one call per page, and Download in multi-chunk calls at offsets that
/// are not chunk-aligned.
let private roundTrip () : byte[] =
    let dir = Path.Combine(Env.dataRoot (), "models", "Qwen3-0.6B")
    let path = Path.Combine(dir, "model.safetensors")

    if not (File.Exists path) then
        skiptest (sprintf "the Qwen3-0.6B weights are not at %s" path)

    let expected =
        File.ReadAllLines(Path.Combine(dir, "SHA256SUMS"))
        |> Array.pick (fun l ->
            let parts = l.Split([| ' '; '*' |], StringSplitOptions.RemoveEmptyEntries)

            if parts.Length = 2 && parts[1] = "model.safetensors" then
                Some parts[0]
            else
                None)

    let length = FileInfo(path).Length
    let first = min length Heap.MaxPageBytes
    let rest = length - first

    let layout =
        HeapLayout.ofPages
            [
                ArenaKind.Weights, Heap.MaxPageBytes
                ArenaKind.Weights, Heap.alignUp Heap.PageGranularity rest
            ]

    use backend = create ()
    let b = backend :> IGpuBackend
    let heap = b.CreateHeap layout
    let clock = Stopwatch.StartNew()

    do
        use fs =
            new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 <<< 20, FileOptions.SequentialScan)

        heap.UploadFrom(1, 0L, fs, first)
        heap.UploadFrom(2, 0L, fs, rest)

    let up = clock.Elapsed
    clock.Restart()
    use sha = IncrementalHash.CreateHash HashAlgorithmName.SHA256
    // Downloads of 200 MiB + 12 bytes: each call is four chunks (three of 64 MiB, then 8 MiB + 12 bytes), so the next
    // chunk's copy runs while the previous one drains, and every call after a page's first starts at an offset that is
    // not a multiple of the chunk size.
    let piece = 3L * Transfer.ChunkBytes + (8L <<< 20) + 12L
    let buffer = Array.zeroCreate<byte>(int piece)
    let mutable calls = 0

    for slot, bytes in [ 1, first; 2, rest ] do
        let mutable at = 0L

        while at < bytes do
            let n = min piece (bytes - at)
            heap.Download(slot, at, Span(buffer, 0, int n))
            sha.AppendData(buffer, 0, int n)
            at <- at + n
            calls <- calls + 1

    Expect.isGreaterThan calls 4 "several multi-chunk downloads"

    let down = clock.Elapsed
    let digest = sha.GetHashAndReset()
    Expect.equal (Convert.ToHexStringLower digest) expected "the weights come back with their SHA-256"
    let gb (t: TimeSpan) = float length / t.TotalSeconds / 1e9

    printfn
        "round trip: %d bytes (2 pages), upload %.2f s (%.2f GB/s), download %.2f s (%.2f GB/s)"
        length
        up.TotalSeconds
        (gb up)
        down.TotalSeconds
        (gb down)

    Telemetry.gauge "gpu.transfer.rate" (gb up * 1e9) "B/s" [ Attr.str "direction" "upload"; Attr.int64 "bytes" length ]

    Telemetry.gauge
        "gpu.transfer.rate"
        (gb down * 1e9)
        "B/s"
        [ Attr.str "direction" "download"; Attr.int64 "bytes" length ]

    digest

let private median (xs: float seq) =
    let a = Array.ofSeq xs |> Array.sort
    if a.Length = 0 then nan else a[a.Length / 2]

/// How a soak times its work.
type private SoakMode =
    /// DeviceOptions.TimeDispatches: every dispatch is a gpu.dispatch timing.
    | TimedDispatches
    /// Dispatches untimed; one gpu.v1.soak timer brackets each submission.
    | SubmissionTimer
    /// No timer at all: what an untimed run keeps per submission.
    | NoTimers

    member m.Name =
        match m with
        | TimedDispatches -> "timed_dispatches"
        | SubmissionTimer -> "submission_timer"
        | NoTimers -> "no_timers"

/// What the backend may keep per dispatch once GC has run, by mode (D1's bounds): a timed dispatch keeps its 24-byte
/// timing and a submission timer its TimerResult; an untimed submission keeps its 4-byte dispatch count.
let private managedBound (mode: SoakMode) =
    match mode with
    | TimedDispatches -> 48.0
    | SubmissionTimer
    | NoTimers -> 16.0

/// The Theil-Sen slope of (x, y) points: the median of the slopes between every pair.
let private slope (points: (float * float) list) =
    let a = Array.ofList points

    median
        [
            for i in 0 .. a.Length - 2 do
                for j in i + 1 .. a.Length - 1 do
                    let (x0, y0), (x1, y1) = a[i], a[j]
                    yield (y1 - y0) / (x1 - x0)
        ]

/// 10,000 dependent scale_f32 dispatches (a = -1, ping-pong between two tokens with a barrier between each): 100
/// submissions of 100, or 1,000 of 10 without timers. The result is exact, and from the first tenth of the submissions
/// to the end host time, GPU time (per dispatch, or per submission from its timer), frames, live Vulkan objects,
/// private memory and managed memory stay flat. No submission is pending after the last Wait.
let private soak (mode: SoakMode) () : byte[] =
    let n = 4096
    let layout = HeapLayout.ofPages [ ArenaKind.Activations, Heap.PageGranularity ]
    let t = TensorTable layout
    let x = t.Add("x", TensorRow.create 1 0L DType.F32 [ int64 n ] TensorFlags.None 0u)

    let y =
        t.Add("y", TensorRow.create 1 16384L DType.F32 [ int64 n ] TensorFlags.None 0u)

    let table = t.Seal()
    use backend = createWith (mode = TimedDispatches)
    let b = backend :> IGpuBackend
    let heap = b.CreateHeap layout
    heap.WriteTable table
    b.LoadKernels(shaders ()) |> ignore
    let x0 = normals "d1.soak" n
    heap.Upload(1, 0L, ReadOnlySpan(bytesOf x0))
    heap.Upload(1, 16384L, ReadOnlySpan(bytesOf (Array.create n 0x7FA5A5A5u)))
    let dispatches = 10_000
    let perSubmission = if mode = NoTimers then 10 else 100
    let submissions = dispatches / perSubmission
    let warm = submissions / 10
    let host = ResizeArray<float>(submissions)
    let gpu = ResizeArray<float>(dispatches + submissions)
    let managed = ResizeArray<float * float>(16)
    let mutable framesAtWarm = 0
    let mutable objectsAtWarm = 0
    let proc = Process.GetCurrentProcess()
    let mutable privateAtWarm = 0L

    for s in 1..submissions do
        let c = b.Begin()

        let timer =
            if mode = SubmissionTimer then
                Some(c.BeginTimer("gpu.v1.soak", []))
            else
                None

        for k in 0 .. perSubmission - 1 do
            let src, dst = if k % 2 = 0 then x, y else y, x

            Kernels.ScaleF32.dispatch
                c
                {
                    X = src
                    Y = dst
                    N = uint32 n
                    A = -1.0f
                }

            c.Barrier()

        timer |> Option.iter c.EndTimer
        let clock = Stopwatch.StartNew()

        let completion =
            use _ =
                Telemetry.scope [ Attr.str TelemetryKeys.Case "soak"; Attr.int "iteration" s ]

            b.Wait(b.Submit c)

        host.Add clock.Elapsed.TotalMilliseconds

        for r in completion.Timers do
            gpu.Add(float r.DurationNs)

        if s % warm = 0 then
            managed.Add(float (s * perSubmission), float (GC.GetTotalMemory true))

        if s = warm then
            framesAtWarm <- backend.FramesCreated
            objectsAtWarm <- VulkanObjects.count ()
            proc.Refresh()
            privateAtWarm <- proc.PrivateMemorySize64

    proc.Refresh()
    let privateGrowth = proc.PrivateMemorySize64 - privateAtWarm
    let samples = List.ofSeq managed
    let managedGrowth = int64 (snd (List.last samples) - snd samples.Head)
    let managedPerDispatch = slope samples
    let managedPerSubmission = managedPerDispatch * float perSubmission
    let bound = managedBound mode
    let hostFirst = median (host |> Seq.skip 1 |> Seq.take warm)
    let hostLast = median (host |> Seq.skip (submissions - warm))
    let perGpu = gpu.Count / submissions
    let gpuWindow = warm * perGpu
    let gpuFirst = median (gpu |> Seq.skip perGpu |> Seq.truncate gpuWindow)
    let gpuLast = median (gpu |> Seq.skip (gpu.Count - gpuWindow))
    // Absolute slack: 2 us per timed dispatch, 20 us per timed submission of 100 dispatches.
    let gpuSlack = if mode = TimedDispatches then 2000.0 else 20000.0
    let gpuFlat = mode = NoTimers || gpuLast <= 1.5 * gpuFirst + gpuSlack

    printfn
        "soak (%s): %d dispatches in %d submissions; host per submission %.3f -> %.3f ms; %s; frames %d -> %d; Vulkan objects %d -> %d; private bytes %+d; managed %+d, slope %.2f B per dispatch (bound %.0f) or %.1f B per submission, samples %s; pending %d"
        mode.Name
        dispatches
        submissions
        hostFirst
        hostLast
        (match mode with
         | NoTimers -> "GPU not timed"
         | TimedDispatches -> sprintf "GPU per dispatch %.1f -> %.1f us" (gpuFirst / 1000.0) (gpuLast / 1000.0)
         | SubmissionTimer -> sprintf "GPU per submission %.1f -> %.1f us" (gpuFirst / 1000.0) (gpuLast / 1000.0))
        framesAtWarm
        backend.FramesCreated
        objectsAtWarm
        (VulkanObjects.count ())
        privateGrowth
        managedGrowth
        managedPerDispatch
        bound
        managedPerSubmission
        (samples
         |> List.map (fun (_, m) -> sprintf "%+d" (int64 m - int64 (snd samples.Head)))
         |> String.concat " ")
        backend.PendingSubmissions

    let read offset =
        let out = Array.zeroCreate<byte>(4 * n)
        heap.Download(1, offset, Span out)
        wordsOf out

    let fx = read 0L
    let fy = read 16384L
    Expect.equal fx x0 "an even number of negations gives the input back"
    Expect.equal fy (x0 |> Array.map (fun w -> w ^^^ 0x80000000u)) "y is its negation"

    let flat =
        backend.FramesCreated = framesAtWarm
        && VulkanObjects.count () = objectsAtWarm
        && hostLast <= 2.0 * hostFirst + 0.5
        && gpuFlat
        && privateGrowth < (64L <<< 20)
        && managedPerDispatch <= bound
        && backend.PendingSubmissions = 0

    Telemetry.check
        "gpu.soak"
        flat
        [
            Attr.str "mode" mode.Name
            Attr.int "dispatches" dispatches
            Attr.int "submissions" submissions
            Attr.float "host_ms_first" hostFirst
            Attr.float "host_ms_last" hostLast
            if mode <> NoTimers then
                Attr.float "gpu_ns_first" gpuFirst
                Attr.float "gpu_ns_last" gpuLast
            Attr.int "frames_first" framesAtWarm
            Attr.int "frames_last" backend.FramesCreated
            Attr.int "objects_first" objectsAtWarm
            Attr.int "objects_last" (VulkanObjects.count ())
            Attr.int64 "private_growth" privateGrowth
            Attr.int64 "managed_growth" managedGrowth
            Attr.float "managed_bytes_per_dispatch" managedPerDispatch
            Attr.float "managed_bytes_per_submission" managedPerSubmission
            Attr.float "managed_bound" bound
            Attr.int "pending" backend.PendingSubmissions
        ]

    Expect.equal backend.FramesCreated framesAtWarm "no new command pools after warm-up"
    Expect.equal (VulkanObjects.count ()) objectsAtWarm "no Vulkan object leaks"
    Expect.isLessThanOrEqual hostLast (2.0 * hostFirst + 0.5) "host time per submission is flat"

    if mode <> NoTimers then
        Expect.isLessThanOrEqual gpuLast (1.5 * gpuFirst + gpuSlack) "GPU time is flat"

    Expect.isLessThan privateGrowth (64L <<< 20) "private memory is flat"

    Expect.isLessThanOrEqual
        managedPerDispatch
        bound
        (sprintf "managed memory kept per dispatch (%s), %d bytes in all" mode.Name managedGrowth)

    Expect.equal backend.PendingSubmissions 0 "no pending submission after the last Wait"
    Array.append (bytesOf fx) (bytesOf fy)

let private leaks () =
    let before = VulkanObjects.count ()

    do
        use b = create ()
        let ib = b :> IGpuBackend
        let suite = SelfTests.Suite.Create ib
        ib.LoadKernels(shaders ()) |> ignore
        let r = suite.Run(4, b.Fp32Denormals.Value)
        Expect.isTrue r.Passed "selftests pass"
        Expect.isGreaterThan (VulkanObjects.count ()) before "the backend holds Vulkan objects"

    Expect.equal (VulkanObjects.count ()) before "every Vulkan object is destroyed by Dispose"

let private hello () =
    let lines = ResizeArray<string>()
    let code = Hello.runWith lines.Add (shaders ()) [||]

    for l in lines do
        printfn "%s" l

    Expect.equal code 0 "hello-vulkan passes"
    Expect.exists lines (fun l -> l.StartsWith "class id") "prints the ExecutionClass id"
    Expect.exists lines (fun l -> l.StartsWith "PASS") "PASS"

/// The D3D12 reference for the cross-backend comparison (the d3d12 group; it runs before the vulkan group).
let d3d12Tests =
    testSequenced
    <| testList
        "vulkan.gpu.d3d12-reference"
        [
            testCase "d3d12 reference: the five selftests on D3D12, for the comparison with Vulkan" d3d12Selftests
        ]

/// Sequenced: one GPU workload at a time; the leak test compares the process-wide live-object count.
let tests =
    testSequenced
    <| testList
        "vulkan.gpu"
        [
            gpuTest "the device check accepts the GPU and the backend opens it" deviceCheck
            gpuTest "interface rules on hardware give RecordingBackend's outcomes" interfaceRules
            gpuTest
                "after Dispose every call raises ObjectDisposedException; a bad SPIR-V blob is an argument error"
                disposedAndBadSpirv
            gpuTest
                "the resource heap writes 64 descriptors at CreateHeap and the probe's heap 64, then none; the reserved range is never written"
                heapWrites
            gpuTestTwice "the fp32 denormal probe is bit-identical twice" probe
            gpuTest "the probe's telemetry follows the interface table, timed and untimed" probeTelemetry
            gpuTestTwice "the five selftests are bit-exact against the CPU and bit-identical twice" selftests
            gpuTest "the negative control: without dispatches every selftest case is unwritten" negativeControl
            gpuTest "the selftest words equal D3D12's where the two classes agree" acrossBackends
            gpuTestTwice "the copied-heap fallback gives the in-place heap's selftest bits, twice" copiedHeap
            gpuTestTwice "copies across and within pages are state-correct, bit-identical twice" copies
            gpuTestTwice "a whole 1 GiB page is one descriptor, bit-identical twice" wholePage
            gpuTest "timers nest, submission values count from 1, waits repeat" timers
            gpuTestTwice "the 1.5 GB weights file round-trips with its SHA-256, twice" roundTrip
            gpuTestTwice "a 10,000-dispatch soak with timed dispatches is exact and flat, twice" (soak TimedDispatches)
            gpuTestTwice
                "a 10,000-dispatch soak with a timer per submission is exact and flat, twice"
                (soak SubmissionTimer)
            gpuTestTwice "a 10,000-dispatch soak without timers is exact and flat, twice" (soak NoTimers)
            gpuTest "every Vulkan object is destroyed" leaks
            gpuTest "hello-vulkan prints the device and the ExecutionClass id" hello
        ]
