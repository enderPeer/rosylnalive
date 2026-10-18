/// gpu tier (the d3d12 mutex: WARP is Direct3D 12, and Lrm.Testing has no d3d12-warp backend): in-box WARP through
/// EnumWarpAdapter and D1's backend. The adapter and its capabilities (shader model, binding tier, the directly indexed
/// heap); the fp32 denormal probe of D1's LoadKernels and the int-0001 control without -denorm; D1's five selftests
/// bit-identical twice and against the CPU, with KernelBounds and the 100 ms budget; their negative control; the same
/// selftests on the RTX 4080 for comparison; copies; descriptor writes; a scale_f32 throughput series for the budget;
/// COM references; then E1's conformance harness (every case, and its negative control) on d3d12-warp.
module Lrm.Spikes.D3D12.GpuTests

open System
open System.Diagnostics
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12
open Lrm.Gpu.D3D12.Interop
open Lrm.Spikes.D3D12.Common
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

let private normals (seed: string) (n: int) =
    let rng = DetRng(SelfTests.Seed, DetRng.streamOf seed)
    Array.init n (fun _ -> (0x3F800000u + ((rng.NextU32() % 8u) <<< 23)) ||| (rng.NextU32() >>> 9))

let private f32 (w: uint32) = BitConverter.UInt32BitsToSingle w
let private bits (x: float32) = BitConverter.SingleToUInt32Bits x

// ---- the adapter -------------------------------------------------------------------------------------------------

/// EnumWarpAdapter gives the Microsoft Basic Render Driver; a device on it reports shader model 6.6 or later and
/// resource binding tier 3, so the d3d12-warp verdict accepts it; D1's enumeration lists the same LUID as a software
/// adapter, which the d3d12 device check rejects on purpose; D1's backend opens it and stays healthy.
let private adapter () =
    recordDevice ()
    let r = WarpAdapter.probe ()
    let a = r.Adapter

    printfn "EnumWarpAdapter: %s, vendor %04x, device %04x, LUID %s, UMD %s" a.Description a.VendorId a.DeviceId (Adapters.luidText a.Luid) a.DriverVersion

    for f, v in r.Files do
        printfn "  %-14s %s" f v

    printfn "  OS %s (%s)" r.Os r.OsEdition
    Expect.isTrue a.Software "WARP is a software adapter"
    Expect.equal a.VendorId WarpAdapter.VendorMicrosoft "Microsoft's vendor id"

    match r.Capabilities with
    | Error e -> failtestf "no D3D12 device on WARP: %s" e
    | Ok c ->
        printfn
            "  feature level %s, shader model %s, binding tier %d, heap tier %d, root signature 1.%d, waves %d..%d, runtime SDK %s, enhanced barriers %b"
            (DeviceCheck.formatFeatureLevel c.MaxFeatureLevel)
            (DeviceCheck.formatShaderModel c.HighestShaderModel)
            (int c.ResourceBindingTier)
            (int c.ResourceHeapTier)
            (int c.HighestRootSignature - 1)
            c.WaveLaneCountMin
            c.WaveLaneCountMax
            (c.RuntimeSdkVersion |> Option.map string |> Option.defaultValue "unknown")
            c.EnhancedBarriers

        Expect.isGreaterThanOrEqual
            (int c.HighestShaderModel)
            (int DeviceCheck.RequiredShaderModel)
            "shader model 6.6 or later (ResourceDescriptorHeap)"

        Expect.isGreaterThanOrEqual
            (int c.ResourceBindingTier)
            (int DeviceCheck.RequiredBindingTier)
            "resource binding tier 3"

    Expect.isEmpty r.Rejections "the d3d12-warp verdict accepts WARP"
    Expect.isGreaterThanOrEqual a.Index 0 "EnumAdapterByGpuPreference lists the WARP adapter"

    let d3d12 = Adapters.enumerate () |> List.find (fun c -> Adapters.luidOf c.Info = Some a.Luid)
    Expect.isFalse d3d12.Accepted "the d3d12 device check rejects the same adapter (d3d12-warp is its own class)"
    Expect.equal d3d12.Info.Index a.Index "the same position in D1's enumeration"

    use b = create ()
    let inner = b.Inner
    Expect.equal inner.DeviceRemovedReason HResult.S_OK "a healthy device"
    Expect.isGreaterThan inner.TimestampFrequency 0UL "timestamp frequency"
    let info = (b :> IGpuBackend).Info
    Expect.equal info.Backend ExecutionBackend.D3D12Warp "d3d12-warp"

    printfn
        "D1's backend on WARP: timestamps at %d Hz, DRED %s, debug layer %s"
        inner.TimestampFrequency
        (if inner.DredEnabled then "forced on" else "unavailable")
        (if inner.DebugLayer then "on" else "not installed")

// ---- the fp32 denormal probe -------------------------------------------------------------------------------------

/// D1's probe (inside LoadKernels) on the embedded set's scale_f32, built with -denorm preserve: the mode and words.
let private probe () : byte[] =
    use b = create ()
    let set = (b :> IGpuBackend).LoadKernels(shaders ())
    let cls = (b :> IGpuBackend).ExecutionClass
    Expect.equal cls.Backend ExecutionBackend.D3D12Warp "the class is d3d12-warp"
    Expect.equal cls.Arch (GpuClass.arch ExecutionBackend.D3D12Warp) "Arch keeps the CPU feature flags"
    Expect.equal cls.ShaderSetHash (Some set.ShaderSetHash) "the DXIL ShaderSetHash"

    match b.Inner.Fp32Denormals, b.Inner.ProbeResults with
    | Some mode, Some words ->
        printfn
            "fp32 denormal probe (d3d12-warp, -denorm preserve): %s, %s; class %s (%s)"
            (Fp32Denormals.name mode)
            (Fp32Denormals.wordsText words)
            (string (ExecutionClass.id cls))
            (ExecutionClass.describe cls)

        if mode <> Fp32DenormalMode.Preserve then
            printfn "FINDING: WARP's fp32 denormal mode is %s with -denorm preserve" (Fp32Denormals.name mode)

        bytesOf words
    | _ -> failtest "the probe did not run"

/// The same probe with scale_f32 built without -denorm (ADR int-0001's control): what WARP does when the blob
/// declares no mode.
let private control () : byte[] =
    match Control.controlSet.Value with
    | Error why -> skiptest why
    | Ok(flags, set) ->
        use _ =
            Telemetry.scope [ Attr.str "blob" "control, no -denorm"; Attr.str "dxil_flags" flags ]

        use b = create ()
        (b :> IGpuBackend).LoadKernels set |> ignore

        match b.Inner.Fp32Denormals, b.Inner.ProbeResults with
        | Some mode, Some words ->
            printfn
                "fp32 denormal probe (d3d12-warp, control without -denorm): %s, %s"
                (Fp32Denormals.name mode)
                (Fp32Denormals.wordsText words)

            bytesOf words
        | _ -> failtest "the probe did not run"

// ---- D1's selftests ----------------------------------------------------------------------------------------------

let private checkWithKernelBounds (r: SelftestRun) =
    for c in r.Cases do
        let k =
            KernelBounds.checkExact c.Kernel 4 (ReadOnlySpan c.Output) (ReadOnlySpan c.Expected)

        Expect.isTrue k.Passed (sprintf "%s %s: KernelBounds counts %g differing elements" c.Kernel c.Case k.Measured)
        Expect.isTrue c.Passed (sprintf "%s %s: %s" c.Kernel c.Case c.Detail)

/// The selftest bytes of the last runs, per backend name, for the comparison with hardware.
let private lastRuns = Collections.Concurrent.ConcurrentDictionary<string, SelftestRun>()

/// D1's five selftests (nine cases) on WARP, in WARP's measured mode.
let private selftests () : byte[] =
    use b = create ()
    let mode () = b.Inner.Fp32Denormals |> Option.defaultWith (fun () -> failtest "no probe")
    let ib = b :> IGpuBackend
    // Suite.Create makes the heap before LoadKernels, as D1's tests and hello do; the mode is known after it.
    let suite = SelfTests.Suite.Create ib
    ib.LoadKernels(shaders ()) |> ignore
    let m = mode ()
    let r1 = suite.Run(1, m)
    let r2 = suite.Run(2, m)
    checkWithKernelBounds r1
    checkWithKernelBounds r2
    Expect.equal r1.Bytes r2.Bytes "two runs on WARP give the same bits"
    Expect.equal r1.Completion.Dispatches 9 "nine dispatches"
    lastRuns["d3d12-warp"] <- r1

    for c in r1.Cases do
        printfn
            "d3d12-warp selftest %-12s %-14s %s, %d elements%s"
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

    printfn
        "d3d12-warp: %d dispatches, %.1f us of GPU time (run 2), slowest %.1f us"
        gpu.Length
        (float (gpu |> List.sumBy _.DurationNs) / 1000.0)
        (float (gpu |> List.map _.DurationNs |> List.max) / 1000.0)

    r1.Bytes

/// Without dispatches every case keeps its poison.
let private negativeControl () =
    use b = create ()
    let ib = b :> IGpuBackend
    let suite = SelfTests.Suite.Create ib
    ib.LoadKernels(shaders ()) |> ignore
    let r = suite.Run(3, Fp32DenormalMode.Preserve, dispatch = false)

    for c in r.Cases do
        Expect.isFalse c.Passed (sprintf "%s %s fails without its dispatch" c.Kernel c.Case)
        Expect.equal c.NotWritten c.Elements (sprintf "%s %s: every element is unwritten" c.Kernel c.Case)

/// The same selftests on the first accepted hardware adapter (the RTX 4080 here), and a byte comparison per case with
/// WARP's. Different classes, so no bit-identity claim: the comparison is a measurement. Skipped without hardware (CI).
let private versusHardware () =
    match hardware () with
    | None -> skiptest "no hardware d3d12 adapter is accepted (a CI runner)"
    | Some info ->
        let hw =
            use b =
                D3D12Backend.Create(
                    info,
                    { DeviceOptions.For ExecutionBackend.D3D12 with
                        TimeDispatches = true
                    }
                )

            let ib = b :> IGpuBackend
            let suite = SelfTests.Suite.Create ib
            ib.LoadKernels(shaders ()) |> ignore
            let r = suite.Run(1, b.Fp32Denormals.Value)
            checkWithKernelBounds r
            r

        let warp =
            match lastRuns.TryGetValue "d3d12-warp" with
            | true, r -> r
            | _ ->
                use b = create ()
                let ib = b :> IGpuBackend
                let suite = SelfTests.Suite.Create ib
                ib.LoadKernels(shaders ()) |> ignore
                suite.Run(1, b.Inner.Fp32Denormals.Value)

        Expect.equal warp.Mode hw.Mode "the same fp32 denormal mode"

        for h, w in List.zip hw.Cases warp.Cases do
            let hw32 = wordsOf h.Output
            let w32 = wordsOf w.Output

            let differing =
                Array.zip hw32 w32 |> Array.filter (fun (a, b) -> a <> b) |> Array.length

            let nanOnly =
                Array.zip hw32 w32
                |> Array.forall (fun (a, b) -> a = b || (Single.IsNaN(f32 a) && Single.IsNaN(f32 b)))

            // Each backend's words against the CPU's expected words, every bit (NaN payloads included).
            let unequal (out: uint32[]) (c: SelftestCase) =
                Array.zip out (wordsOf c.Expected) |> Array.filter (fun (a, b) -> a <> b) |> Array.length

            let warpVsCpu = unequal w32 w
            let hwVsCpu = unequal hw32 h

            Telemetry.check
                "gpu.warp.versus_hardware"
                (differing = 0)
                [
                    Attr.str TelemetryKeys.Kernel h.Kernel
                    Attr.str TelemetryKeys.Case h.Case
                    Attr.int "elements" h.Elements
                    Attr.int "differing" differing
                    Attr.bool "nan_only" nanOnly
                    Attr.str "hardware" info.Name
                    Attr.str "hardware_nan_words" (h.NanWords |> List.map (sprintf "%08x") |> String.concat " ")
                    Attr.str "warp_nan_words" (w.NanWords |> List.map (sprintf "%08x") |> String.concat " ")
                    Attr.int "warp_vs_cpu_bits" warpVsCpu
                    Attr.int "hardware_vs_cpu_bits" hwVsCpu
                ]

            printfn
                "versus %s: %-12s %-14s %s; words unequal to the CPU's, every bit: WARP %d, hardware %d%s"
                info.Name
                h.Kernel
                h.Case
                (if differing = 0 then
                     "same bits"
                 else
                     sprintf "%d of %d words differ%s" differing h.Elements (if nanOnly then ", all NaN" else ""))
                warpVsCpu
                hwVsCpu
                (if h.NanWords.IsEmpty && w.NanWords.IsEmpty then
                     ""
                 else
                     sprintf
                         " (NaN words: hardware %s; WARP %s)"
                         (h.NanWords |> List.map (sprintf "%08x") |> String.concat " ")
                         (w.NanWords |> List.map (sprintf "%08x") |> String.concat " "))

            // Bit-moving kernels move bits on every class; fp32 arithmetic may differ only in NaN payloads.
            Expect.isTrue nanOnly (sprintf "%s %s: only NaN words may differ" h.Kernel h.Case)

// ---- copies, descriptors, throughput, references -----------------------------------------------------------------

/// D1's copy test on WARP: dispatch, copy across pages, dispatch, copy within a page (the scratch buffer), dispatch,
/// copy within a page, in one command list; every intermediate checked against the CPU.
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
    let table = t.Seal()
    use backend = create ()
    let b = backend :> IGpuBackend
    let heap = b.CreateHeap layout
    heap.WriteTable table
    b.LoadKernels(shaders ()) |> ignore
    let x = normals "r5.copies" n
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
        let out = Array.zeroCreate<byte> (int bytes)
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

    Array.concat [ bytesOf (read 1 24576L); bytesOf (read 2 16384L) ]

/// 64 descriptors on the resource heap at CreateHeap and 64 on the probe's transient heap at LoadKernels, then none.
let private heapWrites () =
    use b = create ()
    let ib = b :> IGpuBackend
    let inner = b.Inner
    let layout, table = SelfTests.Suite.Plan()
    let heap = ib.CreateHeap layout
    Expect.equal inner.HeapDescriptorWrites Heap.SlotCount "64 at CreateHeap"
    heap.WriteTable table
    ib.LoadKernels(shaders ()) |> ignore
    Expect.equal inner.ProbeDescriptorWrites Heap.SlotCount "64 on the probe's transient heap"
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
    Expect.equal inner.DescriptorWrites (2 * Heap.SlotCount) "64 + 64, none from dispatches, copies or transfers"
    Expect.equal inner.DeviceRemovedReason HResult.S_OK "healthy"

    Telemetry.check
        "gpu.heap.descriptor_writes"
        (inner.DescriptorWrites = 2 * Heap.SlotCount)
        [
            Attr.str TelemetryKeys.Backend Relabel.To
            Attr.int "slots" Heap.SlotCount
            Attr.int "resource_heap" inner.HeapDescriptorWrites
            Attr.int "probe_heap" inner.ProbeDescriptorWrites
            Attr.int "total" inner.DescriptorWrites
        ]

/// Element counts of the throughput series (scale_f32, x then y in one page).
let throughputSizes = [ 1 <<< 16; 1 <<< 20; 1 <<< 22; 1 <<< 24 ]

/// scale_f32 over 64 Ki to 16 Mi elements (256 KiB to 64 MiB in, as much out), each dispatched alone with timestamps
/// (gpu.dispatch with case throughput_<n> and elements, and no step, so the report's timeline stays on the selftests'
/// last step), twice, exact against the CPU.
/// WARP's time per element says which dispatch sizes fit the 100 ms budget on this many processors.
let private throughput () : byte[] =
    let maxN = List.max throughputSizes
    let yOffset = int64 maxN * 4L
    let layout = HeapLayout.ofPages [ ArenaKind.Activations, 2L * yOffset ]
    let t = TensorTable layout
    let x = t.Add("x", TensorRow.create 1 0L DType.F32 [ int64 maxN ] TensorFlags.ReadOnly 0u)
    let y = t.Add("y", TensorRow.create 1 yOffset DType.F32 [ int64 maxN ] TensorFlags.None 0u)
    let table = t.Seal()
    use backend = create ()
    let b = backend :> IGpuBackend
    let heap = b.CreateHeap layout
    heap.WriteTable table
    b.LoadKernels(shaders ()) |> ignore
    let input = normals "r5.throughput" maxN
    heap.Upload(1, 0L, ReadOnlySpan(bytesOf input))
    let digest = ResizeArray<byte>()

    for n in throughputSizes do
        use _ =
            Telemetry.scope
                [
                    Attr.str TelemetryKeys.Case (sprintf "throughput_%d" n)
                    Attr.int64 "elements" (int64 n)
                ]

        heap.Upload(1, yOffset, ReadOnlySpan(bytesOf (Array.create n 0x7FA5A5A5u)))
        let c = b.Begin()

        Kernels.ScaleF32.dispatch
            c
            {
                X = x
                Y = y
                N = uint32 n
                A = -2.0f
            }

        let clock = Stopwatch.StartNew()
        let done' = b.Wait(b.Submit c)
        let host = clock.Elapsed
        let gpu = done'.Timers |> List.find (fun r -> r.Name = GpuTelemetry.Dispatch)
        let out = Array.zeroCreate<byte> (4 * n)
        heap.Download(1, yOffset, Span out)
        let expected = input[.. n - 1] |> Array.map (fun w -> bits (f32 w * -2.0f))
        Expect.equal (wordsOf out) expected (sprintf "scale_f32 over %d elements" n)
        let nsPerElement = float gpu.DurationNs / float n

        printfn
            "throughput: scale_f32 over %9d elements: %9.3f ms GPU (%.2f ns/element, %.2f GB/s), %9.3f ms host"
            n
            (float gpu.DurationNs / 1e6)
            nsPerElement
            (8.0 * float n / float gpu.DurationNs)
            host.TotalMilliseconds

        digest.AddRange(Security.Cryptography.SHA256.HashData out)

    digest.ToArray()

/// Every COM reference the WARP backend takes is released by Dispose.
let private leaks () =
    let before = Com.liveReferences ()

    do
        use b = create ()
        let ib = b :> IGpuBackend
        let suite = SelfTests.Suite.Create ib
        ib.LoadKernels(shaders ()) |> ignore
        let r = suite.Run(4, b.Inner.Fp32Denormals.Value)
        Expect.isTrue r.Passed "selftests pass"

    Expect.equal (Com.liveReferences ()) before "every COM reference is released by Dispose"

// ---- E1's conformance harness ------------------------------------------------------------------------------------

/// E1's suite (every discovered case, then the negative control) on d3d12-warp, with its own run log of kind kernels.
let conformance () : Test =
    let create () =
        match WarpBackend.tryCreate provider.Value { options () with Debug = false } with
        | Ok b -> b
        | Error e -> failwithf "no d3d12-warp device: %s" e

    let record (_: IGpuBackend) = recordDevice ()

    Lrm.Kernels.Conformance.Suites.BackendRun(
        "d3d12-warp",
        create,
        "Conformance on d3d12-warp (in-box WARP, R5)",
        record
    )
    |> Lrm.Kernels.Conformance.Suites.suite

let tests () =
    testSequenced
    <| testList
        "warp.gpu"
        [
            gpuTest
                "EnumWarpAdapter: WARP has SM 6.6+ and binding tier 3, d3d12-warp accepts it and D1's backend opens it"
                adapter
            gpuTestTwice "D1's fp32 denormal probe on WARP (-denorm preserve), bit-identical twice" probe
            gpuTestTwice "the int-0001 control on WARP: scale_f32 without -denorm, bit-identical twice" control
            gpuTestTwice "D1's five selftests on WARP are bit-exact against the CPU and bit-identical twice" selftests
            gpuTest "the negative control: without dispatches every selftest case is unwritten on WARP" negativeControl
            gpuTest "the selftests on WARP and on the hardware adapter differ at most in NaN payloads" versusHardware
            gpuTestTwice "copies across and within pages are state-correct on WARP, bit-identical twice" copies
            gpuTest "WARP's heap writes 64 descriptors at CreateHeap and the probe's heap 64, then none" heapWrites
            gpuTestTwice "scale_f32 throughput on WARP, 64 Ki to 16 Mi elements, exact and bit-identical twice" throughput
            gpuTest "every COM reference of the WARP backend is released" leaks
            conformance ()
        ]
