/// The harness's own tests (cpu tier): that it catches every way a kernel can be wrong, through faulty backends built
/// around CpuDevice, that its references follow the class's denormal mode, the poison vectors, case validation and
/// discovery.
module Lrm.Kernels.Conformance.HarnessTests

open System
open System.Runtime.InteropServices
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Testing

/// A provider whose constructor throws, as a backend's would without its native library: discovery must report it.
type ThrowingProvider() =
    do raise (DllNotFoundException "no device library")

    interface IGpuBackendProvider with
        member _.Backend = ExecutionBackend.D3D12
        member _.Enumerate() = []
        member _.Create(_, _) = invalidOp "never created"

/// What a faulty backend does differently from the CpuDevice it wraps.
type Faults =
    {
        /// Rewrites a dispatch's block before it is recorded (None drops the dispatch).
        Rewrite: KernelId -> ArgBlock -> ArgBlock option
        /// Runs after submission n (from 1) has executed, with the heap: tampers with memory as a broken kernel would.
        AfterSubmit: int -> IGpuHeap -> unit
        /// Reports this class instead of CpuDevice's.
        Class: ExecutionClass option
        /// Rewrites each completion (its timers).
        Completion: Completion -> Completion
    }

    static member None =
        {
            Rewrite = fun _ a -> Some a
            AfterSubmit = fun _ _ -> ()
            Class = None
            Completion = id
        }

/// A backend that delegates to a CpuDevice and applies Faults.
type FaultyBackend(inner: IGpuBackend, faults: Faults) =
    let mutable heap: IGpuHeap option = None
    let mutable submissions = 0

    interface IGpuBackend with
        member _.Info = inner.Info
        member _.ExecutionClass = defaultArg faults.Class inner.ExecutionClass

        member _.CreateHeap layout =
            let h = inner.CreateHeap layout
            heap <- Some h
            h

        member _.LoadKernels shaders = inner.LoadKernels shaders

        member _.Begin() =
            FaultyList(inner.Begin(), faults) :> ICommandList

        member _.Submit commands =
            match commands with
            | :? FaultyList as f ->
                let s = inner.Submit f.Inner
                submissions <- submissions + 1
                faults.AfterSubmit submissions heap.Value
                s
            | _ -> invalidArg "commands" "not a faulty list"

        member _.Wait s = faults.Completion(inner.Wait s)
        member _.Dispose() = inner.Dispose()

and FaultyList(inner: ICommandList, faults: Faults) =
    member _.Inner = inner

    interface ICommandList with
        member _.Dispatch(k, a) =
            match faults.Rewrite k a with
            | Some a' -> inner.Dispatch(k, a')
            | None -> ()

        member _.Barrier() = inner.Barrier()
        member _.Copy(s, d) = inner.Copy(s, d)
        member _.BeginTimer(n, a) = inner.BeginTimer(n, a)
        member _.EndTimer t = inner.EndTimer t

let private cpu () =
    CpuDevice.create
        CpuDeviceOptions.Default
        { DeviceOptions.For ExecutionBackend.CpuDevice with
            TimeDispatches = true
        }

let private sessionOn (backend: IGpuBackend) (cases: KernelCase list) =
    Session.Load(backend, Lrm.Shaders.Shaders.embedded (), cases, HarnessOptions.Default)

let private faulty (faults: Faults) (cases: KernelCase list) =
    sessionOn (new FaultyBackend(cpu (), faults)) cases

let private scaleCase =
    let r = CaseGen.rng "harness-tests.scale"

    Case.create "scale_f32" "n1000"
    |> Case.f32 "x" (CaseGen.uniform r 1000 -4.0f 4.0f)
    |> Case.output "y" 1024L
    |> Case.u32 "n" 1000u
    |> Case.f32Arg "a" 0.5f

/// The byte offset of element `e` of a case tensor, from the sealed table.
let private elementAt (heap: IGpuHeap) (case: KernelCase) (field: string) (e: int64) =
    let t = heap.Table.Value
    let token = (t.TryFind(case.Id + "/" + field)).Value
    let row = t.Row token
    row.Slot, int64 row.ByteOffset + 4L * e

let private writeWord (heap: IGpuHeap) (slot: int, offset: int64) (word: uint32) =
    heap.Upload(slot, offset, ReadOnlySpan(BitConverter.GetBytes word))

let private flipWord (heap: IGpuHeap) (slot: int, offset: int64) =
    let b = Array.zeroCreate<byte> 4
    heap.Download(slot, offset, Span b)
    writeWord heap (slot, offset) (BitConverter.ToUInt32(b, 0) ^^^ 1u)

/// The probe runs as submissions 1 and 2; the case's two runs are 3 and 4; its splits follow.
let private caseRuns (n: int) = n = 3 || n = 4

let private fails (fragment: string) (o: CaseOutcome) =
    Expect.isFalse o.Passed "the case fails"

    Expect.isTrue
        (o.Failures |> List.exists (fun f -> f.Contains fragment))
        (sprintf "a failure mentions '%s': %A" fragment o.Failures)

let private faultTests =
    testList
        "faults"
        [
            testCase "the unbroken wrapper passes, so the faults below are what fails"
            <| fun () ->
                use s = faulty Faults.None [ scaleCase ]
                let o = s.Run scaleCase
                Expect.isTrue o.Passed o.Detail
                Expect.equal o.Splits 3 "three batch splits"
                Expect.equal o.SplitsMatched 3 "all match"

            testCase "a kernel that writes nothing fails: its poison is left (negative control)"
            <| fun () ->
                use s = faulty Faults.None [ scaleCase ]
                s.Control scaleCase |> fails "still hold their poison"

            testCase "a dispatch dropped by the backend fails the same way"
            <| fun () ->
                let faults =
                    { Faults.None with
                        Rewrite = fun _ a -> if a[3] = 1000u then None else Some a
                    }

                use s = faulty faults [ scaleCase ]
                s.Run scaleCase |> fails "still hold their poison"

            testCase "one wrong bit fails the bit-exact bound and the strict CpuRef check"
            <| fun () ->
                let faults =
                    { Faults.None with
                        AfterSubmit =
                            fun n h ->
                                if n >= 3 then
                                    flipWord h (elementAt h scaleCase "y" 3L)
                    }

                use s = faulty faults [ scaleCase ]
                let o = s.Run scaleCase
                o |> fails "differ from CpuRef F32"
                o |> fails "bit for bit"
                Expect.equal (o.Check |> Option.map _.Measured) (Some 1.0) "one element differs"

            testCase "a run that differs from the other fails run-twice"
            <| fun () ->
                let faults =
                    { Faults.None with
                        AfterSubmit =
                            fun n h ->
                                if n = 4 then
                                    flipWord h (elementAt h scaleCase "y" 10L)
                    }

                use s = faulty faults [ scaleCase ]
                s.Run scaleCase |> fails "runs 1 and 2 differ at element 10"

            testCase "a write past the tensor's end fails its guard"
            <| fun () ->
                let faults =
                    { Faults.None with
                        AfterSubmit =
                            fun n h ->
                                if caseRuns n then
                                    writeWord h (elementAt h scaleCase "y" 1024L) 0u
                    }

                use s = faulty faults [ scaleCase ]
                s.Run scaleCase |> fails "guard word 0 changed"

            testCase "a write before the tensor fails its guard"
            <| fun () ->
                let faults =
                    { Faults.None with
                        AfterSubmit =
                            fun n h ->
                                if caseRuns n then
                                    writeWord h (elementAt h scaleCase "y" -1L) 0u
                    }

                use s = faulty faults [ scaleCase ]
                s.Run scaleCase |> fails "guard word -1 changed"

            testCase "a write inside the capacity but past n fails"
            <| fun () ->
                let faults =
                    { Faults.None with
                        AfterSubmit =
                            fun n h ->
                                if caseRuns n then
                                    writeWord h (elementAt h scaleCase "y" 1000L) 0u
                    }

                use s = faulty faults [ scaleCase ]
                s.Run scaleCase |> fails "written outside the written elements"

            testCase "a kernel that writes its input fails"
            <| fun () ->
                let faults =
                    { Faults.None with
                        AfterSubmit =
                            fun n h ->
                                if caseRuns n then
                                    writeWord h (elementAt h scaleCase "x" 0L) 0u
                    }

                use s = faulty faults [ scaleCase ]
                s.Run scaleCase |> fails "an input changed"

            testCase "a kernel whose rows depend on the batch fails batch invariance, and only that"
            <| fun () ->
                // When the dispatch has one element (a split alone), the scale factor is one ulp off.
                let faults =
                    { Faults.None with
                        Rewrite = fun _ a -> if a[3] = 1u then Some(a.With(4, a[4] + 1u)) else Some a
                    }

                use s = faulty faults [ scaleCase ]
                let o = s.Run scaleCase
                o |> fails "batch invariance: split row 0 alone"
                Expect.isTrue (o.Check |> Option.exists _.Passed) "the whole case is within its bound"

            testCase "a dispatch over the 100 ms budget fails on a shader backend"
            <| fun () ->
                let gpuClass =
                    {
                        Backend = ExecutionBackend.D3D12
                        DeviceId = "pci:0000:0000"
                        Driver = "faulty"
                        ShaderSetHash = Some(Hash.sha256Utf8 "faulty")
                        Arch = "win-x64"
                    }

                let slow (ms: int64) =
                    { Faults.None with
                        Class = Some gpuClass
                        Completion =
                            fun c ->
                                { c with
                                    Timers =
                                        c.Timers
                                        |> List.map (fun t ->
                                            if t.Name = GpuTelemetry.Dispatch then
                                                { t with DurationNs = ms * 1_000_000L }
                                            else
                                                t)
                                }
                    }

                use over = faulty (slow 150L) [ scaleCase ]
                let o = over.Run scaleCase
                o |> fails "over the 100 ms budget"
                Expect.isTrue o.BudgetGating "a shader backend's budget gates"
                Expect.equal o.TimeSource "gpu" "timed by the GPU timers"
                use under = faulty (slow 50L) [ scaleCase ]
                Expect.isTrue (under.Run scaleCase).Passed "50 ms is within the budget"

            testCase "on CpuDevice the time is measured but does not gate"
            <| fun () ->
                let faults =
                    { Faults.None with
                        Completion =
                            fun c ->
                                { c with
                                    Timers = c.Timers |> List.map (fun t -> { t with DurationNs = 150_000_000L })
                                }
                    }

                use s = faulty faults [ scaleCase ]
                let o = s.Run scaleCase
                Expect.isTrue o.Passed o.Detail
                Expect.isFalse o.BudgetGating "emulation time says nothing about a GPU"
                Expect.equal o.TimeNs 150_000_000L "the faster run's dispatch time"
                Expect.equal o.TimeSource "emulated" "CpuDevice's timers are emulation time, not GPU time"

            testCase "an in-place case whose results equal its inputs is refused"
            <| fun () ->
                let c =
                    Case.create "scale_f32" "identity_in_place"
                    |> Case.f32 "x" [| 1.0f; 2.0f |]
                    |> Case.inPlace "y" "x"
                    |> Case.u32 "n" 2u
                    |> Case.f32Arg "a" 1.0f

                use s = sessionOn (cpu ()) [ c ]
                s.Run c |> fails "cannot detect a missing write"
        ]

/// A CpuDevice with a GPU arithmetic model, through the harness: every case passes when the references follow the
/// measured mode.
let private modelTests =
    let run (cpu: CpuDeviceOptions) (expectMode: Fp32DenormalMode) =
        let backend =
            CpuDevice.create
                { cpu with
                    Threads = min 4 Environment.ProcessorCount
                }
                (DeviceOptions.For ExecutionBackend.CpuDevice)

        use s = sessionOn backend (Discovery.cases typeof<KernelCase>.Assembly)
        Expect.equal s.Mode expectMode "the probe measures the model's mode"
        let failed = s.Cases |> List.map s.Run |> List.filter (fun o -> not o.Passed)

        Expect.isEmpty
            (failed |> List.map (fun o -> o.Case.Id + ": " + o.Detail))
            "every case passes in the class's mode"

    testList
        "models"
        [
            testCase "a flushing CpuDevice: the probe measures flush and every case passes against flushed references"
            <| fun () ->
                run
                    { CpuDeviceOptions.Default with
                        Fp32Denormals = Fp32DenormalMode.Flush
                    }
                    Fp32DenormalMode.Flush

            testCase "a CpuDevice that returns 0x7FFFFFFF for every NaN: NaN results match by class"
            <| fun () ->
                run
                    { CpuDeviceOptions.Default with
                        CanonicalNaN = true
                    }
                    Fp32DenormalMode.Preserve

            testCase "both models at once"
            <| fun () ->
                run
                    { CpuDeviceOptions.Default with
                        Fp32Denormals = Fp32DenormalMode.Flush
                        CanonicalNaN = true
                    }
                    Fp32DenormalMode.Flush

            testCase "a flushing device's scale_f32 is wrong against a preserve reference"
            <| fun () ->
                // What the harness would see if it ignored the measured mode.
                let backend =
                    CpuDevice.create
                        { CpuDeviceOptions.Default with
                            Fp32Denormals = Fp32DenormalMode.Flush
                        }
                        (DeviceOptions.For ExecutionBackend.CpuDevice)

                use s = sessionOn backend []
                Expect.equal s.Mode Fp32DenormalMode.Flush "flush"

                let preserve = Fp32Denormals.expected Fp32DenormalMode.Preserve
                let flush = Fp32Denormals.expected Fp32DenormalMode.Flush
                Expect.notEqual preserve flush "the modes differ on the probe"
        ]

let private refInputs (c: KernelCase) (mode: Fp32DenormalMode) : RefInputs =
    let id = (Kernels.tryFind c.Kernel).Value
    let spec = Kernels.all[id.Value]
    let w = Array.zeroCreate<uint32> 32

    spec.Fields
    |> List.iteri (fun i f ->
        match c.Scalars |> List.tryFind (fun (n, _) -> n = f.Name) with
        | Some(_, v) -> w[i + 1] <- v
        | None -> ())

    w[0] <- 1u

    {
        Spec = spec
        Args = ArgBlock.OfWords w
        Input =
            fun field ->
                match (c.Io.Tensors |> List.find (fun t -> t.Field = field)).Data with
                | CaseData.Input b -> b
                | _ -> failwith "not an input"
        Row = fun _ -> TensorRow.Zero
        Mode = mode
        Exec = CpuExec.Serial
    }

let private referenceTests =
    testList
        "references"
        [
            testCase "scale_f32 on the probe's inputs gives the probe's expected words in each mode"
            <| fun () ->
                for mode in [ Fp32DenormalMode.Preserve; Fp32DenormalMode.Flush ] do
                    let out = References.compute (refInputs Session.ProbeCase mode) |> List.exactlyOne
                    Expect.equal out.Bits (Fp32Denormals.expected mode) (Fp32Denormals.name mode)

            testCase "mixed: the denormal elements of scale_f32 are compared within the class only"
            <| fun () ->
                let out =
                    References.compute (refInputs Session.ProbeCase Fp32DenormalMode.Mixed)
                    |> List.exactlyOne

                // Inputs 0 and 1 are the normal controls; 2 and 3 give denormal results; 4 to 6 are denormal inputs.
                Expect.equal
                    out.WithinClassOnly
                    (Some [| false; false; true; true; true; true; true |])
                    "denormal inputs and results"

            testCase "add: a denormal operand flushes in a flush class (2^-149 + 2^-149 = 0)"
            <| fun () ->
                let c =
                    Case.create "add" "t"
                    |> Case.f32Bits "a" [| 0x00000001u; 0x3F800000u |]
                    |> Case.f32Bits "b" [| 0x00000001u; 0x00000001u |]
                    |> Case.output "y" 2L
                    |> Case.u32 "n" 2u

                let bitsIn mode =
                    (References.compute (refInputs c mode) |> List.exactlyOne).Bits

                Expect.equal (bitsIn Fp32DenormalMode.Preserve) [| 0x00000002u; 0x3F800000u |] "preserve"
                Expect.equal (bitsIn Fp32DenormalMode.Flush) [| 0x00000000u; 0x3F800000u |] "flush"

            testCase "argmax_rows in a flush class: a denormal ties with zero and the lowest index wins"
            <| fun () ->
                let c =
                    Case.create "argmax_rows" "t"
                    |> Case.f32Bits "x" [| 0x00000000u; 0x00000005u; 0xBF800000u |]
                    |> Case.outputU32 "y" 1L
                    |> Case.u32 "rows" 1u
                    |> Case.u32 "cols" 3u

                let at mode =
                    (References.compute (refInputs c mode) |> List.exactlyOne).Bits

                Expect.equal (at Fp32DenormalMode.Preserve) [| 1u |] "preserve: the denormal is the maximum"
                Expect.equal (at Fp32DenormalMode.Flush) [| 0u |] "flush: it ties with +0 at index 0"

            testCase "a relative kernel with a denormal input is compared within a mixed class only"
            <| fun () ->
                let c =
                    Case.create "silu_mul" "t"
                    |> Case.f32Bits "gate" [| 0x00000001u; 0x3F800000u |]
                    |> Case.f32 "up" [| 1.0f; 2.0f |]
                    |> Case.output "y" 2L
                    |> Case.u32 "n" 2u

                let out = References.compute (refInputs c Fp32DenormalMode.Mixed) |> List.exactlyOne
                Expect.equal out.WithinClassOnly (Some [| true; true |]) "the whole case"
                Expect.isSome out.F64 "a relative kernel has an F64 reference"

            testCase "the NaN rule: the first NaN operand, quieted, whatever the other operand"
            <| fun () ->
                let mul = References.nanRule (fun a b -> a * b)
                let add = References.nanRule (fun a b -> a + b)
                Expect.equal (mul 0x7FC00001u 0x7FC00003u) 0x7FC00001u "two quiet NaNs: the first"
                Expect.equal (mul 0x3F800000u 0x7F800003u) 0x7FC00003u "a signalling NaN second: quieted"
                Expect.equal (add 0xFF800007u 0xFFC00008u) 0xFFC00007u "the sign stays"
                Expect.equal (add 0x3F800000u 0x3F800000u) 0x40000000u "no NaN: the float32 operation"

            testCase "two NaN operands give the same bits in every run for 1.5 s of one process"
            <| fun () ->
                // Tiered compilation recompiles the emulators and the references partway through a process; before
                // the NaN rule, x64's choice of NaN operand then flipped (scale_f32 from about run 150 to 1,100).
                let cases =
                    Discovery.cases typeof<KernelCase>.Assembly
                    |> List.filter (fun c -> c.Name = "both_nan" || c.Name = "e1_both_nan")

                Expect.equal (cases |> List.map _.Kernel |> List.sort) [ "add"; "scale_f32" ] "both cases"
                use s = sessionOn (cpu ()) cases
                let failures = ResizeArray<string>()

                let clock = Diagnostics.Stopwatch.StartNew()
                let mutable i = 0

                while i < 2000 || (clock.ElapsedMilliseconds < 1500L && i < 50000) do
                    if i % 200 = 0 then
                        Threading.Thread.Sleep 20

                    for c in cases do
                        let o = s.Run c

                        if not o.Passed && failures.Count < 3 then
                            failures.Add(sprintf "run %d of %s: %s" i c.Id o.Detail)

                    i <- i + 1

                if failures.Count > 0 then
                    failtestf "runs that failed: %s" (String.Join("; ", failures))

            testCase "every catalog kernel has a KernelBounds entry, an emulator, a reference and a case"
            <| fun () ->
                let cases = Discovery.cases typeof<KernelCase>.Assembly

                for k in Kernels.all do
                    Expect.isSome (KernelBounds.tryFind k.Name) (k.Name + " has a bound")

                    Expect.isTrue
                        (cases |> List.exists (fun c -> c.Kernel = k.Name))
                        (k.Name + " has at least one conformance case")

                    let c = cases |> List.find (fun c -> c.Kernel = k.Name)

                    if k.Name <> "table_echo" then
                        Expect.isNonEmpty (References.compute (refInputs c Fp32DenormalMode.Preserve)) k.Name
        ]

/// Poison vectors (ConformanceHarness contract, "Vectors"), computed independently in Python (hashlib and a Philox4x32-10
/// written from the Random123 paper).
let private poisonTests =
    testList
        "poison"
        [
            testCase "KAT P1: the seed of (cfg, scale_f32/n1, run1/y)"
            <| fun () ->
                let s = Poison.seed "cfg" "scale_f32/n1" "run1/y"
                Expect.equal s.Hex "d0234eb5b214ca17e194f78018a6ed6db94d9979e8baee26c6eb10d42bdbbd08" "seed"

            testCase "KAT P2: its first four poison words"
            <| fun () ->
                let w = Poison.words (Poison.seed "cfg" "scale_f32/n1" "run1/y") 4

                Expect.equal
                    (w |> Array.map (sprintf "%08x"))
                    [| "7fbe4924"; "ff867e52"; "ff8961ba"; "7f8e1b58" |]
                    "words"

            testCase "KAT P3: avoiding 7f800001 gives 7f800002"
            <| fun () -> Expect.equal (Poison.avoiding 0x7F800001u 0x7F800001u) 0x7F800002u "mantissa 2"

            testCase "KAT S1: the splits of a case with 10 rows"
            <| fun () ->
                let spec = Kernels.all[(Kernels.tryFind "rmsnorm_rows").Value.Value]

                let block =
                    Kernels.RmsnormRows.encode
                        {
                            X = TensorToken 1u
                            W = TensorToken 2u
                            Y = TensorToken 3u
                            Rows = 10u
                            Cols = 4u
                            Eps = 1e-6f
                        }

                let splits = References.splits spec block

                Expect.equal
                    (splits |> List.map _.Label)
                    [ "row 0 alone"; "row 9 alone"; "rows 3 to 5" ]
                    "first, last, middle third"

                Expect.equal
                    (splits |> List.map (fun s -> s.Views))
                    [
                        [ "x", 0L, 4L; "y", 0L, 4L ]
                        [ "x", 36L, 4L; "y", 36L, 4L ]
                        [ "x", 12L, 12L; "y", 12L, 12L ]
                    ]
                    "views at the rows"

            testCase "every poison word is a signalling NaN, and runs and configurations differ"
            <| fun () ->
                let a = Poison.words (Poison.seed "c1" "k/case" "run1/y") 4096
                let b = Poison.words (Poison.seed "c1" "k/case" "run2/y") 4096
                let c = Poison.words (Poison.seed "c2" "k/case" "run1/y") 4096
                Expect.isTrue (a |> Array.forall Poison.isSignallingNaN) "signalling NaNs"
                Expect.isFalse (Array.exists2 (=) a b) "run 1 and run 2 differ in every word"
                Expect.isFalse (Array.exists2 (=) a c) "configurations differ in every word"

            testCase "avoiding never returns the expected word and stays a signalling NaN"
            <| fun () ->
                for p in [ 0x7F800001u; 0xFF800001u; 0x7FBFFFFFu; 0x7F812345u; 0xFFA00000u ] do
                    let q = Poison.avoiding p p
                    Expect.notEqual q p "differs"
                    Expect.isTrue (Poison.isSignallingNaN q) "a signalling NaN"
                    Expect.equal (Poison.avoiding 0u p) p "unchanged when it differs"
        ]

let private validationTests =
    let raises (fragment: string) (c: KernelCase) =
        let raised =
            try
                Session.Load(cpu (), Lrm.Shaders.Shaders.embedded (), [ c ], HarnessOptions.Default)
                |> ignore

                None
            with :? ArgumentException as e ->
                Some e.Message

        match raised with
        | None -> failtestf "expected ArgumentException containing '%s'" fragment
        | Some m when not (m.Contains fragment) -> failtestf "expected '%s' in: %s" fragment m
        | Some _ -> ()

    let ok =
        Case.create "scale_f32" "v"
        |> Case.f32 "x" [| 1.0f |]
        |> Case.output "y" 1L
        |> Case.u32 "n" 1u
        |> Case.f32Arg "a" 2.0f

    testList
        "validation"
        [
            testCase "an unknown kernel"
            <| fun () -> raises "not a catalog kernel" { ok with Kernel = "softmax_rows" }
            testCase "a missing scalar"
            <| fun () ->
                raises
                    "has no value"
                    { ok with
                        Scalars = ok.Scalars |> List.filter (fun (n, _) -> n <> "a")
                    }
            testCase "an unknown scalar"
            <| fun () -> raises "not a scalar field" (ok |> Case.u32 "m" 3u)
            testCase "a field without a tensor"
            <| fun () ->
                raises
                    "has no tensor"
                    { ok with
                        Io =
                            { ok.Io with
                                Tensors = ok.Io.Tensors |> List.filter (fun t -> t.Field <> "y")
                            }
                    }
            testCase "a dtype the field does not accept"
            <| fun () ->
                raises
                    "the kernel accepts"
                    { ok with
                        Io =
                            { ok.Io with
                                Tensors =
                                    ok.Io.Tensors
                                    |> List.map (fun t -> if t.Field = "x" then { t with DType = DType.U32 } else t)
                            }
                    }
            testCase "an alias the catalog does not declare"
            <| fun () ->
                raises
                    "may not alias"
                    (Case.create "matmul_bf16w" "v"
                     |> Case.f32 "x" [| 1.0f |]
                     |> Case.bf16 "w" [| 0x3F80us |]
                     |> Case.inPlace "y" "x"
                     |> Case.u32 "m" 1u
                     |> Case.u32 "n" 1u
                     |> Case.u32 "k" 1u)
            testCase "a written field given input data"
            <| fun () ->
                raises
                    "written and never read"
                    (Case.create "scale_f32" "v"
                     |> Case.f32 "x" [| 1.0f |]
                     |> Case.f32 "y" [| 1.0f |]
                     |> Case.u32 "n" 1u
                     |> Case.f32Arg "a" 1.0f)
            testCase "a case below a kernel's minimum element count, GPU-only or not, names the rule"
            <| fun () ->
                let undersized =
                    Case.create "rmsnorm_rows" "v"
                    |> Case.f32 "x" (Array.create 64 1.0f)
                    |> Case.bf16 "w" (Array.create 64 0x3F80us)
                    |> Case.output "y" 4L
                    |> Case.u32 "rows" 1u
                    |> Case.u32 "cols" 64u
                    |> Case.f32Arg "eps" 1e-6f

                raises "rule y >= rows * cols" undersized
                raises "rmsnorm_rows/v" (undersized |> Case.gpuOnly)

                raises
                    "x >= n"
                    (Case.create "scale_f32" "v"
                     |> Case.f32 "x" [| 1.0f |]
                     |> Case.output "y" 16L
                     |> Case.u32 "n" 16u
                     |> Case.f32Arg "a" 1.0f
                     |> Case.gpuOnly)

            testCase "a case that breaks a kernel's argument rule names the rule"
            <| fun () ->
                raises
                    "heads must divide rows"
                    (Case.create "rope_apply" "v"
                     |> Case.f32 "x" (Array.zeroCreate 12)
                     |> Case.f32 "cos_table" (Array.zeroCreate 8)
                     |> Case.f32 "sin_table" (Array.zeroCreate 8)
                     |> Case.output "y" 12L
                     |> Case.u32 "rows" 3u
                     |> Case.u32 "heads" 2u
                     |> Case.u32 "head_dim" 4u
                     |> Case.u32 "pos0" 0u)

            testCase "two cases with one id"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () ->
                        Session.Load(cpu (), Lrm.Shaders.Shaders.embedded (), [ ok; ok ], HarnessOptions.Default)
                        |> ignore)
                    "a repeated id"
        ]

let private discoveryTests =
    testList
        "discovery"
        [
            testCase "a provider exposed as a module value is found"
            <| fun () ->
                let ps = Discovery.providersIn typeof<CpuDeviceBackend>.Assembly
                Expect.isTrue (ps |> List.exists (fun p -> p.Backend = ExecutionBackend.CpuDevice)) "CpuDevice.provider"

            testCase "looking for a GPU provider never throws"
            <| fun () ->
                for b, _ in Discovery.assemblies do
                    Discovery.provider b |> ignore

            testCase "a provider that throws is reported as an error, not dropped"
            <| fun () ->
                let found = Discovery.discoverIn typeof<ThrowingProvider>.Assembly
                Expect.isFalse (found.Providers |> List.exists (fun p -> p :? ThrowingProvider)) "not a provider"

                Expect.isTrue
                    (found.Errors
                     |> List.exists (fun e -> e.Contains "ThrowingProvider" && e.Contains "no device library"))
                    (sprintf "the error names the type and the exception: %A" found.Errors)

            testCase "the gpu tier fails a required backend that has no provider"
            <| fun () ->
                let groups = Suites.gpuWith [ Backend.D3D12; Backend.Vulkan ]

                for backend, testing in
                    [
                        ExecutionBackend.D3D12, Backend.D3D12
                        ExecutionBackend.Vulkan, Backend.Vulkan
                    ] do
                    let name = ExecutionBackend.name backend
                    let tests = groups |> List.filter (fst >> (=) testing) |> List.collect snd
                    Expect.isNonEmpty tests (name + " has gpu tests")

                    match Discovery.discover backend with
                    | Some _, _ -> ()
                    | None, _ ->
                        // No provider yet (D1 and V1 add them): the group is a test that fails and says why.
                        let code =
                            tests
                            |> List.tryPick (function
                                | TestLabel(n, TestCase(Sync f, _), _) when
                                    n.EndsWith "a required backend has a provider"
                                    ->
                                    Some f
                                | _ -> None)

                        match code with
                        | None -> failtestf "%s: no failing provider test among %A" name tests
                        | Some f ->
                            let message =
                                try
                                    f ()
                                    None
                                with :? AssertException as e ->
                                    Some e.Message

                            Expect.isTrue
                                (message |> Option.exists (fun m -> m.Contains "LRM_REQUIRED_BACKENDS"))
                                (sprintf "%s: the test fails naming LRM_REQUIRED_BACKENDS (%A)" name message)

                Expect.isEmpty
                    (Suites.gpuWith []
                     |> List.collect snd
                     |> List.choose (function
                         | TestLabel(n, _, _) when n.EndsWith "a required backend has a provider" -> Some n
                         | _ -> None))
                    "nothing is required without LRM_REQUIRED_BACKENDS"

            testCase "the case modules are found, ids are unique"
            <| fun () ->
                let cases = Discovery.cases typeof<KernelCase>.Assembly
                Expect.isTrue (cases |> List.exists (fun c -> c.Kernel = "groups_echo")) "selftest cases"
                Expect.isTrue (cases |> List.exists (fun c -> c.Kernel = "matmul_bf16w")) "harness cases"
                Expect.equal (cases |> List.distinctBy _.Id |> List.length) cases.Length "unique ids"
        ]

let tests =
    testList
        "harness"
        [
            faultTests
            modelTests
            referenceTests
            poisonTests
            validationTests
            discoveryTests
        ]
