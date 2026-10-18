namespace Lrm.Gpu.D3D12

open System
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core

/// The outcome of one selftest case on one run.
type SelftestCase =
    {
        Kernel: string
        Case: string
        /// Elements the case writes and compares.
        Elements: int
        /// Elements whose bits differ from the expectation (after the class's rules: any NaN matches any NaN for
        /// scale_f32, and the fp32 denormal mode).
        Differing: int
        /// Written elements that still hold their poison: the kernel wrote nothing there.
        NotWritten: int
        /// Elements past the written region that lost their poison: the kernel wrote where it must not.
        Clobbered: int
        /// Elements compared within the class only (a mixed denormal mode: an input or result is a denormal).
        ClassOnly: int
        /// The distinct NaN words the GPU returned (scale_f32), for the NaN rule's record.
        NanWords: uint32 list
        /// The written region's bits as the GPU returned them, little-endian.
        Output: byte[]
        /// The expected bits of the written region.
        Expected: byte[]
        /// Every written element matched and nothing outside was written.
        Passed: bool
        Detail: string
    }

/// One run of the selftest suite: one submission with every selftest dispatch.
type SelftestRun =
    {
        Run: int
        Mode: Fp32DenormalMode
        Cases: SelftestCase list
        Completion: Completion
    }

    member r.Passed = r.Cases |> List.forall _.Passed

    /// Every case's written region, concatenated in case order: the bytes the run-twice check compares (the poison,
    /// which differs per run, is not part of them).
    member r.Bytes = r.Cases |> List.map _.Output |> Array.concat

/// The five selftests of the catalog (scale_f32, table_echo, args_echo, groups_echo, bf16_unpack), over any shader
/// backend through IGpuBackend, so they read the same on D3D12 and Vulkan. Every output is poisoned before each run
/// with a pattern that differs per run and per case and never equals the expected word, and the words past each
/// written region are poisoned too and must keep it; expectations are the CPU's, in the class's fp32 denormal mode for
/// scale_f32 (KernelAbiAndCatalog, kernel rules; KernelBounds).
[<RequireQualifiedAccess>]
module SelfTests =
    /// scale_f32: elements, not a multiple of 256, so the last group's bound check matters.
    [<Literal>]
    let ScaleN = 4099

    /// scale_f32's factor: 0.75 makes some normal inputs denormal and keeps every denormal input denormal or zero.
    [<Literal>]
    let ScaleA = 0.75f

    /// scale_f32 in place (y aliases x): elements.
    [<Literal>]
    let InPlaceN = 300

    /// groups_echo without folding.
    [<Literal>]
    let UnfoldedGroups = 300

    /// groups_echo above 2 * 65,535 groups: folded into 3 rows of 43,693, so 2 physical groups have no work.
    [<Literal>]
    let FoldedGroups = 131077

    /// bf16_unpack: every bf16 bit pattern.
    [<Literal>]
    let Bf16N = 65536

    /// Poisoned words after each written region, which must keep their poison.
    [<Literal>]
    let Spare = 16

    /// The seed of the selftest inputs ("LRM-D1ST").
    [<Literal>]
    let Seed = 0x4C524D2D44315354UL

    let private isNaN32 (w: uint32) =
        w &&& 0x7F800000u = 0x7F800000u && w &&& 0x007FFFFFu <> 0u

    let private isDenormal (w: uint32) =
        w &&& 0x7F800000u = 0u && w &&& 0x007FFFFFu <> 0u

    /// scale_f32's inputs: the probe's words, special values (zeros, infinities, NaNs with payloads, the extremes),
    /// then random bit patterns over every exponent.
    let scaleInputs: uint32[] =
        let specials =
            [|
                0x00000000u
                0x80000000u
                0x7F800000u
                0xFF800000u
                0x7FC00000u
                0xFFC00001u
                0x7F800001u // a signalling NaN
                0x7FBFFFFFu
                0x7F7FFFFFu // the largest finite
                0xFF7FFFFFu
                0x00000001u // the smallest denormal
                0x80000001u
                0x007FFFFFu // the largest denormal
                0x00FFFFFFu
                0x3F800000u
                0xBF800000u
            |]

        let rng = DetRng(Seed, DetRng.streamOf "d1.selftest.scale_f32")
        let fixedWords = Array.append Fp32Denormals.probeInputs specials

        Array.append fixedWords (Array.init (ScaleN - fixedWords.Length) (fun _ -> rng.NextU32()))

    /// scale_f32 in place: values in [1, 2), so 0.75 * x never equals x and a missing write shows.
    let inPlaceInputs: uint32[] =
        let rng = DetRng(Seed, DetRng.streamOf "d1.selftest.scale_f32.in_place")
        Array.init InPlaceN (fun _ -> 0x3F800000u ||| (rng.NextU32() >>> 9))

    /// args_echo's words 2..31: edge values, then random ones.
    let argsWords: uint32[] =
        let rng = DetRng(Seed, DetRng.streamOf "d1.selftest.args_echo")

        let edges =
            [|
                0u
                1u
                0xFFFFFFFFu
                0x80000000u
                0x7FFFFFFFu
                0xDEADBEEFu
                0x7FC00000u
                0x00800000u
            |]

        Array.append edges (Array.init (30 - edges.Length) (fun _ -> rng.NextU32()))

    /// bf16_unpack's input: every bf16 bit pattern, in order.
    let bf16Inputs: uint16[] = Array.init Bf16N uint16

    /// One fp32 multiply as the CPU reference computes it (IEEE, denormals kept).
    let scaleBits (a: float32) (x: uint32) =
        BitConverter.SingleToUInt32Bits(BitConverter.UInt32BitsToSingle x * a)

    /// scale_f32's expected word in a denormal mode: preserve, the CPU's multiply; flush, the multiply with every
    /// denormal input and result replaced by zero of the same sign (Fp32Denormals.flushBits). Mixed has no single
    /// answer: the elements whose input or result is a denormal are compared within the class only (classOnly).
    let scaleExpected (mode: Fp32DenormalMode) (a: float32) (x: uint32) =
        match mode with
        | Fp32DenormalMode.Flush -> Fp32Denormals.flushBits (scaleBits a (Fp32Denormals.flushBits x))
        | _ -> scaleBits a x

    /// True when a mixed class compares this element within the class only.
    let classOnly (mode: Fp32DenormalMode) (a: float32) (x: uint32) =
        mode = Fp32DenormalMode.Mixed && (isDenormal x || isDenormal (scaleBits a x))

    /// The poison of word `i` of case `case` on run `run`: the expected word XOR a non-zero mask, never a NaN where
    /// a NaN would match under the NaN rule. Spare words (past the written region) take the mask itself.
    let poison (fp32Arithmetic: bool) (case: int) (run: int) (i: int) (expected: uint32 voption) =
        let mask =
            uint32 (SplitMix64.Mix((uint64 (case + 1) <<< 44) ^^^ (uint64 run <<< 32) ^^^ uint64 i))
            ||| 1u

        match expected with
        | ValueNone -> mask
        | ValueSome e ->
            let p = e ^^^ mask

            if fp32Arithmetic && isNaN32 e && isNaN32 p then
                p ^^^ 0x40000000u
            else
                p

    let private words (bytes: byte[]) =
        MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()

    let private bytesOf (ws: uint32[]) =
        MemoryMarshal.AsBytes(ReadOnlySpan ws).ToArray()

    /// Judges one case. `out` holds the written region then the spare words; `expected` and `poisonWords` the same
    /// layout (expected only for the written region). `notWrittenOf i` decides a missing write for an in-place case,
    /// whose written region starts as its input instead of poison.
    let judge
        (kernel: string)
        (case: string)
        (fp32Arithmetic: bool)
        (expected: uint32[])
        (unwritten: uint32[])
        (spare: uint32[])
        (isClassOnly: int -> bool)
        (out: uint32[])
        : SelftestCase =
        let n = expected.Length
        let mutable differing = 0
        let mutable notWritten = 0
        let mutable clobbered = 0
        let mutable classOnlyCount = 0
        let nans = Collections.Generic.SortedSet<uint32>()
        let notes = ResizeArray<string>()

        for i in 0 .. n - 1 do
            let o = out[i]

            if o = unwritten[i] then
                notWritten <- notWritten + 1

                if notes.Count < 4 then
                    notes.Add(sprintf "y[%d] = %08x is still unwritten" i o)
            elif isClassOnly i then
                classOnlyCount <- classOnlyCount + 1
            elif o <> expected[i] then
                if fp32Arithmetic && isNaN32 o && isNaN32 expected[i] then
                    nans.Add o |> ignore
                else
                    differing <- differing + 1

                    if notes.Count < 4 then
                        notes.Add(sprintf "y[%d] = %08x, expected %08x" i o expected[i])
            elif fp32Arithmetic && isNaN32 o then
                nans.Add o |> ignore

        for j in 0 .. spare.Length - 1 do
            if out[n + j] <> spare[j] then
                clobbered <- clobbered + 1

                if notes.Count < 4 then
                    notes.Add(sprintf "y[%d] = %08x past the written region (poison %08x)" (n + j) out[n + j] spare[j])

        let passed = differing = 0 && notWritten = 0 && clobbered = 0

        {
            Kernel = kernel
            Case = case
            Elements = n
            Differing = differing
            NotWritten = notWritten
            Clobbered = clobbered
            ClassOnly = classOnlyCount
            NanWords = List.ofSeq nans
            Output = bytesOf out[0 .. n - 1]
            Expected = bytesOf expected
            Passed = passed
            Detail =
                if notes.Count > 0 then
                    String.concat "; " notes
                elif nans.Count > 0 then
                    "NaN results " + (nans |> Seq.map (sprintf "%08x") |> String.concat " ")
                else
                    ""
        }

    /// The suite's tensors on one backend's heap: one placement per tensor, a sealed table, and the cases.
    type Suite private (backend: IGpuBackend, heap: IGpuHeap, table: SealedTable, layout: HeapLayout) =
        let token (name: string) =
            match table.TryFind name with
            | Some t -> t
            | None -> invalidOp ("no tensor " + name)

        let row (name: string) = table.Row(token name)
        let backendName = ExecutionBackend.name backend.Info.Backend

        let upload (name: string) (ws: uint32[]) =
            let r = row name
            heap.Upload(r.Slot, int64 r.ByteOffset, MemoryMarshal.AsBytes(ReadOnlySpan ws))

        let download (name: string) (count: int) =
            let r = row name
            let out = Array.zeroCreate<uint32> count
            heap.Download(r.Slot, int64 r.ByteOffset, MemoryMarshal.AsBytes(Span out))
            out

        /// The heap layout of the suite: the bf16 input in Weights, the f32 tensors in Activations and the u32
        /// outputs in Scratch, each page rounded up to 64 KiB.
        static member Plan() =
            let planner = HeapPlanner()

            let tensors =
                [
                    "bf16.x", ArenaKind.Weights, DType.BF16, Bf16N, TensorFlags.ReadOnly
                    "scale.x", ArenaKind.Activations, DType.F32, ScaleN, TensorFlags.ReadOnly
                    "scale.y", ArenaKind.Activations, DType.F32, ScaleN + Spare, TensorFlags.None
                    "scale.in_place", ArenaKind.Activations, DType.F32, InPlaceN + Spare, TensorFlags.None
                    "table_echo.target", ArenaKind.Activations, DType.F32, 167, TensorFlags.None
                    "bf16.y", ArenaKind.Activations, DType.F32, Bf16N + Spare, TensorFlags.None
                    "table_echo.y.view", ArenaKind.Scratch, DType.U32, 16 + Spare, TensorFlags.None
                    "table_echo.y.null", ArenaKind.Scratch, DType.U32, 16 + Spare, TensorFlags.None
                    "table_echo.y.bf16", ArenaKind.Scratch, DType.U32, 16 + Spare, TensorFlags.None
                    "args_echo.y", ArenaKind.Scratch, DType.U32, 32, TensorFlags.None
                    "groups_echo.y.unfolded", ArenaKind.Scratch, DType.U32, UnfoldedGroups + Spare, TensorFlags.None
                    "groups_echo.y.folded", ArenaKind.Scratch, DType.U32, FoldedGroups + Spare, TensorFlags.None
                ]

            let placed =
                tensors
                |> List.map (fun (name, arena, dtype, count, flags) ->
                    name, planner.Allocate(arena, int64 count * int64 (DType.size dtype)), dtype, count, flags)

            let layout = planner.Build()
            let t = TensorTable layout

            for name, placement, dtype, count, flags in placed do
                if name = "table_echo.target" then
                    // A non-contiguous rank-3 row (strides 64, 8, 1 over dims 3, 5, 7): table_echo reads only the row.
                    let slot = layout.SlotOf(placement.Arena, placement.Page)

                    t.Add(
                        name,
                        TensorRow.strided
                            slot
                            placement.Offset
                            DType.F32
                            [ 3L; 5L; 7L ]
                            [ 64L; 8L; 1L ]
                            TensorFlags.View
                            (TensorRow.debugIdOf name)
                    )
                    |> ignore
                else
                    t.Add(name, placement, dtype, [ int64 count ], flags) |> ignore

            layout, t.Seal()

        /// Creates the suite's heap on `backend` (CreateHeap, which runs once per backend) and writes its table.
        static member Create(backend: IGpuBackend) =
            let layout, table = Suite.Plan()
            let heap = backend.CreateHeap layout
            heap.WriteTable table
            Suite(backend, heap, table, layout)

        member _.Layout = layout
        member _.Table = table
        member _.Heap = heap

        /// Runs every selftest once, in one submission (dispatches off: the negative control, where every case must
        /// fail as unwritten). `run` (from 1) varies the poison and is the `step` of the run's records. `mode` is the
        /// backend's measured fp32 denormal mode.
        member _.Run(run: int, mode: Fp32DenormalMode, ?dispatch: bool) : SelftestRun =
            let dispatch = defaultArg dispatch true
            use _ = Telemetry.scope [ Attr.int "step" run ]

            // Expectations, from the CPU.
            let scaleExp = scaleInputs |> Array.map (scaleExpected mode ScaleA)
            let inPlaceExp = inPlaceInputs |> Array.map (scaleExpected mode ScaleA)

            let viewRow = (row "table_echo.target").ToWords()
            let bf16Row = (row "bf16.x").ToWords()
            let nullRow = Array.zeroCreate<uint32> 16

            let argsExp = Array.append [| 1u; (token "args_echo.y").Value |] argsWords

            let unfoldedExp = Array.init UnfoldedGroups uint32
            let foldedExp = Array.init FoldedGroups uint32
            let bf16Exp = bf16Inputs |> Array.map Bf16.widenBits

            // (name, case index, fp32 arithmetic, expected written words)
            let outputs =
                [
                    "scale.y", "scale_f32", "x_to_y", 0, true, scaleExp
                    "scale.in_place", "scale_f32", "in_place", 1, true, inPlaceExp
                    "table_echo.y.view", "table_echo", "strided_view", 2, false, viewRow
                    "table_echo.y.null", "table_echo", "null_token", 3, false, nullRow
                    "table_echo.y.bf16", "table_echo", "bf16_row", 4, false, bf16Row
                    "args_echo.y", "args_echo", "all_words", 5, false, argsExp
                    "groups_echo.y.unfolded", "groups_echo", "unfolded", 6, false, unfoldedExp
                    "groups_echo.y.folded", "groups_echo", "folded", 7, false, foldedExp
                    "bf16.y", "bf16_unpack", "every_pattern", 8, false, bf16Exp
                ]

            let capacity (name: string) = int (row name).ElemCount

            let poisoned =
                outputs
                |> List.map (fun (name, _, _, case, fp32, expected) ->
                    let n = expected.Length

                    let ws =
                        Array.init (capacity name) (fun i ->
                            poison fp32 case run i (if i < n then ValueSome expected[i] else ValueNone))

                    name, ws)
                |> Map.ofList

            // Inputs and poison.
            upload "scale.x" scaleInputs
            upload "bf16.x" (MemoryMarshal.Cast<uint16, uint32>(ReadOnlySpan bf16Inputs).ToArray())

            for name, _, _, _, _, _ in outputs do
                let ws = poisoned[name]

                if name = "scale.in_place" then
                    upload name (Array.append inPlaceInputs ws[InPlaceN..])
                else
                    upload name ws

            let commands = backend.Begin()

            if dispatch then
                Kernels.ScaleF32.dispatch
                    commands
                    {
                        X = token "scale.x"
                        Y = token "scale.y"
                        N = uint32 ScaleN
                        A = ScaleA
                    }

                Kernels.ScaleF32.dispatch
                    commands
                    {
                        X = token "scale.in_place"
                        Y = token "scale.in_place"
                        N = uint32 InPlaceN
                        A = ScaleA
                    }

                for src, y in
                    [
                        token "table_echo.target", "table_echo.y.view"
                        TensorToken.Null, "table_echo.y.null"
                        token "bf16.x", "table_echo.y.bf16"
                    ] do
                    Kernels.TableEcho.dispatch commands { Src = src; Y = token y }

                let a = argsWords

                Kernels.ArgsEcho.dispatch
                    commands
                    {
                        Y = token "args_echo.y"
                        V2 = a[0]
                        V3 = a[1]
                        V4 = a[2]
                        V5 = a[3]
                        V6 = a[4]
                        V7 = a[5]
                        V8 = a[6]
                        V9 = a[7]
                        V10 = a[8]
                        V11 = a[9]
                        V12 = a[10]
                        V13 = a[11]
                        V14 = a[12]
                        V15 = a[13]
                        V16 = a[14]
                        V17 = a[15]
                        V18 = a[16]
                        V19 = a[17]
                        V20 = a[18]
                        V21 = a[19]
                        V22 = a[20]
                        V23 = a[21]
                        V24 = a[22]
                        V25 = a[23]
                        V26 = a[24]
                        V27 = a[25]
                        V28 = a[26]
                        V29 = a[27]
                        V30 = a[28]
                        V31 = a[29]
                    }

                Kernels.GroupsEcho.dispatch
                    commands
                    {
                        Y = token "groups_echo.y.unfolded"
                        Groups = uint32 UnfoldedGroups
                    }

                Kernels.GroupsEcho.dispatch
                    commands
                    {
                        Y = token "groups_echo.y.folded"
                        Groups = uint32 FoldedGroups
                    }

                Kernels.Bf16Unpack.dispatch
                    commands
                    {
                        X = token "bf16.x"
                        Y = token "bf16.y"
                        N = uint32 Bf16N
                    }

            let completion = backend.Wait(backend.Submit commands)

            let cases =
                outputs
                |> List.map (fun (name, kernel, case, index, fp32, expected) ->
                    let n = expected.Length
                    let ws = poisoned[name]
                    let out = download name (capacity name)

                    let unwritten =
                        if name = "scale.in_place" then
                            inPlaceInputs
                        else
                            ws[.. n - 1]

                    let isClassOnly =
                        if fp32 then
                            let inputs =
                                if name = "scale.in_place" then
                                    inPlaceInputs
                                else
                                    scaleInputs

                            fun i -> classOnly mode ScaleA inputs[i]
                        else
                            fun _ -> false

                    let c = judge kernel case fp32 expected unwritten ws[n..] isClassOnly out

                    Telemetry.check
                        "gpu.selftest"
                        c.Passed
                        [
                            Attr.str TelemetryKeys.Backend backendName
                            Attr.str TelemetryKeys.Kernel kernel
                            Attr.str TelemetryKeys.Case case
                            Attr.int "elements" c.Elements
                            Attr.int "differing" c.Differing
                            Attr.int "not_written" c.NotWritten
                            Attr.int "clobbered" c.Clobbered
                            Attr.int "class_only" c.ClassOnly
                            Attr.str "mode" (Fp32Denormals.name mode)
                            Attr.str "detail" c.Detail
                        ]

                    ignore index
                    c)

            {
                Run = run
                Mode = mode
                Cases = cases
                Completion = completion
            }
