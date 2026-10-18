namespace Lrm.Kernels.Conformance

open System
open System.Collections.Generic
open System.Diagnostics
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// How a harness session runs (ConformanceHarness contract).
type HarnessOptions =
    {
        /// Threads for the CpuRef references (at most 8 on the shared development machine).
        ReferenceThreads: int
        /// The dispatch budget: the faster of the two runs of every dispatch must be below it on a shader backend.
        BudgetNs: int64
    }

    static member Default =
        {
            ReferenceThreads = min 8 Environment.ProcessorCount
            BudgetNs = 100_000_000L
        }

/// The outcome of one case on one backend.
type CaseOutcome =
    {
        Case: KernelCase
        Backend: string
        Passed: bool
        /// The KernelBounds judgement of the first run against CpuRef in the class's mode (None when the run never got
        /// that far).
        Check: KernelCheck option
        /// Every rule the case broke, in the order checked.
        Failures: string list
        /// One line for reports: the failures, or what passed.
        Detail: string
        /// The faster of the two runs' dispatch time, in ns (GPU timers, or host time when the backend has none).
        TimeNs: int64
        TimeSource: string
        BudgetGating: bool
        /// Batch-invariance splits run and how many matched.
        Splits: int
        SplitsMatched: int
    }

/// Element sets of written tensors: the position of an element among the set's elements (written sets never overlap).
module internal Touch =
    /// Guard words before and after every tensor's elements (64 bytes each side).
    [<Literal>]
    let GuardWords = 16

    /// The index of element `e` among the elements of `s` in increasing order, or ValueNone when `e` is not in `s`.
    let indexOf (s: ElementSet) (e: int64) : int64 voption =
        if s.IsEmpty || e < s.Start || e >= s.End then
            ValueNone
        elif s.Count = 1L || s.Stride <= 0L then
            let i = e - s.Start
            if i < s.Elements then ValueSome i else ValueNone
        else
            let r = (e - s.Start) / s.Stride
            let within = e - (s.Start + r * s.Stride)

            if r < s.Count && within < s.Length then
                ValueSome(r * s.Length + within)
            else
                ValueNone

    /// The words at GuardWords + e of a region for every element e of `s`, in increasing order.
    let gather (s: ElementSet) (region: uint32[]) =
        let out = Array.zeroCreate<uint32>(int s.Elements)
        let mutable j = 0

        for struct (lo, hi) in s.Runs do
            let n = int (hi - lo)
            Array.blit region (GuardWords + int lo) out j n
            j <- j + n

        out

    /// The first index where two arrays differ.
    let firstDifference (a: uint32[]) (b: uint32[]) =
        let n = min a.Length b.Length
        let i = ReadOnlySpan(a, 0, n).CommonPrefixLength(ReadOnlySpan(b, 0, n))

        if i < n then Some i
        elif a.Length <> b.Length then Some n
        else None

    /// The element of `s` at index j (the inverse of indexOf).
    let elementAt (s: ElementSet) (j: int64) =
        if s.Count = 1L || s.Stride <= 0L then
            s.Start + j
        else
            s.Start + (j / s.Length) * s.Stride + j % s.Length

/// A tensor of a planned case: its token, and the region (64-byte guards around the elements) it owns in its page.
type internal PlannedTensor =
    {
        Tensor: CaseTensor
        Token: TensorToken
        Slot: int
        RegionOffset: int64
        /// Words of the region: 16 guard words, the elements in whole words, 16 guard words.
        RegionWords: int
        /// True when a written field is passed this tensor.
        Written: bool
        /// The input's words (elements, padded with zero to whole words); empty for outputs and meta tensors.
        InputWords: uint32[]
    }

    member t.DataWords = t.RegionWords - 2 * Touch.GuardWords

/// A case with its tensors placed in the heap and its table rows.
and internal PlannedCase =
    {
        Case: KernelCase
        Kernel: KernelId
        Spec: KernelSpec
        /// By tensor, in the case's tensor order.
        Tensors: PlannedTensor list
        /// The token of every token field (in place: the read field's token; null meta: the null token).
        Tokens: Map<string, TensorToken>
        /// The tensor of every token field that has one.
        FieldTensor: Map<string, PlannedTensor>
        /// The block with the tokens, word 0 left 0.
        Args: ArgBlock
        /// Batch-invariance splits with their view tokens and blocks.
        Splits: (Split * Map<string, TensorToken> * ArgBlock * Map<string, int64>) list
    }

/// The downloaded regions of one run, by tensor token, plus its dispatch timings.
and internal JobResult =
    {
        Regions: Map<uint32, uint32[]>
        Poison: Map<uint32, uint32[]>
        DispatchNs: int64 list
        HostNs: int64
    }

/// The conformance harness on one backend (ConformanceHarness contract). Create plans every case into one heap and
/// one table (the heap is written once), creates the heap, loads the kernels, writes the table and measures the fp32
/// denormal mode with the probe. Run judges one case: it runs it twice with fresh poison, checks that every written
/// word was written and nothing else, that the runs agree bit for bit, the KernelBounds bound against CpuRef in the
/// class's mode (on CpuDevice also CpuRef F32's exact bits), batch invariance over the case's splits and the dispatch
/// budget; then it emits one conformance.case check. It works with any IGpuBackend that has not created its heap yet.
and [<Sealed>] Session
    private
    (
        backend: IGpuBackend,
        heap: IGpuHeap,
        table: SealedTable,
        planned: Map<string, PlannedCase>,
        probe: PlannedCase,
        kernels: IKernelSet,
        options: HarnessOptions
    ) =
    let cls = backend.ExecutionClass
    let configuration = (ExecutionClass.id cls).Value.Hex
    let backendName = ExecutionBackend.name cls.Backend
    let exec = CpuExec.WithThreads(max 1 options.ReferenceThreads)
    let shaderBackend = ExecutionBackend.runsShaders cls.Backend
    let strict = cls.Backend = ExecutionBackend.CpuDevice
    let modelFtz = cls.Arch.Contains "+ftz"
    let modelNaN = cls.Arch.Contains "+nan-canonical"
    let mutable mode = Fp32DenormalMode.Preserve

    member _.Backend = backend
    member _.ExecutionClass = cls
    member _.BackendName = backendName

    /// The fp32 denormal mode the probe measured.
    member _.Mode = mode

    member _.Cases = planned |> Map.toList |> List.map (fun (_, p) -> p.Case)

    /// True when the backend's kernel set has the case's kernel (a GPU backend lacks kernels whose source is missing).
    member _.Contains(c: KernelCase) =
        match planned.TryFind c.Id with
        | Some p -> kernels.Contains p.Kernel
        | None -> false

    // ---- planning ---------------------------------------------------------------------------------------------------

    static member private Validate(c: KernelCase) : KernelId * KernelSpec =
        let fail fmt =
            Printf.kprintf (fun s -> invalidArg "case" (sprintf "case %s: %s" c.Id s)) fmt

        let id =
            match Kernels.tryFind c.Kernel with
            | Some id -> id
            | None -> fail "%s is not a catalog kernel" c.Kernel

        let spec = Kernels.all[id.Value]

        if (KernelBounds.tryFind c.Kernel).IsNone then
            fail "%s has no KernelBounds entry, so the harness refuses it" c.Kernel

        if String.IsNullOrWhiteSpace c.Name || c.Name.Contains '/' then
            fail "a case needs a name without '/'"

        let tokenFields =
            spec.Fields
            |> List.choose (fun f ->
                match f.Kind with
                | FieldKind.Token(access, dtypes, alias) -> Some(f.Name, (access, dtypes, alias))
                | _ -> None)
            |> Map.ofList

        let scalarFields =
            spec.Fields
            |> List.filter (fun f ->
                match f.Kind with
                | FieldKind.Token _ -> false
                | _ -> true)
            |> List.map _.Name

        for t in c.Io.Tensors do
            match tokenFields.TryFind t.Field with
            | None -> fail "%s is not a token field of %s" t.Field c.Kernel
            | Some(access, dtypes, _) ->
                match t.Data with
                | CaseData.Meta _ when access <> Access.Meta -> fail "%s is not a meta field" t.Field
                | CaseData.Input _
                | CaseData.Output _ when access = Access.Meta -> fail "meta field %s takes a Case.meta tensor" t.Field
                | CaseData.Output _ when not (Access.writes access) -> fail "%s is read, so it needs an input" t.Field
                | CaseData.Input _ when not (Access.reads access) ->
                    fail "%s is written and never read; give it an output (Case.output)" t.Field
                | _ -> ()

                if access <> Access.Meta && not (List.contains t.DType dtypes) then
                    fail "%s is %s, the kernel accepts %A" t.Field (DType.name t.DType) dtypes

                match t.Data with
                | CaseData.Input b when b.Length % DType.size t.DType <> 0 || b.Length = 0 ->
                    fail "input %s has %d bytes, not a positive multiple of %d" t.Field b.Length (DType.size t.DType)
                | CaseData.Output n
                | CaseData.Meta n when n < 1L -> fail "%s needs at least one element" t.Field
                | _ -> ()

                match t.Dims with
                | Some dims when List.fold (*) 1L dims <> t.Elements || dims.IsEmpty || dims.Length > 4 ->
                    fail "dims %A of %s do not give %d elements in rank 1..4" dims t.Field t.Elements
                | _ -> ()

                if Access.writes access && DType.size t.DType <> 4 then
                    fail
                        "the harness compares written tensors of 4-byte elements; %s is %s"
                        t.Field
                        (DType.name t.DType)

        let bound =
            (c.Io.Tensors |> List.map _.Field)
            @ (c.Io.InPlace |> List.map fst)
            @ c.Io.NullMeta

        for f in tokenFields |> Map.toList |> List.map fst do
            match bound |> List.filter ((=) f) |> List.length with
            | 1 -> ()
            | 0 -> fail "token field %s has no tensor" f
            | _ -> fail "token field %s is bound more than once" f

        for written, read in c.Io.InPlace do
            match tokenFields.TryFind written with
            | Some(access, _, alias) when Access.writes access && List.contains read alias -> ()
            | _ -> fail "%s may not alias %s (catalog alias)" written read

            if not (c.Io.Tensors |> List.exists (fun t -> t.Field = read)) then
                fail "in-place field %s needs a tensor for %s" written read

        for f in c.Io.NullMeta do
            match tokenFields.TryFind f with
            | Some(Access.Meta, _, _) -> ()
            | _ -> fail "only a meta field takes the null token (%s)" f

        let given = c.Scalars |> List.map fst

        for f in scalarFields do
            match given |> List.filter ((=) f) |> List.length with
            | 1 -> ()
            | 0 -> fail "scalar field %s has no value" f
            | _ -> fail "scalar field %s is given more than once" f

        for f in given do
            if not (List.contains f scalarFields) then
                fail "%s is not a scalar field of %s" f c.Kernel

        if
            (spec.Fields
             |> List.filter (fun f ->
                 match f.Kind with
                 | FieldKind.Token(a, _, _) -> Access.writes a
                 | _ -> false)
             |> List.length)
            <> 1
        then
            fail "the harness judges kernels with exactly one written field"

        // The kernel ABI's argument rules and minimum element counts (kernel-abi.md, "Tensor access"; KernelRules), so
        // that a case breaking one is refused here by name on every backend: a GPU-only case never reaches CpuDevice's
        // dispatch checks, and a GPU would run it out of bounds.
        let words = Array.zeroCreate<uint32> ArgBlock.WordCount

        spec.Fields
        |> List.iteri (fun i f ->
            match c.Scalars |> List.tryFind (fun (n, _) -> n = f.Name) with
            | Some(_, v) -> words[i + 1] <- v
            | None -> ())

        let tensorOf (field: string) =
            let read =
                c.Io.InPlace
                |> List.tryFind (fun (w, _) -> w = field)
                |> Option.map snd
                |> Option.defaultValue field

            c.Io.Tensors |> List.tryFind (fun t -> t.Field = read)

        try
            KernelRules.check
                spec
                (ArgBlock.OfWords words)
                (fun f -> tensorOf f |> Option.map _.Elements |> Option.defaultValue 0L)
                (fun f -> tensorOf f |> Option.map _.DType)
                (fun f ->
                    tensorOf f
                    |> Option.map (fun t -> c.Id + "/" + t.Field)
                    |> Option.defaultValue f)
            |> ignore
        with :? GpuContractException as e ->
            fail "%s" e.Message

        id, spec

    /// The scale_f32 denormal probe as a case (GpuBackendInterface): x = Fp32Denormals.probeInputs, a = 0.5.
    static member ProbeCase =
        Case.create "scale_f32" "fp32_denormal_probe"
        |> Case.f32Bits "x" Fp32Denormals.probeInputs
        |> Case.output "y" (int64 Fp32Denormals.probeInputs.Length)
        |> Case.u32 "n" (uint32 Fp32Denormals.probeInputs.Length)
        |> Case.f32Arg "a" Fp32Denormals.ProbeScale
        |> Case.doc "The fp32 denormal probe (GpuBackendInterface)"

    /// Plans `cases` (and the probe) into one heap of `backend`, whose kernels are loaded (`kernels`, from LoadKernels)
    /// and whose heap does not exist yet: CreateHeap and WriteTable run here, then the denormal probe. The class is
    /// known once kernels are loaded, so a caller can open a run log for it first. Raises ArgumentException for a
    /// malformed case.
    static member Create
        (backend: IGpuBackend, kernels: IKernelSet, cases: KernelCase list, options: HarnessOptions)
        : Session =
        let all = Session.ProbeCase :: cases

        match all |> List.countBy _.Id |> List.filter (fun (_, n) -> n > 1) with
        | (id, _) :: _ -> invalidArg "cases" (sprintf "case %s appears twice" id)
        | [] -> ()

        let validated = all |> List.map (fun c -> c, Session.Validate c)
        let planner = HeapPlanner()
        let guardBytes = int64 (4 * Touch.GuardWords)

        let placements =
            validated
            |> List.map (fun (c, _) ->
                c.Io.Tensors
                |> List.map (fun t ->
                    let dataBytes = Heap.alignUp 4L (t.Elements * int64 (DType.size t.DType))
                    t, planner.Allocate(ArenaKind.Scratch, dataBytes + 2L * guardBytes)))

        let layout = planner.Build()
        let tt = TensorTable layout

        let plannedCases =
            List.zip validated placements
            |> List.map (fun ((c, (id, spec)), places) ->
                let writtenFields =
                    spec.Fields
                    |> List.choose (fun f ->
                        match f.Kind with
                        | FieldKind.Token(a, _, _) when Access.writes a -> Some f.Name
                        | _ -> None)

                let tensors =
                    places
                    |> List.map (fun (t, p) ->
                        let written =
                            List.contains t.Field writtenFields
                            || c.Io.InPlace |> List.exists (fun (_, read) -> read = t.Field)

                        let flags =
                            match t.Data with
                            | CaseData.Input _ when not written -> TensorFlags.ReadOnly
                            | _ -> TensorFlags.None

                        let name = c.Id + "/" + t.Field
                        let dims = defaultArg t.Dims [ t.Elements ]
                        let slot = layout.SlotOf(p.Arena, p.Page)

                        let row =
                            TensorRow.create
                                slot
                                (p.Offset + guardBytes)
                                t.DType
                                dims
                                flags
                                (TensorRow.debugIdOf name)

                        let token = tt.Add(name, row)
                        let dataWords = int (Heap.alignUp 4L (t.Elements * int64 (DType.size t.DType)) / 4L)

                        let inputWords =
                            match t.Data with
                            | CaseData.Input b ->
                                let padded = Array.zeroCreate<byte>(dataWords * 4)
                                Buffer.BlockCopy(b, 0, padded, 0, b.Length)
                                MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan padded).ToArray()
                            | _ -> [||]

                        {
                            Tensor = t
                            Token = token
                            Slot = slot
                            RegionOffset = p.Offset
                            RegionWords = dataWords + 2 * Touch.GuardWords
                            Written = written
                            InputWords = inputWords
                        })

                let fieldTensor =
                    (tensors |> List.map (fun t -> t.Tensor.Field, t))
                    @ (c.Io.InPlace
                       |> List.map (fun (w, r) -> w, tensors |> List.find (fun t -> t.Tensor.Field = r)))
                    |> Map.ofList

                let tokens =
                    (fieldTensor |> Map.toList |> List.map (fun (f, t) -> f, t.Token))
                    @ (c.Io.NullMeta |> List.map (fun f -> f, TensorToken.Null))
                    |> Map.ofList

                let block (tokensOf: Map<string, TensorToken>) (scalars: (string * uint32) list) =
                    let w = Array.zeroCreate<uint32> ArgBlock.WordCount

                    spec.Fields
                    |> List.iteri (fun i f ->
                        w[i + 1] <-
                            match f.Kind with
                            | FieldKind.Token _ -> tokensOf[f.Name].Value
                            | _ -> scalars |> List.find (fun (n, _) -> n = f.Name) |> snd)

                    ArgBlock.OfWords w

                let args = block tokens c.Scalars

                // Batch-invariance splits: views share one token per (tensor, offset, length).
                let viewTokens = Dictionary<(uint32 * int64 * int64), TensorToken>()

                let splits =
                    References.splits spec args
                    |> List.mapi (fun i s ->
                        let viewOf (field: string) (offset: int64) (count: int64) =
                            let source = fieldTensor[field]
                            let key = (source.Token.Value, offset, count)

                            match viewTokens.TryGetValue key with
                            | true, t -> t
                            | _ ->
                                let name = Invariant.format $"{c.Id}/split{i}/{field}"
                                let t = tt.AddView(name, source.Token, [ count ], offset)
                                viewTokens[key] <- t
                                t

                        let viewed =
                            s.Views |> List.map (fun (f, off, n) -> f, viewOf f off n) |> Map.ofList

                        let offsets = s.Views |> List.map (fun (f, off, _) -> f, off) |> Map.ofList

                        let toks =
                            tokens
                            |> Map.map (fun f t ->
                                match viewed.TryFind f with
                                | Some v -> v
                                | None -> t)

                        let scalars =
                            c.Scalars
                            |> List.map (fun (n, v) ->
                                match s.Scalars |> List.tryFind (fun (m, _) -> m = n) with
                                | Some(_, v') -> n, v'
                                | None -> n, v)

                        s, toks, block toks scalars, offsets)

                {
                    Case = c
                    Kernel = id
                    Spec = spec
                    Tensors = tensors
                    Tokens = tokens
                    FieldTensor = fieldTensor
                    Args = args
                    Splits = splits
                })

        let sealedTable = tt.Seal()
        let heap = backend.CreateHeap layout
        heap.WriteTable sealedTable

        let probe = plannedCases.Head

        let map = plannedCases.Tail |> List.map (fun p -> p.Case.Id, p) |> Map.ofList

        let s = new Session(backend, heap, sealedTable, map, probe, kernels, options)
        s.MeasureMode()
        s

    /// Loads the kernels of `shaders`, then Create.
    static member Load(backend: IGpuBackend, shaders: ShaderSet, cases: KernelCase list, options: HarnessOptions) =
        Session.Create(backend, backend.LoadKernels shaders, cases, options)

    // ---- running ----------------------------------------------------------------------------------------------------

    /// The physical grid's x for a block, as Dispatch writes it into word 0.
    static member private PhysicalX(spec: KernelSpec, args: ArgBlock) =
        match Grid.physical spec.Linear (KernelSpec.gridOf spec args) with
        | Ok p -> uint32 p.X
        | Error e -> invalidOp (sprintf "%s: %s" spec.Name e)

    /// One run of a planned case (or of one of its splits): upload every input with poisoned guards, poison every
    /// output region, dispatch (unless `control`, the negative control without dispatches), and download every region.
    /// `expected` gives the reference word of a written element so that no poison word equals it.
    member private _.Job
        (p: PlannedCase, args: ArgBlock, label: string, control: bool, expected: PlannedTensor -> int -> uint32 voption)
        : JobResult =
        let regions = Dictionary<uint32, uint32[]>()
        let poisons = Dictionary<uint32, uint32[]>()

        for t in p.Tensors do
            let poison =
                Poison.words (Poison.seed configuration p.Case.Id (label + "/" + t.Tensor.Field)) t.RegionWords

            let words = Array.copy poison

            match t.Tensor.Data with
            | CaseData.Input _ -> Array.blit t.InputWords 0 words Touch.GuardWords t.InputWords.Length
            | CaseData.Output _ ->
                for i in 0 .. t.DataWords - 1 do
                    match expected t i with
                    | ValueSome e -> words[Touch.GuardWords + i] <- Poison.avoiding e words[Touch.GuardWords + i]
                    | ValueNone -> ()
            | CaseData.Meta _ -> ()

            poisons[t.Token.Value] <- words

            match t.Tensor.Data with
            | CaseData.Meta _ -> ()
            | _ -> heap.Upload(t.Slot, t.RegionOffset, MemoryMarshal.AsBytes(ReadOnlySpan words))

        let clock = Stopwatch.StartNew()
        let commands = backend.Begin()

        if not control then
            commands.Dispatch(p.Kernel, args)

        let completion = backend.Wait(backend.Submit commands)
        let hostNs = clock.Elapsed.Ticks * 100L

        for t in p.Tensors do
            match t.Tensor.Data with
            | CaseData.Meta _ -> ()
            | _ ->
                let bytes = Array.zeroCreate<byte>(4 * t.RegionWords)
                heap.Download(t.Slot, t.RegionOffset, Span bytes)
                regions[t.Token.Value] <- MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()

        {
            Regions = regions |> Seq.map (fun kv -> kv.Key, kv.Value) |> Map.ofSeq
            Poison = poisons |> Seq.map (fun kv -> kv.Key, kv.Value) |> Map.ofSeq
            DispatchNs =
                completion.Timers
                |> List.filter (fun t -> t.Name = GpuTelemetry.Dispatch)
                |> List.map _.DurationNs
            HostNs = hostNs
        }

    /// The reference of a case's full dispatch.
    member private _.Reference(p: PlannedCase, m: Fp32DenormalMode) =
        let args = p.Args.WithGroupsX(Session.PhysicalX(p.Spec, p.Args))

        References.compute
            {
                Spec = p.Spec
                Args = args
                Input =
                    fun field ->
                        let t = p.FieldTensor[field]

                        match t.Tensor.Data with
                        | CaseData.Input b -> b
                        | _ -> invalidOp (sprintf "%s has no input data" field)
                Row = fun field -> table.Row p.Tokens[field]
                Mode = m
                Exec = exec
            }

    member private this.MeasureMode() =
        let p = probe
        let none = fun (_: PlannedTensor) (_: int) -> ValueNone
        let r1 = this.Job(p, p.Args, "run1", false, none)
        let r2 = this.Job(p, p.Args, "run2", false, none)
        let y = p.FieldTensor["y"]
        let n = Fp32Denormals.probeInputs.Length

        let words (r: JobResult) =
            Array.sub r.Regions[y.Token.Value] Touch.GuardWords n

        for i in 0 .. n - 1 do
            if (words r1)[i] = r1.Poison[y.Token.Value][Touch.GuardWords + i] then
                invalidOp (sprintf "the fp32 denormal probe left its poison in word %d: scale_f32 wrote nothing" i)

        if words r1 <> words r2 then
            invalidOp "the fp32 denormal probe differs between its two runs"

        mode <- Fp32Denormals.recordProbe cls.Backend (words r1)

    /// The strict CpuDevice expectation of a written word: CpuRef F32 in the class's mode, with the device's model of
    /// fp32 results applied (flush and canonical NaN, ConformanceHarness contract, "CpuDevice").
    member private _.Strict(p: PlannedCase, t: PlannedTensor, bits: uint32) =
        let f32 = t.Tensor.DType = DType.F32
        let kernel = p.Case.Kernel

        let flushed =
            if f32 && modelFtz && (KernelBounds.find kernel).Kind = KernelBoundKind.Relative then
                Fp32Denormals.flushBits bits
            else
                bits

        let nan = flushed &&& 0x7F800000u = 0x7F800000u && flushed &&& 0x007FFFFFu <> 0u

        if
            f32
            && modelNaN
            && nan
            && References.followsMode kernel
            && kernel <> "argmax_rows"
        then
            0x7FFFFFFFu
        else
            flushed

    /// Runs and judges one case (ConformanceHarness contract). With `control`, the negative control: the same runs
    /// with the dispatches removed, which must fail; it emits no conformance.case record.
    member this.Judge(c: KernelCase, control: bool) : CaseOutcome =
        let p =
            match planned.TryFind c.Id with
            | Some p -> p
            | None -> invalidArg "c" (sprintf "case %s is not in this session" c.Id)

        let failures = ResizeArray<string>()
        let fail (s: string) = failures.Add s
        let refs = this.Reference(p, mode)
        let out = refs |> List.exactlyOne
        let y = p.FieldTensor[out.Field]
        let inPlace = c.Io.InPlace |> List.isEmpty |> not
        let g = Touch.GuardWords
        let touched = out.Touched

        if touched.Elements <> int64 out.Bits.Length then
            invalidOp (
                sprintf "%s: the reference has %d words for %d touched elements" c.Id out.Bits.Length touched.Elements
            )

        let expected (t: PlannedTensor) (i: int) =
            if t.Token = y.Token then
                match Touch.indexOf touched (int64 i) with
                | ValueSome j -> ValueSome out.Bits[int j]
                | ValueNone -> ValueNone
            else
                ValueNone

        if inPlace then
            let inputs = Touch.gather touched (Array.append (Array.zeroCreate g) y.InputWords)

            if Array.forall2 (=) inputs out.Bits then
                fail "an in-place case whose every result equals its input cannot detect a missing write"

        /// The rules every run keeps: guards intact, nothing written outside the written elements, inputs unchanged,
        /// and no written word left at its poison.
        let checkRegions (label: string) (r: JobResult) (written: ElementSet) =
            for t in p.Tensors do
                match t.Tensor.Data with
                | CaseData.Meta _ -> ()
                | data ->
                    let words = r.Regions[t.Token.Value]
                    let before = r.Poison[t.Token.Value]
                    let guard = [ 0 .. g - 1 ] @ [ t.RegionWords - g .. t.RegionWords - 1 ]

                    match guard |> List.tryFind (fun i -> words[i] <> before[i]) with
                    | Some i ->
                        let at = if i < g then i - g else i - (t.RegionWords - g)

                        fail (
                            Invariant.format
                                $"{label}: {t.Tensor.Field} guard word {at} changed (written outside the tensor)"
                        )
                    | None -> ()

                    let isOut = t.Token = y.Token
                    let mutable changed = 0
                    let mutable first = -1
                    let mutable left = 0
                    let mutable firstLeft = -1

                    for i in 0 .. t.DataWords - 1 do
                        let w = words[g + i]
                        let b = before[g + i]

                        if isOut && written.Contains(int64 i) then
                            if not inPlace && w = b then
                                left <- left + 1

                                if firstLeft < 0 then
                                    firstLeft <- i
                        elif w <> b then
                            changed <- changed + 1

                            if first < 0 then
                                first <- i

                    if left > 0 then
                        fail (
                            Invariant.format
                                $"{label}: {left} written word(s) of {t.Tensor.Field} still hold their poison, first at element {firstLeft} (nothing was written there)"
                        )

                    if changed > 0 then
                        let what =
                            match data with
                            | CaseData.Input _ when not t.Written -> "an input changed"
                            | _ -> "written outside the written elements"

                        fail (
                            Invariant.format
                                $"{label}: {changed} word(s) of {t.Tensor.Field} changed, first at element {first} ({what})"
                        )

        let r1 = this.Job(p, p.Args, "run1", control, expected)
        let r2 = this.Job(p, p.Args, "run2", control, expected)
        checkRegions "run 1" r1 touched
        checkRegions "run 2" r2 touched
        let w1 = Touch.gather touched r1.Regions[y.Token.Value]
        let w2 = Touch.gather touched r2.Regions[y.Token.Value]
        let twice = Touch.firstDifference w1 w2

        match twice with
        | Some i -> fail (Invariant.format $"runs 1 and 2 differ at element {Touch.elementAt touched (int64 i)}")
        | None -> ()

        // The bound, against CpuRef in the class's measured mode.
        let bound = KernelBounds.find c.Kernel

        let included =
            match out.WithinClassOnly with
            | Some within -> Array.map not within
            | None -> Array.create w1.Length true

        let excluded = included |> Array.filter not |> Array.length

        let pick (xs: 'a[]) =
            if excluded = 0 then
                xs
            else
                Array.zip xs included |> Array.filter snd |> Array.map fst

        let check =
            match bound.Kind with
            | KernelBoundKind.BitExact ->
                let bytes (ws: uint32[]) =
                    MemoryMarshal.AsBytes(ReadOnlySpan(pick ws)).ToArray()

                KernelBounds.checkExact c.Kernel 4 (ReadOnlySpan(bytes w1)) (ReadOnlySpan(bytes out.Bits))
            | KernelBoundKind.Relative when excluded = w1.Length ->
                {
                    Kernel = c.Kernel
                    Kind = bound.Kind
                    Measured = 0.0
                    ReferenceError = 0.0
                    Bound = 0.0
                    Passed = true
                    Elements = 0L
                }
            | KernelBoundKind.Relative ->
                let f64 (ws: uint32[]) =
                    pick ws |> Array.map (fun w -> float (BitConverter.UInt32BitsToSingle w))

                KernelBounds.checkRelative
                    c.Kernel
                    (ReadOnlySpan(f64 w1))
                    (ReadOnlySpan(f64 out.Bits))
                    (ReadOnlySpan(pick out.F64.Value))

        if not check.Passed then
            fail (
                match check.Kind with
                | KernelBoundKind.BitExact ->
                    let at =
                        match Touch.firstDifference w1 out.Bits with
                        | Some i -> Touch.elementAt touched (int64 i)
                        | None -> -1L

                    Invariant.format
                        $"{check.Measured} of {check.Elements} elements differ from CpuRef F32 ({Fp32Denormals.name mode}), first at element {at}"
                | KernelBoundKind.Relative ->
                    Invariant.format
                        $"relL2 {Invariant.f64 check.Measured} exceeds the bound {Invariant.f64 check.Bound} (reference error {Invariant.f64 check.ReferenceError})"
            )

        if strict then
            let want = out.Bits |> Array.map (fun b -> this.Strict(p, y, b))

            match Touch.firstDifference w1 want with
            | Some i ->
                fail (
                    Invariant.format
                        $"CpuDevice must equal CpuRef F32 bit for bit: element {Touch.elementAt touched (int64 i)} is {w1[i]:x8}, CpuRef F32 gives {want[i]:x8}"
                )
            | None -> ()

        // Batch invariance: each split, alone, writes the same bits at the same positions, and nothing else.
        let mutable matched = 0

        for i, (s, _, sargs, offsets) in p.Splits |> List.indexed do
            let fp =
                KernelRules.footprint p.Spec sargs (fun f -> p.FieldTensor.TryFind f |> Option.map _.Tensor.DType)
                |> List.find (fun f -> f.Field = out.Field)

            let shift = defaultArg (offsets.TryFind out.Field) 0L

            let written =
                { fp.Elements with
                    Start = fp.Elements.Start + shift
                }

            let label = "split " + s.Label
            let before = failures.Count
            let r = this.Job(p, sargs, Invariant.format $"split{i}", control, expected)
            checkRegions label r written
            let alone = Touch.gather written r.Regions[y.Token.Value]
            let whole = Touch.gather written r1.Regions[y.Token.Value]

            match Touch.firstDifference alone whole with
            | Some j ->
                fail (
                    Invariant.format
                        $"batch invariance: {label} gives {alone[j]:x8} at element {Touch.elementAt written (int64 j)}, the whole case {whole[j]:x8}"
                )
            | None -> ()

            if failures.Count = before then
                matched <- matched + 1

        // Timing: the faster of the two runs, per dispatch, from the backend's gpu.dispatch timers when it has them.
        // CpuDevice's timers are host time around each emulated dispatch, so they are labelled emulated, not gpu.
        let timeNs, source =
            match r1.DispatchNs, r2.DispatchNs with
            | xs, ys when not xs.IsEmpty && xs.Length = ys.Length ->
                List.map2 min xs ys |> List.max, (if strict then "emulated" else "gpu")
            | _ -> min r1.HostNs r2.HostNs, "host"

        if shaderBackend && not control && timeNs >= options.BudgetNs then
            fail (
                Invariant.format
                    $"the dispatch takes {float timeNs / 1e6:F3} ms (the faster of two runs), over the {float options.BudgetNs / 1e6:F0} ms budget"
            )

        let failed = List.ofSeq failures
        let passed = failed.IsEmpty

        let detail =
            if passed then
                let what =
                    match check.Kind with
                    | KernelBoundKind.BitExact when strict -> "CpuRef F32 bits"
                    | KernelBoundKind.BitExact ->
                        Invariant.format $"bit-exact to CpuRef F32 ({Fp32Denormals.name mode})"
                    | KernelBoundKind.Relative when strict ->
                        Invariant.format
                            $"CpuRef F32 bits; relL2 {Invariant.f64 check.Measured} within {Invariant.f64 check.Bound}"
                    | KernelBoundKind.Relative ->
                        Invariant.format $"relL2 {Invariant.f64 check.Measured} within {Invariant.f64 check.Bound}"

                let within =
                    if excluded > 0 then
                        Invariant.format $"; {excluded} element(s) compared within the class only (mixed)"
                    else
                        ""

                let splits =
                    if p.Splits.IsEmpty then
                        "no batch splits"
                    else
                        Invariant.format $"batch {matched}/{p.Splits.Length}"

                Invariant.format $"{what}; twice identical; {splits}{within}"
            else
                String.Join("; ", failed)

        let outcome =
            {
                Case = c
                Backend = backendName
                Passed = passed
                Check = Some check
                Failures = failed
                Detail = detail
                TimeNs = timeNs
                TimeSource = source
                BudgetGating = shaderBackend
                Splits = p.Splits.Length
                SplitsMatched = matched
            }

        if not control then
            Telemetry.check
                TelemetryNames.ConformanceCase
                passed
                (KernelBounds.attrs check
                 @ [
                     Attr.str TelemetryKeys.Case c.Name
                     Attr.str TelemetryKeys.Backend backendName
                     Attr.str "detail" detail
                     Attr.str "xclass" configuration
                     Attr.str "fp32_denormals" (Fp32Denormals.name mode)
                     Attr.bool "strict" strict
                     Attr.bool "twice_identical" twice.IsNone
                     Attr.int "batch_splits" p.Splits.Length
                     Attr.int "batch_matched" matched
                     Attr.int "within_class_only" excluded
                     Attr.int64 "time_ns" timeNs
                     Attr.str "time_source" source
                     Attr.int64 "budget_ns" options.BudgetNs
                     Attr.bool "budget_gating" shaderBackend
                 ])

        outcome

    /// Runs and judges one case, emitting its conformance.case check.
    member this.Run(c: KernelCase) = this.Judge(c, false)

    /// The negative control of one case: the dispatches removed. It must fail (poison left, or the result of an
    /// in-place case not computed). Emits nothing.
    member this.Control(c: KernelCase) = this.Judge(c, true)

    interface IDisposable with
        member _.Dispose() = backend.Dispose()
