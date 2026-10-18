namespace Lrm.Gpu.Exec

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Models.Qwen

/// One compute op of a variant's static order (GpuExecutor contract, "Pass one").
type PlanOp =
    {
        /// Its place among the forward's compute ops (taps not counted): the order the ops are matched in.
        Index: int
        /// Its place among every call of the trace, taps included: the clock liveness is measured on.
        Sequence: int
        Kind: PlanOpKind
        Inputs: ValueRef list
        /// The result's capacity, [Rows, Cols] at the plan's largest T. Cols is 0 for an op without a result.
        Rows: int
        Cols: int
        /// The decoder layer the op belongs to (telemetry only); None for the embedding and the head.
        Layer: int option
    }

    member o.HasResult = o.Cols > 0

    /// Bytes of the result at capacity (fp32).
    member o.Bytes = int64 o.Rows * int64 o.Cols * 4L

/// One tap of a variant's static order.
type PlanTap =
    {
        Name: string
        Source: ValueRef
        Sequence: int
        Rows: int
        Cols: int
    }

/// Where the result of an op of a variant lives.
[<RequireQualifiedAccess>]
type ValueHome =
    /// The op has no result (KvWrite).
    | NoResult
    /// At this byte offset of the stage's activations pool.
    | Pool of offset: int64
    /// At the start of the stage's logits pool: the result of the stage with the head.
    | Logits

/// One head variant of a planned stage: the static order traced at the plan's largest T, its liveness and where every
/// result lives. A stage has the variant `all`; the stage with the head also has `last` (HeadRows.Last over more than
/// one row, which adds LastRow) when the plan's T is above 1.
type PlanVariant =
    {
        Name: string
        /// True for the variant whose forward takes the last row before lm_head.
        LastRow: bool
        Ops: PlanOp[]
        Taps: PlanTap list
        /// The op whose result the forward returns.
        Result: int
        Homes: ValueHome[]
        /// Per op: the Sequence of the result's last use (Int32.MaxValue for the forward's result).
        LastUse: int[]
        /// Per op: the Sequence of the first later op whose result overlaps it in the pool (Int32.MaxValue: none). A
        /// handle of the result is readable while the forward has not recorded that op.
        FirstClobber: int[]
        /// Bytes of the activations pool this variant needs.
        PoolBytes: int64
        /// Bytes of the logits pool it needs (0 for a stage without the head).
        LogitsBytes: int64
    }

/// The plan of one planned stage: its variants, pools and tokens.
type StagePlan =
    {
        Stage: Stage
        HasHead: bool
        /// `all`, then `last` when the stage has the head and the plan's T is above 1.
        Variants: PlanVariant[]
        /// The stage's activations pool (Activations arena), of which every pool result is a view.
        Pool: TensorToken
        /// The stage's logits (Logits arena), of which the head's result is a view; the null token without the head.
        LogitsPool: TensorToken
        /// Per variant, per op: the token of its result (the null token for an op without one).
        Values: TensorToken[][]
        /// Per (variant, op) of a split matmul: the x and y views of chunks 1 .. n - 1 (chunk 0 is the op's own tokens).
        Chunks: Map<int * int, TensorToken[] * TensorToken[]>
        /// The tap snapshots of this stage (Taps arena): name -> token [rows, cols] and cols.
        Taps: Map<string, TensorToken * int>
        /// The KV-cache pool (KvCache arena): per model layer of the stage, 2 · Caches tensors [MaxContext, KvDim].
        Caches: Map<int, TensorToken[]>
    }

/// The executor's static plan (GpuExecutor contract): the arenas laid out by HeapPlanner, the tensor table sealed
/// once, and per stage the static order with every result's token. It is a pure function of the model, the OpsOptions
/// and the GpuExecOptions, and touches no device.
type ExecPlan =
    {
        Model: Qwen3Model
        Options: OpsOptions
        Exec: GpuExecOptions
        /// The most rows one forward runs: min(MaxRows, MaxContext).
        MaxRows: int
        Layout: HeapLayout
        Table: SealedTable
        /// The weights the forward reads, in canonical order, with their shapes and tokens (bf16, read only).
        Weights: (string * int64[] * TensorToken) list
        /// cos and sin of T_spec for positions 0 .. MaxContext - 1, [MaxContext, head_dim / 2] f32 (read only).
        RopeCos: TensorToken
        RopeSin: TensorToken
        /// The ids of an embedding forward, u32 [MaxRows] (Scratch).
        Ids: TensorToken
        /// The buffer of IForwardOps.Upload, f32 [MaxRows, hidden] (Scratch).
        Upload: TensorToken
        /// The rows of argmax_rows, u32 [max(MaxRows, MaxContext)] (Logits).
        Argmax: TensorToken
        Stages: StagePlan list
    }

    /// The token of a weight by its safetensors name (lm_head resolves to the embedding when tied).
    member p.Weight(name: string) =
        match p.Weights |> List.tryFind (fun (n, _, _) -> n = name) with
        | Some(_, _, t) -> t
        | None -> raise (ForwardException(sprintf "the plan holds no weight '%s'" name))

    /// The plan of a planned stage.
    member p.StagePlan(stage: Stage) =
        match p.Stages |> List.tryFind (fun s -> s.Stage = stage) with
        | Some s -> s
        | None -> raise (ForwardException(sprintf "stage %d is not in the plan" stage.Index))

/// Liveness and the activations pool of one variant (GpuExecutor contract, "Memory").
[<RequireQualifiedAccess>]
module Liveness =
    /// The FirstClobber of a result nothing ever overwrites.
    [<Literal>]
    let Never = Int32.MaxValue

    let private align (bytes: int64) =
        Heap.alignUp Heap.AllocationAlignment bytes

    /// The layer of every op: the layer of its weight or cache, else the layer of the op before it; the embedding, the
    /// final norm and lm_head belong to no layer.
    let private layers (model: Qwen3Model) (kinds: (PlanOpKind * ValueRef list) list) =
        let byWeight = Dictionary<string, int>(StringComparer.Ordinal)

        model.Params.Layers
        |> Array.iteri (fun i p ->
            for w in
                [
                    p.InputNorm
                    p.Q
                    p.K
                    p.V
                    p.O
                    p.QNorm
                    p.KNorm
                    p.PostNorm
                    p.Gate
                    p.Up
                    p.Down
                ] do
                byWeight[w] <- i)

        let mutable current: int option = None

        [
            for kind, inputs in kinds do
                let cacheLayer =
                    inputs
                    |> List.tryPick (fun r ->
                        match r with
                        | ValueRef.Cache l -> Some l
                        | _ -> None)

                let layer =
                    match kind with
                    | PlanOpKind.Embed -> None
                    | PlanOpKind.RmsNorm(w, _)
                    | PlanOpKind.MatMul w ->
                        match byWeight.TryGetValue w with
                        | true, l -> Some l
                        | _ -> None
                    | _ ->
                        match cacheLayer with
                        | Some l -> Some l
                        | None -> current

                current <- layer
                yield layer
        ]

    /// A variant from a trace: its ops and taps in order, every result's last use, its home (the logits pool for the
    /// result of the stage with the head, else first fit in the activations pool among the results live at the same
    /// time), and FirstClobber.
    let variant (model: Qwen3Model) (name: string) (lastRow: bool) (hasHead: bool) (t: TracedForward) : PlanVariant =
        let computes =
            t.Calls
            |> List.indexed
            |> List.choose (fun (seq, call) ->
                match call with
                | TraceCall.Compute(kind, inputs, rows, cols) -> Some(seq, kind, inputs, rows, cols)
                | TraceCall.Tap _ -> None)

        let layerOf =
            layers model (computes |> List.map (fun (_, k, i, _, _) -> k, i))
            |> Array.ofList

        let ops =
            computes
            |> List.mapi (fun i (seq, kind, inputs, rows, cols) ->
                {
                    Index = i
                    Sequence = seq
                    Kind = kind
                    Inputs = inputs
                    Rows = rows
                    Cols = cols
                    Layer = layerOf[i]
                })
            |> Array.ofList

        let taps =
            t.Calls
            |> List.indexed
            |> List.choose (fun (seq, call) ->
                match call with
                | TraceCall.Tap(n, source, rows, cols) ->
                    Some
                        {
                            Name = n
                            Source = source
                            Sequence = seq
                            Rows = rows
                            Cols = cols
                        }
                | TraceCall.Compute _ -> None)

        let result =
            match t.Result with
            | ValueRef.Value p when p >= 0 && p < ops.Length && ops[p].HasResult -> p
            | r ->
                raise (ForwardException(sprintf "trace: the forward returned %A, not the result of one of its ops" r))

        let lastUse = ops |> Array.map _.Sequence

        let used (r: ValueRef) (at: int) =
            match r with
            | ValueRef.Value p -> lastUse[p] <- max lastUse[p] at
            | _ -> ()

        for o in ops do
            for r in o.Inputs do
                used r o.Sequence

        for tap in taps do
            used tap.Source tap.Sequence

        lastUse[result] <- Never

        let homes = Array.create ops.Length ValueHome.NoResult
        // Results live at the same time never share bytes: first fit among the results still live, in op order.
        let active = List<struct (int64 * int64 * int)>()
        let mutable poolBytes = 0L

        for o in ops do
            if o.HasResult then
                if hasHead && o.Index = result then
                    homes[o.Index] <- ValueHome.Logits
                else
                    active.RemoveAll(fun (struct (_, _, last)) -> last < o.Sequence) |> ignore
                    let placed = active |> Seq.sortBy (fun (struct (off, _, _)) -> off) |> List.ofSeq
                    let size = align o.Bytes
                    let mutable at = 0L
                    let mutable found = false

                    for struct (off, endAt, _) in placed do
                        if not found then
                            if at + size <= off then
                                found <- true
                            else
                                at <- max at (align endAt)

                    homes[o.Index] <- ValueHome.Pool at
                    active.Add(struct (at, at + o.Bytes, lastUse[o.Index]))
                    poolBytes <- max poolBytes (at + o.Bytes)

        let range (i: int) =
            match homes[i] with
            | ValueHome.Pool off -> Some(off, off + ops[i].Bytes)
            | _ -> None

        let firstClobber =
            ops
            |> Array.map (fun p ->
                match range p.Index with
                | None -> Never
                | Some(lo, hi) ->
                    ops
                    |> Array.filter (fun q ->
                        q.Sequence > p.Sequence
                        && (match range q.Index with
                            | Some(a, b) -> a < hi && lo < b
                            | None -> false))
                    |> Array.fold (fun m q -> min m q.Sequence) Never)

        {
            Name = name
            LastRow = lastRow
            Ops = ops
            Taps = taps
            Result = result
            Homes = homes
            LastUse = lastUse
            FirstClobber = firstClobber
            PoolBytes = poolBytes
            LogitsBytes = (if hasHead then ops[result].Bytes else 0L)
        }

[<RequireQualifiedAccess>]
module ExecPlan =
    /// The most rows one forward of the plan runs: min(MaxRows, MaxContext).
    let maxRows (o: OpsOptions) = min o.MaxRows o.MaxContext

    /// The most rows one embed_gather_bf16 dispatch takes (its grid y).
    [<Literal>]
    let MaxEmbedRows = 65535

    /// The kernels the executor dispatches; a backend's loaded set must hold every one.
    let kernels: KernelId list =
        [
            Kernels.EmbedGatherBf16.Id
            Kernels.RmsnormRows.Id
            Kernels.RopeApply.Id
            Kernels.SiluMul.Id
            Kernels.Add.Id
            Kernels.CopyRows.Id
            Kernels.ArgmaxRows.Id
            Kernels.MatmulBf16w.Id
            Kernels.AttentionGqaCausal.Id
        ]

    /// The weights the forward reads, in canonical order (Qwen3Weights.Names), with their shapes.
    let weightShapes (model: Qwen3Model) =
        let c = model.Config

        [
            yield! Qwen3Params.shapes c
            if model.Params.LmHead = Qwen3Params.LmHeadName then
                yield Qwen3Params.LmHeadName, [| int64 c.Vocab; int64 c.Hidden |]
        ]

    let private fits (what: string) (bytes: int64) =
        if bytes > Heap.MaxPageBytes then
            invalidArg
                "options"
                (sprintf "%s needs %d bytes, more than one 1 GiB page; plan fewer rows or positions" what bytes)

    /// Builds the plan (pass one): traces every planned stage, lays out the arenas, adds every tensor to the table and
    /// seals it. Raises ArgumentException for options outside OpsOptions.validate or GpuExecOptions.validate, a T above
    /// 65,535, or a tensor larger than a page; GpuContractException when the table or the heap overflows.
    let build (model: Qwen3Model) (options: OpsOptions) (exec: GpuExecOptions) : ExecPlan =
        let c = model.Config
        OpsOptions.validate c options
        GpuExecOptions.validate exec
        let tmax = maxRows options

        if tmax > MaxEmbedRows then
            invalidArg
                "options"
                (sprintf "a forward of %d rows exceeds %d, the rows of one embedding dispatch" tmax MaxEmbedRows)

        let stagePlans =
            options.Stages
            |> List.map (fun stage ->
                let hasHead = Stage.hasHead c stage
                let all = StaticOrder.trace model stage tmax HeadRows.All options.Taps
                let variants = [ Liveness.variant model "all" false hasHead all ]

                let variants =
                    if hasHead && tmax > 1 then
                        let last = StaticOrder.trace model stage tmax HeadRows.Last options.Taps
                        variants @ [ Liveness.variant model "last" true hasHead last ]
                    else
                        variants

                stage, hasHead, Array.ofList variants)

        let planner = HeapPlanner()
        let shapes = weightShapes model

        let weightPlaces =
            shapes
            |> List.map (fun (name, dims) ->
                let count = dims |> Array.fold (*) 1L
                name, dims, planner.Allocate(ArenaKind.Weights, Heap.alignUp 4L (count * 2L)))

        let half = c.HeadDim / 2
        let ropeBytes = int64 options.MaxContext * int64 half * 4L
        fits "the RoPE table" ropeBytes
        let cosPlace = planner.Allocate(ArenaKind.Rope, ropeBytes)
        let sinPlace = planner.Allocate(ArenaKind.Rope, ropeBytes)
        let idsPlace = planner.Allocate(ArenaKind.Scratch, int64 tmax * 4L)
        let uploadBytes = int64 tmax * int64 c.Hidden * 4L
        fits "the upload buffer" uploadBytes
        let uploadPlace = planner.Allocate(ArenaKind.Scratch, uploadBytes)
        let argmaxRows = max tmax options.MaxContext
        let argmaxPlace = planner.Allocate(ArenaKind.Logits, int64 argmaxRows * 4L)
        let cacheBytes = int64 options.MaxContext * int64 c.KvDim * 4L
        fits "a KV-cache tensor" cacheBytes

        let stagePlaces =
            stagePlans
            |> List.map (fun (stage, hasHead, variants) ->
                let caches =
                    [
                        for layer in stage.FirstLayer .. stage.LastLayer ->
                            layer,
                            Array.init (2 * options.Caches) (fun _ -> planner.Allocate(ArenaKind.KvCache, cacheBytes))
                    ]

                let poolBytes = variants |> Array.map _.PoolBytes |> Array.max
                fits (sprintf "the activations of stage %d" stage.Index) poolBytes
                let pool = planner.Allocate(ArenaKind.Activations, max 4L poolBytes)

                let logits =
                    if hasHead then
                        let b = variants |> Array.map _.LogitsBytes |> Array.max
                        fits (sprintf "the logits of stage %d" stage.Index) b
                        Some(planner.Allocate(ArenaKind.Logits, b))
                    else
                        None

                // One snapshot per tap the stage records, at the largest rows any variant gives it.
                let tapShapes =
                    variants
                    |> Array.collect (fun v -> v.Taps |> Array.ofList)
                    |> Array.groupBy _.Name
                    |> Array.map (fun (n, xs) -> n, (xs |> Array.map _.Rows |> Array.max), xs[0].Cols)
                    |> Array.sortBy (fun (n, _, _) ->
                        variants[0].Taps
                        |> List.tryFindIndex (fun t -> t.Name = n)
                        |> Option.defaultValue Int32.MaxValue)

                let taps =
                    tapShapes
                    |> Array.map (fun (n, rows, cols) ->
                        let b = int64 rows * int64 cols * 4L
                        fits (sprintf "tap %s" n) b
                        n, rows, cols, planner.Allocate(ArenaKind.Taps, b))

                caches, pool, poolBytes, logits, taps)

        let layout = planner.Build()
        let table = TensorTable(layout)

        let weights =
            weightPlaces
            |> List.map (fun (name, dims, place) ->
                name, dims, table.Add(name, place, DType.BF16, List.ofArray dims, TensorFlags.ReadOnly))

        let ropeDims = [ int64 options.MaxContext; int64 half ]
        let cos = table.Add("rope.cos", cosPlace, DType.F32, ropeDims, TensorFlags.ReadOnly)
        let sin = table.Add("rope.sin", sinPlace, DType.F32, ropeDims, TensorFlags.ReadOnly)

        let ids =
            table.Add("scratch.ids", idsPlace, DType.U32, [ int64 tmax ], TensorFlags.None)

        let upload =
            table.Add("scratch.upload", uploadPlace, DType.F32, [ int64 tmax; int64 c.Hidden ], TensorFlags.None)

        let argmax =
            table.Add("logits.argmax", argmaxPlace, DType.U32, [ int64 argmaxRows ], TensorFlags.None)

        let stages =
            List.zip stagePlans stagePlaces
            |> List.map (fun ((stage, hasHead, variants), (caches, pool, poolBytes, logits, taps)) ->
                let s = Invariant.format $"s{stage.Index}"

                let cacheTokens =
                    caches
                    |> List.map (fun (layer, places) ->
                        layer,
                        places
                        |> Array.mapi (fun j place ->
                            table.Add(
                                Invariant.format $"{s}.kv.L{layer}.{j}",
                                place,
                                DType.F32,
                                [ int64 options.MaxContext; int64 c.KvDim ],
                                TensorFlags.None
                            )))
                    |> Map.ofList

                let poolToken =
                    table.Add(s + ".pool", pool, DType.F32, [ max 1L (poolBytes / 4L) ], TensorFlags.None)

                let logitsToken =
                    match logits with
                    | Some place -> table.Add(s + ".logits", place, DType.F32, [ place.Bytes / 4L ], TensorFlags.None)
                    | None -> TensorToken.Null

                let valueName (v: PlanVariant) (o: PlanOp) =
                    let what =
                        match o.Kind with
                        | PlanOpKind.RmsNorm(w, _)
                        | PlanOpKind.MatMul w -> PlanOpKind.name o.Kind + "." + w
                        | k -> PlanOpKind.name k

                    Invariant.format $"{s}.{v.Name}.{o.Index}.{what}"

                let values =
                    variants
                    |> Array.map (fun v ->
                        v.Ops
                        |> Array.map (fun o ->
                            let dims = [ int64 o.Rows; int64 o.Cols ]

                            match v.Homes[o.Index] with
                            | ValueHome.NoResult -> TensorToken.Null
                            | ValueHome.Pool off -> table.AddView(valueName v o, poolToken, dims, off / 4L)
                            | ValueHome.Logits -> table.AddView(valueName v o, logitsToken, dims, 0L)))

                let chunks =
                    [
                        for vi in 0 .. variants.Length - 1 do
                            let v = variants[vi]

                            for o in v.Ops do
                                match o.Kind with
                                | PlanOpKind.MatMul w ->
                                    let k = shapes |> List.find (fun (n, _) -> n = w) |> snd |> fun d -> int d[1]

                                    let parts = Chunking.matmul exec.MatMulMacsPerDispatch o.Rows o.Cols k

                                    if parts.Length > 1 then
                                        let x =
                                            match o.Inputs with
                                            | [ ValueRef.Value p ] -> values[vi][p]
                                            | _ ->
                                                invalidArg
                                                    "options"
                                                    (sprintf
                                                        "%s splits into %d dispatches, but its input is not a result of the forward"
                                                        (PlanOpKind.describe o.Kind)
                                                        parts.Length)

                                        let name = valueName v o

                                        let views (source: TensorToken) (cols: int) (tag: string) =
                                            parts
                                            |> List.tail
                                            |> List.mapi (fun i (r0, n) ->
                                                table.AddView(
                                                    Invariant.format $"{name}.c{i + 1}.{tag}",
                                                    source,
                                                    [ int64 n; int64 cols ],
                                                    int64 r0 * int64 cols
                                                ))
                                            |> Array.ofList

                                        let y = values[vi][o.Index]
                                        yield (vi, o.Index), (views x k "x", views y o.Cols "y")
                                | _ -> ()
                    ]
                    |> Map.ofList

                let tapTokens =
                    taps
                    |> Array.map (fun (n, rows, cols, place) ->
                        n,
                        (table.Add(s + ".tap." + n, place, DType.F32, [ int64 rows; int64 cols ], TensorFlags.None),
                         cols))
                    |> Map.ofArray

                {
                    Stage = stage
                    HasHead = hasHead
                    Variants = variants
                    Pool = poolToken
                    LogitsPool = logitsToken
                    Values = values
                    Chunks = chunks
                    Taps = tapTokens
                    Caches = cacheTokens
                })

        {
            Model = model
            Options = options
            Exec = exec
            MaxRows = tmax
            Layout = layout
            Table = table.Seal()
            Weights = weights
            RopeCos = cos
            RopeSin = sin
            Ids = ids
            Upload = upload
            Argmax = argmax
            Stages = stages
        }
