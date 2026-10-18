namespace Lrm.Gpu.Exec

open System
open System.Collections.Generic
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Gpu.Core
open Lrm.Models.Qwen

/// What a GpuTensor holds.
[<RequireQualifiedAccess>]
type GpuTensorKind =
    /// The result of a compute op of a forward: a view of its stage's activations pool, or of its logits.
    | Value
    /// A tap snapshot, copied into the Taps arena when the tap is recorded.
    | Tap
    /// A KV-cache tensor of the KvCache arena (IForwardOps.NewCache).
    | Cache
    /// The handle of IForwardOps.Upload, in the Scratch arena.
    | Upload

/// The handle of the GPU executor (IForwardOps<GpuTensor>): a tensor token of the sealed table and the logical
/// [Rows, Cols] it holds. The token's row is a capacity; kernels take the actual sizes from their fields.
[<Sealed; AllowNullLiteral>]
type GpuTensor
    internal
    (
        owner: obj,
        kind: GpuTensorKind,
        token: TensorToken,
        slot: int,
        offset: int64,
        rows: int,
        cols: int,
        stage: int,
        position: int,
        forward: int64,
        layer: int,
        cacheSlot: int
    ) =
    let mutable dead = false

    member _.Kind = kind
    member _.Token = token
    member _.Rows = rows
    member _.Cols = cols

    /// The heap slot and the byte offset of element 0.
    member _.Slot = slot
    member _.Offset = offset

    /// The stage index of a value or tap; -1 otherwise.
    member _.Stage = stage

    /// The op of the static order whose result a value is; -1 otherwise.
    member _.Position = position

    /// The model layer of a cache tensor; -1 otherwise.
    member _.Layer = layer

    member internal _.Owner = owner
    member internal _.Forward = forward
    member internal _.CacheSlot = cacheSlot
    member internal _.Dead = dead
    member internal _.Kill() = dead <- true

    override _.ToString() =
        sprintf "GpuTensor(%A %O [%d, %d])" kind token rows cols

/// A byte range of one heap slot.
[<Struct>]
type internal ByteRange = { Slot: int; Lo: int64; Hi: int64 }

/// The accesses recorded since the last barrier of a command list: a dispatch that reads bytes a pending dispatch
/// writes, or writes bytes a pending dispatch reads or writes, needs a Barrier first (GpuBackendInterface, Barrier).
[<Sealed>]
type internal Hazards() =
    let reads = List<ByteRange>()
    let writes = List<ByteRange>()

    static member Overlap (a: ByteRange) (b: ByteRange) =
        a.Slot = b.Slot && a.Lo < b.Hi && b.Lo < a.Hi

    member _.Conflicts(r: ByteRange list, w: ByteRange list) =
        w
        |> List.exists (fun x ->
            writes.Exists(fun y -> Hazards.Overlap x y)
            || reads.Exists(fun y -> Hazards.Overlap x y))
        || r |> List.exists (fun x -> writes.Exists(fun y -> Hazards.Overlap x y))

    member _.Add(r: ByteRange list, w: ByteRange list) =
        reads.AddRange r
        writes.AddRange w

    member _.Clear() =
        reads.Clear()
        writes.Clear()

/// IForwardOps on any IGpuBackend through the static plan (GpuExecutor contract). Pass one (ExecPlan.build) traced
/// the forward, laid out the arenas and sealed the table; pass two is this type: each forward is matched op by op
/// against its stage's static order and recorded into one command list of typed ArgBlocks (Kernels.g.fs), with a
/// barrier before every dependent dispatch and a Copy into the Taps arena for every tap; EndForward submits it and
/// waits. It enforces the plan and the handle lifetimes exactly as CpuOps does.
[<Sealed>]
type GpuOps internal (backend: IGpuBackend, heap: IGpuHeap, plan: ExecPlan, xclass: ExecutionClass) =
    let c = plan.Model.Config
    let options = plan.Options
    let exec = plan.Exec
    let table = plan.Table
    let owner = obj ()
    let scale = CpuRef.attentionScale<float32> c.HeadDim
    let backendKind = backend.Info.Backend
    let backendName = ExecutionBackend.name backendKind

    let weights =
        let d = Dictionary<string, int64[] * TensorToken>(StringComparer.Ordinal)

        for name, dims, token in plan.Weights do
            d[name] <- (dims, token)

        d

    let location (t: TensorToken) =
        let r = table.Row t
        r.Slot, int64 r.ByteOffset

    let idsAt = location plan.Ids
    let uploadAt = location plan.Upload
    let argmaxAt = location plan.Argmax
    let argmaxRows = int (table.Row plan.Argmax).ElemCount
    let uploadElements = int64 (table.Row plan.Upload).ElemCount

    // The plan's KV-cache pool: free tensors per (stage, layer).
    let freeCaches = Dictionary<struct (int * int), Stack<int>>()

    do
        for sp in plan.Stages do
            for KeyValue(layer, tokens) in sp.Caches do
                let s = Stack<int>()

                for i in tokens.Length - 1 .. -1 .. 0 do
                    s.Push i

                freeCaches[struct (sp.Stage.Index, layer)] <- s

    let liveCaches = HashSet<GpuTensor>(HashIdentity.Reference)
    let made = List<GpuTensor>()
    let tapped = Dictionary<string, GpuTensor>(StringComparer.Ordinal)
    let kept = Dictionary<int, GpuTensor list>()
    let steps = Dictionary<int, int>()
    let hazards = Hazards()
    let mutable current: (ForwardCall * StagePlan * int) option = None
    let mutable serial = 0L
    let mutable cursorIndex = 0
    let mutable cursorSeq = -1
    let mutable step = 0
    let mutable phase = "prefill"
    let mutable upload: GpuTensor option = None
    let mutable commands: ICommandList option = None
    let mutable recorded = 0
    let mutable dispatches = 0L
    let mutable submissions = 0L
    let mutable disposed = false

    let live () =
        if disposed then
            raise (ObjectDisposedException("GpuForwardOps", "the ops were disposed"))

    let inside (op: string) =
        live ()

        if current.IsNone then
            raise (ForwardException(sprintf "%s outside a forward (BeginForward .. EndForward)" op))

    let outside (op: string) =
        live ()

        if current.IsSome then
            raise (ForwardException(sprintf "%s inside a forward" op))

    let alive (what: string) (x: GpuTensor) =
        live ()

        if isNull x then
            nullArg what

        if not (obj.ReferenceEquals(x.Owner, owner)) then
            raise (ForwardException(sprintf "%s is a handle of other ops" what))

        if x.Dead then
            raise (
                ForwardException(
                    sprintf "%s: the handle is no longer valid (ForwardOpsAndQwenModel, \"Handle lifetimes\")" what
                )
            )

        match current with
        | Some(_, sp, vi) when x.Kind = GpuTensorKind.Value && x.Forward = serial && x.Stage = sp.Stage.Index ->
            let v = sp.Variants[vi]

            if cursorSeq >= v.FirstClobber[x.Position] then
                raise (
                    ForwardException(
                        sprintf
                            "%s: the result of op %d was overwritten after its last planned use, as the static plan reuses its bytes (GpuExecutor, \"Memory\")"
                            what
                            x.Position
                    )
                )
        | _ -> ()

        x

    let planned (stage: Stage) =
        if not (options.Stages |> List.contains stage) then
            raise (
                ForwardException(
                    sprintf
                        "stage %d (layers %d .. %d) is not in the ops' plan (OpsOptions.Stages)"
                        stage.Index
                        stage.FirstLayer
                        stage.LastLayer
                )
            )

    let commandList () =
        match commands with
        | Some l -> l
        | None ->
            let l = backend.Begin()
            commands <- Some l
            recorded <- 0
            hazards.Clear()
            l

    /// Submits what is recorded and waits: the only place results reach memory the host can read.
    let flush () =
        match commands with
        | Some l ->
            commands <- None
            hazards.Clear()

            if recorded > 0 then
                recorded <- 0
                let s = backend.Submit l
                submissions <- submissions + 1L
                backend.Wait s |> ignore
        | None -> ()

    let discard () =
        commands <- None
        recorded <- 0
        hazards.Clear()

    let rangeOf (x: GpuTensor) (row0: int) (rows: int) =
        let rowBytes = int64 x.Cols * 4L

        {
            Slot = x.Slot
            Lo = x.Offset + int64 row0 * rowBytes
            Hi = x.Offset + int64 (row0 + rows) * rowBytes
        }

    let dispatch
        (kernel: KernelId)
        (layer: int option)
        (args: ArgBlock)
        (reads: ByteRange list)
        (writes: ByteRange list)
        =
        let l = commandList ()

        if hazards.Conflicts(reads, writes) then
            l.Barrier()
            recorded <- recorded + 1
            hazards.Clear()

        hazards.Add(reads, writes)

        let timer =
            if exec.TimeDispatches && Telemetry.enabled () then
                Some(
                    l.BeginTimer(
                        GpuTelemetry.Dispatch,
                        [
                            yield Attr.str TelemetryKeys.Kernel Kernels.all[kernel.Value].Name
                            match layer with
                            | Some i -> yield Attr.int TelemetryKeys.Layer i
                            | None -> ()
                            yield Attr.int "step" step
                            yield Attr.str "phase" phase
                        ]
                    )
                )
            else
                None

        l.Dispatch(kernel, args)
        recorded <- recorded + 1
        dispatches <- dispatches + 1L
        timer |> Option.iter l.EndTimer

    let weight (name: string) =
        match weights.TryGetValue name with
        | true, w -> w
        | _ -> raise (ForwardException(sprintf "the plan holds no weight '%s'" name))

    let refOf (sp: StagePlan) (x: GpuTensor) =
        match x.Kind with
        | GpuTensorKind.Value when x.Stage = sp.Stage.Index && x.Forward = serial -> ValueRef.Value x.Position
        | GpuTensorKind.Cache -> ValueRef.Cache x.Layer
        | _ -> ValueRef.Input

    /// The next op of the static order: the call must be it, with the planned inputs.
    let next (kind: PlanOpKind) (inputs: (string * GpuTensor) list) =
        let _, sp, vi = current.Value

        for what, x in inputs do
            alive what x |> ignore

        let v = sp.Variants[vi]

        if cursorIndex >= v.Ops.Length then
            raise (
                ForwardException(
                    sprintf
                        "%s: the forward has more ops than the static plan of stage %d (%d)"
                        (PlanOpKind.describe kind)
                        sp.Stage.Index
                        v.Ops.Length
                )
            )

        let o = v.Ops[cursorIndex]
        let refs = inputs |> List.map (snd >> refOf sp)

        if o.Kind <> kind || o.Inputs <> refs then
            raise (
                ForwardException(
                    sprintf
                        "op %d departs from the static plan of stage %d (%s): the plan has %s of %A, the forward called %s of %A"
                        cursorIndex
                        sp.Stage.Index
                        v.Name
                        (PlanOpKind.describe o.Kind)
                        o.Inputs
                        (PlanOpKind.describe kind)
                        refs
                )
            )

        cursorIndex <- cursorIndex + 1
        cursorSeq <- o.Sequence
        sp, vi, o

    let value (sp: StagePlan) (vi: int) (o: PlanOp) (rows: int) =
        if rows < 1 || rows > o.Rows then
            raise (
                ForwardException(sprintf "%s of %d rows; the plan holds %d" (PlanOpKind.describe o.Kind) rows o.Rows)
            )

        let token = sp.Values[vi][o.Index]
        let slot, offset = location token

        let h =
            GpuTensor(
                owner,
                GpuTensorKind.Value,
                token,
                slot,
                offset,
                rows,
                o.Cols,
                sp.Stage.Index,
                o.Index,
                serial,
                -1,
                -1
            )

        made.Add h
        h

    let sameShape (what: string) (a: GpuTensor) (b: GpuTensor) =
        if a.Rows <> b.Rows || a.Cols <> b.Cols then
            invalidArg what (sprintf "shapes differ: [%d, %d] and [%d, %d]" a.Rows a.Cols b.Rows b.Cols)

    let cacheOf (what: string) (x: GpuTensor) =
        if x.Kind <> GpuTensorKind.Cache || not (liveCaches.Contains x) then
            raise (ForwardException(sprintf "%s is not a cache tensor of these ops (NewCache)" what))

        x

    let download (x: GpuTensor) (row0: int) (count: int) : IHostTensor =
        flush ()
        let data = Array.zeroCreate<float32>(count * x.Cols)
        let bytes = MemoryMarshal.AsBytes(Span<float32>(data))

        if bytes.Length > 0 then
            heap.Download(x.Slot, x.Offset + int64 row0 * int64 x.Cols * 4L, bytes)

        HostTensor.create DType.F32 (Shape.matrix count x.Cols) data :> IHostTensor

    let zeroChunk = 64 <<< 20

    let zeros =
        lazy (Array.zeroCreate<byte>(int (min (int64 zeroChunk) (int64 options.MaxContext * int64 c.KvDim * 4L))))

    let fill (slot: int) (offset: int64) (bytes: int64) =
        let z = zeros.Force()
        let mutable at = 0L

        while at < bytes do
            let n = int (min (int64 z.Length) (bytes - at))
            heap.Upload(slot, offset + at, ReadOnlySpan(z, 0, n))
            at <- at + int64 n

    /// The plan these ops run.
    member _.Plan = plan

    /// The backend the ops record into.
    member _.Backend = backend

    /// Dispatches recorded so far.
    member _.Dispatches = dispatches

    /// Command lists submitted so far.
    member _.Submissions = submissions

    interface IForwardOps<GpuTensor> with
        member _.ExecutionClass = xclass
        member _.DType = DType.F32
        member _.MaxContext = options.MaxContext
        member _.Rows x = (alive "x" x).Rows
        member _.Cols x = (alive "x" x).Cols

        member _.BeginForward call =
            outside "BeginForward"
            planned call.Stage

            if call.Rows < 1 || call.Rows > options.MaxRows then
                raise (
                    ForwardException(
                        sprintf "%d rows; the ops' plan runs 1 .. %d rows per forward" call.Rows options.MaxRows
                    )
                )

            if call.Pos0 < 0 || call.Pos0 + call.Rows > options.MaxContext then
                raise (
                    ForwardException(
                        sprintf
                            "positions %d .. %d; the ops hold 0 .. %d"
                            call.Pos0
                            (call.Pos0 + call.Rows - 1)
                            (options.MaxContext - 1)
                    )
                )

            if not (options.Taps.Covers call.Taps) then
                raise (ForwardException(sprintf "the taps %A are not in the ops' plan (%A)" call.Taps options.Taps))

            // The previous result and taps of this stage end here.
            match kept.TryGetValue call.Stage.Index with
            | true, xs ->
                for x in xs do
                    x.Kill()

                kept.Remove call.Stage.Index |> ignore
            | _ -> ()

            let sp = plan.StagePlan call.Stage

            let vi =
                if sp.Variants.Length > 1 && call.Head = HeadRows.Last && call.Rows > 1 then
                    1
                else
                    0

            step <-
                match steps.TryGetValue call.Stage.Index with
                | true, n when call.Pos0 > 0 -> n + 1
                | _ -> 0

            steps[call.Stage.Index] <- step
            phase <- Forward.phaseOf call.Rows call.Pos0
            serial <- serial + 1L
            cursorIndex <- 0
            cursorSeq <- -1
            made.Clear()
            tapped.Clear()
            discard ()
            current <- Some(call, sp, vi)

            // The arenas, once per forward from position 0, so a run whose ops were created before its telemetry
            // was installed still has them (mem.arena, the inference template's memory chart).
            if call.Pos0 = 0 && Telemetry.enabled () then
                GpuTelemetry.heapCreated backendKind plan.Layout

        member _.EndForward result =
            inside "EndForward"
            let _, sp, vi = current.Value
            let v = sp.Variants[vi]
            let keep = HashSet<GpuTensor>(HashIdentity.Reference)

            let finish () =
                for x in made do
                    if not (keep.Contains x) then
                        x.Kill()

                for KeyValue(_, t) in tapped do
                    if not (keep.Contains t) then
                        t.Kill()

                kept[sp.Stage.Index] <- List.ofSeq keep

                match upload with
                | Some u ->
                    u.Kill()
                    upload <- None
                | None -> ()

                made.Clear()
                tapped.Clear()
                current <- None

            match result with
            | None ->
                discard ()
                finish ()
            | Some r ->
                let planned =
                    not (isNull r)
                    && obj.ReferenceEquals(r.Owner, owner)
                    && not r.Dead
                    && r.Kind = GpuTensorKind.Value
                    && r.Forward = serial
                    && r.Stage = sp.Stage.Index
                    && r.Position = v.Result

                if not planned || cursorIndex <> v.Ops.Length then
                    discard ()
                    finish ()

                    raise (
                        ForwardException(
                            sprintf
                                "EndForward: the forward of stage %d made %d of the plan's %d ops and returned %s, not the plan's result (op %d)"
                                sp.Stage.Index
                                cursorIndex
                                v.Ops.Length
                                (if isNull r then "null" else string r)
                                v.Result
                        )
                    )

                try
                    flush ()
                    keep.Add r |> ignore

                    for KeyValue(_, t) in tapped do
                        keep.Add t |> ignore
                finally
                    finish ()

        member _.Embed ids =
            inside "Embed"
            let sp, vi, o = next PlanOpKind.Embed []

            for i in 0 .. ids.Length - 1 do
                if ids[i] < 0 || ids[i] >= c.Vocab then
                    raise (
                        ArgumentOutOfRangeException(
                            "ids",
                            sprintf "id %d at %d is outside the vocabulary (%d)" ids[i] i c.Vocab
                        )
                    )

            let y = value sp vi o ids.Length
            let words = ids |> Array.map uint32
            let slot, offset = idsAt
            heap.Upload(slot, offset, MemoryMarshal.AsBytes(ReadOnlySpan<uint32>(words)))
            let _, tableToken = weight plan.Model.Params.Embed

            let args: Kernels.EmbedGatherBf16Args =
                {
                    Ids = plan.Ids
                    Table = tableToken
                    Y = y.Token
                    Rows = uint32 ids.Length
                    Cols = uint32 c.Hidden
                    Vocab = uint32 c.Vocab
                }

            dispatch
                Kernels.EmbedGatherBf16.Id
                o.Layer
                (Kernels.EmbedGatherBf16.encode args)
                [
                    {
                        Slot = slot
                        Lo = offset
                        Hi = offset + int64 ids.Length * 4L
                    }
                ]
                [ rangeOf y 0 y.Rows ]

            y

        member _.RmsNorm(x, weightName, eps) =
            inside "RmsNorm"
            let sp, vi, o = next (PlanOpKind.RmsNorm(weightName, eps)) [ "x", x ]
            let dims, w = weight weightName
            let width = int dims[0]
            let y = value sp vi o x.Rows

            let args: Kernels.RmsnormRowsArgs =
                {
                    X = x.Token
                    W = w
                    Y = y.Token
                    Rows = uint32 (x.Rows * (x.Cols / width))
                    Cols = uint32 width
                    Eps = float32 eps
                }

            dispatch
                Kernels.RmsnormRows.Id
                o.Layer
                (Kernels.RmsnormRows.encode args)
                [ rangeOf x 0 x.Rows ]
                [ rangeOf y 0 y.Rows ]

            y

        member _.MatMul(x, weightName) =
            inside "MatMul"
            let sp, vi, o = next (PlanOpKind.MatMul weightName) [ "x", x ]
            let dims, w = weight weightName
            let n = int dims[0]
            let k = int dims[1]
            let y = value sp vi o x.Rows

            for i, (r0, m) in List.indexed (Chunking.matmul exec.MatMulMacsPerDispatch x.Rows n k) do
                let xt, yt =
                    if i = 0 then
                        x.Token, y.Token
                    else
                        let xs, ys = sp.Chunks[(vi, o.Index)]
                        xs[i - 1], ys[i - 1]

                let args: Kernels.MatmulBf16wArgs =
                    {
                        X = xt
                        W = w
                        Y = yt
                        M = uint32 m
                        N = uint32 n
                        K = uint32 k
                    }

                dispatch
                    Kernels.MatmulBf16w.Id
                    o.Layer
                    (Kernels.MatmulBf16w.encode args)
                    [ rangeOf x r0 m ]
                    [ rangeOf y r0 m ]

            y

        member _.Rope(x, heads, pos0) =
            inside "Rope"
            let sp, vi, o = next (PlanOpKind.Rope heads) [ "x", x ]

            if pos0 < 0 || pos0 + x.Rows > options.MaxContext then
                raise (
                    ForwardException(
                        sprintf
                            "Rope at positions %d .. %d; the table holds 0 .. %d"
                            pos0
                            (pos0 + x.Rows - 1)
                            (options.MaxContext - 1)
                    )
                )

            let y = value sp vi o x.Rows

            let args: Kernels.RopeApplyArgs =
                {
                    X = x.Token
                    CosTable = plan.RopeCos
                    SinTable = plan.RopeSin
                    Y = y.Token
                    Rows = uint32 (x.Rows * heads)
                    Heads = uint32 heads
                    HeadDim = uint32 c.HeadDim
                    Pos0 = uint32 pos0
                }

            dispatch
                Kernels.RopeApply.Id
                o.Layer
                (Kernels.RopeApply.encode args)
                [ rangeOf x 0 x.Rows ]
                [ rangeOf y 0 y.Rows ]

            y

        member _.NewCache(stage, layer, rows) =
            outside "NewCache"
            planned stage

            if layer < stage.FirstLayer || layer > stage.LastLayer then
                invalidArg "layer" (sprintf "layer %d is not in stage %d" layer stage.Index)

            if rows < 1 || rows > options.MaxContext then
                invalidArg "rows" (sprintf "%d rows; the ops hold 1 .. %d positions" rows options.MaxContext)

            let free = freeCaches[struct (stage.Index, layer)]

            if free.Count = 0 then
                raise (
                    ForwardException(
                        sprintf
                            "stage %d holds %d caches already, the plan's limit (OpsOptions.Caches); dispose one first"
                            stage.Index
                            options.Caches
                    )
                )

            let i = free.Pop()
            let token = (plan.StagePlan stage).Caches[layer][i]
            let slot, offset = location token

            try
                fill slot offset (int64 rows * int64 c.KvDim * 4L)
            with _ ->
                free.Push i
                reraise ()

            let h =
                GpuTensor(owner, GpuTensorKind.Cache, token, slot, offset, rows, c.KvDim, stage.Index, -1, 0L, layer, i)

            liveCaches.Add h |> ignore
            h

        member _.ReleaseCache x =
            // After Dispose every cache is released already.
            if not disposed then
                outside "ReleaseCache"

                if isNull x || not (liveCaches.Contains x) then
                    raise (ForwardException "ReleaseCache: not a live cache tensor of these ops")

                liveCaches.Remove x |> ignore
                x.Kill()
                freeCaches[struct (x.Stage, x.Layer)].Push x.CacheSlot

        member _.KvWrite(cache, x, pos0) =
            inside "KvWrite"
            let _, _, o = next PlanOpKind.KvWrite [ "cache", cache; "x", x ]
            let cache = cacheOf "cache" cache

            if x.Cols <> cache.Cols then
                invalidArg "x" (sprintf "%d columns into a cache of %d" x.Cols cache.Cols)

            if pos0 < 0 || pos0 + x.Rows > cache.Rows then
                raise (
                    ForwardException(
                        sprintf "KvWrite of rows %d .. %d into a cache of %d rows" pos0 (pos0 + x.Rows - 1) cache.Rows
                    )
                )

            let args: Kernels.CopyRowsArgs =
                {
                    Src = x.Token
                    Dst = cache.Token
                    Rows = uint32 x.Rows
                    Cols = uint32 x.Cols
                    SrcRow0 = 0u
                    DstRow0 = uint32 pos0
                    SrcStride = uint32 x.Cols
                    DstStride = uint32 cache.Cols
                }

            dispatch
                Kernels.CopyRows.Id
                o.Layer
                (Kernels.CopyRows.encode args)
                [ rangeOf x 0 x.Rows ]
                [ rangeOf cache pos0 x.Rows ]

        member _.Attention(q, k, v, shape, past) =
            inside "Attention"
            let sp, vi, o = next (PlanOpKind.Attention shape) [ "q", q; "k", k; "v", v ]
            let k = cacheOf "k" k
            let v = cacheOf "v" v

            if past < 0 || past + q.Rows > k.Rows || past + q.Rows > v.Rows then
                raise (
                    ForwardException(
                        sprintf
                            "Attention over positions 0 .. %d; the caches hold %d and %d rows"
                            (past + q.Rows - 1)
                            k.Rows
                            v.Rows
                    )
                )

            let y = value sp vi o q.Rows

            for r0, n in Chunking.attention exec.AttentionPairsPerDispatch q.Rows past do
                let args: Kernels.AttentionGqaCausalArgs =
                    {
                        Q = q.Token
                        K = k.Token
                        V = v.Token
                        Y = y.Token
                        Rows = uint32 n
                        QRow0 = uint32 r0
                        Past = uint32 (past + r0)
                        QHeads = uint32 shape.QHeads
                        KvHeads = uint32 shape.KvHeads
                        HeadDim = uint32 shape.HeadDim
                        Scale = scale
                    }

                dispatch
                    Kernels.AttentionGqaCausal.Id
                    o.Layer
                    (Kernels.AttentionGqaCausal.encode args)
                    [ rangeOf q r0 n; rangeOf k 0 (past + r0 + n); rangeOf v 0 (past + r0 + n) ]
                    [ rangeOf y r0 n ]

            y

        member _.SiluMul(gate, up) =
            inside "SiluMul"
            let sp, vi, o = next PlanOpKind.SiluMul [ "gate", gate; "up", up ]
            sameShape "up" gate up
            let y = value sp vi o gate.Rows

            let args: Kernels.SiluMulArgs =
                {
                    Gate = gate.Token
                    Up = up.Token
                    Y = y.Token
                    N = uint32 (gate.Rows * gate.Cols)
                }

            dispatch
                Kernels.SiluMul.Id
                o.Layer
                (Kernels.SiluMul.encode args)
                [ rangeOf gate 0 gate.Rows; rangeOf up 0 up.Rows ]
                [ rangeOf y 0 y.Rows ]

            y

        member _.Add(a, b) =
            inside "Add"
            let sp, vi, o = next PlanOpKind.Add [ "a", a; "b", b ]
            sameShape "b" a b
            let y = value sp vi o a.Rows

            let args: Kernels.AddArgs =
                {
                    A = a.Token
                    B = b.Token
                    Y = y.Token
                    N = uint32 (a.Rows * a.Cols)
                }

            dispatch
                Kernels.Add.Id
                o.Layer
                (Kernels.Add.encode args)
                [ rangeOf a 0 a.Rows; rangeOf b 0 b.Rows ]
                [ rangeOf y 0 y.Rows ]

            y

        member _.LastRow x =
            inside "LastRow"
            let sp, vi, o = next PlanOpKind.LastRow [ "x", x ]
            let y = value sp vi o 1

            let args: Kernels.CopyRowsArgs =
                {
                    Src = x.Token
                    Dst = y.Token
                    Rows = 1u
                    Cols = uint32 x.Cols
                    SrcRow0 = uint32 (x.Rows - 1)
                    DstRow0 = 0u
                    SrcStride = uint32 x.Cols
                    DstStride = uint32 x.Cols
                }

            dispatch
                Kernels.CopyRows.Id
                o.Layer
                (Kernels.CopyRows.encode args)
                [ rangeOf x (x.Rows - 1) 1 ]
                [ rangeOf y 0 1 ]

            y

        member _.Argmax x =
            let x = alive "x" x

            if x.Rows > argmaxRows then
                raise (
                    ForwardException(sprintf "Argmax of %d rows; the plan holds %d (Logits arena)" x.Rows argmaxRows)
                )

            let slot, offset = argmaxAt

            let args: Kernels.ArgmaxRowsArgs =
                {
                    X = x.Token
                    Y = plan.Argmax
                    Rows = uint32 x.Rows
                    Cols = uint32 x.Cols
                }

            dispatch
                Kernels.ArgmaxRows.Id
                None
                (Kernels.ArgmaxRows.encode args)
                [ rangeOf x 0 x.Rows ]
                [
                    {
                        Slot = slot
                        Lo = offset
                        Hi = offset + int64 x.Rows * 4L
                    }
                ]

            flush ()
            let ids = Array.zeroCreate<uint32> x.Rows
            heap.Download(slot, offset, MemoryMarshal.AsBytes(Span<uint32>(ids)))
            ids

        member _.Tap(name, x) =
            inside "Tap"
            let x = alive name x
            let call, sp, _ = current.Value

            if not (call.Taps.Wants name) || not (options.Taps.Wants name) then
                raise (ForwardException(sprintf "Tap '%s': neither the forward nor the plan asks for it" name))

            let token, cols =
                match sp.Taps.TryFind name with
                | Some t -> t
                | None -> raise (ForwardException(sprintf "Tap '%s': stage %d records no such tap" name sp.Stage.Index))

            let capacity = int (table.Row token).ElemCount / cols

            if x.Cols <> cols || x.Rows > capacity then
                raise (
                    ForwardException(
                        sprintf "Tap '%s' of [%d, %d]; the plan holds [%d, %d]" name x.Rows x.Cols capacity cols
                    )
                )

            if tapped.ContainsKey name then
                raise (ForwardException(sprintf "Tap '%s' twice in one forward" name))

            let slot, offset = location token
            let bytes = int64 x.Rows * int64 x.Cols * 4L
            let l = commandList ()

            l.Copy(
                {
                    Slot = x.Slot
                    Offset = x.Offset
                    Bytes = bytes
                },
                {
                    Slot = slot
                    Offset = offset
                    Bytes = bytes
                }
            )

            // Copy is a full barrier on both sides.
            recorded <- recorded + 1
            hazards.Clear()

            let h =
                GpuTensor(
                    owner,
                    GpuTensorKind.Tap,
                    token,
                    slot,
                    offset,
                    x.Rows,
                    x.Cols,
                    sp.Stage.Index,
                    -1,
                    serial,
                    -1,
                    -1
                )

            tapped[name] <- h
            h

        member _.Read x =
            let x = alive "x" x
            download x 0 x.Rows

        member _.ReadRows(x, row0, count) =
            let x = alive "x" x

            if count < 1 || row0 < 0 || row0 + count > x.Rows then
                raise (
                    ArgumentOutOfRangeException(
                        "count",
                        sprintf "rows %d .. %d of a tensor of %d rows" row0 (row0 + count - 1) x.Rows
                    )
                )

            download x row0 count

        member _.Upload x =
            outside "Upload"

            if x.Desc.DType <> DType.F32 then
                invalidArg
                    "x"
                    (sprintf "%s is %s; these ops hold %s" (string x.Desc.Shape) (DType.name x.Desc.DType) "f32")

            if x.Desc.Shape.Rank <> 2 then
                invalidArg "x" (sprintf "%O is not [rows, cols]" x.Desc.Shape)

            let rows = int x.Desc.Shape.Rows
            let cols = int x.Desc.Shape.Cols

            if rows < 1 || cols < 1 || int64 rows * int64 cols > uploadElements then
                raise (
                    ForwardException(
                        sprintf
                            "Upload of [%d, %d]; the plan's upload buffer holds %d elements (MaxRows x hidden)"
                            rows
                            cols
                            uploadElements
                    )
                )

            upload |> Option.iter (fun u -> u.Kill())
            upload <- None
            let slot, offset = uploadAt
            heap.Upload(slot, offset, ReadOnlySpan(x.ToBytes()))

            let h =
                GpuTensor(owner, GpuTensorKind.Upload, plan.Upload, slot, offset, rows, cols, -1, -1, 0L, -1, -1)

            upload <- Some h
            h

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                discard ()

                for x in made do
                    x.Kill()

                for KeyValue(_, xs) in kept do
                    for x in xs do
                        x.Kill()

                for x in liveCaches do
                    x.Kill()

                upload |> Option.iter (fun u -> u.Kill())
                made.Clear()
                tapped.Clear()
                kept.Clear()
                liveCaches.Clear()
                current <- None

                if exec.OwnsBackend then
                    backend.Dispose()

/// The GPU executor's constructors (GpuExecutor contract).
[<RequireQualifiedAccess>]
module GpuForwardOps =
    /// The `component` attribute of the executor's own records (gpu.exec.create and what it encloses).
    [<Literal>]
    let Component = "executor"

    /// `gpu.exec.create` span: pass one, the heap, the table and the weight upload of one GpuForwardOps.create.
    [<Literal>]
    let CreateSpan = "gpu.exec.create"

    let private uploadSpan (heap: IGpuHeap) (slot: int) (offset: int64) (data: ReadOnlySpan<byte>) =
        let chunk = 64 <<< 20
        let whole = data.Length &&& ~~~3
        let mutable at = 0

        while at < whole do
            let n = min chunk (whole - at)
            heap.Upload(slot, offset + int64 at, data.Slice(at, n))
            at <- at + n

        if whole < data.Length then
            // A bf16 tensor of odd length owns its last word (HeapLayoutAndTensorTable, extent).
            let tail = Array.zeroCreate<byte> 4
            data.Slice(whole).CopyTo(Span tail)
            heap.Upload(slot, offset + int64 whole, ReadOnlySpan tail)

    /// GpuForwardOps over `backend` with `exec`: builds the plan for `options` (pass one), loads `shaders`, creates
    /// the heap, writes the sealed table once, streams the weights from the mapped checkpoint and uploads T_spec.
    /// Raises like ExecPlan.build, and GpuContractException when the backend's loaded set lacks a kernel the executor
    /// dispatches. With exec.OwnsBackend the ops own the backend, and a failure disposes it.
    let createWith
        (exec: GpuExecOptions)
        (backend: IGpuBackend)
        (shaders: ShaderSet)
        (weights: Qwen3Weights)
        (options: OpsOptions)
        : GpuOps =
        try
            use _ = Telemetry.scope [ Attr.str TelemetryKeys.Component Component ]

            use span =
                Telemetry.span
                    CreateSpan
                    [ Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend.Info.Backend) ]

            let plan = ExecPlan.build weights.Model options exec
            let kernels = backend.LoadKernels shaders

            match ExecPlan.kernels |> List.filter (fun k -> not (kernels.Contains k)) with
            | [] -> ()
            | missing ->
                raise (
                    GpuContractException(
                        sprintf
                            "the %s backend's shader set lacks kernels the executor dispatches: %s"
                            (ExecutionBackend.name backend.Info.Backend)
                            (missing |> List.map (fun k -> Kernels.all[k.Value].Name) |> String.concat ", ")
                    )
                )

            let xclass = backend.ExecutionClass
            let heap = backend.CreateHeap plan.Layout
            heap.WriteTable plan.Table

            for name, _, token in plan.Weights do
                let row = plan.Table.Row token
                uploadSpan heap row.Slot (int64 row.ByteOffset) (weights.Span name)

            let rope =
                RopeConstants.table (CpuExec.WithThreads options.Threads) weights.Config options.MaxContext

            for token, values in [ plan.RopeCos, rope.Cos; plan.RopeSin, rope.Sin ] do
                let row = plan.Table.Row token
                uploadSpan heap row.Slot (int64 row.ByteOffset) (MemoryMarshal.AsBytes(ReadOnlySpan<float32>(values)))

            span.Add(Attr.int "tensors" plan.Table.Count)
            span.Add(Attr.int "pages" plan.Layout.PageCount)
            span.Add(Attr.int64 "heap_bytes" plan.Layout.TotalBytes)
            span.Add(Attr.int "stages" plan.Stages.Length)
            span.Add(Attr.int "max_rows" plan.MaxRows)
            new GpuOps(backend, heap, plan, xclass)
        with _ ->
            if exec.OwnsBackend then
                backend.Dispose()

            reraise ()

    /// GpuForwardOps with GpuExecOptions.Default (the ops own the backend).
    let create (backend: IGpuBackend) (shaders: ShaderSet) (weights: Qwen3Weights) (options: OpsOptions) =
        createWith GpuExecOptions.Default backend shaders weights options :> IForwardOps<GpuTensor>

/// Registry entries (OpsFactory, Commands/Backends.fs) for GPU backends: each runner gets a fresh backend from
/// `device` and owns it.
[<RequireQualifiedAccess>]
module GpuBackendFactory =
    /// A factory named by the ExecutionBackend's name. `device` creates the backend for a runner's options (its
    /// threads, for CpuDevice); `shaders` gives the shader set to load.
    let create
        (backend: ExecutionBackend)
        (summary: string)
        (exec: GpuExecOptions)
        (shaders: unit -> ShaderSet)
        (device: OpsOptions -> IGpuBackend)
        : OpsFactory =
        let name = ExecutionBackend.name backend

        {
            Name = name
            Summary = summary
            Create =
                fun w o ->
                    let d = device o

                    let ops =
                        GpuForwardOps.createWith { exec with OwnsBackend = true } d (shaders ()) w o

                    QwenRunner.ofOps name w.Model (ops :> IForwardOps<GpuTensor>)
        }
