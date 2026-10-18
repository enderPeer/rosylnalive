namespace Lrm.Gpu.Exec

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Tensor.CpuReference
open Lrm.Models.Qwen

/// Where an op's input comes from in the static order of one forward (GpuExecutor contract, "Pass one").
[<RequireQualifiedAccess>]
type ValueRef =
    /// The result of compute op `position` of the same forward.
    | Value of position: int
    /// A handle from outside the forward: the stage's input residual (StageInput.Hidden), an upload, or another
    /// stage's result.
    | Input
    /// A KV-cache tensor of model layer `layer` (IForwardOps.NewCache).
    | Cache of layer: int

/// One compute op of the static order, with the parameters the plan depends on. Positions (pos0, past) and row
/// counts are not part of it: they are the scalars a dispatch takes.
[<RequireQualifiedAccess>]
type PlanOpKind =
    | Embed
    | RmsNorm of weight: string * eps: float
    | MatMul of weight: string
    | Rope of heads: int
    | KvWrite
    | Attention of shape: AttentionShape
    | SiluMul
    | Add
    | LastRow

[<RequireQualifiedAccess>]
module PlanOpKind =
    /// The op's name in the IForwardOps seam.
    let name (k: PlanOpKind) =
        match k with
        | PlanOpKind.Embed -> "Embed"
        | PlanOpKind.RmsNorm _ -> "RmsNorm"
        | PlanOpKind.MatMul _ -> "MatMul"
        | PlanOpKind.Rope _ -> "Rope"
        | PlanOpKind.KvWrite -> "KvWrite"
        | PlanOpKind.Attention _ -> "Attention"
        | PlanOpKind.SiluMul -> "SiluMul"
        | PlanOpKind.Add -> "Add"
        | PlanOpKind.LastRow -> "LastRow"

    /// The catalog kernel the op runs as (KernelAbiAndCatalog).
    let kernel (k: PlanOpKind) =
        match k with
        | PlanOpKind.Embed -> "embed_gather_bf16"
        | PlanOpKind.RmsNorm _ -> "rmsnorm_rows"
        | PlanOpKind.MatMul _ -> "matmul_bf16w"
        | PlanOpKind.Rope _ -> "rope_apply"
        | PlanOpKind.KvWrite
        | PlanOpKind.LastRow -> "copy_rows"
        | PlanOpKind.Attention _ -> "attention_gqa_causal"
        | PlanOpKind.SiluMul -> "silu_mul"
        | PlanOpKind.Add -> "add"

    /// The op with its parameters, for messages.
    let describe (k: PlanOpKind) =
        match k with
        | PlanOpKind.RmsNorm(w, _)
        | PlanOpKind.MatMul w -> sprintf "%s(%s)" (name k) w
        | PlanOpKind.Rope h -> sprintf "Rope(%d heads)" h
        | PlanOpKind.Attention s -> sprintf "Attention(%d:%d heads of %d)" s.QHeads s.KvHeads s.HeadDim
        | _ -> name k

/// One call of a traced forward, in order.
[<RequireQualifiedAccess>]
type TraceCall =
    /// A compute op; `cols` is 0 for an op without a result (KvWrite).
    | Compute of kind: PlanOpKind * inputs: ValueRef list * rows: int * cols: int
    /// IForwardOps.Tap of a value.
    | Tap of name: string * source: ValueRef * rows: int * cols: int

/// The static order of one forward (ForwardOpsAndQwenModel, "A static order"), as Forward.runStage calls the ops.
type TracedForward =
    {
        Stage: Stage
        /// T, the rows the forward was traced at.
        Rows: int
        Head: HeadRows
        /// The taps the trace recorded (the plan's filter).
        Taps: TapFilter
        Calls: TraceCall list
        /// The handle the forward returned.
        Result: ValueRef
    }

/// A handle of the tracing ops: a shape and where the value comes from.
[<Sealed; AllowNullLiteral>]
type TraceValue internal (rows: int, cols: int, source: ValueRef) =
    member _.Rows = rows
    member _.Cols = cols
    member _.Source = source

/// IForwardOps over shapes only: it records every call Forward.runStage makes, with each input's origin, and computes
/// nothing. Pass one of the executor's plan runs the forward over it (StaticOrder.trace).
[<Sealed>]
type internal TraceOps(model: Qwen3Model, maxContext: int) =
    let c = model.Config
    let shapes = Dictionary<string, int64[]>(StringComparer.Ordinal)

    do
        for name, s in Qwen3Params.shapes c do
            shapes[name] <- s

        if model.Params.LmHead = Qwen3Params.LmHeadName then
            shapes[Qwen3Params.LmHeadName] <- [| int64 c.Vocab; int64 c.Hidden |]

    let calls = List<TraceCall>()
    let mutable computed = 0
    let mutable inside = false
    let mutable result: ValueRef option = None

    let fail (message: string) =
        raise (ForwardException("trace: " + message))

    let shapeOf (weight: string) =
        match shapes.TryGetValue weight with
        | true, s -> s
        | _ -> fail (sprintf "the forward reads '%s', which the model has no tensor for" weight)

    let record (kind: PlanOpKind) (inputs: TraceValue list) (rows: int) (cols: int) =
        if not inside then
            fail (sprintf "%s outside a forward" (PlanOpKind.name kind))

        let position = computed
        computed <- computed + 1
        calls.Add(TraceCall.Compute(kind, inputs |> List.map _.Source, rows, cols))
        TraceValue(rows, cols, ValueRef.Value position)

    member _.Calls = List.ofSeq calls
    member _.Result = result

    /// The stage input of a stage that does not embed: [rows, hidden].
    member _.Input(rows: int) =
        TraceValue(rows, c.Hidden, ValueRef.Input)

    interface IForwardOps<TraceValue> with
        member _.ExecutionClass = ExecutionClass.forCpu ExecutionBackend.CpuDevice
        member _.DType = DType.F32
        member _.MaxContext = maxContext
        member _.Rows x = x.Rows
        member _.Cols x = x.Cols

        member _.BeginForward _ =
            calls.Clear()
            computed <- 0
            result <- None
            inside <- true

        member _.EndForward r =
            inside <- false
            result <- r |> Option.map _.Source

        member _.Embed ids =
            record PlanOpKind.Embed [] ids.Length c.Hidden

        member _.RmsNorm(x, weight, eps) =
            let s = shapeOf weight

            if s.Length <> 1 || s[0] < 1L || int64 x.Cols % s[0] <> 0L then
                fail (sprintf "RmsNorm of %d columns with '%s' of shape %A" x.Cols weight s)

            record (PlanOpKind.RmsNorm(weight, eps)) [ x ] x.Rows x.Cols

        member _.MatMul(x, weight) =
            let s = shapeOf weight

            if s.Length <> 2 || int64 x.Cols <> s[1] then
                fail (sprintf "MatMul of %d columns with '%s' of shape %A" x.Cols weight s)

            record (PlanOpKind.MatMul weight) [ x ] x.Rows (int s[0])

        member _.Rope(x, heads, _) =
            if heads < 1 || x.Cols <> heads * c.HeadDim then
                fail (sprintf "Rope of %d columns as %d heads of %d" x.Cols heads c.HeadDim)

            record (PlanOpKind.Rope heads) [ x ] x.Rows x.Cols

        member _.NewCache(_, layer, rows) =
            TraceValue(rows, c.KvDim, ValueRef.Cache layer)

        member _.ReleaseCache _ = ()

        member _.KvWrite(cache, x, _) =
            record PlanOpKind.KvWrite [ cache; x ] 0 0 |> ignore

        member _.Attention(q, k, v, shape, _) =
            record (PlanOpKind.Attention shape) [ q; k; v ] q.Rows q.Cols

        member _.SiluMul(gate, up) =
            record PlanOpKind.SiluMul [ gate; up ] gate.Rows gate.Cols

        member _.Add(a, b) =
            record PlanOpKind.Add [ a; b ] a.Rows a.Cols

        member _.LastRow x =
            record PlanOpKind.LastRow [ x ] 1 x.Cols

        member _.Argmax _ = fail "Argmax inside a traced forward"

        member _.Tap(name, x) =
            calls.Add(TraceCall.Tap(name, x.Source, x.Rows, x.Cols))
            x

        member _.Read _ = fail "Read inside a traced forward"
        member _.ReadRows(_, _, _) = fail "ReadRows inside a traced forward"
        member _.Upload _ = fail "Upload inside a traced forward"

    interface IDisposable with
        member _.Dispose() = ()

/// Pass one of the static plan: the op order of Forward.runStage, recorded over shapes (TraceOps). The order depends
/// only on the model and the ForwardCall (ForwardOpsAndQwenModel, "A static order"), so one trace per stage and head
/// variant, at the plan's largest T, describes every forward the plan serves.
[<RequireQualifiedAccess>]
module StaticOrder =
    /// The calls of one forward of `stage` over `rows` rows from position 0, with `head` and recording `taps`. Runs
    /// under the no-op telemetry facade, so a trace leaves no record.
    let trace (model: Qwen3Model) (stage: Stage) (rows: int) (head: HeadRows) (taps: TapFilter) : TracedForward =
        if rows < 1 then
            invalidArg "rows" (sprintf "%d rows" rows)

        use _ = Telemetry.install Telemetry.noop
        let ops = new TraceOps(model, rows)
        let iops = ops :> IForwardOps<TraceValue>
        use cache = KvCache.create iops model stage rows

        let input =
            if stage.Embeds then
                StageInput.Ids(Array.zeroCreate rows)
            else
                StageInput.Hidden(ops.Input rows)

        let store =
            match taps with
            | TapFilter.Off -> None
            | f -> Some(TapStore<TraceValue>(f))

        Forward.runStage
            iops
            model
            stage
            cache
            0
            input
            {
                Head = head
                Taps = store
                Step = None
            }
        |> ignore

        {
            Stage = stage
            Rows = rows
            Head = head
            Taps = taps
            Calls = ops.Calls
            Result =
                match ops.Result with
                | Some r -> r
                | None -> raise (ForwardException "trace: the forward returned no result")
        }
