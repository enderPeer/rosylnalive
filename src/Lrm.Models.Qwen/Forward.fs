namespace Lrm.Models.Qwen

open System
open Lrm.Core
open Lrm.Tensor.CpuReference
open Lrm.Telemetry

/// The Qwen3 forward pass, defined once over IForwardOps (ForwardOpsAndQwenModel contract). It follows
/// ActivationNames and the CpuReferenceSemantics op order exactly; the canonical schedule is a prefill, then
/// single-token decode steps, and every row a step computes equals the same row of a prefill, bit for bit, within
/// one ExecutionClass (BatchInvariance).
[<RequireQualifiedAccess>]
module Forward =
    /// `model.forward` span: one call of runStage.
    [<Literal>]
    let ForwardSpan = "model.forward"

    /// `model.layer` span: one decoder layer inside a model.forward span.
    [<Literal>]
    let LayerSpan = "model.layer"

    /// The `component` attribute of the forward's records.
    [<Literal>]
    let Component = "forward"

    /// `prefill` for a call that starts at position 0 or holds more than one token, `decode` for a single-token step
    /// after it (the inference report's phases).
    let phaseOf (rows: int) (pos0: int) =
        if rows = 1 && pos0 > 0 then "decode" else "prefill"

    let private meter (op: Op) =
        if Telemetry.enabled () then
            Metering.op "model" op []

    /// One decoder layer: rows of x at positions pos0 .. pos0 + T - 1, writing K and V into the cache.
    let private layer
        (ops: IForwardOps<'t>)
        (model: Qwen3Model)
        (cache: KvCache<'t>)
        (i: int)
        (pos0: int)
        (rows: int)
        (tap: string -> 't -> 't)
        (x: 't)
        =
        let c = model.Config
        let p = model.Params.Layers[i]
        let eps = c.RmsNormEps
        let n (s: string) = Invariant.format $"L{i}.{s}"

        let shape =
            {
                QHeads = c.Heads
                KvHeads = c.KvHeads
                HeadDim = c.HeadDim
            }

        let h = tap (n "input_layernorm.out") (ops.RmsNorm(x, p.InputNorm, eps))
        let q = tap (n "attn.q_proj.out") (ops.MatMul(h, p.Q))
        let k = tap (n "attn.k_proj.out") (ops.MatMul(h, p.K))
        let v = tap (n "attn.v_proj.out") (ops.MatMul(h, p.V))
        let qn = tap (n "attn.q_norm.out") (ops.RmsNorm(q, p.QNorm, eps))
        let kn = tap (n "attn.k_norm.out") (ops.RmsNorm(k, p.KNorm, eps))
        let qr = tap (n "attn.q_rope") (ops.Rope(qn, c.Heads, pos0))
        let kr = tap (n "attn.k_rope") (ops.Rope(kn, c.KvHeads, pos0))
        ops.KvWrite(cache.Keys i, kr, pos0)
        ops.KvWrite(cache.Values i, v, pos0)
        // The cache stores V raw: the v_cache tap is the v_proj output, which KvWrite copied bit for bit.
        tap (n "attn.v_cache") v |> ignore

        let o =
            tap (n "attn.o_proj.in") (ops.Attention(qr, cache.Keys i, cache.Values i, shape, pos0))

        let oo = tap (n "attn.o_proj.out") (ops.MatMul(o, p.O))
        let r = tap (n "resid_mid") (ops.Add(x, oo))
        let h2 = tap (n "post_attention_layernorm.out") (ops.RmsNorm(r, p.PostNorm, eps))
        let g = tap (n "mlp.gate.out") (ops.MatMul(h2, p.Gate))
        let u = tap (n "mlp.up.out") (ops.MatMul(h2, p.Up))
        let d = tap (n "mlp.down.in") (ops.SiluMul(g, u))
        let dd = tap (n "mlp.down.out") (ops.MatMul(d, p.Down))
        let out = tap (n "out") (ops.Add(r, dd))

        if Telemetry.enabled () then
            for _, op in OpCost.layerOps (Qwen3Config.dims c) (int64 rows) (int64 pos0) do
                Metering.op "model" op []

        out

    /// Runs one stage over T rows at positions pos0 .. pos0 + T - 1 (ForwardOpsAndQwenModel, "runStage").
    ///
    /// - The input is token ids for the stage that starts at layer 0, else the residual stream [T, hidden] that the
    ///   previous stage returned, in the ops' DType (fp32 on every fp32 class).
    /// - pos0 is at most cache.Length, and pos0 + T at most cache.MaxPos. The stage writes cache rows pos0 .. pos0 +
    ///   T - 1 of each of its layers and sets cache.Length to pos0 + T; a pos0 below Length rewrites from there (a
    ///   replay). If the call raises, cache.Length is pos0.
    /// - The result is the residual stream after the stage's last layer, or, for the stage with the head, the
    ///   logits ([T, vocab] for HeadRows.All, [1, vocab] for HeadRows.Last).
    /// - Taps named by ActivationNames go to options.Taps when it wants them, as snapshots (IForwardOps.Tap).
    let runStage
        (ops: IForwardOps<'t>)
        (model: Qwen3Model)
        (stage: Stage)
        (cache: KvCache<'t>)
        (pos0: int)
        (input: StageInput<'t>)
        (options: ForwardOptions<'t>)
        : 't =
        let c = model.Config
        Stage.validate c stage

        if cache.FirstLayer <> stage.FirstLayer || cache.LayerCount <> stage.LayerCount then
            raise (
                ForwardException(
                    sprintf
                        "the cache holds layers %d .. %d, the stage %d .. %d"
                        cache.FirstLayer
                        (cache.FirstLayer + cache.LayerCount - 1)
                        stage.FirstLayer
                        stage.LastLayer
                )
            )

        let rows =
            match input with
            | StageInput.Ids ids when stage.Embeds -> ids.Length
            | StageInput.Hidden h when not stage.Embeds -> ops.Rows h
            | StageInput.Ids _ ->
                raise (
                    ForwardException(
                        sprintf
                            "stage %d starts at layer %d: its input is the residual stream, not ids"
                            stage.Index
                            stage.FirstLayer
                    )
                )
            | StageInput.Hidden _ -> raise (ForwardException "the stage that starts at layer 0 takes token ids")

        if rows < 1 then
            raise (ForwardException "a forward needs at least one row")

        if pos0 < 0 || pos0 > cache.Length then
            raise (
                ForwardException(
                    sprintf
                        "position %d: the cache holds positions 0 .. %d, so a forward starts at most at %d"
                        pos0
                        (cache.Length - 1)
                        cache.Length
                )
            )

        if pos0 + rows > cache.MaxPos then
            raise (
                ForwardException(
                    sprintf "positions %d .. %d do not fit a cache of %d positions" pos0 (pos0 + rows - 1) cache.MaxPos
                )
            )

        let phase = phaseOf rows pos0
        let head = Stage.hasHead c stage

        use span =
            Telemetry.span
                ForwardSpan
                [
                    yield Attr.str "phase" phase
                    match options.Step with
                    | Some s -> yield Attr.int "step" s
                    | None -> ()
                    yield Attr.int "tokens" rows
                    yield Attr.int "pos0" pos0
                    yield Attr.int TelemetryKeys.Stage stage.Index
                    yield Attr.int "first_layer" stage.FirstLayer
                    yield Attr.int "last_layer" stage.LastLayer
                ]

        use _ =
            Telemetry.scope
                [
                    Attr.str TelemetryKeys.Component Component
                    Attr.int TelemetryKeys.Stage stage.Index
                ]

        let tap (name: string) (x: 't) =
            match options.Taps with
            | Some store when store.Wants name -> store.Add(name, ops.Tap(name, x))
            | _ -> ()

            x

        cache.Length <- pos0

        let mutable x =
            match input with
            | StageInput.Ids ids ->
                let e = tap "embed.out" (ops.Embed ids)
                meter (Op.Embed(int64 rows, int64 c.Hidden, DType.BF16))
                e
            | StageInput.Hidden h -> h

        for i in stage.FirstLayer .. stage.LastLayer do
            use _ =
                Telemetry.span
                    LayerSpan
                    [
                        Attr.int TelemetryKeys.Layer i
                        Attr.str "phase" phase
                        Attr.int "tokens" rows
                    ]

            use _ = Telemetry.scope [ Attr.int TelemetryKeys.Layer i ]
            x <- layer ops model cache i pos0 rows tap x

        cache.Length <- pos0 + rows

        if not head then
            x
        else
            // The final norm runs on every row (the FLOP registry's forward); the head on the rows asked for.
            let f = tap "final_norm.out" (ops.RmsNorm(x, model.Params.Norm, c.RmsNormEps))

            let h, headRows =
                match options.Head with
                | HeadRows.All -> f, rows
                | HeadRows.Last -> (if rows = 1 then f else ops.LastRow f), 1

            let logits = tap "logits" (ops.MatMul(h, model.Params.LmHead))

            if Telemetry.enabled () then
                meter (Op.RmsNorm(int64 rows, int64 c.Hidden, DType.BF16))
                meter (Op.MatMul(int64 headRows, int64 c.Vocab, int64 c.Hidden, DType.F32, DType.BF16, DType.F32))
                Metering.tokens "model.tokens" (float rows) []

            span.Add(Attr.int "head_rows" headRows)
            logits

    /// The whole model as one stage: a prefill at pos0 = cache.Length = 0, or a decode or chunk after it.
    let run
        (ops: IForwardOps<'t>)
        (model: Qwen3Model)
        (cache: KvCache<'t>)
        (pos0: int)
        (ids: int[])
        (options: ForwardOptions<'t>)
        =
        runStage ops model (Stage.whole model.Config) cache pos0 (StageInput.Ids ids) options

    /// Several stages in order, each with its own cache, handing the residual stream through `transfer` at every
    /// boundary (a cluster sends it over the network; `hostRoundTrip` reads it back and uploads it, as a node does).
    /// The stages must cover the model in order. Returns the last stage's logits.
    let runStages
        (ops: IForwardOps<'t>)
        (model: Qwen3Model)
        (stages: Stage list)
        (caches: KvCache<'t> list)
        (pos0: int)
        (ids: int[])
        (transfer: 't -> 't)
        (options: ForwardOptions<'t>)
        : 't =
        let c = model.Config

        match stages with
        | [] -> invalidArg "stages" "no stages"
        | first :: _ when first.FirstLayer <> 0 -> invalidArg "stages" "the first stage must start at layer 0"
        | _ -> ()

        for a, b in List.pairwise stages do
            if b.FirstLayer <> a.LastLayer + 1 then
                invalidArg
                    "stages"
                    (sprintf
                        "stage %d ends at layer %d but stage %d starts at %d"
                        a.Index
                        a.LastLayer
                        b.Index
                        b.FirstLayer)

        if (List.last stages).LastLayer <> c.Layers - 1 then
            invalidArg "stages" "the last stage must end at the last layer"

        if caches.Length <> stages.Length then
            invalidArg "caches" (sprintf "%d caches for %d stages" caches.Length stages.Length)

        let mutable input = StageInput.Ids ids
        let mutable out = Unchecked.defaultof<'t>

        let pairs = List.zip stages caches |> Array.ofList

        for i in 0 .. pairs.Length - 1 do
            let s, cache = pairs[i]
            out <- runStage ops model s cache pos0 input options

            if i < pairs.Length - 1 then
                input <- StageInput.Hidden(transfer out)

        out

    /// A stage boundary through the host: read the residual back and upload it again, as a pipeline node does.
    let hostRoundTrip (ops: IForwardOps<'t>) (x: 't) = ops.Upload(ops.Read x)
