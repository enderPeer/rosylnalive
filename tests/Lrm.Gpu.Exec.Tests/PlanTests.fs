/// Pass one (GpuExecutor contract, "Pass one" and "Memory"): the static order the trace records, the planner
/// properties over random plans of the tiny model (results live at the same time never share bytes, an op's result
/// never overlaps its inputs, FirstClobber follows the last use, only the weights the planned stages read, the head's
/// logits rows, every row validates, the plan is deterministic), the chunking rules' vectors, and the plan of
/// Qwen3-0.6B from its config alone (no weights), against the contract's layout vectors.
module Lrm.Gpu.Exec.Tests.PlanTests

open System
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Gpu.Core
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Testing
open Lrm.Gpu.Exec.Tests.Common

/// Qwen3-0.6B's config.json at the pinned revision (the fields the forward reads), so its plan is checked in CI.
let qwenConfigJson =
    """{
  "architectures": [ "Qwen3ForCausalLM" ],
  "attention_bias": false,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"""

/// Qwen3-0.6B as the forward sees it: its config and the tied head.
let qwenModel () =
    let c = Qwen3Config.parse qwenConfigJson

    {
        Config = c
        Params = Qwen3Params.ofConfig c true
    }

let private rangeOf (v: PlanVariant) (i: int) =
    match v.Homes[i] with
    | ValueHome.Pool off -> Some(off, off + v.Ops[i].Bytes)
    | _ -> None

/// The weights the planned stages read, computed from the model's layer parameters (not from the plan's trace): each
/// planned layer's eleven tensors, the embedding when a stage embeds or has the tied head, the final norm and an untied
/// head with the head.
let expectedWeights (model: Qwen3Model) (stages: Stage list) =
    let c = model.Config

    let read =
        set
            [
                for s in stages do
                    if s.Embeds then
                        yield model.Params.Embed

                    for l in s.FirstLayer .. s.LastLayer do
                        let p = model.Params.Layers[l]

                        yield!
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
                            ]

                    if Stage.hasHead c s then
                        yield model.Params.Norm
                        yield model.Params.LmHead
            ]

    ExecPlan.weightShapes model |> List.map fst |> List.filter read.Contains

/// Every property of a plan the contract states (GpuExecutor, "Memory" and "The table").
let checkPlan (plan: ExecPlan) =
    let c = plan.Model.Config
    let t = plan.Table
    Expect.isLessThanOrEqual t.Count (Heap.TableRows - 1) "table rows"
    Expect.isLessThanOrEqual plan.Layout.PageCount (Heap.SlotCount - 1) "pages"

    // Only the weights the planned stages read, in canonical order, each a read-only bf16 row of its shape.
    Expect.equal
        (plan.Weights |> List.map (fun (n, _, _) -> n))
        (expectedWeights plan.Model plan.Options.Stages)
        "the weights of the planned stages"

    for name, dims, token in plan.Weights do
        let row = t.Row token
        Expect.equal row.DType (Some DType.BF16) (name + ": bf16")
        Expect.isTrue (TensorFlags.has TensorFlags.ReadOnly row.Flags) (name + ": read only")
        Expect.equal row.Dims (dims |> Array.map uint32 |> List.ofArray) (name + ": dims")

    // The head's logits: min(T, MaxLogitsRows, one page of rows).
    let pageRows = int (Heap.MaxPageBytes / (int64 c.Vocab * 4L))
    Expect.equal plan.LogitsRows (min plan.MaxRows (min plan.Exec.MaxLogitsRows pageRows)) "the logits rows"

    for i in 1 .. t.Count do
        let row = t.Row(TensorToken(uint32 i))

        match TensorRow.validate plan.Layout row with
        | Ok() -> ()
        | Error e -> failwithf "row %d (%s): %s" i (t.Name(TensorToken(uint32 i))) e

    for sp in plan.Stages do
        let stage = sp.Stage
        let poolRow = t.Row sp.Pool

        for vi in 0 .. sp.Variants.Length - 1 do
            let v = sp.Variants[vi]
            let label = Invariant.format $"stage {stage.Index} {v.Name}"

            let expectedOps =
                19 * stage.LayerCount
                + (if stage.Embeds then 1 else 0)
                + (if sp.HasHead then 2 else 0)
                + (if v.LastRow then 1 else 0)

            Expect.equal v.Ops.Length expectedOps (label + ": ops")

            for o in v.Ops do
                let home = v.Homes[o.Index]

                match o.Kind, home with
                | PlanOpKind.KvWrite, ValueHome.NoResult -> ()
                | PlanOpKind.KvWrite, h -> failwithf "%s: KvWrite at op %d has a home %A" label o.Index h
                | _, ValueHome.Logits when sp.HasHead && o.Index = v.Result ->
                    let rows = if v.LastRow then 1 else plan.LogitsRows
                    Expect.equal (o.Rows, o.Cols) (rows, c.Vocab) (label + ": the logits' capacity")
                    let row = t.Row(sp.Values[vi][o.Index])
                    Expect.equal row.Dims [ uint32 rows; uint32 c.Vocab ] (label + ": the logits view")

                    Expect.isLessThanOrEqual
                        (o.Bytes)
                        (int64 (t.Row sp.LogitsPool).ElemCount * 4L)
                        (label + ": logits pool")
                | _, ValueHome.Pool off ->
                    Expect.equal (off % Heap.AllocationAlignment) 0L (label + ": aligned")
                    Expect.isLessThanOrEqual (off + o.Bytes) v.PoolBytes (label + ": inside the pool")
                    let row = t.Row(sp.Values[vi][o.Index])
                    Expect.equal row.Slot poolRow.Slot (label + ": the pool's page")
                    Expect.equal (int64 row.ByteOffset) (int64 poolRow.ByteOffset + off) (label + ": the view's offset")
                    Expect.isTrue (TensorFlags.has TensorFlags.View row.Flags) (label + ": a view")
                    Expect.equal row.Dims [ uint32 o.Rows; uint32 o.Cols ] (label + ": the view's dims")
                | k, h -> failwithf "%s: op %d (%A) lives at %A" label o.Index k h

                if o.HasResult then
                    Expect.isGreaterThanOrEqual v.LastUse[o.Index] o.Sequence (label + ": last use after the def")

                    if o.Index <> v.Result then
                        Expect.isGreaterThan
                            v.FirstClobber[o.Index]
                            v.LastUse[o.Index]
                            (label + ": clobbered after its last use")

                // An op's result never overlaps its inputs.
                for r in o.Inputs do
                    match r, rangeOf v o.Index with
                    | ValueRef.Value p, Some(lo, hi) ->
                        match rangeOf v p with
                        | Some(a, b) when a < hi && lo < b ->
                            failwithf "%s: op %d overlaps its input op %d" label o.Index p
                        | _ -> ()
                    | _ -> ()

            Expect.equal v.LastUse[v.Result] Liveness.Never (label + ": the result lives on")
            Expect.equal v.FirstClobber[v.Result] Liveness.Never (label + ": the result is never overwritten")

            // Results live at the same time never share bytes.
            for a in v.Ops do
                for b in v.Ops do
                    if a.Index < b.Index then
                        match rangeOf v a.Index, rangeOf v b.Index with
                        | Some(alo, ahi), Some(blo, bhi) ->
                            let liveTogether = b.Sequence <= v.LastUse[a.Index]

                            if liveTogether && alo < bhi && blo < ahi then
                                failwithf "%s: ops %d and %d are live together and share bytes" label a.Index b.Index

                            if alo < bhi && blo < ahi then
                                Expect.isLessThanOrEqual v.FirstClobber[a.Index] b.Sequence (label + ": FirstClobber")
                        | _ -> ()

            // A split matmul has its chunk views.
            for o in v.Ops do
                match o.Kind with
                | PlanOpKind.MatMul w ->
                    let k = ExecPlan.weightShapes plan.Model |> List.find (fun (n, _) -> n = w) |> snd
                    let parts = Chunking.matmul plan.Exec.MatMulMacsPerDispatch o.Rows o.Cols (int k[1])

                    if parts.Length > 1 then
                        let xs, ys = sp.Chunks[(vi, o.Index)]
                        Expect.equal xs.Length (parts.Length - 1) (label + ": x chunk views")
                        Expect.equal ys.Length (parts.Length - 1) (label + ": y chunk views")
                    else
                        Expect.isFalse (sp.Chunks.ContainsKey((vi, o.Index))) (label + ": no chunk views")
                | _ -> ()

            // One snapshot per tap the stage records, as wide as its source and never more rows than it holds.
            for tap in v.Taps do
                let token, cols = sp.Taps[tap.Name]
                Expect.equal cols tap.Cols (label + ": tap columns")
                Expect.isGreaterThanOrEqual (int (t.Row token).Dims[0]) tap.Rows (label + ": tap rows")

                match tap.Source with
                | ValueRef.Value p ->
                    Expect.isLessThanOrEqual tap.Rows v.Ops[p].Rows (label + ": tap within its source")
                | _ -> ()

        // The cache pool: 2 · Caches tensors per layer, [MaxContext, KvDim].
        for l in stage.FirstLayer .. stage.LastLayer do
            let tokens = sp.Caches[l]
            Expect.equal tokens.Length (2 * plan.Options.Caches) "cache tensors per layer"

            for tk in tokens do
                Expect.equal (t.Row tk).Dims [ uint32 plan.Options.MaxContext; uint32 c.KvDim ] "cache dims"

    // Deterministic: the same inputs give the same layout and table.
    let again = ExecPlan.build plan.Model plan.Options plan.Exec
    Expect.equal again.Layout plan.Layout "the same layout"
    Expect.equal again.Table.Id plan.Table.Id "the same table"

let private taps =
    [|
        TapFilter.Off
        TapFilter.All
        TapFilter.Names [ "embed.out"; "L1.attn.q_rope"; "final_norm.out"; "logits" ]
        TapFilter.Names [ "L0.out"; "L1.attn.v_cache" ]
    |]

/// A plan of the tiny model from arbitrary integers: positions 1 .. 96, rows 1 .. 128, the whole model, both stages of
/// a split at layer 1 or one of them alone, a tap filter, 1 to 3 caches, the default or a small matmul and attention
/// budget, and all the head's rows or 1 .. 16.
let randomPlan
    (context: int)
    (rows: int)
    (stages: int)
    (tap: int)
    (caches: int)
    (macs: int)
    (pairs: int)
    (logits: int)
    =
    let pos (x: int) = x &&& 0x7FFFFFFF
    let c = tinyWeights.Value.Config
    let split = Stage.split c [ 1 ]

    let options =
        { OpsOptions.defaults c with
            Threads = 1
            MaxContext = 1 + pos context % 96
            MaxRows = 1 + pos rows % 128
            Stages =
                (match pos stages % 4 with
                 | 0 -> [ Stage.whole c ]
                 | 1 -> split
                 | 2 -> [ split[0] ]
                 | _ -> [ split[1] ])
            Taps = taps[pos tap % taps.Length]
            Caches = 1 + pos caches % 3
        }

    let exec =
        { GpuExecOptions.Default with
            MaxLogitsRows =
                (if logits % 2 = 0 then
                     GpuExecOptions.Default.MaxLogitsRows
                 else
                     1 + pos logits % 16)
            MatMulMacsPerDispatch =
                (if macs % 2 = 0 then
                     GpuExecOptions.Default.MatMulMacsPerDispatch
                 else
                     int64 (1 + pos macs % 60000))
            AttentionPairsPerDispatch =
                (if pairs % 2 = 0 then
                     GpuExecOptions.Default.AttentionPairsPerDispatch
                 else
                     int64 (1 + pos pairs % 5000))
        }

    ExecPlan.build (tinyModel ()) options exec

let tests =
    testList
        "plan"
        [
            testCase
                "the static order of the tiny model: 41 ops, 42 with the last-row head, 39 taps in ActivationNames order"
            <| fun () ->
                let plan = ExecPlan.build (tinyModel ()) (tinyOptions ()) GpuExecOptions.Default
                let sp = plan.Stages |> List.exactlyOne
                Expect.equal (sp.Variants |> Array.map _.Name) [| "all"; "last" |] "variants"
                let all = sp.Variants[0]
                Expect.equal all.Ops.Length 41 "ops"
                Expect.equal sp.Variants[1].Ops.Length 42 "ops with LastRow"
                Expect.equal (all.Taps |> List.map _.Name) (List.ofSeq (ActivationNames.all 2)) "the taps"
                let p = (tinyModel ()).Params.Layers[1]

                let layer1 =
                    all.Ops
                    |> Array.filter (fun o -> o.Layer = Some 1)
                    |> Array.map (fun o -> o.Kind)

                Expect.equal
                    layer1
                    [|
                        PlanOpKind.RmsNorm(p.InputNorm, 1e-6)
                        PlanOpKind.MatMul p.Q
                        PlanOpKind.MatMul p.K
                        PlanOpKind.MatMul p.V
                        PlanOpKind.RmsNorm(p.QNorm, 1e-6)
                        PlanOpKind.RmsNorm(p.KNorm, 1e-6)
                        PlanOpKind.Rope 4
                        PlanOpKind.Rope 2
                        PlanOpKind.KvWrite
                        PlanOpKind.KvWrite
                        PlanOpKind.Attention
                            {
                                QHeads = 4
                                KvHeads = 2
                                HeadDim = 32
                            }
                        PlanOpKind.MatMul p.O
                        PlanOpKind.Add
                        PlanOpKind.RmsNorm(p.PostNorm, 1e-6)
                        PlanOpKind.MatMul p.Gate
                        PlanOpKind.MatMul p.Up
                        PlanOpKind.SiluMul
                        PlanOpKind.MatMul p.Down
                        PlanOpKind.Add
                    |]
                    "layer 1, in the contract's order"

                Expect.equal all.Ops[0].Kind PlanOpKind.Embed "the embedding first"
                Expect.equal all.Ops[0].Layer None "the embedding has no layer"
                Expect.equal all.Ops[40].Kind (PlanOpKind.MatMul Qwen3Params.EmbedName) "the tied head last"
                Expect.equal all.Ops[40].Layer None "the head has no layer"
                Expect.equal all.Homes[40] ValueHome.Logits "the logits in the Logits arena"
                checkPlan plan

            Props.testProperty "random plans of the tiny model keep every planner property"
            <| fun
                   (context: int)
                   (rows: int)
                   (stages: int)
                   (tap: int)
                   (caches: int)
                   (macs: int)
                   (pairs: int)
                   (logits: int) -> checkPlan (randomPlan context rows stages tap caches macs pairs logits)

            testCase "chunking vectors (ADR K2-0001, K3-0001)"
            <| fun () ->
                let qwenLm = 64L * 151936L * 1024L
                Expect.equal GpuExecOptions.Default.MatMulMacsPerDispatch qwenLm "the default matmul budget"
                Expect.equal GpuExecOptions.Default.AttentionPairsPerDispatch 1048576L "the default attention budget"

                for (n, k), rows in
                    [
                        (151936, 1024), 64
                        (2048, 1024), 4736
                        (1024, 1024), 9472
                        (1024, 2048), 4736
                        (3072, 1024), 3136
                        (1024, 3072), 3136
                    ] do
                    Expect.equal
                        (Chunking.matmulRows qwenLm n k)
                        rows
                        (Invariant.format $"rows per dispatch, {n} x {k}")

                Expect.equal (Chunking.matmul qwenLm 900 151936 1024).Length 15 "lm_head over 900 rows"
                Expect.equal (Chunking.matmul qwenLm 900 3072 1024) [ 0, 900 ] "gate_proj over 900 rows"
                Expect.equal (Chunking.attention 1048576L 900 0) [ 0, 900 ] "a 900-token prefill is one dispatch"
                Expect.equal (Chunking.attention 1048576L 1447 0) [ 0, 1447 ] "1,447 tokens fit one dispatch"
                Expect.equal (Chunking.attention 1048576L 1448 0) [ 0, 1447; 1447, 1 ] "1,448 do not"
                Expect.equal (Chunking.attention 1048576L 1 4095) [ 0, 1 ] "a decode step at 4095"

                Expect.equal
                    (Chunking.attention 1048576L 4096 0)
                    [
                        0, 1447
                        1447, 600
                        2047, 460
                        2507, 388
                        2895, 341
                        3236, 309
                        3545, 284
                        3829, 264
                        4093, 3
                    ]
                    "a 4,096-token prefill"

                for r0, n in Chunking.attention 1048576L 4096 0 do
                    Expect.isLessThanOrEqual (Chunking.attentionPairs (int64 n) (int64 r0)) 1048576L "within the budget"

            testCase "Qwen3-0.6B from its config: 900 rows, 1,024 positions, two caches"
            <| fun () ->
                let model = qwenModel ()
                let c = model.Config

                let options =
                    { OpsOptions.defaults c with
                        Threads = 1
                        MaxContext = 1024
                        MaxRows = 900
                        Caches = 2
                    }

                let plan = ExecPlan.build model options GpuExecOptions.Default
                let mib64 (b: int64) = Heap.alignUp Heap.PageGranularity b
                let a256 (b: int64) = Heap.alignUp Heap.AllocationAlignment b
                let bytes (a: ArenaKind) = plan.Layout.ArenaBytes a
                Expect.equal plan.Layout.PageCount 7 "pages: 2 weights, 1 for each other arena"
                Expect.equal plan.Table.Count 1416 "table rows"
                Expect.equal (bytes ArenaKind.Rope) (mib64 (2L * 1024L * 64L * 4L)) "rope"
                Expect.equal (bytes ArenaKind.KvCache) (mib64 (2L * 2L * 28L * 1024L * 1024L * 4L)) "kv_cache"
                Expect.equal (bytes ArenaKind.Scratch) (mib64 (a256 (900L * 4L) + 900L * 1024L * 4L)) "scratch"
                Expect.equal (bytes ArenaKind.Logits) (mib64 (a256 (1024L * 4L) + 900L * 151936L * 4L)) "logits"

                let weightBytes =
                    ExecPlan.weightShapes model
                    |> List.map (fun (_, d) -> Heap.alignUp 4L (Array.fold (*) 1L d * 2L))

                Expect.equal weightBytes.Length 310 "310 weights, the head tied"
                Expect.equal plan.Weights.Length 310 "the whole model reads every weight"
                Expect.isLessThan (List.sum weightBytes) (bytes ArenaKind.Weights) "weights, packed"
                Expect.equal plan.LogitsRows 900 "the head on all 900 rows"

                let weightPages =
                    plan.Layout.Pages
                    |> List.filter (fun pg -> pg.Arena = ArenaKind.Weights)
                    |> List.map _.Bytes

                Expect.equal weightPages [ 1072562176L; 119603200L ] "the two weight pages"
                Expect.equal (bytes ArenaKind.Activations) 44236800L "activations"
                Expect.isFalse (plan.Layout.Arenas |> List.contains ArenaKind.Taps) "no taps arena without taps"

                let sp = plan.Stages |> List.exactlyOne
                Expect.equal sp.Variants[0].Ops.Length 535 "ops"
                Expect.equal sp.Variants[0].PoolBytes 44236800L "the activations pool"
                Expect.equal sp.Chunks.Count 1 "only lm_head splits"
                checkPlan plan

            testCase
                "Qwen3-0.6B, stage 1 of a split at 14 alone: only layers 14 .. 27, the final norm and the tied head"
            <| fun () ->
                let model = qwenModel ()
                let c = model.Config
                let stages = Stage.split c [ 14 ]

                let options =
                    { OpsOptions.defaults c with
                        Threads = 1
                        MaxContext = 1024
                        MaxRows = 900
                        Stages = [ stages[1] ]
                    }

                let plan = ExecPlan.build model options GpuExecOptions.Default
                Expect.equal plan.Weights.Length (14 * 11 + 2) "14 layers, the final norm and the embedding"

                Expect.isTrue
                    (plan.Weights |> List.exists (fun (n, _, _) -> n = Qwen3Params.EmbedName))
                    "the tied head reads the embedding"

                Expect.isFalse
                    (plan.Weights |> List.exists (fun (n, _, _) -> n.StartsWith "model.layers.13."))
                    "no layer of stage 0"

                // Python, from the shapes: 14 layers of 15,730,944 bf16 values each, the embedding and the final norm,
                // each placed at 256-byte alignment in one page rounded up to 64 KiB.
                Expect.equal (plan.Layout.ArenaBytes ArenaKind.Weights) 751697920L "weights of stage 1"
                Expect.isLessThan (plan.Layout.ArenaBytes ArenaKind.Weights) 1192165376L "less than the whole model"
                checkPlan plan

            testCase
                "Qwen3-0.6B at 2,048 rows: the All head holds the 1,766 rows of one page; MaxLogitsRows 1 holds one"
            <| fun () ->
                let model = qwenModel ()
                let c = model.Config
                let mib64 (b: int64) = Heap.alignUp Heap.PageGranularity b
                let a256 (b: int64) = Heap.alignUp Heap.AllocationAlignment b

                let options =
                    { OpsOptions.defaults c with
                        Threads = 1
                        MaxContext = 2048
                        MaxRows = 2048
                        Taps = TapFilter.Names [ "logits"; "final_norm.out" ]
                    }

                let plan = ExecPlan.build model options GpuExecOptions.Default
                Expect.equal (ExecPlan.pageLogitsRows c) 1766 "one page of Qwen3-0.6B logits"
                Expect.equal plan.LogitsRows 1766 "the All head's rows"

                Expect.equal
                    (plan.Layout.ArenaBytes ArenaKind.Logits)
                    (mib64 (a256 (2048L * 4L) + 1766L * 151936L * 4L))
                    "logits: the argmax rows and 1,766 rows of logits"

                let sp = plan.Stages |> List.exactlyOne

                let tap (name: string) =
                    int (plan.Table.Row(fst sp.Taps[name])).Dims[0]

                Expect.equal (tap "logits") 1766 "the logits tap holds what the head holds"
                Expect.equal (tap "final_norm.out") 2048 "other taps hold T rows"
                checkPlan plan

                let one =
                    ExecPlan.build
                        model
                        options
                        { GpuExecOptions.Default with
                            MaxLogitsRows = 1
                        }

                Expect.equal one.LogitsRows 1 "MaxLogitsRows 1"

                Expect.equal
                    (one.Layout.ArenaBytes ArenaKind.Logits)
                    (mib64 (a256 (2048L * 4L) + 151936L * 4L))
                    "logits: the argmax rows and one row"

                checkPlan one

            testCase "plans the executor cannot serve are refused"
            <| fun () ->
                let model = qwenModel ()
                let c = model.Config

                let refusedWith (exec: GpuExecOptions) (label: string) (text: string) (o: OpsOptions) =
                    let e =
                        throws<ArgumentException> label (fun () -> ExecPlan.build model o exec |> ignore)

                    Expect.stringContains e.Message text label

                let refusedPlan = refusedWith GpuExecOptions.Default

                refusedPlan
                    "more rows than one embedding dispatch"
                    "the rows of one embedding dispatch"
                    { OpsOptions.defaults c with
                        MaxContext = 70000
                        MaxRows = 70000
                    }

                refusedPlan
                    "activations beyond a page"
                    "the activations of stage 0 needs"
                    { OpsOptions.defaults c with
                        MaxContext = 40000
                        MaxRows = 40000
                    }

                refusedWith
                    { GpuExecOptions.Default with
                        MaxLogitsRows = 0
                    }
                    "no logits rows"
                    "logits rows"
                    { OpsOptions.defaults c with
                        MaxContext = 8
                        MaxRows = 8
                    }

                refusedWith
                    { GpuExecOptions.Default with
                        MatMulMacsPerDispatch = 0L
                    }
                    "a bad budget"
                    "multiply-adds"
                    { OpsOptions.defaults c with
                        MaxContext = 8
                        MaxRows = 8
                    }
        ]
