/// The routing report (Phase 3): how tokens are routed to Roslyn experts per layer, and the router's entropy
/// (ReportTemplates contract, "routing"). Reads route.usage counters and route.entropy gauges.
module Lrm.Reports.Builders.Routing

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Builds the routing scope.
let build (log: RunLog) : Scope =
    let usage =
        log.Named "route.usage" |> List.filter (fun e -> e.Kind = TelemetryKind.Counter)

    let entropy =
        log.Named "route.entropy" |> List.filter (fun e -> e.Kind = TelemetryKind.Gauge)

    let stepOf (e: RunEvent) =
        e.TryInt "step" |> Option.defaultValue 0L

    let layerOf (e: RunEvent) =
        e.TryInt TelemetryKeys.Layer |> Option.defaultValue -1L

    let expertOf (e: RunEvent) = defaultArg (e.TryStr "expert") "(none)"

    // Experts in order of total use, so the colour order is stable for one run and "none" (no expert) comes last.
    let experts =
        usage
        |> List.groupBy expertOf
        |> List.map (fun (x, es) -> x, es |> List.sumBy _.Value)
        |> List.sortBy (fun (x, total) -> (if x = "none" then 1 else 0), -total, x)
        |> List.map fst

    let expertColor (x: string) =
        if x = "none" then Color.Neutral else colorIn experts x

    let layers = (usage @ entropy) |> List.map layerOf |> List.distinct |> List.sort
    let steps = (usage @ entropy) |> List.map stepOf |> List.distinct |> List.sort
    let last = List.tryLast steps |> Option.defaultValue 0L

    let share (layer: int64) (expert: string) (step: int64) =
        let xs = usage |> List.filter (fun e -> layerOf e = layer && stepOf e = step)
        let total = xs |> List.sumBy _.Value
        let mine = xs |> List.filter (fun e -> expertOf e = expert) |> List.sumBy _.Value
        if total > 0.0 then mine / total else nan

    let heat =
        if usage.IsEmpty then
            noData "Expert usage" "route.usage counter"
        else
            Chart.Heatmap
                {
                    Title = "Expert share by layer"
                    Subtitle = Invariant.format $"Share of each layer's routed tokens at step {last}"
                    RowLabel = "expert"
                    ColLabel = "layer"
                    Rows = experts
                    Cols = layers |> List.map Invariant.i64
                    ColValues = []
                    Values =
                        [|
                            for x in experts ->
                                [|
                                    for l in layers ->
                                        let v = share l x last
                                        if Double.IsNaN v then None else Some v
                                |]
                        |]
                    Log = false
                    ValueFormat = NumFormat.Pct
                    Ceiling = None
                    Aggregate = HeatAggregate.Mean
                }

    let stacked =
        if usage.IsEmpty then
            noData "Expert mix per layer" "route.usage counter"
        else
            Chart.Bar
                {
                    Title = "Expert mix per layer"
                    Subtitle = Invariant.format $"Routed tokens at step {last}, stacked by expert"
                    CategoryLabel = "layer"
                    ValueLabel = "share of tokens"
                    ValueFormat = NumFormat.Pct
                    Categories = layers |> List.map Invariant.i64
                    Series =
                        experts
                        |> List.map (fun x ->
                            {
                                Name = x
                                Color = expertColor x
                                Values = layers |> List.map (fun l -> share l x last)
                            })
                    Stacked = true
                    Horizontal = false
                }

    let checkpoints =
        // Up to six checkpoints are all shown; past six, the first, the last and three between keep it readable.
        let s = Array.ofList steps

        if s.Length <= 6 then
            steps
        else
            [ 0; s.Length / 4; s.Length / 2; 3 * s.Length / 4; s.Length - 1 ]
            |> List.map (fun i -> s[i])
            |> List.distinct

    let entropyAt (layer: int64) (step: int64) =
        entropy
        |> List.filter (fun e -> layerOf e = layer && stepOf e = step)
        |> List.map _.Value
        |> mean

    let entropyByLayer =
        if entropy.IsEmpty then
            noData "Router entropy per layer" "route.entropy gauge"
        else
            Chart.Line
                {
                    Title = "Router entropy per layer"
                    Subtitle = "One line per checkpoint; lower entropy means sharper routing"
                    XLabel = "layer"
                    YLabel = "entropy (nats)"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Plain
                    YLog = false
                    YFromZero = true
                    IntegerX = true
                    Series =
                        checkpoints
                        |> List.mapi (fun i s ->
                            {
                                Name = "step " + Invariant.i64 s
                                Color = Color.slot i
                                Points = layers |> List.map (fun l -> float l, entropyAt l s)
                            })
                    RefLines =
                        [
                            if not experts.IsEmpty then
                                {
                                    Value = DetMath.ln (float experts.Length)
                                    Label = Invariant.format $"uniform over {experts.Length} experts"
                                }
                        ]
                    Bands = []
                }

    let entropyOverSteps =
        if entropy.IsEmpty then
            noData "Mean router entropy" "route.entropy gauge"
        else
            Chart.Line
                {
                    Title = "Mean router entropy over training"
                    Subtitle = "Mean over layers"
                    XLabel = "step"
                    YLabel = "entropy (nats)"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Plain
                    YLog = false
                    YFromZero = true
                    IntegerX = false
                    Series =
                        [
                            {
                                Name = "mean entropy"
                                Color = Color.Cat 0
                                Points =
                                    steps
                                    |> List.map (fun s ->
                                        float s,
                                        entropy |> List.filter (fun e -> stepOf e = s) |> List.map _.Value |> mean)
                            }
                        ]
                    RefLines = []
                    Bands = []
                }

    let topRows =
        layers
        |> List.map (fun l ->
            let ranked =
                experts
                |> List.map (fun x -> x, share l x last)
                |> List.filter (fun (_, v) -> Double.IsFinite v)
                |> List.sortBy (fun (x, v) -> -v, x)

            match ranked with
            | (x, v) :: _ -> [ Invariant.i64 l; x; Fmt.pct v; Fmt.dp 3 (entropyAt l last) ]
            | [] -> [ Invariant.i64 l; "none"; ""; "" ])

    runScope log
    |> Scope.value "experts" (Invariant.i32 experts.Length)
    |> Scope.value "layers" (Invariant.i32 layers.Length)
    |> Scope.value "step" (Invariant.i64 last)
    |> Scope.table "top" (table [ "Layer", R; "Top expert", L; "Share", R; "Entropy (nats)", R ] topRows)
    |> Scope.chart "usage_heatmap" heat
    |> Scope.chart "expert_mix" stacked
    |> Scope.chart "entropy_by_layer" entropyByLayer
    |> Scope.chart "entropy_over_steps" entropyOverSteps
