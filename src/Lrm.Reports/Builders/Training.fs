/// The training report (Phase 2): loss per curriculum stage, gradient and activation statistics per layer
/// (ReportTemplates contract, "training"). Reads train.* gauges.
module Lrm.Reports.Builders.Training

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

let private gauges (log: RunLog) (name: string) =
    log.Named name |> List.filter (fun e -> e.Kind = TelemetryKind.Gauge)

let private stepOf (e: RunEvent) =
    e.TryInt "step" |> Option.defaultValue -1L

/// Builds the training scope.
let build (log: RunLog) : Scope =
    let loss = gauges log "train.loss"
    let grads = gauges log "train.grad_norm"
    let acts = gauges log "train.act_rms"
    let updates = gauges log "train.update_ratio"
    let splitOf (e: RunEvent) = defaultArg (e.TryStr "split") "train"

    let stageOf (e: RunEvent) =
        defaultArg (e.TryText TelemetryKeys.Stage) ""

    let stages =
        loss
        |> List.filter (fun e -> splitOf e = "train")
        |> List.groupBy stageOf
        |> List.map (fun (s, xs) -> s, xs |> List.map stepOf |> List.min, xs |> List.map stepOf |> List.max)
        |> List.sortBy (fun (_, lo, _) -> lo)

    let lossChart =
        if loss.IsEmpty then
            noData "Loss" "train.loss gauge"
        else
            Chart.Line
                {
                    Title = "Loss by step"
                    Subtitle = "Training and held-out loss; shaded bands are curriculum stages"
                    XLabel = "step"
                    YLabel = "loss (nats per token)"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Plain
                    YLog = false
                    YFromZero = false
                    IntegerX = false
                    Series =
                        [ "train"; "eval" ]
                        |> List.mapi (fun i split ->
                            {
                                Name = split
                                Color = Color.Cat i
                                Points =
                                    loss
                                    |> List.filter (fun e -> splitOf e = split)
                                    |> List.map (fun e -> float (stepOf e), e.Value)
                            })
                        |> List.filter (fun s -> not s.Points.IsEmpty)
                    RefLines = []
                    Bands =
                        stages
                        |> List.map (fun (s, lo, hi) ->
                            {
                                From = float lo
                                To = float hi
                                Label = s
                            })
                }

    let layerIds (xs: RunEvent list) =
        xs
        |> List.choose (fun e -> e.TryInt TelemetryKeys.Layer)
        |> List.distinct
        |> List.sort

    let gradHeat =
        if grads.IsEmpty then
            noData "Gradient norm" "train.grad_norm gauge"
        else
            let layers = layerIds grads
            let steps = grads |> List.map stepOf |> List.distinct |> List.sort

            let lookup =
                grads
                |> List.map (fun e -> (e.TryInt TelemetryKeys.Layer |> Option.defaultValue -1L, stepOf e), e.Value)
                |> Map.ofList

            Chart.Heatmap
                {
                    Title = "Gradient norm by layer and step"
                    Subtitle = "L2 norm of each layer's gradient"
                    RowLabel = "layer"
                    ColLabel = "step"
                    Rows = layers |> List.map (fun l -> "L" + Invariant.i64 l)
                    Cols = steps |> List.map Invariant.i64
                    Values =
                        [|
                            for l in layers -> [| for s in steps -> lookup.TryFind(l, s) |> Option.defaultValue nan |]
                        |]
                    Log = true
                    ValueFormat = NumFormat.Sci
                }

    let gradSpread =
        if grads.IsEmpty then
            noData "Gradient norm spread" "train.grad_norm gauge"
        else
            let layers = layerIds grads

            Chart.ErrorBars
                {
                    Title = "Gradient norm per layer over the run"
                    Subtitle = "Dot: median over steps; whiskers: 5th to 95th percentile"
                    XLabel = "layer"
                    YLabel = "gradient norm (log)"
                    YFormat = NumFormat.Sci
                    YLog = true
                    Categories = layers |> List.map Invariant.i64
                    Series =
                        [
                            {
                                Name = "gradient norm"
                                Color = Color.Cat 0
                                Spreads =
                                    layers
                                    |> List.mapi (fun i l ->
                                        let v =
                                            grads
                                            |> List.filter (fun e -> e.TryInt TelemetryKeys.Layer = Some l)
                                            |> List.map _.Value

                                        {
                                            Index = i
                                            Low = quantile 0.05 v
                                            Mid = median v
                                            High = quantile 0.95 v
                                        })
                            }
                        ]
                    RefLines = []
                }

    let actChart =
        if acts.IsEmpty then
            noData "Activation RMS" "train.act_rms gauge"
        else
            let last = acts |> List.map stepOf |> List.max
            let sites = acts |> List.choose (fun e -> e.TryStr "site") |> List.distinct
            let atLast = acts |> List.filter (fun e -> stepOf e = last)

            Chart.Line
                {
                    Title = "Activation RMS per layer"
                    Subtitle = Invariant.format $"At step {last}"
                    XLabel = "layer"
                    YLabel = "RMS"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Plain
                    YLog = false
                    YFromZero = true
                    IntegerX = true
                    Series =
                        sites
                        |> List.mapi (fun i site ->
                            {
                                Name = site
                                Color = Color.slot i
                                Points =
                                    atLast
                                    |> List.filter (fun e -> e.TryStr "site" = Some site)
                                    |> List.choose (fun e ->
                                        e.TryInt TelemetryKeys.Layer |> Option.map (fun l -> float l, e.Value))
                                    |> List.sortBy fst
                            })
                    RefLines = []
                    Bands = []
                }

    let updateHist =
        if updates.IsEmpty then
            noData "Update ratio" "train.update_ratio gauge"
        else
            Chart.Histogram
                {
                    Title = "Update-to-weight ratio"
                    Subtitle = "|step update| / |weights| per layer and step, log scale"
                    XLabel = "ratio"
                    XFormat = NumFormat.Pow10
                    Bins = bins 24 true (updates |> List.map _.Value)
                    Color = Color.Cat 2
                    Marks = [ { At = -3.0; Label = "1e-3" } ]
                }

    let stageRows =
        stages
        |> List.map (fun (s, lo, hi) ->
            let at split step =
                loss
                |> List.filter (fun e -> splitOf e = split && stageOf e = s)
                |> List.sortBy (fun e -> abs (stepOf e - step))
                |> List.tryHead
                |> Option.map (fun e -> Fmt.dp 3 e.Value)
                |> Option.defaultValue "n/a"

            [
                s
                Invariant.i64 lo
                Invariant.i64 hi
                at "train" lo
                at "train" hi
                at "eval" hi
            ])

    let lastTrain =
        loss
        |> List.filter (fun e -> splitOf e = "train")
        |> List.sortBy stepOf
        |> List.tryLast

    runScope log
    |> Scope.value "final_loss" (lastTrain |> Option.map (fun e -> Fmt.dp 3 e.Value) |> Option.defaultValue "n/a")
    |> Scope.value
        "steps"
        (lastTrain
         |> Option.map (fun e -> Invariant.i64 (stepOf e))
         |> Option.defaultValue "0")
    |> Scope.table
        "stages"
        (table
            [
                "Stage", L
                "First step", R
                "Last step", R
                "Loss at start", R
                "Loss at end", R
                "Held-out at end", R
            ]
            stageRows)
    |> Scope.chart "loss" lossChart
    |> Scope.chart "grad_heatmap" gradHeat
    |> Scope.chart "grad_spread" gradSpread
    |> Scope.chart "act_rms" actChart
    |> Scope.chart "update_hist" updateHist
