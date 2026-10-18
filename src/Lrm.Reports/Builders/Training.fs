/// The training report (Phase 2): loss per curriculum stage, the learning rate, gradient and activation statistics
/// per layer (ReportTemplates contract, "training"). Reads train.* gauges. Built to stay readable and small for runs
/// that log every layer on every one of tens of thousands of steps: the heatmap bins its step columns and the line
/// charts thin their paths (ChartSvg).
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
    let lr = gauges log "train.lr"
    let grads = gauges log "train.grad_norm"
    let acts = gauges log "train.act_rms"
    let updates = gauges log "train.update_ratio"
    let splitOf (e: RunEvent) = defaultArg (e.TryStr "split") "train"

    let stageOf (e: RunEvent) =
        defaultArg (e.TryText TelemetryKeys.Stage) ""

    let trainLoss = loss |> List.filter (fun e -> splitOf e = "train")

    let stages =
        trainLoss
        |> List.groupBy stageOf
        |> List.map (fun (s, xs) -> s, xs |> List.map stepOf |> List.min, xs |> List.map stepOf |> List.max)
        |> List.sortBy (fun (_, lo, _) -> lo)

    let stageBands =
        stages
        |> List.map (fun (s, lo, hi) ->
            {
                From = float lo
                To = float hi
                Label = s
            })

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
                    Bands = stageBands
                }

    let lrChart =
        if lr.IsEmpty then
            noData "Learning rate" "train.lr gauge"
        else
            Chart.Line
                {
                    Title = "Learning rate by step"
                    Subtitle = "The optimizer's learning rate; shaded bands are curriculum stages"
                    XLabel = "step"
                    YLabel = "learning rate"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Plain
                    YLog = false
                    YFromZero = true
                    IntegerX = false
                    Series =
                        [
                            {
                                Name = "learning rate"
                                Color = Color.Cat 0
                                Points = lr |> List.map (fun e -> float (stepOf e), e.Value)
                            }
                        ]
                    RefLines = []
                    Bands = stageBands
                }

    let layerOf (e: RunEvent) =
        e.TryInt TelemetryKeys.Layer |> Option.defaultValue -1L

    // Gradient norms per layer, each in record order: one pass over the records, not one per layer.
    let gradsByLayer =
        grads
        |> List.filter (fun e -> (e.TryInt TelemetryKeys.Layer).IsSome)
        |> List.groupBy layerOf
        |> List.sortBy fst

    let gradHeat =
        if gradsByLayer.IsEmpty then
            noData "Gradient norm" "train.grad_norm gauge"
        else
            let steps = grads |> List.map stepOf |> List.distinct |> List.sort |> Array.ofList
            let column = steps |> Array.mapi (fun i s -> s, i) |> dict

            let values =
                gradsByLayer
                |> List.map (fun (_, xs) ->
                    let row = Array.create steps.Length None

                    for e in xs do
                        row[column[stepOf e]] <- Some e.Value

                    row)
                |> Array.ofList

            Chart.Heatmap
                {
                    Title = "Gradient norm by layer and step"
                    Subtitle = "L2 norm of each layer's gradient"
                    RowLabel = "layer"
                    ColLabel = "step"
                    Rows = gradsByLayer |> List.map (fun (l, _) -> "L" + Invariant.i64 l)
                    Cols = steps |> Array.map Invariant.i64 |> List.ofArray
                    ColValues = steps |> Array.map float |> List.ofArray
                    Values = values
                    Log = true
                    ValueFormat = NumFormat.Sci
                    Ceiling = None
                    Aggregate = HeatAggregate.Max
                }

    let gradSpread =
        if gradsByLayer.IsEmpty then
            noData "Gradient norm spread" "train.grad_norm gauge"
        else
            Chart.ErrorBars
                {
                    Title = "Gradient norm per layer over the run"
                    Subtitle = "Dot: median over steps; whiskers: 5th to 95th percentile"
                    XLabel = "layer"
                    YLabel = "gradient norm (log)"
                    YFormat = NumFormat.Sci
                    YLog = true
                    Categories = gradsByLayer |> List.map (fun (l, _) -> Invariant.i64 l)
                    Series =
                        [
                            {
                                Name = "gradient norm"
                                Color = Color.Cat 0
                                Spreads =
                                    gradsByLayer
                                    |> List.mapi (fun i (_, xs) ->
                                        let v = xs |> List.map _.Value
                                        let p = spread i median v

                                        { p with
                                            Low = quantile 0.05 v
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
                    Subtitle = Invariant.format $"At step {Fmt.int last}"
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
            let b = bin 24 true (updates |> List.map _.Value)

            Chart.Histogram
                {
                    Title = "Update-to-weight ratio"
                    Subtitle = "|step update| / |weights| per layer and step, log scale"
                    XLabel = "ratio"
                    XFormat = NumFormat.Pow10
                    Bins = b.Bins
                    Zeros = b.Zeros
                    Invalid = b.Invalid
                    Color = Color.Cat 2
                    Marks =
                        [
                            {
                                At = -3.0
                                Label = ChartSvg.format NumFormat.Pow10 -3.0 + " (typical)"
                            }
                        ]
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

            [ s; Fmt.int lo; Fmt.int hi; at "train" lo; at "train" hi; at "eval" hi ])

    let lastTrain = trainLoss |> List.sortBy stepOf |> List.tryLast
    let trainSteps = trainLoss |> List.map stepOf |> List.distinct |> List.length

    runScope log
    |> Scope.value "final_loss" (lastTrain |> Option.map (fun e -> Fmt.dp 3 e.Value) |> Option.defaultValue "n/a")
    |> Scope.value "last_step" (lastTrain |> Option.map (stepOf >> Fmt.int) |> Option.defaultValue "n/a")
    |> Scope.value "steps" (plural trainSteps "training step" "training steps")
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
    |> Scope.chart "lr" lrChart
    |> Scope.chart "grad_heatmap" gradHeat
    |> Scope.chart "grad_spread" gradSpread
    |> Scope.chart "act_rms" actChart
    |> Scope.chart "update_hist" updateHist
