/// The inference report: prefill and decode speed, per-layer time, memory per arena and the metering table
/// (ReportTemplates contract, "inference"). Reads model.forward and model.layer spans, model.* meters and mem.arena
/// gauges.
module Lrm.Reports.Builders.Inference

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Heap arenas in HeapLayoutAndTensorTable order; unknown arenas follow in ordinal order.
let arenaOrder =
    [
        "weights"
        "rope"
        "kv_cache"
        "activations"
        "scratch"
        "logits"
        "taps"
        "master"
        "grad"
        "adam_m"
        "adam_v"
        "roslyn_in"
        "roslyn_out"
    ]

let private phase (e: RunEvent) = defaultArg (e.TryStr "phase") ""

/// The metering table rows: per node, component and stage (layers summed).
let meterRows (events: RunEvent list) =
    Metering.attribute events
    |> Metering.rollup (fun k -> { k with Layer = None })
    |> List.map (fun (k, t) ->
        [
            k.Node
            (if k.Component = "" then "(none)" else k.Component)
            (if k.Stage = "" then "" else k.Stage)
            Fmt.duration (float t.WallNs)
            Fmt.duration (float t.GpuNs)
            Fmt.si t.Flops
            Fmt.bytes t.Bytes
            Fmt.si t.Tokens
            Fmt.int (int64 t.Records)
        ])

let meterColumns =
    [
        "Node", L
        "Component", L
        "Stage", L
        "Wall (self)", R
        "GPU", R
        "FLOPs", R
        "Bytes moved", R
        "Tokens", R
        "Records", R
    ]

/// Builds the inference scope.
let build (log: RunLog) : Scope =
    let forwards =
        log.Named "model.forward" |> List.filter (fun e -> e.Kind = TelemetryKind.Span)

    let layers =
        log.Named "model.layer" |> List.filter (fun e -> e.Kind = TelemetryKind.Span)

    let nodes = forwards |> List.map _.Node |> List.distinct |> List.sort

    let tokensOf (e: RunEvent) =
        e.TryInt "tokens" |> Option.map float |> Option.defaultValue 1.0

    let phaseRows =
        [
            for ph in [ "prefill"; "decode" ] do
                let xs = forwards |> List.filter (fun e -> phase e = ph)

                if not xs.IsEmpty then
                    let tokens = xs |> List.sumBy tokensOf
                    let ns = xs |> List.sumBy (fun e -> float e.DurationNs)

                    yield
                        ph,
                        [
                            ph
                            Fmt.int (int64 xs.Length)
                            Fmt.int (int64 tokens)
                            Fmt.duration ns
                            Fmt.num (tokens / (ns / 1e9)) + " tok/s"
                            Fmt.duration (median (xs |> List.map (fun e -> float e.DurationNs)))
                        ]
        ]

    let rate ph =
        let xs = forwards |> List.filter (fun e -> phase e = ph)
        let ns = xs |> List.sumBy (fun e -> float e.DurationNs)

        if xs.IsEmpty || ns <= 0.0 then
            "n/a"
        else
            Fmt.num ((xs |> List.sumBy tokensOf) / (ns / 1e9)) + " tok/s"

    let flops = log.Named "model.flops" |> List.sumBy _.Value
    let forwardNs = forwards |> List.sumBy (fun e -> float e.DurationNs)
    let tokens = log.Named "model.tokens" |> List.sumBy _.Value

    let seriesName (node: string) =
        match backendOf log node with
        | "" -> node
        | b -> node + " (" + b + ")"

    let decodeLatency =
        let decode = forwards |> List.filter (fun e -> phase e = "decode")

        if decode.IsEmpty then
            noData "Decode step latency" "model.forward decode span"
        else
            Chart.Line
                {
                    Title = "Decode step latency"
                    Subtitle = "Host time of each single-token step"
                    XLabel = "step"
                    YLabel = "latency"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Duration
                    YLog = false
                    YFromZero = true
                    IntegerX = true
                    Series =
                        nodes
                        |> List.mapi (fun i node ->
                            {
                                Name = seriesName node
                                // One line wears the first slot; several follow their backend's fixed colour.
                                Color =
                                    match nodes, colorIn backendOrder (backendOf log node) with
                                    | [ _ ], _
                                    | _, Color.Neutral -> Color.slot i
                                    | _, c -> c
                                Points =
                                    decode
                                    |> List.filter (fun e -> e.Node = node)
                                    |> List.mapi (fun j e ->
                                        float (defaultArg (e.TryInt "step") (int64 j)), float e.DurationNs)
                            })
                    RefLines = []
                    Bands = []
                }

    let layerTime =
        if layers.IsEmpty then
            noData "Layer time per token" "model.layer span"
        else
            let tokensOfParent =
                forwards |> List.map (fun e -> (e.Node, e.SpanId), tokensOf e) |> Map.ofList

            let perToken (e: RunEvent) =
                let t = tokensOfParent.TryFind(e.Node, e.ParentId) |> Option.defaultValue 1.0
                float e.DurationNs / t

            let layerIds =
                layers
                |> List.choose (fun e -> e.TryInt TelemetryKeys.Layer)
                |> List.distinct
                |> List.sort

            Chart.Line
                {
                    Title = "Layer time per token"
                    Subtitle = "Mean host time of model.layer spans divided by the tokens of their step"
                    XLabel = "layer"
                    YLabel = "time per token"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Duration
                    YLog = false
                    YFromZero = true
                    IntegerX = true
                    Series =
                        [ "prefill"; "decode" ]
                        |> List.mapi (fun i ph ->
                            {
                                Name = ph
                                Color = Color.Cat i
                                Points =
                                    layerIds
                                    |> List.map (fun l ->
                                        let xs =
                                            layers
                                            |> List.filter (fun e ->
                                                phase e = ph && e.TryInt TelemetryKeys.Layer = Some l)
                                            |> List.map perToken

                                        float l, mean xs)
                            })
                        |> List.filter (fun s -> s.Points |> List.exists (fun (_, y) -> Double.IsFinite y))
                    RefLines = []
                    Bands = []
                }

    let latencyHist =
        let xs =
            forwards
            |> List.filter (fun e -> phase e = "decode")
            |> List.map (fun e -> float e.DurationNs)

        if xs.IsEmpty then
            noData "Decode latency distribution" "model.forward decode span"
        else
            Chart.Histogram
                {
                    Title = "Decode latency distribution"
                    Subtitle =
                        Invariant.format
                            $"{xs.Length} steps; median {Fmt.duration (median xs)}, p95 {Fmt.duration (quantile 0.95 xs)}"
                    XLabel = "step latency"
                    XFormat = NumFormat.Duration
                    Bins = bins 20 false xs
                    Color = Color.Cat 0
                    Marks = [ { At = median xs; Label = "median" } ]
                }

    let arenas = Metering.arenas log.Events
    let arenaNames = arenas |> List.map _.Arena |> sortIn arenaOrder
    let arenaNodes = arenas |> List.map _.Node |> List.distinct |> List.sort

    let arenaChart =
        if arenas.IsEmpty then
            noData "Peak memory per arena" "mem.arena gauge"
        else
            Chart.Bar
                {
                    Title = "Peak memory per arena"
                    Subtitle = "Largest mem.arena gauge per heap arena"
                    CategoryLabel = "arena"
                    ValueLabel = "bytes"
                    ValueFormat = NumFormat.Bytes
                    Categories = arenaNames
                    Series =
                        arenaNodes
                        |> List.mapi (fun i node ->
                            {
                                Name = seriesName node
                                Color = Color.slot i
                                Values =
                                    arenaNames
                                    |> List.map (fun a ->
                                        arenas
                                        |> List.tryFind (fun x -> x.Node = node && x.Arena = a)
                                        |> Option.map _.PeakBytes
                                        |> Option.defaultValue 0.0)
                            })
                    Stacked = false
                    Horizontal = true
                }

    runScope log
    |> Scope.value "prefill_rate" (rate "prefill")
    |> Scope.value "decode_rate" (rate "decode")
    |> Scope.value
        "ttft"
        (forwards
         |> List.tryFind (fun e -> phase e = "prefill")
         |> Option.map (fun e -> Fmt.duration (float e.DurationNs))
         |> Option.defaultValue "n/a")
    |> Scope.value
        "flops_per_token"
        (if tokens > 0.0 then
             Fmt.siWith "FLOP" (flops / tokens)
         else
             "n/a")
    |> Scope.value
        "achieved"
        (if forwardNs > 0.0 && flops > 0.0 then
             Fmt.siWith "FLOP/s" (flops / (forwardNs / 1e9))
         else
             "n/a")
    |> Scope.table
        "phases"
        (table
            [ "Phase", L; "Steps", R; "Tokens", R; "Time", R; "Rate", R; "Median step", R ]
            (phaseRows |> List.map snd))
    |> Scope.table "meters" (table meterColumns (meterRows log.Events))
    |> Scope.table
        "arenas"
        (table
            [ "Node", L; "Arena", L; "Peak", R; "Last", R; "Samples", R ]
            [
                for a in arenas |> List.sortBy (fun a -> a.Node, orderIn arenaOrder a.Arena) ->
                    [
                        a.Node
                        a.Arena
                        Fmt.bytes a.PeakBytes
                        Fmt.bytes a.LastBytes
                        Fmt.int (int64 a.Samples)
                    ]
            ])
    |> Scope.chart "decode_latency" decodeLatency
    |> Scope.chart "layer_time" layerTime
    |> Scope.chart "latency_hist" latencyHist
    |> Scope.chart "arena_peak" arenaChart
