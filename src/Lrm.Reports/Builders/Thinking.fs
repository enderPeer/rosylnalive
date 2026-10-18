/// The thinking report (Phase 3): the Roslyn calls a run made from inside the model, as a trace with outcomes
/// (ReportTemplates contract, "thinking"). Reads roslyn.call spans and think.step events.
module Lrm.Reports.Builders.Thinking

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Call outcomes in legend order, with their status colours.
let outcomes =
    [
        "ok", Color.Good
        "diagnostic", Color.Warning
        "timeout", Color.Serious
        "error", Color.Critical
    ]

let private outcomeColor (o: string) =
    outcomes
    |> List.tryFind (fun (n, _) -> n = o)
    |> Option.map snd
    |> Option.defaultValue Color.Neutral

/// Builds the thinking scope.
let build (log: RunLog) : Scope =
    let t0 = runStart log

    let calls =
        log.Named "roslyn.call" |> List.filter (fun e -> e.Kind = TelemetryKind.Span)

    let steps =
        log.Named "think.step" |> List.filter (fun e -> e.Kind = TelemetryKind.Event)

    let apiOf (e: RunEvent) = defaultArg (e.TryStr "api") "(unknown)"
    let outcomeOf (e: RunEvent) = defaultArg (e.TryStr "outcome") "ok"

    let apis =
        calls
        |> List.groupBy apiOf
        |> List.map (fun (a, xs) -> a, xs.Length)
        |> List.sortBy (fun (a, n) -> -n, a)
        |> List.map fst

    let used =
        outcomes
        |> List.filter (fun (o, _) -> calls |> List.exists (fun e -> outcomeOf e = o))

    let trace =
        if calls.IsEmpty then
            noData "Roslyn call trace" "roslyn.call span"
        else
            let lane = apis |> List.mapi (fun i a -> a, i) |> Map.ofList

            Chart.Timeline
                {
                    Title = "Roslyn call trace"
                    Subtitle = "Each call on its API's lane, coloured by outcome; markers are thinking steps"
                    XLabel = "time since the run started"
                    XFormat = NumFormat.Duration
                    Lanes = apis
                    Intervals =
                        calls
                        |> List.map (fun e ->
                            let layer = e.TryText TelemetryKeys.Layer |> Option.defaultValue "?"
                            let token = e.TryText "token" |> Option.defaultValue "?"
                            let detail = e.TryStr "detail" |> Option.defaultValue ""

                            {
                                Lane = lane[apiOf e]
                                Start = float (e.TimeNs - t0)
                                Length = float e.DurationNs
                                Color = outcomeColor (outcomeOf e)
                                Tip =
                                    Invariant.format
                                        $"{apiOf e} at layer {layer}, token {token}: {outcomeOf e}, {Fmt.duration (float e.DurationNs)} {detail}"
                            })
                    Marks =
                        steps
                        |> List.map (fun e ->
                            {
                                At = float (e.TimeNs - t0)
                                Label = defaultArg (e.TryStr "kind") "step"
                                Color = Color.Neutral
                            })
                    Legend = used
                }

    let byApi =
        if calls.IsEmpty then
            noData "Calls per API" "roslyn.call span"
        else
            Chart.Bar
                {
                    Title = "Calls per API by outcome"
                    Subtitle = "Count of roslyn.call spans"
                    CategoryLabel = "API"
                    ValueLabel = "calls"
                    ValueFormat = NumFormat.Int
                    Categories = apis
                    Series =
                        used
                        |> List.map (fun (o, color) ->
                            {
                                Name = o
                                Color = color
                                Values =
                                    apis
                                    |> List.map (fun a ->
                                        calls
                                        |> List.filter (fun e -> apiOf e = a && outcomeOf e = o)
                                        |> List.length
                                        |> float)
                            })
                    Stacked = true
                    Horizontal = true
                }

    let durations = calls |> List.map (fun e -> float e.DurationNs)

    let latency =
        if calls.IsEmpty then
            noData "Call latency" "roslyn.call span"
        else
            let b = bin 20 true durations

            Chart.Histogram
                {
                    Title = "Call latency"
                    Subtitle =
                        Invariant.format
                            $"Log scale; median {Fmt.duration (median durations)}, p95 {Fmt.duration (quantile 0.95 durations)}"
                    XLabel = "duration"
                    XFormat = NumFormat.Decades NumFormat.Duration
                    Bins = b.Bins
                    Zeros = b.Zeros
                    Invalid = b.Invalid
                    Color = Color.Cat 0
                    Marks = []
                }

    let ok = calls |> List.filter (fun e -> outcomeOf e = "ok")

    let callRows =
        calls
        |> List.sortBy (fun e -> e.TimeNs, e.Node, e.Seq)
        |> List.truncate 40
        |> List.mapi (fun i e ->
            [
                Invariant.i32 (i + 1)
                Fmt.duration (float (e.TimeNs - t0))
                e.TryText TelemetryKeys.Layer |> Option.defaultValue ""
                e.TryText "token" |> Option.defaultValue ""
                "`" + apiOf e + "`"
                outcomeOf e
                Fmt.duration (float e.DurationNs)
                e.TryStr "detail" |> Option.defaultValue ""
            ])

    let apiRows =
        apis
        |> List.map (fun a ->
            let xs = calls |> List.filter (fun e -> apiOf e = a)

            let count o =
                xs |> List.filter (fun e -> outcomeOf e = o) |> List.length |> int64 |> Fmt.int

            let d = xs |> List.map (fun e -> float e.DurationNs)

            [
                "`" + a + "`"
                Fmt.int (int64 xs.Length)
                count "ok"
                count "diagnostic"
                count "timeout"
                count "error"
                Fmt.duration (median d)
                Fmt.duration (maxOr 0.0 d)
            ])

    let layersCalling =
        calls
        |> List.choose (fun e -> e.TryInt TelemetryKeys.Layer)
        |> List.distinct
        |> List.length

    runScope log
    |> Scope.value "calls" (Fmt.int (int64 calls.Length))
    |> Scope.value
        "ok_rate"
        (if calls.IsEmpty then
             "n/a"
         else
             Fmt.pct (float ok.Length / float calls.Length))
    |> Scope.value "layers_calling" (Invariant.i32 layersCalling)
    |> Scope.value "steps" (Fmt.int (int64 steps.Length))
    |> Scope.table
        "calls"
        (table
            [
                "#", R
                "Time", R
                "Layer", R
                "Token", R
                "API", L
                "Outcome", L
                "Duration", R
                "Detail", L
            ]
            callRows)
    |> Scope.table
        "apis"
        (table
            [
                "API", L
                "Calls", R
                "ok", R
                "diagnostic", R
                "timeout", R
                "error", R
                "Median", R
                "Max", R
            ]
            apiRows)
    |> Scope.chart "trace" trace
    |> Scope.chart "by_api" byApi
    |> Scope.chart "latency" latency
