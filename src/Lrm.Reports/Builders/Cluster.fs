/// The cluster report: the stage Gantt chart, reissues after node loss, step latency and per-node load
/// (ReportTemplates contract, "cluster"). Reads cluster.unit and cluster.step spans and cluster.* events.
module Lrm.Reports.Builders.Cluster

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Unit outcomes in legend order, with the status colour each one wears.
let statuses =
    [
        "ok", Color.Good
        "reissued", Color.Warning
        "duplicate", Color.Neutral
        "lost", Color.Critical
    ]

let private statusColor (s: string) =
    statuses
    |> List.tryFind (fun (n, _) -> n = s)
    |> Option.map snd
    |> Option.defaultValue Color.Serious

/// Builds the cluster scope.
let build (log: RunLog) : Scope =
    let t0 = runStart log

    let units =
        log.Named "cluster.unit" |> List.filter (fun e -> e.Kind = TelemetryKind.Span)

    let steps =
        log.Named "cluster.step" |> List.filter (fun e -> e.Kind = TelemetryKind.Span)

    let statusOf (e: RunEvent) = defaultArg (e.TryStr "status") "ok"

    let stageOf (e: RunEvent) =
        e.TryText TelemetryKeys.Stage |> Option.defaultValue ""

    // Lanes: worker nodes in (stage, node) order, so a stage's replicas sit together.
    let laneKeys =
        units
        |> List.map (fun e -> e.Node, stageOf e)
        |> List.distinct
        |> List.sortBy (fun (node, stage) -> stage, node)

    let laneIndex = laneKeys |> List.mapi (fun i k -> k, i) |> Map.ofList

    let laneName (node: string, stage: string) =
        if stage = "" then node else node + "  stage " + stage

    let events =
        log.Prefixed "cluster." |> List.filter (fun e -> e.Kind = TelemetryKind.Event)

    let lost = events |> List.filter (fun e -> e.Name = "cluster.node_lost")
    let reissues = events |> List.filter (fun e -> e.Name = "cluster.reissue")
    let divergences = events |> List.filter (fun e -> e.Name = "cluster.divergence")

    let digests =
        log.Named "cluster.replica_digest"
        |> List.filter (fun e -> e.Kind = TelemetryKind.Check)

    let gantt =
        if units.IsEmpty then
            noData "Stage Gantt chart" "cluster.unit span"
        else
            Chart.Timeline
                {
                    Title = "Stage Gantt chart"
                    Subtitle = "Work units per node and stage; markers are node losses"
                    XLabel = "time since the run started"
                    XFormat = NumFormat.Duration
                    Lanes = laneKeys |> List.map laneName
                    Intervals =
                        units
                        |> List.map (fun e ->
                            let step = defaultArg (e.TryText "step") "?"
                            let unitId = defaultArg (e.TryStr "unit") "?"

                            {
                                Lane = laneIndex[(e.Node, stageOf e)]
                                Start = float (e.TimeNs - t0)
                                Length = float e.DurationNs
                                Color = statusColor (statusOf e)
                                Tip =
                                    Invariant.format
                                        $"{e.Node} stage {stageOf e}, step {step}, unit {unitId}: {statusOf e}, {Fmt.duration (float e.DurationNs)}"
                            })
                    Marks =
                        lost
                        |> List.map (fun e ->
                            {
                                At = float (e.TimeNs - t0)
                                Label = defaultArg (e.TryStr "target") "node" + " lost"
                                Color = Color.Critical
                            })
                    Legend =
                        statuses
                        |> List.filter (fun (s, _) -> units |> List.exists (fun e -> statusOf e = s))
                }

    let stepLatency =
        if steps.IsEmpty then
            noData "Step latency" "cluster.step span"
        else
            Chart.Line
                {
                    Title = "End-to-end step latency"
                    Subtitle = "Coordinator time from issuing a step to its accepted result"
                    XLabel = "step"
                    YLabel = "latency"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Duration
                    YLog = false
                    YFromZero = true
                    IntegerX = true
                    Series =
                        [
                            {
                                Name = "step latency"
                                Color = Color.Cat 0
                                Points =
                                    steps
                                    |> List.mapi (fun i e ->
                                        float (defaultArg (e.TryInt "step") (int64 i)), float e.DurationNs)
                            }
                        ]
                    RefLines =
                        [
                            {
                                Value = median (steps |> List.map (fun e -> float e.DurationNs))
                                Label = "median"
                            }
                        ]
                    Bands = []
                }

    let busy =
        if units.IsEmpty then
            noData "Busy time per node" "cluster.unit span"
        else
            let used =
                statuses
                |> List.filter (fun (s, _) -> units |> List.exists (fun e -> statusOf e = s))

            Chart.Bar
                {
                    Title = "Busy time per node by unit outcome"
                    Subtitle = "Sum of cluster.unit span durations"
                    CategoryLabel = "node"
                    ValueLabel = "busy time"
                    ValueFormat = NumFormat.Duration
                    Categories = laneKeys |> List.map laneName
                    Series =
                        used
                        |> List.map (fun (s, color) ->
                            {
                                Name = s
                                Color = color
                                Values =
                                    laneKeys
                                    |> List.map (fun (node, stage) ->
                                        units
                                        |> List.filter (fun e -> e.Node = node && stageOf e = stage && statusOf e = s)
                                        |> List.sumBy (fun e -> float e.DurationNs))
                            })
                    Stacked = true
                    Horizontal = true
                }

    let bytesBy (node: string) =
        log.Events
        |> List.filter (fun e ->
            e.Node = node
            && e.Kind = TelemetryKind.Meter
            && e.Unit = TelemetryUnits.Bytes
            && e.Name.StartsWith("cluster.", StringComparison.Ordinal))
        |> List.sumBy _.Value

    let nodeRows =
        [
            for node, stage in laneKeys ->
                let xs = units |> List.filter (fun e -> e.Node = node && stageOf e = stage)

                [
                    node
                    stage
                    backendOf log node
                    Fmt.int (int64 xs.Length)
                    Fmt.int (int64 (xs |> List.filter (fun e -> statusOf e = "reissued")).Length)
                    Fmt.int (int64 (xs |> List.filter (fun e -> statusOf e = "lost")).Length)
                    Fmt.duration (xs |> List.sumBy (fun e -> float e.DurationNs))
                    Fmt.bytes (bytesBy node)
                ]
        ]

    let eventRows =
        events
        |> List.sortBy (fun e -> e.TimeNs, e.Node, e.Seq)
        |> List.truncate 40
        |> List.map (fun e ->
            let detail =
                e.Attrs
                |> List.filter (fun a -> a.Key <> TelemetryKeys.Component)
                |> List.map (fun a -> a.Key + "=" + defaultArg (e.TryText a.Key) "")
                |> String.concat " "

            [ Fmt.duration (float (e.TimeNs - t0)); e.Node; e.Name; detail ])

    let digestOk = digests |> List.forall (fun e -> e.Value = 1.0)
    let healthy = divergences.IsEmpty && digestOk

    runScope log
    |> Scope.value
        "verdict"
        (if units.IsEmpty then
             "No work units were recorded."
         elif healthy then
             let unitText = plural units.Length "work unit" "work units"
             let lostText = plural lost.Length "node loss" "node losses"

             Invariant.format
                 $"**PASS.** {unitText}, {Fmt.int (int64 reissues.Length)} reissued after {lostText}; no divergence, and every replica digest matches."
         else
             let differ = (digests |> List.filter (fun e -> e.Value <> 1.0)).Length
             let divergenceText = plural divergences.Length "divergence" "divergences"
             let differText = plural differ "replica digest differs" "replica digests differ"
             Invariant.format $"**FAIL.** {divergenceText}; {differText}.")
    |> Scope.value "units" (Fmt.int (int64 units.Length))
    |> Scope.value "reissued" (Fmt.int (int64 reissues.Length))
    |> Scope.value "lost" (Fmt.int (int64 lost.Length))
    |> Scope.value "divergences" (Fmt.int (int64 divergences.Length))
    |> Scope.value
        "digests"
        (Invariant.format $"{(digests |> List.filter (fun e -> e.Value = 1.0)).Length}/{digests.Length}")
    |> Scope.table
        "nodes"
        (table
            [
                "Node", L
                "Stage", R
                "Backend", L
                "Units", R
                "Reissued", R
                "Lost", R
                "Busy", R
                "Bytes sent", R
            ]
            nodeRows)
    |> Scope.table "events" (table [ "Time", R; "Node", L; "Event", L; "Detail", L ] eventRows)
    |> Scope.table "meters" (table Inference.meterColumns (Inference.meterRows log.Events))
    |> Scope.chart "gantt" gantt
    |> Scope.chart "step_latency" stepLatency
    |> Scope.chart "busy" busy
