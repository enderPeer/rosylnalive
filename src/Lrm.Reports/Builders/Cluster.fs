/// The cluster report: the stage Gantt chart, reissues and node losses, step latency, stage deadlines and per-node load
/// (ReportTemplates contract, "cluster"). Reads C1's reserved records (ClusterProtocol contract, "Telemetry"):
/// cluster.unit.done events (the Gantt chart), the cluster.assign, cluster.reissue, cluster.duplicate,
/// cluster.deadline.miss, cluster.replay.* and cluster.chaos counters, cluster.member.* and cluster.demote events,
/// cluster.deadline gauges, cluster.unit.latency histograms and the cluster.run outcome.
///
/// Times. cluster.unit.done carries start_ns and end_ns on the coordinator's clock (virtual time in the simulator, the
/// process's monotonic clock in real runs), while every record's own time is the run log's clock. The coordinator
/// emits unit.done when it accepts the unit, at end_ns, so their difference aligns the two; with no unit the records'
/// own times are used.
module Lrm.Reports.Builders.Cluster

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Unit outcomes in legend order, with the status colour each one wears: accepted from its first assignment, or
/// accepted after its work was reissued to another node (cluster.reissue for the same lane, stage and step).
let statuses = [ "accepted", Color.Good; "reissued", Color.Warning ]

type private Unit =
    {
        Lane: int64
        Stage: int64
        Step: int64
        /// The node whose result the coordinator accepted.
        Node: string
        StartNs: int64
        EndNs: int64
        Status: string
    }

let private nodeOf (e: RunEvent) =
    e.TryStr TelemetryKeys.Node |> Option.defaultValue e.Node

let private intOr (fallback: int64) (key: string) (e: RunEvent) = defaultArg (e.TryInt key) fallback

let private statusColor (s: string) =
    statuses
    |> List.tryFind (fun (n, _) -> n = s)
    |> Option.map snd
    |> Option.defaultValue Color.Serious

/// The counters the counters table lists, in this order, with the attribute that splits each one.
let private counterOrder =
    [
        "cluster.assign", "retransmit"
        "cluster.reissue", ""
        "cluster.duplicate", ""
        "cluster.deadline.miss", ""
        "cluster.replay.served", ""
        "cluster.replay.units", ""
        "cluster.frame.rejected", "reason"
        "cluster.chaos", "fault"
    ]

/// Builds the cluster scope.
let build (log: RunLog) : Scope =
    let named (name: string) (kind: TelemetryKind) =
        log.Named name |> List.filter (fun e -> e.Kind = kind)

    let reissues = named "cluster.reissue" TelemetryKind.Counter

    let unitKey (e: RunEvent) =
        intOr -1L "lane" e, intOr -1L TelemetryKeys.Stage e, intOr -1L "step" e

    let reissued = reissues |> List.map unitKey |> Set.ofList

    let doneEvents =
        named "cluster.unit.done" TelemetryKind.Event
        |> List.filter (fun e -> (e.TryInt "start_ns").IsSome && (e.TryInt "end_ns").IsSome)

    let units =
        doneEvents
        |> List.map (fun e ->
            let key = unitKey e
            let lane, stage, step = key

            {
                Lane = lane
                Stage = stage
                Step = step
                Node = nodeOf e
                StartNs = intOr 0L "start_ns" e
                EndNs = intOr 0L "end_ns" e
                Status = (if reissued.Contains key then "reissued" else "accepted")
            })

    // The run log's clock minus the coordinator's, from the first accepted unit.
    let offset =
        match doneEvents with
        | e :: _ -> e.TimeNs - intOr 0L "end_ns" e
        | [] -> 0L

    let coordTime (e: RunEvent) = e.TimeNs - offset

    // The zero of every time axis: the coordinator's first cluster record (a join, usually), on its own clock.
    let t0 =
        match doneEvents with
        | [] -> runStart log
        | first :: _ ->
            let starts = units |> List.map _.StartNs

            let records =
                log.Prefixed "cluster."
                |> List.filter (fun e -> e.Node = first.Node)
                |> List.map coordTime

            List.min (starts @ records)

    let stageText (s: int64) = Invariant.i64 s

    // Lanes: (node, stage) in stage order, so a stage's replicas sit together.
    let laneKeys =
        units
        |> List.map (fun u -> u.Node, u.Stage)
        |> List.distinct
        |> List.sortBy (fun (node, stage) -> stage, node)

    let laneIndex = laneKeys |> List.mapi (fun i k -> k, i) |> Map.ofList

    let laneName (node: string, stage: int64) = node + "  stage " + stageText stage

    let leaves = named "cluster.member.leave" TelemetryKind.Event
    let demotes = named "cluster.demote" TelemetryKind.Event
    let divergences = named "cluster.divergence" TelemetryKind.Event

    let kills =
        named "cluster.chaos" TelemetryKind.Counter
        |> List.filter (fun e -> e.TryStr "fault" = Some "kill" || e.TryStr "fault" = Some "restart")

    let outcome = named "cluster.run" TelemetryKind.Event |> List.tryLast

    let marks =
        [
            for e in leaves ->
                let reason = defaultArg (e.TryStr "reason") "left"
                coordTime e, nodeOf e + " left (" + reason + ")", Color.Critical
            for e in kills ->
                let killed = e.TryStr "fault" = Some "kill"

                coordTime e,
                nodeOf e + (if killed then " killed" else " restarted"),
                (if killed then Color.Critical else Color.Warning)
            for e in demotes ->
                coordTime e,
                nodeOf e
                + " demoted from stage "
                + defaultArg (e.TryText TelemetryKeys.Stage) "?",
                Color.Warning
        ]
        |> List.filter (fun (t, _, _) -> t >= t0)
        |> List.sortBy (fun (t, label, _) -> t, label)

    let gantt =
        if units.IsEmpty then
            noData "Stage Gantt chart" "cluster.unit.done event"
        else
            Chart.Timeline
                {
                    Title = "Stage Gantt chart"
                    Subtitle =
                        "Each accepted work unit on the node whose result was taken, from assignment to acceptance; markers are node losses, kills and demotions"
                    XLabel = "time since the first cluster record"
                    XFormat = NumFormat.Duration
                    Lanes = laneKeys |> List.map laneName
                    Intervals =
                        units
                        |> List.map (fun u ->
                            {
                                Lane = laneIndex[(u.Node, u.Stage)]
                                Start = float (u.StartNs - t0)
                                Length = float (u.EndNs - u.StartNs)
                                Color = statusColor u.Status
                                Tip =
                                    Invariant.format
                                        $"{u.Node} stage {u.Stage}, lane {u.Lane} step {u.Step}: {u.Status} after {Fmt.duration (float (u.EndNs - u.StartNs))}"
                            })
                    Marks =
                        marks
                        |> List.map (fun (t, label, color) ->
                            {
                                At = float (t - t0)
                                Label = label
                                Color = color
                            })
                    Legend =
                        statuses
                        |> List.filter (fun (s, _) -> units |> List.exists (fun u -> u.Status = s))
                }

    // A step's latency: from the assignment of its first stage to the acceptance of its last, per lane.
    let stageList = units |> List.map _.Stage |> List.distinct |> List.sort

    let firstStage = List.tryHead stageList |> Option.defaultValue 0L
    let lastStage = List.tryLast stageList |> Option.defaultValue 0L
    let lanes = units |> List.map _.Lane |> List.distinct |> List.sort

    let stepLatency (lane: int64) =
        units
        |> List.filter (fun u -> u.Lane = lane)
        |> List.groupBy _.Step
        |> List.choose (fun (step, us) ->
            let first = us |> List.filter (fun u -> u.Stage = firstStage)
            let last = us |> List.filter (fun u -> u.Stage = lastStage)

            if first.IsEmpty || last.IsEmpty then
                None
            else
                let start = first |> List.map _.StartNs |> List.min
                let finish = last |> List.map _.EndNs |> List.max
                Some(float step, float (finish - start)))
        |> List.sortBy fst

    let latencySeries =
        lanes
        |> List.mapi (fun i lane ->
            {
                Name = "lane " + Invariant.i64 lane
                Color = (if i < 8 then Color.Cat i else Color.Neutral)
                Points = stepLatency lane
            })
        |> List.filter (fun s -> not s.Points.IsEmpty)

    let stepChart =
        if latencySeries.IsEmpty then
            noData "Step latency" "cluster.unit.done event for the first and the last stage"
        else
            Chart.Line
                {
                    Title = "End-to-end step latency"
                    Subtitle =
                        Invariant.format
                            $"Per lane: from the assignment of stage {firstStage} to the acceptance of stage {lastStage}"
                    XLabel = "step"
                    YLabel = "latency"
                    XFormat = NumFormat.Int
                    YFormat = NumFormat.Duration
                    YLog = false
                    YFromZero = true
                    IntegerX = true
                    Series = latencySeries
                    RefLines =
                        [
                            {
                                Value = median (latencySeries |> List.collect (fun s -> s.Points |> List.map snd))
                                Label = "median"
                            }
                        ]
                    Bands = []
                }

    let deadlines = named "cluster.deadline" TelemetryKind.Gauge

    let deadlineChart =
        if deadlines.IsEmpty then
            noData "Stage deadlines" "cluster.deadline gauge"
        else
            let stages =
                deadlines
                |> List.map (intOr -1L TelemetryKeys.Stage)
                |> List.distinct
                |> List.sort

            // A deadline holds until it changes, so each series is drawn as steps up to the end of the run.
            let tEnd =
                [
                    yield! units |> List.map _.EndNs
                    yield! deadlines |> List.map coordTime
                    match outcome with
                    | Some e -> yield coordTime e
                    | None -> ()
                ]
                |> List.max

            Chart.Line
                {
                    Title = "Stage deadlines"
                    Subtitle =
                        "A deadline counts only after Ready: 30 s at first, then clamp(4·p95 + 50 ms, 250 ms, 30 s) once a stage has 8 samples"
                    XLabel = "time since the first cluster record"
                    YLabel = "deadline"
                    XFormat = NumFormat.Duration
                    YFormat = NumFormat.Duration
                    YLog = false
                    YFromZero = true
                    IntegerX = false
                    Series =
                        stages
                        |> List.mapi (fun i stage ->
                            let xs =
                                deadlines
                                |> List.filter (fun e -> intOr -1L TelemetryKeys.Stage e = stage)
                                |> List.map (fun e -> float (coordTime e - t0), e.Value)

                            let steps =
                                [
                                    for j, (x, v) in List.indexed xs do
                                        if j > 0 then
                                            x, snd xs[j - 1]

                                        x, v
                                ]

                            {
                                Name = "stage " + stageText stage
                                Color = (if i < 8 then Color.Cat i else Color.Neutral)
                                Points = steps @ [ float (tEnd - t0), snd (List.last xs) ]
                            })
                    RefLines = []
                    Bands = []
                }

    let latencies =
        named "cluster.unit.latency" TelemetryKind.Histogram |> List.map _.Value

    let latencyChart =
        if latencies.IsEmpty then
            noData "Unit latency" "cluster.unit.latency histogram"
        else
            let b = bin 24 true latencies

            Chart.Histogram
                {
                    Title = "Unit latency, deadline start to accepted result"
                    Subtitle = "Every deadline sample of every stage, on a log scale"
                    XLabel = "latency"
                    XFormat = NumFormat.Decades NumFormat.Duration
                    Bins = b.Bins
                    Zeros = b.Zeros
                    Invalid = b.Invalid
                    Color = Color.Cat 0
                    Marks = []
                }

    let busy =
        if units.IsEmpty then
            noData "Accepted work per node" "cluster.unit.done event"
        else
            let used =
                statuses
                |> List.filter (fun (s, _) -> units |> List.exists (fun u -> u.Status = s))

            Chart.Bar
                {
                    Title = "Accepted work per node and stage, by outcome"
                    Subtitle =
                        "Sum of the assignment-to-acceptance time of the units whose result was taken from each node"
                    CategoryLabel = "node"
                    ValueLabel = "time"
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
                                        |> List.filter (fun u -> u.Node = node && u.Stage = stage && u.Status = s)
                                        |> List.sumBy (fun u -> float (u.EndNs - u.StartNs)))
                            })
                    Stacked = true
                    Horizontal = true
                }

    let counted (name: string) (node: string) (stage: int64) =
        named name TelemetryKind.Counter
        |> List.filter (fun e -> nodeOf e = node && intOr -1L TelemetryKeys.Stage e = stage)
        |> List.sumBy _.Value

    // Every (node, stage) that was assigned work or had a result accepted.
    let nodeKeys =
        [
            yield! laneKeys
            yield!
                named "cluster.assign" TelemetryKind.Counter
                |> List.map (fun e -> nodeOf e, intOr -1L TelemetryKeys.Stage e)
        ]
        |> List.distinct
        |> List.sortBy (fun (node, stage) -> stage, node)

    let nodeRows =
        [
            for node, stage in nodeKeys ->
                let xs = units |> List.filter (fun u -> u.Node = node && u.Stage = stage)
                let backend = backendOf log node

                [
                    node
                    stageText stage
                    (if backend = "" then "n/a" else backend)
                    Fmt.int (int64 (counted "cluster.assign" node stage))
                    Fmt.int (int64 xs.Length)
                    Fmt.int (int64 (xs |> List.filter (fun u -> u.Status = "reissued")).Length)
                    Fmt.int (int64 (counted "cluster.duplicate" node stage))
                    Fmt.int (int64 (counted "cluster.deadline.miss" node stage))
                    Fmt.duration (xs |> List.sumBy (fun u -> float (u.EndNs - u.StartNs)))
                ]
        ]

    let counterRows =
        [
            for name, split in counterOrder do
                let xs = named name TelemetryKind.Counter

                if not xs.IsEmpty then
                    let groups =
                        if split = "" then
                            [ "", xs ]
                        else
                            xs
                            |> List.groupBy (fun e -> split + "=" + defaultArg (e.TryText split) "?")
                            |> List.sortBy fst

                    for detail, es in groups -> [ name; detail; Fmt.int (int64 (es |> List.sumBy _.Value)) ]
        ]

    let eventRows =
        [
            yield!
                log.Prefixed "cluster."
                |> List.filter (fun e ->
                    e.Kind = TelemetryKind.Event
                    && e.Name <> "cluster.unit.done"
                    && e.Name <> "cluster.ready")
            yield! kills
        ]
        |> List.sortBy (fun e -> coordTime e, e.Node, e.Seq)
        |> List.truncate 40
        |> List.map (fun e ->
            let detail =
                e.Attrs
                |> List.filter (fun a -> a.Key <> TelemetryKeys.Component && a.Key <> "run_digest")
                |> List.map (fun a -> a.Key + "=" + defaultArg (e.TryText a.Key) "")
                |> String.concat " "

            [ Fmt.duration (float (coordTime e - t0)); e.Node; e.Name; detail ])

    let outcomeInt (key: string) =
        outcome |> Option.bind (fun e -> e.TryInt key)

    let status =
        outcome
        |> Option.bind (fun e -> e.TryStr "status")
        |> Option.defaultValue "not finished"

    let unitCount = outcomeInt "units" |> Option.defaultValue (int64 units.Length)
    let reissueCount = reissues |> List.sumBy _.Value |> int64

    let verdictText =
        let lostText = plural leaves.Length "node loss" "node losses"
        let reissueText = plural (int reissueCount) "unit reissued" "units reissued"
        let divergenceText = plural divergences.Length "divergence" "divergences"

        match outcome, status with
        | None, _ when units.IsEmpty -> "No work units were recorded."
        | None, _ ->
            Invariant.format
                $"**INCOMPLETE.** {Fmt.int (int64 units.Length)} work units were accepted, but no cluster.run outcome was recorded."
        | Some _, "completed" when divergences.IsEmpty ->
            Invariant.format
                $"**PASS.** {Fmt.int unitCount} work units, every one cross-checked by a second replica; {reissueText}, {lostText}, no divergence."
        | Some _, "unverified" ->
            let unconfirmed = outcomeInt "unconfirmed" |> Option.defaultValue 0L

            Invariant.format
                $"**UNVERIFIED.** {Fmt.int unitCount} work units completed, but {Fmt.int unconfirmed} were never cross-checked by a second replica; {divergenceText}."
        | Some _, s -> Invariant.format $"**FAIL.** The run ended {s}: {divergenceText}, {reissueText}, {lostText}."

    runScope log
    |> Scope.value "verdict" verdictText
    |> Scope.value "status" status
    |> Scope.value "units" (Fmt.int unitCount)
    |> Scope.value
        "cross_checked"
        (match outcomeInt "cross_checked" with
         | Some c -> Fmt.int c + " of " + Fmt.int unitCount
         | None -> "n/a")
    |> Scope.value "reissued" (Fmt.int reissueCount)
    |> Scope.value "lost" (Fmt.int (int64 leaves.Length))
    |> Scope.value "divergences" (Fmt.int (int64 divergences.Length))
    |> Scope.value "epoch" (outcomeInt "epoch" |> Option.map Fmt.int |> Option.defaultValue "n/a")
    |> Scope.value
        "digest"
        (outcome
         |> Option.bind (fun e -> e.TryStr "run_digest")
         |> Option.map (fun d -> "`" + d + "`")
         |> Option.defaultValue "not recorded")
    |> Scope.table
        "nodes"
        (table
            [
                "Node", L
                "Stage", R
                "Backend", L
                "Assigned", R
                "Accepted", R
                "Reissued to it", R
                "Duplicates", R
                "Deadline misses", R
                "Accepted time", R
            ]
            nodeRows)
    |> Scope.table "counters" (table [ "Counter", L; "Split", L; "Total", R ] counterRows)
    |> Scope.table "events" (table [ "Time", R; "Node", L; "Event", L; "Detail", L ] eventRows)
    |> Scope.table "meters" (table Inference.meterColumns (Inference.meterRows log.Events))
    |> Scope.chart "gantt" gantt
    |> Scope.chart "step_latency" stepChart
    |> Scope.chart "deadlines" deadlineChart
    |> Scope.chart "latency" latencyChart
    |> Scope.chart "busy" busy
