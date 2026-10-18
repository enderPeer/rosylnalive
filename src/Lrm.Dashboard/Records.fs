/// The dashboard's view of a run whose kind has no report template (ADR T2-0001): every record name with what it
/// adds up to, checks passed and failed per name, span time per name and the metering table. Like T1's builders it
/// is a pure function of the run log and never fails on missing records.
module Lrm.Dashboard.Records

open System
open System.IO
open System.Reflection
open System.Text
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// The template name, also the prefix of its chart files.
[<Literal>]
let Name = "records"

type private Marker = class end

/// The embedded template text (templates/records.md), with LF line ends.
let templateText =
    lazy
        (let asm = typeof<Marker>.Assembly
         use stream = asm.GetManifestResourceStream("dashboard/records.md")

         if isNull stream then
             invalidOp "no embedded template dashboard/records.md"

         use reader = new StreamReader(stream, UTF8Encoding(false, true))
         reader.ReadToEnd().Replace("\r\n", "\n"))

/// The most bars a chart shows; the rest fold into "other".
[<Literal>]
let MaxBars = 16

let private kindOrder = TelemetryKind.all |> List.map TelemetryKind.name

/// What one record name adds up to, in its unit.
let private summary (kind: TelemetryKind) (unit: string) (xs: RunEvent list) =
    let value (v: float) =
        match unit with
        | TelemetryUnits.Nanoseconds -> Fmt.duration v
        | TelemetryUnits.Bytes -> Fmt.bytes v
        | "" -> Fmt.num v
        | u -> Fmt.num v + " " + u

    let values = xs |> List.map _.Value

    match kind with
    | TelemetryKind.Check ->
        let passed = xs |> List.filter (fun e -> e.Value = 1.0) |> List.length

        Fmt.int (int64 passed)
        + " passed, "
        + Fmt.int (int64 (xs.Length - passed))
        + " failed"
    | TelemetryKind.Counter
    | TelemetryKind.Meter -> "total " + value (List.sum values)
    | TelemetryKind.Gauge ->
        "last "
        + value (List.last values)
        + ", range "
        + value (minOr nan values)
        + " to "
        + value (maxOr nan values)
    | TelemetryKind.Histogram -> "median " + value (median values) + ", p95 " + value (quantile 0.95 values)
    | TelemetryKind.Span
    | TelemetryKind.GpuInterval ->
        let ds = xs |> List.map (fun e -> float e.DurationNs)
        "total " + Fmt.duration (List.sum ds) + ", median " + Fmt.duration (median ds)
    | TelemetryKind.Event -> ""

/// Folds the bars past MaxBars into one "other" bar.
let private topBars (bars: (string * float list) list) =
    if bars.Length <= MaxBars then
        bars
    else
        let keep = List.take (MaxBars - 1) bars
        let rest = List.skip (MaxBars - 1) bars
        let width = (snd bars.Head).Length

        keep
        @ [
            Invariant.format $"other ({rest.Length} names)",
            List.init width (fun k -> rest |> List.sumBy (fun (_, vs) -> vs[k]))
        ]

/// Builds the records scope.
let build (log: RunLog) : Scope =
    let groups =
        log.Events
        |> List.groupBy (fun e -> e.Kind, e.Name)
        |> List.sortBy (fun ((k, n), _) -> orderIn kindOrder (TelemetryKind.name k), n)

    let checks =
        groups
        |> List.filter (fun ((k, _), _) -> k = TelemetryKind.Check)
        |> List.map (fun ((_, n), xs) ->
            let passed = xs |> List.filter (fun e -> e.Value = 1.0) |> List.length
            n, passed, xs.Length - passed)
        // Failures first, then the most checked names, then by name.
        |> List.sortBy (fun (n, p, f) -> -f, -(p + f), n)

    let failed = checks |> List.sumBy (fun (_, _, f) -> f)
    let total = checks |> List.sumBy (fun (_, p, f) -> p + f)

    let verdict =
        if total = 0 then
            "This run has no checks."
        elif failed = 0 then
            "**PASS.** " + plural total "check" "checks" + ", every one passed."
        else
            "**FAIL.** "
            + plural failed "check" "checks"
            + " of "
            + Fmt.int (int64 total)
            + " failed."

    let checksChart =
        if checks.IsEmpty then
            noData "Checks by name" "check"
        else
            let bars = topBars [ for n, p, f in checks -> n, [ float p; float f ] ]

            Chart.Bar
                {
                    Title = "Checks by name"
                    Subtitle = "Passed and failed checks per record name, failures first"
                    CategoryLabel = "check"
                    ValueLabel = "checks"
                    ValueFormat = NumFormat.Int
                    Categories = bars |> List.map fst
                    Series =
                        [
                            {
                                Name = "passed"
                                Color = Color.Good
                                Values = bars |> List.map (fun (_, vs) -> vs[0])
                            }
                            {
                                Name = "failed"
                                Color = Color.Critical
                                Values = bars |> List.map (fun (_, vs) -> vs[1])
                            }
                        ]
                    Stacked = true
                    Horizontal = true
                }

    let spanTotals =
        groups
        |> List.filter (fun ((k, _), _) -> k = TelemetryKind.Span || k = TelemetryKind.GpuInterval)
        |> List.map (fun ((k, n), xs) ->
            (if k = TelemetryKind.GpuInterval then n + " (gpu)" else n), xs |> List.sumBy (fun e -> float e.DurationNs))
        |> List.sortBy (fun (n, ns) -> -ns, n)

    let spansChart =
        if spanTotals.IsEmpty then
            noData "Time by record name" "span or gpu"
        else
            let bars = topBars [ for n, ns in spanTotals -> n, [ ns ] ]

            Chart.Bar
                {
                    Title = "Time by record name"
                    Subtitle = "Total duration of the spans and GPU intervals of each name, nested spans included"
                    CategoryLabel = "record"
                    ValueLabel = "time"
                    ValueFormat = NumFormat.Duration
                    Categories = bars |> List.map fst
                    Series =
                        [
                            {
                                Name = "time"
                                Color = Color.slot 0
                                Values = bars |> List.map (fun (_, vs) -> vs[0])
                            }
                        ]
                    Stacked = false
                    Horizontal = true
                }

    let recordRows =
        [
            for (k, n), xs in groups ->
                let unit = xs.Head.Unit

                [ TelemetryKind.name k; n; Fmt.int (int64 xs.Length); unit; summary k unit xs ]
        ]

    let names =
        groups |> List.map (fun ((_, n), _) -> n) |> List.distinct |> List.length

    let summaryText =
        plural log.Events.Length "record" "records"
        + " under "
        + plural names "name" "names"
        + " from "
        + plural log.Nodes.Length "node" "nodes"
        + "."

    runScope log
    |> Scope.value "verdict" verdict
    |> Scope.value "summary" summaryText
    |> Scope.chart "checks" checksChart
    |> Scope.chart "spans" spansChart
    |> Scope.table
        "checks"
        (table
            [ "Check", L; "Passed", R; "Failed", R; "Total", R ]
            [
                for n, p, f in checks -> [ n; Fmt.int (int64 p); Fmt.int (int64 f); Fmt.int (int64 (p + f)) ]
            ])
    |> Scope.table "records" (table [ "Kind", L; "Name", L; "Records", R; "Unit", L; "Adds up to", L ] recordRows)
    |> Scope.table "meters" (table Builders.Inference.meterColumns (Builders.Inference.meterRows log.Events))
