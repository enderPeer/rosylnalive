/// The parity report: agreement of one implementation with the goldens, tap by tap (ReportTemplates contract,
/// "parity"). Reads golden.compare checks.
module Lrm.Reports.Builders.Parity

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

type private Cmp =
    {
        Pair: string
        Case: string
        Tap: string
        Metric: string
        Layer: int option
        Measured: float
        Bound: float
        Passed: bool
    }

    /// measured / bound for at-most metrics; NaN for greedy_match, whose bound is a minimum.
    member c.Ratio =
        if c.Metric = "greedy_match" then nan
        elif c.Bound > 0.0 then c.Measured / c.Bound
        elif c.Measured = 0.0 then 0.0
        else infinity

let private compares (log: RunLog) =
    log.Named TelemetryNames.GoldenCompare
    |> List.filter (fun e -> e.Kind = TelemetryKind.Check)
    |> List.map (fun e ->
        let tap = defaultArg (e.TryStr "tap") ""

        {
            Pair = defaultArg (e.TryStr "pair") "(no pair)"
            Case = defaultArg (e.TryStr "case") ""
            Tap = tap
            Metric = defaultArg (e.TryStr "metric") ""
            Layer = e.TryInt TelemetryKeys.Layer |> Option.map int |> Option.orElse (tapLayer tap)
            Measured = defaultArg (e.TryFloat "measured") nan
            Bound = defaultArg (e.TryFloat "bound") nan
            Passed = e.Value = 1.0
        })

let private relL2 = "rel_l2"

let private heatmapOf (pair: string) (cmps: Cmp list) =
    let layered =
        cmps
        |> List.filter (fun c -> c.Pair = pair && c.Metric = relL2 && c.Layer.IsSome)

    if layered.IsEmpty then
        noData ("Worst measured/bound, " + pair) "per-layer rel_l2 golden.compare"
    else
        let kinds = layered |> List.map (fun c -> tapKind c.Tap) |> List.distinct

        let layers =
            layered |> List.map (fun c -> c.Layer.Value) |> List.distinct |> List.sort

        let cell kind layer =
            layered
            |> List.filter (fun c -> tapKind c.Tap = kind && c.Layer = Some layer)
            |> List.map _.Ratio
            |> maxOr nan

        Chart.Heatmap
            {
                Title = "Worst measured/bound by tap and layer, " + pair
                Subtitle = "rel_l2 over all cases; 1.0 is the tolerance bound, darker is closer to it"
                RowLabel = "tap"
                ColLabel = "layer"
                Rows = kinds
                Cols = layers |> List.map Invariant.i32
                Values = [| for k in kinds -> [| for l in layers -> cell k l |] |]
                Log = true
                ValueFormat = NumFormat.Sci
            }

/// Builds the parity scope.
let build (log: RunLog) : Scope =
    let cmps = compares log
    let failed = cmps |> List.filter (fun c -> not c.Passed)
    let pairs = cmps |> List.map _.Pair |> sortIn pairOrder
    let cases = cmps |> List.map _.Case |> List.distinct |> List.sort

    let verdictText =
        if cmps.IsEmpty then
            "No golden comparisons were recorded."
        elif failed.IsEmpty then
            Invariant.format $"**PASS.** All {Fmt.int (int64 cmps.Length)} comparisons are within their bounds."
        else
            Invariant.format
                $"**FAIL.** {Fmt.int (int64 failed.Length)} of {Fmt.int (int64 cmps.Length)} comparisons exceed their bounds."

    let worst =
        cmps
        |> List.filter (fun c -> Double.IsFinite c.Ratio)
        |> List.sortByDescending _.Ratio
        |> List.tryHead

    let layered = cmps |> List.filter (fun c -> c.Metric = relL2 && c.Layer.IsSome)

    let layers =
        layered |> List.map (fun c -> c.Layer.Value) |> List.distinct |> List.sort

    let errorBars =
        if layered.IsEmpty then
            noData "rel_l2 by layer" "per-layer rel_l2 golden.compare"
        else
            Chart.ErrorBars
                {
                    Title = "rel_l2 by layer"
                    Subtitle =
                        "Dot: median over cases and taps of the layer; whiskers: min to max. Lines: each pair's bound."
                    XLabel = "layer"
                    YLabel = "rel_l2 (log)"
                    YFormat = NumFormat.Sci
                    YLog = true
                    Categories = layers |> List.map Invariant.i32
                    Series =
                        [
                            for p in pairs do
                                let xs = layered |> List.filter (fun c -> c.Pair = p)

                                if not xs.IsEmpty then
                                    {
                                        Name = p
                                        Color = colorIn pairOrder p
                                        Spreads =
                                            layers
                                            |> List.mapi (fun i l ->
                                                let v =
                                                    xs
                                                    |> List.filter (fun c -> c.Layer = Some l)
                                                    |> List.map _.Measured

                                                {
                                                    Index = i
                                                    Low = minOr nan v
                                                    Mid = median v
                                                    High = maxOr nan v
                                                })
                                    }
                        ]
                    RefLines =
                        [
                            for p in pairs do
                                let b =
                                    layered |> List.filter (fun c -> c.Pair = p) |> List.map _.Bound |> maxOr nan

                                if Double.IsFinite b then
                                    {
                                        Value = b
                                        Label = "bound " + p + " " + Fmt.sci 1 b
                                    }
                        ]
                }

    let ratioHist =
        let ratios =
            cmps |> List.map _.Ratio |> List.filter (fun r -> r > 0.0 && Double.IsFinite r)

        if ratios.IsEmpty then
            noData "measured/bound" "golden.compare"
        else
            Chart.Histogram
                {
                    Title = "Distance to the bound, every comparison"
                    Subtitle = "measured / bound on a log scale; everything left of the bound line passes"
                    XLabel = "measured / bound"
                    XFormat = NumFormat.Pow10
                    Bins = bins 24 true ratios
                    Color = Color.Cat 0
                    Marks = [ { At = 0.0; Label = "bound" } ]
                }

    let caseRows =
        [
            for case in cases do
                let xs = cmps |> List.filter (fun c -> c.Case = case)
                let bad = xs |> List.filter (fun c -> not c.Passed)

                let w =
                    xs
                    |> List.filter (fun c -> Double.IsFinite c.Ratio)
                    |> List.sortByDescending _.Ratio
                    |> List.tryHead

                let logits =
                    xs
                    |> List.filter (fun c -> c.Tap = "logits" && c.Metric = relL2)
                    |> List.map (fun c -> c.Pair + " " + Fmt.sci 2 c.Measured)
                    |> String.concat ", "

                let greedy =
                    xs
                    |> List.filter (fun c -> c.Metric = "greedy_match")
                    |> List.map (fun c ->
                        Fmt.dp 0 c.Measured + "/" + Fmt.dp 0 c.Bound + (if c.Passed then "" else " ✗"))
                    |> String.concat ", "

                [
                    case
                    Fmt.int (int64 xs.Length)
                    Fmt.int (int64 bad.Length)
                    (match w with
                     | Some c -> c.Pair + " " + c.Tap + " " + c.Metric
                     | None -> "")
                    (match w with
                     | Some c -> Fmt.dp 3 c.Ratio
                     | None -> "")
                    (if logits = "" then "n/a" else logits)
                    (if greedy = "" then "n/a" else greedy)
                ]
        ]

    let closest =
        (if failed.IsEmpty then cmps else failed)
        |> List.filter (fun c -> Double.IsFinite c.Ratio)
        |> List.sortBy (fun c -> -c.Ratio, c.Pair, c.Case, c.Tap, c.Metric)
        |> List.truncate 15

    let worstRows =
        closest
        |> List.map (fun c ->
            [
                c.Pair
                c.Case
                c.Tap
                c.Metric
                Fmt.sci 2 c.Measured
                Fmt.sci 2 c.Bound
                Fmt.dp 3 c.Ratio
                mark c.Passed
            ])

    let pairItems =
        pairs
        |> List.map (fun p ->
            let xs = cmps |> List.filter (fun c -> c.Pair = p)
            let bad = xs |> List.filter (fun c -> not c.Passed)

            Scope.empty
            |> Scope.value "name" p
            |> Scope.value "checks" (Fmt.int (int64 xs.Length))
            |> Scope.value "failed" (Fmt.int (int64 bad.Length))
            |> Scope.value "verdict" (verdict bad.IsEmpty)
            |> Scope.value "worst" (xs |> List.map _.Ratio |> maxOr nan |> Fmt.dp 3)
            |> Scope.chart "heatmap" (heatmapOf p cmps))

    runScope log
    |> Scope.value "verdict" verdictText
    |> Scope.value "checks" (Fmt.int (int64 cmps.Length))
    |> Scope.value "failed" (Fmt.int (int64 failed.Length))
    |> Scope.value "pairs" (if pairs.IsEmpty then "none" else String.concat ", " pairs)
    |> Scope.value "cases" (Invariant.i32 cases.Length)
    |> Scope.value
        "worst"
        (match worst with
         | Some c -> Invariant.format $"{c.Pair} {c.Case} {c.Tap} {c.Metric}: {Fmt.dp 3 c.Ratio} of the bound"
         | None -> "n/a")
    |> Scope.value
        "closest.title"
        (if failed.IsEmpty then
             "Closest to the bound"
         else
             "Failing comparisons")
    |> Scope.chart "rel_l2_by_layer" errorBars
    |> Scope.chart "ratio_histogram" ratioHist
    |> Scope.table
        "cases"
        (table
            [
                "Case", L
                "Checks", R
                "Failed", R
                "Worst comparison", L
                "Worst ratio", R
                "Logits rel_l2", L
                "Greedy match", L
            ]
            caseRows)
    |> Scope.table
        "closest"
        (table
            [
                "Pair", L
                "Case", L
                "Tap", L
                "Metric", L
                "Measured", R
                "Bound", R
                "Ratio", R
                "Result", L
            ]
            worstRows)
    |> Scope.list "pairs" pairItems
