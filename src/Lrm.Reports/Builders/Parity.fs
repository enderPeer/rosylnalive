/// The parity report: agreement of one implementation with the goldens, tap by tap (ReportTemplates contract,
/// "parity"). Reads golden.compare checks.
///
/// A comparison whose measured value (or bound) is NaN or infinite is the commonest GPU failure, so it is never left
/// out: it sorts first in the failure table, counts as the worst comparison, is a crossed critical cell in the heatmap
/// and a cross in the NaN/inf strip of the error bars.
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
        /// False for a measurement, such as an e_ref (ReferenceOracles), which records a value without a bound.
        HasBound: bool
        Passed: bool
    }

    /// greedy_match has a minimum as its bound, not a maximum: it has no ratio; nor has a measurement without a bound.
    member c.HasRatio = c.Metric <> "greedy_match" && c.HasBound

    /// measured / bound for at-most metrics: NaN when either is NaN, infinity when the bound is 0 and the measured
    /// value is not, NaN for greedy_match.
    member c.Ratio =
        if not c.HasRatio || Double.IsNaN c.Measured || Double.IsNaN c.Bound then
            nan
        elif c.Bound > 0.0 then
            c.Measured / c.Bound
        elif c.Measured = 0.0 then
            0.0
        else
            infinity

    /// True when the comparison has a ratio and it is NaN or infinite.
    member c.IsInvalid = c.HasRatio && not (Double.IsFinite c.Ratio)

/// The order of "worst first": invalid ratios (NaN before infinity), then finite ratios from the largest, then
/// comparisons without a ratio; ties by pair, case, tap and metric.
let private worstFirst (c: Cmp) =
    let rank, key =
        if c.IsInvalid then
            (if Double.IsNaN c.Ratio then 0 else 1), 0.0
        elif c.HasRatio then
            2, -c.Ratio
        else
            3, 0.0

    rank, key, c.Pair, c.Case, c.Tap, c.Metric

/// A ratio as table text: three decimals, "NaN" or "inf", or "n/a" for greedy_match.
let private ratioText (c: Cmp) =
    if c.HasRatio then Fmt.dp 3 c.Ratio else "n/a"

/// The layer of a comparison: `layer` when it is 0 or more, else parsed from `L<i>.` in the tap. A negative `layer` is
/// ComparatorsAndGoldenReader's -1 for a model-level tap (embed.out, final_norm.out, logits), which has no layer.
let private layerOf (e: RunEvent) (tap: string) =
    match e.TryInt TelemetryKeys.Layer with
    | Some l when l >= 0L -> Some(int l)
    | Some _ -> None
    | None -> tapLayer tap

let private compares (log: RunLog) =
    log.Named TelemetryNames.GoldenCompare
    |> List.filter (fun e -> e.Kind = TelemetryKind.Check)
    |> List.map (fun e ->
        let tap = defaultArg (e.TryStr "tap") ""

        {
            Pair = defaultArg (e.TryStr "pair") "(no pair)"
            Case = compareCase e |> Option.defaultValue ""
            Tap = tap
            Metric = metricName (defaultArg (e.TryStr "metric") "")
            Layer = layerOf e tap
            Measured = defaultArg (e.TryFloat "measured") nan
            Bound = defaultArg (e.TryFloat "bound") nan
            HasBound = (e.TryFloat "bound").IsSome
            Passed = e.Value = 1.0
        })

let private relL2 = "rel_l2"

/// The worst ratio of some comparisons as text ("NaN" or "inf" when one is invalid), or "n/a".
let private worstRatioText (xs: Cmp list) =
    xs
    |> List.filter _.HasRatio
    |> List.sortBy worstFirst
    |> List.tryHead
    |> Option.map ratioText
    |> Option.defaultValue "n/a"

let private heatmapOf (pair: string) (cmps: Cmp list) =
    let layered =
        cmps
        |> List.filter (fun c -> c.Pair = pair && c.Metric = relL2 && c.Layer.IsSome && c.HasRatio)

    if layered.IsEmpty then
        noData ("Worst measured/bound, " + pair) "per-layer rel_l2 golden.compare"
    else
        let kinds = layered |> List.map (fun c -> tapKind c.Tap) |> List.distinct

        let layers =
            layered |> List.map (fun c -> c.Layer.Value) |> List.distinct |> List.sort

        let cell kind layer =
            let ratios =
                layered
                |> List.filter (fun c -> tapKind c.Tap = kind && c.Layer = Some layer)
                |> List.map _.Ratio

            if ratios.IsEmpty then
                None
            elif ratios |> List.exists (fun r -> not (Double.IsFinite r)) then
                Some nan
            else
                Some(List.max ratios)

        Chart.Heatmap
            {
                Title = "Worst measured/bound by tap and layer, " + pair
                Subtitle = "rel_l2 over all cases; the scale ends at the bound (1.0), so darker is closer to it"
                RowLabel = "tap"
                ColLabel = "layer"
                Rows = kinds
                Cols = layers |> List.map Invariant.i32
                ColValues = []
                Values = [| for k in kinds -> [| for l in layers -> cell k l |] |]
                Log = true
                ValueFormat = NumFormat.Sci
                Ceiling = Some(1.0, "over bound")
                Aggregate = HeatAggregate.Max
            }

/// The most rows the closest and failing table shows.
let private tableRows = 40

/// Builds the parity scope.
let build (log: RunLog) : Scope =
    let cmps = compares log
    let failed = cmps |> List.filter (fun c -> not c.Passed)
    let pairs = cmps |> List.map _.Pair |> sortIn pairOrder
    let cases = cmps |> List.map _.Case |> List.distinct |> List.sort
    let invalid = cmps |> List.filter _.IsInvalid

    let verdictText =
        if cmps.IsEmpty then
            "No golden comparisons were recorded."
        elif failed.IsEmpty then
            Invariant.format $"**PASS.** All {Fmt.int (int64 cmps.Length)} comparisons are within their bounds."
        else
            Invariant.format
                $"**FAIL.** {Fmt.int (int64 failed.Length)} of {Fmt.int (int64 cmps.Length)} comparisons exceed their bounds"
            + (if invalid.IsEmpty then
                   "."
               else
                   let them = plural invalid.Length "of them is" "of them are"
                   Invariant.format $", {them} NaN or infinite.")

    let worst = cmps |> List.filter _.HasRatio |> List.sortBy worstFirst |> List.tryHead

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
                                                xs
                                                |> List.filter (fun c -> c.Layer = Some l)
                                                |> List.map _.Measured
                                                |> spread i median)
                                    }
                        ]
                    // One line per distinct bound: pairs that share a bound share its line and label, so labels
                    // never print over each other.
                    RefLines =
                        [
                            for p in pairs do
                                let b =
                                    layered
                                    |> List.filter (fun c -> c.Pair = p && c.HasBound)
                                    |> List.map _.Bound
                                    |> maxOr nan

                                if Double.IsFinite b then
                                    p, b
                        ]
                        |> List.groupBy snd
                        |> List.map (fun (b, ps) ->
                            {
                                Value = b
                                Label = "bound " + String.concat ", " (List.map fst ps) + " " + Fmt.sci 1 b
                            })
                }

    let ratioHist =
        let ratios = cmps |> List.filter _.HasRatio |> List.map _.Ratio

        if ratios.IsEmpty then
            noData "measured/bound" "golden.compare"
        else
            let b = bin 24 true ratios

            Chart.Histogram
                {
                    Title = "Distance to the bound, every comparison"
                    Subtitle =
                        "measured / bound on a log scale; left of the bound line passes"
                        + (if b.Zeros > 0 then
                               "; exact matches in the 0 column"
                           else
                               "")
                    XLabel = "measured / bound"
                    XFormat = NumFormat.Pow10
                    Bins = b.Bins
                    Zeros = b.Zeros
                    Invalid = b.Invalid
                    Color = Color.Cat 0
                    Marks = [ { At = 0.0; Label = "bound" } ]
                }

    let caseRows =
        [
            for case in cases do
                let xs = cmps |> List.filter (fun c -> c.Case = case)
                let bad = xs |> List.filter (fun c -> not c.Passed)
                let w = xs |> List.filter _.HasRatio |> List.sortBy worstFirst |> List.tryHead

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
                     | Some c -> ratioText c
                     | None -> "")
                    (if logits = "" then "n/a" else logits)
                    (if greedy = "" then "n/a" else greedy)
                ]
        ]

    // Every failing comparison, whatever its value (NaN and infinity first); without failures, those closest to
    // their bound.
    let shown, closestTitle =
        if failed.IsEmpty then
            cmps |> List.filter _.HasRatio |> List.sortBy worstFirst |> List.truncate 15, "Closest to the bound"
        else
            let rows = failed |> List.sortBy worstFirst

            rows |> List.truncate tableRows,
            (if rows.Length > tableRows then
                 Invariant.format $"Failing comparisons (the worst {tableRows} of {Fmt.int (int64 rows.Length)})"
             else
                 "Failing comparisons")

    let worstRows =
        shown
        |> List.map (fun c ->
            [
                c.Pair
                c.Case
                c.Tap
                c.Metric
                Fmt.sci 2 c.Measured
                Fmt.sci 2 c.Bound
                ratioText c
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
            |> Scope.value "worst" (worstRatioText xs)
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
         | Some c when c.IsInvalid ->
             Invariant.format
                 $"{c.Pair} {c.Case} {c.Tap} {c.Metric}: measured {Fmt.sci 2 c.Measured} against a bound of {Fmt.sci 2 c.Bound}"
         | Some c -> Invariant.format $"{c.Pair} {c.Case} {c.Tap} {c.Metric}: {Fmt.dp 3 c.Ratio} of the bound"
         | None -> "n/a")
    |> Scope.value "closest.title" closestTitle
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
