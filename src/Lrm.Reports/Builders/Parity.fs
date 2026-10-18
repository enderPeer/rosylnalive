/// The parity report: agreement of one implementation with the goldens, tap by tap (ReportTemplates contract,
/// "parity"). Reads golden.compare checks, the oracle self-tests (goldens.selftest checks, and golden.compare checks that
/// name a `check` and no `tap`) and the greedy margins of golden tools (goldens.greedy.* records).
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
        /// An oracle self-test (ReferenceOracles, "Telemetry"): a `check` and no `tap`. It counts in the verdict, the
        /// histogram and the closest table, and has its own table instead of a pair and a case.
        SelfTest: bool
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
        let tap = e.TryStr "tap"
        let selfTest = tap.IsNone && (e.TryStr "check").IsSome
        // Without a tap, the compared thing is a tensor (a gradient e_ref) or a self-test's check.
        let tapText =
            tap
            |> Option.orElse (e.TryStr "tensor")
            |> Option.orElse (e.TryStr "check")
            |> Option.defaultValue ""

        // Without a case or prompt: "(self-test)", or the kind in brackets, such as "(gradient)".
        let case =
            compareCase e
            |> Option.defaultWith (fun () ->
                if selfTest then
                    "(self-test)"
                else
                    match e.TryStr "kind" with
                    | Some k when k <> "measurement" -> "(" + k + ")"
                    | _ -> "(no case)")

        {
            Pair = defaultArg (e.TryStr "pair") "(no pair)"
            Case = case
            Tap = tapText
            Metric = metricName (defaultArg (e.TryStr "metric") "")
            Layer = (if tap.IsSome then layerOf e tapText else None)
            Measured = defaultArg (e.TryFloat "measured") nan
            Bound = defaultArg (e.TryFloat "bound") nan
            HasBound = (e.TryFloat "bound").IsSome
            Passed = e.Value = 1.0
            SelfTest = selfTest
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

/// A pair's measured rel_l2 by tap kind and layer, for a pair whose per-layer checks have no bound (e_ref
/// measurements): the largest value of each cell, on a log scale.
let private valueHeatmapOf (pair: string) (measured: Cmp list) =
    let kinds = measured |> List.map (fun c -> tapKind c.Tap) |> List.distinct

    let layers =
        measured |> List.map (fun c -> c.Layer.Value) |> List.distinct |> List.sort

    let cell kind layer =
        let xs =
            measured
            |> List.filter (fun c -> tapKind c.Tap = kind && c.Layer = Some layer)
            |> List.map _.Measured

        if xs.IsEmpty then
            None
        elif xs |> List.exists (fun x -> not (Double.IsFinite x)) then
            Some nan
        else
            Some(List.max xs)

    Chart.Heatmap
        {
            Title = "Measured rel_l2 by tap and layer, " + pair
            Subtitle = "Largest value over all cases; these checks are measurements with no bound (such as e_ref)"
            RowLabel = "tap"
            ColLabel = "layer"
            Rows = kinds
            Cols = layers |> List.map Invariant.i32
            ColValues = []
            Values = [| for k in kinds -> [| for l in layers -> cell k l |] |]
            Log = true
            ValueFormat = NumFormat.Sci
            Ceiling = None
            Aggregate = HeatAggregate.Max
        }

let private heatmapOf (pair: string) (cmps: Cmp list) =
    let perLayer =
        cmps
        |> List.filter (fun c -> c.Pair = pair && c.Metric = relL2 && c.Layer.IsSome)

    let layered = perLayer |> List.filter _.HasRatio

    if layered.IsEmpty && not perLayer.IsEmpty then
        valueHeatmapOf pair perLayer
    elif layered.IsEmpty then
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

/// The oracle self-tests: golden.compare self-tests with their values, and goldens.selftest checks (pass or fail).
let private selfTestRows (log: RunLog) (cmps: Cmp list) =
    [
        for c in cmps |> List.filter _.SelfTest ->
            [
                c.Tap
                c.Metric
                Fmt.sci 2 c.Measured
                Fmt.sci 2 c.Bound
                ratioText c
                mark c.Passed
            ],
            c.Passed
        for e in
            log.Named "goldens.selftest"
            |> List.filter (fun e -> e.Kind = TelemetryKind.Check) ->
            [ defaultArg (e.TryStr "check") "?"; "exact"; ""; ""; ""; mark (e.Value = 1.0) ], e.Value = 1.0
    ]

/// The greedy margins of golden tools (ReferenceOracles, "Greedy decoding and EOS"): per prompt and oracle, the
/// smallest top-2 margin over the greedy steps, the same in float32 ulps of the top logit, and whether the float32
/// argmax agreed at every step.
let private greedyScope (log: RunLog) =
    let gauges (name: string) =
        log.Named name |> List.filter (fun e -> e.Kind = TelemetryKind.Gauge)

    let margins = gauges "goldens.greedy.min_margin"

    if margins.IsEmpty then
        []
    else
        let key (e: RunEvent) =
            defaultArg (e.TryStr "prompt") "?", defaultArg (e.TryStr "oracle") "?"

        let prompts = margins |> List.map (key >> fst) |> List.distinct |> List.sort

        let oracles =
            margins
            |> List.map (key >> snd)
            |> List.distinct
            |> sortIn [ "torch32"; "torch64p" ]

        let lookup (xs: RunEvent list) (k: string * string) =
            xs |> List.tryFind (fun e -> key e = k) |> Option.map _.Value

        let ulps = gauges "goldens.greedy.min_margin_f32_ulps"

        let agrees =
            log.Named "goldens.greedy.f32_argmax_agrees"
            |> List.filter (fun e -> e.Kind = TelemetryKind.Check)

        let rows =
            [
                for p in prompts do
                    for o in oracles do
                        match lookup margins (p, o) with
                        | Some m ->
                            [
                                p
                                o
                                Fmt.num m
                                (lookup ulps (p, o)
                                 |> Option.map (fun u -> Fmt.int (int64 (Math.Round u)))
                                 |> Option.defaultValue "n/a")
                                (match lookup agrees (p, o) with
                                 | Some v -> mark (v = 1.0)
                                 | None -> "n/a")
                            ]
                        | None -> ()
            ]

        let smallest = margins |> List.minBy _.Value
        let disagree = agrees |> List.filter (fun e -> e.Value <> 1.0)

        let summary =
            Invariant.format
                $"The smallest margin is {Fmt.num smallest.Value} ({fst (key smallest)}, {snd (key smallest)}). "
            + (if agrees.IsEmpty then
                   "No float32 argmax agreement was recorded."
               elif disagree.IsEmpty then
                   "The float32 argmax agrees with the oracle at every step of every prompt."
               else
                   let them = plural disagree.Length "prompt and oracle" "prompts and oracles"

                   Invariant.format
                       $"The float32 argmax disagrees at some step for {them} (a generate() mismatch from that step on is expected).")

        let chart =
            Chart.Bar
                {
                    Title = "Smallest top-2 margin over the greedy steps"
                    Subtitle =
                        "Per prompt and oracle; a gate prompt needs at least max(0.05, 4·τ_logit) (ReferenceOracles)"
                    CategoryLabel = "prompt"
                    ValueLabel = "top-1 minus top-2 logit"
                    ValueFormat = NumFormat.Plain
                    Categories = prompts
                    Series =
                        oracles
                        |> List.mapi (fun i o ->
                            {
                                Name = o
                                Color = Color.Cat i
                                Values =
                                    prompts |> List.map (fun p -> lookup margins (p, o) |> Option.defaultValue nan)
                            })
                    Stacked = false
                    Horizontal = false
                }

        [
            Scope.empty
            |> Scope.value "summary" summary
            |> Scope.chart "margins" chart
            |> Scope.table
                "margins"
                (table
                    [
                        "Prompt", L
                        "Oracle", L
                        "Smallest margin", R
                        "In float32 ulps", R
                        "float32 argmax agrees", L
                    ]
                    rows)
        ]

/// Builds the parity scope.
let build (log: RunLog) : Scope =
    let cmps = compares log
    let failed = cmps |> List.filter (fun c -> not c.Passed)
    let pairCmps = cmps |> List.filter (fun c -> not c.SelfTest)
    let pairs = pairCmps |> List.map _.Pair |> sortIn pairOrder
    let cases = pairCmps |> List.map _.Case |> List.distinct |> List.sort
    let selfTests = selfTestRows log cmps
    let selfFailed = selfTests |> List.filter (fun (_, ok) -> not ok)

    let selfText =
        if selfTests.IsEmpty then
            ""
        elif selfFailed.IsEmpty then
            Invariant.format $" All {Fmt.int (int64 selfTests.Length)} oracle self-tests pass."
        else
            Invariant.format
                $" {Fmt.int (int64 selfFailed.Length)} of {Fmt.int (int64 selfTests.Length)} oracle self-tests fail."

    // The verdict counts the comparisons of pairs and the self-tests apart, so no check counts twice.
    let pairFailed = pairCmps |> List.filter (fun c -> not c.Passed)
    let pairInvalid = pairCmps |> List.filter _.IsInvalid

    let verdictText =
        if pairCmps.IsEmpty && selfTests.IsEmpty then
            "No golden comparisons were recorded."
        elif pairCmps.IsEmpty then
            (if selfFailed.IsEmpty then "**PASS.**" else "**FAIL.**") + selfText
        elif pairFailed.IsEmpty then
            (if selfFailed.IsEmpty then "**PASS.**" else "**FAIL.**")
            + Invariant.format $" All {Fmt.int (int64 pairCmps.Length)} comparisons are within their bounds."
            + selfText
        else
            Invariant.format
                $"**FAIL.** {Fmt.int (int64 pairFailed.Length)} of {Fmt.int (int64 pairCmps.Length)} comparisons exceed their bounds"
            + (if pairInvalid.IsEmpty then
                   "."
               else
                   let them = plural pairInvalid.Length "of them is" "of them are"
                   Invariant.format $", {them} NaN or infinite.")
            + selfText

    let worst = cmps |> List.filter _.HasRatio |> List.sortBy worstFirst |> List.tryHead

    let layered = pairCmps |> List.filter (fun c -> c.Metric = relL2 && c.Layer.IsSome)

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
                let xs = pairCmps |> List.filter (fun c -> c.Case = case)
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
            let xs = pairCmps |> List.filter (fun c -> c.Pair = p)
            let bad = xs |> List.filter (fun c -> not c.Passed)

            Scope.empty
            |> Scope.value "name" p
            |> Scope.value "checks" (Fmt.int (int64 xs.Length))
            |> Scope.value "failed" (Fmt.int (int64 bad.Length))
            |> Scope.value "verdict" (verdict bad.IsEmpty)
            |> Scope.value "worst" (worstRatioText xs)
            |> Scope.chart "heatmap" (heatmapOf p pairCmps))

    runScope log
    |> Scope.value "verdict" verdictText
    |> Scope.value "checks" (Fmt.int (int64 pairCmps.Length))
    |> Scope.value "failed" (Fmt.int (int64 pairFailed.Length))
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
    |> Scope.list
        "selftests"
        (if selfTests.IsEmpty then
             []
         else
             [
                 Scope.empty
                 |> Scope.value
                     "summary"
                     (Invariant.format
                         $"{Fmt.int (int64 (selfTests.Length - selfFailed.Length))} of {Fmt.int (int64 selfTests.Length)} pass.")
                 |> Scope.table
                     "selftests"
                     (table
                         [
                             "Self-test", L
                             "Metric", L
                             "Measured", R
                             "Bound", R
                             "Ratio", R
                             "Result", L
                         ]
                         (selfTests |> List.map fst))
             ])
    |> Scope.list "greedy" (greedyScope log)
