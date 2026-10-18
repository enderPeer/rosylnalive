/// Helpers shared by the report builders: statistics, fixed colour orders, the run header and table shorthands.
/// Builders are pure functions of a RunLog; nothing here reads the clock, the culture or the environment.
module Lrm.Reports.Builders.Common

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports

/// Finite values only, sorted ascending.
let sortedFinite (xs: float seq) =
    xs |> Seq.filter Double.IsFinite |> Seq.sort |> Array.ofSeq

/// The q-quantile (0..1) by linear interpolation between order statistics; NaN for no values.
let quantile (q: float) (xs: float seq) =
    let s = sortedFinite xs

    if s.Length = 0 then
        nan
    else
        let p = q * float (s.Length - 1)
        let i = int (Math.Floor p)
        let j = min (s.Length - 1) (i + 1)
        s[i] + (s[j] - s[i]) * (p - float i)

let median (xs: float seq) = quantile 0.5 xs

/// The mean of the finite values; NaN for none.
let mean (xs: float seq) =
    let s = sortedFinite xs
    if s.Length = 0 then nan else Array.sum s / float s.Length

let maxOr (fallback: float) (xs: float seq) =
    let s = sortedFinite xs
    if s.Length = 0 then fallback else s[s.Length - 1]

let minOr (fallback: float) (xs: float seq) =
    let s = sortedFinite xs
    if s.Length = 0 then fallback else s[0]

/// The position of `x` in a fixed order, or the order's length (then ordinal order) for unknown entries; used to keep
/// colours and rows stable across reports.
let orderIn (order: string list) (x: string) =
    match List.tryFindIndex (Invariant.equals x) order with
    | Some i -> i, ""
    | None -> order.Length, x

/// Sorts names by a fixed order, unknown names last in ordinal order.
let sortIn (order: string list) (xs: string seq) =
    xs |> Seq.distinct |> Seq.sortBy (orderIn order) |> List.ofSeq

/// Backend names in ExecutionClass order; their colour slots follow this order.
let backendOrder = ExecutionBackend.all |> List.map ExecutionBackend.name

/// ToleranceSpec pairs in contract order; their colour slots follow this order.
let pairOrder =
    [
        "cpu64-vs-torch64p"
        "cpu32-vs-torch32"
        "cpu32-vs-torch64p"
        "gpu-vs-cpu32"
        "gpu-vs-torch32"
        "gpu-vs-torch64p"
        "vulkan-vs-d3d12"
    ]

/// A colour that follows an entity's position in `order`, not its rank in one chart.
let colorIn (order: string list) (x: string) =
    match List.tryFindIndex (Invariant.equals x) order with
    | Some i -> Color.slot i
    | None -> Color.Neutral

/// "PASS" or "FAIL": status text that never relies on colour.
let verdict (passed: bool) = if passed then "PASS" else "FAIL"

/// "✓ pass" or "✗ FAIL" for table cells.
let mark (passed: bool) = if passed then "✓ pass" else "✗ FAIL"

/// A table from headers (with alignment) and rows.
let table (columns: (string * Align) list) (rows: string list list) = { Columns = columns; Rows = rows }

let L = Align.Left
let R = Align.Right

/// A golden.compare metric name in the templates' snake_case: "relL2" -> "rel_l2", "maxAbs" -> "max_abs",
/// "top2Margin" -> "top2_margin"; snake_case names stay as they are. ComparatorsAndGoldenReader (O2's Compare) and
/// O1's tools write camelCase names, and the templates read either form.
let metricName (metric: string) =
    if isNull metric then
        ""
    else
        let sb = Text.StringBuilder(metric.Length + 4)

        for i in 0 .. metric.Length - 1 do
            let c = metric[i]

            if Char.IsAsciiLetterUpper c then
                if i > 0 && metric[i - 1] <> '_' then
                    sb.Append '_' |> ignore

                sb.Append(Char.ToLowerInvariant c) |> ignore
            else
                sb.Append c |> ignore

        sb.ToString()

/// The case of a golden.compare check: `case`, else `prompt` (the ComparatorsAndGoldenReader caller key, which O2's
/// tiny-model checks write).
let compareCase (e: RunEvent) =
    e.TryStr TelemetryKeys.Case |> Option.orElse (e.TryStr "prompt")

/// The tap kind of an activation name: "L3.q_proj" -> "q_proj"; names without a layer prefix stay as they are.
let tapKind (tap: string) =
    if tap.Length > 1 && tap[0] = 'L' && Char.IsDigit tap[1] then
        match tap.IndexOf '.' with
        | i when i > 0 -> tap.Substring(i + 1)
        | _ -> tap
    else
        tap

/// The layer of an activation name: "L3.q_proj" -> Some 3.
let tapLayer (tap: string) =
    if tap.Length > 1 && tap[0] = 'L' && Char.IsDigit tap[1] then
        match tap.IndexOf '.' with
        | i when i > 1 -> Invariant.tryParseI32 (tap.Substring(1, i - 1))
        | _ -> None
    else
        None

/// The earliest timestamp of the run, the zero of every time axis.
let runStart (log: RunLog) =
    if log.Events.IsEmpty then
        0L
    else
        log.Events |> List.map _.TimeNs |> List.min

/// The backend of a node, from its execution class, or "".
let backendOf (log: RunLog) (node: string) =
    log.ClassOf node
    |> Option.map (fun c -> ExecutionBackend.name c.Backend)
    |> Option.defaultValue ""

/// "Synthetic data" banner text for synthetic runs, "" otherwise.
let banner (log: RunLog) =
    if log.Manifest.Synthetic then
        "> **Synthetic data.** This report was rendered from a run log made by a synthetic generator (`lrm report --synthetic`), so the visuals can be reviewed before the real runs exist. None of its numbers are measurements.\n"
    else
        ""

let private classText (c: ExecutionClass option) =
    match c with
    | Some c -> ExecutionClass.describe c
    | None -> "not recorded"

/// The header every template starts with (ReportTemplates contract, "Common values"): run.* values and the
/// run.summary, run.nodes, run.params and run.problems tables.
let runScope (log: RunLog) : Scope =
    let m = log.Manifest
    let records = log.Events.Length

    let summary =
        table
            [ "Field", L; "Value", L ]
            [
                [ "Run"; "`" + m.RunId + "`" ]
                [ "Title"; m.Title ]
                [ "Kind"; m.Kind ]
                [ "Status"; RunStatus.name m.Status ]
                [ "Started"; m.CreatedUtc ]
                [ "Ended"; defaultArg m.EndedUtc "not finished" ]
                [ "Execution class"; classText m.ExecutionClass ]
                [
                    "Class id"
                    match m.ExecutionClass with
                    | Some c -> "`" + (ExecutionClass.id c).Value.Short + "`"
                    | None -> "n/a"
                ]
                [ "Driver"; defaultArg m.Driver "n/a" ]
                [
                    "Commit"
                    m.GitSha
                    |> Option.map (fun s -> "`" + s + "`")
                    |> Option.defaultValue "not recorded"
                ]
                [ "Nodes"; Invariant.i32 log.Nodes.Length ]
                [ "Records"; Fmt.int (int64 records) ]
                [ "Synthetic"; (if m.Synthetic then "yes" else "no") ]
            ]

    let nodes =
        table
            [
                "Node", L
                "Role", L
                "Execution class", L
                "Records", R
                "Finished", L
                "Dropped", R
            ]
            [
                for n in log.Nodes ->
                    [
                        n.Node
                        (if n.Role = "" then "?" else n.Role)
                        classText (log.ClassOf n.Node)
                        Fmt.int (int64 n.Records)
                        (match n.Status with
                         | Some s -> RunStatus.name s
                         | None -> "no")
                        Fmt.int (n.Dropped + n.Invalid)
                    ]
            ]

    let pars = table [ "Parameter", L; "Value", L ] [ for k, v in m.Params -> [ k; v ] ]

    let problems =
        match log.Problems with
        | [] -> "None."
        | ps -> ps |> List.map (fun p -> "- " + p) |> String.concat "\n"

    Scope.empty
    |> Scope.value "run.id" m.RunId
    |> Scope.value "run.title" m.Title
    |> Scope.value "run.kind" m.Kind
    |> Scope.value "run.status" (RunStatus.name m.Status)
    |> Scope.value "run.created" m.CreatedUtc
    |> Scope.value "run.class" (classText m.ExecutionClass)
    |> Scope.value "run.driver" (defaultArg m.Driver "n/a")
    |> Scope.value "run.banner" (banner log)
    |> Scope.value "run.problems" problems
    |> Scope.table "run.summary" summary
    |> Scope.table "run.nodes" nodes
    |> Scope.table "run.params" pars

/// Samples binned for a Histogram: the bins and the samples that have no place on the axis.
type Binned =
    {
        Bins: HistBin list
        /// Exact zeros, in log10 space (in linear space they are in the bins).
        Zeros: int
        /// NaN and infinite samples, and in log10 space negative ones.
        Invalid: int
    }

/// Equal-width histogram bins. In log10 space the bins hold the positive samples, exact zeros are counted in Zeros
/// and NaN, infinite and negative samples in Invalid; in linear space the bins hold every finite sample and Invalid
/// counts the rest. Nothing is dropped silently: Histogram draws Zeros and Invalid as their own columns.
let bin (count: int) (log10Space: bool) (samples: float seq) : Binned =
    let all = Array.ofSeq samples

    let usable (v: float) =
        Double.IsFinite v && (not log10Space || v > 0.0)

    let xs =
        all
        |> Array.filter usable
        |> Array.map (fun v -> if log10Space then DetMath.log10 v else v)

    let zeros =
        if log10Space then
            all |> Array.filter (fun v -> v = 0.0) |> Array.length
        else
            0

    let invalid = all.Length - xs.Length - zeros

    let bins =
        if xs.Length = 0 then
            []
        else
            let lo = Array.min xs
            let hi = Array.max xs
            let lo, hi = if hi > lo then lo, hi else lo - 0.5, hi + 0.5
            let width = (hi - lo) / float count
            let counts = Array.zeroCreate count

            for x in xs do
                let i = max 0 (min (count - 1) (int (Math.Floor((x - lo) / width))))
                counts[i] <- counts[i] + 1

            [
                for i in 0 .. count - 1 ->
                    {
                        Lo = lo + float i * width
                        Hi = lo + float (i + 1) * width
                        Count = counts[i]
                    }
            ]

    {
        Bins = bins
        Zeros = zeros
        Invalid = invalid
    }

/// One category of an error-bar chart: min, `mid` and max of the finite values, and how many were NaN or infinite.
let spread (index: int) (mid: float list -> float) (xs: float list) : ErrorPoint =
    let finite = xs |> List.filter Double.IsFinite

    {
        Index = index
        Low = minOr nan finite
        Mid = (if finite.IsEmpty then nan else mid finite)
        High = maxOr nan finite
        Invalid = xs.Length - finite.Length
    }

/// A count and its noun: "1 node loss", "2 node losses".
let plural (count: int) (one: string) (many: string) =
    Fmt.int (int64 count) + " " + (if count = 1 then one else many)

/// A placeholder chart that says which records were missing.
let noData (title: string) (what: string) =
    Chart.Empty(title, "this run has no " + what + " records")
