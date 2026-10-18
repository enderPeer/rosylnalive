namespace Lrm.Reports

open System
open Lrm.Core

/// How axis ticks and tooltips print values.
[<RequireQualifiedAccess>]
type NumFormat =
    /// Plain decimals sized to the tick step.
    | Plain
    /// Whole numbers with separators.
    | Int
    /// SI suffixes (K, M, G, T): FLOPs, tokens, counts.
    | Si
    /// Binary byte units.
    | Bytes
    /// A duration given in ns.
    | Duration
    /// A ratio shown as a percentage.
    | Pct
    /// Scientific notation.
    | Sci
    /// The value is a base-10 exponent of a plain number: 2 is 100, -6 is 10⁻⁶.
    | Pow10
    /// The value is a base-10 exponent of a quantity in the inner format: with Duration, 6 is 1 ms. Decade ticks are
    /// labelled in the inner format's units (1 µs, 10 µs, 100 µs, 1 ms).
    | Decades of NumFormat

/// One line of a line chart: points (x, y) in order. A non-finite point breaks the line.
type Series =
    {
        Name: string
        Color: Color
        Points: (float * float) list
    }

/// A horizontal reference line, such as a tolerance bound.
type RefLine = { Value: float; Label: string }

/// A labelled x range drawn behind the data, such as a training stage.
type Band =
    {
        From: float
        To: float
        Label: string
    }

/// Line chart: metrics over steps, or per-layer multi-series with IntegerX (ReportTemplates contract). A series with
/// more points than the plot has pixels is drawn through the first, lowest, highest and last point of every pixel
/// column, so its extremes survive. On a log y axis, exact zeros sit in a strip labelled 0 below the decades.
type LineChart =
    {
        Title: string
        Subtitle: string
        XLabel: string
        YLabel: string
        XFormat: NumFormat
        YFormat: NumFormat
        YLog: bool
        /// Include 0 in a linear y range.
        YFromZero: bool
        /// Integer x ticks (layers, steps) and point markers.
        IntegerX: bool
        Series: Series list
        RefLines: RefLine list
        Bands: Band list
    }

/// One series of a bar chart, one value per category.
type BarSeries =
    {
        Name: string
        Color: Color
        Values: float list
    }

/// Bar chart: grouped or stacked, horizontal bars or vertical columns.
type BarChart =
    {
        Title: string
        Subtitle: string
        CategoryLabel: string
        ValueLabel: string
        ValueFormat: NumFormat
        Categories: string list
        Series: BarSeries list
        Stacked: bool
        Horizontal: bool
    }

/// How a heatmap column bin combines the columns it holds.
[<RequireQualifiedAccess>]
type HeatAggregate =
    /// The largest value: spikes stay visible.
    | Max
    /// The mean of the values.
    | Mean

/// Heatmap: rows x columns on the sequential ramp. More than ChartSvg.MaxHeatColumns columns are binned.
type Heatmap =
    {
        Title: string
        Subtitle: string
        RowLabel: string
        ColLabel: string
        Rows: string list
        Cols: string list
        /// The numeric position of every column (such as its step), increasing, or [] for categorical columns. With
        /// positions the column axis gets nice ticks instead of column labels.
        ColValues: float list
        /// Row-major cells, Values[row][col]. None is missing data. Some NaN or an infinity is an invalid value, drawn
        /// in the critical colour with a cross. On a log scale Some 0 is an exact zero, drawn as an outlined cell.
        Values: float option[][]
        Log: bool
        ValueFormat: NumFormat
        /// A fixed top of the colour scale and the name of the cells above it, such as (1.0, "over bound") for
        /// ratios to a tolerance bound: the ramp then ends at the ceiling and cells above it are critical.
        Ceiling: (float * string) option
        /// How binned columns are combined.
        Aggregate: HeatAggregate
    }

/// One histogram bin [Lo, Hi).
type HistBin = { Lo: float; Hi: float; Count: int }

/// A vertical reference line on a histogram.
type VLine = { At: float; Label: string }

/// Histogram of pre-binned counts, with optional side columns for samples that have no place on the axis.
type Histogram =
    {
        Title: string
        Subtitle: string
        XLabel: string
        /// The format of bin edges (Pow10 or Decades when the bins are in log10 space).
        XFormat: NumFormat
        Bins: HistBin list
        /// Samples that are exactly 0 when the bins are in log10 space: a column labelled 0 left of the axis.
        Zeros: int
        /// Samples that are NaN, infinite or (in log10 space) negative: a critical column labelled NaN/inf at the right.
        Invalid: int
        Color: Color
        Marks: VLine list
    }

/// One bar on a timeline lane.
type Interval =
    {
        Lane: int
        Start: float
        Length: float
        /// A status colour other than Good (Warning, Serious, Critical) marks an outcome to notice: it is drawn at least
        /// 4 px wide, outlined and on top of the markers.
        Color: Color
        /// Hover text.
        Tip: string
    }

/// A vertical event marker on a timeline, such as a node kill.
type TimeMark =
    {
        At: float
        Label: string
        Color: Color
    }

/// Timeline of intervals on lanes: the stacked kernel timeline, the cluster stage Gantt chart, a Roslyn call trace.
type Timeline =
    {
        Title: string
        Subtitle: string
        XLabel: string
        XFormat: NumFormat
        Lanes: string list
        Intervals: Interval list
        Marks: TimeMark list
        Legend: (string * Color) list
    }

/// One category's spread: whiskers from Low to High around Mid, over the finite values; Invalid counts the values
/// that were NaN or infinite. A point with only invalid values is drawn as a cross in the NaN/inf strip.
type ErrorPoint =
    {
        Index: int
        Low: float
        Mid: float
        High: float
        Invalid: int
    }

type ErrorSeries =
    {
        Name: string
        Color: Color
        Spreads: ErrorPoint list
    }

/// Per-layer error bars: for each category, each series' Mid with Low-High whiskers. On a log y axis, exact zeros sit
/// in a strip labelled 0 below the decades; invalid values get a cross in a strip labelled NaN/inf above them.
type ErrorBarChart =
    {
        Title: string
        Subtitle: string
        XLabel: string
        YLabel: string
        YFormat: NumFormat
        YLog: bool
        Categories: string list
        Series: ErrorSeries list
        RefLines: RefLine list
    }

/// Every chart kind (ReportTemplates contract).
[<RequireQualifiedAccess>]
type Chart =
    | Line of LineChart
    | Bar of BarChart
    | Heatmap of Heatmap
    | Histogram of Histogram
    | Timeline of Timeline
    | ErrorBars of ErrorBarChart
    /// A placeholder when a run has no data for a chart.
    | Empty of title: string * message: string

/// An axis: Lo and Hi are values (linear) or base-10 exponents (log).
type private Axis =
    {
        Lo: float
        Hi: float
        Log: bool
        Ticks: (float * string) list
    }

type private LegendKind =
    | LineKey
    | BoxKey
    | DotKey

/// The vertical layout of a plot whose y axis may have strips beside the value range: one at the bottom for exact
/// zeros on a log axis, one at the top for NaN or infinite values.
type private YLayout =
    {
        Axis: Axis
        /// The value range: from RangeTop down RangeH pixels.
        RangeTop: float
        RangeH: float
        /// The y of the zero strip's row, when there is one.
        ZeroY: float option
        /// The y of the NaN/inf strip's row, when there is one.
        InvalidY: float option
    }

/// What a heatmap cell shows.
type private CellKind =
    | Scaled of float
    | Exact
    | Over of float
    | Bad
    | Missing

/// Deterministic SVG rendering of every chart kind (ReportTemplates contract, "Charts").
[<RequireQualifiedAccess>]
module ChartSvg =
    let private n = SvgWriter.Num

    [<Literal>]
    let Width = 720.0

    /// Heatmaps with more columns than this bin them (HeatAggregate), so a long run stays a readable, small file.
    [<Literal>]
    let MaxHeatColumns = 120

    /// The height of the zero strip and the NaN/inf strip.
    [<Literal>]
    let private StripH = 20.0

    /// Magnitudes are clamped to this before layout, so no span or position overflows.
    [<Literal>]
    let private Big = 1e300

    /// Values beyond ±Big are drawn at ±Big.
    let private clampV (v: float) =
        if v > Big then Big
        elif v < -Big then -Big
        else v

    /// Positive values on a log axis are drawn within [1e-300, 1e300].
    let private clampPos (v: float) =
        if v < 1e-300 then 1e-300
        elif v > Big then Big
        else v

    let private superDigits = "⁰¹²³⁴⁵⁶⁷⁸⁹"

    /// 10^k as a power of ten with a superscript exponent: 10⁻⁶, 10⁰, 10³.
    let private powerText (k: int) =
        let digits =
            Invariant.i32 (abs k)
            |> Seq.map (fun c -> superDigits[int c - int '0'])
            |> Array.ofSeq

        "10" + (if k < 0 then "⁻" else "") + String(digits)

    /// 10^k as a plain decimal: 0.01, 1, 1,000.
    let private plainPower (k: int) =
        if k >= 0 then
            Fmt.int (int64 (DetMath.pow10 k))
        else
            "0." + String('0', -k - 1) + "1"

    /// m times 10^k as a plain decimal, for m in 1, 2, 5: 0.02, 5, 2,000.
    let private plainMultiple (m: int) (k: int) =
        if k >= 0 then
            Fmt.int (int64 m * int64 (DetMath.pow10 k))
        else
            "0." + String('0', -k - 1) + Invariant.i32 m

    /// m times 10^k in its own unit of `f`, for m in 1, 2, 5: 100 ns, 2 µs, 50 ms; None for formats without units.
    let private multipleInUnit (f: NumFormat) (m: int) (k: int) =
        let split (units: (int * string) list) =
            let u, suffix =
                units
                |> List.filter (fun (e, _) -> e <= k)
                |> List.tryLast
                |> Option.defaultValue units.Head

            Some(plainMultiple m (k - u) + suffix)

        match f with
        | NumFormat.Duration -> split [ 0, " ns"; 3, " µs"; 6, " ms"; 9, " s" ]
        | NumFormat.Si -> split [ 0, ""; 3, "K"; 6, "M"; 9, "G"; 12, "T" ]
        | _ -> None

    /// The decade 10^k in its own unit of `f`: 100 ns, 1 µs, 10 ms; None for formats without decade units.
    let private decadeInUnit (f: NumFormat) (k: int) =
        let split (units: (int * string) list) =
            let u, suffix =
                units
                |> List.filter (fun (e, _) -> e <= k)
                |> List.tryLast
                |> Option.defaultValue units.Head

            Some(plainPower (k - u) + suffix)

        match f with
        | NumFormat.Duration -> split [ 0, " ns"; 3, " µs"; 6, " ms"; 9, " s" ]
        | NumFormat.Si -> split [ 0, ""; 3, "K"; 6, "M"; 9, "G"; 12, "T" ]
        | _ -> None

    /// Labels for the decade ticks of one axis, in one style: plain decimals when every tick lies within 10^-2 to
    /// 10^3, powers of ten otherwise; in the units of `f` for a unit format.
    let private decadeLabels (f: NumFormat) (ks: int list) =
        let plain = ks |> List.forall (fun k -> k >= -2 && k <= 3)

        ks
        |> List.map (fun k ->
            match decadeInUnit f k with
            | Some text -> text
            | None -> if plain then plainPower k else powerText k)

    /// A value in a format, for tooltips and labels.
    let rec format (f: NumFormat) (x: float) =
        match f with
        | NumFormat.Plain -> Fmt.num x
        | NumFormat.Int ->
            if Double.IsFinite x then
                Fmt.int (int64 (Math.Round(clampV x)))
            else
                Fmt.num x
        | NumFormat.Si -> Fmt.si x
        | NumFormat.Bytes -> Fmt.bytes x
        | NumFormat.Duration -> Fmt.duration x
        | NumFormat.Pct -> Fmt.pct x
        | NumFormat.Sci -> Fmt.sci 1 x
        | NumFormat.Pow10 ->
            if not (Double.IsFinite x) then
                Fmt.num x
            elif x = Math.Round x && abs x <= 300.0 then
                (decadeLabels f [ int x ]).Head
            else
                Fmt.sci 2 (DetMath.exp10 (max -300.0 (min 300.0 x)))
        | NumFormat.Decades inner ->
            if Double.IsFinite x then
                format inner (DetMath.exp10 (max -300.0 (min 300.0 x)))
            else
                Fmt.num x

    let private isDecades (f: NumFormat) =
        match f with
        | NumFormat.Pow10
        | NumFormat.Decades _ -> true
        | _ -> false

    // ---- axes -------------------------------------------------------------------------------------------------------

    let private niceStep (raw: float) =
        let k = DetMath.floorLog10 raw
        let p = DetMath.pow10 k
        let f = raw / p

        let nf =
            if f <= 1.0 then 1.0
            elif f <= 2.0 then 2.0
            elif f <= 2.5 then 2.5
            elif f <= 5.0 then 5.0
            else 10.0

        nf * p

    let private stepDecimals (step: float) =
        let k = DetMath.floorLog10 step
        let mantissa = step / DetMath.pow10 k
        min 6 (max 0 (-k + (if mantissa = 2.5 then 1 else 0)))

    let private tickText (f: NumFormat) (step: float) (lo: float) (hi: float) (v: float) =
        match f with
        | NumFormat.Plain ->
            let big = max (abs lo) (abs hi)

            if big >= 1e15 then Fmt.sci 1 v
            elif big >= 1e5 then Fmt.si v
            elif step < 1e-4 then (if v = 0.0 then "0" else Fmt.sci 1 v)
            else Fmt.dp (stepDecimals step) v
        | NumFormat.Int -> Fmt.int (int64 (Math.Round v))
        | f -> format f v

    /// The display unit of an axis in a unit format: the scale it divides by and the suffix it prints.
    let private unitFor (f: NumFormat) (big: float) =
        match f with
        | NumFormat.Duration ->
            if big >= 1e9 then 1e9, " s"
            elif big >= 1e6 then 1e6, " ms"
            elif big >= 1e3 then 1e3, " µs"
            else 1.0, " ns"
        | NumFormat.Bytes ->
            if big >= 1099511627776.0 then 1099511627776.0, " TiB"
            elif big >= 1073741824.0 then 1073741824.0, " GiB"
            elif big >= 1048576.0 then 1048576.0, " MiB"
            elif big >= 1024.0 then 1024.0, " KiB"
            else 1.0, " B"
        | NumFormat.Si ->
            if big >= 1e12 then 1e12, "T"
            elif big >= 1e9 then 1e9, "G"
            elif big >= 1e6 then 1e6, "M"
            elif big >= 1e3 then 1e3, "K"
            else 1.0, ""
        | NumFormat.Pct -> 0.01, "%"
        | _ -> 1.0, ""

    let private isNormalPositive (x: float) =
        Double.IsFinite x && x >= 2.2250738585072014e-308

    /// A linear axis with about `target` ticks at 1, 2, 2.5 or 5 times a power of ten of the display unit. A tight
    /// axis spans exactly [lo, hi] and keeps only the ticks inside it; otherwise the axis extends to whole ticks.
    /// Decade formats tick whole exponents, labelled in one style for the axis. A span too small to divide (subnormal
    /// values) falls back to a unit span; magnitudes beyond 1e300 are clamped.
    let rec private linearAxis (f: NumFormat) (lo: float) (hi: float) (target: int) (integer: bool) (tight: bool) =
        let lo = clampV lo
        let hi = clampV hi

        let lo, hi =
            if hi > lo then lo, hi
            elif lo = 0.0 then 0.0, 1.0
            else lo - abs lo * 0.1, hi + abs hi * 0.1

        let scale, suffix = unitFor f (max (abs lo) (abs hi))
        let slo = lo / scale
        let shi = hi / scale
        let raw = (shi - slo) / float target

        let fallback () =
            let lo', hi' =
                if lo >= 0.0 then lo, lo + 1.0
                elif hi <= 0.0 then hi - 1.0, hi
                else lo * 0.5 + hi * 0.5 - 0.5, lo * 0.5 + hi * 0.5 + 0.5

            linearAxis f lo' hi' target integer tight

        if not (isNormalPositive raw) then
            fallback ()
        else
            let step =
                if isDecades f then
                    // Every decade while at most eight fit, then every second, third, ...
                    float (max 1 (int (Math.Ceiling((shi - slo) / 8.0 - 1e-9))))
                elif integer then
                    max 1.0 (Math.Round(niceStep raw))
                else
                    niceStep raw

            let first = Math.Floor(slo / step) * step
            let last = Math.Ceiling(shi / step) * step
            let span = (last - first) / step

            if not (Double.IsFinite span) || span > 1000.0 then
                fallback ()
            else
                let count = int (Math.Round span)
                let eps = step * 1e-9

                let values =
                    [ for i in 0..count -> first + float i * step ]
                    |> List.filter (fun v -> not tight || (v >= slo - eps && v <= shi + eps))

                // A decade axis over at most three decades also ticks 2 and 5 times each decade, when those labels
                // read plainly (a unit format, or plain decimals).
                let inner =
                    match f with
                    | NumFormat.Decades inner -> inner
                    | _ -> NumFormat.Plain

                let decadeTicks =
                    isDecades f
                    && shi - slo <= 3.0
                    && (values |> List.forall (fun v -> v = Math.Round v && abs v <= 300.0))

                let minor =
                    if not decadeTicks then
                        []
                    else
                        [
                            for k in int (Math.Floor slo) .. int (Math.Ceiling shi) do
                                for m in [ 2; 5 ] do
                                    let v = float k + DetMath.log10 (float m)

                                    if v >= slo - eps && v <= shi + eps then
                                        match multipleInUnit inner m k with
                                        | Some text -> yield v, text
                                        | None when k >= -2 && k <= 3 -> yield v, plainMultiple m k
                                        | None -> ()
                        ]

                let labels =
                    match f with
                    | NumFormat.Duration
                    | NumFormat.Bytes
                    | NumFormat.Si
                    | NumFormat.Pct ->
                        values
                        |> List.map (fun v ->
                            if v = 0.0 then
                                "0"
                            else
                                Fmt.dp (stepDecimals step) v + suffix)
                    | f when
                        isDecades f
                        && values |> List.forall (fun v -> v = Math.Round v && abs v <= 300.0)
                        ->
                        decadeLabels inner (values |> List.map int)
                    | f -> values |> List.map (tickText f step first last)

                let major = List.zip (values |> List.map (fun v -> v * scale)) labels

                // Minor decade ticks join only when every major label is plain too (one style per axis).
                let plainMajors =
                    decadeTicks
                    && (match inner with
                        | NumFormat.Duration
                        | NumFormat.Si -> true
                        | _ -> values |> List.forall (fun v -> v >= -2.0 && v <= 3.0))

                {
                    Lo = (if tight then lo else first * scale)
                    Hi = (if tight then hi else last * scale)
                    Log = false
                    Ticks =
                        (if plainMajors then
                             major @ minor |> List.sortBy fst
                         else
                             major)
                }

    /// A log axis over positive values, with decade ticks labelled in one style (and in the units of `f` for a unit
    /// format such as Duration).
    let private logAxis (f: NumFormat) (lo: float) (hi: float) =
        let lo = clampPos lo
        let hi = clampPos hi
        let lo, hi = if hi > lo then lo, hi else lo / 10.0, hi * 10.0
        let kLo = DetMath.floorLog10 lo

        let kHi =
            let k = DetMath.floorLog10 hi
            if DetMath.pow10 k < hi then k + 1 else k

        let kHi = if kHi = kLo then kHi + 1 else kHi
        let every = max 1 ((kHi - kLo + 7) / 8)
        let ks = [ kLo..kHi ] |> List.filter (fun k -> (k - kLo) % every = 0)

        let inner =
            match f with
            | NumFormat.Duration
            | NumFormat.Si -> f
            | _ -> NumFormat.Plain

        {
            Lo = float kLo
            Hi = float kHi
            Log = true
            Ticks = List.zip (ks |> List.map float) (decadeLabels inner ks)
        }

    /// The position of a value on an axis, 0 at Lo and 1 at Hi.
    let private pos (a: Axis) (v: float) =
        if a.Log then
            (DetMath.log10 (clampPos v) - a.Lo) / (a.Hi - a.Lo)
        else
            (clampV v - a.Lo) / (a.Hi - a.Lo)

    /// The position of a tick (ticks of a log axis are exponents).
    let private tickPos (a: Axis) (t: float) = (t - a.Lo) / (a.Hi - a.Lo)

    /// The layout of a plot from `top` down `plotH` pixels, with the strips it needs.
    let private yLayout (a: Axis) (top: float) (plotH: float) (zero: bool) (invalid: bool) =
        let invalidH = if invalid then StripH else 0.0
        let zeroH = if zero then StripH else 0.0

        {
            Axis = a
            RangeTop = top + invalidH
            RangeH = plotH - invalidH - zeroH
            ZeroY = (if zero then Some(top + plotH - zeroH / 2.0 + 2.0) else None)
            InvalidY = (if invalid then Some(top + invalidH / 2.0) else None)
        }

    /// The y of a value: exact zeros on a log axis go to the zero strip.
    let private yOf (l: YLayout) (v: float) =
        match l.ZeroY with
        | Some y when l.Axis.Log && v = 0.0 -> y
        | _ -> l.RangeTop + l.RangeH * (1.0 - pos l.Axis v)

    // ---- frame ------------------------------------------------------------------------------------------------------

    let private headerHeight (subtitle: string) =
        if String.IsNullOrEmpty subtitle then 38.0 else 56.0

    let private openSvg (w: SvgWriter) (id: string) (height: float) (title: string) (subtitle: string) (desc: string) =
        w.Open(
            "svg",
            [
                "xmlns", "http://www.w3.org/2000/svg"
                "width", n Width
                "height", n height
                "viewBox", "0 0 " + n Width + " " + n height
                "role", "img"
                "aria-labelledby", id + "-title " + id + "-desc"
            ]
        )

        w.Text("title", [ "id", id + "-title" ], title)
        w.Text("desc", [ "id", id + "-desc" ], desc)
        w.Raw("<style>" + Theme.style + "</style>\n")

        w.Leaf(
            "rect",
            [
                "class", "bg"
                "x", "0"
                "y", "0"
                "width", n Width
                "height", n height
                "rx", "8"
            ]
        )

        w.Leaf(
            "rect",
            [
                "class", "frame"
                "x", "0.5"
                "y", "0.5"
                "width", n (Width - 1.0)
                "height", n (height - 1.0)
                "rx", "8"
            ]
        )

        w.Text(
            "text",
            [ "class", "t1"; "x", "16"; "y", "26"; "font-size", "15"; "font-weight", "600" ],
            TextMetrics.fit 15.0 (Width - 32.0) title
        )

        if not (String.IsNullOrEmpty subtitle) then
            w.Text(
                "text",
                [ "class", "t2"; "x", "16"; "y", "45"; "font-size", "12" ],
                TextMetrics.fit 12.0 (Width - 32.0) subtitle
            )

    let private closeSvg (w: SvgWriter) =
        w.Close "svg"
        w.ToString()

    /// The rows of a legend that wraps within the chart width: each row is a list of (x, label, color).
    let private legendRows (entries: (string * Color) list) =
        let maxX = Width - 16.0
        let mutable rows = []
        let mutable row = []
        let mutable x = 16.0

        for label, color in entries do
            let w = 16.0 + 6.0 + TextMetrics.width 12.0 label

            if x + w > maxX && not row.IsEmpty then
                rows <- List.rev row :: rows
                row <- []
                x <- 16.0

            row <- (x, label, color) :: row
            x <- x + w + 18.0

        if not row.IsEmpty then
            rows <- List.rev row :: rows

        List.rev rows

    let private drawLegend (w: SvgWriter) (kind: LegendKind) (y: float) (entries: (string * Color) list) =
        legendRows entries
        |> List.iteri (fun r row ->
            let cy = y + float r * 20.0

            for x, label, color in row do
                match kind with
                | LineKey ->
                    w.Leaf(
                        "rect",
                        [
                            "class", Color.fill color
                            "x", n x
                            "y", n (cy - 1.0)
                            "width", "16"
                            "height", "2.5"
                            "rx", "1"
                        ]
                    )
                | BoxKey ->
                    w.Leaf(
                        "rect",
                        [
                            "class", Color.fill color
                            "x", n (x + 3.0)
                            "y", n (cy - 5.0)
                            "width", "10"
                            "height", "10"
                            "rx", "2"
                        ]
                    )
                | DotKey -> w.Leaf("circle", [ "class", Color.fill color; "cx", n (x + 8.0); "cy", n cy; "r", "4" ])

                w.Text("text", [ "class", "t2"; "x", n (x + 22.0); "y", n (cy + 4.0); "font-size", "12" ], label))

    let private legendHeight (entries: (string * Color) list) =
        if entries.IsEmpty then
            0.0
        else
            float (legendRows entries).Length * 20.0

    let private gridLine (w: SvgWriter) (x1: float) (y1: float) (x2: float) (y2: float) (dashed: bool) =
        w.Leaf(
            "line",
            [
                "class", "grid"
                "x1", n x1
                "y1", n y1
                "x2", n x2
                "y2", n y2
                "stroke-width", "1"
                if dashed then
                    "stroke-dasharray", "3 3"
            ]
        )

    let private tickLabel (w: SvgWriter) (x: float) (y: float) (anchor: string) (text: string) =
        w.Text("text", [ "class", "t3"; "x", n x; "y", n y; "font-size", "11"; "text-anchor", anchor ], text)

    /// The y grid and tick labels, and the labels and separators of the zero and NaN/inf strips.
    let private yGrid (w: SvgWriter) (l: YLayout) (left: float) (right: float) =
        for t, label in l.Axis.Ticks do
            let y = l.RangeTop + l.RangeH * (1.0 - tickPos l.Axis t)
            gridLine w left y right y false
            tickLabel w (left - 8.0) (y + 4.0) "end" label

        match l.ZeroY with
        | Some y ->
            gridLine w left (l.RangeTop + l.RangeH + 4.0) right (l.RangeTop + l.RangeH + 4.0) true
            tickLabel w (left - 8.0) (y + 4.0) "end" "0"
        | None -> ()

        match l.InvalidY with
        | Some y ->
            gridLine w left (l.RangeTop - 4.0) right (l.RangeTop - 4.0) true
            tickLabel w (left - 8.0) (y + 4.0) "end" "NaN/inf"
        | None -> ()

    let private xTicks (w: SvgWriter) (a: Axis) (left: float) (plotW: float) (y: float) =
        let mutable lastEnd = Double.NegativeInfinity

        for t, label in a.Ticks do
            let x = left + plotW * tickPos a t
            let half = TextMetrics.width 11.0 label / 2.0

            if x - half > lastEnd + 6.0 then
                tickLabel w x y "middle" label
                lastEnd <- x + half

    let private axisLabel (w: SvgWriter) (x: float) (y: float) (anchor: string) (text: string) =
        if not (String.IsNullOrEmpty text) then
            w.Text("text", [ "class", "t3"; "x", n x; "y", n y; "font-size", "11"; "text-anchor", anchor ], text)

    let private baseline (w: SvgWriter) (x1: float) (x2: float) (y: float) =
        w.Leaf(
            "line",
            [
                "class", "axis"
                "x1", n x1
                "y1", n y
                "x2", n x2
                "y2", n y
                "stroke-width", "1"
            ]
        )

    let private refLines (w: SvgWriter) (l: YLayout) (refs: RefLine list) (left: float) (right: float) =
        for r in refs do
            if Double.IsFinite r.Value && (not l.Axis.Log || r.Value > 0.0) then
                let p = pos l.Axis r.Value

                if p >= -0.001 && p <= 1.001 then
                    let y = l.RangeTop + l.RangeH * (1.0 - p)

                    w.LeafWithTitle(
                        "line",
                        [
                            "class", "ref"
                            "x1", n left
                            "y1", n y
                            "x2", n right
                            "y2", n y
                            "stroke-width", "1"
                            "stroke-opacity", "0.7"
                        ],
                        r.Label
                    )

                    w.Text(
                        "text",
                        [
                            "class", "t2"
                            "x", n (right - 4.0)
                            "y", n (y - 5.0)
                            "font-size", "10"
                            "text-anchor", "end"
                        ],
                        r.Label
                    )

    let private maxTickWidth (a: Axis) =
        a.Ticks |> List.fold (fun m (_, l) -> max m (TextMetrics.width 11.0 l)) 0.0

    /// The width of the y tick column, including the strip labels.
    let private yLabelWidth (l: YLayout) =
        maxTickWidth l.Axis
        |> max (
            if l.InvalidY.IsSome then
                TextMetrics.width 11.0 "NaN/inf"
            else
                0.0
        )

    /// A cross, the mark of a NaN or infinite value.
    let private crossPath (x: float) (y: float) (r: float) =
        Invariant.format
            $"M{n (x - r)} {n (y - r)}L{n (x + r)} {n (y + r)}M{n (x + r)} {n (y - r)}L{n (x - r)} {n (y + r)}"

    /// A placeholder chart: the title and a message saying why there is nothing to plot.
    let private renderEmpty (id: string) (title: string) (message: string) =
        let w = SvgWriter()
        let h = 96.0
        openSvg w id h title "" ("No data: " + message)

        w.Text(
            "text",
            [ "class", "t2"; "x", "16"; "y", "66"; "font-size", "12" ],
            TextMetrics.fit 12.0 (Width - 32.0) message
        )

        closeSvg w

    // ---- line -------------------------------------------------------------------------------------------------------

    /// Pixel points of one unbroken run, thinned to the first, lowest, highest and last point of every pixel column.
    /// Deterministic, and every extreme survives.
    let private thin (points: (float * float) list) =
        let out = ResizeArray<float * float>()
        let group = ResizeArray<float * float>()

        let flush () =
            if group.Count <= 4 then
                out.AddRange group
            else
                let mutable lo = 0
                let mutable hi = 0

                for i in 1 .. group.Count - 1 do
                    // In pixels, y grows downwards: the smallest y is the highest value.
                    if snd group[i] > snd group[lo] then
                        lo <- i

                    if snd group[i] < snd group[hi] then
                        hi <- i

                [ 0; lo; hi; group.Count - 1 ]
                |> List.distinct
                |> List.sort
                |> List.iter (fun i -> out.Add group[i])

            group.Clear()

        let mutable column = Int32.MinValue

        for x, y in points do
            let c = int (Math.Floor x)

            if c <> column && group.Count > 0 then
                flush ()

            column <- c
            group.Add(x, y)

        flush ()
        List.ofSeq out

    let private renderLine (id: string) (c: LineChart) =
        let ok (x: float, y: float) =
            Double.IsFinite x && Double.IsFinite y && (not c.YLog || y >= 0.0)

        let all = c.Series |> List.collect (fun s -> s.Points |> List.filter ok)

        if all.IsEmpty then
            renderEmpty id c.Title "no finite points"
        else
            let xs = all |> List.map fst

            let bandXs =
                c.Bands
                |> List.collect (fun b -> [ b.From; b.To ])
                |> List.filter Double.IsFinite

            let refYs =
                c.RefLines
                |> List.map _.Value
                |> List.filter (fun v -> Double.IsFinite v && (not c.YLog || v > 0.0))

            let ys =
                (all |> List.map snd |> List.filter (fun y -> not c.YLog || y > 0.0)) @ refYs

            let zero = c.YLog && all |> List.exists (fun (_, y) -> y = 0.0)

            let xa =
                linearAxis c.XFormat (List.min (xs @ bandXs)) (List.max (xs @ bandXs)) 6 c.IntegerX true

            let ya =
                if c.YLog then
                    if ys.IsEmpty then
                        logAxis c.YFormat 1.0 1.0
                    else
                        logAxis c.YFormat (List.min ys) (List.max ys)
                else
                    let lo = List.min ys
                    linearAxis c.YFormat (if c.YFromZero then min 0.0 lo else lo) (List.max ys) 5 false false

            let legend =
                if c.Series.Length >= 2 then
                    c.Series |> List.map (fun s -> s.Name, s.Color)
                else
                    []

            let height = 380.0
            let top = headerHeight c.Subtitle + legendHeight legend + 22.0
            let bottom = 46.0
            let plotH = height - top - bottom
            let layout = yLayout ya top plotH zero false
            let py (v: float) = yOf layout v

            // Direct end labels for 2 to 4 series, only when their ends are at least 14 px apart.
            let ends =
                c.Series
                |> List.choose (fun s ->
                    s.Points
                    |> List.filter ok
                    |> List.tryLast
                    |> Option.map (fun (_, y) -> s.Name, py y))

            let direct =
                c.Series.Length >= 2
                && c.Series.Length <= 4
                && ends.Length = c.Series.Length
                && (ends
                    |> List.map snd
                    |> List.sort
                    |> List.pairwise
                    |> List.forall (fun (a, b) -> b - a >= 14.0))

            let left = 16.0 + yLabelWidth layout + 10.0

            let right =
                if direct then
                    16.0
                    + (ends |> List.fold (fun m (name, _) -> max m (TextMetrics.width 11.0 name)) 0.0)
                    + 10.0
                else
                    24.0

            let plotW = Width - left - right
            let px (v: float) = left + plotW * pos xa v

            let desc =
                Invariant.format
                    $"Line chart of {c.YLabel} against {c.XLabel}: {c.Series.Length} series, {all.Length} points."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            drawLegend w LineKey (headerHeight c.Subtitle + 6.0) legend

            for b in c.Bands do
                let x1 = px (max b.From xa.Lo)
                let x2 = px (min b.To xa.Hi)

                w.LeafWithTitle(
                    "rect",
                    [
                        "class", "band"
                        "x", n x1
                        "y", n top
                        "width", n (max 0.0 (x2 - x1))
                        "height", n plotH
                    ],
                    b.Label
                )

                w.Leaf(
                    "line",
                    [
                        "class", "axis"
                        "x1", n x1
                        "y1", n top
                        "x2", n x1
                        "y2", n (top + plotH)
                        "stroke-width", "1"
                    ]
                )

                w.Text(
                    "text",
                    [ "class", "t3"; "x", n (x1 + 4.0); "y", n (top + 12.0); "font-size", "10" ],
                    TextMetrics.fit 10.0 (max 0.0 (x2 - x1 - 8.0)) b.Label
                )

            yGrid w layout left (left + plotW)
            refLines w layout c.RefLines left (left + plotW)
            xTicks w xa left plotW (top + plotH + 17.0)
            baseline w left (left + plotW) (top + plotH)
            axisLabel w left (top - 10.0) "start" c.YLabel
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel

            for s in c.Series do
                // Unbroken runs of drawable points, each thinned in pixel space.
                let runs = ResizeArray<(float * float) list>()
                let run = ResizeArray<float * float>()

                for p in s.Points do
                    if ok p then
                        run.Add(px (fst p), py (snd p))
                    elif run.Count > 0 then
                        runs.Add(List.ofSeq run)
                        run.Clear()

                if run.Count > 0 then
                    runs.Add(List.ofSeq run)

                let sb = Text.StringBuilder()

                for r in runs do
                    thin r
                    |> List.iteri (fun i (x, y) ->
                        sb.Append(if i = 0 then " M" else " L").Append(n x).Append(' ').Append(n y)
                        |> ignore)

                if sb.Length > 0 then
                    w.LeafWithTitle(
                        "path",
                        [ "class", "line " + Color.stroke s.Color; "d", sb.ToString().TrimStart() ],
                        s.Name
                    )

            let yText (y: float) =
                if c.YLog && y = 0.0 then
                    "0 (exact)"
                else
                    format c.YFormat y

            for s in c.Series do
                let pts = s.Points |> List.filter ok

                if pts.Length <= 40 then
                    for x, y in pts do
                        w.LeafWithTitle(
                            "circle",
                            [
                                "class", Color.fill s.Color + " ring"
                                "cx", n (px x)
                                "cy", n (py y)
                                "r", "4"
                                "stroke-width", "2"
                            ],
                            s.Name + ", " + c.XLabel + " " + format c.XFormat x + ": " + yText y
                        )

            if direct then
                for name, y in ends do
                    w.Text(
                        "text",
                        [
                            "class", "t2"
                            "x", n (left + plotW + 8.0)
                            "y", n (y + 4.0)
                            "font-size", "11"
                        ],
                        name
                    )

            closeSvg w

    // ---- bars -------------------------------------------------------------------------------------------------------

    /// A bar with a 4 px rounded data end and a square base. Horizontal bars grow right; columns grow up.
    let private barPath (horizontal: bool) (x: float) (y: float) (w: float) (h: float) (rounded: bool) =
        let r = if rounded then min 4.0 (min w h / 2.0) else 0.0

        if r <= 0.0 then
            Invariant.format $"M{n x} {n y}h{n w}v{n h}h{n (-w)}Z"
        elif horizontal then
            // Square left end, rounded right end.
            Invariant.format
                $"M{n x} {n y}H{n (x + w - r)}Q{n (x + w)} {n y} {n (x + w)} {n (y + r)}V{n (y + h - r)}Q{n (x + w)} {n (y + h)} {n (x + w - r)} {n (y + h)}H{n x}Z"
        else
            // Square base, rounded top.
            Invariant.format
                $"M{n x} {n (y + h)}V{n (y + r)}Q{n x} {n y} {n (x + r)} {n y}H{n (x + w - r)}Q{n (x + w)} {n y} {n (x + w)} {n (y + r)}V{n (y + h)}Z"

    let private renderBar (id: string) (c: BarChart) =
        let ncat = c.Categories.Length

        let value (s: BarSeries) (i: int) =
            match List.tryItem i s.Values with
            | Some v when Double.IsFinite v && v > 0.0 -> clampV v
            | _ -> 0.0

        let totals =
            [ for i in 0 .. ncat - 1 -> c.Series |> List.sumBy (fun s -> value s i) ]

        let peak =
            if c.Stacked then
                List.fold max 0.0 totals
            else
                c.Series
                |> List.collect (fun s -> [ for i in 0 .. ncat - 1 -> value s i ])
                |> List.fold max 0.0

        if ncat = 0 || c.Series.IsEmpty || peak <= 0.0 then
            renderEmpty id c.Title "no positive values"
        else
            let va = linearAxis c.ValueFormat 0.0 peak 5 false false

            let legend =
                if c.Series.Length >= 2 then
                    c.Series |> List.map (fun s -> s.Name, s.Color)
                else
                    []

            let top = headerHeight c.Subtitle + legendHeight legend + 22.0
            let nser = c.Series.Length

            let desc =
                Invariant.format $"Bar chart of {c.ValueLabel} for {ncat} {c.CategoryLabel} values, {nser} series."

            let w = SvgWriter()

            if c.Horizontal then
                let band = 26.0

                let labelW =
                    c.Categories |> List.fold (fun m l -> max m (TextMetrics.width 12.0 l)) 0.0

                let left = 16.0 + min 240.0 labelW + 10.0
                let right = 16.0 + 64.0
                let plotW = Width - left - right
                let plotH = band * float ncat
                let height = top + plotH + 46.0
                openSvg w id height c.Title c.Subtitle desc
                drawLegend w BoxKey (headerHeight c.Subtitle + 6.0) legend
                let px (v: float) = left + plotW * pos va v

                for t, _ in va.Ticks do
                    let x = left + plotW * tickPos va t
                    gridLine w x top x (top + plotH) false

                xTicks w va left plotW (top + plotH + 17.0)
                axisLabel w left (top - 10.0) "start" c.CategoryLabel
                axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.ValueLabel

                c.Categories
                |> List.iteri (fun i cat ->
                    let y0 = top + band * float i

                    w.Text(
                        "text",
                        [
                            "class", "t2"
                            "x", n (left - 8.0)
                            "y", n (y0 + band / 2.0 + 4.0)
                            "font-size", "12"
                            "text-anchor", "end"
                        ],
                        TextMetrics.fit 12.0 240.0 cat
                    )

                    if c.Stacked || nser = 1 then
                        let thick = 16.0
                        let y = y0 + (band - thick) / 2.0
                        let mutable x = left
                        let segs = c.Series |> List.filter (fun s -> value s i > 0.0)

                        segs
                        |> List.iteri (fun j s ->
                            let v = value s i
                            let wd = px v - left
                            let last = j = segs.Length - 1
                            let shown = if last then wd else max 0.5 (wd - 2.0)

                            w.LeafWithTitle(
                                "path",
                                [ "class", Color.fill s.Color; "d", barPath true x y shown thick last ],
                                cat + ", " + s.Name + ": " + format c.ValueFormat v
                            )

                            x <- x + wd)

                        w.Text(
                            "text",
                            [
                                "class", "t2"
                                "x", n (x + 6.0)
                                "y", n (y + thick / 2.0 + 4.0)
                                "font-size", "11"
                            ],
                            format c.ValueFormat totals[i]
                        )
                    else
                        let thick = min 16.0 ((band - 6.0) / float nser)

                        c.Series
                        |> List.iteri (fun j s ->
                            let v = value s i
                            let y = y0 + 3.0 + float j * thick

                            if v > 0.0 then
                                w.LeafWithTitle(
                                    "path",
                                    [
                                        "class", Color.fill s.Color
                                        "d", barPath true left (y + 1.0) (px v - left) (thick - 2.0) true
                                    ],
                                    cat + ", " + s.Name + ": " + format c.ValueFormat v
                                )))

                baseline w left left top

                w.Leaf(
                    "line",
                    [
                        "class", "axis"
                        "x1", n left
                        "y1", n top
                        "x2", n left
                        "y2", n (top + plotH)
                        "stroke-width", "1"
                    ]
                )

                closeSvg w
            else
                let height = 380.0
                let bottom = 46.0
                let plotH = height - top - bottom
                let left = 16.0 + maxTickWidth va + 10.0
                let right = 24.0
                let plotW = Width - left - right
                let band = plotW / float ncat
                let colW = min 24.0 (band * 0.72)
                let layout = yLayout va top plotH false false
                openSvg w id height c.Title c.Subtitle desc
                drawLegend w BoxKey (headerHeight c.Subtitle + 6.0) legend
                yGrid w layout left (left + plotW)
                axisLabel w left (top - 10.0) "start" c.ValueLabel
                axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.CategoryLabel

                let labelW =
                    c.Categories |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0

                let every = max 1 (int (Math.Ceiling((labelW + 6.0) / band)))

                c.Categories
                |> List.iteri (fun i cat ->
                    let cx = left + band * (float i + 0.5)

                    if i % every = 0 then
                        tickLabel w cx (top + plotH + 17.0) "middle" cat

                    if c.Stacked || nser = 1 then
                        let x = cx - colW / 2.0
                        let mutable base' = top + plotH
                        let segs = c.Series |> List.filter (fun s -> value s i > 0.0)

                        segs
                        |> List.iteri (fun j s ->
                            let v = value s i
                            let h = plotH * pos va v
                            let last = j = segs.Length - 1
                            let shown = if last then h else max 0.5 (h - 2.0)

                            w.LeafWithTitle(
                                "path",
                                [
                                    "class", Color.fill s.Color
                                    "d", barPath false x (base' - shown) colW shown last
                                ],
                                cat + ", " + s.Name + ": " + format c.ValueFormat v
                            )

                            base' <- base' - h)
                    else
                        let sub = colW / float nser

                        c.Series
                        |> List.iteri (fun j s ->
                            let v = value s i
                            let h = plotH * pos va v

                            if v > 0.0 then
                                w.LeafWithTitle(
                                    "path",
                                    [
                                        "class", Color.fill s.Color
                                        "d",
                                        barPath
                                            false
                                            (cx - colW / 2.0 + float j * sub + 0.5)
                                            (top + plotH - h)
                                            (sub - 1.0)
                                            h
                                            true
                                    ],
                                    cat + ", " + s.Name + ": " + format c.ValueFormat v
                                )))

                baseline w left (left + plotW) (top + plotH)
                closeSvg w

    // ---- heatmap ----------------------------------------------------------------------------------------------------

    let private renderHeatmap (id: string) (c: Heatmap) =
        let nr = c.Rows.Length
        let nc = c.Cols.Length

        let raw (i: int) (j: int) =
            if i < c.Values.Length && j < c.Values[i].Length then
                c.Values[i][j]
            else
                None

        // Columns binned to at most MaxHeatColumns: bin b holds columns [b*k, min(nc, (b+1)*k)).
        let k = max 1 ((nc + MaxHeatColumns - 1) / MaxHeatColumns)
        let nb = (nc + k - 1) / k
        let binStart (b: int) = b * k
        let binEnd (b: int) = min nc ((b + 1) * k)

        let invalidValue (v: float) =
            not (Double.IsFinite v) || (c.Log && v < 0.0)

        let aggregate (i: int) (b: int) =
            let xs =
                [
                    for j in binStart b .. binEnd b - 1 do
                        match raw i j with
                        | Some v -> yield v
                        | None -> ()
                ]

            if xs.IsEmpty then
                None
            elif xs |> List.exists invalidValue then
                Some nan
            else
                match c.Aggregate with
                | HeatAggregate.Max -> Some(List.max xs)
                | HeatAggregate.Mean -> Some(List.sum xs / float xs.Length)

        let cells = Array.init nr (fun i -> Array.init nb (fun b -> aggregate i b))

        let kindOf (v: float option) =
            match v with
            | None -> Missing
            | Some v when invalidValue v -> Bad
            | Some v when c.Log && v = 0.0 -> Exact
            | Some v ->
                match c.Ceiling with
                | Some(top, _) when v > top -> Over v
                | _ -> Scaled v

        let kinds = cells |> Array.map (Array.map kindOf)
        let all = kinds |> Array.concat

        let scaled =
            all
            |> Array.choose (fun k ->
                match k with
                | Scaled v -> Some v
                | _ -> None)

        let has (pred: CellKind -> bool) = all |> Array.exists pred

        if nr = 0 || nc = 0 || all |> Array.forall (fun k -> k = Missing) then
            renderEmpty id c.Title "no cells with values"
        else
            let tr (v: float) =
                if c.Log then DetMath.log10 (clampPos v) else clampV v

            let lo, hi =
                match c.Ceiling with
                | Some(top, _) ->
                    let hi = tr top

                    (if scaled.Length = 0 then
                         hi
                     else
                         min hi (scaled |> Array.map tr |> Array.min)),
                    hi
                | None ->
                    if scaled.Length = 0 then
                        0.0, 0.0
                    else
                        scaled |> Array.map tr |> Array.min, scaled |> Array.map tr |> Array.max

            let step (v: float) =
                if hi <= lo then
                    Theme.steps / 2
                else
                    max 0 (min (Theme.steps - 1) (int (Math.Floor((tr v - lo) / (hi - lo) * float Theme.steps))))

            let aggName =
                match c.Aggregate with
                | HeatAggregate.Max -> "max"
                | HeatAggregate.Mean -> "mean"

            let subtitle =
                if k = 1 then
                    c.Subtitle
                else
                    let note = Invariant.format $"columns bin {k} {c.ColLabel}s each ({aggName})"
                    if c.Subtitle = "" then note else c.Subtitle + "; " + note

            let binLabel (b: int) =
                if binEnd b - binStart b = 1 then
                    c.Cols[binStart b]
                else
                    c.Cols[binStart b] + "–" + c.Cols[binEnd b - 1]

            let rowW = c.Rows |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0
            let left = 16.0 + min 200.0 rowW + 10.0
            let right = 16.0
            let top = headerHeight subtitle + 22.0 + 22.0
            let plotW = Width - left - right
            let colPx = plotW / float nc
            let cellH = max 8.0 (min 26.0 (300.0 / float nr))
            let plotH = cellH * float nr
            let bottom = 46.0
            let height = top + plotH + bottom

            let present = all |> Array.filter (fun k -> k <> Missing) |> Array.length

            let desc =
                Invariant.format
                    $"Heatmap of {nr} {c.RowLabel} rows by {nc} {c.ColLabel} columns, {present} cells with values."

            let w = SvgWriter()
            openSvg w id height c.Title subtitle desc

            // Colour key: the lowest and highest value with the ramp between them, then the special cells present.
            let keyY = headerHeight subtitle + 2.0
            let mutable kx = 16.0

            let keyText (text: string) =
                w.Text("text", [ "class", "t3"; "x", n kx; "y", n (keyY + 9.0); "font-size", "11" ], text)
                kx <- kx + TextMetrics.width 11.0 text + 6.0

            let swatch (cls: string) (outlined: bool) (cross: bool) =
                if outlined then
                    w.Leaf(
                        "rect",
                        [
                            "class", cls
                            "x", n (kx + 0.5)
                            "y", n (keyY + 0.5)
                            "width", "12"
                            "height", "9"
                            "stroke-width", "1"
                        ]
                    )
                else
                    w.Leaf("rect", [ "class", cls; "x", n kx; "y", n keyY; "width", "13"; "height", "10" ])

                if cross then
                    w.Leaf(
                        "path",
                        [
                            "class", "gap"
                            "d", crossPath (kx + 6.5) (keyY + 5.0) 3.0
                            "stroke-width", "1.5"
                            "fill", "none"
                        ]
                    )

                kx <- kx + 18.0

            if scaled.Length > 0 || c.Ceiling.IsSome then
                let lowValue = if c.Log then DetMath.exp10 lo else lo

                let highValue =
                    match c.Ceiling with
                    | Some(top, _) -> top
                    | None -> if c.Log then DetMath.exp10 hi else hi

                let lowText =
                    if scaled.Length > 0 then
                        format c.ValueFormat (scaled |> Array.min)
                    else
                        format c.ValueFormat lowValue

                let highText =
                    if c.Ceiling.IsSome || scaled.Length = 0 then
                        format c.ValueFormat highValue
                    else
                        format c.ValueFormat (scaled |> Array.max)

                keyText lowText

                for s in 0 .. Theme.steps - 1 do
                    w.Leaf(
                        "rect",
                        [
                            "class", "q" + Invariant.i32 s
                            "x", n (kx + float s * 14.0)
                            "y", n keyY
                            "width", "13"
                            "height", "10"
                        ]
                    )

                kx <- kx + float Theme.steps * 14.0 + 5.0
                keyText (highText + (if c.Log then " (log scale)" else ""))
                kx <- kx + 8.0

            if has (fun k -> k = Exact) then
                swatch "qz" true false
                keyText "0 (exact)"

            if
                has (fun k ->
                    match k with
                    | Over _ -> true
                    | _ -> false)
            then
                swatch "fcrit" false false
                keyText (c.Ceiling |> Option.map snd |> Option.defaultValue "over")

            if has (fun k -> k = Bad) then
                swatch "fcrit" false true
                keyText "NaN/inf"

            if has (fun k -> k = Missing) then
                swatch "qx" false false
                keyText "no data"

            axisLabel w left (top - 8.0) "start" c.RowLabel

            let gap = if colPx * float k > 4.0 then 1.0 else 0.0

            c.Rows
            |> List.iteri (fun i label ->
                let every = max 1 (int (Math.Ceiling(12.0 / cellH)))

                if i % every = 0 then
                    tickLabel
                        w
                        (left - 6.0)
                        (top + cellH * float i + cellH / 2.0 + 4.0)
                        "end"
                        (TextMetrics.fit 11.0 200.0 label)

                for b in 0 .. nb - 1 do
                    let x = left + colPx * float (binStart b)
                    let cw = colPx * float (binEnd b - binStart b) - gap
                    let y = top + cellH * float i
                    let ch = cellH - 1.0
                    let prefix = if k = 1 then "" else aggName + " "

                    let tip (text: string) =
                        label + ", " + c.ColLabel + " " + binLabel b + ": " + text

                    match kinds[i][b] with
                    | Scaled v ->
                        w.LeafWithTitle(
                            "rect",
                            [
                                "class", "q" + Invariant.i32 (step v)
                                "x", n x
                                "y", n y
                                "width", n cw
                                "height", n ch
                            ],
                            tip (prefix + format c.ValueFormat v)
                        )
                    | Missing ->
                        w.LeafWithTitle(
                            "rect",
                            [ "class", "qx"; "x", n x; "y", n y; "width", n cw; "height", n ch ],
                            tip "no data"
                        )
                    | Exact ->
                        w.LeafWithTitle(
                            "rect",
                            [
                                "class", "qz"
                                "x", n (x + 0.5)
                                "y", n (y + 0.5)
                                "width", n (max 0.0 (cw - 1.0))
                                "height", n (max 0.0 (ch - 1.0))
                                "stroke-width", "1"
                            ],
                            tip (prefix + "0 (exact)")
                        )
                    | Over v ->
                        w.LeafWithTitle(
                            "rect",
                            [ "class", "fcrit"; "x", n x; "y", n y; "width", n cw; "height", n ch ],
                            tip (
                                prefix
                                + format c.ValueFormat v
                                + " ("
                                + (c.Ceiling |> Option.map snd |> Option.defaultValue "over")
                                + ")"
                            )
                        )
                    | Bad ->
                        w.LeafWithTitle(
                            "rect",
                            [ "class", "fcrit"; "x", n x; "y", n y; "width", n cw; "height", n ch ],
                            tip "NaN or infinite"
                        )

                        if cw >= 6.0 && ch >= 6.0 then
                            w.Leaf(
                                "path",
                                [
                                    "class", "gap"
                                    "d", crossPath (x + cw / 2.0) (y + ch / 2.0) (min 3.5 (min cw ch / 2.0 - 1.5))
                                    "stroke-width", "1.5"
                                    "fill", "none"
                                ]
                            ))

            let labelY = top + plotH + 16.0
            let values = Array.ofList c.ColValues

            let numeric =
                values.Length = nc
                && nc >= 2
                && values |> Array.forall Double.IsFinite
                && values |> Array.pairwise |> Array.forall (fun (a, b) -> b > a)

            if numeric then
                // Nice ticks over the column values, placed between the columns they fall between.
                let a = linearAxis NumFormat.Int values[0] values[nc - 1] 6 true true

                let colOf (v: float) =
                    let mutable i = 0

                    while i < nc - 2 && values[i + 1] < v do
                        i <- i + 1

                    float i + (v - values[i]) / (values[i + 1] - values[i])

                let mutable lastEnd = Double.NegativeInfinity

                for t, label in a.Ticks do
                    let x = left + colPx * (colOf t + 0.5)
                    let half = TextMetrics.width 11.0 label / 2.0

                    if x - half > lastEnd + 6.0 && x + half <= Width - 4.0 then
                        w.Leaf(
                            "line",
                            [
                                "class", "axis"
                                "x1", n x
                                "y1", n (top + plotH)
                                "x2", n x
                                "y2", n (top + plotH + 4.0)
                                "stroke-width", "1"
                            ]
                        )

                        tickLabel w x labelY "middle" label
                        lastEnd <- x + half
            else
                let labels = [ for b in 0 .. nb - 1 -> binLabel b ]
                let colW = labels |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0
                let every = max 1 (int (Math.Ceiling((colW + 6.0) / (colPx * float k))))

                labels
                |> List.iteri (fun b label ->
                    if b % every = 0 then
                        let x = left + colPx * (float (binStart b + binEnd b) / 2.0)
                        tickLabel w x labelY "middle" label)

            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.ColLabel
            closeSvg w

    // ---- histogram --------------------------------------------------------------------------------------------------

    let private renderHistogram (id: string) (c: Histogram) =
        let bins =
            c.Bins
            |> List.filter (fun b -> Double.IsFinite b.Lo && Double.IsFinite b.Hi && b.Hi > b.Lo && b.Count >= 0)

        let zeros = max 0 c.Zeros
        let invalid = max 0 c.Invalid
        let binned = bins |> List.sumBy (fun b -> int64 b.Count)
        let total = binned + int64 zeros + int64 invalid

        if total = 0L then
            renderEmpty id c.Title "no samples"
        else
            let marks = c.Marks |> List.filter (fun m -> Double.IsFinite m.At)

            let lo, hi =
                match bins with
                | [] ->
                    match marks with
                    | [] -> 0.0, 1.0
                    | ms -> (ms |> List.map _.At |> List.min) - 1.0, (ms |> List.map _.At |> List.max) + 1.0
                | _ -> (List.head bins).Lo, (List.last bins).Hi

            let xLo = marks |> List.fold (fun m v -> min m v.At) lo
            let xHi = marks |> List.fold (fun m v -> max m v.At) hi
            let xa = linearAxis c.XFormat xLo xHi 6 (isDecades c.XFormat) true

            let peak =
                bins
                |> List.fold (fun m b -> max m (int64 b.Count)) (max (int64 zeros) (int64 invalid))

            let ya = linearAxis NumFormat.Int 0.0 (float peak) 4 true false
            let height = 320.0
            let markRow = if marks.IsEmpty then 0.0 else 14.0
            let top = headerHeight c.Subtitle + 22.0 + markRow
            let bottom = 46.0
            let plotH = height - top - bottom
            let left = 16.0 + maxTickWidth ya + 10.0
            let right = 24.0
            let zeroW = if zeros > 0 then 30.0 else 0.0
            let invalidW = if invalid > 0 then 48.0 else 0.0
            let sideGap = 12.0
            let mainLeft = left + (if zeros > 0 then zeroW + sideGap else 0.0)
            let mainRight = Width - right - (if invalid > 0 then invalidW + sideGap else 0.0)
            let plotW = mainRight - mainLeft
            let px (v: float) = mainLeft + plotW * pos xa v
            let layout = yLayout ya top plotH false false
            let py (v: float) = yOf layout v

            let extra =
                (if zeros > 0 then
                     Invariant.format $", {zeros} exactly 0"
                 else
                     "")
                + (if invalid > 0 then
                       Invariant.format $", {invalid} NaN, infinite or out of range"
                   else
                       "")

            let desc =
                Invariant.format $"Histogram of {c.XLabel}: {total} samples, {binned} in {bins.Length} bins{extra}."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            yGrid w layout left (Width - right)
            xTicks w xa mainLeft plotW (top + plotH + 17.0)
            axisLabel w left (top - 10.0 - markRow) "start" "count"
            axisLabel w (mainLeft + plotW / 2.0) (height - 10.0) "middle" c.XLabel

            let column (x: float) (wd: float) (count: int64) (cls: string) (tip: string) =
                if count > 0L then
                    let h = top + plotH - py (float count)

                    w.LeafWithTitle("path", [ "class", cls; "d", barPath false x (top + plotH - h) wd h true ], tip)

            for b in bins do
                let x1 = px b.Lo
                let x2 = px b.Hi

                column
                    (x1 + 1.0)
                    (max 1.0 (x2 - x1 - 2.0))
                    (int64 b.Count)
                    (Color.fill c.Color)
                    ("["
                     + format c.XFormat b.Lo
                     + ", "
                     + format c.XFormat b.Hi
                     + "): "
                     + Invariant.i32 b.Count)

            if zeros > 0 then
                column
                    (left + 2.0)
                    (zeroW - 4.0)
                    (int64 zeros)
                    (Color.fill c.Color)
                    (Invariant.format $"exactly 0: {zeros}")

                tickLabel w (left + zeroW / 2.0) (top + plotH + 17.0) "middle" "0"
                gridLine w (mainLeft - sideGap / 2.0) top (mainLeft - sideGap / 2.0) (top + plotH) true

            if invalid > 0 then
                let x = mainRight + sideGap

                column
                    (x + 2.0)
                    (invalidW - 4.0)
                    (int64 invalid)
                    "fcrit"
                    (Invariant.format $"NaN, infinite or out of range: {invalid}")

                tickLabel w (x + invalidW / 2.0) (top + plotH + 17.0) "middle" "NaN/inf"
                gridLine w (mainRight + sideGap / 2.0) top (mainRight + sideGap / 2.0) (top + plotH) true

            for m in marks do
                let x = px m.At

                w.LeafWithTitle(
                    "line",
                    [
                        "class", "ref"
                        "x1", n x
                        "y1", n (top - 4.0)
                        "x2", n x
                        "y2", n (top + plotH)
                        "stroke-width", "1"
                    ],
                    m.Label
                )

                // The label sits above the plot, beside its line, and turns to the left of it at the right edge.
                let labelW = TextMetrics.width 10.0 m.Label
                let toLeft = x + 4.0 + labelW > Width - 8.0

                w.Text(
                    "text",
                    [
                        "class", "t2"
                        "x", n (if toLeft then x - 4.0 else x + 4.0)
                        "y", n (top - 6.0)
                        "font-size", "10"
                        "text-anchor", (if toLeft then "end" else "start")
                    ],
                    m.Label
                )

            baseline w left (Width - right) (top + plotH)
            closeSvg w

    // ---- timeline ---------------------------------------------------------------------------------------------------

    let private emphasized (c: Color) =
        match c with
        | Color.Warning
        | Color.Serious
        | Color.Critical -> true
        | _ -> false

    let private renderTimeline (id: string) (c: Timeline) =
        let intervals =
            c.Intervals
            |> List.filter (fun i ->
                Double.IsFinite i.Start
                && Double.IsFinite i.Length
                && Double.IsFinite(i.Start + i.Length)
                && i.Length >= 0.0
                && i.Lane >= 0
                && i.Lane < c.Lanes.Length)

        if intervals.IsEmpty then
            renderEmpty id c.Title "no intervals"
        else
            let marks = c.Marks |> List.filter (fun m -> Double.IsFinite m.At)
            let lo = (intervals |> List.map _.Start) @ (marks |> List.map _.At) |> List.min

            let hi =
                (intervals |> List.map (fun i -> i.Start + i.Length)) @ (marks |> List.map _.At)
                |> List.max

            let xa = linearAxis c.XFormat lo hi 6 false true
            let laneH = 22.0
            let thick = 14.0
            let labelW = c.Lanes |> List.fold (fun m l -> max m (TextMetrics.width 12.0 l)) 0.0
            let left = 16.0 + min 220.0 labelW + 10.0
            let right = 24.0
            let plotW = Width - left - right
            // Marker labels take two staggered rows above the plot.
            let top =
                headerHeight c.Subtitle
                + legendHeight c.Legend
                + 26.0
                + (if marks.IsEmpty then 0.0 else 12.0)

            let plotH = laneH * float c.Lanes.Length
            let height = top + plotH + 46.0
            let px (v: float) = left + plotW * pos xa v

            let desc =
                Invariant.format
                    $"Timeline of {intervals.Length} intervals on {c.Lanes.Length} lanes, {marks.Length} event marks."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            drawLegend w BoxKey (headerHeight c.Subtitle + 6.0) c.Legend

            for t, _ in xa.Ticks do
                let x = left + plotW * tickPos xa t
                gridLine w x top x (top + plotH) false

            c.Lanes
            |> List.iteri (fun i lane ->
                w.Text(
                    "text",
                    [
                        "class", "t2"
                        "x", n (left - 8.0)
                        "y", n (top + laneH * float i + laneH / 2.0 + 4.0)
                        "font-size", "12"
                        "text-anchor", "end"
                    ],
                    TextMetrics.fit 12.0 220.0 lane
                ))

            let drawInterval (iv: Interval) =
                let x1 = px iv.Start
                let strong = emphasized iv.Color
                let minW = if strong then 4.0 else 1.0
                let wd = max minW (px (iv.Start + iv.Length) - x1 - 1.0)
                let x1 = min x1 (left + plotW - wd)
                let y = top + laneH * float iv.Lane + (laneH - thick) / 2.0
                let r = min 3.0 (wd / 2.0)

                w.LeafWithTitle(
                    "rect",
                    [
                        "class", Color.fill iv.Color + (if strong then " emph" else "")
                        "x", n x1
                        "y", n y
                        "width", n wd
                        "height", n thick
                        "rx", n r
                        if strong then
                            "stroke-width", "1"
                    ],
                    iv.Tip
                )

            for iv in intervals do
                if not (emphasized iv.Color) then
                    drawInterval iv

            let marks = marks |> List.sortBy _.At

            for m in marks do
                let x = px m.At

                w.LeafWithTitle(
                    "line",
                    [
                        "class", Color.stroke m.Color
                        "x1", n x
                        "y1", n (top - 6.0)
                        "x2", n x
                        "y2", n (top + plotH)
                        "stroke-width", "2"
                    ],
                    m.Label
                )

                w.Leaf(
                    "path",
                    [
                        "class", Color.fill m.Color
                        "d", Invariant.format $"M{n (x - 5.0)} {n (top - 12.0)}h10l-5 7Z"
                    ]
                )

            // Outcomes to notice go on top of the markers, so a marker never hides one.
            for iv in intervals do
                if emphasized iv.Color then
                    drawInterval iv

            // Labels go above the markers in two staggered rows. A label that fits in neither row is left to the
            // marker's tooltip.
            let rowEnd = [| Double.NegativeInfinity; Double.NegativeInfinity |]

            for m in marks do
                let x = px m.At
                let labelW = TextMetrics.width 10.0 m.Label
                let anchorEnd = x + 8.0 + labelW > Width - 8.0
                let x0 = if anchorEnd then x - 8.0 - labelW else x + 8.0

                match [ 0; 1 ] |> List.tryFind (fun r -> x0 > rowEnd[r] + 4.0) with
                | Some r ->
                    rowEnd[r] <- x0 + labelW

                    w.Text(
                        "text",
                        [
                            "class", "t2"
                            "x", n (if anchorEnd then x - 8.0 else x + 8.0)
                            "y", n (top - 6.0 - 12.0 * float r)
                            "font-size", "10"
                            "text-anchor", (if anchorEnd then "end" else "start")
                        ],
                        m.Label
                    )
                | None -> ()

            xTicks w xa left plotW (top + plotH + 17.0)
            baseline w left (left + plotW) (top + plotH)
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel
            closeSvg w

    // ---- error bars -------------------------------------------------------------------------------------------------

    let private renderErrorBars (id: string) (c: ErrorBarChart) =
        let inRange (p: ErrorPoint) =
            p.Index >= 0 && p.Index < c.Categories.Length

        let ok (p: ErrorPoint) =
            inRange p
            && [ p.Low; p.Mid; p.High ]
               |> List.forall (fun v -> Double.IsFinite v && (not c.YLog || v >= 0.0))

        let bad (p: ErrorPoint) = inRange p && p.Invalid > 0
        let pts = c.Series |> List.collect (fun s -> s.Spreads |> List.filter ok)
        let invalid = c.Series |> List.exists (fun s -> s.Spreads |> List.exists bad)

        if pts.IsEmpty && not invalid then
            renderEmpty id c.Title "no finite points"
        else
            let refYs =
                c.RefLines
                |> List.map _.Value
                |> List.filter (fun v -> Double.IsFinite v && (not c.YLog || v > 0.0))

            let ys =
                (pts
                 |> List.collect (fun p -> [ p.Low; p.Mid; p.High ])
                 |> List.filter (fun v -> not c.YLog || v > 0.0))
                @ refYs

            let zero =
                c.YLog
                && pts |> List.exists (fun p -> p.Low = 0.0 || p.Mid = 0.0 || p.High = 0.0)

            let ya =
                if ys.IsEmpty then
                    if c.YLog then
                        logAxis c.YFormat 1.0 1.0
                    else
                        linearAxis c.YFormat 0.0 1.0 5 false false
                elif c.YLog then
                    logAxis c.YFormat (List.min ys) (List.max ys)
                else
                    linearAxis c.YFormat (List.min ys) (List.max ys) 5 false false

            let legend =
                if c.Series.Length >= 2 then
                    c.Series |> List.map (fun s -> s.Name, s.Color)
                else
                    []

            let height = 380.0
            let top = headerHeight c.Subtitle + legendHeight legend + 22.0
            let bottom = 46.0
            let plotH = height - top - bottom
            let layout = yLayout ya top plotH zero invalid
            let left = 16.0 + yLabelWidth layout + 10.0
            let right = 24.0
            let plotW = Width - left - right
            let ncat = c.Categories.Length
            let band = plotW / float ncat
            let py (v: float) = yOf layout v
            let nser = c.Series.Length
            let dodge = min 9.0 (band / float (nser + 1))

            let desc =
                Invariant.format
                    $"Error bars of {c.YLabel} per {c.XLabel}: {nser} series over {ncat} categories, whiskers from low to high."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            drawLegend w DotKey (headerHeight c.Subtitle + 6.0) legend
            yGrid w layout left (left + plotW)
            refLines w layout c.RefLines left (left + plotW)
            axisLabel w left (top - 10.0) "start" c.YLabel
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel

            let labelW =
                c.Categories |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0

            let every = max 1 (int (Math.Ceiling((labelW + 6.0) / band)))

            c.Categories
            |> List.iteri (fun i cat ->
                if i % every = 0 then
                    tickLabel w (left + band * (float i + 0.5)) (top + plotH + 17.0) "middle" cat)

            let yText (v: float) =
                if c.YLog && v = 0.0 then
                    "0 (exact)"
                else
                    format c.YFormat v

            c.Series
            |> List.iteri (fun j s ->
                let offset = (float j - float (nser - 1) / 2.0) * dodge

                for p in s.Spreads do
                    let x = left + band * (float p.Index + 0.5) + offset

                    if ok p then
                        let yl = py p.Low
                        let yh = py p.High

                        let tip =
                            s.Name
                            + ", "
                            + c.XLabel
                            + " "
                            + c.Categories[p.Index]
                            + ": "
                            + yText p.Mid
                            + " ("
                            + yText p.Low
                            + " to "
                            + yText p.High
                            + ")"
                            + (if p.Invalid > 0 then
                                   Invariant.format $", and {p.Invalid} NaN or infinite"
                               else
                                   "")

                        w.Leaf(
                            "path",
                            [
                                "class", Color.stroke s.Color
                                "d",
                                Invariant.format $"M{n x} {n yl}V{n yh}M{n (x - 3.0)} {n yl}h6M{n (x - 3.0)} {n yh}h6"
                                "stroke-width", "1.5"
                                "fill", "none"
                            ]
                        )

                        w.LeafWithTitle(
                            "circle",
                            [
                                "class", Color.fill s.Color + " ring"
                                "cx", n x
                                "cy", n (py p.Mid)
                                "r", "4"
                                "stroke-width", "1.5"
                            ],
                            tip
                        )

                    if bad p then
                        match layout.InvalidY with
                        | Some y ->
                            w.LeafWithTitle(
                                "path",
                                [
                                    "class", "scrit"
                                    "d", crossPath x y 4.0
                                    "stroke-width", "2"
                                    "fill", "none"
                                ],
                                Invariant.format
                                    $"{s.Name}, {c.XLabel} {c.Categories[p.Index]}: {p.Invalid} NaN or infinite"
                            )
                        | None -> ())

            baseline w left (left + plotW) (top + plotH)
            closeSvg w

    /// The chart's title.
    let title (c: Chart) =
        match c with
        | Chart.Line x -> x.Title
        | Chart.Bar x -> x.Title
        | Chart.Heatmap x -> x.Title
        | Chart.Histogram x -> x.Title
        | Chart.Timeline x -> x.Title
        | Chart.ErrorBars x -> x.Title
        | Chart.Empty(t, _) -> t

    /// The SVG text of a chart. `id` (a safe name, usually the file stem) prefixes the ids inside the SVG, so several
    /// charts can be inlined on one page. The same chart always renders to the same bytes, under any culture. A chart
    /// that cannot be drawn becomes a placeholder naming the failure, so one bad series never stops a report.
    let render (id: string) (c: Chart) : string =
        try
            match c with
            | Chart.Line x -> renderLine id x
            | Chart.Bar x -> renderBar id x
            | Chart.Heatmap x -> renderHeatmap id x
            | Chart.Histogram x -> renderHistogram id x
            | Chart.Timeline x -> renderTimeline id x
            | Chart.ErrorBars x -> renderErrorBars id x
            | Chart.Empty(t, m) -> renderEmpty id t m
        with e ->
            renderEmpty id (title c) ("this chart could not be drawn (" + e.GetType().Name + ")")
