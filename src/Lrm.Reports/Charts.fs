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
    /// The value is a base-10 exponent: 2 prints as "100", -6 as "1e-6".
    | Pow10

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

/// Line chart: metrics over steps, or per-layer multi-series with IntegerX (ReportTemplates contract).
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

/// Heatmap: rows x columns on the sequential ramp. NaN cells are missing.
type Heatmap =
    {
        Title: string
        Subtitle: string
        RowLabel: string
        ColLabel: string
        Rows: string list
        Cols: string list
        /// Row-major values: Values[row][col].
        Values: float[][]
        Log: bool
        ValueFormat: NumFormat
    }

/// One histogram bin [Lo, Hi).
type HistBin = { Lo: float; Hi: float; Count: int }

/// A vertical reference line on a histogram.
type VLine = { At: float; Label: string }

/// Histogram of pre-binned counts.
type Histogram =
    {
        Title: string
        Subtitle: string
        XLabel: string
        /// The format of bin edges (Pow10 when the bins are in log10 space).
        XFormat: NumFormat
        Bins: HistBin list
        Color: Color
        Marks: VLine list
    }

/// One bar on a timeline lane.
type Interval =
    {
        Lane: int
        Start: float
        Length: float
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

/// One category's spread: whiskers from Low to High around Mid.
type ErrorPoint =
    {
        Index: int
        Low: float
        Mid: float
        High: float
    }

type ErrorSeries =
    {
        Name: string
        Color: Color
        Spreads: ErrorPoint list
    }

/// Per-layer error bars: for each category, each series' Mid with Low-High whiskers.
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

/// Deterministic SVG rendering of every chart kind (ReportTemplates contract, "Charts").
[<RequireQualifiedAccess>]
module ChartSvg =
    let private n = SvgWriter.Num

    [<Literal>]
    let Width = 720.0

    let private pow10Label (k: float) =
        if k <> Math.Round k then
            "1e" + Fmt.dp 1 k
        else
            match int k with
            | 0 -> "1"
            | 1 -> "10"
            | 2 -> "100"
            | -1 -> "0.1"
            | -2 -> "0.01"
            | k -> "1e" + Invariant.i32 k

    /// A value in a format, for tooltips and labels.
    let format (f: NumFormat) (x: float) =
        match f with
        | NumFormat.Plain -> Fmt.num x
        | NumFormat.Int ->
            if Double.IsFinite x then
                Fmt.int (int64 (Math.Round x))
            else
                Fmt.num x
        | NumFormat.Si -> Fmt.si x
        | NumFormat.Bytes -> Fmt.bytes x
        | NumFormat.Duration -> Fmt.duration x
        | NumFormat.Pct -> Fmt.pct x
        | NumFormat.Sci -> Fmt.sci 1 x
        | NumFormat.Pow10 -> pow10Label x

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

    let private tickText (f: NumFormat) (step: float) (lo: float) (hi: float) (v: float) =
        match f with
        | NumFormat.Plain ->
            let big = max (abs lo) (abs hi)

            if big >= 1e5 then
                Fmt.si v
            elif step < 1e-4 then
                (if v = 0.0 then "0" else Fmt.sci 1 v)
            else
                let k = DetMath.floorLog10 step
                let mantissa = step / DetMath.pow10 k
                let decimals = max 0 (-k + (if mantissa = 2.5 then 1 else 0))
                Fmt.dp (min decimals 6) v
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

    /// A linear axis with about `target` ticks at 1, 2, 2.5 or 5 times a power of ten of the display unit. A tight
    /// axis spans exactly [lo, hi] and keeps only the ticks inside it; otherwise the axis extends to whole ticks.
    let private linearAxis (f: NumFormat) (lo: float) (hi: float) (target: int) (integer: bool) (tight: bool) =
        let lo, hi =
            if hi > lo then lo, hi
            elif lo = 0.0 then 0.0, 1.0
            else lo - abs lo * 0.1, hi + abs hi * 0.1

        let scale, suffix = unitFor f (max (abs lo) (abs hi))
        let slo = lo / scale
        let shi = hi / scale
        let step = niceStep ((shi - slo) / float target)
        let step = if integer then max 1.0 (Math.Round step) else step
        let first = Math.Floor(slo / step) * step
        let last = Math.Ceiling(shi / step) * step
        let count = int (Math.Round((last - first) / step))
        let eps = step * 1e-9

        let label (v: float) =
            match f with
            | NumFormat.Duration
            | NumFormat.Bytes
            | NumFormat.Si
            | NumFormat.Pct ->
                if v = 0.0 then
                    "0"
                else
                    let k = DetMath.floorLog10 step
                    let mantissa = step / DetMath.pow10 k
                    let decimals = max 0 (-k + (if mantissa = 2.5 then 1 else 0))
                    Fmt.dp (min decimals 6) v + suffix
            | f -> tickText f step first last v

        let ticks =
            [ for i in 0..count -> first + float i * step ]
            |> List.filter (fun v -> not tight || (v >= slo - eps && v <= shi + eps))
            |> List.map (fun v -> v * scale, label v)

        {
            Lo = (if tight then lo else first * scale)
            Hi = (if tight then hi else last * scale)
            Log = false
            Ticks = ticks
        }

    let private logAxis (lo: float) (hi: float) =
        let lo, hi = if hi > lo then lo, hi else lo / 10.0, hi * 10.0
        let kLo = DetMath.floorLog10 lo

        let kHi =
            let k = DetMath.floorLog10 hi
            if DetMath.pow10 k < hi then k + 1 else k

        let kHi = if kHi = kLo then kHi + 1 else kHi
        let every = max 1 ((kHi - kLo + 7) / 8)

        let ticks =
            [ kLo..kHi ]
            |> List.filter (fun k -> (k - kLo) % every = 0)
            |> List.map (fun k -> float k, pow10Label (float k))

        {
            Lo = float kLo
            Hi = float kHi
            Log = true
            Ticks = ticks
        }

    /// The position of a value on an axis, 0 at Lo and 1 at Hi.
    let private pos (a: Axis) (v: float) =
        if a.Log then
            (DetMath.log10 v - a.Lo) / (a.Hi - a.Lo)
        else
            (v - a.Lo) / (a.Hi - a.Lo)

    /// The position of a tick (ticks of a log axis are exponents).
    let private tickPos (a: Axis) (t: float) = (t - a.Lo) / (a.Hi - a.Lo)

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

    let private yGrid (w: SvgWriter) (a: Axis) (left: float) (right: float) (top: float) (plotH: float) =
        for t, label in a.Ticks do
            let y = top + plotH * (1.0 - tickPos a t)

            w.Leaf(
                "line",
                [
                    "class", "grid"
                    "x1", n left
                    "y1", n y
                    "x2", n right
                    "y2", n y
                    "stroke-width", "1"
                ]
            )

            w.Text(
                "text",
                [
                    "class", "t3"
                    "x", n (left - 8.0)
                    "y", n (y + 4.0)
                    "font-size", "11"
                    "text-anchor", "end"
                ],
                label
            )

    let private xTicks (w: SvgWriter) (a: Axis) (left: float) (plotW: float) (y: float) =
        let mutable lastEnd = Double.NegativeInfinity

        for t, label in a.Ticks do
            let x = left + plotW * tickPos a t
            let half = TextMetrics.width 11.0 label / 2.0

            if x - half > lastEnd + 6.0 then
                w.Text(
                    "text",
                    [
                        "class", "t3"
                        "x", n x
                        "y", n y
                        "font-size", "11"
                        "text-anchor", "middle"
                    ],
                    label
                )

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

    let private refLines
        (w: SvgWriter)
        (a: Axis)
        (refs: RefLine list)
        (left: float)
        (right: float)
        (top: float)
        (plotH: float)
        =
        for r in refs do
            if Double.IsFinite r.Value && (not a.Log || r.Value > 0.0) then
                let p = pos a r.Value

                if p >= -0.001 && p <= 1.001 then
                    let y = top + plotH * (1.0 - p)

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

    let private renderLine (id: string) (c: LineChart) =
        let ok (x: float, y: float) =
            Double.IsFinite x && Double.IsFinite y && (not c.YLog || y > 0.0)

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

            let ys = (all |> List.map snd) @ refYs

            let xa =
                linearAxis c.XFormat (List.min (xs @ bandXs)) (List.max (xs @ bandXs)) 6 c.IntegerX true

            let ya =
                if c.YLog then
                    logAxis (List.min ys) (List.max ys)
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
            let py (v: float) = top + plotH * (1.0 - pos ya v)

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

            let left = 16.0 + maxTickWidth ya + 10.0

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

            yGrid w ya left (left + plotW) top plotH
            refLines w ya c.RefLines left (left + plotW) top plotH
            xTicks w xa left plotW (top + plotH + 17.0)
            baseline w left (left + plotW) (top + plotH)
            axisLabel w left (top - 10.0) "start" c.YLabel
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel

            for s in c.Series do
                let sb = Text.StringBuilder()
                let mutable pen = false

                for p in s.Points do
                    if ok p then
                        sb.Append(if pen then " L" else " M").Append(n (px (fst p))).Append(' ').Append(n (py (snd p)))
                        |> ignore

                        pen <- true
                    else
                        pen <- false

                if sb.Length > 0 then
                    w.LeafWithTitle(
                        "path",
                        [ "class", "line " + Color.stroke s.Color; "d", sb.ToString().TrimStart() ],
                        s.Name
                    )

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
                            s.Name + ", " + c.XLabel + " " + format c.XFormat x + ": " + format c.YFormat y
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
            | Some v when Double.IsFinite v && v > 0.0 -> v
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

                for t, label in va.Ticks do
                    let x = left + plotW * tickPos va t

                    w.Leaf(
                        "line",
                        [
                            "class", "grid"
                            "x1", n x
                            "y1", n top
                            "x2", n x
                            "y2", n (top + plotH)
                            "stroke-width", "1"
                        ]
                    )

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
                let py (v: float) = top + plotH * (1.0 - pos va v)
                openSvg w id height c.Title c.Subtitle desc
                drawLegend w BoxKey (headerHeight c.Subtitle + 6.0) legend
                yGrid w va left (left + plotW) top plotH
                axisLabel w left (top - 10.0) "start" c.ValueLabel
                axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.CategoryLabel

                let labelW =
                    c.Categories |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0

                let every = max 1 (int (Math.Ceiling((labelW + 6.0) / band)))

                c.Categories
                |> List.iteri (fun i cat ->
                    let cx = left + band * (float i + 0.5)

                    if i % every = 0 then
                        w.Text(
                            "text",
                            [
                                "class", "t3"
                                "x", n cx
                                "y", n (top + plotH + 17.0)
                                "font-size", "11"
                                "text-anchor", "middle"
                            ],
                            cat
                        )

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

        let usable (v: float) =
            Double.IsFinite v && (not c.Log || v > 0.0)

        let cells =
            [
                for i in 0 .. nr - 1 do
                    for j in 0 .. nc - 1 do
                        if i < c.Values.Length && j < c.Values[i].Length && usable (c.Values[i][j]) then
                            yield c.Values[i][j]
            ]

        if cells.IsEmpty then
            renderEmpty id c.Title "no finite cells"
        else
            let tr (v: float) = if c.Log then DetMath.log10 v else v
            let lo = cells |> List.map tr |> List.min
            let hi = cells |> List.map tr |> List.max

            let step (v: float) =
                if hi <= lo then
                    Theme.steps / 2
                else
                    min (Theme.steps - 1) (int (Math.Floor((tr v - lo) / (hi - lo) * float Theme.steps)))

            let rowW = c.Rows |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0
            let left = 16.0 + min 200.0 rowW + 10.0
            let right = 16.0
            let top = headerHeight c.Subtitle + 22.0 + 22.0
            let plotW = Width - left - right
            let cellW = plotW / float nc
            let cellH = max 8.0 (min 26.0 (300.0 / float nr))
            let plotH = cellH * float nr
            let bottom = 46.0
            let height = top + plotH + bottom

            let desc =
                Invariant.format
                    $"Heatmap of {nr} {c.RowLabel} rows by {nc} {c.ColLabel} columns, {cells.Length} cells with values."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            // Colour key: the lowest and highest value with the ramp between them.
            let keyY = headerHeight c.Subtitle + 2.0

            let lowText = format c.ValueFormat (cells |> List.min)

            let highText = format c.ValueFormat (cells |> List.max)

            w.Text("text", [ "class", "t3"; "x", "16"; "y", n (keyY + 9.0); "font-size", "11" ], lowText)

            let kx = 16.0 + TextMetrics.width 11.0 lowText + 6.0

            for k in 0 .. Theme.steps - 1 do
                w.Leaf(
                    "rect",
                    [
                        "class", "q" + Invariant.i32 k
                        "x", n (kx + float k * 14.0)
                        "y", n keyY
                        "width", "13"
                        "height", "10"
                    ]
                )

            w.Text(
                "text",
                [
                    "class", "t3"
                    "x", n (kx + float Theme.steps * 14.0 + 5.0)
                    "y", n (keyY + 9.0)
                    "font-size", "11"
                ],
                highText + (if c.Log then " (log scale)" else "")
            )

            axisLabel w left (top - 8.0) "start" c.RowLabel

            c.Rows
            |> List.iteri (fun i label ->
                let every = max 1 (int (Math.Ceiling(12.0 / cellH)))

                if i % every = 0 then
                    w.Text(
                        "text",
                        [
                            "class", "t3"
                            "x", n (left - 6.0)
                            "y", n (top + cellH * float i + cellH / 2.0 + 4.0)
                            "font-size", "11"
                            "text-anchor", "end"
                        ],
                        TextMetrics.fit 11.0 200.0 label
                    )

                for j in 0 .. nc - 1 do
                    let v =
                        if i < c.Values.Length && j < c.Values[i].Length then
                            c.Values[i][j]
                        else
                            nan

                    let cls = if usable v then "q" + Invariant.i32 (step v) else "qx"
                    let gap = if cellW > 4.0 then 1.0 else 0.0

                    w.LeafWithTitle(
                        "rect",
                        [
                            "class", cls
                            "x", n (left + cellW * float j)
                            "y", n (top + cellH * float i)
                            "width", n (cellW - gap)
                            "height", n (cellH - 1.0)
                        ],
                        label
                        + ", "
                        + c.ColLabel
                        + " "
                        + c.Cols[j]
                        + ": "
                        + (if usable v then format c.ValueFormat v else "no data")
                    ))

            let colW = c.Cols |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0
            let every = max 1 (int (Math.Ceiling((colW + 6.0) / cellW)))

            c.Cols
            |> List.iteri (fun j label ->
                if j % every = 0 then
                    w.Text(
                        "text",
                        [
                            "class", "t3"
                            "x", n (left + cellW * (float j + 0.5))
                            "y", n (top + plotH + 16.0)
                            "font-size", "11"
                            "text-anchor", "middle"
                        ],
                        label
                    ))

            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.ColLabel
            closeSvg w

    // ---- histogram --------------------------------------------------------------------------------------------------

    let private renderHistogram (id: string) (c: Histogram) =
        let bins =
            c.Bins
            |> List.filter (fun b -> Double.IsFinite b.Lo && Double.IsFinite b.Hi && b.Hi > b.Lo)

        let total = bins |> List.sumBy _.Count

        if bins.IsEmpty || total = 0 then
            renderEmpty id c.Title "no samples"
        else
            let lo = (List.head bins).Lo
            let hi = (List.last bins).Hi
            let marks = c.Marks |> List.filter (fun m -> Double.IsFinite m.At)
            let xLo = marks |> List.fold (fun m v -> min m v.At) lo
            let xHi = marks |> List.fold (fun m v -> max m v.At) hi
            let xa = linearAxis c.XFormat xLo xHi 6 (c.XFormat = NumFormat.Pow10) true
            let peak = bins |> List.fold (fun m b -> max m b.Count) 0
            let ya = linearAxis NumFormat.Int 0.0 (float peak) 4 true false
            let height = 320.0
            let top = headerHeight c.Subtitle + 22.0
            let bottom = 46.0
            let plotH = height - top - bottom
            let left = 16.0 + maxTickWidth ya + 10.0
            let right = 24.0
            let plotW = Width - left - right
            let px (v: float) = left + plotW * pos xa v
            let py (v: float) = top + plotH * (1.0 - pos ya v)

            let desc =
                Invariant.format $"Histogram of {c.XLabel}: {total} samples in {bins.Length} bins."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            yGrid w ya left (left + plotW) top plotH
            xTicks w xa left plotW (top + plotH + 17.0)
            axisLabel w left (top - 10.0) "start" "count"
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel

            for b in bins do
                if b.Count > 0 then
                    let x1 = px b.Lo
                    let x2 = px b.Hi
                    let h = top + plotH - py (float b.Count)
                    let wd = max 1.0 (x2 - x1 - 2.0)

                    w.LeafWithTitle(
                        "path",
                        [
                            "class", Color.fill c.Color
                            "d", barPath false (x1 + 1.0) (top + plotH - h) wd h true
                        ],
                        "["
                        + format c.XFormat b.Lo
                        + ", "
                        + format c.XFormat b.Hi
                        + "): "
                        + Invariant.i32 b.Count
                    )

            for m in marks do
                let x = px m.At

                w.LeafWithTitle(
                    "line",
                    [
                        "class", "ref"
                        "x1", n x
                        "y1", n top
                        "x2", n x
                        "y2", n (top + plotH)
                        "stroke-width", "1"
                    ],
                    m.Label
                )

                w.Text("text", [ "class", "t2"; "x", n (x + 4.0); "y", n (top + 11.0); "font-size", "10" ], m.Label)

            baseline w left (left + plotW) (top + plotH)
            closeSvg w

    // ---- timeline ---------------------------------------------------------------------------------------------------

    let private renderTimeline (id: string) (c: Timeline) =
        let intervals =
            c.Intervals
            |> List.filter (fun i ->
                Double.IsFinite i.Start
                && Double.IsFinite i.Length
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
            let top = headerHeight c.Subtitle + legendHeight c.Legend + 26.0
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

                w.Leaf(
                    "line",
                    [
                        "class", "grid"
                        "x1", n x
                        "y1", n top
                        "x2", n x
                        "y2", n (top + plotH)
                        "stroke-width", "1"
                    ]
                )

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

            for iv in intervals do
                let x1 = px iv.Start
                let wd = max 1.0 (px (iv.Start + iv.Length) - x1 - 1.0)
                let y = top + laneH * float iv.Lane + (laneH - thick) / 2.0
                let r = min 3.0 (wd / 2.0)

                w.LeafWithTitle(
                    "rect",
                    [
                        "class", Color.fill iv.Color
                        "x", n x1
                        "y", n y
                        "width", n wd
                        "height", n thick
                        "rx", n r
                    ],
                    iv.Tip
                )

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

            // Labels go on top of every marker line. One that would run into the previous label is left to the
            // marker's tooltip.
            let mutable labelEnd = Double.NegativeInfinity

            for m in marks do
                let x = px m.At
                let labelW = TextMetrics.width 10.0 m.Label
                let anchorEnd = x + 8.0 + labelW > Width - 8.0
                let x0 = if anchorEnd then x - 8.0 - labelW else x + 8.0

                if x0 > labelEnd + 4.0 then
                    labelEnd <- x0 + labelW

                    w.Text(
                        "text",
                        [
                            "class", "t2"
                            "x", n (if anchorEnd then x - 8.0 else x + 8.0)
                            "y", n (top - 6.0)
                            "font-size", "10"
                            "text-anchor", (if anchorEnd then "end" else "start")
                        ],
                        m.Label
                    )

            xTicks w xa left plotW (top + plotH + 17.0)
            baseline w left (left + plotW) (top + plotH)
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel
            closeSvg w

    // ---- error bars -------------------------------------------------------------------------------------------------

    let private renderErrorBars (id: string) (c: ErrorBarChart) =
        let ok (p: ErrorPoint) =
            [ p.Low; p.Mid; p.High ]
            |> List.forall (fun v -> Double.IsFinite v && (not c.YLog || v > 0.0))
            && p.Index >= 0
            && p.Index < c.Categories.Length

        let pts = c.Series |> List.collect (fun s -> s.Spreads |> List.filter ok)

        if pts.IsEmpty then
            renderEmpty id c.Title "no finite points"
        else
            let refYs =
                c.RefLines
                |> List.map _.Value
                |> List.filter (fun v -> Double.IsFinite v && (not c.YLog || v > 0.0))

            let ys = (pts |> List.collect (fun p -> [ p.Low; p.High ])) @ refYs

            let ya =
                if c.YLog then
                    logAxis (List.min ys) (List.max ys)
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
            let left = 16.0 + maxTickWidth ya + 10.0
            let right = 24.0
            let plotW = Width - left - right
            let ncat = c.Categories.Length
            let band = plotW / float ncat
            let py (v: float) = top + plotH * (1.0 - pos ya v)
            let nser = c.Series.Length
            let dodge = min 9.0 (band / float (nser + 1))

            let desc =
                Invariant.format
                    $"Error bars of {c.YLabel} per {c.XLabel}: {nser} series over {ncat} categories, whiskers from low to high."

            let w = SvgWriter()
            openSvg w id height c.Title c.Subtitle desc
            drawLegend w DotKey (headerHeight c.Subtitle + 6.0) legend
            yGrid w ya left (left + plotW) top plotH
            refLines w ya c.RefLines left (left + plotW) top plotH
            axisLabel w left (top - 10.0) "start" c.YLabel
            axisLabel w (left + plotW / 2.0) (height - 10.0) "middle" c.XLabel

            let labelW =
                c.Categories |> List.fold (fun m l -> max m (TextMetrics.width 11.0 l)) 0.0

            let every = max 1 (int (Math.Ceiling((labelW + 6.0) / band)))

            c.Categories
            |> List.iteri (fun i cat ->
                if i % every = 0 then
                    w.Text(
                        "text",
                        [
                            "class", "t3"
                            "x", n (left + band * (float i + 0.5))
                            "y", n (top + plotH + 17.0)
                            "font-size", "11"
                            "text-anchor", "middle"
                        ],
                        cat
                    ))

            c.Series
            |> List.iteri (fun j s ->
                let offset = (float j - float (nser - 1) / 2.0) * dodge

                for p in s.Spreads |> List.filter ok do
                    let x = left + band * (float p.Index + 0.5) + offset
                    let yl = py p.Low
                    let yh = py p.High

                    let tip =
                        s.Name
                        + ", "
                        + c.XLabel
                        + " "
                        + c.Categories[p.Index]
                        + ": "
                        + format c.YFormat p.Mid
                        + " ("
                        + format c.YFormat p.Low
                        + " to "
                        + format c.YFormat p.High
                        + ")"

                    w.Leaf(
                        "path",
                        [
                            "class", Color.stroke s.Color
                            "d", Invariant.format $"M{n x} {n yl}V{n yh}M{n (x - 3.0)} {n yl}h6M{n (x - 3.0)} {n yh}h6"
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
                    ))

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
    /// charts can be inlined on one page. The same chart always renders to the same bytes, under any culture.
    let render (id: string) (c: Chart) : string =
        match c with
        | Chart.Line x -> renderLine id x
        | Chart.Bar x -> renderBar id x
        | Chart.Heatmap x -> renderHeatmap id x
        | Chart.Histogram x -> renderHistogram id x
        | Chart.Timeline x -> renderTimeline id x
        | Chart.ErrorBars x -> renderErrorBars id x
        | Chart.Empty(t, m) -> renderEmpty id t m
