/// Charts and builders at the edges: extreme magnitudes, exact zeros on log scales, NaN and infinite values, long
/// runs, and text that XML cannot hold.
module Lrm.Telemetry.Tests.ChartEdgeTests

open System
open System.Globalization
open System.IO
open System.Text.RegularExpressions
open System.Xml
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Telemetry.Tests.Common
open Lrm.Telemetry.Tests.ReportTests

// ---- chart shorthands -----------------------------------------------------------------------------------------------

let private line (title: string) (ylog: bool) (points: (float * float) list) =
    Chart.Line
        {
            Title = title
            Subtitle = ""
            XLabel = "x"
            YLabel = "y"
            XFormat = NumFormat.Plain
            YFormat = NumFormat.Plain
            YLog = ylog
            YFromZero = false
            IntegerX = false
            Series =
                [
                    {
                        Name = "s"
                        Color = Color.Cat 0
                        Points = points
                    }
                ]
            RefLines = []
            Bands = []
        }

let private bars (values: float list) =
    Chart.Bar
        {
            Title = "bars"
            Subtitle = ""
            CategoryLabel = "c"
            ValueLabel = "v"
            ValueFormat = NumFormat.Plain
            Categories = values |> List.mapi (fun i _ -> "c" + string i)
            Series =
                [
                    {
                        Name = "s"
                        Color = Color.Cat 0
                        Values = values
                    }
                ]
            Stacked = true
            Horizontal = false
        }

let private histogram (format: NumFormat) (bins: HistBin list) (zeros: int) (invalid: int) =
    Chart.Histogram
        {
            Title = "hist"
            Subtitle = ""
            XLabel = "x"
            XFormat = format
            Bins = bins
            Zeros = zeros
            Invalid = invalid
            Color = Color.Cat 0
            Marks = [ { At = 0.0; Label = "bound" } ]
        }

let private point (i: int) (lo: float) (mid: float) (hi: float) =
    {
        Index = i
        Low = lo
        Mid = mid
        High = hi
        Invalid = 0
    }

let private errorBars (ylog: bool) (points: ErrorPoint list) =
    Chart.ErrorBars
        {
            Title = "errors"
            Subtitle = ""
            XLabel = "layer"
            YLabel = "y"
            YFormat = NumFormat.Sci
            YLog = ylog
            Categories = [ for i in 0..3 -> string i ]
            Series =
                [
                    {
                        Name = "s"
                        Color = Color.Cat 0
                        Spreads = points
                    }
                ]
            RefLines = []
        }

let private heat (log: bool) (ceiling: (float * string) option) (values: float option[][]) (cols: float list) =
    Chart.Heatmap
        {
            Title = "heat"
            Subtitle = ""
            RowLabel = "row"
            ColLabel = "step"
            Rows = [ for i in 0 .. values.Length - 1 -> "r" + string i ]
            Cols = [ for j in 0 .. values[0].Length - 1 -> string j ]
            ColValues = cols
            Values = values
            Log = log
            ValueFormat = NumFormat.Sci
            Ceiling = ceiling
            Aggregate = HeatAggregate.Max
        }

let private timeline (intervals: Interval list) (marks: TimeMark list) =
    Chart.Timeline
        {
            Title = "trace"
            Subtitle = ""
            XLabel = "time"
            XFormat = NumFormat.Duration
            Lanes = [ "a"; "b" ]
            Intervals = intervals
            Marks = marks
            Legend = []
        }

let private interval (lane: int) (start: float) (length: float) (color: Color) =
    {
        Lane = lane
        Start = start
        Length = length
        Color = color
        Tip = "tip"
    }

let private count (pattern: string) (text: string) = Regex.Matches(text, pattern).Count

/// Heatmap cells of a class: cells carry a tooltip, the colour key's swatches do not.
let private cells (cls: string) (svg: string) =
    count ("<rect class=\"" + cls + "\"[^>]*><title>") svg

let private inv (s: string) =
    Double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture)

/// Parses the SVG and checks that every positioned element and every line path lies inside the chart's frame.
let private insideFrame (name: string) (svg: string) =
    let root = parseSvg name svg
    let width = inv (root.GetAttribute "width")
    let height = inv (root.GetAttribute "height")

    for node in root.SelectNodes "//*" do
        let el = node :?> XmlElement

        for attr in [ "x"; "x1"; "x2"; "cx"; "y"; "y1"; "y2"; "cy" ] do
            let v = el.GetAttribute attr

            if v <> "" then
                let limit = if attr.StartsWith "x" || attr = "cx" then width else height
                let f = inv v

                Expect.isTrue
                    (f >= -0.5 && f <= limit + 0.5)
                    (sprintf "%s: <%s %s=%s> outside the frame" name el.Name attr v)

        if el.Name = "path" && el.GetAttribute("class").StartsWith "line" then
            let numbers =
                Regex.Matches(el.GetAttribute "d", @"-?[0-9.]+")
                |> Seq.map (fun m -> inv m.Value)
                |> Array.ofSeq

            numbers
            |> Array.iteri (fun i f ->
                let limit = if i % 2 = 0 then width else height
                Expect.isTrue (f >= -0.5 && f <= limit + 0.5) (sprintf "%s: line point %g outside the frame" name f))

    root

/// Magnitudes that used to stop the whole report (the axis code overflowed or underflowed its span).
let private extremes: (string * Chart) list =
    [
        "line_pm_max", line "±max" false [ 0.0, -1e308; 1.0, 1e308 ]
        "line_x_max", line "x ±max" false [ -1e308, 0.0; 1e308, 1.0 ]
        "line_subnormal", line "subnormal" false [ 0.0, 5e-324; 1.0, 1e-323 ]
        "line_same_subnormal", line "same subnormal" false [ 0.0, 5e-324; 1.0, 5e-324 ]
        "line_log_max", line "log max" true [ 0.0, Double.MaxValue; 1.0, Double.MaxValue ]
        "line_log_subnormal", line "log subnormal" true [ 0.0, 5e-324; 1.0, 1e-300; 2.0, 1.0 ]
        "bar_subnormal", bars [ 5e-324 ]
        "bar_max", bars [ Double.MaxValue; Double.MaxValue ]
        "hist_wide", histogram NumFormat.Plain [ { Lo = -1e308; Hi = 1e308; Count = 3 } ] 0 0
        "hist_huge_counts",
        histogram
            NumFormat.Plain
            [
                {
                    Lo = 0.0
                    Hi = 1.0
                    Count = Int32.MaxValue
                }
                {
                    Lo = 1.0
                    Hi = 2.0
                    Count = Int32.MaxValue
                }
            ]
            0
            0
        "timeline_huge", timeline [ interval 0 1e308 1e307 Color.Good; interval 1 0.0 1e300 Color.Critical ] []
        "error_bars_huge", errorBars false [ point 0 -1e308 0.0 1e308 ]
    ]

// ---- an in-memory run log -------------------------------------------------------------------------------------------

let private manifest (kind: string) =
    {
        Schema = RunManifest.Schema
        RunId = "probe"
        Kind = kind
        Title = "probe"
        Status = RunStatus.Ok
        CreatedUtc = "2026-10-18T09:00:00.000Z"
        EndedUtc = None
        Node = "n0"
        Synthetic = false
        GitSha = None
        ExecutionClass = None
        Driver = None
        Params = []
    }

let private memoryLog (kind: string) (events: RunEvent list) =
    {
        Directory = ""
        Manifest = manifest kind
        Nodes =
            [
                {
                    Node = "n0"
                    File = ""
                    Role = "creator"
                    ExecutionClass = None
                    Records = events.Length
                    Ended = true
                    Status = Some RunStatus.Ok
                    Dropped = 0L
                    Invalid = 0L
                    RestartOf = ""
                    ClockOffsetNs = 0L
                    Problems = []
                }
            ]
        Events = events
    }

let private gauge (seq: int64) (name: string) (value: float) (attrs: Attr list) =
    {
        Node = "n0"
        Seq = seq
        Kind = TelemetryKind.Gauge
        Name = name
        TimeNs = seq * 1000L
        DurationNs = 0L
        Value = value
        Unit = ""
        SpanId = 0L
        ParentId = 0L
        GpuStartNs = 0L
        Attrs = attrs
    }

/// A training log that records the loss, the learning rate and every layer's gradient norm on every step.
let trainingLog (steps: int) (layers: int) =
    let layerAttrs = Array.init layers (fun l -> Attr.int TelemetryKeys.Layer l)
    let stage = Attr.str TelemetryKeys.Stage "pretrain"
    let train = Attr.str "split" "train"
    let events = ResizeArray<RunEvent>(steps * (layers + 2))
    let mutable seq = 0L

    let add name value attrs =
        events.Add(gauge seq name value attrs)
        seq <- seq + 1L

    for s in 0 .. steps - 1 do
        let step = Attr.int "step" s
        add "train.loss" (4.0 / (1.0 + float s / 500.0)) [ step; train; stage ]
        add "train.lr" (3e-4 * (1.0 - float s / float steps)) [ step ]

        for l in 0 .. layers - 1 do
            add "train.grad_norm" (0.5 + 0.01 * float l + (if s = steps / 3 then 9.0 else 0.0)) [ step; layerAttrs[l] ]

    memoryLog "training" (List.ofSeq events)

/// A parity run with 12 rel_l2 comparisons, three of them failing: one 3x over its bound, one NaN, one infinite.
let private parityProbe (root: string) =
    use run =
        RunContext.Create(
            { RunOptions.create "parity" "probe" with
                RunsRoot = Some root
                RunId = Some(RunId.parse "parity-probe")
                Clock = Some(steppingClock 0L 1000L)
                FlushInterval = TimeSpan.Zero
            }
        )

    let emit (tap: string) (layer: int) (measured: float) =
        run.Telemetry.Check(
            TelemetryNames.GoldenCompare,
            measured <= 1e-5,
            [
                Attr.str "pair" "gpu-vs-cpu32"
                Attr.str TelemetryKeys.Case "c1"
                Attr.str "tap" ("L" + string layer + "." + tap)
                Attr.str "metric" "rel_l2"
                Attr.float "measured" measured
                Attr.float "bound" 1e-5
                Attr.int TelemetryKeys.Layer layer
            ]
        )

    for layer in 0..5 do
        for tap in [ "q_proj"; "down.out" ] do
            let measured =
                match tap, layer with
                | "q_proj", 2 -> 3e-5
                | "down.out", 3 -> nan
                | "q_proj", 4 -> infinity
                | "q_proj", 0 -> 0.0
                | _ -> 1e-7 * float (layer + 1)

            emit tap layer measured

    run.Finish RunStatus.Failed
    run.Directory

let tests =
    testList
        "charts at the edges"
        [
            testCase "extreme magnitudes render as charts, inside their frame"
            <| fun () ->
                for name, chart in extremes do
                    let svg = ChartSvg.render name chart
                    insideFrame name svg |> ignore
                    Expect.isFalse (svg.Contains "could not be drawn") (name + ": drawn")
                    Expect.isFalse (svg.Contains "No data:") (name + ": has data")

            testCase "every chart kind with awkward values stays inside its frame"
            <| fun () ->
                for name, chart in chartsOfEveryKind do
                    insideFrame name (ChartSvg.render name chart) |> ignore

            testCase "exact zeros on log scales stay visible"
            <| fun () ->
                let h = ChartSvg.render "h" (heat true None [| [| Some 0.0; Some 0.5 |] |] [])
                parseSvg "h" h |> ignore
                Expect.stringContains h "class=\"qz\"" "heatmap: an exact cell"
                Expect.stringContains h "0 (exact)" "heatmap: exact tooltip and key"
                Expect.isFalse (h.Contains "no data") "heatmap: zero is not missing"

                let e =
                    ChartSvg.render "e" (errorBars true [ point 0 0.0 0.0 0.0; point 1 0.0 1e-7 2e-7 ])

                parseSvg "e" e |> ignore
                Expect.isFalse (e.Contains "No data:") "error bars: drawn"
                Expect.equal (count "<circle" e) 2 "error bars: both points"
                Expect.stringContains e ">0</text>" "error bars: the zero strip"
                Expect.stringContains e "0 (exact)" "error bars: exact tooltip"

                let l =
                    ChartSvg.render "l" (line "log zero" true [ 0.0, 1e-7; 1.0, 0.0; 2.0, 1e-6 ])

                parseSvg "l" l |> ignore
                Expect.stringContains l "0 (exact)" "line: exact tooltip"
                let d = Regex.Match(l, "class=\"line s0\" d=\"([^\"]*)\"").Groups[1].Value
                Expect.equal (count " L" d) 2 "line: one path through all three points"

                let b = Builders.Common.bin 10 true [ 0.0; 0.0; -0.0; 1e-3; 1e-2; nan; -1.0 ]
                Expect.equal (b.Zeros, b.Invalid, b.Bins |> List.sumBy _.Count) (3, 2, 2) "bin counts"
                let hs = ChartSvg.render "hs" (histogram NumFormat.Pow10 b.Bins b.Zeros b.Invalid)
                parseSvg "hs" hs |> ignore
                Expect.stringContains hs "exactly 0: 3" "histogram: zero column"
                Expect.stringContains hs "NaN, infinite or out of range: 2" "histogram: invalid column"
                Expect.stringContains hs "7 samples" "histogram: desc counts every sample"

                let allZero = Builders.Common.bin 10 true [ 0.0; 0.0; 0.0; 0.0 ]

                let z =
                    ChartSvg.render "z" (histogram NumFormat.Pow10 allZero.Bins allZero.Zeros allZero.Invalid)

                Expect.isFalse (z.Contains "No data:") "histogram: only zeros is still a chart"

            testCase "a ratio heatmap ends its scale at the bound"
            <| fun () ->
                let passing =
                    ChartSvg.render "p" (heat true (Some(1.0, "over bound")) [| [| Some 0.001; Some 0.01 |] |] [])

                Expect.equal (cells "fcrit" passing) 0 "passing: no critical cell"
                Expect.stringContains passing "1.0e+00 (log scale)" "passing: the key ends at the bound"
                // 0.01 sits a third of the way up a ramp from 0.001 to 1: a light step, not the darkest.
                Expect.equal (cells "q4" passing) 1 "passing: 0.01 is step 4 of 13"
                Expect.equal (cells "q12" passing) 0 "passing: no cell at the darkest step"

                let failing =
                    ChartSvg.render "f" (heat true (Some(1.0, "over bound")) [| [| Some 0.01; Some 3.0 |] |] [])

                Expect.stringContains failing "3.0e+00 (over bound)" "failing: the cell says it is over"
                Expect.equal (cells "fcrit" failing) 1 "failing: a critical cell"

            testCase "NaN and infinite heatmap cells are crossed and critical, apart from missing ones"
            <| fun () ->
                let svg =
                    ChartSvg.render "n" (heat true None [| [| Some nan; Some infinity; None; Some 1e-3 |] |] [])

                parseSvg "n" svg |> ignore
                Expect.equal (count "NaN or infinite</title>" svg) 2 "two invalid cells"
                Expect.equal (count "no data</title>" svg) 1 "one missing cell"
                Expect.stringContains svg ">NaN/inf</text>" "key entry"
                Expect.stringContains svg ">no data</text>" "key entry for missing"

            testCase "wide heatmaps bin their columns and tick numeric columns nicely"
            <| fun () ->
                let values =
                    [| for r in 0..2 -> [| for j in 0..999 -> Some(1.0 + float ((r + j) % 17)) |] |]

                let svg = ChartSvg.render "w" (heat true None values [ for j in 0..999 -> float j ])
                parseSvg "w" svg |> ignore
                Expect.equal (cells "q[0-9]+" svg) (3 * 112) "112 bins of 9 columns per row"
                Expect.stringContains svg "columns bin 9 steps each (max)" "the subtitle says so"
                Expect.stringContains svg "step 990–998: max" "a bin's tooltip names its columns"
                Expect.stringContains svg "step 999: max" "the last bin holds one column"

                for tick in [ "200"; "400"; "600"; "800" ] do
                    Expect.stringContains svg (">" + tick + "</text>") ("tick " + tick)

            testCase "long line series are thinned and keep their extremes"
            <| fun () ->
                let points = [ for i in 0..19_999 -> float i, (if i = 12_345 then 1.0 else 0.0) ]
                let svg = ChartSvg.render "t" (line "spike" false points)
                let d = Regex.Match(svg, "class=\"line s0\" d=\"([^\"]*)\"").Groups[1].Value

                let ys =
                    Regex.Matches(d, @"[ML](-?[0-9.]+) (-?[0-9.]+)")
                    |> Seq.map (fun m -> inv m.Groups[2].Value)
                    |> Array.ofSeq

                Expect.isLessThan ys.Length 3000 "at most a few points per pixel column"
                Expect.equal (ys |> Array.distinct |> Array.length) 2 "the spike and the floor"
                Expect.stringContains svg "1 series, 20000 points" "the description counts every point"

            testCase "a 20,000-step training log renders small (bins and thinning)"
            <| fun () ->
                let o = Reports.render "training" (trainingLog 20_000 28)

                let size file =
                    (o.Svgs |> List.find (fun (f, _) -> f = file) |> snd).Length

                for file, svg in o.Svgs do
                    parseSvg file svg |> ignore

                Expect.isLessThan (size "training-grad_heatmap.svg") 600_000 "the gradient heatmap"
                Expect.isLessThan (size "training-loss.svg") 100_000 "the loss chart"
                Expect.isLessThan (size "training-lr.svg") 100_000 "the learning-rate chart"
                Expect.isLessThan (o.Svgs |> List.sumBy (fun (_, s) -> s.Length)) 1_500_000 "every chart together"
                Expect.stringContains o.Markdown "after 20,000 training steps" "the step count"

            testCase "parity lists NaN and infinite failures first and marks them in every chart"
            <| fun () ->
                use tmp = new TempDir()
                let o = Reports.render "parity" (RunLogReader.loadOrFail (parityProbe tmp.Path))
                let md = o.Markdown

                let svg file =
                    o.Svgs |> List.find (fun (f, _) -> f = file) |> snd

                Expect.stringContains
                    md
                    "**FAIL.** 3 of 12 comparisons exceed their bounds, 2 of them are NaN or infinite."
                    "verdict"

                Expect.stringContains md "## Failing comparisons" "the failure table"

                let at (tap: string) =
                    md.IndexOf("| gpu-vs-cpu32 | c1 | " + tap + " |", StringComparison.Ordinal)

                Expect.isGreaterThan (at "L3.down.out") 0 "the NaN failure is listed"
                Expect.isGreaterThan (at "L4.q_proj") (at "L3.down.out") "inf after NaN"
                Expect.isGreaterThan (at "L2.q_proj") (at "L4.q_proj") "the finite failure after both"
                Expect.stringContains md "| NaN | 1.00e-05 | NaN | ✗ FAIL |" "the NaN row"
                Expect.stringContains md "| inf | 1.00e-05 | inf | ✗ FAIL |" "the inf row"

                Expect.stringContains
                    md
                    "Closest to its bound: gpu-vs-cpu32 c1 L3.down.out rel_l2: measured NaN"
                    "worst"

                Expect.stringContains md "worst ratio NaN of the bound" "per-pair worst"

                let h = svg "parity-pairs-0-heatmap.svg"
                Expect.equal (count "NaN or infinite</title>" h) 2 "heatmap: two invalid cells"
                Expect.stringContains h "(over bound)</title>" "heatmap: the finite failure is over the bound"
                Expect.stringContains h "0 (exact)</title>" "heatmap: the exact match"

                let e = svg "parity-rel_l2_by_layer.svg"
                Expect.equal (count "class=\"scrit\"" e) 2 "error bars: a cross for layers 3 and 4"
                Expect.stringContains e "layer 3: 1 NaN or infinite</title>" "error bars: the cross's tooltip"
                Expect.stringContains e ">NaN/inf</text>" "error bars: the NaN/inf strip"

                let r = svg "parity-ratio_histogram.svg"
                Expect.stringContains r "NaN, infinite or out of range: 2" "histogram: invalid column"
                Expect.stringContains r "exactly 0: 1" "histogram: exact column"

            testCase "SVG text holds only characters XML allows, and cuts never split a surrogate pair"
            <| fun () ->
                let bad = ChartSvg.render "x" (Chart.Empty("a￾b￿c\uD800d\uDC00e", "m"))
                parseSvg "x" bad |> ignore
                Expect.stringContains bad "a�b�c�d�e" "replaced with U+FFFD"

                for prefix in [ ""; "a" ] do
                    let long = ChartSvg.render "y" (Chart.Empty(prefix + String.replicate 300 "😀", "m"))
                    parseSvg "y" long |> ignore
                    Expect.isFalse (long.Contains "�") "no replacement character from a cut"
                    Expect.stringContains long "😀…" "cut between pairs"

            testCase "a render replaces the charts of the previous render of the same template"
            <| fun () ->
                use tmp = new TempDir()

                let output svgs : Reports.Output =
                    {
                        Template = "parity"
                        Markdown = "x\n"
                        Svgs = svgs
                    }

                let charts = Path.Combine(tmp.Path, "charts")

                Reports.write tmp.Path (output [ "parity-a.svg", "<svg/>"; "parity-b.svg", "<svg/>" ])
                |> ignore

                File.WriteAllText(Path.Combine(charts, "gate1-a.svg"), "<svg/>")
                File.WriteAllText(Path.Combine(charts, "parityx-a.svg"), "<svg/>")
                Reports.write tmp.Path (output [ "parity-a.svg", "<svg/>" ]) |> ignore

                Expect.equal
                    (Directory.GetFiles charts
                     |> Array.map Path.GetFileName
                     |> Array.sort
                     |> List.ofArray)
                    [ "gate1-a.svg"; "parity-a.svg"; "parityx-a.svg" ]
                    "only the stale chart of this template is gone"

            testCase "timeline outcomes to notice are at least 4 px wide and drawn over the markers"
            <| fun () ->
                let svg =
                    ChartSvg.render
                        "tl"
                        (timeline
                            [ interval 0 0.0 10e6 Color.Good; interval 1 2e6 0.0 Color.Critical ]
                            [
                                {
                                    At = 2e6
                                    Label = "step"
                                    Color = Color.Neutral
                                }
                            ])

                let crit = Regex.Match(svg, "<rect class=\"fcrit emph\"[^>]* width=\"([0-9.]+)\"")
                Expect.isTrue crit.Success "an outlined critical interval"
                Expect.isGreaterThanOrEqual (inv crit.Groups[1].Value) 4.0 "at least 4 px"

                Expect.isGreaterThan
                    crit.Index
                    (svg.IndexOf("<line class=\"sn\"", StringComparison.Ordinal))
                    "over the marker"

            testCase "timeline marker labels stagger instead of disappearing"
            <| fun () ->
                let mark at label =
                    {
                        At = at
                        Label = label
                        Color = Color.Neutral
                    }

                let svg =
                    ChartSvg.render
                        "st"
                        (timeline [ interval 0 0.0 10e6 Color.Good ] [ mark 1e6 "revise"; mark 1.1e6 "revise again" ])

                Expect.stringContains svg ">revise</text>" "first label"
                Expect.stringContains svg ">revise again</text>" "second label, on the second row"

            testCase "decade axes label each decade in one style and in its unit"
            <| fun () ->
                let bins lo hi =
                    [
                        for k in lo .. hi - 1 ->
                            {
                                Lo = float k
                                Hi = float (k + 1)
                                Count = 1
                            }
                    ]

                let d =
                    ChartSvg.render "d" (histogram (NumFormat.Decades NumFormat.Duration) (bins 3 7) 0 0)

                for tick in [ "1 µs"; "10 µs"; "100 µs"; "1 ms"; "10 ms" ] do
                    Expect.stringContains d (">" + tick + "</text>") tick

                let wide = ChartSvg.render "p" (histogram NumFormat.Pow10 (bins -6 0) 0 0)

                for tick in [ "10⁻⁶"; "10⁻³"; "10⁰" ] do
                    Expect.stringContains wide (">" + tick + "</text>") tick

                Expect.isFalse (wide.Contains ">0.01</text>") "one style per axis"

                let narrow = ChartSvg.render "q" (histogram NumFormat.Pow10 (bins -2 1) 0 0)

                for tick in [ "0.01"; "0.1"; "1"; "10" ] do
                    Expect.stringContains narrow (">" + tick + "</text>") tick

            testCase "the samples exercise every column and series they show"
            <| fun () ->
                let render kind = Reports.render kind (syntheticLog kind)
                let cluster = render "cluster"

                let nodeRows =
                    cluster.Markdown.Split('\n')
                    |> Array.filter (fun l -> Regex.IsMatch(l, "^\\| s[01][ab] \\||^\\| spare \\|"))

                Expect.isNonEmpty nodeRows "node rows"

                for row in nodeRows do
                    Expect.isFalse (row.EndsWith "| 0 B |") ("bytes sent: " + row)

                let inference = render "inference"
                Expect.isFalse (inference.Markdown.Contains "| 0 ns |") "GPU time in the metering table"

                let routing = render "routing"

                let byLayer =
                    routing.Svgs
                    |> List.find (fun (f, _) -> f = "routing-entropy_by_layer.svg")
                    |> snd

                for step in [ 0; 500; 1000; 1500; 2000; 2500 ] do
                    Expect.stringContains byLayer (">step " + string step + "</title>") ("checkpoint " + string step)

                let training = render "training"
                let lr = training.Svgs |> List.find (fun (f, _) -> f = "training-lr.svg") |> snd
                Expect.isFalse (lr.Contains "No data:") "the learning rate is charted"
                Expect.stringContains training.Markdown "at step 599, after 600 training steps" "steps"
        ]
