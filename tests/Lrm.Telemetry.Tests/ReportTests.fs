module Lrm.Telemetry.Tests.ReportTests

open System
open System.IO
open System.Xml
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Telemetry.Tests.Common

// ---- fixtures -------------------------------------------------------------------------------------------------------

/// Log-space bins over awkward samples: exact zeros, a negative, a NaN.
let private histogramSample =
    Builders.Common.bin 12 true [ 1e-6; 2e-6; 1e-4; 0.5; 0.0; 0.0; -1.0; nan; 3e-3 ]

/// One chart of every kind, with awkward values mixed in (NaN, infinities, zeros and negatives on log axes).
let chartsOfEveryKind: (string * Chart) list =
    [
        "line",
        Chart.Line
            {
                Title = "Loss & <friends>"
                Subtitle = "A \"quoted\" subtitle"
                XLabel = "step"
                YLabel = "loss"
                XFormat = NumFormat.Int
                YFormat = NumFormat.Plain
                YLog = false
                YFromZero = false
                IntegerX = false
                Series =
                    [
                        {
                            Name = "train"
                            Color = Color.Cat 0
                            Points = [ for i in 0..99 -> float i, 4.0 / (1.0 + float i * 0.1) ] @ [ 100.0, nan ]
                        }
                        {
                            Name = "eval"
                            Color = Color.Cat 1
                            Points = [ 10.0, 3.0; 50.0, infinity; 90.0, 1.0 ]
                        }
                    ]
                RefLines = [ { Value = 1.5; Label = "target" } ]
                Bands =
                    [
                        {
                            From = 0.0
                            To = 40.0
                            Label = "warmup"
                        }
                    ]
            }
        "per_layer_log",
        Chart.Line
            {
                Title = "Per layer"
                Subtitle = ""
                XLabel = "layer"
                YLabel = "rel_l2"
                XFormat = NumFormat.Int
                YFormat = NumFormat.Sci
                YLog = true
                YFromZero = false
                IntegerX = true
                Series =
                    [
                        for s in 0..2 ->
                            {
                                Name = "s" + string s
                                Color = Color.slot s
                                Points =
                                    [ for l in 0..27 -> float l, (if l = 5 then 0.0 else 1e-7 * float (l + 1 + s)) ]
                            }
                    ]
                RefLines = [ { Value = 1e-5; Label = "bound" } ]
                Bands = []
            }
        "bars_stacked_columns",
        Chart.Bar
            {
                Title = "Columns"
                Subtitle = "stacked"
                CategoryLabel = "layer"
                ValueLabel = "time"
                ValueFormat = NumFormat.Duration
                Categories = [ for l in 0..27 -> string l ]
                Series =
                    [
                        for s in 0..3 ->
                            {
                                Name = "k" + string s
                                Color = Color.slot s
                                Values = [ for l in 0..27 -> float ((l + 1) * (s + 1)) * 1000.0 ]
                            }
                    ]
                Stacked = true
                Horizontal = false
            }
        "bars_grouped_horizontal",
        Chart.Bar
            {
                Title = "Bars"
                Subtitle = ""
                CategoryLabel = "arena"
                ValueLabel = "bytes"
                ValueFormat = NumFormat.Bytes
                Categories =
                    [
                        "weights"
                        "kv_cache"
                        "a very long arena name that will not fit in the label column at all"
                    ]
                Series =
                    [
                        {
                            Name = "n0"
                            Color = Color.Cat 0
                            Values = [ 1.5e9; 1.2e8; nan ]
                        }
                        {
                            Name = "n1"
                            Color = Color.Cat 1
                            Values = [ 1.4e9; -5.0 ]
                        }
                    ]
                Stacked = false
                Horizontal = true
            }
        "heatmap",
        Chart.Heatmap
            {
                Title = "Heat"
                Subtitle = ""
                RowLabel = "expert"
                ColLabel = "layer"
                Rows = [ "a"; "b"; "c" ]
                Cols = [ for l in 0..27 -> string l ]
                ColValues = []
                Values =
                    [|
                        for r in 0..2 ->
                            [|
                                for l in 0..27 ->
                                    if l = 3 then None
                                    elif l = 4 then Some 0.0
                                    elif l = 5 then Some nan
                                    elif l = 6 then Some -infinity
                                    else Some(float (r * 28 + l + 1))
                            |]
                    |]
                Log = true
                ValueFormat = NumFormat.Plain
                Ceiling = Some(60.0, "over 60")
                Aggregate = HeatAggregate.Max
            }
        "histogram",
        Chart.Histogram
            {
                Title = "Hist"
                Subtitle = ""
                XLabel = "ratio"
                XFormat = NumFormat.Pow10
                Bins = histogramSample.Bins
                Zeros = histogramSample.Zeros
                Invalid = histogramSample.Invalid
                Color = Color.Cat 0
                Marks = [ { At = 0.0; Label = "bound" } ]
            }
        "timeline",
        Chart.Timeline
            {
                Title = "Gantt"
                Subtitle = ""
                XLabel = "time"
                XFormat = NumFormat.Duration
                Lanes = [ "n0"; "n1" ]
                Intervals =
                    [
                        {
                            Lane = 0
                            Start = 0.0
                            Length = 5e6
                            Color = Color.Good
                            Tip = "ok"
                        }
                        {
                            Lane = 1
                            Start = 2e6
                            Length = 0.0
                            Color = Color.Critical
                            Tip = "lost"
                        }
                        {
                            Lane = 7
                            Start = 0.0
                            Length = 1.0
                            Color = Color.Good
                            Tip = "bad lane"
                        }
                    ]
                Marks =
                    [
                        {
                            At = 2e6
                            Label = "n1 lost"
                            Color = Color.Critical
                        }
                        {
                            At = 2.1e6
                            Label = "overlapping label"
                            Color = Color.Neutral
                        }
                    ]
                Legend = [ "ok", Color.Good; "lost", Color.Critical ]
            }
        "error_bars",
        Chart.ErrorBars
            {
                Title = "Errors"
                Subtitle = ""
                XLabel = "layer"
                YLabel = "rel_l2"
                YFormat = NumFormat.Sci
                YLog = true
                Categories = [ for l in 0..27 -> string l ]
                Series =
                    [
                        {
                            Name = "a"
                            Color = Color.Cat 0
                            Spreads =
                                [
                                    for l in 0..27 ->
                                        {
                                            Index = l
                                            Low = (if l = 3 then 0.0 else 1e-8)
                                            Mid = 1e-7
                                            High = (if l = 2 then nan else 1e-6)
                                            Invalid = (if l = 2 || l = 4 then 1 else 0)
                                        }
                                ]
                        }
                        {
                            Name = "b"
                            Color = Color.Cat 1
                            Spreads =
                                [
                                    {
                                        Index = 40
                                        Low = 1.0
                                        Mid = 1.0
                                        High = 1.0
                                        Invalid = 0
                                    }
                                ]
                        }
                    ]
                RefLines = [ { Value = 1e-5; Label = "bound" } ]
            }
        "empty_line",
        Chart.Line
            {
                Title = "Nothing"
                Subtitle = ""
                XLabel = "x"
                YLabel = "y"
                XFormat = NumFormat.Plain
                YFormat = NumFormat.Plain
                YLog = true
                YFromZero = false
                IntegerX = false
                Series =
                    [
                        {
                            Name = "s"
                            Color = Color.Cat 0
                            Points = [ 1.0, -1.0; 2.0, nan ]
                        }
                    ]
                RefLines = []
                Bands = []
            }
        "empty", Chart.Empty("No data", "this run has no x records")
    ]

/// Parses SVG text as XML and returns the root element.
let parseSvg (name: string) (svg: string) =
    let doc = XmlDocument()
    doc.XmlResolver <- null

    try
        doc.LoadXml svg
    with e ->
        failtestf "%s does not parse as XML: %s" name e.Message

    let root = doc.DocumentElement
    Expect.equal root.LocalName "svg" (name + ": root")
    Expect.equal root.NamespaceURI "http://www.w3.org/2000/svg" (name + ": namespace")
    Expect.isNotEmpty (root.GetAttribute "viewBox") (name + ": viewBox")
    Expect.equal root.FirstChild.LocalName "title" (name + ": title first")
    root

/// Synthetic runs, written once per test process and deleted when it exits.
let private syntheticRoot =
    lazy
        (let root =
            Path.Combine(Path.GetTempPath(), "lrm-telemetry-tests", "synthetic-" + Guid.NewGuid().ToString("N"))

         AppDomain.CurrentDomain.ProcessExit.Add(fun _ ->
             try
                 Directory.Delete(root, true)
             with _ ->
                 ())

         for k in Synthetic.kinds do
             Synthetic.write k root |> ignore

         root)

let syntheticLog (kind: string) =
    RunLogReader.loadOrFail (Path.Combine(syntheticRoot.Force(), "synthetic-" + kind))

let private repoSamples () =
    Path.Combine(Lrm.Testing.Env.repoRoot (), "docs", "reports", "samples")

let private regenerate =
    "regenerate them with: dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report --synthetic all --out docs/reports/samples"

let tests =
    testList
        "reports"
        [
            testCase "DetMath agrees with the platform functions to about 1e-15"
            <| fun () ->
                for i in -300 .. 300 do
                    let x = 1.37 * DetMath.pow10 (i / 10) * (1.0 + float (i % 10) / 10.0)

                    if x > 0.0 && Double.IsFinite x then
                        let rel = abs (DetMath.ln x - Math.Log x) / max 1.0 (abs (Math.Log x))
                        Expect.isLessThan rel 1e-15 (sprintf "ln %g" x)

                for i in -700 .. 700 do
                    let x = float i * 0.999

                    if x < 709.0 then
                        let e = Math.Exp x
                        Expect.isLessThan (abs (DetMath.exp x - e) / e) 4e-15 (sprintf "exp %g" x)

                Expect.equal (DetMath.pow10 22) 1e22 "10^22 is exact"
                Expect.equal (DetMath.pow10 -3) (1.0 / 1000.0) "10^-3"
                Expect.equal (DetMath.floorLog10 999.9) 2 "floorLog10 999.9"
                Expect.equal (DetMath.floorLog10 1000.0) 3 "floorLog10 1000"
                Expect.equal (DetMath.floorLog10 0.00099) -4 "floorLog10 0.00099"
                Expect.isTrue (Double.IsNaN(DetMath.ln -1.0)) "ln of a negative"
                Expect.isLessThan (abs (DetMath.ln 5e-324 / Math.Log 5e-324 - 1.0)) 1e-15 "ln of the smallest subnormal"

            testCase "number formats are invariant and never print -0"
            <| fun () ->
                let check () =
                    [
                        Fmt.dp 2 3.14159
                        Fmt.dp 2 -0.001
                        Fmt.sci 2 0.000321
                        Fmt.num 1234567.0
                        Fmt.num 1234.4
                        Fmt.num 0.5
                        Fmt.num 0.0001
                        Fmt.int 1234567L
                        Fmt.si 1.25e9
                        Fmt.si 512.0
                        Fmt.bytes (1.5 * 1073741824.0)
                        Fmt.duration 3.4e6
                        Fmt.duration 800.0
                        Fmt.pct 0.1234
                        Fmt.num nan
                        Fmt.num -infinity
                    ]

                let expected =
                    [
                        "3.14"
                        "0.00"
                        "3.21e-04"
                        "1.23e+06"
                        "1,234"
                        "0.5000"
                        "1.00e-04"
                        "1,234,567"
                        "1.25G"
                        "512"
                        "1.50 GiB"
                        "3.40 ms"
                        "800 ns"
                        "12.3%"
                        "NaN"
                        "-inf"
                    ]

                Expect.equal (check ()) expected "invariant"
                Expect.equal (withCulture "de-DE" check) expected "de-DE"
                Expect.equal (withCulture "tr-TR" check) expected "tr-TR"

            testCase "every chart kind renders to valid SVG with a title and a description"
            <| fun () ->
                for name, chart in chartsOfEveryKind do
                    let svg = ChartSvg.render name chart
                    let root = parseSvg name svg

                    Expect.equal
                        (root.GetAttribute "aria-labelledby")
                        (name + "-title " + name + "-desc")
                        (name + ": labels")

                    Expect.isNotNull (root.SelectSingleNode "//*[local-name()='desc']") (name + ": desc")
                    // "NaN" may only appear in the labels of the NaN/inf strips and cells, never as a printed number.
                    let numbers =
                        svg.Replace("NaN/inf", "").Replace("NaN or infinite", "").Replace("NaN, infinite", "")

                    Expect.isFalse (numbers.Contains "NaN") (name + ": no NaN")
                    Expect.isFalse (svg.Contains "Infinity") (name + ": no Infinity")
                    Expect.isFalse (svg.Contains "could not be drawn") (name + ": drawn")
                    Expect.isTrue (svg.EndsWith "</svg>\n") (name + ": ends with a line break")

            testCase "charts are byte-identical when rendered twice and under de-DE and tr-TR"
            <| fun () ->
                for name, chart in chartsOfEveryKind do
                    let a = ChartSvg.render name chart
                    Expect.equal (ChartSvg.render name chart) a (name + ": twice")
                    Expect.equal (withCulture "de-DE" (fun () -> ChartSvg.render name chart)) a (name + ": de-DE")
                    Expect.equal (withCulture "tr-TR" (fun () -> ChartSvg.render name chart)) a (name + ": tr-TR")

            testCase "charts with nothing to plot render as a placeholder"
            <| fun () ->
                let svg =
                    ChartSvg.render "empty_line" (chartsOfEveryKind |> List.find (fst >> (=) "empty_line") |> snd)

                Expect.stringContains svg "no finite points" "placeholder text"
                Expect.stringContains svg "height=\"96\"" "placeholder size"

            testCase "every chart carries both themes and the categorical palette"
            <| fun () ->
                let svg = ChartSvg.render "line" (snd chartsOfEveryKind.Head)
                Expect.stringContains svg "@media (prefers-color-scheme:dark)" "dark theme"
                Expect.stringContains svg ".bg{fill:#fcfcfb}" "light surface"
                Expect.stringContains svg ".bg{fill:#1a1a19}" "dark surface"

                for c in Array.append Theme.categoricalLight Theme.categoricalDark do
                    Expect.stringContains svg c ("slot " + c)

            testCase "the template engine fills values, tables, charts and lists"
            <| fun () ->
                let text =
                    "{{! a comment line disappears }}\n# {{value title}}\n{{#each items}}\n- {{value name}} in {{value title}} {{chart c}}\n{{/each}}\n{{table t}}\n{{ chart c }}\n"

                let scope =
                    Scope.empty
                    |> Scope.value "title" "T"
                    |> Scope.table
                        "t"
                        {
                            Columns = [ "A", Align.Left; "B", Align.Right ]
                            Rows = [ [ "x|y"; "1" ]; [ "multi\nline" ] ]
                        }
                    |> Scope.chart "c" (Chart.Empty("C [1]", "none"))
                    |> Scope.list
                        "items"
                        [ Scope.empty |> Scope.value "name" "a"; Scope.empty |> Scope.value "name" "b" ]

                let r = Template.render "x" (Template.parse "x" text) scope

                Expect.equal
                    r.Markdown
                    "# T\n- a in T ![C 1](charts/x-items-0-c.svg)\n- b in T ![C 1](charts/x-items-1-c.svg)\n| A | B |\n| --- | ---: |\n| x\\|y | 1 |\n| multi line |  |\n![C 1](charts/x-c.svg)\n"
                    "markdown"

                Expect.equal
                    (r.Charts |> List.map fst)
                    [ "x-items-0-c.svg"; "x-items-1-c.svg"; "x-c.svg" ]
                    "chart files"

                Expect.equal
                    (Template.renderTable
                        {
                            Columns = [ "A", Align.Center ]
                            Rows = []
                        })
                    "_No rows._"
                    "empty table"

                Expect.equal
                    (Template.render
                        "x"
                        (Template.parse "x" "a\r\n{{#each e}}\r\nx\r\n{{/each}}\r\nb")
                        (Scope.empty |> Scope.list "e" []))
                        .Markdown
                    "a\nb"
                    "CRLF templates and an empty list"

            testCase "template errors name the template and the line"
            <| fun () ->
                let fails (text: string) (scope: Scope) (fragment: string) =
                    try
                        Template.render "t" (Template.parse "t" text) scope |> ignore
                        failtestf "no error for %s" text
                    with TemplateError msg ->
                        Expect.stringContains msg fragment text

                fails "a\n{{value missing}}" Scope.empty "t line 2: no value 'missing'"
                fails "{{table missing}}" Scope.empty "no table 'missing'"
                fails "{{chart missing}}" Scope.empty "no chart 'missing'"
                fails "{{#each missing}}{{/each}}" Scope.empty "no list 'missing'"
                fails "{{frobnicate x}}" Scope.empty "unknown directive 'frobnicate'"
                fails "{{value Bad}}" Scope.empty "not a valid name"
                fails "{{value}}" Scope.empty "exactly one name"
                fails "x\n\n{{#each l}}\nbody" Scope.empty "t line 3: {{#each l}} is never closed"
                fails "{{/each}}" Scope.empty "without {{#each}}"
                fails "{{value x" Scope.empty "unclosed {{"
                fails "{{! never closed" Scope.empty "unclosed comment"

            testCase "synthetic runs are byte-identical when written again and under de-DE"
            <| fun () ->
                use a = new TempDir()
                use b = new TempDir()

                for k in Synthetic.kinds do
                    Synthetic.write k a.Path |> ignore
                    withCulture "de-DE" (fun () -> Synthetic.write k b.Path |> ignore)

                let fa = filesUnder a.Path
                let fb = filesUnder b.Path
                Expect.equal (fa |> List.map fst) (fb |> List.map fst) "same files"

                for (path, x), (_, y) in List.zip fa fb do
                    Expect.equal (Lrm.Testing.Determinism.firstDifference x y) None path

            testCase "every synthetic run is a clean log of its kind"
            <| fun () ->
                for k in Synthetic.kinds do
                    let log = syntheticLog k
                    Expect.equal log.Manifest.Kind k "kind"
                    Expect.isTrue log.Manifest.Synthetic "synthetic"
                    Expect.isNonEmpty log.Events "records"

                    let expected =
                        if k = "cluster" then
                            // The killed node never writes its trailer; that is the point of the scenario.
                            [
                                "s0a: no run.end trailer (the node is still running, or it stopped without finishing)"
                            ]
                        else
                            []

                    Expect.equal log.Problems expected (k + ": problems")

            testCase "every template renders its synthetic run identically twice and under de-DE and tr-TR"
            <| fun () ->
                for t in Reports.templates do
                    let log = syntheticLog t.Name
                    let a = Reports.render t.Name log
                    Expect.equal (Reports.render t.Name log) a (t.Name + ": twice")
                    Expect.equal (withCulture "de-DE" (fun () -> Reports.render t.Name log)) a (t.Name + ": de-DE")
                    Expect.equal (withCulture "tr-TR" (fun () -> Reports.render t.Name log)) a (t.Name + ": tr-TR")
                    Expect.stringContains a.Markdown "Synthetic data." (t.Name + ": banner")
                    Expect.isFalse (a.Markdown.Contains "{{") (t.Name + ": no placeholder left")
                    Expect.isNonEmpty a.Svgs (t.Name + ": charts")

                    for file, svg in a.Svgs do
                        parseSvg file svg |> ignore
                        Expect.stringContains a.Markdown ("](charts/" + file + ")") (file + " is linked")
                        Expect.isFalse (svg.Contains "No data:") (file + " has data")

            testCase "every template renders a run with no records"
            <| fun () ->
                use tmp = new TempDir()

                for t in Reports.templates do
                    let dir =
                        use run =
                            RunContext.Create(
                                { RunOptions.create t.Name "empty" with
                                    RunsRoot = Some tmp.Path
                                    RunId = Some(RunId.parse ("empty-" + t.Name))
                                    Clock = Some(steppingClock 0L 1L)
                                }
                            )

                        run.Finish RunStatus.Ok
                        run.Directory

                    let o = Reports.render t.Name (RunLogReader.loadOrFail dir)

                    for file, svg in o.Svgs do
                        parseSvg file svg |> ignore
                        Expect.stringContains svg "No data:" (file + " says there is no data")

            testCase "the committed samples in docs/reports/samples are up to date"
            <| fun () ->
                let committed = repoSamples ()
                Expect.isTrue (Directory.Exists committed) (committed + " exists; " + regenerate)
                use runs = new TempDir()
                use out = new TempDir()
                Reports.renderSamples runs.Path out.Path |> ignore
                let fresh = filesUnder out.Path
                let have = filesUnder committed
                Expect.equal (have |> List.map fst) (fresh |> List.map fst) ("the same files; " + regenerate)

                for (path, x), (_, y) in List.zip have fresh do
                    Expect.equal (Lrm.Testing.Determinism.firstDifference x y) None (path + " differs; " + regenerate)

            testCase "every template is registered, embedded and parses"
            <| fun () ->
                Expect.equal
                    Reports.names
                    [
                        "parity"
                        "inference"
                        "kernels"
                        "cluster"
                        "gate1"
                        "training"
                        "routing"
                        "thinking"
                    ]
                    "templates"

                Expect.equal Synthetic.kinds Reports.names "a synthetic run for every template"

                for t in Reports.templates do
                    Template.parse t.Name (Reports.templateText t.Name) |> ignore
        ]
