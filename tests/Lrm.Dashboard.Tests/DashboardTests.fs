module Lrm.Dashboard.Tests.DashboardTests

open System
open System.IO
open System.Text
open System.Xml
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Dashboard
open Lrm.Dashboard.Tests.Common

/// Measured runs (three parity, one tokenizer) plus T1's synthetic runs, the page's usual mix.
let private mixed (tmp: TempDir) =
    let runs = tmp.Sub "runs"
    let synthetic = tmp.Sub "synthetic"

    writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
    |> ignore

    writeRun runs "r-b" "parity" "B" (unixNs 2026 10 18 11 0) RunStatus.Ok (parityBody 0.6)
    |> ignore

    writeRun runs "r-c" "parity" "C" (unixNs 2026 10 18 12 0) RunStatus.Failed (parityBody 3.0)
    |> ignore

    writeRun runs "r-t" "tokenizer" "T" (unixNs 2026 10 18 10 30) RunStatus.Ok recordsBody
    |> ignore

    writeSynthetic synthetic
    DashboardOptions.over [ RunsRoot.runs tmp.Path runs; RunsRoot.samples tmp.Path tmp.Path synthetic ]

let private regenerate =
    "regenerate it from the repo root with: dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll dashboard --synthetic "
    + "--out tests/Lrm.Dashboard.Tests/reports/samples.html"

/// The SVG a report writes for `file` (ChartSvg.render with the file stem), as the dashboard inlines it for a run:
/// ids prefixed with r-<run>- and the <style> element left out (the page carries it once).
let private inlined (runId: string) (file: string) (reportSvg: string) =
    let stem = Path.GetFileNameWithoutExtension file
    let id = RunView.chartId runId file

    reportSvg
        .Replace("<style>" + Theme.style + "</style>\n", "")
        .Replace(
            "aria-labelledby=\"" + stem + "-title " + stem + "-desc\"",
            "aria-labelledby=\"" + id + "-title " + id + "-desc\""
        )
        .Replace("id=\"" + stem + "-title\"", "id=\"" + id + "-title\"")
        .Replace("id=\"" + stem + "-desc\"", "id=\"" + id + "-desc\"")
        .TrimEnd('\n')

/// Every value of the id attribute, as a set, and the duplicates.
let private ids (html: string) =
    let all = attributeValues "id" html
    set all, all |> List.countBy id |> List.filter (fun (_, n) -> n > 1) |> List.map fst

let tests =
    testList
        "cpu.dashboard.page"
        [
            testCase "the same runs give byte-identical pages, twice and under de-DE and tr-TR"
            <| fun () ->
                use tmp = new TempDir()
                let o = mixed tmp
                let a = (Dashboard.build o).Bytes
                Expect.equal (Dashboard.build o).Bytes a "twice"
                Expect.equal (withCulture "de-DE" (fun () -> (Dashboard.build o).Bytes)) a "de-DE"
                Expect.equal (withCulture "tr-TR" (fun () -> (Dashboard.build o).Bytes)) a "tr-TR"
                Expect.equal (Dashboard.build { o with Threads = 1 }).Bytes a "one thread"

            testCase "the committed samples dashboard is byte-identical to a fresh render"
            <| fun () ->
                use tmp = new TempDir()
                writeSynthetic tmp.Path
                let fresh = (Dashboard.build (samplesOptions tmp.Path)).Bytes
                let path = samplesHtml ()
                Expect.isTrue (File.Exists path) ("no " + path + "; " + regenerate)

                Expect.isTrue
                    (File.ReadAllBytes path = fresh)
                    ("the committed samples dashboard is stale; " + regenerate)

            testCase "the samples dashboard is the same under de-DE"
            <| fun () ->
                use tmp = new TempDir()
                writeSynthetic tmp.Path
                let a = (Dashboard.build (samplesOptions tmp.Path)).Html

                Expect.equal
                    (withCulture "de-DE" (fun () -> (Dashboard.build (samplesOptions tmp.Path)).Html))
                    a
                    "de-DE"

            testCase "every run's report and chart is the one lrm report renders"
            <| fun () ->
                use tmp = new TempDir()
                let output = Dashboard.build (mixed tmp)

                for v in output.Model.Views do
                    match v.Template with
                    | None -> ()
                    | Some name ->
                        let report = Reports.render name (RunLogReader.loadOrFail v.Entry.Dir)
                        Expect.equal v.Markdown report.Markdown (v.Entry.Id + ": the report Markdown")
                        Expect.equal (v.Charts |> List.map fst) (report.Svgs |> List.map fst) (v.Entry.Id + ": charts")

                        for file, svg in report.Svgs do
                            Expect.stringContains
                                output.Html
                                (inlined v.Entry.Id file svg)
                                (v.Entry.Id + ": " + file + " is the report's chart")

            testCase "the chart theme is T1's, once, with overrides for both toggle states"
            <| fun () ->
                use tmp = new TempDir()
                let html = (Dashboard.build (mixed tmp)).Html
                Expect.equal (occurrences Theme.style html) 1 "Theme.style once"
                Expect.equal (occurrences "<style>" html) 2 "page CSS and chart CSS, no per-chart style"
                Expect.stringContains html ":root[data-theme=light] .d-chart .bg{fill:#fcfcfb}" "light override"
                Expect.stringContains html ":root[data-theme=dark] .d-chart .bg{fill:#1a1a19}" "dark override"

                Expect.stringContains
                    html
                    ":root[data-theme=dark] .d-chart .q0{fill:#0d366b}"
                    "dark ramp runs the other way"

                Expect.stringContains
                    html
                    "@media (prefers-color-scheme:dark){:root:not([data-theme=light])"
                    "page dark mode"

            testCase "the page is self-contained: nothing is fetched, no link leaves the machine"
            <| fun () ->
                use tmp = new TempDir()
                let html = (Dashboard.build (mixed tmp)).Html

                for forbidden in
                    [
                        "<link"
                        " src="
                        "@import"
                        "url("
                        "<iframe"
                        "<img"
                        "<object"
                        "<embed"
                        "fetch("
                        "XMLHttpRequest"
                        "//cdn"
                    ] do
                    Expect.isFalse (html.Contains forbidden) ("no " + forbidden)

                let namespaceUri = "http://www.w3.org/2000/svg"
                let withoutNamespace = html.Replace("xmlns=\"" + namespaceUri + "\"", "")
                Expect.isFalse (withoutNamespace.Contains "http:") "no http: outside the SVG namespace name"
                Expect.isFalse (withoutNamespace.Contains "https:") "no https:"

                for href in attributeValues "href" html do
                    Expect.isTrue (Html.isLocalHref href) ("a local href: " + href)

                // No absolute path either: the page's bytes never depend on where the folders are.
                Expect.isFalse (html.Contains "file:") "no file: URI"
                Expect.isFalse (html.Contains tmp.Path) "no absolute path"
                Expect.isFalse (html.Contains(tmp.Path.Replace('\\', '/'))) "no absolute path with slashes"

            testCase "ids are unique and every fragment link, compare slot and label resolves"
            <| fun () ->
                use tmp = new TempDir()
                let html = (Dashboard.build (mixed tmp)).Html
                let all, duplicates = ids html
                Expect.isEmpty duplicates "unique ids"

                for href in attributeValues "href" html do
                    if href.StartsWith "#" then
                        Expect.isTrue (all.Contains(Uri.UnescapeDataString(href.Substring 1))) ("target of " + href)

                let slots = attributeValues "data-src" html
                Expect.isNonEmpty slots "compare slots"

                for src in slots do
                    Expect.isTrue (all.Contains src) ("slot source " + src)

                for attribute in [ "aria-labelledby"; "aria-describedby" ] do
                    for labels in attributeValues attribute html do
                        for l in labels.Split ' ' do
                            Expect.isTrue (all.Contains l) (attribute + " " + l)

            testCase "every chart parses as XML, and so does every table"
            <| fun () ->
                use tmp = new TempDir()
                let html = (Dashboard.build (mixed tmp)).Html
                let mutable i = html.IndexOf("<svg ", StringComparison.Ordinal)
                let mutable n = 0

                while i >= 0 do
                    let stop = html.IndexOf("</svg>", i, StringComparison.Ordinal) + 6
                    let doc = XmlDocument()
                    doc.LoadXml(html.Substring(i, stop - i))
                    n <- n + 1
                    i <- html.IndexOf("<svg ", stop, StringComparison.Ordinal)

                Expect.isGreaterThan n 40 "charts parsed"

            testCase "the index lists every run with kind, status, start, duration, nodes and class"
            <| fun () ->
                use tmp = new TempDir()
                let output = Dashboard.build (mixed tmp)
                let html = output.Html

                let index =
                    html.Substring(html.IndexOf("<section class=\"d-view\" id=\"runs\"", StringComparison.Ordinal))

                Expect.equal (occurrences "<tr data-kind=" index) 13 "a row per run"
                Expect.equal (occurrences "<td" index) (13 * 8) "eight cells per row"

                for label in
                    [
                        "Started (UTC)"
                        "Run"
                        "Kind"
                        "Status"
                        "Duration"
                        "Nodes"
                        "Execution class"
                        "Title"
                    ] do
                    Expect.stringContains index (">" + label + "</th>") ("a " + label + " column")

                let row =
                    let i =
                        index.IndexOf("<tr data-kind=\"parity\" data-status=\"failed\">", StringComparison.Ordinal)

                    index.Substring(i, index.IndexOf("</tr>", i, StringComparison.Ordinal) - i)

                Expect.stringContains row "<td class=\"d-nw\">2026-10-18 12:00:00</td>" "start"
                Expect.stringContains row "<a href=\"#run-r-c\"><code>r-c</code></a>" "the run, linked to its view"
                Expect.stringContains row "<a href=\"#kind-parity\">parity</a>" "kind"
                Expect.stringContains row "<span class=\"d-st d-st-failed\">failed</span>" "status as text"
                Expect.stringContains row "<td class=\"d-r d-nw\">" "duration"
                Expect.stringContains row "<td class=\"d-r\">1</td>" "nodes"
                Expect.stringContains row "cpu-f32 <span class=\"d-muted\">dotnet-10</span>" "class"

            testCase
                "a view per template kind, Phase 2 and 3 included, and a placeholder for kinds without measured runs"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                let html =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])).Html

                for t in Reports.templates do
                    Expect.stringContains html ("id=\"kind-" + t.Name + "\"") (t.Name + " view")

                Expect.stringContains html "Training runs start in Phase 2" "phase 2 placeholder"
                Expect.stringContains html "in Phase 3 (Roslyn experts)" "phase 3 placeholder"
                Expect.stringContains html "No measured inference runs yet." "phase 1 placeholder"

            testCase "a run view has its report sections, metering totals and the link to its Markdown report"
            <| fun () ->
                use tmp = new TempDir()
                let output = Dashboard.build (mixed tmp)
                let html = output.Html
                let v = output.Model.Views |> List.find (fun v -> v.Entry.Id = "r-c")
                let flops = 2.0 * 2.0 * 2.0 * 2048.0
                Expect.equal v.Totals.Flops flops "FLOPs metered"

                let view =
                    let i =
                        html.IndexOf("<section class=\"d-view\" id=\"run-r-c\"", StringComparison.Ordinal)

                    html.Substring(i, html.IndexOf("</section>", i, StringComparison.Ordinal) - i)

                Expect.stringContains view ("<span>FLOPs</span><b>" + Fmt.si flops + "</b>") "FLOPs tile"
                Expect.stringContains view "<span>Wall (self)</span>" "wall tile"

                // Totals that are zero get no tile; one line names them.
                for zero in [ "GPU"; "Bytes moved"; "Tokens"; "Metered time"; "Log problems" ] do
                    Expect.isFalse (view.Contains("<span>" + zero + "</span>")) ("no tile for " + zero)

                Expect.stringContains
                    view
                    "<p class=\"d-muted d-small\">Nothing metered for GPU time, bytes moved, tokens, metered time.</p>"
                    "zero totals in one line"

                Expect.stringContains html "<a href=\"runs/r-c/report/parity.md\">parity.md</a>" "report link"
                Expect.stringContains html "<code>lrm report r-c</code>" "the command"
                Expect.stringContains html "<h3 id=\"run-r-c-error-by-layer\">Error by layer</h3>" "a report section"
                Expect.stringContains html "id=\"r-r-c-parity-rel_l2_by_layer-fig\"" "a report chart"
                Expect.stringContains html "<a href=\"parity.md\">parity.md</a>" "a sample's report link"

            testCase "a kind without a template gets the records view"
            <| fun () ->
                use tmp = new TempDir()
                let output = Dashboard.build (mixed tmp)
                let v = output.Model.Views |> List.find (fun v -> v.Entry.Id = "r-t")
                Expect.isNone v.Template "no T1 template"
                Expect.isNone v.Error "rendered"
                Expect.stringContains v.Markdown "**FAIL.** 1 check of 10 failed." "verdict"
                Expect.stringContains v.Markdown "| tokenizer.case | 9 | 1 | 10 |" "checks table"

                Expect.stringContains
                    v.Markdown
                    "| counter | tokenizer.merges | 1 | count | total 151,387 count |"
                    "a counter"

                Expect.stringContains
                    v.Markdown
                    "| gauge | tokenizer.cache | 1 | B | last 4.00 KiB, range 4.00 KiB to 4.00 KiB |"
                    "a gauge"

                Expect.equal (v.Charts |> List.map fst) [ "records-checks.svg"; "records-spans.svg" ] "charts"
                Expect.stringContains output.Html "id=\"r-r-t-records-checks-fig\"" "chart inlined"
                Expect.stringContains output.Html "lrm report r-t --template &lt;name&gt;" "how to get a report"

            testCase "a compare view shows both runs' fields, values, metering and charts"
            <| fun () ->
                use tmp = new TempDir()
                let output = Dashboard.build (mixed tmp)
                let html = output.Html
                let start = html.IndexOf("id=\"cmp-r-a~r-c\"", StringComparison.Ordinal)
                Expect.isGreaterThan start 0 "the pair's view"
                let stop = html.IndexOf("</section>", start, StringComparison.Ordinal)
                let view = html.Substring(start, stop - start)

                Expect.stringContains
                    view
                    "<td>Status</td><td class=\"d-wrap\">ok</td><td class=\"d-wrap\">failed</td><td>changed</td>"
                    "status"

                Expect.stringContains view "<code>verdict</code>" "values"
                Expect.stringContains view "<td>FLOPs</td>" "metering"
                let a = output.Model.Views |> List.find (fun v -> v.Entry.Id = "r-a")
                Expect.equal (occurrences "class=\"d-slot\"" view) (2 * a.Charts.Length) "a slot per chart and run"

                for file, _ in a.Charts do
                    Expect.stringContains view ("data-src=\"" + RunView.chartId "r-a" file + "-fig\"") file

            testCase "watch mode reloads through the script, which keeps the fragment; a meta refresh only without it"
            <| fun () ->
                use tmp = new TempDir()
                let o = mixed tmp
                let still = (Dashboard.build o).Html

                let live =
                    (Dashboard.build
                        { o with
                            Watch = true
                            IntervalSeconds = 7
                        })
                        .Html

                Expect.isFalse (still.Contains "http-equiv") "no refresh"
                Expect.isFalse (still.Contains "data-watch=") "no watch marker"
                Expect.isFalse (still.Contains "<noscript>") "no fallback"
                Expect.stringContains live "<html lang=\"en\" data-watch=\"7\">" "watch marker with the interval"

                // A meta refresh reloads the URL the page was opened with; Chromium drops the #fragment the reader
                // navigated to since, so every reload went back to the runs index. Only <noscript> may hold one.
                let fallback = "<noscript><meta http-equiv=\"refresh\" content=\"7\"></noscript>"
                Expect.equal (occurrences fallback live) 1 "the fallback without JavaScript"
                Expect.isFalse (live.Replace(fallback, "").Contains "http-equiv") "no bare meta refresh"
                Expect.stringContains live "location.reload()" "the script reloads, keeping the fragment"

            testCase "nothing from a run log can inject markup"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                writeRun
                    runs
                    "r-x"
                    "parity"
                    "<script>alert(1)</script> & \"q\""
                    (unixNs 2026 10 18 10 0)
                    RunStatus.Ok
                    (fun () ->
                        Telemetry.check
                            "golden.compare"
                            true
                            [
                                Attr.str "pair" "<b>p</b>"
                                Attr.str "case" "<i>c"
                                Attr.str "tap" "L0.q_proj"
                                Attr.str "metric" "rel_l2"
                                Attr.float "measured" 1e-6
                                Attr.float "bound" 1e-5
                            ])
                |> ignore

                let html =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])).Html

                Expect.isFalse (html.Contains "<script>alert") "title escaped"
                Expect.isFalse (html.Contains "<b>p</b>") "pair escaped"
                Expect.isFalse (html.Contains "<i>c") "case escaped"
                Expect.stringContains html "&lt;script&gt;alert(1)&lt;/script&gt; &amp; \"q\"" "escaped title"

            testCase "a link in a run log never points at the network"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                // The template puts the title into the report's Markdown, where a link is a link.
                for id, title in
                    [
                        "r-n1", "[click](//evil.example/share/x)"
                        "r-n2", "[click](\\\\evil.example\\share\\x)"
                        "r-n3", "[click](file://evil.example/share/x)"
                        "r-n4", "[click](https://evil.example/x)"
                    ] do
                    writeRun runs id "parity" title (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
                    |> ignore

                let o =
                    { DashboardOptions.over [ RunsRoot.runs tmp.Path runs ] with
                        PerKind = 4
                    }

                let html = (Dashboard.build o).Html
                Expect.stringContains html "click (<code>//evil.example/share/x</code>)" "shown as text"

                for href in attributeValues "href" html do
                    Expect.isTrue (Html.isLocalHref href) ("a local href: " + href)

                Expect.isFalse (html.Contains "href=\"//") "no network path"
                Expect.isFalse (html.Contains "href=\"\\") "no UNC path"

            testCase "a damaged log still gets its view, with its problems"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                let dir =
                    writeRun runs "r-d" "parity" "D" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)

                File.AppendAllText(Path.Combine(dir, "events.n0.jsonl"), "{not json}\n{\"v\":1")
                let output = Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])
                let v = output.Model.Views.Head
                Expect.isNone v.Error "still rendered"
                Expect.isGreaterThanOrEqual v.Problems.Length 2 "problems"
                Expect.stringContains output.Html "<span>Log problems</span><b>" "problems tile"
                Expect.stringContains output.Html "the last line is cut off" "problem listed"

            testCase "an empty runs folder gives a page that says so"
            <| fun () ->
                use tmp = new TempDir()

                let html =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path (tmp.Sub "runs") ])).Html

                Expect.stringContains html "No runs yet." "empty index"
                Expect.stringContains html "No two runs of one kind have a view yet." "empty compare"

            testCase "unreadable folders are listed with their reasons"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                Directory.CreateDirectory(Path.Combine(runs, "legacy")) |> ignore

                File.WriteAllText(
                    Path.Combine(runs, "legacy", "manifest.json"),
                    "{\"schema\":\"lrm.run/1\",\"runId\":\"legacy\"}"
                )

                let html =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])).Html

                Expect.stringContains html "1 folder is not readable runs" "summary"
                Expect.stringContains html "<code>legacy</code>" "folder"

            testCase "writeReports writes what lrm report writes"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                let dir =
                    writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)

                let output = Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])
                let written = Dashboard.writeReports output
                let md = Path.Combine(dir, "report", "parity.md")
                Expect.equal written [ md ] "one report"
                let expected = Reports.render "parity" (RunLogReader.loadOrFail dir)
                Expect.equal (File.ReadAllText md) expected.Markdown "Markdown"

                for file, svg in expected.Svgs do
                    Expect.equal (File.ReadAllText(Path.Combine(dir, "report", "charts", file))) svg file

            testCase "a run outside the page's folder names its report as text, with no absolute path"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                let build (pageDir: string) =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs pageDir runs ])).Html

                let html = build (tmp.Sub "pages")

                Expect.stringContains
                    html
                    "Markdown report: <code>runs/r-a/report/parity.md</code> beside the run log"
                    "the report's path from the runs folder"

                Expect.isFalse (html.Contains "file:") "no file: URI"
                Expect.isFalse (html.Contains(tmp.Path.Replace('\\', '/'))) "no absolute path"
                Expect.equal (build (tmp.Sub "elsewhere")) html "the same bytes from another folder"

            testCase "Until keeps the runs that started by then, so a snapshot stays the same while runs are added"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                for id, hour in [ "r-a", 10; "r-b", 11 ] do
                    writeRun runs id "parity" id (unixNs 2026 10 18 hour 0) RunStatus.Ok (parityBody 0.5)
                    |> ignore

                let o =
                    { DashboardOptions.over [ RunsRoot.runs tmp.Path runs ] with
                        Until = Some(DateTimeOffset(2026, 10, 18, 11, 0, 0, TimeSpan.Zero))
                    }

                let before = (Dashboard.build o).Bytes

                writeRun runs "r-c" "parity" "r-c" (unixNs 2026 10 18 12 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                let output = Dashboard.build o
                Expect.equal (output.Model.Runs |> List.map _.Id) [ "r-b"; "r-a" ] "the runs that started by 11:00"
                Expect.isTrue (output.Bytes = before) "the same bytes after a later run is written"

            testCase "the count axis of the runs chart prints whole, evenly spaced ticks"
            <| fun () ->
                // NumFormat.Int rounds every tick label, so a fractional step printed "0 0 0 1 1 1" for one run.
                for peak in 1..60 do
                    let chart =
                        Chart.Bar
                            {
                                Title = "t"
                                Subtitle = ""
                                CategoryLabel = "kind"
                                ValueLabel = "runs"
                                ValueFormat = Page.countFormat peak
                                Categories = [ "a"; "b" ]
                                Series =
                                    [
                                        {
                                            Name = "ok"
                                            Color = Color.Good
                                            Values = [ float peak; 1.0 ]
                                        }
                                    ]
                                Stacked = true
                                Horizontal = true
                            }

                    let svg = ChartSvg.render "c" chart
                    Expect.isFalse (svg.Contains ">1.000<") (Invariant.format $"a count prints whole for {peak}")
                    let doc = XmlDocument()
                    doc.LoadXml svg
                    let ns = XmlNamespaceManager(doc.NameTable)
                    ns.AddNamespace("s", "http://www.w3.org/2000/svg")

                    let ticks =
                        [
                            for node in doc.SelectNodes("//s:text[@class='t3' and @text-anchor='middle']", ns) do
                                match
                                    Double.TryParse(
                                        node.InnerText.Replace(",", ""),
                                        Globalization.NumberStyles.Float,
                                        Invariant.culture
                                    )
                                with
                                | true, v -> yield v
                                | _ -> ()
                        ]

                    Expect.isGreaterThanOrEqual ticks.Length 2 (Invariant.format $"ticks for {peak}")
                    Expect.equal ticks.Head 0.0 (Invariant.format $"starts at 0 for {peak}")
                    Expect.isGreaterThanOrEqual (List.last ticks) (float peak) (Invariant.format $"reaches {peak}")
                    let step = ticks[1] - ticks[0]

                    for i in 1 .. ticks.Length - 1 do
                        Expect.floatClose
                            Accuracy.high
                            (ticks[i] - ticks[i - 1])
                            step
                            (sprintf "evenly spaced for %d: %A" peak ticks)

                    if peak >= 5 && not (peak > 10 && peak <= 12) then
                        Expect.equal (Page.countFormat peak) NumFormat.Int (Invariant.format $"Int for {peak}")
                        Expect.isTrue (ticks |> List.forall (fun v -> v = Math.Round v)) "whole ticks"

            testCase "a kind's durations are dots on a log axis, so a long run does not flatten the others"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                // Two quick runs and one that lasts hours (as the golden generation does beside the parity probes).
                for id, hour, body in
                    [
                        "r-a", 10, parityBody 0.5
                        "r-b", 11, parityBody 0.6
                        "r-c", 12, (fun () -> Telemetry.check "golden.compare" true [])
                    ] do
                    writeRun runs id "parity" id (unixNs 2026 10 18 hour 0) RunStatus.Ok body
                    |> ignore

                let entries, _ = Catalog.scan [ RunsRoot.runs tmp.Path runs ]
                let timed = entries |> List.rev

                match Page.durationChart timed with
                | Chart.ErrorBars c ->
                    Expect.isTrue c.YLog "log axis"
                    Expect.equal c.YFormat NumFormat.Duration "durations"
                    Expect.equal c.Categories.Length 3 "a category per run"

                    Expect.equal
                        (c.Series |> List.collect (fun s -> s.Spreads |> List.map _.Index) |> List.sort)
                        [ 0; 1; 2 ]
                        "a dot per run"
                | other -> failtestf "expected error bars, got %s" (ChartSvg.title other)

                let html =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])).Html

                Expect.stringContains html "id=\"d-kind-parity-durations-fig\"" "the kind view has the chart"
                Expect.stringContains html "oldest first, by status; log scale" "its subtitle says log scale"

            testCase "a long kind name never makes a pill wider than the page"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                let kind = "a-very-long-kind-name-for-an-experimental-benchmark-suite-x"

                writeRun runs "r-l" kind "L" (unixNs 2026 10 18 10 0) RunStatus.Ok recordsBody
                |> ignore

                let html =
                    (Dashboard.build (DashboardOptions.over [ RunsRoot.runs tmp.Path runs ])).Html

                Expect.stringContains
                    html
                    ("<a href=\"#kind-"
                     + kind
                     + "\" title=\""
                     + kind
                     + "\"><span>"
                     + kind
                     + "</span><b>1</b></a>")
                    "the pill's name can be cut, its title holds it all"

                Expect.stringContains html ".d-nav a span{min-width:0;overflow:hidden;text-overflow:ellipsis}" "cut"
                Expect.stringContains html ".d-nav a{display:inline-flex;max-width:100%" "at most the page's width"
                Expect.stringContains html ("<span class=\"d-tag\" title=\"" + kind + "\">") "the run's kind tag"

            testCase "the page has one h1, a skip link, focusable view headings and chart toggle buttons"
            <| fun () ->
                use tmp = new TempDir()
                let output = Dashboard.build (mixed tmp)
                let html = output.Html
                Expect.equal (occurrences "<h1" html) 1 "one h1"
                Expect.stringContains html "<h1 class=\"d-brand\"><a href=\"#runs\">LRM dashboard</a></h1>" "the brand"
                Expect.stringContains html "<a class=\"d-skip\" href=\"#d-main\" hidden>Skip to the view</a>" "skip"
                Expect.stringContains html "<main id=\"d-main\">" "its target"

                let views = occurrences "<section class=\"d-view\"" html
                Expect.equal (occurrences "<h2 id=\"runs-h\" tabindex=\"-1\">" html) 1 "the index heading"
                Expect.equal (occurrences "<h2 tabindex=\"-1\">" html + 1) views "every view heading takes focus"

                let figures = occurrences "<figure class=\"d-chart\"" html
                Expect.isGreaterThan figures 40 "charts"

                Expect.equal
                    (occurrences "tabindex=\"0\" role=\"button\" aria-pressed=\"false\" aria-label=\"Zoom chart: " html)
                    figures
                    "every chart is a named toggle button"

                Expect.stringContains
                    html
                    "aria-label=\"Zoom chart: Runs by kind and status\" aria-describedby=\"d-runs-by_kind-desc\""
                    "named after the chart, described by its description"

            testCase "write replaces the file atomically and leaves no temporary file"
            <| fun () ->
                use tmp = new TempDir()
                let o = mixed tmp
                let path = Path.Combine(tmp.Path, "out", "dashboard.html")
                let output = Dashboard.build o
                Dashboard.write path output
                Dashboard.write path output
                Expect.isTrue (File.ReadAllBytes path = output.Bytes) "bytes"
                Expect.isFalse (File.Exists(path + ".tmp")) "no temporary file"
                Expect.isFalse (output.Html.Contains "\r") "LF only"
                Expect.isFalse (output.Bytes.Length >= 3 && output.Bytes[0] = 0xEFuy) "no byte order mark"
        ]
