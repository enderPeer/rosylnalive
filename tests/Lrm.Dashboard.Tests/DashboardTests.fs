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
                    Expect.isTrue
                        (href.StartsWith "#" || href.StartsWith "file:///" || not (href.Contains ":"))
                        ("a local href: " + href)

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

                for labels in attributeValues "aria-labelledby" html do
                    for l in labels.Split ' ' do
                        Expect.isTrue (all.Contains l) ("label " + l)

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
                Expect.stringContains html ("<span>FLOPs</span><b>" + Fmt.si flops + "</b>") "FLOPs tile"
                Expect.stringContains html "<span>Wall (self)</span>" "wall tile"
                Expect.stringContains html "<span>GPU</span>" "GPU tile"
                Expect.stringContains html "<span>Bytes moved</span>" "bytes tile"
                Expect.stringContains html "<span>Tokens</span>" "tokens tile"
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

            testCase "watch mode adds the meta refresh, and only watch mode"
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
                Expect.stringContains live "<meta http-equiv=\"refresh\" content=\"7\">" "refresh"
                Expect.stringContains live "<html lang=\"en\" data-watch=\"\">" "watch marker"

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
