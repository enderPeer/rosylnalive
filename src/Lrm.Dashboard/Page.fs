namespace Lrm.Dashboard

open System
open System.Text
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports

/// Everything one dashboard page shows. Built by Dashboard.build from the run folders; rendered by Page.render.
type DashboardModel =
    {
        Title: string
        /// Watch mode: the page reloads itself every IntervalSeconds.
        Watch: bool
        IntervalSeconds: int
        /// How many runs roots were read.
        Roots: int
        /// Every readable run, newest first.
        Runs: RunEntry list
        Unreadable: Unreadable list
        /// The runs with a view, newest first.
        Views: RunView list
        /// Why the other runs have no view, by run id.
        NoView: Map<string, NoView>
        /// Comparisons as (older, newer): runs of one kind, both synthetic or both not.
        Pairs: (RunView * RunView) list
    }

/// The HTML page (ADR T2-0001): one file with inline CSS, JavaScript and SVG, the same bytes for the same run logs.
[<RequireQualifiedAccess>]
module Page =
    let private t = Html.text
    let private a = Html.attr

    // ---- small pieces ----------------------------------------------------------------------------------------------

    let private startText (r: RunEntry) =
        match r.Started with
        | Some s -> s.UtcDateTime.ToString("yyyy-MM-dd HH:mm:ss", Invariant.culture)
        | None -> r.Manifest.CreatedUtc

    let private shortStart (r: RunEntry) =
        match r.Started with
        | Some s -> s.UtcDateTime.ToString("MM-dd HH:mm:ss", Invariant.culture)
        | None -> r.Id

    let private durationText (r: RunEntry) =
        match r.DurationNs with
        | Some ns -> Fmt.duration (float ns)
        | None when r.Manifest.Status = RunStatus.Running -> "running"
        | None -> "n/a"

    let private statusName (r: RunEntry) = RunStatus.name r.Manifest.Status

    let private status (r: RunEntry) =
        let s = statusName r
        "<span class=\"d-st d-st-" + s + "\">" + s + "</span>"

    let private classShort (c: ExecutionClass option) =
        match c with
        | None -> "<span class=\"d-muted\">not recorded</span>"
        | Some c ->
            let detail = if String.IsNullOrEmpty c.Driver then c.Arch else c.Driver

            "<span title=\""
            + a (ExecutionClass.describe c)
            + "\">"
            + t (ExecutionBackend.name c.Backend)
            + " <span class=\"d-muted\">"
            + t detail
            + "</span></span>"

    let private cnt (n: int) = Fmt.int (int64 n)

    let private tile (label: string) (value: string) =
        "<div class=\"d-tile\"><span>" + t label + "</span><b>" + t value + "</b></div>"

    let private tiles (items: (string * string) list) =
        "<div class=\"d-tiles\">"
        + (items |> List.map (fun (l, v) -> tile l v) |> String.concat "")
        + "</div>\n"

    let private styleElement = "<style>" + Theme.style + "</style>\n"

    /// A chart as an inline SVG figure. The SVG is ChartSvg.render's text for this id with its <style> element left
    /// out: the page carries those rules once (Assets.chartCss). The figure is a toggle button (Enter, Space or a click
    /// shows the chart at full size), named after the chart and described by the chart's own description.
    let figure (id: string) (chart: Chart) =
        let svg = (ChartSvg.render id chart).Replace(styleElement, "").TrimEnd('\n')

        "<figure class=\"d-chart\" id=\""
        + a (id + "-fig")
        + "\" tabindex=\"0\" role=\"button\" aria-pressed=\"false\" aria-label=\""
        + a ("Zoom chart: " + ChartSvg.title chart)
        + "\" aria-describedby=\""
        + a (id + "-desc")
        + "\">"
        + svg
        + "</figure>"

    let private viewHref (r: RunEntry) = "#run-" + r.Id

    let pairId (older: RunView) (newer: RunView) =
        "cmp-" + older.Entry.Id + "~" + newer.Entry.Id

    let private crumbs (items: (string * string option) list) =
        let parts =
            items
            |> List.map (fun (label, href) ->
                match href with
                | Some h -> "<a href=\"" + a h + "\">" + t label + "</a>"
                | None -> "<span>" + t label + "</span>")

        "<nav class=\"d-crumbs\" aria-label=\"Breadcrumbs\">"
        + String.concat " › " parts
        + "</nav>\n"

    let private inlineMd (text: string) =
        Markdown.inlineHtml (fun _ _ -> None) text

    // ---- the runs table ----------------------------------------------------------------------------------------------

    let private runRow (m: DashboardModel) (withKind: bool) (r: RunEntry) =
        let sb = StringBuilder()

        sb.Append("<tr data-kind=\"").Append(a r.Kind).Append("\" data-status=\"").Append(statusName r).Append("\">")
        |> ignore

        // The label names the column; on a phone the CSS prints it before the cell (nth-child), so rows stay small.
        let cell (_label: string) (cls: string) (html: string) =
            sb.Append(if cls = "" then "<td>" else "<td class=\"" + cls + "\">").Append(html).Append("</td>")
            |> ignore

        cell "Started" "d-nw" (t (startText r))

        let run =
            let id = "<code>" + t r.Id + "</code>"

            let linked =
                match m.NoView.TryFind r.Id with
                | None -> "<a href=\"" + a (viewHref r) + "\">" + id + "</a>"
                | Some(NoView.TooLarge bytes) ->
                    id
                    + " <span class=\"d-tag\">log "
                    + t (Fmt.bytes (float bytes))
                    + ", too large for a view</span>"
                | Some NoView.Older -> id

            if r.Manifest.Synthetic then
                linked + " <span class=\"d-tag\">synthetic</span>"
            else
                linked

        cell "Run" "" run

        if withKind then
            cell "Kind" "" ("<a href=\"#kind-" + a r.Kind + "\">" + t r.Kind + "</a>")

        cell "Status" "d-nw" (status r)
        cell "Duration" "d-r d-nw" (t (durationText r))
        cell "Nodes" "d-r" (cnt r.Nodes)
        cell "Class" "" (classShort r.Manifest.ExecutionClass)
        cell "Title" "d-wrap" (t r.Manifest.Title)
        sb.Append("</tr>\n").ToString()

    let private runTable (m: DashboardModel) (withKind: bool) (bodyId: string option) (runs: RunEntry list) =
        let sb = StringBuilder()

        sb
            .Append("<div class=\"d-scroll\"><table class=\"d-table d-rows")
            .Append(if withKind then "" else " d-nokind")
            .Append("\">\n<thead><tr>")
        |> ignore

        for h, cls in
            [
                "Started (UTC)", ""
                "Run", ""
                if withKind then
                    "Kind", ""
                "Status", ""
                "Duration", "d-r"
                "Nodes", "d-r"
                "Execution class", ""
                "Title", ""
            ] do
            sb.Append("<th scope=\"col\"").Append(if cls = "" then "" else " class=\"" + cls + "\"").Append('>')
            |> ignore

            sb.Append(t h).Append("</th>") |> ignore

        sb.Append("</tr></thead>\n<tbody") |> ignore

        match bodyId with
        | Some id -> sb.Append(" id=\"").Append(id).Append('"') |> ignore
        | None -> ()

        sb.Append(">\n") |> ignore

        for r in runs do
            sb.Append(runRow m withKind r) |> ignore

        sb.Append("</tbody></table></div>\n").ToString()

    let private statuses =
        [
            RunStatus.Ok, Color.Good
            RunStatus.Failed, Color.Critical
            RunStatus.Running, Color.Warning
            RunStatus.Aborted, Color.Serious
        ]

    let private countStatus (s: RunStatus) (runs: RunEntry list) =
        runs |> List.filter (fun r -> r.Manifest.Status = s) |> List.length

    /// The value format of a count axis whose largest bar is `peak` runs. T1's bar axis has about five ticks at 1, 2,
    /// 2.5 or 5 times a power of ten of peak/5, and NumFormat.Int rounds each tick label to a whole number, so a
    /// fractional step would print repeated or wrong labels ("0 0 0 1 1 1" for one run). Int is used only where the
    /// step is whole (peak/5 at least 1 and not in (2, 2.5]); otherwise Si prints the fractional ticks as they are
    /// ("0.2") and the bar values as whole counts ("1"). The cpu tier checks the labels for every peak from 1 to 60
    /// (T2 asks T1 for integer steps on Int axes).
    let countFormat (peak: int) =
        let fifth = float peak / 5.0

        if fifth >= 1.0 && not (fifth > 2.0 && fifth <= 2.5) then
            NumFormat.Int
        else
            NumFormat.Si

    // ---- views -----------------------------------------------------------------------------------------------------

    let private indexView (m: DashboardModel) (kinds: string list) =
        let sb = StringBuilder()
        let runs = m.Runs

        sb.Append("<section class=\"d-view\" id=\"runs\" data-nav=\"runs\" aria-labelledby=\"runs-h\">\n")
        |> ignore

        sb.Append("<h2 id=\"runs-h\" tabindex=\"-1\">Runs</h2>\n") |> ignore

        sb.Append(
            "<p class=\"d-muted\">Every run in the runs folders, newest first. A run with a view links to it: "
            + "the sections of its report, its metering totals and a link to the Markdown report.</p>\n"
        )
        |> ignore

        sb.Append(
            tiles (
                [ "Runs", cnt runs.Length ]
                @ [ for s, _ in statuses -> RunStatus.name s, cnt (countStatus s runs) ]
                @ [ "With a view", cnt m.Views.Length; "Not readable", cnt m.Unreadable.Length ]
            )
        )
        |> ignore

        let present =
            kinds |> List.filter (fun k -> runs |> List.exists (fun r -> r.Kind = k))

        if not present.IsEmpty then
            let series =
                statuses
                |> List.filter (fun (s, _) -> countStatus s runs > 0)
                |> List.map (fun (s, c) ->
                    {
                        Name = RunStatus.name s
                        Color = c
                        Values =
                            present
                            |> List.map (fun k ->
                                runs
                                |> List.filter (fun r -> r.Kind = k && r.Manifest.Status = s)
                                |> List.length
                                |> float)
                    })

            let peak =
                present
                |> List.map (fun k -> runs |> List.filter (fun r -> r.Kind = k) |> List.length)
                |> List.fold max 0

            let chart =
                Chart.Bar
                    {
                        Title = "Runs by kind and status"
                        Subtitle = "Every readable run in the runs folders"
                        CategoryLabel = "kind"
                        ValueLabel = "runs"
                        ValueFormat = countFormat peak
                        Categories = present
                        Series = series
                        Stacked = true
                        Horizontal = true
                    }

            sb.Append(figure "d-runs-by_kind" chart).Append('\n') |> ignore

        sb.Append("<div class=\"d-filters\" id=\"d-filters\" hidden>") |> ignore

        sb.Append("<label>Kind <select id=\"d-f-kind\"><option value=\"\">all kinds</option>")
        |> ignore

        for k in present do
            sb.Append("<option value=\"").Append(a k).Append("\">").Append(t k).Append("</option>")
            |> ignore

        sb.Append("</select></label>") |> ignore

        sb.Append("<label>Status <select id=\"d-f-status\"><option value=\"\">any status</option>")
        |> ignore

        for s, _ in statuses do
            let n = RunStatus.name s

            sb.Append("<option value=\"").Append(n).Append("\">").Append(n).Append("</option>")
            |> ignore

        sb.Append("</select></label>") |> ignore

        sb.Append(
            "<label>Search <input id=\"d-f-q\" type=\"search\" placeholder=\"run id, title, class\" autocomplete=\"off\"></label>"
        )
        |> ignore

        sb.Append("<span id=\"d-f-count\" class=\"d-muted\" aria-live=\"polite\"></span></div>\n")
        |> ignore

        if runs.IsEmpty then
            sb.Append("<div class=\"d-card\"><p>No runs yet. Runs appear here as soon as a host records one ")
            |> ignore

            sb.Append("(RunLog contract: <code>LRM_DATA/runs/&lt;runId&gt;/</code>).</p></div>\n")
            |> ignore
        else
            sb.Append(runTable m true (Some "d-runs-body") runs) |> ignore

        if not m.Unreadable.IsEmpty then
            sb.Append("<details class=\"d-card\"><summary>") |> ignore

            sb.Append(
                t (Builders.Common.plural m.Unreadable.Length "folder is" "folders are")
                + " not readable runs"
            )
            |> ignore

            sb.Append("</summary>\n<div class=\"d-scroll\"><table class=\"d-table\"><thead><tr>")
            |> ignore

            if m.Roots > 1 then
                sb.Append("<th scope=\"col\" class=\"d-r\">Root</th>") |> ignore

            sb.Append("<th scope=\"col\">Folder</th><th scope=\"col\">Why</th></tr></thead><tbody>\n")
            |> ignore

            for u in m.Unreadable do
                sb.Append("<tr>") |> ignore

                if m.Roots > 1 then
                    sb.Append("<td class=\"d-r\">").Append(cnt (u.Root + 1)).Append("</td>")
                    |> ignore

                sb
                    .Append("<td><code>")
                    .Append(t u.Folder)
                    .Append("</code></td><td class=\"d-wrap\">")
                    .Append(t u.Reason)
                    .Append("</td></tr>\n")
                |> ignore

            sb.Append("</tbody></table></div></details>\n") |> ignore

        sb.Append("</section>\n").ToString()

    let private pairLink (older: RunView) (newer: RunView) =
        "<a href=\"#"
        + a (pairId older newer)
        + "\"><code>"
        + t newer.Entry.Id
        + "</code> against <code>"
        + t older.Entry.Id
        + "</code></a> <span class=\"d-muted\">("
        + t (shortStart newer.Entry)
        + " and "
        + t (shortStart older.Entry)
        + ")</span>"

    let private pairList (pairs: (RunView * RunView) list) =
        "<ul class=\"d-pairs\">\n"
        + (pairs
           |> List.map (fun (o, n) -> "<li>" + pairLink o n + "</li>\n")
           |> String.concat "")
        + "</ul>\n"

    let private phaseNote (kind: string) (phase: int) =
        match phase with
        | 2 -> "Training runs start in Phase 2 (the backward pass and training)."
        | 3 -> "Runs of this kind come with the LRM itself, in Phase 3 (Roslyn experts)."
        | _ ->
            "Runs of this kind appear here once a host records one with kind <code>"
            + t kind
            + "</code>."

    /// The most runs a kind's view lists; the runs index lists every run.
    [<Literal>]
    let KindRows = 15

    /// The duration of each of `timed` (finished runs, oldest first) as a dot per run, coloured by status, on a log
    /// axis: one kind's runs can last from a millisecond to hours (the parity probes and the golden generation), and a
    /// linear axis would flatten all but the longest. A run of zero length sits in the axis's zero strip.
    let durationChart (timed: RunEntry list) =
        let timed = Array.ofList timed

        Chart.ErrorBars
            {
                Title = "Duration of each run"
                Subtitle =
                    Invariant.format $"The {timed.Length} newest finished runs, oldest first, by status; log scale"
                XLabel = "start (UTC)"
                YLabel = "duration"
                YFormat = NumFormat.Duration
                YLog = true
                Categories = timed |> Array.map shortStart |> List.ofArray
                Series =
                    statuses
                    |> List.filter (fun (s, _) -> timed |> Array.exists (fun r -> r.Manifest.Status = s))
                    |> List.map (fun (s, c) ->
                        {
                            Name = RunStatus.name s
                            Color = c
                            Spreads =
                                [
                                    for i in 0 .. timed.Length - 1 do
                                        let r = timed[i]

                                        if r.Manifest.Status = s then
                                            let ns = float r.DurationNs.Value

                                            yield
                                                {
                                                    Index = i
                                                    Low = ns
                                                    Mid = ns
                                                    High = ns
                                                    Invalid = 0
                                                }
                                ]
                        })
                RefLines = []
            }

    let private kindView (m: DashboardModel) (kind: string) =
        let sb = StringBuilder()
        let runs = m.Runs |> List.filter (fun r -> r.Kind = kind)
        let views = m.Views |> List.filter (fun v -> v.Entry.Kind = kind)
        let pairs = m.Pairs |> List.filter (fun (o, _) -> o.Entry.Kind = kind)
        let template = Catalog.templateOf kind
        let id = "kind-" + kind

        sb.Append("<section class=\"d-view\" id=\"" + a id + "\" data-nav=\"" + a id + "\">\n")
        |> ignore

        sb.Append(crumbs [ "Runs", Some "#runs"; kind, None ]) |> ignore
        sb.Append("<h2 tabindex=\"-1\">").Append(t kind) |> ignore

        match template with
        | Some tpl ->
            sb.Append(" <span class=\"d-tag\">phase ").Append(cnt tpl.Phase).Append("</span></h2>\n")
            |> ignore

            sb.Append("<p>").Append(t tpl.Description).Append(".</p>\n") |> ignore
        | None ->
            sb.Append("</h2>\n<p>No report template reads this kind, so its runs show the dashboard's records view: ")
            |> ignore

            sb.Append("every record name and what it adds up to.</p>\n") |> ignore

        let real = runs |> List.filter (fun r -> not r.Manifest.Synthetic)

        if real.IsEmpty then
            sb.Append("<div class=\"d-card\"><p><b>No measured ").Append(t kind).Append(" runs yet.</b> ")
            |> ignore

            sb.Append(phaseNote kind (template |> Option.map _.Phase |> Option.defaultValue 1))
            |> ignore

            if not runs.IsEmpty then
                sb.Append(" The synthetic sample below shows what this view will show; its numbers are made up.")
                |> ignore

            sb.Append("</p></div>\n") |> ignore

        if not runs.IsEmpty then
            sb.Append(
                tiles
                    [
                        "Runs", cnt runs.Length
                        "ok", cnt (countStatus RunStatus.Ok runs)
                        "failed", cnt (countStatus RunStatus.Failed runs)
                        "With a view", cnt views.Length
                    ]
            )
            |> ignore

            match views with
            | latest :: _ ->
                sb.Append("<p class=\"d-links\">Newest view: <a href=\"") |> ignore

                sb
                    .Append(a (viewHref latest.Entry))
                    .Append("\">")
                    .Append(t latest.Entry.Manifest.Title)
                    .Append("</a> <span class=\"d-muted\">(")
                    .Append(t (startText latest.Entry))
                    .Append(")</span></p>\n")
                |> ignore
            | [] -> ()

            let timed =
                runs
                |> List.filter (fun r -> r.DurationNs.IsSome)
                |> List.truncate 40
                |> List.rev

            if timed.Length >= 2 then
                sb.Append(figure ("d-kind-" + kind + "-durations") (durationChart timed)).Append('\n')
                |> ignore

            if not pairs.IsEmpty then
                sb.Append("<h3>Compare</h3>\n").Append(pairList pairs) |> ignore

            let shown = List.truncate KindRows runs
            sb.Append("<h3>Runs</h3>\n").Append(runTable m false None shown) |> ignore

            if runs.Length > shown.Length then
                sb
                    .Append("<p class=\"d-muted\">The ")
                    .Append(t (Builders.Common.plural (runs.Length - shown.Length) "older run is" "older runs are"))
                    .Append(" in the <a href=\"#runs\">runs index</a>.</p>\n")
                |> ignore

        sb.Append("</section>\n").ToString()

    let private compareIndex (m: DashboardModel) (kinds: string list) =
        let sb = StringBuilder()

        sb.Append("<section class=\"d-view\" id=\"compare\" data-nav=\"compare\">\n")
        |> ignore

        sb.Append(crumbs [ "Runs", Some "#runs"; "Compare", None ]) |> ignore
        sb.Append("<h2 tabindex=\"-1\">Compare</h2>\n") |> ignore

        sb.Append(
            "<p>Two runs of the same kind side by side: what changed in the run, its headline values, its metering "
            + "totals and every chart. Every pair of runs with a view is listed; synthetic runs are compared only "
            + "with synthetic runs.</p>\n"
        )
        |> ignore

        if m.Pairs.IsEmpty then
            sb.Append("<div class=\"d-card\"><p>No two runs of one kind have a view yet.</p></div>\n")
            |> ignore
        else
            for k in kinds do
                let pairs = m.Pairs |> List.filter (fun (o, _) -> o.Entry.Kind = k)

                if not pairs.IsEmpty then
                    sb.Append("<h3><a href=\"#kind-").Append(a k).Append("\">").Append(t k).Append("</a></h3>\n")
                    |> ignore

                    sb.Append(pairList pairs) |> ignore

        sb.Append("</section>\n").ToString()

    /// A run's metering totals: (tile label, name in a sentence, value text, whether it is non-zero). Wall time first.
    let private totalsRows (x: MeterTotals) =
        [
            "Wall (self)", "wall time", Fmt.duration (float x.WallNs), x.WallNs <> 0L
            "GPU", "GPU time", Fmt.duration (float x.GpuNs), x.GpuNs <> 0L
            "FLOPs", "FLOPs", Fmt.si x.Flops, x.Flops <> 0.0
            "Bytes moved", "bytes moved", Fmt.bytes x.Bytes, x.Bytes <> 0.0
            "Tokens", "tokens", Fmt.si x.Tokens, x.Tokens <> 0.0
            "Metered time", "metered time", Fmt.duration x.MeterNs, x.MeterNs <> 0.0
        ]

    /// The metering totals that are zero, in one line ("Nothing metered for GPU time, tokens."), or "". Zero totals
    /// get no tile, so a phone shows the report sooner.
    let private notMetered (rows: (string * string * string * bool) list) =
        match rows |> List.filter (fun (_, _, _, nonZero) -> not nonZero) with
        | [] -> ""
        | zero ->
            "<p class=\"d-muted d-small\">Nothing metered for "
            + t (zero |> List.map (fun (_, name, _, _) -> name) |> String.concat ", ")
            + ".</p>\n"

    let private runView (m: DashboardModel) (v: RunView) =
        let sb = StringBuilder()
        let e = v.Entry
        let id = "run-" + e.Id
        let nav = "kind-" + e.Kind

        sb.Append("<section class=\"d-view\" id=\"" + a id + "\" data-nav=\"" + a nav + "\">\n")
        |> ignore

        sb.Append(crumbs [ "Runs", Some "#runs"; e.Kind, Some("#" + nav); e.Id, None ])
        |> ignore

        sb.Append("<h2 tabindex=\"-1\">").Append(t e.Manifest.Title).Append("</h2>\n")
        |> ignore

        sb.Append("<p class=\"d-tags\">").Append(status e) |> ignore

        sb.Append(" <span class=\"d-tag\" title=\"").Append(a e.Kind).Append("\">").Append(t e.Kind).Append("</span>")
        |> ignore

        if e.Manifest.Synthetic then
            sb.Append(" <span class=\"d-tag\">synthetic</span>") |> ignore

        sb.Append(" ").Append(classShort e.Manifest.ExecutionClass) |> ignore
        sb.Append(" <code>").Append(t e.Id).Append("</code>") |> ignore

        sb.Append(" <span class=\"d-muted\">").Append(t (startText e)).Append(" UTC</span></p>\n")
        |> ignore

        // Only the totals that are not zero get a tile; the others are named in one line below the tiles.
        let totals = totalsRows v.Totals

        sb.Append(
            tiles (
                [ "Duration", durationText e; "Nodes", cnt e.Nodes; "Records", cnt v.Records ]
                @ [
                    for label, _, value, nonZero in totals do
                        if nonZero then
                            yield label, value
                ]
                @ (if v.Problems.IsEmpty then
                       []
                   else
                       [ "Log problems", cnt v.Problems.Length ])
            )
        )
        |> ignore

        sb.Append(notMetered totals) |> ignore
        sb.Append("<p class=\"d-links\">") |> ignore

        match v.Template with
        | Some tpl ->
            match v.Report with
            | Some(ReportLink.Href href) ->
                sb
                    .Append("Markdown report: <a href=\"")
                    .Append(a href)
                    .Append("\">")
                    .Append(t (tpl + ".md"))
                    .Append("</a>, rendered by ")
                |> ignore
            | Some(ReportLink.Text path) ->
                sb.Append("Markdown report: <code>").Append(t path).Append("</code> beside the run log, rendered by ")
                |> ignore
            | None -> sb.Append("The Markdown report is rendered by ") |> ignore

            sb.Append("<code>lrm report ").Append(t e.Id).Append("</code>.") |> ignore
        | None ->
            sb.Append("No report template reads kind <code>").Append(t e.Kind).Append("</code>; ")
            |> ignore

            sb.Append("<code>lrm report ").Append(t e.Id).Append(" --template &lt;name&gt;</code> renders one.")
            |> ignore

        let mine =
            m.Pairs |> List.filter (fun (o, n) -> o.Entry.Id = e.Id || n.Entry.Id = e.Id)

        if not mine.IsEmpty then
            sb.Append(" Compare with ") |> ignore

            mine
            |> List.iteri (fun i (o, n) ->
                let other = if o.Entry.Id = e.Id then n else o

                sb
                    .Append(if i = 0 then "" else ", ")
                    .Append("<a href=\"#")
                    .Append(a (pairId o n))
                    .Append("\"><code>")
                    .Append(t other.Entry.Id)
                    .Append("</code></a>")
                |> ignore)

            sb.Append('.') |> ignore

        sb.Append("</p>\n") |> ignore

        match v.Error with
        | Some reason ->
            sb.Append("<div class=\"d-card\"><p><b>No view:</b> ").Append(t reason).Append("</p></div>\n")
            |> ignore
        | None ->
            let charts = v.Charts |> Map.ofList

            let fig (alt: string) (src: string) =
                if src.StartsWith("charts/", StringComparison.Ordinal) then
                    let file = src.Substring 7

                    charts.TryFind file
                    |> Option.map (fun c -> figure (RunView.chartId e.Id file) c)
                else
                    None

            // The report's first heading is its title, which the view already shows.
            let body =
                if v.Markdown.StartsWith("# ", StringComparison.Ordinal) then
                    match v.Markdown.IndexOf '\n' with
                    | i when i >= 0 -> v.Markdown.Substring(i + 1)
                    | _ -> ""
                else
                    v.Markdown

            let html, headings =
                Markdown.render
                    {
                        HeadingShift = 1
                        IdPrefix = id
                        Figure = fig
                    }
                    body

            let sections = headings |> List.filter (fun h -> h.Level = 3)

            if sections.Length > 1 then
                sb.Append("<nav class=\"d-toc\" aria-label=\"Sections\">") |> ignore

                for h in sections do
                    sb.Append("<a href=\"#").Append(a h.Id).Append("\">").Append(inlineMd h.Text).Append("</a>")
                    |> ignore

                sb.Append("</nav>\n") |> ignore

            if not v.Charts.IsEmpty then
                sb.Append("<p class=\"d-hint\">Tap a chart to see it at full size; tap again to fit it.</p>\n")
                |> ignore

            sb.Append("<article class=\"d-report\">\n").Append(html).Append("</article>\n")
            |> ignore

        sb.Append("</section>\n").ToString()

    let private changeOf (fmt: float -> string) (x: float) (y: float) =
        if x = y then
            ""
        elif x = 0.0 then
            "new"
        else
            let d = y - x
            let sign = if d > 0.0 then "+" else "−"
            sign + fmt (abs d) + " (×" + Fmt.dp 2 (y / x) + ")"

    let private compareTable (headers: string list) (rows: (string * string * string * string) list) =
        let sb = StringBuilder()

        sb.Append("<div class=\"d-scroll\"><table class=\"d-table\"><thead><tr>")
        |> ignore

        for h in headers do
            sb.Append("<th scope=\"col\">").Append(t h).Append("</th>") |> ignore

        sb.Append("</tr></thead><tbody>\n") |> ignore

        for label, x, y, change in rows do
            sb.Append(if change = "" then "<tr>" else "<tr class=\"d-changed\">") |> ignore

            sb
                .Append("<td>")
                .Append(label)
                .Append("</td><td class=\"d-wrap\">")
                .Append(x)
                .Append("</td><td class=\"d-wrap\">")
                .Append(y)
                .Append("</td><td>")
                .Append(change)
                .Append("</td></tr>\n")
            |> ignore

        sb.Append("</tbody></table></div>\n").ToString()

    let private pairView (index: int) (older: RunView) (newer: RunView) =
        let sb = StringBuilder()
        let kind = older.Entry.Kind
        let id = pairId older newer

        sb.Append("<section class=\"d-view\" id=\"" + a id + "\" data-nav=\"compare\">\n")
        |> ignore

        sb.Append(crumbs [ "Runs", Some "#runs"; "Compare", Some "#compare"; kind, None ])
        |> ignore

        sb.Append("<h2 tabindex=\"-1\">Compare ").Append(t kind).Append(" runs</h2>\n<div class=\"d-pair-grid\">\n")
        |> ignore

        for label, v in [ "A, older", older; "B, newer", newer ] do
            let e = v.Entry

            sb
                .Append("<div class=\"d-card\"><p class=\"d-slot-label\">")
                .Append(label)
                .Append("</p><p><a href=\"")
                .Append(a (viewHref e))
                .Append("\"><code>")
                .Append(t e.Id)
                .Append("</code></a></p><p>")
                .Append(t e.Manifest.Title)
                .Append("</p><p class=\"d-tags\">")
                .Append(status e)
                .Append(" <span class=\"d-muted\">")
                .Append(t (startText e))
                .Append(" UTC</span></p></div>\n")
            |> ignore

        sb.Append("</div>\n") |> ignore

        let same (x: string) (y: string) = if x = y then "" else "changed"

        let field (label: string) (f: RunView -> string) =
            let x = f older
            let y = f newer
            t label, t x, t y, same x y

        let classOf (v: RunView) =
            v.Entry.Manifest.ExecutionClass
            |> Option.map ExecutionClass.describe
            |> Option.defaultValue "not recorded"

        sb.Append("<h3>Run</h3>\n") |> ignore

        sb.Append(
            compareTable
                [ "Field"; "A"; "B"; "Change" ]
                [
                    field "Status" (fun v -> statusName v.Entry)
                    field "Duration" (fun v -> durationText v.Entry)
                    field "Nodes" (fun v -> cnt v.Entry.Nodes)
                    field "Records" (fun v -> cnt v.Records)
                    field "Execution class" classOf
                    field "Driver" (fun v -> defaultArg v.Entry.Manifest.Driver "n/a")
                    field "Commit" (fun v -> defaultArg v.Entry.Manifest.GitSha "not recorded")
                    field "Log problems" (fun v -> cnt v.Problems.Length)
                ]
        )
        |> ignore

        let names =
            (older.Values |> List.map fst) @ (newer.Values |> List.map fst)
            |> List.distinct
            |> List.sortWith (fun x y -> String.CompareOrdinal(x, y))

        if not names.IsEmpty then
            let lookup (v: RunView) (n: string) =
                v.Values |> List.tryFind (fun (k, _) -> k = n) |> Option.map snd

            sb.Append("<h3>Values</h3>\n") |> ignore

            sb.Append(
                compareTable
                    [ "Value"; "A"; "B"; "Change" ]
                    [
                        for n in names ->
                            let x = lookup older n
                            let y = lookup newer n

                            let show =
                                Option.map inlineMd >> Option.defaultValue "<span class=\"d-muted\">none</span>"

                            "<code>" + t n + "</code>", show x, show y, (if x = y then "" else "changed")
                    ]
            )
            |> ignore

        sb.Append("<h3>Metering</h3>\n") |> ignore
        let x = older.Totals
        let y = newer.Totals

        // Totals that are zero in both runs are left out; wall time is always shown.
        let metric (always: bool) (label: string) (fmt: float -> string) (f: MeterTotals -> float) =
            if always || f x <> 0.0 || f y <> 0.0 then
                [ t label, t (fmt (f x)), t (fmt (f y)), t (changeOf fmt (f x) (f y)) ]
            else
                []

        sb.Append(
            compareTable
                [ "Total"; "A"; "B"; "Change" ]
                (metric true "Wall (self)" Fmt.duration (fun m -> float m.WallNs)
                 @ metric false "GPU" Fmt.duration (fun m -> float m.GpuNs)
                 @ metric false "FLOPs" Fmt.si _.Flops
                 @ metric false "Bytes moved" Fmt.bytes _.Bytes
                 @ metric false "Tokens" Fmt.si _.Tokens
                 @ metric false "Metered time" Fmt.duration _.MeterNs)
        )
        |> ignore

        let keyed (v: RunView) =
            v.Charts
            |> List.map (fun (file, c) -> RunView.chartKey v.TemplateName file, (file, c))

        let ka = keyed older
        let kb = keyed newer
        let keys = (ka |> List.map fst) @ (kb |> List.map fst) |> List.distinct

        if not keys.IsEmpty then
            sb.Append("<h3>Charts</h3>\n") |> ignore

            sb.Append(
                "<p class=\"d-muted\">Each chart of A beside the same chart of B. Click a chart to see it at full size.</p>\n"
            )
            |> ignore

            keys
            |> List.iteri (fun k key ->
                let ca = ka |> List.tryFind (fun (x, _) -> x = key) |> Option.map snd
                let cb = kb |> List.tryFind (fun (x, _) -> x = key) |> Option.map snd

                let title =
                    match ca, cb with
                    | Some(_, c), _
                    | None, Some(_, c) -> ChartSvg.title c
                    | None, None -> key

                sb.Append("<h4>").Append(t title).Append("</h4>\n<div class=\"d-pair-grid\">\n")
                |> ignore

                for side, v, chart in [ "a", older, ca; "b", newer, cb ] do
                    sb
                        .Append("<div><p class=\"d-slot-label\">")
                        .Append(if side = "a" then "A" else "B")
                        .Append(" · <code>")
                        .Append(t v.Entry.Id)
                        .Append("</code></p>")
                    |> ignore

                    match chart with
                    | Some(file, _) ->
                        let fig = RunView.chartId v.Entry.Id file + "-fig"

                        sb
                            .Append("<div class=\"d-slot\" data-src=\"")
                            .Append(a fig)
                            .Append("\" data-n=\"")
                            .Append(Invariant.format $"c{index}-{k}{side}")
                            .Append("\"><a href=\"#")
                            .Append(a fig)
                            .Append("\">Open the chart in the run's view</a></div></div>\n")
                        |> ignore
                    | None ->
                        sb.Append("<p class=\"d-muted\">This run has no such chart.</p></div>\n")
                        |> ignore

                sb.Append("</div>\n") |> ignore)

        sb.Append("</section>\n").ToString()

    // ---- the page --------------------------------------------------------------------------------------------------

    /// The whole page.
    let render (m: DashboardModel) : string =
        let sb = StringBuilder(1 <<< 20)
        let kinds = Catalog.kindOrder m.Runs

        let navKinds =
            kinds
            |> List.filter (fun k -> (Catalog.templateOf k).IsSome || m.Runs |> List.exists (fun r -> r.Kind = k))

        sb.Append("<!doctype html>\n") |> ignore

        // Watch mode: data-watch holds the reload interval, and the script reloads the page with location.reload(),
        // which keeps the view the reader navigated to (its #fragment). A meta refresh would reload the URL the page
        // was opened with, which Chromium does without the fragment, so it is only the fallback without JavaScript.
        sb.Append(
            if m.Watch then
                Invariant.format $"<html lang=\"en\" data-watch=\"{m.IntervalSeconds}\">\n"
            else
                "<html lang=\"en\">\n"
        )
        |> ignore

        sb.Append("<head>\n<meta charset=\"utf-8\">\n") |> ignore

        sb.Append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
        |> ignore

        if m.Watch then
            sb.Append(
                Invariant.format $"<noscript><meta http-equiv=\"refresh\" content=\"{m.IntervalSeconds}\"></noscript>\n"
            )
            |> ignore

        sb.Append("<meta name=\"generator\" content=\"lrm dashboard (Lrm.Dashboard, ADR T2-0001)\">\n")
        |> ignore

        sb.Append("<title>").Append(t m.Title).Append("</title>\n") |> ignore
        sb.Append("<script>").Append(Assets.headScript).Append("</script>\n") |> ignore
        sb.Append("<style>\n").Append(Assets.pageCss).Append("\n</style>\n") |> ignore
        sb.Append("<style>\n").Append(Assets.chartCss).Append("\n</style>\n") |> ignore
        sb.Append("</head>\n<body>\n") |> ignore

        // The skip link moves focus to the current view's heading (the script shows it; without JavaScript it is hidden,
        // because a fragment outside every view would switch to the runs index).
        sb.Append("<a class=\"d-skip\" href=\"#d-main\" hidden>Skip to the view</a>\n")
        |> ignore

        // Header and navigation.
        sb.Append("<header class=\"d-top\">\n<div class=\"d-bar\">") |> ignore

        sb.Append("<h1 class=\"d-brand\"><a href=\"#runs\">").Append(t m.Title).Append("</a></h1>")
        |> ignore

        sb.Append("<span class=\"d-sum\">") |> ignore

        sb.Append(
            t (
                Builders.Common.plural m.Runs.Length "run" "runs"
                + " · "
                + Builders.Common.plural m.Views.Length "view" "views"
                + " · "
                + Builders.Common.plural m.Pairs.Length "comparison" "comparisons"
            )
        )
        |> ignore

        sb.Append("</span>") |> ignore

        if m.Watch then
            sb.Append(
                Invariant.format
                    $"<span class=\"d-live\">live: rebuilt when the run logs change, reloads every {m.IntervalSeconds} s</span>"
            )
            |> ignore

        sb.Append("<button id=\"d-theme\" type=\"button\">Theme: auto</button></div>\n")
        |> ignore

        sb.Append("<nav class=\"d-nav\" aria-label=\"Views\">") |> ignore

        // A pill's name is cut with an ellipsis when it is wider than the page (kind names are free strings); its
        // title holds the whole name.
        let pill (href: string) (name: string) (n: int) =
            sb
                .Append("<a href=\"")
                .Append(a href)
                .Append("\" title=\"")
                .Append(a name)
                .Append("\"><span>")
                .Append(t name)
                .Append("</span><b>")
                .Append(cnt n)
                .Append("</b></a>")
            |> ignore

        pill "#runs" "Runs" m.Runs.Length

        for k in navKinds do
            pill ("#kind-" + k) k (m.Runs |> List.filter (fun r -> r.Kind = k) |> List.length)

        pill "#compare" "Compare" m.Pairs.Length
        sb.Append("</nav>\n</header>\n<main id=\"d-main\">\n") |> ignore

        // Views; the runs index is last, so the CSS fallback can hide it behind a targeted view.
        for k in navKinds do
            sb.Append(kindView m k) |> ignore

        sb.Append(compareIndex m kinds) |> ignore

        for v in m.Views do
            sb.Append(runView m v) |> ignore

        m.Pairs |> List.iteri (fun i (o, n) -> sb.Append(pairView i o n) |> ignore)
        sb.Append(indexView m kinds) |> ignore
        sb.Append("</main>\n<footer>") |> ignore

        sb.Append(
            "Rendered by <code>lrm dashboard</code> from run logs only (RunLog contract), with the report templates' "
            + "builders and charts (ReportTemplates contract). The file is self-contained and makes no network "
            + "requests; the same run logs give the same bytes."
        )
        |> ignore

        sb.Append("</footer>\n<script>\n").Append(Assets.bodyScript).Append("\n</script>\n</body>\n</html>\n")
        |> ignore

        sb.ToString()
