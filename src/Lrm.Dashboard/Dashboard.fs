namespace Lrm.Dashboard

open System
open System.IO
open System.Text
open System.Threading
open System.Threading.Tasks
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports

/// What `lrm dashboard` builds (ADR T2-0001).
type DashboardOptions =
    {
        /// The folders the runs are read from, in order; an earlier root shadows a run id in a later one.
        Roots: RunsRoot list
        /// The page title.
        Title: string
        /// Watch mode: the page reloads itself every IntervalSeconds.
        Watch: bool
        IntervalSeconds: int
        /// The newest runs of each kind that get a view (the runs index lists every run).
        PerKind: int
        /// Run ids that always get a view.
        Include: string list
        /// Runs whose events files are larger than this get no view unless included by id.
        MaxLogBytes: int64
        /// The most runs whose comparisons are listed, per kind (the newest; synthetic runs separately).
        ComparePerKind: int
        /// The most run logs read at once.
        Threads: int
        /// Only runs that started at or before this time are shown: a snapshot that stays the same while new runs are
        /// written (the committed runs.html). None shows every run.
        Until: DateTimeOffset option
    }

[<RequireQualifiedAccess>]
module DashboardOptions =
    /// The defaults over these roots.
    let over (roots: RunsRoot list) =
        {
            Roots = roots
            Title = "LRM dashboard"
            Watch = false
            IntervalSeconds = 3
            PerKind = 3
            Include = []
            MaxLogBytes = 64L * 1024L * 1024L
            ComparePerKind = 4
            Threads = 4
            Until = None
        }

/// One built page and what it holds.
type DashboardOutput =
    {
        Html: string
        Model: DashboardModel
    }

    /// The file bytes: UTF-8 without a byte order mark.
    member o.Bytes = UTF8Encoding(false).GetBytes o.Html

/// The dashboard over run logs (ADR T2-0001): one self-contained HTML file built from the run folders through T1's
/// RunLogReader and report builders, so its charts are the reports' charts.
[<RequireQualifiedAccess>]
module Dashboard =
    /// Which runs get a view: every run of a ViewAll root and every included id; otherwise the newest PerKind runs of
    /// each kind whose logs are at most MaxLogBytes. Returns the runs with a view (newest first) and why the others
    /// have none.
    let select (o: DashboardOptions) (runs: RunEntry list) =
        let viewAll (r: RunEntry) =
            o.Roots
            |> List.tryItem r.Root
            |> Option.map _.ViewAll
            |> Option.defaultValue false

        let included (r: RunEntry) = List.contains r.Id o.Include
        let counts = Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal)
        let views = ResizeArray<RunEntry>()
        let mutable none = Map.empty

        for r in runs do
            if r.EventBytes > o.MaxLogBytes && not (included r) then
                none <- none.Add(r.Id, NoView.TooLarge r.EventBytes)
            elif viewAll r || included r then
                views.Add r
            else
                let n =
                    match counts.TryGetValue r.Kind with
                    | true, n -> n
                    | _ -> 0

                if n < o.PerKind then
                    counts[r.Kind] <- n + 1
                    views.Add r
                else
                    none <- none.Add(r.Id, NoView.Older)

        List.ofSeq views, none

    /// Every pair (older, newer) among the newest ComparePerKind views of each kind, synthetic runs apart from the
    /// others. Ordered by kind, then newest pair first.
    let pairs (o: DashboardOptions) (kinds: string list) (views: RunView list) =
        [
            for k in kinds do
                for synthetic in [ false; true ] do
                    let group =
                        views
                        |> List.filter (fun v -> v.Entry.Kind = k && v.Entry.Manifest.Synthetic = synthetic)
                        |> List.truncate o.ComparePerKind
                        |> Array.ofList

                    for i in 0 .. group.Length - 1 do
                        for j in i + 1 .. group.Length - 1 do
                            yield group[j], group[i]
        ]

    /// Reads the run folders and builds the page. Pure apart from reading the folders: the same run logs give the
    /// same bytes, under any culture.
    let build (o: DashboardOptions) : DashboardOutput =
        use span =
            Telemetry.span "dashboard.render" [ Attr.str TelemetryKeys.Component "dashboard" ]

        let runs, unreadable = Catalog.scan o.Roots

        // A run whose start does not parse has no time to compare, so it stays.
        let runs =
            match o.Until with
            | None -> runs
            | Some until ->
                runs
                |> List.filter (fun r ->
                    match r.Started with
                    | Some s -> s <= until
                    | None -> true)

        let chosen, none = select o runs
        let chosen = Array.ofList chosen
        let views: RunView[] = Array.zeroCreate chosen.Length

        let load (i: int) =
            let r = chosen[i]
            views[i] <- RunView.load o.Roots[r.Root] r

        Parallel.For(0, chosen.Length, ParallelOptions(MaxDegreeOfParallelism = max 1 o.Threads), load)
        |> ignore

        let views = List.ofArray views
        let kinds = Catalog.kindOrder runs

        let model =
            {
                Title = o.Title
                Watch = o.Watch
                IntervalSeconds = max 1 o.IntervalSeconds
                Roots = o.Roots.Length
                Runs = runs
                Unreadable = unreadable
                Views = views
                NoView = none
                Pairs = pairs o kinds views
            }

        let html = Page.render model
        span.Add(Attr.int "runs" runs.Length)
        span.Add(Attr.int "views" views.Length)
        span.Add(Attr.int "comparisons" model.Pairs.Length)
        span.Add(Attr.int "chars" html.Length)
        { Html = html; Model = model }

    /// Writes the page atomically: a temporary file beside it, then a rename over the old one, so a browser that
    /// reloads mid-write never sees half a page. Creates the folder.
    let write (path: string) (output: DashboardOutput) =
        let full = Path.GetFullPath path
        Directory.CreateDirectory(Path.GetDirectoryName full) |> ignore
        let tmp = full + ".tmp"
        File.WriteAllBytes(tmp, output.Bytes)
        File.Move(tmp, full, true)

    /// Renders the Markdown report of every view with a T1 template into <run>/report, as `lrm report` does by
    /// default, so the dashboard's report links resolve. Returns the reports written.
    let writeReports (output: DashboardOutput) =
        [
            for v in output.Model.Views do
                match v.Template, v.Error with
                | Some name, None ->
                    let report: Reports.Output =
                        {
                            Template = name
                            Markdown = v.Markdown
                            Svgs =
                                v.Charts
                                |> List.map (fun (file, c) ->
                                    file, ChartSvg.render (Path.GetFileNameWithoutExtension file) c)
                        }

                    yield!
                        Reports.write (Path.Combine(v.Entry.Dir, RunLogNames.ReportDir)) report
                        |> List.truncate 1
                | _ -> ()
        ]

    /// What the folders hold now: every run folder with the names, sizes and write times of its manifest and events
    /// files. Watch mode rebuilds when it changes. A root that is missing, cannot be listed or goes away while it is
    /// listed holds no runs; a run folder that cannot be listed adds only its name.
    let fingerprint (roots: RunsRoot list) =
        let sb = StringBuilder()

        for root in roots do
            sb.Append(root.Path).Append('\n') |> ignore

            let dirs =
                try
                    if Directory.Exists root.Path then
                        Directory.GetDirectories root.Path
                    else
                        [||]
                with
                | :? IOException
                | :? UnauthorizedAccessException -> [||]

            for dir in dirs |> Array.sortWith (fun x y -> String.CompareOrdinal(x, y)) do
                sb.Append(Path.GetFileName dir).Append('\n') |> ignore

                let files =
                    try
                        Directory.GetFiles dir
                        |> Array.filter (fun f ->
                            let n = Path.GetFileName f
                            n = RunManifest.FileName || (RunLogReader.nodeOfFile f).IsSome)
                        |> Array.sortWith (fun x y -> String.CompareOrdinal(x, y))
                    with
                    | :? IOException
                    | :? UnauthorizedAccessException -> [||]

                for f in files do
                    try
                        let info = FileInfo f

                        sb
                            .Append(' ')
                            .Append(info.Name)
                            .Append(' ')
                            .Append(Invariant.i64 info.Length)
                            .Append(' ')
                            .Append(Invariant.i64 info.LastWriteTimeUtc.Ticks)
                            .Append('\n')
                        |> ignore
                    with
                    | :? IOException
                    | :? UnauthorizedAccessException -> ()

        sb.ToString()

    let private sameBytes (x: byte[]) (y: byte[]) =
        x.Length = y.Length
        && MemoryExtensions.SequenceEqual(ReadOnlySpan<byte>(x), ReadOnlySpan<byte>(y))

    /// Watch mode: builds and writes the page, then polls the folders every `poll` and rebuilds when they
    /// change, writing only when the bytes differ. `onWrite` is told about every write. Stops when `stop` is
    /// cancelled. An I/O error while the folders are listed or read (a run deleted, renamed or collected meanwhile) or
    /// while the page is written (the file held open) is retried at the next poll; it never ends the watch.
    let watch
        (o: DashboardOptions)
        (path: string)
        (poll: TimeSpan)
        (stop: CancellationToken)
        (onWrite: DashboardOutput -> unit)
        =
        let o = { o with Watch = true }
        let mutable seen = None
        let mutable written: byte[] = null

        while not stop.IsCancellationRequested do
            try
                let now = fingerprint o.Roots

                if seen <> Some now then
                    let output = build o
                    let bytes = output.Bytes

                    if isNull written || not (sameBytes bytes written) then
                        write path output
                        written <- bytes
                        onWrite output

                    seen <- Some now
            with
            | :? IOException
            | :? UnauthorizedAccessException -> ()

            stop.WaitHandle.WaitOne poll |> ignore
