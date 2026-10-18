namespace Lrm.Dashboard

open System
open System.Globalization
open System.IO
open Lrm.Core
open Lrm.Telemetry

/// Where a run view points for its Markdown report.
[<RequireQualifiedAccess>]
type ReportLink =
    /// A relative href from the dashboard file.
    | Href of string
    /// The report's path from its runs folder ("runs/<runId>/report/parity.md"), shown as text: the run lies outside
    /// the dashboard file's folder, and its only link would be an absolute file: URI, which would put a local path into
    /// the page and make its bytes depend on where the folders are (ADR T2-0001, decision 7).
    | Text of string

/// A folder the dashboard reads runs from (RunLog contract: one sub-folder per run).
type RunsRoot =
    {
        /// The folder that holds the run folders.
        Path: string
        /// Every run under this root gets a view, whatever DashboardOptions.PerKind says (the synthetic samples).
        ViewAll: bool
        /// Where a run's Markdown report is (run folder, template name), or None.
        Report: string -> string -> ReportLink option
    }

[<RequireQualifiedAccess>]
module RunsRoot =
    /// A folder of recorded runs. A run's report is <run>/report/<template>.md, where `lrm report` writes by default:
    /// a relative link when the run folder lies inside `outDir` (the dashboard file's folder), otherwise its path from
    /// the runs folder as text.
    let runs (outDir: string) (path: string) =
        let root = Path.GetFullPath path

        {
            Path = root
            ViewAll = false
            Report =
                fun runDir template ->
                    let file = Path.Combine(runDir, RunLogNames.ReportDir, template + ".md")

                    match Html.href outDir file with
                    | Some href -> Some(ReportLink.Href href)
                    | None ->
                        let name = Path.GetFileName(Path.TrimEndingDirectorySeparator root)
                        let rel = Path.GetRelativePath(root, file).Replace(Path.DirectorySeparatorChar, '/')
                        Some(ReportLink.Text(if String.IsNullOrEmpty name then rel else name + "/" + rel))
        }

    /// A folder of T1's synthetic runs (Synthetic.write): every run gets a view, and its report link points at the
    /// committed sample report <samplesDir>/<template>.md, as a relative href from `outDir`.
    let samples (outDir: string) (samplesDir: string) (path: string) =
        {
            Path = Path.GetFullPath path
            ViewAll = true
            Report =
                fun _ template ->
                    Some(ReportLink.Href(Html.relativeHref outDir (Path.Combine(samplesDir, template + ".md"))))
        }

/// One run folder with a readable manifest, as the runs index shows it. Reading it needs only the manifest and the
/// names and sizes of the events files, so the index stays cheap for any number of runs.
type RunEntry =
    {
        /// The run folder, a full path.
        Dir: string
        /// The position of its root in the root list.
        Root: int
        Manifest: RunManifest
        /// The number of events files (one per node).
        Nodes: int
        /// The total size of the events files in bytes.
        EventBytes: int64
        /// Start and end as parsed from the manifest, when they parse.
        Started: DateTimeOffset option
        Ended: DateTimeOffset option
    }

    member r.Id = r.Manifest.RunId
    member r.Kind = r.Manifest.Kind

    /// End minus start in ns, when the run has finished and both times parse.
    member r.DurationNs =
        match r.Started, r.Ended with
        | Some s, Some e -> Some((e - s).Ticks * 100L)
        | _ -> None

/// A folder under a runs root that is not a readable run: no manifest, or one RunManifest.parse rejects.
type Unreadable =
    {
        Root: int
        /// The folder name.
        Folder: string
        Reason: string
    }

/// Finding runs and choosing which ones get a view (Lrm.Dashboard; ADR T2-0001).
[<RequireQualifiedAccess>]
module Catalog =
    let private parseTime (s: string) =
        match
            DateTimeOffset.TryParse(
                s,
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal ||| DateTimeStyles.AdjustToUniversal
            )
        with
        | true, t -> Some t
        | _ -> None

    /// Reads one run folder: an entry, or why it is not a readable run. The reason never holds a path.
    let read (root: int) (dir: string) : Result<RunEntry, Unreadable> =
        let folder = Path.GetFileName(Path.TrimEndingDirectorySeparator dir)
        let path = Path.Combine(dir, RunManifest.FileName)

        let unreadable reason =
            Error
                {
                    Root = root
                    Folder = folder
                    Reason = reason
                }

        if not (File.Exists path) then
            unreadable "no manifest.json"
        else
            // Shared for reading, writing and deleting: a live run replaces its manifest with a rename
            // (RunManifest.write), which Windows refuses while any reader holds the file without delete sharing, so a
            // plain File.ReadAllText from a watching dashboard could make the run's Finish fail.
            let text =
                try
                    use stream =
                        new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite ||| FileShare.Delete)

                    use reader = new StreamReader(stream, Text.UTF8Encoding(false), true)
                    Ok(reader.ReadToEnd())
                with
                | :? IOException
                | :? UnauthorizedAccessException -> Error "manifest.json cannot be read"

            match text |> Result.bind RunManifest.parse with
            | Error e -> unreadable ("manifest.json: " + e)
            | Ok m ->
                // The folder can change or go away between the manifest and the listing (a run being deleted or
                // collected): then it is listed as not readable, and the page is still built.
                let files =
                    try
                        Ok(RunLogReader.eventFiles dir)
                    with
                    | :? IOException
                    | :? UnauthorizedAccessException -> Error()

                match files with
                | Error() -> unreadable "its files changed while they were read"
                | Ok files ->
                    let bytes =
                        files
                        |> List.sumBy (fun (_, f) ->
                            try
                                FileInfo(f).Length
                            with
                            | :? IOException
                            | :? UnauthorizedAccessException -> 0L)

                    Ok
                        {
                            Dir = Path.GetFullPath dir
                            Root = root
                            Manifest = m
                            Nodes = files.Length
                            EventBytes = bytes
                            Started = parseTime m.CreatedUtc
                            Ended = m.EndedUtc |> Option.bind parseTime
                        }

    /// Newest first: by start time (as parsed, else the text), then by run id, both descending; ordinal throughout.
    let newestFirst (runs: RunEntry list) =
        runs
        |> List.sortWith (fun a b ->
            let byTime =
                match a.Started, b.Started with
                | Some x, Some y -> compare y.UtcTicks x.UtcTicks
                | Some _, None -> -1
                | None, Some _ -> 1
                | None, None -> String.CompareOrdinal(b.Manifest.CreatedUtc, a.Manifest.CreatedUtc)

            if byTime <> 0 then
                byTime
            else
                String.CompareOrdinal(b.Id, a.Id))

    /// Every run folder under the roots, in root order and then folder name (ordinal). A run id seen under an earlier
    /// root shadows the later folder, which is listed as unreadable. Missing roots hold no runs.
    let scan (roots: RunsRoot list) : RunEntry list * Unreadable list =
        let runs = ResizeArray<RunEntry>()
        let bad = ResizeArray<Unreadable>()
        let seen = Collections.Generic.HashSet<string>(StringComparer.Ordinal)

        roots
        |> List.iteri (fun i root ->
            // A root that is missing or cannot be listed holds no runs.
            let dirs =
                try
                    if Directory.Exists root.Path then
                        Directory.GetDirectories root.Path
                    else
                        [||]
                with
                | :? IOException
                | :? UnauthorizedAccessException -> [||]

            for dir in dirs |> Array.sortWith (fun a b -> String.CompareOrdinal(a, b)) do
                match read i dir with
                | Ok r when seen.Add r.Id -> runs.Add r
                | Ok r ->
                    bad.Add
                        {
                            Root = i
                            Folder = Path.GetFileName dir
                            Reason = "run id " + r.Id + " is already listed from an earlier folder"
                        }
                | Error u -> bad.Add u)

        newestFirst (List.ofSeq runs), List.ofSeq bad

    /// The report template of a kind, if T1 has one.
    let templateOf (kind: string) = Lrm.Reports.Reports.tryFind kind

    /// Kinds in display order: the report templates in report order, then every other kind (ordinal).
    let kindOrder (runs: RunEntry list) =
        let known = Lrm.Reports.Reports.names

        let others =
            runs
            |> List.map _.Kind
            |> List.filter (fun k -> not (List.contains k known))
            |> List.distinct
            |> List.sortWith (fun a b -> String.CompareOrdinal(a, b))

        known @ others

/// Why a run has no view.
[<RequireQualifiedAccess>]
type NoView =
    /// Older than the newest PerKind runs of its kind.
    | Older
    /// Its events files are larger than MaxLogBytes.
    | TooLarge of bytes: int64
