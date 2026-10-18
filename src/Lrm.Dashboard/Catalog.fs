namespace Lrm.Dashboard

open System
open System.Globalization
open System.IO
open Lrm.Core
open Lrm.Telemetry

/// A folder the dashboard reads runs from (RunLog contract: one sub-folder per run).
type RunsRoot =
    {
        /// The folder that holds the run folders.
        Path: string
        /// Every run under this root gets a view, whatever DashboardOptions.PerKind says (the synthetic samples).
        ViewAll: bool
        /// The href of a run's Markdown report from the dashboard file (run folder, template name), or None.
        ReportHref: string -> string -> string option
    }

[<RequireQualifiedAccess>]
module RunsRoot =
    /// A folder of recorded runs. A run's report link points at <run>/report/<template>.md, where `lrm report` writes
    /// by default: a relative href when the run folder lies inside `outDir` (the dashboard file's folder), otherwise a
    /// file: URI.
    let runs (outDir: string) (path: string) =
        {
            Path = Path.GetFullPath path
            ViewAll = false
            ReportHref =
                fun runDir template ->
                    Some(Html.href outDir (Path.Combine(runDir, RunLogNames.ReportDir, template + ".md")))
        }

    /// A folder of T1's synthetic runs (Synthetic.write): every run gets a view, and its report link points at the
    /// committed sample report <samplesDir>/<template>.md, as a relative href from `outDir`.
    let samples (outDir: string) (samplesDir: string) (path: string) =
        {
            Path = Path.GetFullPath path
            ViewAll = true
            ReportHref = fun _ template -> Some(Html.relativeHref outDir (Path.Combine(samplesDir, template + ".md")))
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
            let text =
                try
                    Ok(File.ReadAllText path)
                with
                | :? IOException
                | :? UnauthorizedAccessException -> Error "manifest.json cannot be read"

            match text |> Result.bind RunManifest.parse with
            | Error e -> unreadable ("manifest.json: " + e)
            | Ok m ->
                let files = RunLogReader.eventFiles dir

                let bytes =
                    files
                    |> List.sumBy (fun (_, f) ->
                        try
                            FileInfo(f).Length
                        with _ ->
                            0L)

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
