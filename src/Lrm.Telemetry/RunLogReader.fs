namespace Lrm.Telemetry

open System
open System.IO
open System.Text
open Lrm.Core

/// What one node's events file says about the node (RunLog contract).
type NodeLog =
    {
        Node: string
        /// The events file.
        File: string
        /// "creator" or "joined", from the run.node header; "" without a header.
        Role: string
        /// The node's execution class, from the run.node header.
        ExecutionClass: ExecutionClass option
        /// Data records read (structural records excluded).
        Records: int
        /// True when the file ends with a run.end trailer.
        Ended: bool
        /// The status in the trailer.
        Status: RunStatus option
        /// The trailer's counts: records lost to I/O errors or written after close, and records with invalid names.
        Dropped: int64
        Invalid: int64
        /// Everything wrong with this file: malformed lines, seq gaps, a missing header or trailer, a cut-off last line.
        Problems: string list
    }

/// A whole run, read back (RunLog contract).
type RunLog =
    {
        Directory: string
        Manifest: RunManifest
        /// One entry per events file, ordered by node id (ordinal).
        Nodes: NodeLog list
        /// Every data record of every node: nodes in id order, each node in seq order. Structural records (run.node,
        /// run.end) are not included.
        Events: RunEvent list
    }

    /// Every problem found in any file, prefixed by its node.
    member r.Problems =
        r.Nodes
        |> List.collect (fun n -> n.Problems |> List.map (fun p -> n.Node + ": " + p))

    /// The records named `name`.
    member r.Named(name: string) =
        r.Events |> List.filter (fun e -> Invariant.equals e.Name name)

    /// The records whose name starts with `prefix`.
    member r.Prefixed(prefix: string) =
        r.Events
        |> List.filter (fun e -> e.Name.StartsWith(prefix, StringComparison.Ordinal))

    /// The execution class of a node: its header's, else the manifest's for the creating node.
    member r.ClassOf(node: string) =
        r.Nodes
        |> List.tryFind (fun n -> Invariant.equals n.Node node)
        |> Option.bind _.ExecutionClass
        |> Option.orElse (
            if Invariant.equals node r.Manifest.Node then
                r.Manifest.ExecutionClass
            else
                None
        )

/// Reads run folders written by RunContext and JsonlSink (RunLog contract). The reader is tolerant: a bad line is
/// skipped and reported as a problem of its node, so a partly written or damaged log still gives a report.
[<RequireQualifiedAccess>]
module RunLogReader =
    /// The node of an events file name ("events.n0.jsonl" -> "n0").
    let nodeOfFile (path: string) =
        let name = Path.GetFileName path

        if
            name.StartsWith("events.", StringComparison.Ordinal)
            && name.EndsWith(".jsonl", StringComparison.Ordinal)
            && name.Length > "events..jsonl".Length
        then
            Some(name.Substring(7, name.Length - 7 - 6))
        else
            None

    /// The events files of a run folder, ordered by node id.
    let eventFiles (runDir: string) =
        if Directory.Exists runDir then
            Directory.GetFiles(runDir, "events.*.jsonl")
            |> Array.choose (fun f -> nodeOfFile f |> Option.map (fun n -> n, f))
            |> Array.sortBy fst
            |> List.ofArray
        else
            []

    let private endsWithNewline (path: string) =
        use fs =
            new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite ||| FileShare.Delete)

        if fs.Length = 0L then
            true
        else
            fs.Seek(-1L, SeekOrigin.End) |> ignore
            fs.ReadByte() = int '\n'

    /// Streams the lines of one events file as (line number from 1, parsed record or error). A last line without its
    /// LF (cut off while being written) is reported as an error instead of being parsed.
    let readLines (path: string) : seq<int * Result<RunEvent, string>> =
        seq {
            let complete = endsWithNewline path

            use fs =
                new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite ||| FileShare.Delete)

            use reader = new StreamReader(fs, UTF8Encoding(false, true))
            let mutable number = 0
            let mutable line = reader.ReadLine()

            while not (isNull line) do
                number <- number + 1
                let next = reader.ReadLine()

                if isNull next && not complete then
                    yield number, Error "the last line is cut off (no final LF)"
                elif line.Length = 0 then
                    yield number, Error "empty line"
                else
                    yield number, EventLine.decode line

                line <- next
        }

    let private classOfHeader (e: RunEvent) =
        match e.TryStr "backend" |> Option.bind ExecutionBackend.tryParse with
        | None -> None
        | Some backend ->
            let c =
                {
                    Backend = backend
                    DeviceId = defaultArg (e.TryStr "device_id") ""
                    Driver = defaultArg (e.TryStr "driver") ""
                    ShaderSetHash = e.TryStr "shader_set_hash" |> Option.bind ContentId.tryParse
                    Arch = defaultArg (e.TryStr "arch") ""
                }

            match ExecutionClass.validate c with
            | Ok c -> Some c
            | Error _ -> None

    /// Reads one events file: its node summary and its data records in seq order.
    let readNode (node: string) (path: string) : NodeLog * RunEvent list =
        let problems = ResizeArray<string>()
        let events = ResizeArray<RunEvent>()
        let mutable expected = 0L
        let mutable header: RunEvent option = None
        let mutable trailer: RunEvent option = None

        for number, parsed in readLines path do
            match parsed with
            | Error e -> problems.Add(sprintf "line %d: %s" number e)
            | Ok ev ->
                if not (Invariant.equals ev.Node node) then
                    problems.Add(sprintf "line %d: node '%s' in the file of node '%s'" number ev.Node node)

                if ev.Seq <> expected then
                    problems.Add(sprintf "line %d: seq %d where %d was expected (records lost)" number ev.Seq expected)

                expected <- ev.Seq + 1L

                if trailer.IsSome then
                    problems.Add(sprintf "line %d: record after the run.end trailer" number)

                match ev.Name with
                | RunLogNames.NodeHeader when ev.Kind = TelemetryKind.Event ->
                    if ev.Seq <> 0L || header.IsSome then
                        problems.Add(sprintf "line %d: run.node header that is not the first record" number)

                    header <- Some ev
                | RunLogNames.NodeEnd when ev.Kind = TelemetryKind.Event ->
                    match ev.TryInt "records" with
                    | Some n when n <> ev.Seq ->
                        problems.Add(sprintf "run.end counts %d records but is record %d" n ev.Seq)
                    | _ -> ()

                    trailer <- Some ev
                | _ -> events.Add ev

        if header.IsNone then
            problems.Add "no run.node header"

        match trailer with
        | None -> problems.Add "no run.end trailer (the node is still running, or it stopped without finishing)"
        | Some t ->
            match t.TryInt "dropped" with
            | Some d when d > 0L -> problems.Add(sprintf "%d records were dropped by the sink" d)
            | _ -> ()

            match t.TryInt "invalid" with
            | Some i when i > 0L -> problems.Add(sprintf "%d records with invalid names were refused" i)
            | _ -> ()

        let log =
            {
                Node = node
                File = path
                Role = header |> Option.bind (fun h -> h.TryStr "role") |> Option.defaultValue ""
                ExecutionClass = header |> Option.bind classOfHeader
                Records = events.Count
                Ended = trailer.IsSome
                Status =
                    trailer
                    |> Option.bind (fun t -> t.TryStr "status")
                    |> Option.bind RunStatus.tryParse
                Dropped = trailer |> Option.bind (fun t -> t.TryInt "dropped") |> Option.defaultValue 0L
                Invalid = trailer |> Option.bind (fun t -> t.TryInt "invalid") |> Option.defaultValue 0L
                Problems = List.ofSeq problems
            }

        log, List.ofSeq events

    /// Streams the data records of a run: nodes in id order, each in seq order. Bad lines are skipped.
    let stream (runDir: string) : seq<RunEvent> =
        seq {
            for node, path in eventFiles runDir do
                for _, parsed in readLines path do
                    match parsed with
                    | Ok ev when
                        not (
                            ev.Kind = TelemetryKind.Event
                            && (Invariant.equals ev.Name RunLogNames.NodeHeader
                                || Invariant.equals ev.Name RunLogNames.NodeEnd)
                        )
                        && Invariant.equals ev.Node node
                        ->
                        yield ev
                    | _ -> ()
        }

    /// Reads a whole run folder. Error only when the manifest is missing or invalid; problems in event files are
    /// reported in RunLog.Problems.
    let load (runDir: string) : Result<RunLog, string> =
        let dir = Path.GetFullPath runDir

        match RunManifest.read dir with
        | Error e -> Error e
        | Ok manifest ->
            let nodes = eventFiles dir |> List.map (fun (node, path) -> readNode node path)

            Ok
                {
                    Directory = dir
                    Manifest = manifest
                    Nodes = nodes |> List.map fst
                    Events = nodes |> List.collect snd
                }

    /// Reads a whole run folder; raises InvalidOperationException when the manifest is missing or invalid.
    let loadOrFail (runDir: string) =
        match load runDir with
        | Ok log -> log
        | Error e -> invalidOp e
