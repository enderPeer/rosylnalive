namespace Lrm.Telemetry

open System
open System.IO
open System.Text
open System.Text.Unicode
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
        /// The node id this process had before it restarted (the header's restart_of), or "".
        RestartOf: string
        /// This node's clock minus the creating node's clock, in ns, from the creator's last run.clock_offset record
        /// for it; 0 when none was recorded. The reader has already subtracted it from every TimeNs of this node.
        ClockOffsetNs: int64
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
        /// run.end, run.clock_offset) are not included. Times are on the creating node's clock (see ClockOffsetNs).
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

    /// The bytes of one line as text: strict UTF-8, so a damaged byte makes the line a problem instead of an
    /// exception or a silent U+FFFD.
    let private decodeLine (bytes: byte[]) (offset: int) (count: int) : Result<RunEvent, string> =
        if count = 0 then
            Error "empty line"
        elif not (Utf8.IsValid(ReadOnlySpan<byte>(bytes, offset, count))) then
            Error "invalid UTF-8"
        else
            EventLine.decode (Encoding.UTF8.GetString(bytes, offset, count))

    /// Streams the lines of one events file as (line number from 1, parsed record or error). The file is split on
    /// LF bytes and each line is checked as UTF-8 on its own, so a file cut inside a multi-byte character or holding
    /// a stray byte still reads: the bad line is an error and every other line parses. A last line without its LF
    /// (cut off while being written) is reported as an error instead of being parsed.
    let readLines (path: string) : seq<int * Result<RunEvent, string>> =
        seq {
            use fs =
                new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite ||| FileShare.Delete, 65536)

            let buffer: byte[] = Array.zeroCreate 65536
            let mutable line: byte[] = Array.zeroCreate 4096
            let mutable length = 0
            let mutable number = 0
            let mutable read = fs.Read(buffer, 0, buffer.Length)

            while read > 0 do
                let mutable start = 0

                while start < read do
                    let lf = Array.IndexOf(buffer, 10uy, start, read - start)
                    let stop = if lf < 0 then read else lf
                    let n = stop - start

                    if length + n > line.Length then
                        Array.Resize(&line, max (length + n) (line.Length * 2))

                    Buffer.BlockCopy(buffer, start, line, length, n)
                    length <- length + n

                    if lf < 0 then
                        start <- read
                    else
                        number <- number + 1
                        // A UTF-8 byte order mark is not part of the format, but it is not worth losing a line over.
                        let skip =
                            if
                                number = 1
                                && length >= 3
                                && line[0] = 0xEFuy
                                && line[1] = 0xBBuy
                                && line[2] = 0xBFuy
                            then
                                3
                            else
                                0

                        yield number, decodeLine line skip (length - skip)
                        length <- 0
                        start <- lf + 1

                read <- fs.Read(buffer, 0, buffer.Length)

            if length > 0 then
                yield number + 1, Error "the last line is cut off (no final LF)"
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

    let private isStructural (e: RunEvent) =
        e.Kind = TelemetryKind.Event
        && (Invariant.equals e.Name RunLogNames.NodeHeader
            || Invariant.equals e.Name RunLogNames.NodeEnd
            || Invariant.equals e.Name RunLogNames.ClockOffset)

    /// One file read: the node summary (no clock offset applied yet), its data records, and the clock offsets it
    /// records for other nodes as (target, offset ns), the last record per target, sorted by target.
    let private readFile (node: string) (path: string) =
        let problems = ResizeArray<string>()
        let events = ResizeArray<RunEvent>()
        let offsets = Collections.Generic.Dictionary<string, int64>(StringComparer.Ordinal)
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
                | RunLogNames.ClockOffset when ev.Kind = TelemetryKind.Event ->
                    match ev.TryStr "target", ev.TryInt "offset_ns" with
                    | Some target, Some offset -> offsets[target] <- offset
                    | _ -> problems.Add(sprintf "line %d: run.clock_offset without target and offset_ns" number)
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
                RestartOf = header |> Option.bind (fun h -> h.TryStr "restart_of") |> Option.defaultValue ""
                ClockOffsetNs = 0L
                Problems = List.ofSeq problems
            }

        let recorded =
            offsets
            |> Seq.map (fun kv -> kv.Key, kv.Value)
            |> Seq.sortWith (fun (a, _) (b, _) -> String.CompareOrdinal(a, b))
            |> List.ofSeq

        log, List.ofSeq events, recorded

    /// Moves a node's records onto the creating node's clock.
    let private shift (offset: int64) (events: RunEvent list) =
        if offset = 0L then
            events
        else
            events |> List.map (fun e -> { e with TimeNs = e.TimeNs - offset })

    /// Reads one events file on its own: its node summary and its data records in seq order, on the node's own clock
    /// (clock offsets live in the creating node's file; RunLogReader.load applies them).
    let readNode (node: string) (path: string) : NodeLog * RunEvent list =
        let log, events, _ = readFile node path
        log, events

    /// The clock offsets the creating node recorded (RunLog contract, "Runs across hosts"), by target node.
    let private creatorOffsets (creator: string) (files: (string * string) list) =
        match files |> List.tryFind (fun (n, _) -> Invariant.equals n creator) with
        | None -> Map.empty
        | Some(node, path) ->
            let _, _, offsets = readFile node path
            Map.ofList offsets

    /// Streams the data records of a run: nodes in id order, each in seq order, times on the creating node's clock.
    /// Bad lines are skipped.
    let stream (runDir: string) : seq<RunEvent> =
        seq {
            let files = eventFiles runDir

            let offsets =
                match RunManifest.read runDir with
                | Ok m -> creatorOffsets m.Node files
                | Error _ -> Map.empty

            for node, path in files do
                let offset = offsets.TryFind node |> Option.defaultValue 0L

                for _, parsed in readLines path do
                    match parsed with
                    | Ok ev when not (isStructural ev) && Invariant.equals ev.Node node ->
                        yield { ev with TimeNs = ev.TimeNs - offset }
                    | _ -> ()
        }

    /// Reads a whole run folder. Error only when the manifest is missing or invalid; problems in event files are
    /// reported in RunLog.Problems.
    let load (runDir: string) : Result<RunLog, string> =
        let dir = Path.GetFullPath runDir

        match RunManifest.read dir with
        | Error e -> Error e
        | Ok manifest ->
            let read = eventFiles dir |> List.map (fun (node, path) -> readFile node path)

            let offsets =
                read
                |> List.tryFind (fun (n, _, _) -> Invariant.equals n.Node manifest.Node)
                |> Option.map (fun (_, _, o) -> Map.ofList o)
                |> Option.defaultValue Map.empty

            let nodes =
                read
                |> List.map (fun (n, events, recorded) ->
                    let offset = offsets.TryFind n.Node |> Option.defaultValue 0L

                    let problems =
                        if recorded.IsEmpty || Invariant.equals n.Node manifest.Node then
                            n.Problems
                        else
                            n.Problems
                            @ [
                                "run.clock_offset records from a node that did not create the run are ignored"
                            ]

                    { n with
                        ClockOffsetNs = offset
                        Problems = problems
                    },
                    shift offset events)

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
