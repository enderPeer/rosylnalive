namespace Lrm.Telemetry

open System
open System.IO
open Lrm.Core

/// Names of the structural records and files of a run (RunLog contract).
[<RequireQualifiedAccess>]
module RunLogNames =
    /// The header record, seq 0 of every events file: who the node is and its execution class.
    [<Literal>]
    let NodeHeader = "run.node"

    /// The trailer record, the last line of a finished events file: counts of written, dropped and invalid records.
    [<Literal>]
    let NodeEnd = "run.end"

    /// A clock offset the creating node measured for another node (RunLog contract, "Runs across hosts").
    [<Literal>]
    let ClockOffset = "run.clock_offset"

    /// events.<node>.jsonl
    let eventsFile (node: NodeId) = "events." + node.Value + ".jsonl"

    /// Free-form files a run produces.
    [<Literal>]
    let ArtifactsDir = "artifacts"

    /// Rendered reports (the default output of lrm report).
    [<Literal>]
    let ReportDir = "report"

    /// The node id of a process that restarts as `node` for the `attempt`-th time (1, 2, ...): "<node>-r<attempt>".
    /// Events files are never reopened, so a restarted process always writes a new file under a new id.
    let restartedNode (node: NodeId) (attempt: int) =
        if attempt < 1 then
            invalidArg "attempt" "restart attempts count from 1"

        NodeId.parse (node.Value + "-r" + Invariant.i32 attempt)

/// Options for a new run (RunContext.Create).
type RunOptions =
    {
        /// What the run is for: a template name such as "parity", or another safe name.
        Kind: string
        Title: string
        /// This process's node id; "n0" for a single-process run.
        Node: NodeId
        /// The execution class of this node, recorded in the manifest and in the node header.
        ExecutionClass: ExecutionClass option
        /// Free-form parameters for the manifest.
        Params: (string * string) list
        /// True for synthetic runs made to review reports.
        Synthetic: bool
        GitSha: string option
        /// The run id; a fresh one (RunId.fresh) when None.
        RunId: RunId option
        /// The folder that holds run folders; LrmPaths.runs () when None.
        RunsRoot: string option
        /// The clock for every timestamp (Unix ns); TelemetryClock.system when None. Tests pass a virtual clock.
        Clock: (unit -> int64) option
        /// How often buffered lines reach the file; TimeSpan.Zero flushes only when the buffer fills and at the end.
        FlushInterval: TimeSpan
    }

[<RequireQualifiedAccess>]
module RunOptions =
    /// Options for a single-node run of `kind` with defaults for everything else.
    let create (kind: string) (title: string) =
        {
            Kind = kind
            Title = title
            Node = NodeId.local
            ExecutionClass = None
            Params = []
            Synthetic = false
            GitSha = None
            RunId = None
            RunsRoot = None
            Clock = None
            FlushInterval = TimeSpan.FromSeconds 1.0
        }

/// Options for joining an existing run as another node (RunContext.Join).
type JoinOptions =
    {
        /// This process's node id; never the creator's.
        Node: NodeId
        /// The execution class of this node, recorded in its node header.
        ExecutionClass: ExecutionClass option
        /// The clock for every timestamp (Unix ns); TelemetryClock.system when None.
        Clock: (unit -> int64) option
        /// How often buffered lines reach the file; TimeSpan.Zero flushes only when the buffer fills and at the end.
        FlushInterval: TimeSpan
        /// For a node on another host, which has no copy of the run folder: the creator's manifest. Join creates the
        /// local run folder when it is missing and writes this manifest into it when it has none. None: the folder and
        /// its manifest must already exist.
        Manifest: RunManifest option
        /// The node id this process had before it restarted, recorded in its header as restart_of.
        RestartOf: NodeId option
    }

[<RequireQualifiedAccess>]
module JoinOptions =
    /// Options for joining as `node`, with defaults for everything else.
    let create (node: NodeId) =
        {
            Node = node
            ExecutionClass = None
            Clock = None
            FlushInterval = TimeSpan.FromSeconds 1.0
            Manifest = None
            RestartOf = None
        }

/// One node's part of a run (RunLog contract): the run folder, this node's events file, and the telemetry facade that
/// writes to it. The node that creates the run also owns the manifest. Typical use:
///
///     use run = RunContext.Create(RunOptions.create "parity" "CPU parity")
///     use _ = run.Install()
///     ... work that reports through Telemetry ...
///     run.Finish RunStatus.Ok
[<Sealed>]
type RunContext
    private
    (dir: string, runId: RunId, node: NodeId, manifest: RunManifest option, sink: JsonlSink, clock: unit -> int64) =
    let telemetry = Telemetry.ofSinkWithClock clock sink
    let gate = obj ()
    let mutable finished = false

    static let classAttrs (c: ExecutionClass option) =
        match c with
        | None -> []
        | Some c ->
            [
                Attr.str "backend" (ExecutionBackend.name c.Backend)
                Attr.str "device_id" c.DeviceId
                Attr.str "driver" c.Driver
                Attr.str "shader_set_hash" (c.ShaderSetHash |> Option.map _.Hex |> Option.defaultValue "")
                Attr.str "arch" c.Arch
                Attr.str "xclass" (ExecutionClass.id c).Value.Hex
            ]

    static let header (clock: unit -> int64) (role: string) (c: ExecutionClass option) (restartOf: NodeId option) =
        {
            Kind = TelemetryKind.Event
            Name = RunLogNames.NodeHeader
            TimeNs = clock ()
            DurationNs = 0L
            Value = 0.0
            Unit = TelemetryUnits.NoUnit
            GpuStartNs = 0L
            Attrs =
                Attr.str "role" role :: classAttrs c
                @ (restartOf
                   |> Option.map (fun n -> Attr.str "restart_of" n.Value)
                   |> Option.toList)
            SpanId = 0L
            ParentId = 0L
        }

    /// Creates the run folder, writes the manifest (status running) and opens this node's events file. The folder
    /// must not exist yet.
    static member Create(options: RunOptions) : RunContext =
        let kind = SafeName.check "run kind" options.Kind
        let runId = options.RunId |> Option.defaultWith RunId.fresh
        let root = options.RunsRoot |> Option.defaultWith LrmPaths.runs
        let dir = Path.Combine(Path.GetFullPath root, runId.Value)
        let clock = options.Clock |> Option.defaultValue TelemetryClock.system

        match options.ExecutionClass |> Option.map ExecutionClass.validate with
        | Some(Error e) -> invalidArg "options" (sprintf "invalid ExecutionClass: %s" e)
        | _ -> ()

        if Directory.Exists dir then
            invalidOp (sprintf "run folder %s already exists" dir)

        Directory.CreateDirectory dir |> ignore

        let manifest =
            {
                Schema = RunManifest.Schema
                RunId = runId.Value
                Kind = kind
                Title = options.Title
                Status = RunStatus.Running
                CreatedUtc = Json.isoTime (clock ())
                EndedUtc = None
                Node = options.Node.Value
                Synthetic = options.Synthetic
                GitSha = options.GitSha
                ExecutionClass = options.ExecutionClass
                Driver = options.ExecutionClass |> Option.map _.Driver
                Params = options.Params |> List.sortBy fst
            }

        RunManifest.write dir manifest

        let sink =
            new JsonlSink(Path.Combine(dir, RunLogNames.eventsFile options.Node), options.Node, options.FlushInterval)

        sink.Write(header clock "creator" options.ExecutionClass None)
        new RunContext(dir, runId, options.Node, Some manifest, sink, clock)

    /// Joins an existing run as another node: opens events.<node>.jsonl in `runDir` (it must not exist yet) and
    /// writes the node header. The manifest stays with the creating node. With options.Manifest, `runDir` may be a
    /// local copy of the run folder on another host (RunLog contract, "Runs across hosts").
    static member Join(runDir: string, options: JoinOptions) : RunContext =
        let dir = Path.GetFullPath runDir
        let node = options.Node

        match options.Manifest with
        | Some m ->
            if not (Invariant.equals (Path.GetFileName dir) m.RunId) then
                invalidArg "runDir" (sprintf "the run folder %s is not named after run %s" dir m.RunId)

            Directory.CreateDirectory dir |> ignore

            if not (File.Exists(Path.Combine(dir, RunManifest.FileName))) then
                RunManifest.write dir m
        | None -> ()

        let manifest =
            match RunManifest.read dir with
            | Ok m -> m
            | Error e -> invalidOp (sprintf "cannot join run: %s" e)

        match options.Manifest with
        | Some m when not (Invariant.equals m.RunId manifest.RunId) ->
            invalidOp (sprintf "cannot join run %s: the folder holds run %s" m.RunId manifest.RunId)
        | _ -> ()

        if Invariant.equals manifest.Node node.Value then
            invalidArg "node" (sprintf "node %s created this run; a joining node needs its own id" node.Value)

        let clock = options.Clock |> Option.defaultValue TelemetryClock.system

        let sink =
            new JsonlSink(Path.Combine(dir, RunLogNames.eventsFile node), node, options.FlushInterval)

        sink.Write(header clock "joined" options.ExecutionClass options.RestartOf)
        new RunContext(dir, RunId.parse manifest.RunId, node, None, sink, clock)

    /// Joins an existing run in a shared run folder as another node, flushing every second.
    static member Join
        (runDir: string, node: NodeId, executionClass: ExecutionClass option, clock: (unit -> int64) option)
        : RunContext =
        RunContext.Join(
            runDir,
            { JoinOptions.create node with
                ExecutionClass = executionClass
                Clock = clock
            }
        )

    member _.RunId = runId
    member _.Directory = dir
    member _.Node = node

    /// The facade that writes this node's events file. Install it to make it ambient.
    member _.Telemetry = telemetry

    /// The sink, for its counters.
    member _.Sink = sink

    /// True for the node that created the run and owns its manifest.
    member _.IsCreator = manifest.IsSome

    /// <run>/artifacts, created on first use.
    member _.ArtifactsDirectory =
        Directory.CreateDirectory(Path.Combine(dir, RunLogNames.ArtifactsDir)).FullName

    /// Makes this run's facade ambient until the result is disposed.
    member _.Install() = Telemetry.install telemetry

    /// Records, on the creating node, that `target`'s clock reads `offsetNs` ns ahead of this node's (negative: behind),
    /// measured with a round trip of `rttNs` ns. Readers subtract the last offset recorded for a node from all of its
    /// times, so every node's records share the creator's time axis.
    member _.RecordClockOffset(target: NodeId, offsetNs: int64, rttNs: int64) =
        if manifest.IsNone then
            invalidOp "only the node that created the run records clock offsets"

        if Invariant.equals target.Value node.Value then
            invalidArg "target" "the creating node's own clock is the reference"

        sink.Write
            {
                Kind = TelemetryKind.Event
                Name = RunLogNames.ClockOffset
                TimeNs = clock ()
                DurationNs = 0L
                Value = 0.0
                Unit = TelemetryUnits.NoUnit
                GpuStartNs = 0L
                Attrs =
                    [
                        Attr.str "target" target.Value
                        Attr.int64 "offset_ns" offsetNs
                        Attr.int64 "rtt_ns" rttNs
                    ]
                SpanId = 0L
                ParentId = 0L
            }

    /// Writes the run.end trailer, closes the events file and, on the creating node, records the status and end time
    /// in the manifest. Later calls do nothing.
    member _.Finish(status: RunStatus) =
        let first =
            lock gate (fun () ->
                if finished then
                    false
                else
                    finished <- true
                    true)

        if first then
            let now = clock ()

            sink.Write
                {
                    Kind = TelemetryKind.Event
                    Name = RunLogNames.NodeEnd
                    TimeNs = now
                    DurationNs = 0L
                    Value = 0.0
                    Unit = TelemetryUnits.NoUnit
                    GpuStartNs = 0L
                    Attrs =
                        [
                            Attr.str "status" (RunStatus.name status)
                            Attr.int64 "records" sink.Records
                            Attr.int64 "dropped" sink.Dropped
                            Attr.int64 "invalid" sink.Invalid
                        ]
                    SpanId = 0L
                    ParentId = 0L
                }

            sink.Close()

            match manifest with
            | Some m ->
                RunManifest.write
                    dir
                    { m with
                        Status = status
                        EndedUtc = Some(Json.isoTime now)
                    }
            | None -> ()

    interface IDisposable with
        /// Finishes with status aborted unless Finish was called.
        member x.Dispose() = x.Finish RunStatus.Aborted
