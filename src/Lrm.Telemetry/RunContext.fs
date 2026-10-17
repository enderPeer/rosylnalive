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

    /// events.<node>.jsonl
    let eventsFile (node: NodeId) = "events." + node.Value + ".jsonl"

    /// Free-form files a run produces.
    [<Literal>]
    let ArtifactsDir = "artifacts"

    /// Rendered reports (the default output of lrm report).
    [<Literal>]
    let ReportDir = "report"

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

    static let header (clock: unit -> int64) (role: string) (c: ExecutionClass option) =
        {
            Kind = TelemetryKind.Event
            Name = RunLogNames.NodeHeader
            TimeNs = clock ()
            DurationNs = 0L
            Value = 0.0
            Unit = TelemetryUnits.NoUnit
            GpuStartNs = 0L
            Attrs = Attr.str "role" role :: classAttrs c
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

        sink.Write(header clock "creator" options.ExecutionClass)
        new RunContext(dir, runId, options.Node, Some manifest, sink, clock)

    /// Joins an existing run as another node: opens events.<node>.jsonl in `runDir` (it must not exist yet) and
    /// writes the node header. The manifest stays with the creating node.
    static member Join
        (runDir: string, node: NodeId, executionClass: ExecutionClass option, clock: (unit -> int64) option)
        : RunContext =
        let dir = Path.GetFullPath runDir

        let manifest =
            match RunManifest.read dir with
            | Ok m -> m
            | Error e -> invalidOp (sprintf "cannot join run: %s" e)

        if Invariant.equals manifest.Node node.Value then
            invalidArg "node" (sprintf "node %s created this run; a joining node needs its own id" node.Value)

        let clock = clock |> Option.defaultValue TelemetryClock.system

        let sink =
            new JsonlSink(Path.Combine(dir, RunLogNames.eventsFile node), node, TimeSpan.FromSeconds 1.0)

        sink.Write(header clock "joined" executionClass)
        new RunContext(dir, RunId.parse manifest.RunId, node, None, sink, clock)

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
