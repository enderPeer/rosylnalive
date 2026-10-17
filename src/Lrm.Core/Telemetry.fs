namespace Lrm.Core

open System
open System.Diagnostics
open System.Threading

/// The value of a telemetry attribute. There are only these four kinds, so every sink encodes them the same way.
[<RequireQualifiedAccess>]
type AttrValue =
    | Str of string
    | Int of int64
    | Float of float
    | Bool of bool

/// One telemetry attribute.
[<Struct>]
type Attr = { Key: string; Value: AttrValue }

/// Attribute constructors: Attr.int "layer" 3, Attr.str "backend" "vulkan".
[<RequireQualifiedAccess>]
module Attr =
    let str (key: string) (value: string) =
        {
            Key = key
            Value = AttrValue.Str value
        }

    let int64 (key: string) (value: int64) =
        {
            Key = key
            Value = AttrValue.Int value
        }

    let int (key: string) (value: int) =
        {
            Key = key
            Value = AttrValue.Int(Operators.int64 value)
        }

    let float (key: string) (value: float) =
        {
            Key = key
            Value = AttrValue.Float value
        }

    let bool (key: string) (value: bool) =
        {
            Key = key
            Value = AttrValue.Bool value
        }

/// What a telemetry record measures.
[<RequireQualifiedAccess>]
type TelemetryKind =
    /// A timed region of host work (Telemetry.span); Value is its duration in ns.
    | Span
    /// A count of occurrences; Value is the increment.
    | Counter
    /// A level at one moment, such as bytes in use or a queue depth; Value is the level.
    | Gauge
    /// One sample of a distribution, such as a latency; Value is the sample.
    | Histogram
    /// Something that happened, described by its attributes; Value is 0.
    | Event
    /// A GPU-timed region from timestamp queries; Value is its GPU duration in ns.
    | GpuInterval
    /// Resource use attributed to the current scope (wall or GPU time, bytes moved, FLOPs, tokens); Value is the amount.
    | Meter
    /// The outcome of a verification, such as a golden comparison; Value is 1 when it passed and 0 when it failed.
    | Check

[<RequireQualifiedAccess>]
module TelemetryKind =
    /// Every kind, in declaration order.
    let all =
        [
            TelemetryKind.Span
            TelemetryKind.Counter
            TelemetryKind.Gauge
            TelemetryKind.Histogram
            TelemetryKind.Event
            TelemetryKind.GpuInterval
            TelemetryKind.Meter
            TelemetryKind.Check
        ]

    /// The short name sinks write: "span", "counter", "gauge", "histogram", "event", "gpu", "meter", "check".
    let name (kind: TelemetryKind) =
        match kind with
        | TelemetryKind.Span -> "span"
        | TelemetryKind.Counter -> "counter"
        | TelemetryKind.Gauge -> "gauge"
        | TelemetryKind.Histogram -> "histogram"
        | TelemetryKind.Event -> "event"
        | TelemetryKind.GpuInterval -> "gpu"
        | TelemetryKind.Meter -> "meter"
        | TelemetryKind.Check -> "check"

/// One telemetry record, as a sink receives it. Records describe the computation; they never feed back into it.
type TelemetryRecord =
    {
        Kind: TelemetryKind
        /// Dotted lowercase name, such as "model.layer" (see TelemetryNames).
        Name: string
        /// Host wall-clock time in Unix nanoseconds (UTC): the start of a span, otherwise when the record was made.
        /// Timestamps are for logs and charts only and never enter a content hash.
        TimeNs: int64
        /// Span: host duration in ns. GpuInterval: GPU duration in ns. Otherwise 0.
        DurationNs: int64
        /// The measurement; see TelemetryKind.
        Value: float
        /// The unit of Value: "ns", "B", "flop", "token", "count", or "" when Value has none.
        Unit: string
        /// GpuInterval: start in the GPU's own timestamp domain, in ns (comparable only within one queue). Otherwise 0.
        GpuStartNs: int64
        /// Scope attributes from Child, outermost first, then the call's own attributes.
        Attrs: Attr list
        /// Span: its id, unique within one root telemetry and starting at 1. Otherwise 0.
        SpanId: int64
        /// The enclosing span's id, or 0 at top level.
        ParentId: int64
    }

/// A span in progress. Disposing it ends the span and emits its record; a second Dispose does nothing.
type ISpan =
    inherit IDisposable
    /// The span's id, or 0 for a no-op span.
    abstract Id: int64
    /// Adds an attribute reported when the span ends, such as a result size.
    abstract Add: attr: Attr -> unit

/// The telemetry facade (TelemetryFacade contract). Every LRM component reports through it; the ambient instance is a
/// no-op until a host installs one (Telemetry.install). Implementations must be thread-safe.
type ITelemetry =
    /// False for the no-op facade. Callers may skip building expensive attributes when it is false.
    abstract Enabled: bool
    /// Starts a span; dispose it to end it. Spans nest through an async-local parent.
    abstract Span: name: string * attrs: Attr list -> ISpan
    /// Adds `delta` occurrences.
    abstract Counter: name: string * delta: float * unit: string * attrs: Attr list -> unit
    /// Records a level.
    abstract Gauge: name: string * value: float * unit: string * attrs: Attr list -> unit
    /// Records one sample of a distribution.
    abstract Histogram: name: string * value: float * unit: string * attrs: Attr list -> unit
    /// Records an event.
    abstract Event: name: string * attrs: Attr list -> unit
    /// Records a GPU-timed region: its start in the GPU timestamp domain and its duration, both in ns.
    abstract GpuInterval: name: string * gpuStartNs: int64 * gpuDurationNs: int64 * attrs: Attr list -> unit
    /// Records resource use attributed to this scope.
    abstract Meter: name: string * amount: float * unit: string * attrs: Attr list -> unit
    /// Records the outcome of a check.
    abstract Check: name: string * passed: bool * attrs: Attr list -> unit
    /// A facade whose records also carry `attrs` (component, layer, stage, node, ...), after this one's own scope.
    abstract Child: attrs: Attr list -> ITelemetry

/// Where records go. T1's run-log writer is one; InMemorySink is the test sink.
type ITelemetrySink =
    /// Receives one record. Called from any thread. A production sink must not throw; it drops and counts instead.
    abstract Write: record: TelemetryRecord -> unit

/// Naming rules and the reserved names (TelemetryFacade contract).
[<RequireQualifiedAccess>]
module TelemetryNames =
    /// GPU backends and the GPU executor.
    [<Literal>]
    let GpuPrefix = "gpu."

    /// The Qwen model and its forward pass.
    [<Literal>]
    let ModelPrefix = "model."

    /// The cluster core and the serving pipeline.
    [<Literal>]
    let ClusterPrefix = "cluster."

    /// One kernel conformance case (ConformanceHarness).
    [<Literal>]
    let ConformanceCase = "conformance.case"

    /// One tokenizer corpus case.
    [<Literal>]
    let TokenizerCase = "tokenizer.case"

    /// One comparison against a golden.
    [<Literal>]
    let GoldenCompare = "golden.compare"

    /// Memory use of one heap arena.
    [<Literal>]
    let MemArena = "mem.arena"

    /// Name prefixes reserved for their owning components.
    let reservedPrefixes = [ GpuPrefix; ModelPrefix; ClusterPrefix ]

    /// Exact names reserved for their owning components.
    let reservedNames = [ ConformanceCase; TokenizerCase; GoldenCompare; MemArena ]

    /// True when `name` is reserved: it starts with gpu., model. or cluster., or is one of the reserved names.
    let isReserved (name: string) =
        not (isNull name)
        && (reservedNames |> List.exists (Invariant.equals name)
            || reservedPrefixes
               |> List.exists (fun p -> name.StartsWith(p, StringComparison.Ordinal)))

    let private isSegmentChar (c: char) =
        (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c = '_'

    /// A record name or attribute key: 1 to 128 characters, dot-separated non-empty segments of a-z, 0-9 and '_',
    /// starting with a letter. For example "model.layer", "gpu.dispatch", "layer", "bytes_moved".
    let isValid (name: string) =
        not (isNull name)
        && name.Length >= 1
        && name.Length <= 128
        && name[0] >= 'a'
        && name[0] <= 'z'
        && name.Split('.')
           |> Array.forall (fun s -> s.Length > 0 && s |> Seq.forall isSegmentChar)

    /// A unit: at most 32 printable ASCII characters without spaces; "" means no unit.
    let isValidUnit (unit: string) =
        not (isNull unit)
        && unit.Length <= 32
        && unit |> Seq.forall (fun c -> c > ' ' && c <= '~')

    /// The first contract violation in a record, or None. InMemorySink enforces this.
    let check (r: TelemetryRecord) : string option =
        let badAttr =
            r.Attrs
            |> List.tryPick (fun a ->
                if not (isValid a.Key) then
                    Some(sprintf "attribute key '%s' is not a valid name" a.Key)
                else
                    match a.Value with
                    | AttrValue.Str null -> Some(sprintf "attribute '%s' has a null string" a.Key)
                    | _ -> None)

        if not (isValid r.Name) then
            Some(sprintf "telemetry name '%s' is not valid (dotted segments of a-z, 0-9 and '_')" r.Name)
        elif not (isValidUnit r.Unit) then
            Some(sprintf "unit '%s' of '%s' is not valid" r.Unit r.Name)
        else
            badAttr

/// Conventional attribute keys for attribution (component, layer, stage, node) and identification.
[<RequireQualifiedAccess>]
module TelemetryKeys =
    [<Literal>]
    let Component = "component"

    [<Literal>]
    let Layer = "layer"

    [<Literal>]
    let Stage = "stage"

    [<Literal>]
    let Node = "node"

    [<Literal>]
    let Backend = "backend"

    [<Literal>]
    let Kernel = "kernel"

    [<Literal>]
    let Arena = "arena"

    [<Literal>]
    let Case = "case"

/// Conventional units.
[<RequireQualifiedAccess>]
module TelemetryUnits =
    [<Literal>]
    let NoUnit = ""

    [<Literal>]
    let Nanoseconds = "ns"

    [<Literal>]
    let Bytes = "B"

    [<Literal>]
    let Flops = "flop"

    [<Literal>]
    let Tokens = "token"

    [<Literal>]
    let Count = "count"

/// Clocks for the facade. A clock returns host wall-clock time in Unix nanoseconds.
[<RequireQualifiedAccess>]
module TelemetryClock =
    let private originTimestamp = Stopwatch.GetTimestamp()
    let private originUnixNs = (DateTime.UtcNow - DateTime.UnixEpoch).Ticks * 100L

    /// The UTC time when the clock was first used plus a monotonic Stopwatch offset, so it never runs backwards
    /// within a process.
    let system () : int64 =
        let elapsed = Stopwatch.GetTimestamp() - originTimestamp
        let f = Stopwatch.Frequency

        originUnixNs
        + (elapsed / f) * 1_000_000_000L
        + (elapsed % f) * 1_000_000_000L / f

[<Sealed>]
type private NoopSpan() =
    static member val Instance = new NoopSpan()

    interface ISpan with
        member _.Id = 0L
        member _.Add _ = ()
        member _.Dispose() = ()

[<Sealed>]
type private NoopTelemetry() =
    static member val Instance = NoopTelemetry()

    interface ITelemetry with
        member _.Enabled = false
        member _.Span(_, _) = NoopSpan.Instance :> ISpan
        member _.Counter(_, _, _, _) = ()
        member _.Gauge(_, _, _, _) = ()
        member _.Histogram(_, _, _, _) = ()
        member _.Event(_, _) = ()
        member _.GpuInterval(_, _, _, _) = ()
        member _.Meter(_, _, _, _) = ()
        member _.Check(_, _, _) = ()
        member this.Child _ = this :> ITelemetry

/// Span ids and the async-local current span, shared by a root facade and all its children.
[<Sealed>]
type private SpanContext() =
    let current = AsyncLocal<int64>()
    let mutable next = 0L

    member _.Current
        with get () = current.Value
        and set (v: int64) = current.Value <- v

    member _.NextId() = Interlocked.Increment(&next)

[<Sealed>]
type private SinkSpan
    (
        sink: ITelemetrySink,
        clock: unit -> int64,
        context: SpanContext,
        name: string,
        attrs: Attr list,
        id: int64,
        parent: int64,
        start: int64
    ) =
    let gate = obj ()
    let mutable extra: Attr list = []
    let mutable ended = 0

    interface ISpan with
        member _.Id = id

        member _.Add(attr: Attr) =
            lock gate (fun () -> extra <- attr :: extra)

        member _.Dispose() =
            if Interlocked.Exchange(&ended, 1) = 0 then
                let finish = clock ()

                if context.Current = id then
                    context.Current <- parent

                let added = lock gate (fun () -> List.rev extra)
                let duration = finish - start

                sink.Write
                    {
                        Kind = TelemetryKind.Span
                        Name = name
                        TimeNs = start
                        DurationNs = duration
                        Value = float duration
                        Unit = TelemetryUnits.Nanoseconds
                        GpuStartNs = 0L
                        Attrs = (if added.IsEmpty then attrs else attrs @ added)
                        SpanId = id
                        ParentId = parent
                    }

[<Sealed>]
type private SinkTelemetry(sink: ITelemetrySink, clock: unit -> int64, context: SpanContext, scope: Attr list) =
    let withScope (attrs: Attr list) =
        if scope.IsEmpty then attrs
        elif attrs.IsEmpty then scope
        else scope @ attrs

    let point kind name value unit attrs gpuStart duration =
        sink.Write
            {
                Kind = kind
                Name = name
                TimeNs = clock ()
                DurationNs = duration
                Value = value
                Unit = unit
                GpuStartNs = gpuStart
                Attrs = withScope attrs
                SpanId = 0L
                ParentId = context.Current
            }

    interface ITelemetry with
        member _.Enabled = true

        member _.Span(name, attrs) =
            let id = context.NextId()
            let parent = context.Current
            let start = clock ()
            context.Current <- id
            new SinkSpan(sink, clock, context, name, withScope attrs, id, parent, start) :> ISpan

        member _.Counter(name, delta, unit, attrs) =
            point TelemetryKind.Counter name delta unit attrs 0L 0L

        member _.Gauge(name, value, unit, attrs) =
            point TelemetryKind.Gauge name value unit attrs 0L 0L

        member _.Histogram(name, value, unit, attrs) =
            point TelemetryKind.Histogram name value unit attrs 0L 0L

        member _.Event(name, attrs) =
            point TelemetryKind.Event name 0.0 TelemetryUnits.NoUnit attrs 0L 0L

        member _.GpuInterval(name, gpuStartNs, gpuDurationNs, attrs) =
            point
                TelemetryKind.GpuInterval
                name
                (float gpuDurationNs)
                TelemetryUnits.Nanoseconds
                attrs
                gpuStartNs
                gpuDurationNs

        member _.Meter(name, amount, unit, attrs) =
            point TelemetryKind.Meter name amount unit attrs 0L 0L

        member _.Check(name, passed, attrs) =
            point TelemetryKind.Check name (if passed then 1.0 else 0.0) TelemetryUnits.NoUnit attrs 0L 0L

        member _.Child(attrs) =
            SinkTelemetry(sink, clock, context, withScope attrs) :> ITelemetry

/// A strict sink for tests: it keeps every record in memory and raises ArgumentException on a record that breaks
/// the naming rules (TelemetryNames.check), so tests catch bad names at once.
[<Sealed>]
type InMemorySink() =
    let records = ResizeArray<TelemetryRecord>()

    /// Every record so far, in arrival order.
    member _.Records: TelemetryRecord list = lock records (fun () -> List.ofSeq records)

    /// The number of records so far.
    member _.Count = lock records (fun () -> records.Count)

    /// Forgets every record.
    member _.Clear() =
        lock records (fun () -> records.Clear())

    interface ITelemetrySink with
        member _.Write(record: TelemetryRecord) =
            match TelemetryNames.check record with
            | Some problem -> invalidArg "record" problem
            | None -> lock records (fun () -> records.Add record)

/// The ambient telemetry (TelemetryFacade contract). Components call these functions; the host decides where the
/// records go. Until a host installs a facade the ambient one is a no-op, and telemetry never changes a computed
/// value either way.
[<RequireQualifiedAccess>]
module Telemetry =
    let private ambient = AsyncLocal<ITelemetry>()

    /// The facade that records nothing.
    let noop: ITelemetry = NoopTelemetry.Instance :> ITelemetry

    /// The ambient facade of this async context, or noop.
    let current () : ITelemetry =
        let t = ambient.Value
        if isNull (box t) then noop else t

    /// Makes `telemetry` ambient for this async context and everything it starts. Disposing the result restores the
    /// previous facade.
    let install (telemetry: ITelemetry) : IDisposable =
        if isNull (box telemetry) then
            nullArg "telemetry"

        let previous = ambient.Value
        ambient.Value <- telemetry

        { new IDisposable with
            member _.Dispose() = ambient.Value <- previous
        }

    /// Makes current().Child attrs ambient until the result is disposed: use _ = Telemetry.scope [ Attr.int "layer" 3 ].
    let scope (attrs: Attr list) : IDisposable = install ((current ()).Child attrs)

    /// A facade that writes to `sink`, timed by `clock` (Unix ns; for example a virtual clock in tests).
    let ofSinkWithClock (clock: unit -> int64) (sink: ITelemetrySink) : ITelemetry =
        if isNull (box sink) then
            nullArg "sink"

        SinkTelemetry(sink, clock, SpanContext(), []) :> ITelemetry

    /// A facade that writes to `sink`, timed by TelemetryClock.system.
    let ofSink (sink: ITelemetrySink) : ITelemetry =
        ofSinkWithClock TelemetryClock.system sink

    /// True when the ambient facade records anything.
    let enabled () = (current ()).Enabled

    /// Starts a span on the ambient facade: use _ = Telemetry.span "model.layer" [ Attr.int "layer" i ].
    let span (name: string) (attrs: Attr list) = (current ()).Span(name, attrs)

    let counter (name: string) (delta: float) (unit: string) (attrs: Attr list) =
        (current ()).Counter(name, delta, unit, attrs)

    let gauge (name: string) (value: float) (unit: string) (attrs: Attr list) =
        (current ()).Gauge(name, value, unit, attrs)

    let histogram (name: string) (value: float) (unit: string) (attrs: Attr list) =
        (current ()).Histogram(name, value, unit, attrs)

    let event (name: string) (attrs: Attr list) = (current ()).Event(name, attrs)

    let gpuInterval (name: string) (gpuStartNs: int64) (gpuDurationNs: int64) (attrs: Attr list) =
        (current ()).GpuInterval(name, gpuStartNs, gpuDurationNs, attrs)

    let meter (name: string) (amount: float) (unit: string) (attrs: Attr list) =
        (current ()).Meter(name, amount, unit, attrs)

    let check (name: string) (passed: bool) (attrs: Attr list) = (current ()).Check(name, passed, attrs)
