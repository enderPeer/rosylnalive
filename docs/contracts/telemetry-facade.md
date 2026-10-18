# TelemetryFacade contract

Owner: S1 (the facade, in Lrm.Core). T1 owns the runtime: the run-log sink, metering, reports and the RunLog contract. Status: v1, frozen at the contracts-v1 tag. Consumers: every component. Code: `src/Lrm.Core/Telemetry.fs`. Tests: `tests/Lrm.Core.Tests/TelemetryTests.fs`.

Observability is built into every component, but no component depends on a telemetry backend. Components report through the ambient facade, and the host (an app, a test or a node) decides where the records go.

## Rules

1. **Telemetry never changes values.** No telemetry call returns data into a computation. The only value a component may read is `Enabled`, and only to skip building expensive attributes, never to change what it computes. A test runs one instrumented computation three times, with the no-op facade, with an in-memory sink, and with a sink on a virtual clock, and requires identical bits.
2. **Timestamps never enter hashes.** The canonical encoder has no time, date or duration encoder, and ExecutionClass has no time field; a reflection test checks both. A content id therefore cannot depend on when something ran.
3. **The facade is ambient and a no-op by default.**
   - `Telemetry.current ()` is async-local. It flows into `Task.Run`, `Parallel.For` and everything else that captures the execution context.
   - Until a host calls `Telemetry.install`, it is `Telemetry.noop`, which records nothing.
4. **Thread safety.** Implementations and sinks are thread-safe.

## API

```fsharp
type ITelemetry =
    abstract Enabled: bool
    abstract Span: name: string * attrs: Attr list -> ISpan
    abstract Counter: name: string * delta: float * unit: string * attrs: Attr list -> unit
    abstract Gauge: name: string * value: float * unit: string * attrs: Attr list -> unit
    abstract Histogram: name: string * value: float * unit: string * attrs: Attr list -> unit
    abstract Event: name: string * attrs: Attr list -> unit
    abstract GpuInterval: name: string * gpuStartNs: int64 * gpuDurationNs: int64 * attrs: Attr list -> unit
    abstract Meter: name: string * amount: float * unit: string * attrs: Attr list -> unit
    abstract Check: name: string * passed: bool * attrs: Attr list -> unit
    abstract Child: attrs: Attr list -> ITelemetry
```

The `Telemetry` module forwards each verb to the ambient facade:
- `Telemetry.span`, `counter`, `gauge`, `histogram`, `event`, `gpuInterval`, `meter` and `check`;
- `Telemetry.scope attrs`, which makes `current().Child attrs` ambient until it is disposed;
- `Telemetry.install`, which makes a facade ambient until it is disposed, then restores the previous one;
- `Telemetry.ofSink` and `Telemetry.ofSinkWithClock`, which build a facade over a sink.

```fsharp
use _ = Telemetry.scope [ Attr.str TelemetryKeys.Component "forward"; Attr.int TelemetryKeys.Stage 0 ]
for layer in 0 .. 27 do
    use span = Telemetry.span "model.layer" [ Attr.int TelemetryKeys.Layer layer ]
    // ... compute ...
    span.Add(Attr.int "rows" rows)
    Telemetry.meter "model.flops" flops TelemetryUnits.Flops []
```

| Verb | Kind (`k`) | Value | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `Span` | `span` | duration in ns | `ns` | A timed region of host work. The record is emitted when the span is disposed |
| `Counter` | `counter` | the increment | given | A count of occurrences |
| `Gauge` | `gauge` | the level | given | A level at one moment, such as bytes in use or a queue depth |
| `Histogram` | `histogram` | the sample | given | One sample of a distribution, such as a latency |
| `Event` | `event` | 0 | `""` | Something that happened, described by its attributes |
| `GpuInterval` | `gpu` | GPU duration in ns | `ns` | A region timed by GPU timestamp queries |
| `Meter` | `meter` | the amount | given | Resource use attributed to the current scope: wall or GPU time, bytes moved, FLOPs, tokens |
| `Check` | `check` | 1 if passed, 0 if failed | `""` | The outcome of a verification, such as a golden comparison |
| `Child` | (none) | | | A facade whose records also carry these attributes, after its parent's scope |

- **Attributes.** `Attr = { Key: string; Value: AttrValue }`. `AttrValue` is `Str`, `Int` (int64), `Float` or `Bool`, so every sink encodes attributes the same way. The constructors are `Attr.str`, `Attr.int`, `Attr.int64`, `Attr.float` and `Attr.bool`.
- **Spans.**
  - `ISpan` is `IDisposable`. Disposing it ends the span, and a second `Dispose` does nothing.
  - `ISpan.Add` attaches attributes that are reported when the span ends.
  - `ISpan.Id` is 0 for a no-op span.
- **Cost** (bench tier, informational, on the 32-thread development machine): a span on the no-op facade costs 16 ns, and a span on a facade with a null sink 118 ns (129 ns with one attribute).

## Records and sinks

A sink implements `ITelemetrySink.Write: TelemetryRecord -> unit`. It is called from any thread.

| Field | Type | Meaning |
| --- | --- | --- |
| `Kind` | `TelemetryKind` | One of the eight kinds. `TelemetryKind.name` gives the short name in the table above |
| `Name` | string | The record's name (see Names) |
| `TimeNs` | int64 | Host wall-clock time in Unix ns (UTC): the start of a span, otherwise when the record was made |
| `DurationNs` | int64 | Span: the host duration. GpuInterval: the GPU duration. Otherwise 0 |
| `Value` | float | The measurement, as listed in the API table |
| `Unit` | string | The unit of Value |
| `GpuStartNs` | int64 | GpuInterval: its start in the GPU's own timestamp domain, in ns, comparable only within one queue. Otherwise 0 |
| `Attrs` | `Attr list` | Scope attributes (outermost first), then the call's own attributes, then those added to a span |
| `SpanId` | int64 | Span: its id, unique within one root facade and counting from 1. Otherwise 0 |
| `ParentId` | int64 | The enclosing span's id, or 0 at top level. Point records (counters, events and so on) carry it too |

- **Production sinks** (T1's run-log writer) never throw. On failure they drop the record and count the drop.
- **InMemorySink** is the strict test sink. It keeps every record, and it raises `ArgumentException` for a record that breaks the naming rules (`TelemetryNames.check`), so tests catch bad names at once.
- **RunLog mapping.** RunLog (T1) writes the keys `v, t, k, n, d, x, u, node, seq, a`. The natural mapping is:
  - `t` = TimeNs, `k` = the kind name, `n` = Name, `d` = DurationNs, `x` = Value, `u` = Unit;
  - `a` = Attrs, plus the span id, the parent id and the GPU start;
  - `node` and `seq` come from the sink.

  T1's contract is authoritative.
- **Clocks.** `Telemetry.ofSink` uses `TelemetryClock.system`: the UTC time at first use plus a monotonic Stopwatch offset, so it never runs backwards within a process. `ofSinkWithClock` takes any `unit -> int64` clock in Unix ns, for example C1's virtual clock.

## Names, keys and units

- **Names.** A record name or attribute key is 1 to 128 characters: dot-separated, non-empty segments of `a-z`, `0-9` and `_`, starting with a letter. Examples: `model.layer`, `gpu.dispatch`, `layer`, `bytes_moved`.
- **Units.** A unit is at most 32 printable ASCII characters with no spaces. The conventional units (`TelemetryUnits`) are `ns`, `B`, `flop`, `token`, `count`, and `""` for none.
- **Attribution keys.** The conventional keys (`TelemetryKeys`) are `component`, `layer`, `stage`, `node`, `backend`, `kernel`, `arena` and `case`. Reports attribute meters by component, layer, stage and node, so components set these through `Child` or `Telemetry.scope`.

### Reserved names

Reserved names have fixed meanings that reports and the dashboard depend on. Only the components listed emit them; other components use their own prefixes. `TelemetryNames.isReserved` recognizes them.

| Name | Emitted by | Typical use |
| --- | --- | --- |
| `gpu.*` | GPU backends and the GPU executor (D1, V1, X1, kernels) | `gpu.dispatch` GpuIntervals, `gpu.bytes_moved` meters, device faults |
| `model.*` | The Qwen model and forward pass (M2) | `model.layer` spans, `model.flops` and `model.tokens` meters |
| `cluster.*` | The cluster core and serving (C1, C2) | `cluster.divergence` events, reissues, leases, latencies |
| `run.*` | The run log itself (T1's RunContext; RunLog contract) | the `run.node` header, the `run.end` trailer and `run.clock_offset` records. No component emits them through the facade |
| `conformance.case` | The conformance harness (E1) | One Check per kernel case, with its bound and measured error as attributes |
| `tokenizer.case` | The tokenizer (M1) | One Check per corpus case |
| `golden.compare` | The comparators (O2) and their callers | One Check per golden comparison, with the metric and tolerance as attributes |
| `mem.arena` | The heap and executor (R4a, X1) | A Gauge in bytes per heap arena, with `arena` as an attribute |

## Verification

The tests in `TelemetryTests.fs` (cpu tier) check that:
- the default facade is a no-op;
- spans nest, carry their timing and order attributes as specified;
- every verb produces its kind, value and unit;
- a span ends only once;
- `install` and `scope` restore the previous facade;
- the ambient facade flows into parallel work;
- telemetry never changes values;
- no time type can be canonically encoded;
- names are validated and the reserved set is recognized;
- the strict sink rejects malformed records.

The span cost is measured in the bench tier only.
