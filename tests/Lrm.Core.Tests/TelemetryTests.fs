module Lrm.Core.Tests.TelemetryTests

open System
open System.Reflection
open System.Threading.Tasks
open Expecto
open Lrm.Core

/// A clock that advances by `step` ns on every read, from `start`.
let private steppingClock (start: int64) (step: int64) =
    let t = ref (start - step)

    fun () ->
        t.Value <- t.Value + step
        t.Value

let private withSink (clock: unit -> int64) (f: unit -> 'a) =
    let sink = InMemorySink()

    let result =
        use _ = Telemetry.install (Telemetry.ofSinkWithClock clock sink)
        f ()

    result, sink.Records

/// A small computation that reports through the ambient facade at every step. Its result must not depend on
/// telemetry in any way.
let private instrumented () =
    use _ = Telemetry.span "model.forward" [ Attr.str "backend" "cpu-f32" ]
    let rng = DetRng(7UL, DetRng.streamOf "telemetry-test")
    let values = Array.init 256 (fun _ -> rng.NextF32() * 2.0f - 1.0f)
    let mutable acc = 0.0f

    for layer in 0..3 do
        use _ = Telemetry.scope [ Attr.int TelemetryKeys.Layer layer ]
        use span = Telemetry.span "model.layer" []

        for i in 0 .. values.Length - 1 do
            values[i] <- Bf16.widen (Bf16.narrowRne (values[i] * 1.5f + float32 layer))
            acc <- acc + values[i]

        span.Add(Attr.float "acc" (float acc))
        Telemetry.meter "model.flops" (float values.Length * 2.0) TelemetryUnits.Flops []
        Telemetry.counter "model.steps" 1.0 TelemetryUnits.Count []

    let id =
        Canonical.hash "lrm.test/1" (fun w ->
            w.Seq(values, w.F32)
            w.F32 acc)

    Telemetry.check TelemetryNames.GoldenCompare true [ Attr.str "id" id.Short ]
    id, acc

let tests =
    testList
        "telemetry"
        [
            testCase "the ambient facade is a no-op by default"
            <| fun () ->
                Expect.isFalse (Telemetry.enabled ()) "disabled"
                use span = Telemetry.span "model.layer" [ Attr.int "layer" 1 ]
                Expect.equal span.Id 0L "no-op span"
                Telemetry.event "model.event" []
                Expect.isFalse (Telemetry.current().Child([ Attr.int "x" 1 ]).Enabled) "children stay no-op"

            testCase "spans nest and carry scope attributes"
            <| fun () ->
                let (), records =
                    withSink (steppingClock 1_000L 10L) (fun () ->
                        use _ = Telemetry.scope [ Attr.str TelemetryKeys.Component "forward" ]
                        use outer = Telemetry.span "model.forward" [ Attr.int "tokens" 5 ]

                        do
                            use inner = Telemetry.span "model.layer" [ Attr.int TelemetryKeys.Layer 0 ]
                            inner.Add(Attr.bool "cached" true)
                            Telemetry.event "model.tap" [ Attr.str "tap" "embed.out" ]

                        Expect.isGreaterThan outer.Id 0L "outer id")

                let byName n =
                    records |> List.find (fun r -> r.Name = n)

                let outer, inner, tap =
                    byName "model.forward", byName "model.layer", byName "model.tap"

                Expect.equal (records |> List.map _.Name) [ "model.tap"; "model.layer"; "model.forward" ] "order"
                Expect.equal outer.ParentId 0L "outer is top level"
                Expect.equal inner.ParentId outer.SpanId "inner parent"
                Expect.equal tap.ParentId inner.SpanId "event parent"
                Expect.equal tap.SpanId 0L "events have no span id"
                Expect.equal (outer.TimeNs, outer.DurationNs) (1_000L, 40L) "outer timing"
                Expect.equal (inner.TimeNs, inner.DurationNs) (1_010L, 20L) "inner timing"
                Expect.equal inner.Value 20.0 "span value is its duration"
                Expect.equal inner.Unit "ns" "span unit"

                Expect.equal
                    inner.Attrs
                    [ Attr.str "component" "forward"; Attr.int "layer" 0; Attr.bool "cached" true ]
                    "scope, call, then added attributes"

                Expect.equal outer.Attrs [ Attr.str "component" "forward"; Attr.int "tokens" 5 ] "outer attributes"

            testCase "every kind is recorded with its value and unit"
            <| fun () ->
                let (), records =
                    withSink (steppingClock 0L 1L) (fun () ->
                        let t = (Telemetry.current ()).Child [ Attr.int TelemetryKeys.Stage 1 ]
                        t.Counter("cluster.reissues", 2.0, TelemetryUnits.Count, [])
                        t.Gauge(TelemetryNames.MemArena, 4096.0, TelemetryUnits.Bytes, [ Attr.str "arena" "weights" ])
                        t.Histogram("cluster.latency", 1.5e6, TelemetryUnits.Nanoseconds, [])
                        t.Event("cluster.ready", [])
                        t.GpuInterval("gpu.dispatch", 123_456_789L, 2_500L, [ Attr.str "kernel" "matmul_bf16w" ])
                        t.Meter("gpu.bytes_moved", 1.0e9, TelemetryUnits.Bytes, [])
                        t.Check(TelemetryNames.ConformanceCase, false, [ Attr.str "case" "add/0" ]))

                let summary =
                    records
                    |> List.map (fun r ->
                        TelemetryKind.name r.Kind, r.Name, r.Value, r.Unit, r.GpuStartNs, r.DurationNs)

                Expect.equal
                    summary
                    [
                        "counter", "cluster.reissues", 2.0, "count", 0L, 0L
                        "gauge", "mem.arena", 4096.0, "B", 0L, 0L
                        "histogram", "cluster.latency", 1.5e6, "ns", 0L, 0L
                        "event", "cluster.ready", 0.0, "", 0L, 0L
                        "gpu", "gpu.dispatch", 2500.0, "ns", 123_456_789L, 2_500L
                        "meter", "gpu.bytes_moved", 1.0e9, "B", 0L, 0L
                        "check", "conformance.case", 0.0, "", 0L, 0L
                    ]
                    "records"

                for r in records do
                    Expect.equal (List.head r.Attrs) (Attr.int "stage" 1) (sprintf "%s carries the scope" r.Name)

            testCase "a span ends once"
            <| fun () ->
                let (), records =
                    withSink (steppingClock 0L 1L) (fun () ->
                        let s = Telemetry.span "model.layer" []
                        s.Dispose()
                        s.Dispose())

                Expect.equal records.Length 1 "one record"

            testCase "install and scope restore the previous facade"
            <| fun () ->
                let a = Telemetry.ofSink (InMemorySink())
                let b = Telemetry.ofSink (InMemorySink())

                do
                    use _ = Telemetry.install a
                    Expect.isTrue (obj.ReferenceEquals(Telemetry.current (), a)) "a"

                    do
                        use _ = Telemetry.install b
                        Expect.isTrue (obj.ReferenceEquals(Telemetry.current (), b)) "b"

                    Expect.isTrue (obj.ReferenceEquals(Telemetry.current (), a)) "a again"

                    do
                        use _ = Telemetry.scope [ Attr.int "x" 1 ]
                        Expect.isFalse (obj.ReferenceEquals(Telemetry.current (), a)) "scoped child"

                    Expect.isTrue (obj.ReferenceEquals(Telemetry.current (), a)) "a after scope"

                Expect.isFalse (Telemetry.enabled ()) "no-op again"

            testCase "the ambient facade flows into parallel work"
            <| fun () ->
                let (), records =
                    withSink TelemetryClock.system (fun () ->
                        Parallel.For(0, 1000, (fun i -> Telemetry.counter "model.rows" 1.0 TelemetryUnits.Count []))
                        |> ignore

                        Task.Run(fun () -> Telemetry.event "model.task" []).Wait())

                Expect.equal (records |> List.filter (fun r -> r.Name = "model.rows")).Length 1000 "1000 counters"
                Expect.isTrue (records |> List.exists (fun r -> r.Name = "model.task")) "task event"

            testCase "telemetry never changes values"
            <| fun () ->
                let plain = instrumented ()
                let recorded, records = withSink TelemetryClock.system instrumented
                let virtualTime, _ = withSink (steppingClock 5_000_000_000L 7L) instrumented
                Expect.isGreaterThan records.Length 10 "records were made"

                let bits (id: ContentId, acc: float32) =
                    id.Hex, BitConverter.SingleToUInt32Bits acc

                Expect.equal (bits recorded) (bits plain) "with a sink"
                Expect.equal (bits virtualTime) (bits plain) "with another clock"

            testCase "timestamps cannot enter canonical hashes"
            <| fun () ->
                let timeTypes =
                    Collections.Generic.HashSet<Type>(
                        [
                            typeof<DateTime>
                            typeof<DateTimeOffset>
                            typeof<TimeSpan>
                            typeof<TelemetryRecord>
                        ]
                    )

                let offenders =
                    typeof<CanonicalWriter>.GetMethods(BindingFlags.Public ||| BindingFlags.Instance)
                    |> Array.filter (fun m ->
                        m.GetParameters() |> Array.exists (fun p -> timeTypes.Contains p.ParameterType))
                    |> Array.map _.Name

                Expect.isEmpty offenders "CanonicalWriter has no time encoder"

                let classFields =
                    typeof<ExecutionClass>.GetProperties()
                    |> Array.map _.PropertyType
                    |> Array.filter timeTypes.Contains

                Expect.isEmpty classFields "ExecutionClass has no time field"

            testCase "names and the reserved set"
            <| fun () ->
                for good in [ "model.layer"; "gpu.dispatch"; "layer"; "bytes_moved"; "l2.cache_hits" ] do
                    Expect.isTrue (TelemetryNames.isValid good) good

                for bad in
                    [
                        ""
                        "Model.layer"
                        "model..layer"
                        "model."
                        ".model"
                        "1abc"
                        "model layer"
                        null
                    ] do
                    Expect.isFalse (TelemetryNames.isValid bad) (sprintf "%A" bad)

                for r in
                    [
                        "gpu.dispatch"
                        "model.layer"
                        "cluster.divergence"
                        "run.node"
                        "run.clock_offset"
                        "conformance.case"
                        "tokenizer.case"
                        "golden.compare"
                        "mem.arena"
                    ] do
                    Expect.isTrue (TelemetryNames.isReserved r) r

                for r in [ "gpux.dispatch"; "mem.arena2"; "report.chart"; "runs.count" ] do
                    Expect.isFalse (TelemetryNames.isReserved r) r

            testCase "the in-memory sink rejects malformed records"
            <| fun () ->
                let t = Telemetry.ofSink (InMemorySink())

                Expect.throwsT<ArgumentException> (fun () -> t.Event("Bad Name", [])) "name"
                Expect.throwsT<ArgumentException> (fun () -> t.Event("model.ok", [ Attr.str "Bad" "x" ])) "key"
                Expect.throwsT<ArgumentException> (fun () -> t.Event("model.ok", [ Attr.str "k" null ])) "null value"

                Expect.throwsT<ArgumentException> (fun () -> t.Gauge("model.ok", 1.0, "has space", [])) "unit"
        ]
