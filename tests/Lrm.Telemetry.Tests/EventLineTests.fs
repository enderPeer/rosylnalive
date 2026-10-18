module Lrm.Telemetry.Tests.EventLineTests

open System
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Testing
open Lrm.Telemetry.Tests.Common

let private t0 = 1_760_000_000_000_000_000L

/// The known-answer lines of the RunLog contract (vectors E1 to E5).
let vectors: (string * string * int64 * TelemetryRecord * string) list =
    [
        "E1 span",
        "n0",
        7L,
        { record TelemetryKind.Span "model.layer" with
            DurationNs = 1500L
            Value = 1500.0
            Unit = "ns"
            SpanId = 3L
            ParentId = 1L
            Attrs = [ Attr.str "component" "forward"; Attr.int "layer" 3 ]
        },
        """{"v":1,"t":1760000000000000000,"k":"span","n":"model.layer","d":1500,"x":1500,"u":"ns","node":"n0","seq":7,"a":{"$span":3,"$parent":1,"component":"forward","layer":3}}"""

        "E2 gpu interval",
        "n1",
        0L,
        { record TelemetryKind.GpuInterval "gpu.dispatch" with
            DurationNs = 9400L
            Value = 9400.0
            Unit = "ns"
            GpuStartNs = 0L
            Attrs = [ Attr.str "kernel" "matmul_bf16w" ]
        },
        """{"v":1,"t":1760000000000000000,"k":"gpu","n":"gpu.dispatch","d":9400,"x":9400,"u":"ns","node":"n1","seq":0,"a":{"$gpu_start":0,"kernel":"matmul_bf16w"}}"""

        "E3 check with typed floats",
        "n0",
        1L,
        { record TelemetryKind.Check "golden.compare" with
            Value = 1.0
            Attrs =
                [
                    Attr.float "measured" 3.2e-7
                    Attr.float "bound" 1e-5
                    Attr.float "exact" 1.0
                    Attr.float "neg" -0.0
                    Attr.float "bad" nan
                    Attr.bool "ok" true
                    Attr.str "tap" "L3.q_proj"
                ]
        },
        """{"v":1,"t":1760000000000000000,"k":"check","n":"golden.compare","d":0,"x":1,"u":"","node":"n0","seq":1,"a":{"measured":3.2E-07,"bound":1E-05,"exact":1.0,"neg":-0.0,"bad":{"$f":"NaN"},"ok":true,"tap":"L3.q_proj"}}"""

        "E4 meter: non-finite value, escapes, repeated key",
        "n0",
        2L,
        { record TelemetryKind.Meter "model.flops" with
            Value = infinity
            Unit = "flop"
            Attrs =
                [
                    Attr.int "layer" 1
                    Attr.str "note" "tab\there \"q\" \\ é 😀 \u2028"
                    Attr.int "layer" 2
                ]
        },
        """{"v":1,"t":1760000000000000000,"k":"meter","n":"model.flops","d":0,"x":"Infinity","u":"flop","node":"n0","seq":2,"a":{"layer":2,"note":"tab\there \"q\" \\ é 😀 \u2028"}}"""

        "E5 event: control character and a lone surrogate",
        "coord",
        12L,
        { record TelemetryKind.Event "cluster.node_lost" with
            TimeNs = -5L
            Attrs = [ Attr.str "target" "s0a"; Attr.str "reason" "a\u0001b\ud800" ]
        },
        """{"v":1,"t":-5,"k":"event","n":"cluster.node_lost","d":0,"x":0,"u":"","node":"coord","seq":12,"a":{"target":"s0a","reason":"a\u0001b�"}}"""
    ]

let private bits (x: float) = BitConverter.DoubleToInt64Bits x

let private sameFloat (a: float) (b: float) =
    (Double.IsNaN a && Double.IsNaN b) || bits a = bits b

let private sameAttr (a: AttrValue) (b: AttrValue) =
    match a, b with
    | AttrValue.Float x, AttrValue.Float y -> sameFloat x y
    | x, y -> x = y

/// Random records with every kind, attribute type and awkward value, from a fixed DetRng stream.
let private randomRecords (count: int) =
    let r = DetRng(11UL, DetRng.streamOf "eventline-tests")

    let floats =
        [|
            0.0
            -0.0
            1.0
            0.1
            1e-300
            5e-324
            1.7976931348623157e308
            nan
            infinity
            -infinity
            123456789.125
        |]

    let pickFloat () =
        if r.NextBelow 3u = 0u then
            floats[int (r.NextBelow(uint32 floats.Length))]
        else
            BitConverter.UInt64BitsToDouble(r.NextU64())

    let strings =
        [|
            ""
            "x"
            "é"
            "😀"
            "a\"b"
            "\\"
            "\n\r\t"
            "\u0000"
            "\u2029"
            "plain text"
        |]

    let pickValue () =
        match r.NextBelow 4u with
        | 0u -> AttrValue.Str strings[int (r.NextBelow(uint32 strings.Length))]
        | 1u -> AttrValue.Int(int64 (r.NextU64()))
        | 2u -> AttrValue.Float(pickFloat ())
        | _ -> AttrValue.Bool(r.NextBelow 2u = 0u)

    [
        for i in 1..count do
            let kind = TelemetryKind.all[int (r.NextBelow 8u)]
            let keys = [| "a"; "b"; "layer"; "c.d"; "x_1" |]

            let attrs =
                [
                    for _ in 1 .. int (r.NextBelow 6u) ->
                        {
                            Key = keys[int (r.NextBelow(uint32 keys.Length))]
                            Value = pickValue ()
                        }
                ]

            yield
                {
                    Kind = kind
                    Name = "test.n" + Invariant.i32 (i % 7)
                    TimeNs = int64 (r.NextU64())
                    DurationNs = int64 (r.NextU64())
                    Value = pickFloat ()
                    Unit = [| ""; "ns"; "B"; "flop" |][int (r.NextBelow 4u)]
                    GpuStartNs =
                        (if kind = TelemetryKind.GpuInterval then
                             int64 (r.NextU64())
                         else
                             0L)
                    Attrs = attrs
                    SpanId =
                        (if kind = TelemetryKind.Span then
                             1L + int64 (r.NextBelow 1000u)
                         else
                             0L)
                    ParentId = int64 (r.NextBelow 3u)
                }
    ]

let tests =
    testList
        "eventline"
        [
            for name, node, seq, r, expected in vectors do
                testCase ("known answer " + name)
                <| fun () ->
                    Expect.equal (EventLine.encode node seq r) expected "line"
                    Expect.equal (withCulture "de-DE" (fun () -> EventLine.encode node seq r)) expected "de-DE"

            testCase "known answers decode to their records"
            <| fun () ->
                for name, node, seq, r, line in vectors do
                    match EventLine.decode line with
                    | Error e -> failtestf "%s: %s" name e
                    | Ok ev ->
                        Expect.equal ev.Node node name
                        Expect.equal ev.Seq seq name
                        Expect.equal ev.Kind r.Kind name
                        Expect.equal ev.Name r.Name name
                        Expect.equal ev.TimeNs r.TimeNs name
                        Expect.isTrue (sameFloat ev.Value r.Value) name

            testCase "every record round-trips (1000 random records)"
            <| fun () ->
                randomRecords 1000
                |> List.iteri (fun i r ->
                    let line = EventLine.encode "n0" (int64 i) r

                    match EventLine.decode line with
                    | Error e -> failtestf "record %d: %s\n%s" i e line
                    | Ok ev ->
                        let expectedAttrs = EventLine.dedupe r.Attrs
                        Expect.equal ev.Kind r.Kind "kind"
                        Expect.equal ev.Name r.Name "name"
                        Expect.equal ev.TimeNs r.TimeNs "t"
                        Expect.equal ev.DurationNs r.DurationNs "d"
                        Expect.isTrue (sameFloat ev.Value r.Value) (sprintf "x %s" line)
                        Expect.equal ev.Unit r.Unit "u"
                        Expect.equal ev.Seq (int64 i) "seq"
                        Expect.equal ev.SpanId r.SpanId "span"
                        Expect.equal ev.ParentId r.ParentId "parent"
                        Expect.equal ev.GpuStartNs r.GpuStartNs "gpu start"
                        Expect.equal (ev.Attrs |> List.map _.Key) (expectedAttrs |> List.map _.Key) "attr keys"

                        for a, b in List.zip ev.Attrs expectedAttrs do
                            Expect.isTrue (sameAttr a.Value b.Value) (sprintf "attr %s in %s" a.Key line))

            Props.testProperty "every float value round-trips bit-exactly (NaN stays NaN)"
            <| fun (b: uint64) ->
                let x = BitConverter.UInt64BitsToDouble b

                let line =
                    EventLine.encode
                        "n0"
                        0L
                        { record TelemetryKind.Meter "m" with
                            Value = x
                            Attrs = [ Attr.float "f" x ]
                        }

                match EventLine.decode line with
                | Ok ev ->
                    sameFloat ev.Value x
                    && (match ev.TryAttr "f" with
                        | Some(AttrValue.Float y) -> sameFloat y x
                        | _ -> false)
                | Error _ -> false

            testCase "integral floats stay floats and integers stay integers"
            <| fun () ->
                let line =
                    EventLine.encode
                        "n0"
                        0L
                        { record TelemetryKind.Event "e" with
                            Attrs = [ Attr.float "f" 3.0; Attr.int "i" 3; Attr.float "big" 1e20 ]
                        }

                match EventLine.decode line with
                | Ok ev ->
                    Expect.equal (ev.TryAttr "f") (Some(AttrValue.Float 3.0)) "float"
                    Expect.equal (ev.TryAttr "i") (Some(AttrValue.Int 3L)) "int"
                    Expect.equal (ev.TryAttr "big") (Some(AttrValue.Float 1e20)) "1E+20 is a float"
                | Error e -> failtest e

            testCase "malformed lines are rejected with a reason"
            <| fun () ->
                let good =
                    """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""

                Expect.isOk (EventLine.decode good) "the base line is valid"

                let bad =
                    [
                        "not JSON", "{"
                        "array", "[]"
                        "missing key", """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","a":{}}"""
                        "duplicate key",
                        """{"v":1,"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""
                        "unknown key",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{},"z":1}"""
                        "version 2",
                        """{"v":2,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""
                        "unknown kind",
                        """{"v":1,"t":0,"k":"blip","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""
                        "fractional t",
                        """{"v":1,"t":0.5,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""
                        "span without $span",
                        """{"v":1,"t":0,"k":"span","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""
                        "$span on an event",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"$span":1}}"""
                        "gpu without $gpu_start",
                        """{"v":1,"t":0,"k":"gpu","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{}}"""
                        "unknown structural key",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"$z":1}}"""
                        "null attribute",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"q":null}}"""
                        "bad $f",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"q":{"$f":"Inf"}}}"""
                        "string x",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":"1","u":"","node":"n0","seq":0,"a":{}}"""
                        "duplicate attribute",
                        """{"v":1,"t":0,"k":"event","n":"e","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"q":1,"q":2}}"""
                    ]

                for name, line in bad do
                    match EventLine.decode line with
                    | Ok _ -> failtestf "%s was accepted" name
                    | Error e -> Expect.isNotEmpty e name

            testCase "dedupe keeps the first position and the last value"
            <| fun () ->
                let attrs = [ Attr.int "a" 1; Attr.int "b" 2; Attr.int "a" 3 ]
                Expect.equal (EventLine.dedupe attrs) [ Attr.int "a" 3; Attr.int "b" 2 ] "dedupe"

            testCase "JSON text: escapes, typed floats, ISO times"
            <| fun () ->
                Expect.equal (Json.quote "a\u001fb\u007f") "\"a\\u001fb\u007f\"" "control below 0x20 only"
                Expect.equal (Json.quote "\b\f") "\"\\b\\f\"" "short escapes"
                Expect.equal (Json.isoTime 0L) "1970-01-01T00:00:00.000Z" "epoch"
                Expect.equal (Json.isoTime -1L) "1969-12-31T23:59:59.999Z" "floor before the epoch"
                Expect.equal (Json.isoTime 1_792_310_400_123_456_789L) "2026-10-18T08:00:00.123Z" "milliseconds"

                Expect.equal
                    (Json.pretty (
                        JsonValue.Obj
                            [
                                "a", JsonValue.Arr [ JsonValue.Int 1L; JsonValue.Null ]
                                "b", JsonValue.Obj []
                            ]
                    ))
                    "{\n  \"a\": [\n    1,\n    null\n  ],\n  \"b\": {}\n}\n"
                    "pretty"
        ]
