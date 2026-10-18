module Lrm.Telemetry.Tests.MeteringTests

open System
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Telemetry.Tests.Common

let private ev (node: string) (kind: TelemetryKind) (name: string) (attrs: Attr list) =
    {
        Node = node
        Seq = 0L
        Kind = kind
        Name = name
        TimeNs = 0L
        DurationNs = 0L
        Value = 0.0
        Unit = ""
        SpanId = 0L
        ParentId = 0L
        GpuStartNs = 0L
        Attrs = attrs
    }

let private span node id parent dur attrs =
    { ev node TelemetryKind.Span "s" attrs with
        SpanId = id
        ParentId = parent
        DurationNs = dur
        Value = float dur
        Unit = "ns"
    }

let private meter node unit amount attrs =
    { ev node TelemetryKind.Meter "m" attrs with
        Value = amount
        Unit = unit
    }

let private comp = Attr.str "component"
let private layer = Attr.int "layer"

let tests =
    testList
        "metering"
        [
            testCase "FLOP registry known answers"
            <| fun () ->
                let cases =
                    [
                        "matmul 1x1024x1024 f32 x bf16",
                        Op.MatMul(1L, 1024L, 1024L, DType.F32, DType.BF16, DType.F32),
                        (2_097_152L, 4_096L + 2_097_152L, 4_096L)
                        "rmsnorm 1x1024", Op.RmsNorm(1L, 1024L, DType.BF16), (4_098L, 4_096L + 2_048L, 4_096L)
                        "rmsnorm 16x128 (q_norm)", Op.RmsNorm(16L, 128L, DType.BF16), (8_224L, 8_192L + 256L, 8_192L)
                        "rope 1x16x128", Op.Rope(1L, 16L, 128L), (6_144L, 8_192L + 512L, 8_192L)
                        "attention rows 1 past 0",
                        Op.Attention(1L, 0L, 16L, 8L, 128L),
                        (8_240L, 8_192L + 8_192L, 8_192L)
                        "attention rows 4 past 10",
                        Op.Attention(4L, 10L, 16L, 8L, 128L),
                        (16L * 50L * 515L, 32_768L + 114_688L, 32_768L)
                        "silu_mul 3072", Op.SiluMul 3072L, (15_360L, 24_576L, 12_288L)
                        "add 1024", Op.Add 1024L, (1_024L, 8_192L, 4_096L)
                        "embed 2x1024 bf16", Op.Embed(2L, 1024L, DType.BF16), (0L, 4_096L + 8L, 8_192L)
                        "copy_rows 3x10 f32", Op.CopyRows(3L, 10L, DType.F32), (0L, 120L, 120L)
                        "argmax 1x151936", Op.Argmax(1L, 151_936L), (151_936L, 607_744L, 4L)
                    ]

                for name, op, (flops, read, written) in cases do
                    let c = OpCost.ofOp op
                    Expect.equal (c.Flops, c.BytesRead, c.BytesWritten) (flops, read, written) name

            testCase "causal pairs: rows*past + rows*(rows+1)/2"
            <| fun () ->
                Expect.equal (OpCost.causalPairs 1L 0L) 1L "first token"
                Expect.equal (OpCost.causalPairs 1L 511L) 512L "decode after 511"
                Expect.equal (OpCost.causalPairs 4L 10L) 50L "chunk"
                Expect.equal (OpCost.causalPairs 512L 0L) 131_328L "prefill"

            testCase "Qwen3-0.6B: one layer and one decode step at position 0"
            <| fun () ->
                let d = TransformerDims.qwen3_0_6b
                let layer = OpCost.layerOps d 1L 0L |> List.map snd |> OpCost.sum
                Expect.equal layer.Flops 31_512_676L "layer FLOPs"
                let step = OpCost.forward d 1L 0L 1L
                Expect.equal step.Flops 1_193_675_890L "decode step FLOPs"
                Expect.equal (OpCost.forwardOps d 1L 0L 1L).Length (1 + 28 * 17 + 3) "op count"

                Expect.equal
                    (OpCost.forwardOps d 1L 0L 1L |> List.map fst |> List.take 3)
                    [ "embed"; "L0.input_layernorm"; "L0.q_proj" ]
                    "tap names"

            testCase "attribution: span self time, GPU time and meters by unit, per key"
            <| fun () ->
                let events =
                    [
                        span "n0" 1L 0L 1000L [ comp "forward"; Attr.int "stage" 0 ]
                        span "n0" 2L 1L 300L [ comp "forward"; Attr.int "stage" 0; layer 0 ]
                        span "n0" 3L 1L 500L [ comp "forward"; Attr.int "stage" 0; layer 1 ]
                        span "n0" 4L 3L 700L [ comp "forward"; Attr.int "stage" 0; layer 1 ] // overlapping child: self 0
                        { ev
                              "n0"
                              TelemetryKind.GpuInterval
                              "gpu.dispatch"
                              [ comp "forward"; Attr.int "stage" 0; layer 1 ] with
                            DurationNs = 90L
                        }
                        meter "n0" "flop" 2e6 [ comp "forward"; Attr.int "stage" 0; layer 0 ]
                        meter "n0" "B" 4096.0 [ comp "forward"; Attr.int "stage" 0; layer 0 ]
                        meter "n0" "token" 1.0 [ comp "forward"; Attr.int "stage" 0 ]
                        meter "n0" "ns" 55.0 [ comp "forward"; Attr.int "stage" 0 ]
                        meter "n1" "flop" 3.0 [ comp "head"; Attr.str "stage" "s1" ]
                        { ev "n0" TelemetryKind.Gauge "mem.arena" [ Attr.str "arena" "weights" ] with
                            Value = 1e9
                        }
                    ]

                let rows = Metering.attribute events

                let find node c stage l =
                    rows
                    |> List.find (fun (k, _) ->
                        k =
                            {
                                Node = node
                                Component = c
                                Stage = stage
                                Layer = l
                            })
                    |> snd

                let top = find "n0" "forward" "0" None
                Expect.equal (top.WallNs, top.Tokens, top.MeterNs, top.Records) (200L, 1.0, 55.0, 3) "stage level"
                let l0 = find "n0" "forward" "0" (Some 0L)
                Expect.equal (l0.WallNs, l0.Flops, l0.Bytes) (300L, 2e6, 4096.0) "layer 0"
                let l1 = find "n0" "forward" "0" (Some 1L)
                Expect.equal (l1.WallNs, l1.GpuNs) (700L, 90L) "layer 1: self 0 + 700 + GPU"
                Expect.equal (find "n1" "head" "s1" None).Flops 3.0 "named stage on another node"

                Expect.equal
                    (rows |> List.map fst |> List.map _.Node)
                    [ "n0"; "n0"; "n0"; "n1" ]
                    "sorted, gauges not counted"

                let perComponent = Metering.rollup (fun k -> { k with Layer = None }) rows
                let fwd = perComponent |> List.find (fun (k, _) -> k.Node = "n0") |> snd
                Expect.equal fwd.WallNs 1200L "self times sum to the root span plus the overlap"

            testCase "arena peaks and last values per node and arena"
            <| fun () ->
                let g node arena v =
                    { ev node TelemetryKind.Gauge "mem.arena" [ Attr.str "arena" arena ] with
                        Value = v
                        Unit = "B"
                    }

                let arenas =
                    Metering.arenas
                        [
                            g "n0" "kv_cache" 10.0
                            g "n0" "kv_cache" 30.0
                            g "n0" "kv_cache" 20.0
                            g "n0" "weights" 5.0
                        ]

                Expect.equal
                    (arenas |> List.map (fun a -> a.Arena, a.PeakBytes, a.LastBytes, a.Samples))
                    [ "kv_cache", 30.0, 20.0, 3; "weights", 5.0, 5.0, 1 ]
                    "arenas"

            testCase "Metering.op emits <prefix>.flops and <prefix>.bytes_moved with the op attribute"
            <| fun () ->
                let sink = InMemorySink()

                do
                    use _ = Telemetry.install (Telemetry.ofSinkWithClock (steppingClock 0L 1L) sink)
                    Metering.op "model" (Op.Add 1024L) [ Attr.int "layer" 3 ]
                    Metering.arena "weights" 123L []
                    Metering.tokens "model.tokens" 1.0 []

                let rs = sink.Records

                Expect.equal
                    (rs |> List.map (fun r -> r.Name, r.Value, r.Unit))
                    [
                        "model.flops", 1024.0, "flop"
                        "model.bytes_moved", 12288.0, "B"
                        "mem.arena", 123.0, "B"
                        "model.tokens", 1.0, "token"
                    ]
                    "records"

                Expect.equal rs.Head.Attrs [ Attr.str "op" "add"; Attr.int "layer" 3 ] "attributes"

                // Once the facade is uninstalled the ambient one is the no-op: nothing more is recorded.
                Metering.op "model" (Op.Add 1024L) []
                Expect.equal sink.Count 4 "nothing after uninstall"
        ]
