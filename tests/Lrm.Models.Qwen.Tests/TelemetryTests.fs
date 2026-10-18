/// The forward's telemetry (ForwardOpsAndQwenModel, "Telemetry"): model.generate, model.forward and model.layer spans
/// with the attributes the inference template reads, model.tokens and the FLOP registry's model.flops and
/// model.bytes_moved meters, and a real run log that the inference and parity templates render with data in every
/// chart the forward feeds.
module Lrm.Models.Qwen.Tests.TelemetryTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

let private str (key: string) (r: TelemetryRecord) =
    r.Attrs
    |> List.tryFindBack (fun a -> a.Key = key)
    |> Option.bind (fun a ->
        match a.Value with
        | AttrValue.Str s -> Some s
        | _ -> None)

let private int' (key: string) (r: TelemetryRecord) =
    r.Attrs
    |> List.tryFindBack (fun a -> a.Key = key)
    |> Option.bind (fun a ->
        match a.Value with
        | AttrValue.Int i -> Some i
        | _ -> None)

let private named (name: string) (kind: TelemetryKind) (records: TelemetryRecord list) =
    records |> List.filter (fun r -> r.Name = name && r.Kind = kind)

let tests =
    testList
        "telemetry"
        [
            testCase
                "greedy decoding: one model.generate, model.forward per step, model.layer per layer, with their attributes"
            <| fun () ->
                let sink = InMemorySink()
                let ids = promptIds "p0"

                do
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    Greedy.run (tinyF32 ()) (tinyModel ()) ids (GreedyOptions.parity 4) |> ignore

                let records = sink.Records

                let generate =
                    named Greedy.GenerateSpan TelemetryKind.Span records |> List.exactlyOne

                Expect.equal (int' "prompt_tokens" generate, int' "steps" generate) (Some 13L, Some 4L) "generate"

                let forwards = named Forward.ForwardSpan TelemetryKind.Span records
                Expect.equal forwards.Length 4 "one model.forward per step"

                Expect.equal
                    (forwards
                     |> List.map (fun r -> str "phase" r, int' "step" r, int' "tokens" r, int' "pos0" r))
                    [
                        Some "prefill", Some 0L, Some 13L, Some 0L
                        Some "decode", Some 1L, Some 1L, Some 13L
                        Some "decode", Some 2L, Some 1L, Some 14L
                        Some "decode", Some 3L, Some 1L, Some 15L
                    ]
                    "phase, step, tokens, pos0"

                for f in forwards do
                    Expect.equal f.ParentId generate.SpanId "inside model.generate"
                    Expect.equal (int' TelemetryKeys.Stage f) (Some 0L) "stage"

                let layers = named Forward.LayerSpan TelemetryKind.Span records
                Expect.equal layers.Length 8 "two layers per step"

                for l in layers do
                    let parent = forwards |> List.find (fun f -> f.SpanId = l.ParentId)
                    Expect.equal (str "phase" l) (str "phase" parent) "phase of its step"
                    Expect.equal (str TelemetryKeys.Component l) (Some Forward.Component) "component"
                    Expect.equal (int' TelemetryKeys.Stage l) (Some 0L) "stage"
                    Expect.isSome (int' TelemetryKeys.Layer l) "layer"

                let tokens = named "model.tokens" TelemetryKind.Meter records |> List.sumBy _.Value
                Expect.equal tokens 16.0 "13 prompt tokens and 3 decode steps"

            testCase "model.flops and model.bytes_moved equal the FLOP registry's forward, per step and in total"
            <| fun () ->
                let sink = InMemorySink()
                let ids = promptIds "p0"
                let model = tinyModel ()

                do
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    Greedy.run (tinyF64 ()) model ids (GreedyOptions.parity 4) |> ignore

                let dims = Qwen3Config.dims model.Config

                let expected =
                    [
                        OpCost.forward dims 13L 0L 1L
                        OpCost.forward dims 1L 13L 1L
                        OpCost.forward dims 1L 14L 1L
                        OpCost.forward dims 1L 15L 1L
                    ]
                    |> List.fold (+) OpCost.Zero

                let sum (name: string) =
                    named name TelemetryKind.Meter sink.Records |> List.sumBy _.Value

                Expect.equal (sum "model.flops") (float expected.Flops) "flops"
                Expect.equal (sum "model.bytes_moved") (float expected.BytesMoved) "bytes"

                let layerFlops =
                    named "model.flops" TelemetryKind.Meter sink.Records
                    |> List.filter (fun r -> int' TelemetryKeys.Layer r = Some 1L)
                    |> List.sumBy _.Value

                let oneLayer =
                    [
                        OpCost.layerOps dims 13L 0L
                        OpCost.layerOps dims 1L 13L
                        OpCost.layerOps dims 1L 14L
                        OpCost.layerOps dims 1L 15L
                    ]
                    |> List.sumBy (fun ops -> (OpCost.sum (List.map snd ops)).Flops)

                Expect.equal layerFlops (float oneLayer) "layer 1's meters carry its layer"

            testCase "two stages: each model.forward names its stage and layers; tokens are metered once, by the head"
            <| fun () ->
                let sink = InMemorySink()
                let ops = tinyF32 ()
                let model = tinyModel ()
                let stages = Stage.split model.Config [ 1 ]
                let caches = stages |> List.map (fun s -> KvCache.create ops model s 13)

                do
                    use _ = Telemetry.install (Telemetry.ofSink sink)

                    Forward.runStages
                        ops
                        model
                        stages
                        caches
                        0
                        (promptIds "p0")
                        (Forward.hostRoundTrip ops)
                        ForwardOptions.last
                    |> ignore

                let forwards = named Forward.ForwardSpan TelemetryKind.Span sink.Records

                Expect.equal
                    (forwards
                     |> List.map (fun r -> int' TelemetryKeys.Stage r, int' "first_layer" r, int' "last_layer" r))
                    [ Some 0L, Some 0L, Some 0L; Some 1L, Some 1L, Some 1L ]
                    "stages"

                let layerStages =
                    named Forward.LayerSpan TelemetryKind.Span sink.Records
                    |> List.map (fun r -> int' TelemetryKeys.Layer r, int' TelemetryKeys.Stage r)

                Expect.equal layerStages [ Some 0L, Some 0L; Some 1L, Some 1L ] "layer spans per stage"

                Expect.equal
                    (named "model.tokens" TelemetryKind.Meter sink.Records |> List.sumBy _.Value)
                    13.0
                    "tokens once"

            testCase "a real run log renders the inference template with data in every chart the forward feeds"
            <| fun () ->
                let root = tempDir "runs"

                try
                    let dir =
                        use run =
                            RunContext.Create(
                                { RunOptions.create "inference" "M2a telemetry test" with
                                    ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuF32)
                                    RunsRoot = Some root
                                }
                            )

                        do
                            use _ = run.Install()

                            Greedy.run (tinyF32 ()) (tinyModel ()) (promptIds "p0") (GreedyOptions.parity 8)
                            |> ignore

                        run.Finish RunStatus.Ok
                        run.Directory

                    let log = RunLogReader.loadOrFail dir
                    Expect.isEmpty log.Problems "a clean log"
                    let out = Reports.render "inference" log
                    Expect.stringContains out.Markdown "tok/s" "rates"
                    Expect.isFalse (out.Markdown.Contains "| n/a |") "every headline value"

                    for chart in [ "decode_latency"; "layer_time"; "latency_hist" ] do
                        let _, svg = out.Svgs |> List.find (fun (f, _) -> f.Contains chart)
                        Expect.isFalse (svg.Contains "this run has no") (chart + " has data")
                finally
                    deleteDir root
        ]
