/// Local timing (bench tier: informational, never gates). Results are printed and also reported as gauges through the
/// ambient facade, so a host that installs a run log can chart them.
module Lrm.Telemetry.Tests.Bench

open System
open System.Diagnostics
open System.IO
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Telemetry.Tests.Common

let private best (runs: int) (f: unit -> unit) =
    [ 1..runs ]
    |> List.map (fun _ ->
        let clock = Stopwatch.StartNew()
        f ()
        clock.Elapsed.TotalMilliseconds)
    |> List.min

let tests =
    testList
        "bench"
        [
            testCase "run-log sink cost per record"
            <| fun () ->
                use tmp = new TempDir()
                let n = 200_000
                let mutable run = 0

                let ms =
                    best 3 (fun () ->
                        run <- run + 1

                        use ctx =
                            RunContext.Create(
                                { RunOptions.create "inference" "bench" with
                                    RunsRoot = Some tmp.Path
                                    RunId = Some(RunId.parse ("bench-" + Invariant.i32 run))
                                }
                            )

                        do
                            use _ = ctx.Install()

                            for i in 1..n do
                                use _ = Telemetry.span "model.layer" [ Attr.int "layer" (i % 28) ]
                                ()

                        ctx.Finish RunStatus.Ok)

                let ns = ms * 1e6 / float n
                printfn "bench: run-log span (with 1 attribute) costs %.0f ns, %.2f M records/s" ns (1e3 / ns)
                Telemetry.gauge "telemetry.sink.ns_per_record" ns TelemetryUnits.Nanoseconds []

            testCase "report rendering time per template"
            <| fun () ->
                use tmp = new TempDir()

                for kind in Synthetic.kinds do
                    let log = RunLogReader.loadOrFail (Synthetic.write kind tmp.Path)
                    let ms = best 3 (fun () -> Reports.render kind log |> ignore)
                    printfn "bench: %-10s %6d records, render %.1f ms" kind log.Events.Length ms
                    Telemetry.gauge "report.render_ms" ms "ms" [ Attr.str "template" kind ]

            testCase "a 20,000-step training report: render time and size"
            <| fun () ->
                let log = ChartEdgeTests.trainingLog 20_000 28
                let mutable output = None

                let ms = best 3 (fun () -> output <- Some(Reports.render "training" log))

                let svgs = output.Value.Svgs
                let bytes = svgs |> List.sumBy (fun (_, s) -> s.Length)

                for file, svg in svgs do
                    printfn "bench: %-28s %8d bytes" file svg.Length

                printfn "bench: training, %d records: render %.0f ms, %d SVG bytes" log.Events.Length ms bytes
                Telemetry.gauge "report.render_ms" ms "ms" [ Attr.str "template" "training-20k" ]

                Telemetry.gauge
                    "report.svg_bytes"
                    (float bytes)
                    TelemetryUnits.Bytes
                    [ Attr.str "template" "training-20k" ]

            testCase "run-log read speed"
            <| fun () ->
                use tmp = new TempDir()
                let dir = Synthetic.write "kernels" tmp.Path
                let records = (RunLogReader.loadOrFail dir).Events.Length
                let ms = best 3 (fun () -> RunLogReader.loadOrFail dir |> ignore)

                let bytes =
                    Directory.GetFiles(dir, "events.*.jsonl")
                    |> Array.sumBy (fun f -> FileInfo(f).Length)

                printfn "bench: read %d records (%d bytes) in %.1f ms" records bytes ms
                Telemetry.gauge "telemetry.read_records_per_s" (float records / (ms / 1e3)) TelemetryUnits.Count []
        ]
