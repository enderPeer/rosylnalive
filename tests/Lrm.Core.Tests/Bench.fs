/// Local timing measurements (bench tier: informational, never gates).
module Lrm.Core.Tests.Bench

open System
open System.Diagnostics
open Expecto
open Lrm.Core

/// Discards every record, so a measurement sees only the facade's own cost.
type private NullSink() =
    interface ITelemetrySink with
        member _.Write _ = ()

/// Nanoseconds per call of `f`, the best of five runs of `n` calls.
let private nsPerCall (n: int) (f: unit -> unit) =
    let best =
        [ 1..5 ]
        |> List.map (fun _ ->
            let clock = Stopwatch.StartNew()

            for _ in 1..n do
                f ()

            clock.Elapsed.TotalMilliseconds * 1e6 / float n)
        |> List.min

    best

let tests =
    testList
        "bench"
        [
            testCase "telemetry span cost"
            <| fun () ->
                let n = 1_000_000

                let noop =
                    nsPerCall n (fun () ->
                        use _ = Telemetry.span "model.layer" []
                        ())

                let sinkTelemetry = Telemetry.ofSink (NullSink())

                let nullSink =
                    nsPerCall n (fun () ->
                        use _ = sinkTelemetry.Span("model.layer", [])
                        ())

                let withAttrs =
                    nsPerCall n (fun () ->
                        use _ = sinkTelemetry.Span("model.layer", [ Attr.int "layer" 3 ])
                        ())

                printfn "span: no-op %.1f ns, null sink %.1f ns, null sink + 1 attr %.1f ns" noop nullSink withAttrs

            testCase "Philox4x32-10 and SHA-256 throughput"
            <| fun () ->
                let r = DetRng(1UL, 2UL)
                let words = Array.zeroCreate<uint32>(1 <<< 20)
                let clock = Stopwatch.StartNew()
                r.Fill(Span words)
                let philox = float words.Length / clock.Elapsed.TotalSeconds / 1e6
                let data = Array.zeroCreate<byte>(64 <<< 20)
                clock.Restart()
                Hash.sha256 data |> ignore
                let sha = 64.0 / clock.Elapsed.TotalSeconds
                printfn "DetRng: %.0f M words/s; SHA-256: %.0f MiB/s" philox sha
        ]
