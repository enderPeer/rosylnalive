/// Bench tier (local, informational): tokenizer load time and throughput in tokens per second, recorded as
/// tokenizer.bench.* gauges in the run log (TokenizerApi contract, "Telemetry"). The timed loops run on the no-op
/// facade: with a run log installed every Encode would write a span, and the bench would time the sink.
module Lrm.Tokenizer.Tests.Bench

open System
open System.Diagnostics
open System.Threading.Tasks
open Expecto
open Lrm.Core
open Lrm.Tokenizer
open Lrm.Tokenizer.Tests.Common

/// Runs `f` (which returns the tokens it handled) at least 3 times and for at least 1 s; returns tokens per second.
let private rate (f: unit -> int) =
    f () |> ignore
    let sw = Stopwatch.StartNew()
    let mutable runs = 0
    let mutable tokens = 0L

    while runs < 3 || sw.Elapsed.TotalSeconds < 1.0 do
        tokens <- tokens + int64 (f ())
        runs <- runs + 1

    float tokens / sw.Elapsed.TotalSeconds

type private Rate =
    {
        Mode: string
        Threads: int
        Input: string
        TokensPerS: float
    }

let tests =
    testList
        "bench"
        [
            testCase "load, encode and decode throughput on the corpus"
            <| fun () ->
                let path = QwenTokenizer.defaultPath ()

                if not (IO.File.Exists path) then
                    tokenizerOrSkip "bench" |> ignore

                // Measure on the no-op facade, whatever a caller installed.
                let rates, loadMs =
                    use _ = Telemetry.install Telemetry.noop

                    let loads =
                        [
                            for _ in 1..3 do
                                let sw = Stopwatch.StartNew()
                                QwenTokenizer.load path |> ignore
                                sw.Elapsed.TotalMilliseconds
                        ]

                    let t = QwenTokenizer.load path

                    let texts =
                        [|
                            for e in jsonLines (corpusPath ()) do
                                if text e "kind" = "encode" then
                                    yield text e "text"
                        |]

                    let ids = texts |> Array.map t.Encode
                    let total = ids |> Array.sumBy Array.length

                    let encodeAll () =
                        let mutable n = 0

                        for s in texts do
                            n <- n + t.Encode(s).Length

                        n

                    let parallelEncode () =
                        Parallel.For(
                            0,
                            texts.Length,
                            ParallelOptions(MaxDegreeOfParallelism = 8),
                            fun i -> t.Encode texts[i] |> ignore
                        )
                        |> ignore

                        total

                    let decodeAll () =
                        let mutable n = 0

                        for x in ids do
                            t.Decode x |> ignore
                            n <- n + x.Length

                        n

                    // The corpus's C# cases joined: one prompt-sized input.
                    let csharp =
                        String.Join(
                            "\n\n",
                            [|
                                for e in jsonLines (corpusPath ()) do
                                    if text e "cat" = "csharp" then
                                        yield text e "text"
                            |]
                        )

                    // Long runs of one character (indentation, rules such as '=====', blobs): every BPE step looks up
                    // a self-pair such as ('a', 'a'), the worst case for a weak merge-key hash.
                    let runs =
                        String.Join("\n", [| for c in [ 'a'; ' '; '='; '!' ] -> String(c, 1 <<< 16) |])

                    let measure mode threads input f =
                        {
                            Mode = mode
                            Threads = threads
                            Input = input
                            TokensPerS = rate f
                        }

                    [
                        measure "encode_cold" 1 "corpus" (fun () ->
                            t.ClearCache()
                            encodeAll ())
                        measure "encode_warm" 1 "corpus" encodeAll
                        measure "encode_warm" 8 "corpus" parallelEncode
                        measure "decode" 1 "corpus" decodeAll
                        measure "encode_cold" 1 "csharp" (fun () ->
                            t.ClearCache()
                            t.Encode(csharp).Length)
                        measure "encode_cold" 1 "runs" (fun () ->
                            t.ClearCache()
                            t.Encode(runs).Length)
                    ],
                    List.min loads

                use _ = TokenizerRun.install ()
                use _ = Telemetry.scope [ Attr.str TelemetryKeys.Component "tokenizer" ]
                printfn "bench: load tokenizer.json                    %8s ms" (Invariant.format $"{loadMs:F1}")
                Telemetry.gauge "tokenizer.bench.load" (loadMs * 1e6) TelemetryUnits.Nanoseconds []

                for r in rates do
                    printfn
                        "bench: %-34s %2d thread(s)  %12s tokens/s"
                        (r.Mode + " " + r.Input)
                        r.Threads
                        (Invariant.format $"{r.TokensPerS:N0}")

                    Telemetry.gauge
                        "tokenizer.bench.tokens_per_s"
                        r.TokensPerS
                        "token/s"
                        [
                            Attr.str "mode" r.Mode
                            Attr.int "threads" r.Threads
                            Attr.str "input" r.Input
                        ]
        ]
