/// Bench tier (local, informational): tokenizer load time and throughput in tokens per second, recorded as
/// tokenizer.bench.* gauges in the run log (TokenizerApi contract, "Telemetry").
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

let private record (mode: string) (threads: int) (input: string) (tokensPerS: float) =
    printfn
        "bench: %-34s %2d thread(s)  %12s tokens/s"
        (mode + " " + input)
        threads
        (Invariant.format $"{tokensPerS:N0}")

    Telemetry.gauge
        "tokenizer.bench.tokens_per_s"
        tokensPerS
        "token/s"
        [ Attr.str "mode" mode; Attr.int "threads" threads; Attr.str "input" input ]

let tests =
    testList
        "bench"
        [
            testCase "load, encode and decode throughput on the corpus"
            <| fun () ->
                use _ = TokenizerRun.install ()
                use _ = Telemetry.scope [ Attr.str TelemetryKeys.Component "tokenizer" ]
                let path = QwenTokenizer.defaultPath ()

                if not (IO.File.Exists path) then
                    tokenizerOrSkip "bench" |> ignore

                let loads =
                    [
                        for _ in 1..3 do
                            let sw = Stopwatch.StartNew()
                            QwenTokenizer.load path |> ignore
                            sw.Elapsed.TotalMilliseconds
                    ]

                let loadMs = List.min loads
                printfn "bench: load tokenizer.json                    %8s ms" (Invariant.format $"{loadMs:F1}")
                Telemetry.gauge "tokenizer.bench.load" (loadMs * 1e6) TelemetryUnits.Nanoseconds []

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

                record
                    "encode_cold"
                    1
                    "corpus"
                    (rate (fun () ->
                        t.ClearCache()
                        encodeAll ()))

                record "encode_warm" 1 "corpus" (rate encodeAll)

                let parallelEncode () =
                    Parallel.For(
                        0,
                        texts.Length,
                        ParallelOptions(MaxDegreeOfParallelism = 8),
                        fun i -> t.Encode texts[i] |> ignore
                    )
                    |> ignore

                    total

                record "encode_warm" 8 "corpus" (rate parallelEncode)

                let decodeAll () =
                    let mutable n = 0

                    for x in ids do
                        t.Decode x |> ignore
                        n <- n + x.Length

                    n

                record "decode" 1 "corpus" (rate decodeAll)

                // One long C# text: the corpus's csharp cases joined, as a prompt-sized input.
                let csharp =
                    String.Join(
                        "\n\n",
                        [|
                            for e in jsonLines (corpusPath ()) do
                                if text e "cat" = "csharp" then
                                    yield text e "text"
                        |]
                    )

                record
                    "encode_cold"
                    1
                    "csharp"
                    (rate (fun () ->
                        t.ClearCache()
                        t.Encode(csharp).Length))
        ]
