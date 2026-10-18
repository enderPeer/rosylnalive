/// Greedy generation on the canonical schedule (ForwardOpsAndQwenModel contract): one prefill, then single-token decode
/// steps, on one or more registered backends (Backends.fs). Owner: M2a.
///
///     lrm generate [--model DIR|tiny|qwen3-0.6b] [--backend cpu-f32[,cpu-f64,...]]
///                  (--ids 1,2,3 | --ids-from FILE[:TENSOR] | --prompt TEXT) [--steps 16] [--stop-at-eos] [--margins]
///                  [--expect FILE:TENSOR] [--case NAME] [--threads 8] [--warmup 0] [--run-log] [--title TEXT]
///
/// EOS is ignored unless --stop-at-eos is given (parity runs ignore it). --expect compares the ids with an integer
/// tensor, such as goldens/tiny/g1/greedy.safetensors:p0.ids.torch32, and records a greedy_match golden.compare check.
/// --run-log writes a run log of kind inference (LRM_DATA/runs), one node per backend, for `lrm report`. --warmup N
/// runs N untimed, unrecorded greedy steps first, so JIT compilation and the weight copies stay out of the rates.
module Lrm.Cli.Commands.Generate

open System
open System.IO
open Lrm.Core
open Lrm.Tensor
open Lrm.Telemetry
open Lrm.Models.Qwen
open Lrm.Cli.Commands.Backends

let private usage () =
    eprintfn "usage: lrm generate [--model DIR|tiny|qwen3-0.6b] [--backend NAME[,NAME...]]"
    eprintfn "                    (--ids 1,2,3 | --ids-from FILE[:TENSOR] | --prompt TEXT) [--steps 16]"
    eprintfn "                    [--stop-at-eos] [--margins] [--expect FILE:TENSOR] [--case NAME] [--threads 8]"
    eprintfn "                    [--run-log] [--title TEXT]"
    eprintfn ""
    eprintfn "  --model       a model folder (REVISION, SHA256SUMS); tiny = goldens/tiny/g1/model; default qwen3-0.6b"
    eprintfn "  --backend     registered backends (lrm backends); default cpu-f32"
    eprintfn "  --ids-from    an integer tensor of a safetensors file (default tensor input_ids)"
    eprintfn "  --prompt      text; UTF-8 bytes for a byte-level model (the tiny model)"
    eprintfn "  --steps       tokens to generate (default 16)"
    eprintfn "  --stop-at-eos stop after the config's EOS id (parity runs ignore EOS)"
    eprintfn "  --margins     print each step's top-2 logit margin"
    eprintfn "  --warmup      untimed, unrecorded greedy steps before the measured run (default 0)"
    eprintfn "  --expect      the ids the run must give, as FILE:TENSOR; exit 1 when they differ"
    eprintfn "  --run-log     write a run log of kind inference under LRM_DATA/runs"
    2

let private values =
    [
        "model"
        "backend"
        "ids"
        "ids-from"
        "prompt"
        "steps"
        "warmup"
        "expect"
        "case"
        "threads"
        "title"
    ]

let private switches = [ "stop-at-eos"; "margins"; "run-log" ]

/// The case of an --expect spec: the tensor's first dot segment (p0 for greedy.safetensors:p0.ids.torch32).
let private caseOfSpec (spec: string) =
    let tensor =
        match spec.LastIndexOf ':' with
        | i when i > 1 -> spec.Substring(i + 1)
        | _ -> Path.GetFileNameWithoutExtension spec

    match tensor.IndexOf '.' with
    | i when i > 0 -> tensor.Substring(0, i)
    | _ -> tensor

let private greedyMatch (pair: TolerancePair option) (case: string) (got: int[]) (expected: int[]) =
    let matched =
        Seq.zip got expected |> Seq.takeWhile (fun (a, b) -> a = b) |> Seq.length

    Compare.check
        "greedy_match"
        BoundOp.AtLeast
        (float matched)
        (float expected.Length)
        (int64 expected.Length)
        [
            match pair with
            | Some p -> yield Attr.str "pair" (TolerancePair.name p)
            | None -> ()
            yield Attr.str "prompt" case
            yield Attr.str "tap" "greedy.ids"
            yield Attr.int TelemetryKeys.Layer -1
            yield Attr.str "kind" "chain"
        ]

let private run (argv: string[]) : int =
    match Cli.parse values switches argv with
    | Error e ->
        eprintfn "lrm generate: %s" e
        usage ()
    | Ok p ->
        let setup =
            Cli.result {
                let! dir = Cli.modelDir (p.Get "model")
                let! backends = Cli.backends (p.Get "backend")
                let! steps = Cli.intOption p "steps" 16
                let! threads = Cli.intOption p "threads" (min 8 Environment.ProcessorCount)
                let! warmup = Cli.intOption p "warmup" 0

                if steps < 1 || threads < 1 || warmup < 0 then
                    return! Error "--steps and --threads must be at least 1, --warmup at least 0"
                else
                    let! expected =
                        match p.Get "expect" with
                        | None -> Ok None
                        | Some spec -> Cli.idsFrom spec "" |> Result.map (fun ids -> Some(spec, ids))

                    return dir, backends, steps, threads, warmup, expected
            }

        match setup with
        | Error e ->
            eprintfn "lrm generate: %s" e
            2
        | Ok(dir, backends, steps, threads, warmup, expected) ->
            try
                use w = Qwen3Weights.Open dir

                match Cli.promptIds p w with
                | Error e ->
                    eprintfn "lrm generate: %s" e
                    2
                | Ok(prompt, source) ->
                    let options =
                        {
                            Steps = steps
                            StopAtEos = p.Has "stop-at-eos"
                            Margins = p.Has "margins"
                        }

                    let case =
                        p.Get "case"
                        |> Option.orElse (expected |> Option.map (fst >> caseOfSpec))
                        |> Option.defaultValue "cli"

                    let title =
                        p.Get "title"
                        |> Option.defaultValue (
                            Invariant.format
                                $"lrm generate: {Path.GetFileName w.Dir} ({w.Revision.Text}), {prompt.Length} prompt tokens, {steps} steps"
                        )

                    let host =
                        Cli.RunHost(
                            "inference",
                            title,
                            p.Has "run-log",
                            [
                                "model", Cli.displayPath w.Dir
                                "revision", w.Revision.Text
                                "model_sha256", w.Sha256.Hex
                                "prompt", source
                                "prompt_tokens", Invariant.i32 prompt.Length
                                "steps", Invariant.i32 steps
                                "stop_at_eos", (if options.StopAtEos then "true" else "false")
                                "threads", Invariant.i32 threads
                                "warmup_steps", Invariant.i32 warmup
                                "backends", backends |> List.map _.Name |> String.concat ","
                                "command", "lrm generate"
                            ]
                        )

                    printfn
                        "lrm generate: %s (%s), prompt %d tokens (%s), %d steps, EOS %s"
                        w.Dir
                        w.Revision.Text
                        prompt.Length
                        source
                        steps
                        (if options.StopAtEos then "stops" else "ignored")

                    let mutable failed = false
                    let results = Collections.Generic.List<string * int[]>()

                    for i, f in List.indexed backends do
                        let runner =
                            f.Create
                                w
                                {
                                    Threads = threads
                                    MaxContext = Greedy.positions prompt.Length steps
                                }

                        printfn "%s: %s" f.Name (Cli.describe runner.ExecutionClass)

                        if warmup > 0 then
                            runner.Greedy(prompt, GreedyOptions.parity (min warmup steps)) |> ignore

                        do
                            use _ = host.Install(i, runner.ExecutionClass)
                            let r = runner.Greedy(prompt, options)
                            results.Add(f.Name, r.Ids)
                            printfn "  ids: %s" (r.Ids |> Array.map Invariant.i32 |> String.concat " ")

                            match Cli.text w r.Ids with
                            | Some t -> printfn "  text: \"%s\"" (Cli.printable t)
                            | None -> ()

                            if r.StoppedAtEos then
                                printfn "  stopped at EOS after %d tokens" r.Ids.Length

                            if r.Margins.Length > 0 then
                                let m = Array.min r.Margins
                                let at = Array.IndexOf(r.Margins, m)

                                printfn
                                    "  margins: min %s at step %d; %s"
                                    (Invariant.f64 m)
                                    at
                                    (r.Margins
                                     |> Array.map (fun x ->
                                         x.ToString("G4", Globalization.CultureInfo.InvariantCulture))
                                     |> String.concat " ")

                            match expected with
                            | None -> ()
                            | Some(spec, ids) ->
                                let oracle =
                                    if spec.Contains "torch64p" then "torch64p"
                                    elif spec.Contains "torch32" then "torch32"
                                    else ""

                                let expectedIds =
                                    if r.StoppedAtEos && r.Ids.Length < ids.Length then
                                        ids[.. r.Ids.Length - 1]
                                    else
                                        ids

                                let c = greedyMatch (Cli.pairFor f.Name oracle) case r.Ids expectedIds

                                printfn
                                    "  expect: %d of %d steps match %s%s"
                                    (int c.Measured)
                                    expectedIds.Length
                                    spec
                                    (if c.Passed then "" else " (MISMATCH)")

                                if not c.Passed then
                                    failed <- true

                        host.PrintRates i

                    if results.Count > 1 then
                        let first = snd results[0]

                        let same = results |> Seq.forall (fun (_, ids) -> ids = first)

                        printfn
                            "backends %s: %s"
                            (results |> Seq.map fst |> String.concat ", ")
                            (if same then
                                 "identical ids"
                             else
                                 "the ids differ (different classes may)")

                    host.Finish(if failed then RunStatus.Failed else RunStatus.Ok)
                    if failed then 1 else 0
            with
            | :? Qwen3WeightsException as e ->
                eprintfn "lrm generate: %s" e.Message
                1
            | :? Qwen3ConfigException as e ->
                eprintfn "lrm generate: %s" e.Message
                1
            | :? IntegrityException as e ->
                eprintfn "lrm generate: %s" e.Message
                1
            | :? ForwardException as e ->
                eprintfn "lrm generate: %s" e.Message
                1
            | :? ArgumentException as e ->
                eprintfn "lrm generate: %s" e.Message
                1
            | :? IOException as e ->
                eprintfn "lrm generate: %s" e.Message
                1

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "generate"
            Summary = "Greedy generation on the Qwen forward (prefill, then single-token decode steps)."
            Run = run
        |}
    ]
