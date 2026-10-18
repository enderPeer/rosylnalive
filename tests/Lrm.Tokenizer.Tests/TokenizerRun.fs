/// The run log of the tokenizer tests (RunLog contract, kind "tokenizer"). The corpus, oracle-probe and bench tests
/// report through the facade with the reserved name tokenizer.case and the tokenizer.* records of
/// docs/contracts/tokenizer.md ("Telemetry"); this module also writes them to LRM_DATA/runs/<runId>, so
/// `lrm report <runId>` renders them. The run is created by the first test that asks for it, and Main finishes it.
module Lrm.Tokenizer.Tests.TokenizerRun

open System
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Telemetry

let private gate = obj ()
let mutable private run: RunContext option = None

/// The ICU version behind string.Normalize, read from the runtime's internal interop (informational: the oracle's
/// NFC is Unicode 9, and ADR M1-0001 records what the split points cover).
let icuVersion () =
    try
        let t = typeof<obj>.Assembly.GetType "Interop+Globalization"

        let m =
            t.GetMethod("GetICUVersion", Reflection.BindingFlags.NonPublic ||| Reflection.BindingFlags.Static)

        let v = m.Invoke(null, [||]) :?> int
        sprintf "%d.%d.%d.%d" ((v >>> 24) &&& 0xFF) ((v >>> 16) &&& 0xFF) ((v >>> 8) &&& 0xFF) (v &&& 0xFF)
    with _ ->
        "unknown"

let private create () =
    let tier = Lrm.Testing.Env.tryGet "LRM_TEST_TIER" |> Option.defaultValue "cpu"

    let r =
        RunContext.Create(
            { RunOptions.create "tokenizer" ("Tokenizer (M1), " + tier + " tier") with
                GitSha = Lrm.Testing.Env.commit ()
                Params =
                    [
                        "tier", tier
                        "source", "tests/Lrm.Tokenizer.Tests (M1)"
                        "corpus", "goldens/tokenizer/g1/cases.jsonl"
                        "oracle", "HF tokenizers 0.23.3 (goldens/tokenizer/g1, tests/Lrm.Tokenizer.Tests/oracle)"
                        "tokenizer_sha256", Lrm.Tokenizer.QwenTokenizer.PinnedSha256
                        "runtime", RuntimeInformation.FrameworkDescription
                        "os", RuntimeInformation.OSDescription
                        "icu", icuVersion ()
                    ]
            }
        )

    printfn "tokenizer run log: %s" r.Directory
    r

/// The run's facade (the run is created on first use).
let telemetry () =
    lock gate (fun () ->
        match run with
        | Some r -> r.Telemetry
        | None ->
            let r = create ()
            run <- Some r
            r.Telemetry)

/// Makes the run's facade ambient until disposed.
let install () : IDisposable = Telemetry.install (telemetry ())

/// The run folder, if a test created the run.
let directory () =
    lock gate (fun () -> run |> Option.map _.Directory)

/// Finishes the run, if a test created one: ok when the exe passed, failed otherwise.
let finish (exitCode: int) =
    lock gate (fun () ->
        match run with
        | None -> ()
        | Some r ->
            r.Finish(if exitCode = 0 then RunStatus.Ok else RunStatus.Failed)
            run <- None)
