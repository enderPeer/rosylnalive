/// Shared helpers of the tokenizer tests: paths, the pinned tokenizer, JSON lines and the visible skip.
module Lrm.Tokenizer.Tests.Common

open System
open System.IO
open System.Text.Json
open Expecto
open Lrm.Core
open Lrm.Tokenizer

/// The repo root (LRM_REPO_ROOT from tools/test.ps1, else the folder holding Lrm.slnx above the exe).
let repo () = Lrm.Testing.Env.repoRoot ()

/// A path under the repo root.
let repoPath (relative: string) = Path.Combine(repo (), relative)

/// The oracle probe data written by tests/Lrm.Tokenizer.Tests/oracle/probe_oracle.py.
let oraclePath (name: string) =
    repoPath (Path.Combine("tests", "Lrm.Tokenizer.Tests", "oracle", name))

/// O1's tokenizer corpus (GoldenSet contract, "The tokenizer corpus").
let corpusPath () =
    repoPath (Path.Combine("goldens", "tokenizer", "g1", "cases.jsonl"))

/// Every line of a JSON-lines file, parsed.
let jsonLines (path: string) : JsonElement list =
    [
        for line in File.ReadLines path do
            if line.Length > 0 then
                use doc = JsonDocument.Parse line
                doc.RootElement.Clone()
    ]

/// An int array property.
let ints (e: JsonElement) (name: string) =
    [| for x in e.GetProperty(name).EnumerateArray() -> x.GetInt32() |]

/// A string property.
let text (e: JsonElement) (name: string) = e.GetProperty(name).GetString()

/// The pinned tokenizer from LRM_DATA, loaded once; None when tokenizer.json is absent.
let private pinned: Lazy<QwenTokenizer option> =
    lazy (QwenTokenizer.tryLoadDefault ())

/// Why tests that need tokenizer.json are skipped when it is absent.
let skipReason () =
    sprintf
        "tokenizer.json is not at %s: the CI download waits for the user's AP-1 approval (DataRootAndToolchain), so this test is SKIPPED, not passed"
        (QwenTokenizer.defaultPath ())

/// The pinned tokenizer, or a visible skip: a line on stderr, a tokenizer.skipped event, and an ignored test.
let tokenizerOrSkip (test: string) : QwenTokenizer =
    match pinned.Force() with
    | Some t -> t
    | None ->
        let reason = skipReason ()
        eprintfn "SKIPPED %s: %s" test reason
        use _ = TokenizerRun.install ()

        Telemetry.event
            "tokenizer.skipped"
            [
                Attr.str "test" test
                Attr.str "reason" reason
                Attr.str TelemetryKeys.Component "tokenizer"
            ]

        skiptest reason

/// A string from UTF-16 code units in big-endian hex, 4 digits per unit (the corpus's utf16 field). Lone surrogates
/// are built here at run time, so no source file holds one.
let ofUtf16Hex (h: string) =
    String(Array.init (h.Length / 4) (fun i -> char (Convert.ToUInt16(h.Substring(4 * i, 4), 16))))

/// Code points as a string ("U+0041 U+1F600").
let describe (s: string) =
    s.EnumerateRunes()
    |> Seq.map (fun r -> sprintf "U+%04X" r.Value)
    |> String.concat " "

/// The first index where two arrays differ, or -1.
let firstDiff (a: int[]) (b: int[]) =
    let n = min a.Length b.Length
    let mutable i = 0

    while i < n && a[i] = b[i] do
        i <- i + 1

    if i = n && a.Length = b.Length then -1 else i
