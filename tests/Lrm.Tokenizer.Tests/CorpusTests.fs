/// Acceptance on the pinned tokenizer.json (TokenizerApi contract, "Verification"): O1's corpus exact, the probe's
/// random texts and id lists exact, the 26 added tokens, no BOS, and the known-answer vectors E1 to E11 and D1 to D6.
/// Without tokenizer.json (CI runners until the user approves the download, AP-1) each test is SKIPPED visibly: an
/// ignored test, a line on stderr and a tokenizer.skipped event in the run log.
module Lrm.Tokenizer.Tests.CorpusTests

open System
open System.IO
open System.Text
open System.Text.Json
open Expecto
open Lrm.Core
open Lrm.Tokenizer
open Lrm.Tokenizer.Tests.Common

/// Runs a JSON-lines file through the library's runner (Corpus.run, or Corpus.verify for O1's corpus) on the run log,
/// and fails with the first differences.
let private expectExact (run: unit -> CorpusResult) =
    let r =
        use _ = TokenizerRun.install ()
        run ()

    if not r.Exact then
        failtestf
            "%s: %d of %d cases differ from the oracle:\n%s"
            r.Source
            r.Failures.Length
            r.Cases
            (r.Failures
             |> Seq.truncate 25
             |> Seq.map (fun c -> c.Id + ": " + c.Detail)
             |> String.concat "\n")

    r

/// Vectors E1 to E11 (Encode) and D1 to D6 (Decode) of docs/contracts/tokenizer.md, from the oracle.
let encodeVectors: (string * int list) list =
    [
        "Hello world", [ 9707; 1879 ]
        "using System;", [ 970; 739; 26 ]
        "<|im_start|>user\nHi<|im_end|>", [ 151644; 872; 198; 13048; 151645 ]
        "e\u0301", [ 963 ]
        Char.ConvertFromUtf32 0x1F600, [ 141334 ]
        "a\u07fd\u0334", [ 64; 155; 121; 136; 112 ]
        "x\ufffe\u0301", [ 87; 5564; 122; 53839 ]
        "public static int Add(int a, int b) => a + b;",
        [ 888; 1099; 526; 2691; 1548; 264; 11; 526; 293; 8; 589; 264; 488; 293; 26 ]
        "", []
        " ", [ 220 ]
        "\u4e2d\u6587", [ 104811 ]
    ]

let decodeVectors: (int list * string) list =
    [
        [ 151643 ], "<|endoftext|>"
        [ 9707; 1879 ], "Hello world"
        [ 160; 151700; 116; 255 ], "\u4e2d"
        [ 5691 ], "\ufffd"
        [ 151669 ], ""
        [], ""
    ]

/// Lines of every kind for the toy tokenizer of ToyTests (" the" is id 258, and each byte b is id b), each meant to pass
/// or fail by one rule of GoldenSet's "The tokenizer corpus". Escapes stay JSON escapes, so the source is ASCII.
let private toyLines =
    [
        true, """{"cat":"t","id":"t-0001","ids":[258],"kind":"encode","text":" the"}"""
        false, """{"cat":"t","id":"t-0002","ids":[258,1],"kind":"encode","text":" the"}"""
        true, """{"bytes":"ff","cat":"t","id":"t-0003","ids":[239,191,189],"kind":"encode_bytes","text":"\ufffd"}"""
        false, """{"bytes":"ff","cat":"t","id":"t-0004","ids":[239,191,189],"kind":"encode_bytes","text":"x"}"""
        true, """{"bytes":"20746865","cat":"t","id":"t-0005","ids":[258],"kind":"decode","text":" the"}"""
        false, """{"bytes":"20746865","cat":"t","id":"t-0006","ids":[258],"kind":"decode","text":" thf"}"""
        false, """{"bytes":"00","cat":"t","id":"t-0007","ids":[258],"kind":"decode","text":" the"}"""
        true, """{"cat":"t","error":"TypeError","id":"t-0008","kind":"encode_reject","utf16":"0061d8000062"}"""
        false, """{"cat":"t","error":"TypeError","id":"t-0009","kind":"encode_reject","utf16":"00610062"}"""
        false, """{"cat":"t","id":"t-0010","kind":"tokenize","text":"x"}"""
        false, """{"cat":"t","id":"t-0011","kind":"encode","text":"x"}"""
        true, """{"ids":[258],"kind":"encode","text":" the"}"""
    ]

/// Writes lines to a fresh temporary file, runs `f` on its path and deletes it.
let private withFile (lines: string list) (f: string -> 'a) =
    let path =
        Path.Combine(Path.GetTempPath(), "lrm-m1-corpus-" + Guid.NewGuid().ToString("N") + ".jsonl")

    File.WriteAllText(path, String.concat "\n" lines + "\n")

    try
        f path
    finally
        File.Delete path

let private strAttr (key: string) (r: TelemetryRecord) =
    r.Attrs
    |> List.tryPick (fun a ->
        match a.Value with
        | AttrValue.Str s when a.Key = key -> Some s
        | _ -> None)

let private intAttr (key: string) (r: TelemetryRecord) =
    r.Attrs
    |> List.tryPick (fun a ->
        match a.Value with
        | AttrValue.Int v when a.Key = key -> Some v
        | _ -> None)

/// The library's corpus runner (Corpus) on the toy tokenizer; needs no tokenizer.json, so CI runs it.
let private runnerTests =
    testList
        "runner"
        [
            testCase "Corpus.run applies each kind's rule and records one tokenizer.case check per line"
            <| fun () ->
                let t = QwenTokenizer.parse (ToyTests.toyJson ignore)
                let sink = InMemorySink()

                let r =
                    withFile (List.map snd toyLines) (fun path ->
                        use _ = Telemetry.install (Telemetry.ofSink sink)
                        Corpus.run t "toy" path)

                Expect.equal r.Source "toy" "source"
                Expect.equal r.Cases toyLines.Length "cases"
                Expect.equal r.Passed (toyLines |> List.filter fst |> List.length) "passed"
                Expect.isFalse r.Exact "not exact"

                Expect.equal
                    (r.Failures |> List.map _.Id)
                    [ "t-0002"; "t-0004"; "t-0006"; "t-0007"; "t-0009"; "t-0010"; "t-0011" ]
                    "the failing cases, in file order"

                let detail id =
                    (r.Failures |> List.find (fun c -> c.Id = id)).Detail

                Expect.stringStarts (detail "t-0002") "ids differ at 1 (expected 2 ids, got 1)" "ids"
                Expect.stringContains (detail "t-0004") "do not decode" "encode_bytes text"
                Expect.stringContains (detail "t-0006") "Decode gave" "decode text"
                Expect.stringContains (detail "t-0007") "DecodeBytes differs" "decode bytes"
                Expect.stringContains (detail "t-0009") "accepted" "encode_reject"
                Expect.stringContains (detail "t-0010") "unknown kind 'tokenize'" "kind"
                Expect.stringContains (detail "t-0011") "no 'ids'" "a missing field fails the case"

                let checks =
                    sink.Records |> List.filter (fun x -> x.Name = TelemetryNames.TokenizerCase)

                Expect.equal checks.Length toyLines.Length "one tokenizer.case per line"

                Expect.equal
                    (checks |> List.map (fun c -> c.Value = 1.0))
                    (List.map fst toyLines)
                    "each check's outcome"

                for c in checks do
                    Expect.equal (strAttr "source" c) (Some "toy") "source"
                    Expect.equal (strAttr TelemetryKeys.Component c) (Some "tokenizer") "component"
                    Expect.equal (strAttr "detail" c |> Option.isSome) (c.Value = 0.0) "detail on failure only"

                let last = List.last checks
                Expect.equal (strAttr TelemetryKeys.Case last) (Some "toy-0012") "a line without id"
                Expect.equal (strAttr "cat" last) (Some "probe-encode") "a line without cat"
                Expect.equal (intAttr "tokens" last) (Some 1L) "expected ids"

                let corpus = sink.Records |> List.filter (fun x -> x.Name = "tokenizer.corpus")
                Expect.equal corpus.Length 1 "one tokenizer.corpus event"
                Expect.equal (intAttr "cases" corpus.Head) (Some 12L) "cases"
                Expect.equal (intAttr "passed" corpus.Head) (Some 5L) "passed"
                Expect.equal (intAttr "failed" corpus.Head) (Some 7L) "failed"
                Expect.equal (strAttr "sha256" corpus.Head) (Some r.Sha256) "sha256"

            testCase "Corpus.run of passing lines is exact, and telemetry changes no outcome"
            <| fun () ->
                let t = QwenTokenizer.parse (ToyTests.toyJson ignore)
                let lines = toyLines |> List.filter fst |> List.map snd

                withFile lines (fun path ->
                    let plain = Corpus.run t "toy" path

                    let recorded =
                        use _ = Telemetry.install (Telemetry.ofSink (InMemorySink()))
                        Corpus.run t "toy" path

                    Expect.isTrue plain.Exact "exact"
                    Expect.equal recorded plain "same result with a sink")

            testCase "Corpus.verify refuses any file but O1's corpus, naming both hashes"
            <| fun () ->
                let t = QwenTokenizer.parse (ToyTests.toyJson ignore)

                withFile (List.map snd toyLines) (fun path ->
                    let sha = ContentId.hex (Hash.sha256 (File.ReadAllBytes path))

                    let message =
                        try
                            Corpus.verify t path |> ignore
                            failtest "Corpus.verify ran a file that is not O1's corpus"
                        with :? InvalidDataException as e ->
                            e.Message

                    Expect.stringContains message sha "the file's hash"
                    Expect.stringContains message Corpus.PinnedSha256 "the pin")

            testCase "a line that is not JSON raises InvalidDataException naming the case"
            <| fun () ->
                let t = QwenTokenizer.parse (ToyTests.toyJson ignore)

                withFile [ snd toyLines.Head; "{not json" ] (fun path ->
                    let message =
                        try
                            Corpus.run t "toy" path |> ignore
                            failtest "Corpus.run accepted a line that is not JSON"
                        with :? InvalidDataException as e ->
                            e.Message

                    Expect.stringContains message "case 2 is not JSON" "names the case")

            testCase "the pins name GoldenSet v1's corpus"
            <| fun () ->
                Expect.equal Corpus.RepoPath "goldens/tokenizer/g1/cases.jsonl" "path"
                Expect.equal Corpus.PinnedCases 5597 "cases"

                Expect.equal
                    Corpus.PinnedSha256
                    "35ff0c43a1bc4325588b1a22953f861784e4ca76e0f3f863a5f70ff9ceb16c66"
                    "GoldenSet, The tokenizer set"
        ]

let tests =
    testList
        "corpus"
        [
            runnerTests

            testCase "O1's corpus is exact (goldens/tokenizer/g1, 5,597 cases)"
            <| fun () ->
                let path = corpusPath ()
                // The digest is checked without tokenizer.json too, so CI holds the corpus to its pin.
                let sha = ContentId.hex (Hash.sha256 (File.ReadAllBytes path))
                Expect.equal sha Corpus.PinnedSha256 "cases.jsonl is the GoldenSet v1 corpus"
                Expect.equal (jsonLines path).Length Corpus.PinnedCases "cases"
                let t = tokenizerOrSkip "corpus"
                let r = expectExact (fun () -> Corpus.verify t path)
                Expect.equal r.Cases 5597 "cases"
                Expect.equal r.Passed 5597 "exact"

            testCase "the probe's random texts and id lists are exact (oracle/encode-cases.jsonl)"
            <| fun () ->
                let t = tokenizerOrSkip "encode-cases"

                let r =
                    expectExact (fun () -> Corpus.run t "probe" (oraclePath "encode-cases.jsonl"))

                Expect.equal r.Cases 1800 "cases"

            testCase "vectors E1 to E11 and D1 to D6"
            <| fun () ->
                let t = tokenizerOrSkip "vectors"

                for input, ids in encodeVectors do
                    Expect.equal (t.Encode input) (Array.ofList ids) (describe input)

                for ids, output in decodeVectors do
                    Expect.equal (t.Decode(Array.ofList ids)) output (sprintf "%A" ids)

            testCase "all 26 added tokens round-trip, with the manifest's ids and flags"
            <| fun () ->
                let t = tokenizerOrSkip "added-tokens"

                use manifest =
                    JsonDocument.Parse(File.ReadAllText(repoPath "goldens/tokenizer/g1/manifest.json"))

                let want =
                    [
                        for a in manifest.RootElement.GetProperty("added_tokens").EnumerateArray() ->
                            {
                                Id = a.GetProperty("id").GetInt32()
                                Content = a.GetProperty("content").GetString()
                                Special = a.GetProperty("special").GetBoolean()
                            }
                    ]

                Expect.equal want.Length 26 "26 added tokens"
                Expect.equal t.AddedTokens want "ids, contents and special flags"

                for a in want do
                    Expect.equal (t.IdOf a.Content) (Some a.Id) a.Content
                    Expect.equal (t.Encode a.Content) [| a.Id |] a.Content
                    Expect.equal (t.Decode [| a.Id |]) a.Content a.Content
                    Expect.equal (t.Encode("x" + a.Content + "x")) [| 87; a.Id; 87 |] ("inside text: " + a.Content)

            testCase "no BOS: Encode adds no id that the text does not hold"
            <| fun () ->
                let t = tokenizerOrSkip "no-bos"
                Expect.equal (t.Encode "") [||] "the empty text"

                for e in jsonLines (corpusPath ()) do
                    if text e "kind" = "encode" then
                        let s = text e "text"
                        let ids = t.Encode s

                        if ids.Length > 0 && (ids[0] = 151643 || ids[0] = 151644) then
                            Expect.isTrue (s.StartsWith(t.TokenOf ids[0] |> Option.get, StringComparison.Ordinal)) s

            testCase "151,669 real ids, 151,387 merges, and every id maps to its token and back"
            <| fun () ->
                let t = tokenizerOrSkip "vocabulary"
                Expect.equal t.VocabSize QwenTokenizer.RealIds "real ids"
                Expect.equal t.VocabSize 151_669 "TokenizerApi"
                Expect.equal t.MergeCount 151_387 "merges"
                Expect.equal QwenTokenizer.ModelRows 151_936 "model rows"
                Expect.equal t.Sha256 QwenTokenizer.PinnedSha256 "the pinned file"

                for id in 0 .. t.VocabSize - 1 do
                    match t.TokenOf id with
                    | Some token -> Expect.equal (t.IdOf token) (Some id) token
                    | None -> failtestf "id %d has no token" id

                Expect.equal (t.TokenOf t.VocabSize) None "151,669 has no token"
        ]
