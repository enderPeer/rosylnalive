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

/// The corpus digest of docs/contracts/golden-set.md ("The tokenizer set").
let CorpusSha256 =
    "35ff0c43a1bc4325588b1a22953f861784e4ca76e0f3f863a5f70ff9ceb16c66"

type private Outcome =
    {
        Passed: bool
        Detail: string
        Tokens: int
    }

let private window (ids: int[]) (at: int) =
    ids
    |> Array.skip (max 0 (at - 2))
    |> Array.truncate 6
    |> Array.map string
    |> String.concat " "

let private compareIds (want: int[]) (got: int[]) =
    match firstDiff want got with
    | -1 ->
        {
            Passed = true
            Detail = ""
            Tokens = want.Length
        }
    | i ->
        {
            Passed = false
            Detail =
                sprintf
                    "ids differ at %d (expected %d ids, got %d): expected ... %s ..., got ... %s ..."
                    i
                    want.Length
                    got.Length
                    (window want i)
                    (window got i)
            Tokens = want.Length
        }

/// Runs one corpus or probe line (GoldenSet, "The tokenizer corpus").
let private runCase (t: QwenTokenizer) (e: JsonElement) : Outcome =
    let fail detail =
        {
            Passed = false
            Detail = detail
            Tokens = 0
        }

    try
        match text e "kind" with
        | "encode" -> compareIds (ints e "ids") (t.Encode(text e "text"))
        | "encode_bytes" ->
            let bytes = Convert.FromHexString(text e "bytes")

            if Encoding.UTF8.GetString bytes <> text e "text" then
                fail "the bytes do not decode to the case's text (errors=replace)"
            else
                compareIds (ints e "ids") (t.EncodeBytes bytes)
        | "decode" ->
            let ids = ints e "ids"
            let decoded = t.Decode ids

            let bytesOk =
                match e.TryGetProperty "bytes" with
                | true, b -> t.DecodeBytes ids = Convert.FromHexString(b.GetString())
                | _ -> true

            if not bytesOk then
                fail "DecodeBytes differs from the case's bytes"
            elif decoded <> text e "text" then
                fail (sprintf "Decode gave %s, expected %s" (describe decoded) (describe (text e "text")))
            else
                {
                    Passed = true
                    Detail = ""
                    Tokens = ids.Length
                }
        | "encode_reject" ->
            let s = ofUtf16Hex (text e "utf16")

            try
                t.Encode s |> ignore
                fail "Encode accepted a text with a lone surrogate"
            with :? ArgumentException ->
                {
                    Passed = true
                    Detail = ""
                    Tokens = 0
                }
        | other -> fail (sprintf "unknown kind %s" other)
    with ex ->
        fail (sprintf "%s: %s" (ex.GetType().Name) ex.Message)

/// Runs every line of a JSON-lines file, emitting one tokenizer.case check per line, and fails with the first
/// differences.
let private runFile (t: QwenTokenizer) (source: string) (lines: JsonElement list) =
    use _ = TokenizerRun.install ()

    use _ =
        Telemetry.scope [ Attr.str TelemetryKeys.Component "tokenizer"; Attr.str "source" source ]

    let failures = ResizeArray<string>()

    lines
    |> List.iteri (fun i e ->
        let o = runCase t e

        let id =
            match e.TryGetProperty "id" with
            | true, v -> v.GetString()
            | _ -> sprintf "%s-%04d" source (i + 1)

        let cat =
            match e.TryGetProperty "cat" with
            | true, v -> v.GetString()
            | _ -> "probe-" + text e "kind"

        let attrs =
            [
                Attr.str TelemetryKeys.Case id
                Attr.str "cat" cat
                Attr.str "kind" (text e "kind")
                Attr.int "tokens" o.Tokens
            ]

        Telemetry.check
            TelemetryNames.TokenizerCase
            o.Passed
            (if o.Passed then
                 attrs
             else
                 attrs @ [ Attr.str "detail" o.Detail ])

        if not o.Passed then
            failures.Add(id + ": " + o.Detail))

    Telemetry.event
        "tokenizer.corpus"
        [
            Attr.int "cases" lines.Length
            Attr.int "passed" (lines.Length - failures.Count)
            Attr.int "failed" failures.Count
        ]

    if failures.Count > 0 then
        failtestf
            "%s: %d of %d cases differ from the oracle:\n%s"
            source
            failures.Count
            lines.Length
            (failures |> Seq.truncate 25 |> String.concat "\n")

/// Vectors E1 to E11 (Encode) and D1 to D6 (Decode) of docs/contracts/tokenizer.md, from the oracle.
let encodeVectors: (string * int list) list =
    [
        "Hello world", [ 9707; 1879 ]
        "using System;", [ 970; 739; 26 ]
        "<|im_start|>user\nHi<|im_end|>", [ 151644; 872; 198; 13048; 151645 ]
        "é", [ 963 ]
        Char.ConvertFromUtf32 0x1F600, [ 141334 ]
        "a̴߽", [ 64; 155; 121; 136; 112 ]
        "x￾́", [ 87; 5564; 122; 53839 ]
        "public static int Add(int a, int b) => a + b;",
        [ 888; 1099; 526; 2691; 1548; 264; 11; 526; 293; 8; 589; 264; 488; 293; 26 ]
        "", []
        " ", [ 220 ]
        "中文", [ 104811 ]
    ]

let decodeVectors: (int list * string) list =
    [
        [ 151643 ], "<|endoftext|>"
        [ 9707; 1879 ], "Hello world"
        [ 160; 151700; 116; 255 ], "中"
        [ 5691 ], "�"
        [ 151669 ], ""
        [], ""
    ]

let tests =
    testList
        "corpus"
        [
            testCase "O1's corpus is exact (goldens/tokenizer/g1, 5,597 cases)"
            <| fun () ->
                let path = corpusPath ()
                let sha = ContentId.hex (Hash.sha256 (File.ReadAllBytes path))
                Expect.equal sha CorpusSha256 "cases.jsonl is the GoldenSet v1 corpus"
                let t = tokenizerOrSkip "corpus"
                let lines = jsonLines path
                Expect.equal lines.Length 5597 "cases"
                runFile t "corpus" lines

            testCase "the probe's random texts and id lists are exact (oracle/encode-cases.jsonl)"
            <| fun () ->
                let t = tokenizerOrSkip "encode-cases"
                runFile t "probe" (jsonLines (oraclePath "encode-cases.jsonl"))

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
