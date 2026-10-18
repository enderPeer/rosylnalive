/// The whole pipeline on a toy tokenizer.json with Qwen's configuration and a 263-token vocabulary, so Encode, Decode,
/// the loader's checks, telemetry and thread safety are tested in CI without the 11 MB file.
module Lrm.Tokenizer.Tests.ToyTests

open System
open System.IO
open System.Text
open System.Text.Json.Nodes
open System.Threading.Tasks
open Expecto
open Lrm.Core
open Lrm.Tokenizer
open Lrm.Tokenizer.Tests.Common

/// The byte-level string of a text.
let private bl (s: string) =
    ByteLevel.encode (Encoding.UTF8.GetBytes s)

/// Merges of the toy model, in rank order: (left, right) in byte-level form.
let private toyMerges =
    [
        bl " ", bl "t" // 256 " t"      rank 0
        bl "h", bl "e" // 257 "he"      rank 1
        bl " t", bl "he" // 258 " the"  rank 2
        bl "a", bl "a" // 259 "aa"      rank 3
        bl "b", bl "c" // 260 "bc"      rank 4
        bl "a", bl "b" // 261 "ab"      rank 5
        bl "aa", bl "aa" // 262 "aaaa"  rank 6
    ]

/// Added tokens of the toy model: ids 263, 264, 265.
let private toyAdded = [ "<|a|>", true; "<|ab|>", true; "<x>", false ]

/// A toy tokenizer.json; `edit` changes the JSON tree before it is written.
let toyJson (edit: JsonObject -> unit) : byte[] =
    let vocab = JsonObject()

    for b in 0..255 do
        vocab[string (ByteLevel.charOf (byte b))] <- JsonValue.Create b

    toyMerges
    |> List.iteri (fun i (l, r) -> vocab[l + r] <- JsonValue.Create(256 + i))

    let merges = JsonArray()

    for l, r in toyMerges do
        merges.Add(JsonArray(JsonValue.Create l, JsonValue.Create r))

    let added = JsonArray()

    toyAdded
    |> List.iteri (fun i (content, special) ->
        let t = JsonObject()
        t["id"] <- JsonValue.Create(256 + toyMerges.Length + i)
        t["content"] <- JsonValue.Create content
        t["single_word"] <- JsonValue.Create false
        t["lstrip"] <- JsonValue.Create false
        t["rstrip"] <- JsonValue.Create false
        t["normalized"] <- JsonValue.Create false
        t["special"] <- JsonValue.Create special
        added.Add t)

    let byteLevel () =
        JsonObject(
            [
                Collections.Generic.KeyValuePair("type", JsonValue.Create "ByteLevel" :> JsonNode)
                Collections.Generic.KeyValuePair("add_prefix_space", JsonValue.Create false)
                Collections.Generic.KeyValuePair("trim_offsets", JsonValue.Create false)
                Collections.Generic.KeyValuePair("use_regex", JsonValue.Create false)
            ]
        )

    let root =
        JsonNode
            .Parse(
                """{"version":"1.0","truncation":null,"padding":null,"normalizer":{"type":"NFC"},
                "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":""},
                "behavior":"Isolated","invert":false}]},
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"",
                "end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false}}"""
            )
            .AsObject()

    root["pre_tokenizer"].["pretokenizers"].[0].["pattern"].["Regex"] <- JsonValue.Create PreTokenizer.SplitPattern
    root["pre_tokenizer"].["pretokenizers"].AsArray().Add(byteLevel ())
    root["post_processor"] <- byteLevel ()
    root["decoder"] <- byteLevel ()
    root["added_tokens"] <- added
    root["model"].["vocab"] <- vocab
    root["model"].["merges"] <- merges
    edit root
    Encoding.UTF8.GetBytes(root.ToJsonString())

let private toy = lazy (QwenTokenizer.parse (toyJson ignore))

let private idOf (s: string) = (toy.Force().IdOf(bl s)).Value

/// The id of each byte of `s`, as single-byte tokens.
let private byteIds (s: string) =
    [|
        for b in Encoding.UTF8.GetBytes s -> (toy.Force().IdOf(string (ByteLevel.charOf b))).Value
    |]

let private encodeTests =
    testList
        "encode"
        [
            testCase "BPE merges the lowest rank first, and the leftmost pair among equal ranks"
            <| fun () ->
                let t = toy.Force()
                Expect.equal (t.Encode " the") [| idOf " the" |] "' the' merges through ' t' and 'he'"
                Expect.equal (t.Encode "abc") [| idOf "a"; idOf "bc" |] "'bc' (rank 4) before 'ab' (rank 5)"
                Expect.equal (t.Encode "aaa") [| idOf "aa"; idOf "a" |] "the leftmost 'aa' first"
                Expect.equal (t.Encode "aaaa") [| idOf "aaaa" |] "'aa' 'aa' then 'aaaa'"
                Expect.equal (t.Encode "aaaaa") [| idOf "aaaa"; idOf "a" |] "five a"
                Expect.equal (t.Encode "") [||] "the empty text"

            testCase "added tokens match first, leftmost-longest, on the raw text"
            <| fun () ->
                let t = toy.Force()
                Expect.equal (t.Encode "<|ab|>") [| 264 |] "the longer of <|a|> and <|ab|>"
                Expect.equal (t.Encode "<|a|><|ab|>") [| 263; 264 |] "two in a row"
                Expect.equal (t.Encode "x<x>y") (Array.concat [ byteIds "x"; [| 265 |]; byteIds "y" ]) "inside a word"
                Expect.equal (t.Encode "<|a") (byteIds "<|a") "an unfinished token is text"
                Expect.equal (t.Encode "<<x>") (Array.append (byteIds "<") [| 265 |]) "the match starts late"

            testCase "NFC runs on each stretch between added tokens"
            <| fun () ->
                let t = toy.Force()
                Expect.equal (t.Encode "é") (byteIds "é") "e + U+0301 composes to U+00E9"

                Expect.equal
                    (t.Encode "e<x>́")
                    (Array.concat [ byteIds "e"; [| 265 |]; byteIds "́" ])
                    "not across a token"

            testCase "a lone surrogate is rejected, a pair is encoded"
            <| fun () ->
                let t = toy.Force()

                for h in [ "d800"; "0061dc00"; "0061d800d8000062"; "dfff0061" ] do
                    Expect.throwsT<ArgumentException> (fun () -> t.Encode(ofUtf16Hex h) |> ignore) h

                Expect.equal (t.Encode(ofUtf16Hex "d83dde00")) (byteIds (ofUtf16Hex "d83dde00")) "U+1F600"

            testCase "EncodeBytes replaces each maximal invalid subpart with one U+FFFD"
            <| fun () ->
                let t = toy.Force()
                Expect.equal (t.EncodeBytes [| 0x80uy |]) (byteIds "�") "a lone continuation byte"
                Expect.equal (t.EncodeBytes [| 0xE4uy; 0xB8uy |]) (byteIds "�") "a truncated sequence"

                Expect.equal
                    (t.EncodeBytes [| 0x61uy; 0xF0uy; 0x80uy; 0x62uy |])
                    (byteIds "a��b")
                    "F0 80 is two subparts"
        ]

let private decodeTests =
    testList
        "decode"
        [
            testCase "bytes of tokens, added tokens and rows with no token"
            <| fun () ->
                let t = toy.Force()
                Expect.equal (t.DecodeBytes [| idOf " the" |]) (Encoding.UTF8.GetBytes " the") "a merged token"
                Expect.equal (t.Decode [| 263; 265 |]) "<|a|><x>" "added tokens"
                Expect.equal (t.DecodeBytes [||]) [||] "the empty list"
                Expect.equal (t.DecodeBytes [| t.VocabSize; 151_935 |]) [||] "rows with no token add nothing"

                let zhong = byteIds "中"

                Expect.equal
                    (t.Decode [| zhong[0]; 151_700; zhong[1]; zhong[2] |])
                    "中"
                    "even between the bytes of one character"

            testCase "Decode is errors=replace"
            <| fun () ->
                let t = toy.Force()
                let zhong = byteIds "中"
                Expect.equal (t.Decode [| zhong[0] |]) "�" "a truncated character"
                Expect.equal (t.Decode [| zhong[0]; zhong[1]; idOf "a" |]) "�a" "one U+FFFD per maximal subpart"

            testCase "ids outside the model raise"
            <| fun () ->
                let t = toy.Force()

                for id in [ -1; QwenTokenizer.ModelRows; Int32.MaxValue ] do
                    Expect.throwsT<ArgumentOutOfRangeException> (fun () -> t.DecodeBytes [| id |] |> ignore) (string id)

            testCase "IdOf and TokenOf"
            <| fun () ->
                let t = toy.Force()
                Expect.equal (t.IdOf "<|ab|>") (Some 264) "added"
                Expect.equal (t.IdOf(bl " the")) (Some 258) "vocabulary, byte-level form"
                Expect.equal (t.IdOf " the") None "not in byte-level form"
                Expect.equal (t.TokenOf 258) (Some(bl " the")) "token of an id"
                Expect.equal (t.TokenOf 265) (Some "<x>") "added"
                Expect.equal (t.TokenOf 266) None "no token"
                Expect.equal t.VocabSize 266 "263 vocabulary tokens and 3 added"
                Expect.equal t.MergeCount 7 "merges"
                Expect.equal (t.AddedTokens |> List.map _.Special) [ true; true; false ] "special flags"
        ]

/// Each edit must make the loader refuse the file.
let private refusals: (string * (JsonObject -> unit)) list =
    let pre (root: JsonObject) = root["pre_tokenizer"].["pretokenizers"]

    [
        "another regex", (fun r -> (pre r).[0].["pattern"].["Regex"] <- JsonValue.Create @"\p{L}+|\s+")
        "behaviour Removed", (fun r -> (pre r).[0].["behavior"] <- JsonValue.Create "Removed")
        "ByteLevel with use_regex", (fun r -> (pre r).[1].["use_regex"] <- JsonValue.Create true)
        "ByteLevel without use_regex", (fun r -> (pre r).[1].AsObject().Remove "use_regex" |> ignore)
        "add_prefix_space", (fun r -> (pre r).[1].["add_prefix_space"] <- JsonValue.Create true)
        "NFKC", (fun r -> r["normalizer"].["type"] <- JsonValue.Create "NFKC")
        "dropout", (fun r -> r["model"].["dropout"] <- JsonValue.Create 0.1)
        "byte fallback", (fun r -> r["model"].["byte_fallback"] <- JsonValue.Create true)
        "ignore_merges", (fun r -> r["model"].["ignore_merges"] <- JsonValue.Create true)
        "an unknown token", (fun r -> r["model"].["unk_token"] <- JsonValue.Create "<unk>")
        "a normalized added token", (fun r -> r["added_tokens"].[0].["normalized"] <- JsonValue.Create true)
        "an lstrip added token", (fun r -> r["added_tokens"].[1].["lstrip"] <- JsonValue.Create true)
        "an added token id gap", (fun r -> r["added_tokens"].[2].["id"] <- JsonValue.Create 300)
        "a merge of an unknown token",
        (fun r -> r["model"].["merges"].AsArray().Add(JsonArray(JsonValue.Create "zz", JsonValue.Create "a")))
        "a merge whose result is missing",
        (fun r -> r["model"].["merges"].AsArray().Add(JsonArray(JsonValue.Create "b", JsonValue.Create "b")))
        "a vocabulary id gap", (fun r -> r["model"].["vocab"].[bl " the"] <- JsonValue.Create 999)
        "a decoder that is not ByteLevel", (fun r -> r["decoder"].["type"] <- JsonValue.Create "WordPiece")
        "a model that is not BPE", (fun r -> r["model"].["type"] <- JsonValue.Create "Unigram")
    ]

let private loaderTests =
    testList
        "loader"
        [
            testCase "the toy file parses"
            <| fun () -> Expect.equal (toy.Force().VocabSize) 266 "parsed"

            for name, edit in refusals do
                testCase (sprintf "refuses %s" name)
                <| fun () ->
                    Expect.throwsT<InvalidDataException>
                        (fun () -> QwenTokenizer.parse (toyJson edit) |> ignore)
                        "an unsupported tokenizer.json"

            testCase "load refuses a file whose SHA-256 is not the pin"
            <| fun () ->
                let path =
                    Path.Combine(Path.GetTempPath(), sprintf "lrm-m1-toy-%s.json" (Guid.NewGuid().ToString "N"))

                File.WriteAllBytes(path, toyJson ignore)

                try
                    let e = Expect.throwsC (fun () -> QwenTokenizer.load path |> ignore) (fun e -> e)

                    Expect.isTrue (e :? InvalidDataException) "InvalidDataException"
                    Expect.stringContains e.Message QwenTokenizer.PinnedSha256 "names the pin"
                finally
                    File.Delete path

            testCase "the pin is the contract's and toolchain.json's"
            <| fun () ->
                Expect.equal
                    QwenTokenizer.PinnedSha256
                    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
                    "TokenizerApi vector"

                use doc =
                    Text.Json.JsonDocument.Parse(File.ReadAllText(repoPath "tools/bootstrap/toolchain.json"))

                let entry = doc.RootElement.GetProperty("tokenizer")

                Expect.equal
                    QwenTokenizer.PinnedSha256
                    (entry.GetProperty("sha256").GetString())
                    "toolchain.json sha256"

                Expect.equal
                    QwenTokenizer.PinnedPath
                    (entry.GetProperty("install").GetString())
                    "toolchain.json install"

                Expect.isTrue
                    (Lrm.Testing.Env.samePath
                        (Path.GetDirectoryName(QwenTokenizer.defaultPath ()))
                        (Path.Combine(LrmPaths.dataRoot (), "models", "Qwen3-0.6B")))
                    "defaultPath is under LRM_DATA"
        ]

let private behaviourTests =
    testList
        "behaviour"
        [
            testCase "telemetry records tokenizer.encode and tokenizer.tokens and never changes ids"
            <| fun () ->
                let t = toy.Force()
                let texts = [ " the aaa"; "<|ab|>x"; "é bc"; "" ]
                let plain = [ for s in texts -> t.Encode s ]
                let sink = InMemorySink()

                let recorded =
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    [ for s in texts -> t.Encode s ]

                Expect.equal recorded plain "same ids with and without telemetry"
                let spans = sink.Records |> List.filter (fun r -> r.Name = "tokenizer.encode")
                Expect.equal spans.Length texts.Length "one span per Encode"

                let tokens =
                    sink.Records
                    |> List.filter (fun r -> r.Name = "tokenizer.tokens")
                    |> List.sumBy _.Value

                Expect.equal tokens (float (plain |> List.sumBy Array.length)) "metered tokens"

            testCase "Encode is thread-safe: 8 threads give the sequential ids"
            <| fun () ->
                let t = QwenTokenizer.parse (toyJson ignore)

                let texts =
                    [| for e in jsonLines (oraclePath "split-cases.jsonl") -> text e "text" |]

                let sequential = texts |> Array.map t.Encode
                t.ClearCache()
                let parallel' = Array.zeroCreate<int[]> texts.Length

                Parallel.For(
                    0,
                    texts.Length,
                    ParallelOptions(MaxDegreeOfParallelism = 8),
                    fun i -> parallel'[i] <- t.Encode texts[i]
                )
                |> ignore

                Expect.equal parallel' sequential "parallel = sequential"
                Expect.isGreaterThan t.CachedPieces 0 "the cache filled"

            testCase "decoding the encoding gives the NFC text back"
            <| fun () ->
                let t = toy.Force()

                for e in jsonLines (oraclePath "split-cases.jsonl") do
                    let s = text e "text"
                    Expect.equal (t.Decode(t.Encode s)) (Nfc.normalize s) (describe s)
        ]

let tests = testList "toy" [ encodeTests; decodeTests; loaderTests; behaviourTests ]
