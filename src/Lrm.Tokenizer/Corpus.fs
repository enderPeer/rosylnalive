namespace Lrm.Tokenizer

open System
open System.IO
open System.Text
open System.Text.Json
open Lrm.Core

/// The outcome of one corpus line (TokenizerApi contract, "The corpus check").
type CorpusCase =
    {
        /// The line's `id`, or `<source>-<line number>` for a line without one (the probe's files).
        Id: string
        /// The line's `cat`, or `probe-<kind>` for a line without one.
        Category: string
        /// `encode`, `encode_bytes`, `decode` or `encode_reject`.
        Kind: string
        Passed: bool
        /// The number of expected ids (0 for encode_reject).
        Tokens: int
        /// Why the case failed (for ids, the first differing index and the ids around it); empty when it passed.
        Detail: string
    }

/// The outcome of one corpus file.
type CorpusResult =
    {
        /// `corpus` for O1's corpus, or the caller's name for another file (the tests use `probe`).
        Source: string
        Path: string
        /// The file's SHA-256, lowercase hex.
        Sha256: string
        Cases: int
        Passed: int
        /// Every failing case, in file order.
        Failures: CorpusCase list
    }

    /// At least one case ran, and every case passed.
    member r.Exact = r.Cases > 0 && r.Failures.IsEmpty

/// O1's tokenizer corpus as a check (TokenizerApi contract, "The corpus check"; GoldenSet, "The tokenizer corpus").
/// The cpu tier, `lrm gate1` (I3) and the Gate 1 suite (I1) all run it through here, so the rules per kind and the
/// digest check exist once.
[<RequireQualifiedAccess>]
module Corpus =
    /// The SHA-256 of O1's corpus, goldens/tokenizer/g1/cases.jsonl (GoldenSet v1, "The tokenizer set").
    [<Literal>]
    let PinnedSha256 =
        "35ff0c43a1bc4325588b1a22953f861784e4ca76e0f3f863a5f70ff9ceb16c66"

    /// The number of cases in it.
    [<Literal>]
    let PinnedCases = 5_597

    /// Its path under the repo root.
    [<Literal>]
    let RepoPath = "goldens/tokenizer/g1/cases.jsonl"

    let private property (e: JsonElement) (name: string) =
        match e.TryGetProperty name with
        | true, v -> v
        | _ -> raise (InvalidDataException(sprintf "the case has no '%s'" name))

    let private str (e: JsonElement) (name: string) = (property e name).GetString()

    let private ints (e: JsonElement) (name: string) =
        [| for x in (property e name).EnumerateArray() -> x.GetInt32() |]

    let private optionalStr (e: JsonElement) (name: string) (fallback: string) =
        match e.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.String -> v.GetString()
        | _ -> fallback

    /// The text of UTF-16 code units in big-endian hex, 4 digits per unit (the corpus's `utf16` field). Lone
    /// surrogates are built here at run time, so no file holds one.
    let private ofUtf16Hex (h: string) =
        String(Array.init (h.Length / 4) (fun i -> char (Convert.ToUInt16(h.Substring(4 * i, 4), 16))))

    /// Code points as text ("U+0041 U+1F600").
    let private describe (s: string) =
        s.EnumerateRunes()
        |> Seq.map (fun r -> sprintf "U+%04X" r.Value)
        |> String.concat " "

    let private window (ids: int[]) (at: int) =
        ids
        |> Array.skip (min ids.Length (max 0 (at - 2)))
        |> Array.truncate 6
        |> Array.map string
        |> String.concat " "

    /// None when the ids are equal, else the first differing index and the ids around it.
    let private compareIds (want: int[]) (got: int[]) =
        let n = min want.Length got.Length
        let mutable i = 0

        while i < n && want[i] = got[i] do
            i <- i + 1

        if i = n && want.Length = got.Length then
            None
        else
            Some(
                sprintf
                    "ids differ at %d (expected %d ids, got %d): expected ... %s ..., got ... %s ..."
                    i
                    want.Length
                    got.Length
                    (window want i)
                    (window got i)
            )

    /// Runs one line by its kind (GoldenSet, "The tokenizer corpus"); an exception fails the case.
    let private runCase (t: QwenTokenizer) (fallbackId: string) (e: JsonElement) : CorpusCase =
        let kind = optionalStr e "kind" ""

        let case =
            {
                Id = optionalStr e "id" fallbackId
                Category = optionalStr e "cat" ("probe-" + kind)
                Kind = kind
                Passed = false
                Tokens = 0
                Detail = ""
            }

        let ids (want: int[]) (got: int[]) =
            match compareIds want got with
            | None ->
                { case with
                    Passed = true
                    Tokens = want.Length
                }
            | Some detail ->
                { case with
                    Tokens = want.Length
                    Detail = detail
                }

        let fail detail = { case with Detail = detail }

        try
            match kind with
            | "encode" -> ids (ints e "ids") (t.Encode(str e "text"))
            | "encode_bytes" ->
                let bytes = Convert.FromHexString(str e "bytes")

                if Encoding.UTF8.GetString bytes <> str e "text" then
                    fail "the bytes do not decode to the case's text (errors=replace)"
                else
                    ids (ints e "ids") (t.EncodeBytes bytes)
            | "decode" ->
                let want = ints e "ids"
                let decoded = t.Decode want

                let bytesOk =
                    match e.TryGetProperty "bytes" with
                    | true, b -> t.DecodeBytes want = Convert.FromHexString(b.GetString())
                    | _ -> true

                if not bytesOk then
                    fail "DecodeBytes differs from the case's bytes"
                elif decoded <> str e "text" then
                    fail (sprintf "Decode gave %s, expected %s" (describe decoded) (describe (str e "text")))
                else
                    { case with
                        Passed = true
                        Tokens = want.Length
                    }
            | "encode_reject" ->
                let s = ofUtf16Hex (str e "utf16")

                try
                    t.Encode s |> ignore
                    fail "Encode accepted a text with a lone surrogate"
                with :? ArgumentException ->
                    { case with Passed = true }
            | other -> fail (sprintf "unknown kind '%s'" other)
        with ex ->
            fail (sprintf "%s: %s" (ex.GetType().Name) ex.Message)

    let private runBytes (t: QwenTokenizer) (source: string) (path: string) (bytes: byte[]) (sha: string) =
        use _ =
            Telemetry.scope [ Attr.str TelemetryKeys.Component "tokenizer"; Attr.str "source" source ]

        let failures = ResizeArray<CorpusCase>()
        let mutable cases = 0
        use reader = new StringReader(Encoding.UTF8.GetString bytes)
        let mutable line = reader.ReadLine()

        while not (isNull line) do
            if line.Length > 0 then
                cases <- cases + 1

                let c =
                    use doc =
                        try
                            JsonDocument.Parse line
                        with :? JsonException as ex ->
                            raise (
                                InvalidDataException(sprintf "%s: case %d is not JSON: %s" path cases ex.Message, ex)
                            )

                    runCase t (sprintf "%s-%04d" source cases) doc.RootElement

                let attrs =
                    [
                        Attr.str TelemetryKeys.Case c.Id
                        Attr.str "cat" c.Category
                        Attr.str "kind" c.Kind
                        Attr.int "tokens" c.Tokens
                    ]

                Telemetry.check
                    TelemetryNames.TokenizerCase
                    c.Passed
                    (if c.Passed then
                         attrs
                     else
                         attrs @ [ Attr.str "detail" c.Detail ])

                if not c.Passed then
                    failures.Add c

            line <- reader.ReadLine()

        Telemetry.event
            "tokenizer.corpus"
            [
                Attr.int "cases" cases
                Attr.int "passed" (cases - failures.Count)
                Attr.int "failed" failures.Count
                Attr.str "sha256" sha
            ]

        {
            Source = source
            Path = path
            Sha256 = sha
            Cases = cases
            Passed = cases - failures.Count
            Failures = List.ofSeq failures
        }

    /// Runs every line of the JSON-lines file at `path` in order, and emits through the ambient facade one
    /// tokenizer.case check per line and one tokenizer.corpus event, all with `source`. It checks no digest: `verify`
    /// is the check of O1's corpus. Raises InvalidDataException for a line that is not JSON.
    let run (tokenizer: QwenTokenizer) (source: string) (path: string) : CorpusResult =
        let bytes = File.ReadAllBytes path
        runBytes tokenizer source path bytes (ContentId.hex (Hash.sha256 bytes))

    /// O1's corpus at `path` (RepoPath under the repo root): its SHA-256 must be PinnedSha256, else
    /// InvalidDataException naming both hashes; then every case runs as in `run` with source `corpus`. The corpus is
    /// exact when the result's Exact is true and its Cases is PinnedCases.
    let verify (tokenizer: QwenTokenizer) (path: string) : CorpusResult =
        let bytes = File.ReadAllBytes path
        let sha = ContentId.hex (Hash.sha256 bytes)

        if sha <> PinnedSha256 then
            raise (
                InvalidDataException(
                    sprintf "%s has SHA-256 %s, but O1's tokenizer corpus (GoldenSet v1) is %s" path sha PinnedSha256
                )
            )

        runBytes tokenizer "corpus" path bytes sha
