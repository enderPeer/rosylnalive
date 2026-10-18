namespace Lrm.Tokenizer

open System
open System.Collections.Generic
open System.IO
open System.Reflection
open System.Text
open Lrm.Core

type internal TokenizerMarker = class end

/// The Qwen3 byte-level BPE tokenizer (TokenizerApi contract, docs/contracts/tokenizer.md). Exact id parity with HF
/// tokenizers 0.23.3 on the pinned tokenizer.json is the specification. Instances are immutable apart from a piece
/// cache that never changes a result, and every member is safe to call from many threads at once.
///
/// Encode runs, in order: added tokens on the raw text (leftmost-longest), then for each stretch between them the
/// oracle's NFC (Nfc), the Split pre-tokenizer (PreTokenizer) and byte-level BPE (BpeModel). It adds no BOS or EOS.
[<Sealed>]
type QwenTokenizer internal (parsed: TokenizerJson.Parsed, sha256: string) =
    let bpe = BpeModel(parsed.TokenBytes, parsed.ByteIds, parsed.Merges, 1 <<< 16)
    let trie = AddedTokenTrie parsed.Added
    let addedById = Dictionary<int, AddedToken>()
    let addedByContent = Dictionary<string, int>(StringComparer.Ordinal)
    let tokenOf = Array.zeroCreate<string> parsed.TokenBytes.Length

    do
        for t in parsed.Added do
            addedById[t.Id] <- t
            addedByContent[t.Content] <- t.Id

        for KeyValue(token, id) in parsed.Vocab do
            tokenOf[id] <- token

    let vocabSize = parsed.TokenBytes.Length + parsed.Added.Length

    /// Raises ArgumentException at the first lone surrogate (the oracle refuses such text: Python cannot pass it to
    /// Rust as a string).
    static let checkScalars (text: string) =
        let mutable i = 0

        while i < text.Length do
            let c = text[i]

            if Char.IsHighSurrogate c && i + 1 < text.Length && Char.IsLowSurrogate text[i + 1] then
                i <- i + 2
            elif Char.IsSurrogate c then
                invalidArg
                    "text"
                    (sprintf
                        "the text holds a lone surrogate U+%04X at index %d; only Unicode scalar values can be encoded"
                        (int c)
                        i)
            else
                i <- i + 1

    let encodeInto (text: string) (ids: ResizeArray<int>) =
        let mutable pieces = 0
        let mutable added = 0

        trie.Split(
            text,
            (fun id ->
                ids.Add id
                added <- added + 1),
            fun start length ->
                let segment = Nfc.normalize (text.Substring(start, length))

                PreTokenizer.split segment (fun s l ->
                    pieces <- pieces + 1
                    bpe.EncodePiece(segment, s, l, ids))
        )

        struct (pieces, added)

    /// The number of real token ids: the vocabulary and the added tokens (151,669 for Qwen3).
    member _.VocabSize = vocabSize

    /// The added tokens, by id.
    member _.AddedTokens: AddedToken list = parsed.Added

    /// The number of merges (151,387 for Qwen3).
    member _.MergeCount = parsed.Merges.Count

    /// The SHA-256 of the tokenizer.json this tokenizer was read from, lowercase hex.
    member _.Sha256 = sha256

    /// The token ids of `text`. Raises ArgumentException when `text` holds a lone surrogate.
    member _.Encode(text: string) : int[] =
        if isNull text then
            nullArg "text"

        checkScalars text
        let t = Telemetry.current ()

        if t.Enabled then
            use span =
                t.Span("tokenizer.encode", [ Attr.str TelemetryKeys.Component "tokenizer" ])

            let ids = ResizeArray<int>()
            let struct (pieces, added) = encodeInto text ids
            span.Add(Attr.int "chars" text.Length)
            span.Add(Attr.int "tokens" ids.Count)
            span.Add(Attr.int "pieces" pieces)
            span.Add(Attr.int "added" added)

            t.Meter(
                "tokenizer.tokens",
                float ids.Count,
                TelemetryUnits.Tokens,
                [ Attr.str TelemetryKeys.Component "tokenizer" ]
            )

            ids.ToArray()
        else
            let ids = ResizeArray<int>(text.Length / 3 + 4)
            encodeInto text ids |> ignore
            ids.ToArray()

    /// The token ids of UTF-8 `bytes`, decoded with one U+FFFD per maximal invalid subpart (errors=replace, as
    /// Python's bytes.decode and .NET's Encoding.UTF8 both do).
    member this.EncodeBytes(bytes: byte[]) : int[] =
        if isNull bytes then
            nullArg "bytes"

        this.Encode(Encoding.UTF8.GetString bytes)

    /// The bytes of `ids`: each vocabulary token's bytes, each added token's content as the ByteLevel decoder gives it,
    /// and nothing for ids from VocabSize to ModelRows - 1 (lm_head rows with no token). Raises
    /// ArgumentOutOfRangeException for an id below 0 or at ModelRows and above.
    member _.DecodeBytes(ids: IReadOnlyList<int>) : byte[] =
        if isNull ids then
            nullArg "ids"

        // Two passes: check every id and size the output, then copy.
        let parts = Array.zeroCreate<byte[]> ids.Count
        let mutable total = 0

        for i in 0 .. ids.Count - 1 do
            let id = ids[i]

            let part =
                if id >= 0 && id < bpe.Count then
                    bpe.BytesOf id
                else
                    match addedById.TryGetValue id with
                    | true, t -> ByteLevel.decoderBytes t.Content
                    | _ when id >= vocabSize && id < QwenTokenizer.ModelRows -> Array.Empty<byte>()
                    | _ ->
                        raise (
                            ArgumentOutOfRangeException(
                                "ids",
                                id,
                                sprintf
                                    "token id %d at index %d is outside the model's %d rows"
                                    id
                                    i
                                    QwenTokenizer.ModelRows
                            )
                        )

            parts[i] <- part
            total <- total + part.Length

        let bytes = Array.zeroCreate<byte> total
        let mutable at = 0

        for part in parts do
            Buffer.BlockCopy(part, 0, bytes, at, part.Length)
            at <- at + part.Length

        let t = Telemetry.current ()

        if t.Enabled then
            t.Event(
                "tokenizer.decode",
                [
                    Attr.str TelemetryKeys.Component "tokenizer"
                    Attr.int "ids" ids.Count
                    Attr.int "bytes" bytes.Length
                ]
            )

        bytes

    /// The text of `ids`: DecodeBytes decoded as UTF-8 with one U+FFFD per maximal invalid subpart (errors=replace).
    member this.Decode(ids: IReadOnlyList<int>) : string =
        Encoding.UTF8.GetString(this.DecodeBytes ids)

    /// The id of a token: an added token's content, or a vocabulary token in its byte-level form (for example
    /// "Ġthe" for " the").
    member _.IdOf(token: string) : int option =
        match addedByContent.TryGetValue token with
        | true, id -> Some id
        | _ ->
            match parsed.Vocab.TryGetValue token with
            | true, id -> Some id
            | _ -> None

    /// The token of an id: the byte-level vocabulary string or the added token's content; None for other ids.
    member _.TokenOf(id: int) : string option =
        if id >= 0 && id < tokenOf.Length then
            Some tokenOf[id]
        else
            match addedById.TryGetValue id with
            | true, t -> Some t.Content
            | _ -> None

    /// Forgets every cached piece (for cold-cache measurements); results never depend on the cache.
    member _.ClearCache() = bpe.ClearCache()

    /// The number of cached pieces.
    member _.CachedPieces = bpe.CacheCount

    /// The rows of Qwen3-0.6B's embedding and lm_head (config.json vocab_size). Ids from VocabSize up to here have no
    /// token and decode to nothing; ids at or above it are outside the model.
    static member ModelRows = 151_936

/// Loading (TokenizerApi contract, "Loading").
[<RequireQualifiedAccess>]
module QwenTokenizer =
    let private metadata (key: string) =
        typeof<TokenizerMarker>.Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
        |> Seq.tryFind (fun a -> a.Key = key && not (String.IsNullOrWhiteSpace a.Value))
        |> Option.map _.Value
        |> Option.defaultWith (fun () ->
            failwithf "Lrm.Tokenizer was built without the %s metadata (see Lrm.Tokenizer.lrm.targets)" key)

    /// The SHA-256 of the pinned Qwen3-0.6B tokenizer.json, lowercase hex: $(LrmTokenizerSha256) from
    /// tools/bootstrap/toolchain.json, built into the assembly.
    let PinnedSha256: string = metadata "Lrm.TokenizerSha256"

    /// The pinned tokenizer.json's path under LRM_DATA, with forward slashes ("models/Qwen3-0.6B/tokenizer.json"):
    /// $(LrmTokenizerJson) relative to $(LrmData), built into the assembly.
    let PinnedPath: string = metadata "Lrm.TokenizerPath"

    /// The real ids of Qwen3: 151,643 vocabulary tokens and 26 added tokens.
    [<Literal>]
    let RealIds = 151_669

    /// The pinned tokenizer.json under the data root (LRM_DATA).
    let defaultPath () = LrmPaths.under PinnedPath

    /// Builds a tokenizer from tokenizer.json bytes without checking their hash; the structure is still checked.
    /// For tests with small vocabularies. `load` is the production path.
    let parse (json: byte[]) : QwenTokenizer =
        if isNull json then
            nullArg "json"

        QwenTokenizer(TokenizerJson.parse json, ContentId.hex (Hash.sha256 json))

    /// Reads tokenizer.json at `path` and checks that its SHA-256 is PinnedSha256 before parsing it. Raises
    /// InvalidDataException on another hash or an unsupported configuration, FileNotFoundException when the file is
    /// missing.
    let load (path: string) : QwenTokenizer =
        let t = Telemetry.current ()

        use span =
            t.Span("tokenizer.load", [ Attr.str TelemetryKeys.Component "tokenizer" ])

        let bytes = File.ReadAllBytes path
        let sha = ContentId.hex (Hash.sha256 bytes)

        if sha <> PinnedSha256 then
            raise (
                InvalidDataException(
                    sprintf
                        "%s has SHA-256 %s, but the TokenizerApi pin is %s (tokenizer.json of Qwen3-0.6B)"
                        path
                        sha
                        PinnedSha256
                )
            )

        let tokenizer = QwenTokenizer(TokenizerJson.parse bytes, sha)

        if tokenizer.VocabSize <> RealIds then
            raise (InvalidDataException(sprintf "%s has %d real ids, expected %d" path tokenizer.VocabSize RealIds))

        span.Add(Attr.int64 "bytes" bytes.LongLength)
        span.Add(Attr.int "vocab" tokenizer.VocabSize)
        span.Add(Attr.int "merges" tokenizer.MergeCount)
        span.Add(Attr.int "added" tokenizer.AddedTokens.Length)
        span.Add(Attr.str "sha256" sha)
        tokenizer

    /// The pinned tokenizer from the data root, or None when the file is absent (as on CI runners until the download
    /// is approved).
    let tryLoadDefault () : QwenTokenizer option =
        let path = defaultPath ()
        if File.Exists path then Some(load path) else None
