/// Reads and checks tokenizer.json (TokenizerApi contract, "tokenizer.json"). Only the configuration that Qwen3-0.6B
/// uses is accepted: NFC, the exact Split regex with behaviour Isolated, ByteLevel without prefix space or regex, a BPE
/// model without dropout, unknown token, affixes, byte fallback or ignore_merges, and added tokens that are matched on
/// the raw text (normalized false) without stripping or word rules. Anything else raises InvalidDataException, so
/// a tokenizer that would need other code is never silently run with this one.
module internal Lrm.Tokenizer.TokenizerJson

open System
open System.Collections.Generic
open System.IO
open System.Text.Json

/// What the loader builds from the file.
type Parsed =
    {
        TokenBytes: byte[][]
        ByteIds: int[]
        Merges: Dictionary<int64, int64>
        Vocab: Dictionary<string, int>
        Added: AddedToken list
    }

let private fail (what: string) =
    raise (InvalidDataException("tokenizer.json: " + what))

let private prop (e: JsonElement) (path: string) (name: string) =
    match e.TryGetProperty name with
    | true, v -> v
    | _ -> fail (sprintf "%s has no '%s'" path name)

let private str (e: JsonElement) (path: string) (name: string) =
    let v = prop e path name

    if v.ValueKind <> JsonValueKind.String then
        fail (sprintf "%s.%s is not a string" path name)

    v.GetString()

let private expectStr (e: JsonElement) (path: string) (name: string) (expected: string) =
    let v = str e path name

    if v <> expected then
        fail (sprintf "%s.%s is '%s', only '%s' is supported" path name v expected)

let private expectBool (e: JsonElement) (path: string) (name: string) (expected: bool) =
    let v = prop e path name

    let ok =
        (v.ValueKind = JsonValueKind.True && expected)
        || (v.ValueKind = JsonValueKind.False && not expected)

    if not ok then
        fail (sprintf "%s.%s must be %b" path name expected)

let private isNullOr (e: JsonElement) (name: string) (ok: JsonElement -> bool) =
    match e.TryGetProperty name with
    | false, _ -> true
    | true, v -> v.ValueKind = JsonValueKind.Null || ok v

let private expectAbsentOrNull (e: JsonElement) (path: string) (name: string) =
    if not (isNullOr e name (fun _ -> false)) then
        fail (sprintf "%s.%s must be null" path name)

let private expectEmptyAffix (e: JsonElement) (path: string) (name: string) =
    if not (isNullOr e name (fun v -> v.ValueKind = JsonValueKind.String && v.GetString() = "")) then
        fail (sprintf "%s.%s must be empty" path name)

let private expectFalseOrAbsent (e: JsonElement) (path: string) (name: string) =
    match e.TryGetProperty name with
    | false, _ -> ()
    | true, v when v.ValueKind = JsonValueKind.False || v.ValueKind = JsonValueKind.Null -> ()
    | _ -> fail (sprintf "%s.%s must be false" path name)

/// A ByteLevel step. As a pre-tokenizer it must say use_regex false (tokenizers' default is true, which would split
/// again with the GPT-2 regex); as the decoder the flag does nothing.
let private checkByteLevel (e: JsonElement) (path: string) (preTokenizer: bool) =
    expectStr e path "type" "ByteLevel"
    expectBool e path "add_prefix_space" false

    if preTokenizer then
        expectBool e path "use_regex" false

let private checkPipeline (root: JsonElement) =
    expectAbsentOrNull root "$" "truncation"
    expectAbsentOrNull root "$" "padding"
    expectStr (prop root "$" "normalizer") "normalizer" "type" "NFC"

    let pre = prop root "$" "pre_tokenizer"
    expectStr pre "pre_tokenizer" "type" "Sequence"
    let steps = prop pre "pre_tokenizer" "pretokenizers"

    if steps.ValueKind <> JsonValueKind.Array || steps.GetArrayLength() <> 2 then
        fail "pre_tokenizer.pretokenizers must be [Split, ByteLevel]"

    let split = steps[0]
    expectStr split "pre_tokenizer.pretokenizers[0]" "type" "Split"
    expectStr (prop split "Split" "pattern") "Split.pattern" "Regex" PreTokenizer.SplitPattern
    expectStr split "Split" "behavior" "Isolated"
    expectBool split "Split" "invert" false
    checkByteLevel steps[1] "pre_tokenizer.pretokenizers[1]" true

    match root.TryGetProperty "post_processor" with
    | true, pp when pp.ValueKind <> JsonValueKind.Null -> expectStr pp "post_processor" "type" "ByteLevel"
    | _ -> ()

    checkByteLevel (prop root "$" "decoder") "decoder" false

let private checkModel (model: JsonElement) =
    expectStr model "model" "type" "BPE"
    expectAbsentOrNull model "model" "dropout"
    expectAbsentOrNull model "model" "unk_token"
    expectEmptyAffix model "model" "continuing_subword_prefix"
    expectEmptyAffix model "model" "end_of_word_suffix"
    expectFalseOrAbsent model "model" "fuse_unk"
    expectFalseOrAbsent model "model" "byte_fallback"
    expectFalseOrAbsent model "model" "ignore_merges"

let private readVocab (model: JsonElement) =
    let vocab = prop model "model" "vocab"

    if vocab.ValueKind <> JsonValueKind.Object then
        fail "model.vocab is not an object"

    let ids = Dictionary<string, int>(StringComparer.Ordinal)

    for p in vocab.EnumerateObject() do
        if p.Value.ValueKind <> JsonValueKind.Number then
            fail (sprintf "model.vocab[%s] is not a number" p.Name)

        if not (ids.TryAdd(p.Name, p.Value.GetInt32())) then
            fail (sprintf "model.vocab has '%s' twice" p.Name)

    let n = ids.Count
    let bytes = Array.zeroCreate<byte[]> n

    for KeyValue(token, id) in ids do
        if id < 0 || id >= n || not (isNull bytes[id]) then
            fail (sprintf "model.vocab ids are not 0..%d, each once (%s -> %d)" (n - 1) token id)

        match ByteLevel.tryDecode token with
        | Some b when b.Length > 0 -> bytes[id] <- b
        | _ -> fail (sprintf "model.vocab token '%s' is not a byte-level string" token)

    let byteIds =
        Array.init 256 (fun b ->
            match ids.TryGetValue(string (ByteLevel.charOf (byte b))) with
            | true, id -> id
            | _ -> fail (sprintf "model.vocab has no token for byte 0x%02X" b))

    ids, bytes, byteIds

let private readMerges (model: JsonElement) (ids: Dictionary<string, int>) =
    let merges = prop model "model" "merges"

    if merges.ValueKind <> JsonValueKind.Array then
        fail "model.merges is not an array"

    // The default Int64 hash collides on these keys (MergeKey).
    let table = Dictionary<int64, int64>(merges.GetArrayLength(), MergeKey.comparer)
    let mutable rank = 0

    for m in merges.EnumerateArray() do
        let left, right =
            match m.ValueKind with
            | JsonValueKind.Array when m.GetArrayLength() = 2 -> m[0].GetString(), m[1].GetString()
            | JsonValueKind.String ->
                match m.GetString().Split(' ') with
                | [| a; b |] -> a, b
                | _ -> fail (sprintf "model.merges[%d] is not 'a b'" rank)
            | _ -> fail (sprintf "model.merges[%d] is not a pair" rank)

        let idOf (t: string) =
            match ids.TryGetValue t with
            | true, id -> id
            | _ -> fail (sprintf "model.merges[%d] names '%s', which is not in the vocabulary" rank t)

        let l = idOf left
        let r = idOf right
        let merged = idOf (left + right)
        let key = MergeKey.pack l r

        if not (table.TryAdd(key, (int64 rank <<< 32) ||| int64 merged)) then
            fail (sprintf "model.merges[%d] repeats the pair '%s' '%s'" rank left right)

        rank <- rank + 1

    table

let private readAdded (root: JsonElement) (vocabCount: int) =
    match root.TryGetProperty "added_tokens" with
    | false, _ -> []
    | true, a when a.ValueKind = JsonValueKind.Null -> []
    | true, a ->
        let tokens =
            [
                for t in a.EnumerateArray() do
                    let id = (prop t "added_tokens[]" "id").GetInt32()
                    let path = sprintf "added_tokens[id %d]" id
                    let content = str t path "content"
                    expectBool t path "normalized" false
                    expectBool t path "lstrip" false
                    expectBool t path "rstrip" false
                    expectBool t path "single_word" false

                    let special =
                        match (prop t path "special").ValueKind with
                        | JsonValueKind.True -> true
                        | JsonValueKind.False -> false
                        | _ -> fail (path + ".special is not a boolean")

                    {
                        Id = id
                        Content = content
                        Special = special
                    }
            ]
            |> List.sortBy _.Id

        tokens
        |> List.iteri (fun i t ->
            if t.Id <> vocabCount + i then
                fail (
                    sprintf "added token ids must follow the vocabulary (%d expected, %d found)" (vocabCount + i) t.Id
                ))

        tokens

/// Parses and checks tokenizer.json.
let parse (json: byte[]) : Parsed =
    use doc =
        try
            JsonDocument.Parse(ReadOnlyMemory json, JsonDocumentOptions(MaxDepth = 64))
        with :? JsonException as e ->
            raise (InvalidDataException("tokenizer.json is not valid JSON: " + e.Message, e))

    let root = doc.RootElement
    checkPipeline root
    let model = prop root "$" "model"
    checkModel model
    let ids, bytes, byteIds = readVocab model
    let merges = readMerges model ids
    let added = readAdded root ids.Count

    {
        TokenBytes = bytes
        ByteIds = byteIds
        Merges = merges
        Vocab = ids
        Added = added
    }
