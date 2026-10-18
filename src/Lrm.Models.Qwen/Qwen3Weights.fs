namespace Lrm.Models.Qwen

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.IO
open System.Text
open Lrm.Core
open Lrm.Tensor

/// A model folder that breaks the loader rules (ForwardOpsAndQwenModel contract): a missing or malformed REVISION or
/// SHA256SUMS, an unpinned revision, or tensors that do not match the config. Hash mismatches raise Lrm.Tensor's
/// IntegrityException instead.
type Qwen3WeightsException(message: string) =
    inherit FormatException("qwen3 weights: " + message)

/// Where a model folder comes from: its REVISION file.
[<RequireQualifiedAccess>]
type ModelRevision =
    /// `local:<name>`, such as `local:tiny-g1`: verified through SHA256SUMS only (GoldenSet, model folders).
    | Local of name: string
    /// A Hugging Face commit (40 lowercase hex digits): verified through SHA256SUMS and against a pin.
    | HuggingFace of commit: string

    /// The REVISION text, without the newline.
    member r.Text =
        match r with
        | Local n -> "local:" + n
        | HuggingFace c -> c

/// The pinned SHA-256 of files at one Hugging Face revision.
type RevisionPin =
    {
        /// The Hugging Face model id, for messages.
        Model: string
        /// The commit.
        Revision: string
        /// File name and SHA-256 of every pinned file; each must also be in the folder's SHA256SUMS.
        Files: (string * ContentId) list
    }

/// The Hugging Face revisions this build knows (tools/bootstrap/toolchain.json, `model`, and the folder's verified
/// SHA256SUMS).
[<RequireQualifiedAccess>]
module Qwen3Pins =
    /// Qwen/Qwen3-0.6B at the plan's revision.
    let qwen3_0_6b =
        {
            Model = "Qwen/Qwen3-0.6B"
            Revision = "c1899de289a04d12100db370d81485cdf75e47ca"
            Files =
                [
                    "config.json", ContentId.parse "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
                    "model.safetensors",
                    ContentId.parse "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
                    "tokenizer.json", ContentId.parse "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
                ]
        }

    /// Every built-in pin.
    let all = [ qwen3_0_6b ]

/// The tensor names of one decoder layer.
type Qwen3LayerParams =
    {
        InputNorm: string
        Q: string
        K: string
        V: string
        O: string
        QNorm: string
        KNorm: string
        PostNorm: string
        Gate: string
        Up: string
        Down: string
    }

/// The tensor names the forward reads, after aliasing: LmHead is the embedding's name when the head is tied.
type Qwen3Params =
    {
        Embed: string
        Layers: Qwen3LayerParams[]
        Norm: string
        LmHead: string
    }

/// The static description the forward runs on: the config and the tensor names.
type Qwen3Model =
    {
        Config: Qwen3Config
        Params: Qwen3Params
    }

[<RequireQualifiedAccess>]
module Qwen3Params =
    /// The embedding's safetensors name.
    [<Literal>]
    let EmbedName = "model.embed_tokens.weight"

    /// lm_head's safetensors name.
    [<Literal>]
    let LmHeadName = "lm_head.weight"

    /// The final norm's safetensors name.
    [<Literal>]
    let NormName = "model.norm.weight"

    let private layer (i: int) =
        let p = Invariant.format $"model.layers.{i}."

        {
            InputNorm = p + "input_layernorm.weight"
            Q = p + "self_attn.q_proj.weight"
            K = p + "self_attn.k_proj.weight"
            V = p + "self_attn.v_proj.weight"
            O = p + "self_attn.o_proj.weight"
            QNorm = p + "self_attn.q_norm.weight"
            KNorm = p + "self_attn.k_norm.weight"
            PostNorm = p + "post_attention_layernorm.weight"
            Gate = p + "mlp.gate_proj.weight"
            Up = p + "mlp.up_proj.weight"
            Down = p + "mlp.down_proj.weight"
        }

    /// The names for a config, with lm_head tied to the embedding or separate.
    let ofConfig (c: Qwen3Config) (lmHeadTied: bool) =
        {
            Embed = EmbedName
            Layers = Array.init c.Layers layer
            Norm = NormName
            LmHead = if lmHeadTied then EmbedName else LmHeadName
        }

    /// Every tensor of a checkpoint and its BF16 shape, in canonical order, without lm_head.
    let shapes (c: Qwen3Config) : (string * int64[]) list =
        let h, q, kv, i, d =
            int64 c.Hidden, int64 c.QDim, int64 c.KvDim, int64 c.Intermediate, int64 c.HeadDim

        [
            yield EmbedName, [| int64 c.Vocab; h |]
            for l in 0 .. c.Layers - 1 do
                let p = layer l
                yield p.InputNorm, [| h |]
                yield p.Q, [| q; h |]
                yield p.K, [| kv; h |]
                yield p.V, [| kv; h |]
                yield p.O, [| h; q |]
                yield p.QNorm, [| d |]
                yield p.KNorm, [| d |]
                yield p.PostNorm, [| h |]
                yield p.Gate, [| i; h |]
                yield p.Up, [| i; h |]
                yield p.Down, [| h; i |]
            yield NormName, [| h |]
        ]

/// How lm_head was resolved.
[<RequireQualifiedAccess>]
type LmHeadSource =
    /// The checkpoint has no lm_head.weight; the head is the embedding (the tiny model, Qwen3-0.6B-Base).
    | Missing
    /// lm_head.weight is present and equal to the embedding by SHA-256, checked once at open; it is not used.
    | AliasVerified
    /// tie_word_embeddings is false: lm_head.weight is its own tensor.
    | Separate

/// The options of Qwen3Weights.Open.
type Qwen3WeightsOptions =
    {
        /// The Hugging Face revisions accepted; a Hugging Face revision without a pin is refused.
        Pins: RevisionPin list
    }

    static member Default = { Pins = Qwen3Pins.all }

/// A verified Qwen3 checkpoint folder (ForwardOpsAndQwenModel contract): config.json, model.safetensors (memory-mapped
/// and pinned open), REVISION and SHA256SUMS. Every file SHA256SUMS lists is hashed before use; model.safetensors over
/// the mapped bytes the reader then serves. Tensors are BF16 and match the config exactly. Dispose to unmap; tensors
/// already returned by Bf16 stay valid.
[<Sealed>]
type Qwen3Weights
    private
    (
        dir: string,
        revision: ModelRevision,
        config: Qwen3Config,
        model: Qwen3Model,
        file: SafetensorsFile,
        lmHead: LmHeadSource,
        sums: (string * ContentId) list
    ) =
    let cache =
        ConcurrentDictionary<string, Lazy<HostTensor<uint16>>>(StringComparer.Ordinal)

    let used =
        HashSet<string>(
            (Qwen3Params.shapes config |> List.map fst)
            @ (if lmHead = LmHeadSource.Separate then
                   [ Qwen3Params.LmHeadName ]
               else
                   []),
            StringComparer.Ordinal
        )

    static let fail (msg: string) : 'a = raise (Qwen3WeightsException msg)

    static let isHex (s: string) =
        s |> Seq.forall (fun c -> (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))

    /// `<64 hex> *<name>` (binary mode) or `<64 hex>  <name>` (text mode); names are plain file names.
    static let parseSums (path: string) (text: string) =
        let lines =
            text.Replace("\r\n", "\n").Split('\n') |> Array.filter (fun l -> l.Length > 0)

        let seen = HashSet<string>(StringComparer.Ordinal)

        [
            for l in lines ->
                if
                    l.Length < 67
                    || l[64] <> ' '
                    || (l[65] <> '*' && l[65] <> ' ')
                    || not (isHex (l.Substring(0, 64)))
                then
                    fail (sprintf "%s: '%s' is not a sha256sum line" path l)

                let name = l.Substring 66

                if
                    name.Length = 0
                    || name.IndexOfAny([| '/'; '\\'; ':' |]) >= 0
                    || name = "."
                    || name = ".."
                    || name = "SHA256SUMS"
                then
                    fail (sprintf "%s: '%s' is not a file of the folder" path name)

                if not (seen.Add name) then
                    fail (sprintf "%s lists %s twice" path name)

                name, ContentId.parse (l.Substring(0, 64))
        ]

    static let parseRevision (path: string) (bytes: byte[]) =
        let text = Encoding.UTF8.GetString bytes
        let t = text.TrimEnd('\n', '\r')

        if t.Contains '\n' || t.Contains '\r' then
            fail (sprintf "%s holds more than one line" path)

        if t.StartsWith("local:", StringComparison.Ordinal) then
            let name = t.Substring 6

            if
                name.Length = 0
                || not (name |> Seq.forall (fun c -> Char.IsAsciiLetterOrDigit c || "-_.".Contains c))
            then
                fail (sprintf "%s: '%s' is not local:<name>" path t)

            ModelRevision.Local name
        elif t.Length = 40 && isHex t then
            ModelRevision.HuggingFace t
        else
            fail (sprintf "%s: '%s' is neither local:<name> nor a 40-digit commit" path t)

    static let readVerified (path: string) (expected: ContentId) =
        if not (File.Exists path) then
            fail (sprintf "%s is listed in SHA256SUMS but missing" path)

        let bytes = File.ReadAllBytes path
        let actual = Hash.sha256 bytes

        if actual <> expected then
            raise (IntegrityException(path, "sha256 " + expected.Hex, "sha256 " + actual.Hex))

        bytes

    /// The folder.
    member _.Dir = dir

    /// The REVISION it was verified as.
    member _.Revision = revision

    member _.Config = config

    /// The config and the tensor names the forward reads.
    member _.Model = model

    member _.Params = model.Params

    /// How lm_head was resolved.
    member _.LmHead = lmHead

    /// The SHA-256 of model.safetensors (verified over the mapped bytes).
    member _.Sha256 =
        sums
        |> List.pick (fun (n, h) -> if n = "model.safetensors" then Some h else None)

    /// Every SHA256SUMS line, each verified at open.
    member _.Sums = sums

    /// The memory-mapped checkpoint (open for reading only; nobody can write it while it is open).
    member _.File = file

    /// The tensors the forward reads, in canonical order (lm_head only when it is separate).
    member _.Names =
        [|
            yield! Qwen3Params.shapes config |> List.map fst
            if lmHead = LmHeadSource.Separate then
                yield Qwen3Params.LmHeadName
        |]

    /// A tensor's BF16 elements (a copy out of the mapping, made once and shared). Raises KeyNotFoundException for a
    /// name the forward does not read; lm_head.weight resolves to the embedding when the head is tied.
    member this.Bf16(name: string) : HostTensor<uint16> =
        let name =
            if name = Qwen3Params.LmHeadName && lmHead <> LmHeadSource.Separate then
                Qwen3Params.EmbedName
            else
                name

        if not (used.Contains name) then
            raise (KeyNotFoundException(sprintf "%s holds no tensor '%s' that the forward reads" dir name))

        cache.GetOrAdd(name, (fun n -> lazy (file.ReadBf16 n))).Force()

    /// A tensor's raw bytes straight from the mapping (no copy), valid until Dispose. Aliases resolve as in Bf16.
    member this.Span(name: string) : ReadOnlySpan<byte> =
        let name =
            if name = Qwen3Params.LmHeadName && lmHead <> LmHeadSource.Separate then
                Qwen3Params.EmbedName
            else
                name

        if not (used.Contains name) then
            raise (KeyNotFoundException(sprintf "%s holds no tensor '%s' that the forward reads" dir name))

        file.Span name

    /// Opens and verifies a model folder (ForwardOpsAndQwenModel contract, "Loading").
    static member Open(dir: string, ?options: Qwen3WeightsOptions) : Qwen3Weights =
        let options = defaultArg options Qwen3WeightsOptions.Default
        let dir = Path.GetFullPath(Path.TrimEndingDirectorySeparator dir)

        if not (Directory.Exists dir) then
            fail (sprintf "%s is not a folder" dir)

        let at (name: string) = Path.Combine(dir, name)
        let sumsPath = at "SHA256SUMS"

        if not (File.Exists sumsPath) then
            fail (sprintf "%s has no SHA256SUMS" dir)

        let sums = parseSums sumsPath (File.ReadAllText sumsPath)

        let sumOf (name: string) =
            sums |> List.tryPick (fun (n, h) -> if n = name then Some h else None)

        for required in [ "REVISION"; "config.json"; "model.safetensors" ] do
            if (sumOf required).IsNone then
                fail (sprintf "%s does not list %s" sumsPath required)

        let revision =
            parseRevision (at "REVISION") (readVerified (at "REVISION") (sumOf "REVISION").Value)

        match revision with
        | ModelRevision.Local _ -> ()
        | ModelRevision.HuggingFace commit ->
            match options.Pins |> List.tryFind (fun p -> p.Revision = commit) with
            | None -> fail (sprintf "%s: revision %s is not pinned (Qwen3Pins)" dir commit)
            | Some pin ->
                for name, expected in pin.Files do
                    match sumOf name with
                    | None -> fail (sprintf "%s does not list %s, which the %s pin covers" sumsPath name pin.Model)
                    | Some listed when listed <> expected ->
                        raise (
                            IntegrityException(
                                at name,
                                "sha256 " + expected.Hex + " (pin " + pin.Model + ")",
                                "sha256 " + listed.Hex + " (SHA256SUMS)"
                            )
                        )
                    | Some _ -> ()

        let config =
            Qwen3Config.parseBytes (at "config.json") (readVerified (at "config.json") (sumOf "config.json").Value)

        for name, expected in sums do
            if name <> "model.safetensors" && name <> "config.json" && name <> "REVISION" then
                readVerified (at name) expected |> ignore

        let stPath = at "model.safetensors"

        if not (File.Exists stPath) then
            fail (sprintf "%s is listed in SHA256SUMS but missing" stPath)

        let file =
            SafetensorsFile.OpenVerified(stPath, FileInfo(stPath).Length, (sumOf "model.safetensors").Value)

        try
            let expected = Qwen3Params.shapes config

            let expectedNames =
                HashSet<string>(expected |> List.map fst, StringComparer.Ordinal)

            for name in file.Names do
                if name <> Qwen3Params.LmHeadName && not (expectedNames.Contains name) then
                    fail (sprintf "%s holds '%s', which a %d-layer Qwen3 does not have" stPath name config.Layers)

            let check (name: string) (shape: int64[]) =
                match file.TryEntry name with
                | None -> fail (sprintf "%s has no '%s'" stPath name)
                | Some e ->
                    if e.Desc.DType <> DType.BF16 then
                        fail (sprintf "%s: '%s' is %s, not BF16" stPath name (DType.name e.Desc.DType))

                    if not (e.Desc.Shape.Equals(Shape.ofArray shape)) then
                        fail (
                            sprintf "%s: '%s' is %O; the config gives %O" stPath name e.Desc.Shape (Shape.ofArray shape)
                        )

            for name, shape in expected do
                check name shape

            let lmHead =
                match file.Contains Qwen3Params.LmHeadName, config.TieWordEmbeddings with
                | false, true -> LmHeadSource.Missing
                | false, false ->
                    fail (sprintf "%s: tie_word_embeddings is false but there is no lm_head.weight" stPath)
                | true, tied ->
                    check Qwen3Params.LmHeadName [| int64 config.Vocab; int64 config.Hidden |]

                    if not tied then
                        LmHeadSource.Separate
                    else
                        let a = Hash.sha256Span (file.Span Qwen3Params.LmHeadName)
                        let b = Hash.sha256Span (file.Span Qwen3Params.EmbedName)

                        if a <> b then
                            fail (
                                sprintf
                                    "%s: tie_word_embeddings is true but lm_head.weight (sha256 %s) differs from the embedding (sha256 %s)"
                                    stPath
                                    a.Hex
                                    b.Hex
                            )

                        LmHeadSource.AliasVerified

            let model =
                {
                    Config = config
                    Params = Qwen3Params.ofConfig config (lmHead <> LmHeadSource.Separate)
                }

            new Qwen3Weights(dir, revision, config, model, file, lmHead, sums)
        with _ ->
            file.Dispose()
            reraise ()

    member _.Dispose() = file.Dispose()

    interface IDisposable with
        member this.Dispose() = this.Dispose()
