namespace Lrm.Models.Qwen

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Lrm.Core

/// A config.json that the Qwen3 forward cannot run, or that is not well formed (ForwardOpsAndQwenModel contract).
type Qwen3ConfigException(message: string) =
    inherit FormatException("qwen3 config: " + message)

/// The dimensions of a dense Qwen3 decoder (ForwardOpsAndQwenModel contract), read from a Hugging Face config.json.
/// Only what the forward pass uses is kept; everything else in the file is validated or ignored.
type Qwen3Config =
    {
        /// num_hidden_layers.
        Layers: int
        /// hidden_size.
        Hidden: int
        /// num_attention_heads.
        Heads: int
        /// num_key_value_heads.
        KvHeads: int
        /// head_dim, read explicitly (Qwen3-0.6B has 16 heads of 128 over a hidden size of 1024).
        HeadDim: int
        /// intermediate_size.
        Intermediate: int
        /// vocab_size: the rows of the embedding and of lm_head.
        Vocab: int
        /// rms_norm_eps, as float64; each mode rounds it once to its own type.
        RmsNormEps: float
        /// rope_theta, from the top level or from rope_parameters.rope_theta.
        RopeTheta: float
        /// max_position_embeddings.
        MaxPositionEmbeddings: int
        /// tie_word_embeddings: lm_head is the embedding (false when absent, as in transformers).
        TieWordEmbeddings: bool
        /// bos_token_id, when the file has one.
        BosTokenId: int option
        /// eos_token_id: one id, a list of ids, or none.
        EosTokenIds: int list
    }

    /// Query features: Heads * HeadDim.
    member c.QDim = c.Heads * c.HeadDim

    /// Key and value features: KvHeads * HeadDim.
    member c.KvDim = c.KvHeads * c.HeadDim

/// Loading and validating Qwen3 configs.
[<RequireQualifiedAccess>]
module Qwen3Config =
    let private fail (msg: string) : 'a = raise (Qwen3ConfigException msg)

    /// The properties of an object, refusing duplicate names (System.Text.Json keeps the last one silently).
    let private properties (where: string) (o: JsonElement) =
        if o.ValueKind <> JsonValueKind.Object then
            fail (sprintf "%s is not a JSON object" where)

        let d = Dictionary<string, JsonElement>(StringComparer.Ordinal)

        for p in o.EnumerateObject() do
            if d.ContainsKey p.Name then
                fail (sprintf "%s has the key '%s' twice" where p.Name)

            d[p.Name] <- p.Value

        d

    let private tryGet (d: Dictionary<string, JsonElement>) (key: string) =
        match d.TryGetValue key with
        | true, v when v.ValueKind <> JsonValueKind.Null -> Some v
        | _ -> None

    let private int' (d: Dictionary<string, JsonElement>) (key: string) =
        match tryGet d key with
        | Some v when v.ValueKind = JsonValueKind.Number ->
            match v.TryGetInt32() with
            | true, i -> i
            | _ -> fail (sprintf "'%s' is not a 32-bit integer" key)
        | Some _ -> fail (sprintf "'%s' is not a number" key)
        | None -> fail (sprintf "'%s' is missing" key)

    let private positive (d: Dictionary<string, JsonElement>) (key: string) =
        let v = int' d key

        if v < 1 then
            fail (sprintf "'%s' is %d; it must be at least 1" key v)

        v

    let private float' (key: string) (v: JsonElement) =
        if v.ValueKind <> JsonValueKind.Number then
            fail (sprintf "'%s' is not a number" key)

        let x = v.GetDouble()

        if not (Double.IsFinite x) || x <= 0.0 then
            fail (sprintf "'%s' is %s; it must be positive and finite" key (Invariant.f64 x))

        x

    let private str (d: Dictionary<string, JsonElement>) (key: string) =
        match tryGet d key with
        | Some v when v.ValueKind = JsonValueKind.String -> Some(v.GetString())
        | Some _ -> fail (sprintf "'%s' is not a string" key)
        | None -> None

    let private bool' (d: Dictionary<string, JsonElement>) (key: string) =
        match tryGet d key with
        | Some v when v.ValueKind = JsonValueKind.True -> Some true
        | Some v when v.ValueKind = JsonValueKind.False -> Some false
        | Some _ -> fail (sprintf "'%s' is not a boolean" key)
        | None -> None

    let private tokenIds (d: Dictionary<string, JsonElement>) (key: string) =
        let id (v: JsonElement) =
            match v.ValueKind, v.TryGetInt32() with
            | JsonValueKind.Number, (true, i) when i >= 0 -> i
            | _ -> fail (sprintf "'%s' holds %s, not a token id" key (v.GetRawText()))

        match tryGet d key with
        | None -> []
        | Some v when v.ValueKind = JsonValueKind.Array -> [ for x in v.EnumerateArray() -> id x ]
        | Some v -> [ id v ]

    /// rope_theta from the top level or from rope_parameters (transformers 5 writes the latter). Both may be present
    /// when they agree. rope_parameters may hold only rope_type "default" and rope_theta: any other field or type
    /// changes the RoPE table, which the pinned T_spec does not cover.
    let private ropeTheta (d: Dictionary<string, JsonElement>) =
        let top = tryGet d "rope_theta" |> Option.map (float' "rope_theta")

        let nested =
            match tryGet d "rope_parameters" with
            | None -> None
            | Some rp ->
                let p = properties "rope_parameters" rp

                for k in p.Keys do
                    if k <> "rope_type" && k <> "rope_theta" then
                        fail (sprintf "rope_parameters.%s is not supported (only rope_type default and rope_theta)" k)

                match str p "rope_type" with
                | None
                | Some "default" -> ()
                | Some t -> fail (sprintf "rope_parameters.rope_type '%s' is not supported (only default)" t)

                tryGet p "rope_theta" |> Option.map (float' "rope_parameters.rope_theta")

        match top, nested with
        | Some a, Some b when BitConverter.DoubleToInt64Bits a <> BitConverter.DoubleToInt64Bits b ->
            fail (
                sprintf "rope_theta %s and rope_parameters.rope_theta %s disagree" (Invariant.f64 a) (Invariant.f64 b)
            )
        | Some a, _ -> a
        | None, Some b -> b
        | None, None -> fail "neither rope_theta nor rope_parameters.rope_theta is present"

    /// Parses and validates config.json text. `source` names the file in messages.
    let parseBytes (source: string) (bytes: byte[]) : Qwen3Config =
        let doc =
            try
                JsonDocument.Parse(ReadOnlyMemory bytes)
            with :? JsonException as e ->
                fail (
                    sprintf
                        "%s is not valid JSON (byte %d)"
                        source
                        (defaultArg (Option.ofNullable e.BytePositionInLine) 0L)
                )

        use doc = doc
        let d = properties source doc.RootElement

        match str d "model_type" with
        | Some "qwen3" -> ()
        | Some t -> fail (sprintf "model_type '%s' is not qwen3" t)
        | None -> fail "'model_type' is missing"

        match tryGet d "architectures" with
        | Some a when a.ValueKind = JsonValueKind.Array ->
            let names =
                [
                    for x in a.EnumerateArray() ->
                        if x.ValueKind = JsonValueKind.String then
                            x.GetString()
                        else
                            ""
                ]

            if not (List.contains "Qwen3ForCausalLM" names) then
                fail (sprintf "architectures %s do not include Qwen3ForCausalLM" (a.GetRawText()))
        | Some _ -> fail "'architectures' is not an array"
        | None -> fail "'architectures' is missing"

        match str d "hidden_act" with
        | None
        | Some "silu" -> ()
        | Some act -> fail (sprintf "hidden_act '%s' is not supported (only silu)" act)

        if bool' d "attention_bias" = Some true then
            fail "attention_bias is true; the forward has no attention bias"

        if bool' d "use_sliding_window" = Some true then
            fail "use_sliding_window is true; the forward has full causal attention only"

        match tryGet d "layer_types" with
        | Some lt when lt.ValueKind = JsonValueKind.Array ->
            for x in lt.EnumerateArray() do
                if x.ValueKind <> JsonValueKind.String || x.GetString() <> "full_attention" then
                    fail (sprintf "layer_types holds %s; only full_attention is supported" (x.GetRawText()))
        | Some _ -> fail "'layer_types' is not an array"
        | None -> ()

        match tryGet d "rope_scaling" with
        | None -> ()
        | Some v -> fail (sprintf "rope_scaling is %s; only null is supported" (v.GetRawText()))

        let theta = ropeTheta d

        let headDim =
            match tryGet d "head_dim" with
            | None -> fail "'head_dim' is missing (Qwen3 configs name it explicitly)"
            | Some _ -> positive d "head_dim"

        if headDim % 2 <> 0 then
            fail (sprintf "head_dim %d is odd; NeoX RoPE rotates halves" headDim)

        let heads = positive d "num_attention_heads"
        let kvHeads = positive d "num_key_value_heads"

        if heads % kvHeads <> 0 then
            fail (sprintf "%d attention heads are not a multiple of %d key-value heads" heads kvHeads)

        let eps =
            match tryGet d "rms_norm_eps" with
            | Some v -> float' "rms_norm_eps" v
            | None -> fail "'rms_norm_eps' is missing"

        let bos =
            match tokenIds d "bos_token_id" with
            | [] -> None
            | [ b ] -> Some b
            | _ -> fail "'bos_token_id' holds more than one id"

        {
            Layers = positive d "num_hidden_layers"
            Hidden = positive d "hidden_size"
            Heads = heads
            KvHeads = kvHeads
            HeadDim = headDim
            Intermediate = positive d "intermediate_size"
            Vocab = positive d "vocab_size"
            RmsNormEps = eps
            RopeTheta = theta
            MaxPositionEmbeddings = positive d "max_position_embeddings"
            TieWordEmbeddings = defaultArg (bool' d "tie_word_embeddings") false
            BosTokenId = bos
            EosTokenIds = tokenIds d "eos_token_id"
        }

    /// Parses and validates config.json text.
    let parse (json: string) =
        parseBytes "config.json" (Text.Encoding.UTF8.GetBytes json)

    /// Reads and validates a config.json file. Qwen3Weights.open reads it through SHA256SUMS instead.
    let load (path: string) =
        parseBytes path (File.ReadAllBytes path)

    /// The FLOP-registry dimensions of this config (RunLog contract, "FLOP registry").
    let dims (c: Qwen3Config) : Lrm.Telemetry.TransformerDims =
        {
            Hidden = int64 c.Hidden
            Heads = int64 c.Heads
            KvHeads = int64 c.KvHeads
            HeadDim = int64 c.HeadDim
            Intermediate = int64 c.Intermediate
            Layers = int64 c.Layers
            Vocab = int64 c.Vocab
            Weight = DType.BF16
        }
