/// Qwen3Config.load (ForwardOpsAndQwenModel, "Config"): both rope_theta spellings, an explicit head_dim, and the
/// refusals for what the forward does not implement.
module Lrm.Models.Qwen.Tests.ConfigTests

open System.IO
open System.Text.Json.Nodes
open Expecto
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

/// The config.json of Qwen/Qwen3-0.6B at revision c1899de2 (SHA-256 660db3b7…, Qwen3Pins), as text.
let qwenConfigText =
    """{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"""

/// The Qwen config with `edit` applied to its JSON object.
let private edited (edit: JsonObject -> unit) =
    let o = JsonNode.Parse(qwenConfigText).AsObject()
    edit o
    o.ToJsonString()

let private refuses (label: string) (json: string) (fragment: string) =
    let e =
        throws<Qwen3ConfigException> label (fun () -> Qwen3Config.parse json |> ignore)

    Expect.stringContains e.Message fragment label

let tests =
    testList
        "config"
        [
            testCase "Qwen3-0.6B: every field, head_dim explicit (16 x 128 over a hidden size of 1024)"
            <| fun () ->
                let c = Qwen3Config.parse qwenConfigText
                Expect.equal (c.Layers, c.Hidden, c.Heads, c.KvHeads, c.HeadDim) (28, 1024, 16, 8, 128) "dims"
                Expect.equal (c.Intermediate, c.Vocab, c.MaxPositionEmbeddings) (3072, 151936, 40960) "sizes"
                Expect.equal c.RmsNormEps 1e-6 "eps"
                Expect.equal c.RopeTheta 1e6 "theta"
                Expect.isTrue c.TieWordEmbeddings "tied"
                Expect.equal c.BosTokenId (Some 151643) "bos"
                Expect.equal c.EosTokenIds [ 151645 ] "eos"
                Expect.equal (c.QDim, c.KvDim) (2048, 1024) "q and kv features"

            testCase "the tiny config: 2 layers, hidden 64, 4 and 2 heads of 32, vocabulary 256, eos 7"
            <| fun () ->
                let c = Qwen3Config.load (Path.Combine(tinyModelDir (), "config.json"))
                Expect.equal (c.Layers, c.Hidden, c.Heads, c.KvHeads, c.HeadDim) (2, 64, 4, 2, 32) "dims"
                Expect.equal (c.Intermediate, c.Vocab, c.MaxPositionEmbeddings) (128, 256, 512) "sizes"
                Expect.equal (c.BosTokenId, c.EosTokenIds) (Some 0, [ 7 ]) "special ids"
                Expect.isTrue c.TieWordEmbeddings "tied"

            testCase "rope_parameters.rope_theta (transformers 5) is accepted and equal to rope_theta"
            <| fun () ->
                let nested =
                    edited (fun o ->
                        o.Remove "rope_theta" |> ignore
                        o["rope_parameters"] <- JsonNode.Parse """{"rope_type": "default", "rope_theta": 1000000.0}""")

                let both =
                    edited (fun o -> o["rope_parameters"] <- JsonNode.Parse """{"rope_theta": 1e6}""")

                Expect.equal (Qwen3Config.parse nested) (Qwen3Config.parse qwenConfigText) "nested only"
                Expect.equal (Qwen3Config.parse both) (Qwen3Config.parse qwenConfigText) "both, agreeing"

            testCase "eos_token_id may be a list; a missing tie_word_embeddings is false"
            <| fun () ->
                let c =
                    Qwen3Config.parse (
                        edited (fun o ->
                            o["eos_token_id"] <- JsonNode.Parse "[151645, 151643]"
                            o.Remove "tie_word_embeddings" |> ignore)
                    )

                Expect.equal c.EosTokenIds [ 151645; 151643 ] "eos list"
                Expect.isFalse c.TieWordEmbeddings "untied by default"

            testCase "refuses what the forward does not implement"
            <| fun () ->
                let cases =
                    [
                        "rope_theta disagreement",
                        edited (fun o -> o["rope_parameters"] <- JsonNode.Parse """{"rope_theta": 10000}"""),
                        "disagree"
                        "no rope_theta", edited (fun o -> o.Remove "rope_theta" |> ignore), "rope_theta"
                        "rope_scaling",
                        edited (fun o -> o["rope_scaling"] <- JsonNode.Parse """{"rope_type": "yarn", "factor": 4.0}"""),
                        "rope_scaling"
                        "rope_type",
                        edited (fun o ->
                            o["rope_parameters"] <- JsonNode.Parse """{"rope_type": "linear", "rope_theta": 1e6}"""),
                        "rope_type"
                        "rope_parameters field",
                        edited (fun o -> o["rope_parameters"] <- JsonNode.Parse """{"rope_theta": 1e6, "factor": 2}"""),
                        "factor"
                        "sliding window", edited (fun o -> o["use_sliding_window"] <- JsonValue.Create true), "sliding"
                        "layer_types",
                        edited (fun o ->
                            o["layer_types"] <- JsonNode.Parse """["full_attention", "sliding_attention"]"""),
                        "layer_types"
                        "attention bias",
                        edited (fun o -> o["attention_bias"] <- JsonValue.Create true),
                        "attention_bias"
                        "no head_dim", edited (fun o -> o.Remove "head_dim" |> ignore), "head_dim"
                        "odd head_dim", edited (fun o -> o["head_dim"] <- JsonValue.Create 127), "odd"
                        "heads", edited (fun o -> o["num_key_value_heads"] <- JsonValue.Create 3), "multiple"
                        "activation", edited (fun o -> o["hidden_act"] <- JsonValue.Create "gelu"), "hidden_act"
                        "model type", edited (fun o -> o["model_type"] <- JsonValue.Create "qwen3_moe"), "qwen3_moe"
                        "architecture",
                        edited (fun o -> o["architectures"] <- JsonNode.Parse """["Qwen2ForCausalLM"]"""),
                        "architectures"
                        "zero layers", edited (fun o -> o["num_hidden_layers"] <- JsonValue.Create 0), "at least 1"
                        "eps", edited (fun o -> o["rms_norm_eps"] <- JsonValue.Create -1.0), "rms_norm_eps"
                        "duplicate key",
                        qwenConfigText.Replace("\"head_dim\": 128,", "\"head_dim\": 128, \"head_dim\": 64,"),
                        "twice"
                        "not JSON", "{ \"model_type\": ", "not valid JSON"
                    ]

                for label, json, fragment in cases do
                    refuses label json fragment

            testCase "the FLOP-registry dimensions equal Lrm.Telemetry's Qwen3-0.6B entry"
            <| fun () ->
                Expect.equal
                    (Qwen3Config.dims (Qwen3Config.parse qwenConfigText))
                    Lrm.Telemetry.TransformerDims.qwen3_0_6b
                    "dims"
        ]
