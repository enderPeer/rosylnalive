/// Qwen3Weights.Open (ForwardOpsAndQwenModel, "Loading"): REVISION and SHA256SUMS, pins, the lm_head alias and the
/// tensor checks, on the tiny model and on tampered copies of it.
module Lrm.Models.Qwen.Tests.WeightsTests

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open System.Text.Json.Nodes
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

let private modelFiles = [ "REVISION"; "config.json"; "model.safetensors" ]

/// A copy of the tiny model folder, changed by `edit`, with SHA256SUMS rewritten unless `keepSums`.
let private tampered (label: string) (keepSums: bool) (edit: string -> unit) =
    let d = copyModel (tinyModelDir ()) label
    edit d

    if not keepSums then
        writeSums d modelFiles

    d

/// Rewrites model.safetensors in `dir` after `change` maps its tensors.
let private rewriteTensors (dir: string) (change: SafetensorsTensor list -> SafetensorsTensor list) =
    let path = Path.Combine(dir, "model.safetensors")
    let tensors, metadata = tensorsOf path
    Safetensors.writeFile path metadata (change tensors)

/// Rewrites config.json in `dir`.
let private rewriteConfig (dir: string) (change: JsonObject -> unit) =
    let path = Path.Combine(dir, "config.json")
    let o = JsonNode.Parse(File.ReadAllText path).AsObject()
    change o
    File.WriteAllText(path, o.ToJsonString(JsonSerializerOptions(WriteIndented = true)))

let private embedTensor (tensors: SafetensorsTensor list) =
    tensors |> List.find (fun t -> t.Name = Qwen3Params.EmbedName)

let private withDir (d: string) (f: string -> unit) =
    try
        f d
    finally
        deleteDir d

let private openFails<'e when 'e :> exn>
    (label: string)
    (fragment: string)
    (options: Qwen3WeightsOptions option)
    (d: string)
    =
    withDir d (fun d ->
        let e =
            throws<'e> label (fun () ->
                let w =
                    match options with
                    | Some o -> Qwen3Weights.Open(d, o)
                    | None -> Qwen3Weights.Open d

                w.Dispose())

        Expect.stringContains e.Message fragment label)

let tests =
    testList
        "weights"
        [
            testCase "the tiny model opens as local:tiny-g1 with 24 BF16 tensors and the head aliased to the embedding"
            <| fun () ->
                let w = tinyWeights.Value
                Expect.equal w.Revision (ModelRevision.Local "tiny-g1") "revision"
                Expect.equal w.LmHead LmHeadSource.Missing "no lm_head in the file"
                Expect.equal w.Params.LmHead Qwen3Params.EmbedName "lm_head reads the embedding"
                Expect.equal w.Names.Length 24 "tensors the forward reads"
                Expect.equal (w.Sums |> List.map fst) [ "REVISION"; "config.json"; "model.safetensors" ] "SHA256SUMS"

                Expect.equal
                    w.Sha256
                    (tiny.Force().File "model/model.safetensors").Sha256
                    "model.safetensors equals the golden manifest's entry"

                for name, shape in Qwen3Params.shapes w.Config do
                    let t = w.Bf16 name
                    Expect.equal t.DType DType.BF16 name
                    Expect.equal t.Shape (Shape.ofArray shape) name

            testCase "Bf16 copies a tensor once and shares it; lm_head.weight resolves to the embedding"
            <| fun () ->
                let w = tinyWeights.Value
                Expect.isTrue (obj.ReferenceEquals(w.Bf16 Qwen3Params.EmbedName, w.Bf16 Qwen3Params.EmbedName)) "shared"
                Expect.isTrue (obj.ReferenceEquals(w.Bf16 Qwen3Params.LmHeadName, w.Bf16 Qwen3Params.EmbedName)) "alias"

                Expect.equal
                    (w.Bf16 Qwen3Params.EmbedName).Data[0..3]
                    [| 0xbe06us; 0x3cb6us; 0x3db7us; 0x3d27us |]
                    "first bits (GoldenSet)"

                Expect.equal (w.Span Qwen3Params.EmbedName).Length (256 * 64 * 2) "span"

                Expect.throwsT<KeyNotFoundException>
                    (fun () -> w.Bf16 "model.layers.9.mlp.up_proj.weight" |> ignore)
                    "unknown"

            testCase "a changed byte, file or listing is refused before use"
            <| fun () ->
                tampered "flip" true (fun d ->
                    let p = Path.Combine(d, "model.safetensors")
                    let b = File.ReadAllBytes p
                    b[b.Length - 1] <- b[b.Length - 1] ^^^ 1uy
                    File.WriteAllBytes(p, b))
                |> openFails<IntegrityException> "flipped tensor byte" "sha256" None

                tampered "config" true (fun d -> File.AppendAllText(Path.Combine(d, "config.json"), " "))
                |> openFails<IntegrityException> "changed config" "sha256" None

                tampered "unlisted" true (fun d ->
                    let p = Path.Combine(d, "SHA256SUMS")

                    File.WriteAllLines(
                        p,
                        File.ReadAllLines p |> Array.filter (fun l -> not (l.EndsWith "config.json"))
                    ))
                |> openFails<Qwen3WeightsException> "config.json not listed" "does not list config.json" None

                tampered "nosums" true (fun d -> File.Delete(Path.Combine(d, "SHA256SUMS")))
                |> openFails<Qwen3WeightsException> "no SHA256SUMS" "no SHA256SUMS" None

                tampered "missing" true (fun d ->
                    File.AppendAllText(Path.Combine(d, "SHA256SUMS"), String('0', 64) + " *extra.txt\n"))
                |> openFails<Qwen3WeightsException> "a listed file is missing" "extra.txt" None

                tampered "badline" true (fun d -> File.AppendAllText(Path.Combine(d, "SHA256SUMS"), "not a line\n"))
                |> openFails<Qwen3WeightsException> "a malformed line" "sha256sum line" None

            testCase "REVISION: local:<name>, a pinned commit, or nothing else"
            <| fun () ->
                tampered "rev" false (fun d -> File.WriteAllText(Path.Combine(d, "REVISION"), "main\n"))
                |> openFails<Qwen3WeightsException> "a branch name" "neither" None

                let commit = String('a', 40)

                tampered "unpinned" false (fun d -> File.WriteAllText(Path.Combine(d, "REVISION"), commit + "\n"))
                |> openFails<Qwen3WeightsException> "an unpinned commit" "not pinned" None

                let d =
                    tampered "pinned" false (fun d -> File.WriteAllText(Path.Combine(d, "REVISION"), commit + "\n"))

                let pin (configSha: ContentId) =
                    {
                        Pins =
                            [
                                {
                                    Model = "test/tiny"
                                    Revision = commit
                                    Files =
                                        [
                                            "config.json", configSha
                                            "model.safetensors", Hash.sha256File (Path.Combine(d, "model.safetensors"))
                                        ]
                                }
                            ]
                    }

                withDir d (fun d ->
                    let good = pin (Hash.sha256File (Path.Combine(d, "config.json")))
                    use w = Qwen3Weights.Open(d, good)
                    Expect.equal w.Revision (ModelRevision.HuggingFace commit) "pinned commit opens"

                    let e =
                        throws<IntegrityException> "a pin that disagrees with SHA256SUMS" (fun () ->
                            Qwen3Weights.Open(d, pin (Hash.sha256Utf8 "other")).Dispose())

                    Expect.stringContains e.Message "test/tiny" "names the pin")

            testCase "lm_head: present and equal is one tensor; different, or missing while untied, is refused"
            <| fun () ->
                let addHead (perturb: bool) (d: string) =
                    rewriteTensors d (fun ts ->
                        let e = embedTensor ts
                        let bytes = e.Data.ToArray()

                        if perturb then
                            bytes[0] <- bytes[0] ^^^ 1uy

                        SafetensorsTensor.ofBytes Qwen3Params.LmHeadName e.DType e.Shape bytes :: ts)

                withDir (tampered "head-equal" false (addHead false)) (fun d ->
                    use w = Qwen3Weights.Open d
                    Expect.equal w.LmHead LmHeadSource.AliasVerified "verified alias"
                    Expect.equal w.Params.LmHead Qwen3Params.EmbedName "reads the embedding"
                    Expect.equal w.Names.Length 24 "one tensor, not two")

                tampered "head-differs" false (addHead true)
                |> openFails<Qwen3WeightsException> "a different lm_head" "differs from the embedding" None

                withDir
                    (tampered "untied" false (fun d ->
                        addHead true d
                        rewriteConfig d (fun o -> o["tie_word_embeddings"] <- JsonValue.Create false)))
                    (fun d ->
                        use w = Qwen3Weights.Open d
                        Expect.equal w.LmHead LmHeadSource.Separate "separate head"
                        Expect.equal w.Params.LmHead Qwen3Params.LmHeadName "reads lm_head"

                        Expect.isFalse
                            (obj.ReferenceEquals(w.Bf16 Qwen3Params.LmHeadName, w.Bf16 Qwen3Params.EmbedName))
                            "own tensor")

                tampered "untied-missing" false (fun d ->
                    rewriteConfig d (fun o -> o["tie_word_embeddings"] <- JsonValue.Create false))
                |> openFails<Qwen3WeightsException> "untied without lm_head" "no lm_head" None

            testCase "tensors must match the config: names, shapes and BF16"
            <| fun () ->
                tampered "extra" false (fun d ->
                    rewriteTensors d (fun ts ->
                        SafetensorsTensor.ofBytes
                            "model.layers.2.mlp.up_proj.weight"
                            DType.BF16
                            (Shape.vector 1)
                            [| 0uy; 0uy |]
                        :: ts))
                |> openFails<Qwen3WeightsException> "an extra layer" "does not have" None

                tampered "shape" false (fun d ->
                    rewriteConfig d (fun o -> o["intermediate_size"] <- JsonValue.Create 64))
                |> openFails<Qwen3WeightsException> "a shape the config does not give" "the config gives" None

                tampered "dtype" false (fun d ->
                    rewriteTensors d (fun ts ->
                        ts
                        |> List.map (fun t ->
                            if t.Name = Qwen3Params.NormName then
                                SafetensorsTensor.ofBytes t.Name DType.F32 (Shape.vector 32) (t.Data.ToArray())
                            else
                                t)))
                |> openFails<Qwen3WeightsException> "an F32 tensor" "not BF16" None

                tampered "removed" false (fun d ->
                    rewriteTensors d (List.filter (fun t -> t.Name <> "model.layers.1.self_attn.k_norm.weight")))
                |> openFails<Qwen3WeightsException> "a missing tensor" "k_norm" None

            testCase "the Qwen3-0.6B pin equals tools/bootstrap/toolchain.json's model entry"
            <| fun () ->
                use doc =
                    JsonDocument.Parse(
                        File.ReadAllBytes(Path.Combine(repoRoot (), "tools", "bootstrap", "toolchain.json"))
                    )

                let m = doc.RootElement.GetProperty "model"
                let pin = Qwen3Pins.qwen3_0_6b
                Expect.equal pin.Revision (m.GetProperty("revision").GetString()) "revision"
                Expect.equal pin.Model (m.GetProperty("id").GetString()) "model id"

                for p in m.GetProperty("sha256").EnumerateObject() do
                    let pinned =
                        pin.Files
                        |> List.tryFind (fun (n, _) -> n = p.Name)
                        |> Option.map (snd >> _.Hex)

                    Expect.equal pinned (Some(p.Value.GetString())) p.Name
        ]
