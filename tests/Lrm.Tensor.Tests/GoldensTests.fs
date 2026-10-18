/// ComparatorsAndGoldenReader: Goldens.Open validates manifests and verifies every file before use.
module Lrm.Tensor.Tests.GoldensTests

open System
open System.IO
open System.Text.Json
open System.Text.Json.Nodes
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.Tests.Common

/// A minimal golden set in a temporary tree: <tmp>/goldens/<set>/<version>/ with the given repo files and a manifest
/// that lists them (`edit` can change the manifest before it is written). Returns the set folder and the temp root.
let makeSet (set: string) (version: string) (files: (string * byte[]) list) (edit: JsonObject -> unit) =
    let root = tempDir "golden"
    let dir = Path.Combine(root, "goldens", set, version)
    Directory.CreateDirectory dir |> ignore
    let list = JsonArray()

    for name, bytes in files |> List.sortBy fst do
        let path = Path.Combine(dir, name)
        Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
        File.WriteAllBytes(path, bytes)
        let e = JsonObject()
        e["path"] <- JsonValue.Create(sprintf "goldens/%s/%s/%s" set version name)
        e["location"] <- JsonValue.Create "repo"
        e["bytes"] <- JsonValue.Create bytes.LongLength
        e["sha256"] <- JsonValue.Create (Hash.sha256 bytes).Hex
        e["role"] <- JsonValue.Create "test"
        list.Add e

    let m = JsonObject()
    m["schema"] <- JsonValue.Create "lrm.golden/1"
    m["set"] <- JsonValue.Create set
    m["version"] <- JsonValue.Create version
    m["kind"] <- JsonValue.Create "test"
    m["files"] <- list
    edit m
    File.WriteAllText(Path.Combine(dir, "manifest.json"), m.ToJsonString())
    dir, root

let private withSet set version files edit (f: string -> string -> unit) =
    let dir, root = makeSet set version files edit

    try
        f dir root
    finally
        Directory.Delete(root, true)

let private expectFormat (label: string) (edit: JsonObject -> unit) =
    withSet "s" "g1" [ "a.bin", [| 1uy |]; "b.bin", [| 2uy |] ] edit (fun dir _ ->
        Expect.throwsT<GoldenFormatException> (fun () -> Goldens.Open dir |> ignore) label)

let private firstFile (m: JsonObject) = (m["files"].AsArray()[0]).AsObject()

let tests =
    testList
        "goldens"
        [
            testCase "the tiny set opens and every listed file verifies"
            <| fun () ->
                let set = tiny.Force()
                Expect.equal (set.Set, set.Version, set.Kind) ("tiny", "g1", "tiny-model") "identity"
                Expect.equal set.Files.Length 15 "15 files"
                let (), records = withSink set.VerifyAll
                let verified = records |> List.filter (fun r -> r.Name = "goldens.verify")
                Expect.equal verified.Length 15 "one goldens.verify per file"
                Expect.isTrue (verified |> List.forall (fun r -> r.Value = 1.0)) "all passed"

            testCase "the manifest's tap names are the ActivationNames order"
            <| fun () ->
                let names =
                    tiny.Force().Manifest.GetProperty("taps").GetProperty("names").EnumerateArray()
                    |> Seq.map _.GetString()
                    |> Array.ofSeq

                Expect.sequenceEqual (ActivationNames.all 2) names "39 names"
                Expect.equal (ActivationNames.layer "L11.attn.q_rope") 11 "layer"
                Expect.equal (ActivationNames.kind "L11.attn.q_rope") "attn.q_rope" "kind"
                Expect.equal (ActivationNames.layer "logits") -1 "model level"
                Expect.equal (ActivationNames.kind "final_norm.out") "final_norm.out" "model-level kind"

            testCase "a changed byte or length is caught before use"
            <| fun () ->
                let src = tiny.Force()
                let bytes = File.ReadAllBytes(Path.Combine(src.Dir, "rope.safetensors"))

                withSet "tiny" "g1" [ "rope.safetensors", bytes ] ignore (fun dir _ ->
                    let set = Goldens.Open dir
                    (set.OpenSafetensors "rope.safetensors" :> IDisposable).Dispose()
                    let path = Path.Combine(dir, "rope.safetensors")
                    let flipped = Array.copy bytes
                    flipped[flipped.Length - 1] <- flipped[flipped.Length - 1] ^^^ 1uy
                    File.WriteAllBytes(path, flipped)

                    let (), records =
                        withSink (fun () ->
                            Expect.throwsT<IntegrityException>
                                (fun () -> set.OpenSafetensors("rope.safetensors").Dispose())
                                "hash")

                    Expect.equal
                        (records |> List.map (fun r -> r.Name, r.Value))
                        [ "goldens.verify", 0.0 ]
                        "failed check"

                    Expect.throwsT<IntegrityException>
                        (fun () -> set.ReadAllBytes "rope.safetensors" |> ignore)
                        "bytes"

                    File.WriteAllBytes(path, Array.append bytes [| 0uy |])

                    Expect.throwsT<IntegrityException>
                        (fun () -> set.OpenSafetensors("rope.safetensors").Dispose())
                        "size")

            testCase "data files resolve under the data root and verify the same way"
            <| fun () ->
                let payload = Text.Encoding.ASCII.GetBytes "large golden"

                withSet
                    "tiny"
                    "g9"
                    [ "a.bin", [| 1uy |] ]
                    (fun m ->
                        let e = JsonObject()
                        e["path"] <- JsonValue.Create "tiny/g9/big.bin"
                        e["location"] <- JsonValue.Create "data"
                        e["bytes"] <- JsonValue.Create payload.LongLength
                        e["sha256"] <- JsonValue.Create (Hash.sha256 payload).Hex
                        e["role"] <- JsonValue.Create "big"
                        m["files"].AsArray().Add e)
                    (fun dir root ->
                        let dataRoot = Path.Combine(root, "data")
                        let target = Path.Combine(dataRoot, "golden", "tiny", "g9", "big.bin")
                        Directory.CreateDirectory(Path.GetDirectoryName target) |> ignore
                        File.WriteAllBytes(target, payload)
                        let set = Goldens.Open(dir, dataRoot)
                        Expect.equal (set.File "big.bin").Location GoldenLocation.Data "location"
                        Expect.sequenceEqual (set.ReadAllBytes "big.bin") payload "verified bytes")

            testList
                "malformed manifests are refused"
                [
                    testCase "unknown schema" (fun () ->
                        expectFormat "schema" (fun m -> m["schema"] <- JsonValue.Create "lrm.golden/2"))
                    testCase "set name differs from the folder" (fun () ->
                        expectFormat "set" (fun m -> m["set"] <- JsonValue.Create "other"))
                    testCase "version differs from the folder" (fun () ->
                        expectFormat "version" (fun m -> m["version"] <- JsonValue.Create "g2"))
                    testCase "files not sorted" (fun () ->
                        expectFormat "order" (fun m ->
                            let a = m["files"].AsArray()
                            let first = a[0].DeepClone()
                            a.RemoveAt 0
                            a.Add first))
                    testCase "a path outside the set" (fun () ->
                        expectFormat "escape" (fun m ->
                            (firstFile m)["path"] <- JsonValue.Create "goldens/s/g1/../x.bin"))
                    testCase "a path under another set" (fun () ->
                        expectFormat "prefix" (fun m -> (firstFile m)["path"] <- JsonValue.Create "goldens/t/g1/a.bin"))
                    testCase "the manifest listing itself" (fun () ->
                        expectFormat "self" (fun m ->
                            (firstFile m)["path"] <- JsonValue.Create "goldens/s/g1/manifest.json"))
                    testCase "an unknown location" (fun () ->
                        expectFormat "location" (fun m -> (firstFile m)["location"] <- JsonValue.Create "web"))
                    testCase "an uppercase hash" (fun () ->
                        expectFormat "sha" (fun m ->
                            let f = firstFile m
                            f["sha256"] <- JsonValue.Create(f["sha256"].GetValue<string>().ToUpperInvariant())))
                    testCase "a negative size" (fun () ->
                        expectFormat "bytes" (fun m -> (firstFile m)["bytes"] <- JsonValue.Create -1))
                    testCase "a lone surrogate escape in a string" (fun () ->
                        withSet "s" "g1" [ "a.bin", [| 1uy |] ] ignore (fun dir _ ->
                            let path = Path.Combine(dir, "manifest.json")
                            let text = File.ReadAllText path

                            let edited =
                                text.Replace("\"kind\":\"test\"", "\"kind\":\"t" + string (char 92) + "ud800\"")

                            Expect.notEqual edited text "the manifest was edited"
                            File.WriteAllText(path, edited)

                            Expect.throwsT<GoldenFormatException>
                                (fun () -> Goldens.Open dir |> ignore)
                                "GoldenFormatException, not InvalidOperationException"))
                ]
        ]
