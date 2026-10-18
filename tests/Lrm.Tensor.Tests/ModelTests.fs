/// Model tier (local only): the real Qwen3-0.6B checkpoint under LRM_DATA opens through the validating reader.
module Lrm.Tensor.Tests.ModelTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Tensor

let private modelDir () = LrmPaths.model "Qwen3-0.6B"

let tests =
    testList
        "qwen3-0.6b"
        [
            testCase "model.safetensors opens with 311 BF16 entries, data at 35,560, and the writer rebuilds its header"
            <| fun () ->
                let path = Path.Combine(modelDir (), "model.safetensors")
                use f = SafetensorsFile.Open path
                Expect.equal f.Entries.Length 311 "entries"
                Expect.isTrue (f.Entries |> Array.forall (fun e -> e.Desc.DType = DType.BF16)) "all BF16"
                Expect.equal f.HeaderLength 35_552L "header length"
                Expect.equal f.DataOffset 35_560L "data offset"
                Expect.sequenceEqual f.Metadata [| "format", "pt" |] "metadata"
                Expect.isTrue f.IsAligned "aligned"
                Expect.equal (f.Entry "lm_head.weight").Desc.Shape (Shape.matrix 151_936 1024) "lm_head"
                Expect.sequenceEqual (f.RebuildHeader()) (f.HeaderBytes()) "header rebuilt byte for byte"

            testCase "model.safetensors hashes to its SHA256SUMS line, at the pinned revision"
            <| fun () ->
                let dir = modelDir ()
                let revision = File.ReadAllText(Path.Combine(dir, "REVISION")).Trim()
                Expect.equal revision "c1899de289a04d12100db370d81485cdf75e47ca" "revision"

                let expected =
                    File.ReadAllLines(Path.Combine(dir, "SHA256SUMS"))
                    |> Array.pick (fun l ->
                        let parts = l.Split([| ' ' |], 2, StringSplitOptions.RemoveEmptyEntries)

                        if parts.Length = 2 && parts[1].TrimStart('*') = "model.safetensors" then
                            Some parts[0]
                        else
                            None)

                use f = SafetensorsFile.Open(Path.Combine(dir, "model.safetensors"))
                Expect.equal (f.Sha256()).Hex expected "SHA-256"
        ]
