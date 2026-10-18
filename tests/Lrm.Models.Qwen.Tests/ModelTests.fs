/// Model tier (local only): the real Qwen3-0.6B under LRM_DATA opens through Qwen3Weights.Open (pinned revision, every
/// SHA256SUMS file, lm_head equal to the embedding by hash) and runs a short forward on both CPU classes, where prefill
/// equals decode. The full-model parity against G2's goldens is M2b's.
module Lrm.Models.Qwen.Tests.ModelTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

let private dir () = LrmPaths.model "Qwen3-0.6B"

/// Opened once for the tier (the open hashes 1.5 GB).
let private weights = lazy (Qwen3Weights.Open(dir ()))

/// Prompt ids of the real vocabulary (no tokenizer needed): arbitrary ids below 151,643.
let private prompt = [| 785; 3974; 13876; 38835 |]

/// The plan of these short runs: 8 positions and rows on the whole model, two caches.
let private options () =
    { OpsOptions.defaults weights.Value.Config with
        Threads = Threads
        MaxContext = 8
        MaxRows = 8
        Caches = 2
    }

let private shortForward (ops: IForwardOps<'t>) =
    let w = weights.Value
    let model = w.Model
    let n = prompt.Length

    use c1 = KvCache.create ops model (Stage.whole model.Config) n
    let all = ops.Read(Forward.run ops model c1 0 prompt ForwardOptions.all)
    Expect.equal all.Desc.Shape (Shape.matrix n 151936) "logits"
    Expect.isTrue (all.ToFloat64() |> Array.forall Double.IsFinite) "finite logits"
    use cache = KvCache.create ops model (Stage.whole model.Config) n

    for t in 0 .. n - 1 do
        let step =
            ops.Read(Forward.run ops model cache t [| prompt[t] |] ForwardOptions.last)

        expectSameBits
            (Invariant.format $"{ExecutionBackend.name ops.ExecutionClass.Backend} decode row {t}")
            step
            (rowsOf all t (t + 1))

let tests =
    testList
        "qwen3-0.6b"
        [
            testCase "opens at the pinned revision: every SHA256SUMS file verified, lm_head equal to the embedding"
            <| fun () ->
                let w = weights.Value
                Expect.equal w.Revision (ModelRevision.HuggingFace Qwen3Pins.qwen3_0_6b.Revision) "revision"
                Expect.equal w.LmHead LmHeadSource.AliasVerified "lm_head is the embedding, by SHA-256"
                Expect.equal w.Params.LmHead Qwen3Params.EmbedName "one tensor"
                Expect.equal w.Names.Length 310 "310 tensors read (lm_head aliased)"
                Expect.equal w.Sums.Length 10 "SHA256SUMS lines"

                for name, pinned in Qwen3Pins.qwen3_0_6b.Files do
                    Expect.equal (w.Sums |> List.find (fun (n, _) -> n = name) |> snd) pinned name

                Expect.equal w.Config (Qwen3Config.parse ConfigTests.qwenConfigText) "config"

            testCase "cpu-f32: a 4-token prefill equals 4 decode steps, bit for bit"
            <| fun () ->
                use ops = CpuOps.f32 weights.Value (options ())
                shortForward ops

            testCase "cpu-f64: a 4-token prefill equals 4 decode steps, bit for bit"
            <| fun () ->
                use ops = CpuOps.f64 weights.Value (options ())
                shortForward ops

            testCase "greedy decoding gives the same ids twice and on the runner"
            <| fun () ->
                let w = weights.Value
                use ops = CpuOps.f32 w (options ())
                let a = Greedy.run ops w.Model prompt (GreedyOptions.parity 3)
                use runner = CpuBackends.cpuF32.Create w (options ())
                let b = runner.Greedy(prompt, GreedyOptions.parity 3)

                Expect.sequenceEqual a.Ids b.Ids "same ids"
                Expect.equal a.Ids.Length 3 "3 ids"
                printfn "Qwen3-0.6B cpu-f32 greedy after %A: %A" prompt a.Ids
        ]
