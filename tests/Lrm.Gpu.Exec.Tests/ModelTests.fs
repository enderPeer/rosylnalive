/// The model tier (local; Local\LrmModelCpu): Qwen3-0.6B on the executor over a CpuDevice, bit-identical to CpuOps F32
/// over a 4-token prefill and 3 decode steps (logits per step, the cache digest), and M2a's measured greedy ids. The
/// full-size plan has two weight pages, the tied head read as the embedding, and lm_head in its own Logits arena.
module Lrm.Gpu.Exec.Tests.ModelTests

open System
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Gpu.Core
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Gpu.Exec.Tests.Common

let private weights = lazy (Qwen3Weights.Open(LrmPaths.model "Qwen3-0.6B"))

let tests =
    testList
        "qwen3-0.6b"
        [
            testCase "a 4-token prefill and 3 decode steps on cpu-device = cpu-f32, bit for bit"
            <| fun () ->
                let w = weights.Force()
                let ids = [| 785; 3974; 13876; 38835 |]

                let options =
                    { OpsOptions.defaults w.Config with
                        Threads = Threads
                        MaxContext = 8
                        MaxRows = 4
                    }

                let run (ops: IForwardOps<'t>) =
                    use cache = KvCache.create ops w.Model (Stage.whole w.Config) 7
                    let logits = Collections.Generic.List<IHostTensor>()
                    let mutable next = ids
                    let mutable pos = 0
                    let generated = Collections.Generic.List<int>()

                    for _ in 0..3 do
                        let l = Forward.run ops w.Model cache pos next ForwardOptions.last
                        logits.Add(ops.Read l)
                        pos <- pos + next.Length
                        let id = lastArgmax ops l
                        generated.Add id
                        next <- [| id |]

                    List.ofSeq logits, generated.ToArray(), KvCache.digest ops cache

                use g =
                    GpuForwardOps.createWith
                        GpuExecOptions.Default
                        (cpuDevice Threads)
                        (Lrm.Shaders.Shaders.embedded ())
                        w
                        options

                Expect.equal g.Plan.Layout.PageCount 7 "two weight pages and one per other arena"
                let a, ia, da = run (g :> IForwardOps<GpuTensor>)
                use c = CpuOps.f32 w options
                let b, ib, db = run c
                Expect.equal ia ib "greedy ids"
                Expect.equal ia.[0..2] [| 34208; 916; 279 |] "M2a's measured ids after the prompt"
                Expect.equal da db "cache digests"

                for i, (x, y) in List.indexed (List.zip a b) do
                    expectSameBits (Invariant.format $"step {i} logits") x y
        ]
