/// KV rebuild: exact replay of a chain's hash-chained units (what C1's node does with the unit log) gives back every
/// output and digest; the fast rebuild (one forward of many rows) equals replay bit for bit, on both CPU classes.
module Lrm.Serving.Tests.RebuildTests

open System
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Models.Qwen
open Lrm.Serving
open Lrm.Serving.Tests.Common

/// Every output and digest of applying `units` in order to a fresh state.
let private replay (ex: IQwenStageExecutor) (job: ContentId) (chain: ChainKey) (units: (WorkUnit * byte[]) list) =
    use s = ex.Open(job, chain)

    let steps =
        [
            for work, input in units do
                let output = s.Apply(work, input)
                yield output, s.Digest
        ]

    steps, QwenStageExecutor.cacheDigest ex s

/// The fast rebuild of `units` on a fresh state, its cache digest, and the state (for further units).
let private rebuild (ex: IQwenStageExecutor) (job: ContentId) (chain: ChainKey) (units: (WorkUnit * byte[]) list) =
    let s = ex.Open(job, chain)
    let steps = (s :?> IFastRebuild).Rebuild units
    steps, QwenStageExecutor.cacheDigest ex s, s

let private fastOptions (job: ServingJob) =
    { executorOptions job with
        FastRebuild = true
    }

let private checkEveryChain (spec: JobSpec) (chunk: int) =
    let job = jobOf spec
    let jobId = ServingJob.id job
    let chains = plannedChains job
    let backend = spec.Backend
    use replayer = executor backend (executorOptions job)

    use fast =
        executor
            backend
            { fastOptions job with
                RebuildChunkRows = chunk
            }

    for KeyValue(chain, units) in chains do
        let expected, expectedCache = replay replayer jobId chain units
        let got, gotCache, state = rebuild fast jobId chain units

        try
            Expect.equal got.Length expected.Length "one output per unit"

            for i in 0 .. got.Length - 1 do
                let (go, gd), (eo, ed) = got[i], expected[i]
                Expect.equal go eo (sprintf "%s chain %A step %d: output" backend chain i)
                Expect.equal gd ed (sprintf "%s chain %A step %d: digest" backend chain i)

            Expect.equal gotCache expectedCache (sprintf "%s chain %A: KvCache.digest" backend chain)
            Expect.isSome gotCache "a cache"
            Expect.equal state.Digest (snd (List.last expected)) "the state's digest"
        finally
            state.Dispose()

let tests =
    testList
        "rebuild"
        [
            testCase "exact replay of a chain's logged units gives every output and digest of the plan-order run"
            <| fun () ->
                let job = jobOf tinySpec
                let jobId = ServingJob.id job
                let r = referenceOf tinySpec
                use ex = executor "cpu-f32" (executorOptions job)

                let accepted =
                    r.Pipeline.Accepted |> List.map (fun a -> WorkUnit.id a.Unit, a) |> Map.ofList

                for KeyValue(chain, units) in plannedChains job do
                    let steps, _ = replay ex jobId chain units

                    for (work, _), (output, digest) in List.zip units steps do
                        let a = accepted[WorkUnit.id work]
                        Expect.equal (BlobId.ofBytes output) a.Output (sprintf "%A step %d output id" chain work.Step)
                        Expect.equal digest a.StateDigest (sprintf "%A step %d digest" chain work.Step)

                    let final = r.Pipeline.Chains |> List.find (fun c -> c.Chain = chain)
                    Expect.equal (snd (List.last steps)) final.Digest "the final digest"

            testCase "fast rebuild equals replay bit for bit on cpu-f32 (one forward per chain)"
            <| fun () -> checkEveryChain tinySpec 0

            testCase "fast rebuild equals replay bit for bit on cpu-f64"
            <| fun () -> checkEveryChain tinySpec64 0

            testCase "fast rebuild in chunks of 1, 7 and 16 rows equals replay"
            <| fun () ->
                for chunk in [ 1; 7; 16 ] do
                    checkEveryChain tinySpec chunk

            testCase "a rebuilt chain continues exactly like the replayed one"
            <| fun () ->
                let job = jobOf tinySpec
                let jobId = ServingJob.id job
                let chain = { Lane = 0u; Stage = 1u }
                let units = (plannedChains job)[chain]
                let head, tail = List.splitAt 9 units
                use ex = executor "cpu-f32" (fastOptions job)
                let expected, _ = replay ex jobId chain units
                let _, _, state = rebuild ex jobId chain head

                try
                    for i, (work, input) in List.indexed tail do
                        let output = state.Apply(work, input)
                        let eo, ed = expected[9 + i]
                        Expect.equal output eo (sprintf "step %d output" work.Step)
                        Expect.equal state.Digest ed (sprintf "step %d digest" work.Step)
                finally
                    state.Dispose()

            testCase "fast rebuild is refused when it is off, on a used state, and for units that are not a chain"
            <| fun () ->
                let job = jobOf tinySpec
                let jobId = ServingJob.id job
                let chain = { Lane = 1u; Stage = 0u }
                let units = (plannedChains job)[chain]
                use off = executor "cpu-f32" (executorOptions job)
                use on = executor "cpu-f32" (fastOptions job)
                let refused (label: string) (f: unit -> unit) = Expect.throwsT<StageException> f label

                use a = off.Open(jobId, chain)
                refused "the flag is off" (fun () -> (a :?> IFastRebuild).Rebuild units |> ignore)

                use b = on.Open(jobId, chain)
                b.Apply(units.Head) |> ignore
                refused "a state that applied a unit" (fun () -> (b :?> IFastRebuild).Rebuild units |> ignore)

                use c = on.Open(jobId, chain)
                let fresh = c.Digest

                refused "a gap in the chain" (fun () ->
                    (c :?> IFastRebuild).Rebuild(units.Head :: units[2..]) |> ignore)

                refused "not from step 0" (fun () -> (c :?> IFastRebuild).Rebuild units.Tail |> ignore)
                Expect.equal c.Digest fresh "a refused rebuild changes nothing"
                Expect.isNone (QwenStageExecutor.cacheDigest on c) "and allocates no cache"
                Expect.isEmpty ((c :?> IFastRebuild).Rebuild []) "no units, nothing to do"
        ]
