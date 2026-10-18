/// QwenStageExecutor on the tiny model: the pipeline equals the single-process run bit for bit, the digest rule, the
/// executor's refusals, its chain limit and its ops lock.
module Lrm.Serving.Tests.ExecutorTests

open System
open System.Threading
open System.Threading.Tasks
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Models.Qwen
open Lrm.Serving
open Lrm.Serving.Tests.Common

let private stageChain (lane: int) (stage: int) =
    {
        Lane = uint32 lane
        Stage = uint32 stage
    }

let tests =
    testList
        "executor"
        [
            testCase "two stages in plan order equal the whole model in one process (cpu-f32): logits and greedy ids"
            <| fun () ->
                let r = referenceOf tinySpec
                Expect.equal r.PipelineResult.LogitsDigest r.Single.LogitsDigest "logits digest"
                Expect.equal r.PipelineResult.HeadOutputs r.Single.HeadOutputs "every head output, bit for bit"
                Expect.equal r.PipelineResult.Tokens r.Single.Tokens "greedy ids"
                Expect.equal r.Pipeline.Units 96 "3 lanes x 16 steps x 2 stages"

                Expect.equal
                    (r.Single.Ends |> List.map LaneEnd.name)
                    [ "done"; "done"; "done" ]
                    "every lane ran 16 steps"

            testCase "two stages in plan order equal the whole model in one process (cpu-f64)"
            <| fun () ->
                let r = referenceOf tinySpec64
                Expect.equal r.PipelineResult.LogitsDigest r.Single.LogitsDigest "logits digest"
                Expect.equal r.PipelineResult.Tokens r.Single.Tokens "greedy ids"
                Expect.notEqual r.JobId (referenceOf tinySpec).JobId "another class, another job"

            testCase "the greedy ids are M2a's Greedy.run ids and the golden ones"
            <| fun () ->
                let r = referenceOf tinySpec
                let job = jobOf tinySpec
                let w = tinyWeights ()

                use ops =
                    CpuOps.f32
                        w
                        { OpsOptions.defaults w.Config with
                            Threads = 2
                            MaxContext = ServingJob.maxPositions job
                            MaxRows = ServingJob.maxRows job
                        }

                for lane, prompt in List.indexed job.Prompts do
                    let g = Greedy.run ops w.Model prompt (GreedyOptions.parity job.Steps)
                    Expect.equal r.Single.Tokens[lane] (List.ofArray g.Ids) (sprintf "lane %d = Greedy.run" lane)

                for lane, p in [ 0, "p0"; 1, "p1" ] do
                    Expect.equal
                        r.Single.Tokens[lane]
                        (goldenGreedy p "torch32" |> Array.truncate job.Steps |> List.ofArray)
                        (sprintf "%s = torch32's first 16 ids" p)

            testCase "the digest changes with every unit and with any output, and stays changed"
            <| fun () ->
                let job = jobOf tinySpec
                let chains = plannedChains job
                let jobId = ServingJob.id job
                use ex = executor "cpu-f32" (executorOptions job)

                for KeyValue(chain, units) in chains do
                    use honest = ex.Open(jobId, chain)
                    use other = ex.Open(jobId, chain)
                    let seen = Collections.Generic.HashSet<ContentId>([ honest.Digest ])

                    for work, input in units do
                        honest.Apply(work, input) |> ignore
                        other.Apply(work, input) |> ignore
                        Expect.isTrue (seen.Add honest.Digest) (sprintf "%A step %d: a new digest" chain work.Step)
                        Expect.equal other.Digest honest.Digest "two states of the same units agree"

                // A chain whose first stage-1 input differs in one bit: its outputs and digests differ from there on.
                let units = chains[stageChain 2 1]
                use good = ex.Open(jobId, stageChain 2 1)
                use bad = ex.Open(jobId, stageChain 2 1)

                for i, (work, input) in List.indexed units do
                    let flipped =
                        if i = 0 then
                            // The sign bit of the residual's first element.
                            let x = Array.copy input
                            x[3] <- x[3] ^^^ 0x80uy
                            x
                        else
                            input

                    let badWork =
                        if i = 0 then
                            { work with
                                Input = BlobId.ofBytes flipped
                            }
                        else
                            work

                    let a = good.Apply(work, input)

                    // Later units name the honest chain's Prev; the bad state follows its own (it is a test of the
                    // digest, so the state does not check Prev itself).
                    let b = bad.Apply(badWork, flipped)

                    if i = 0 then
                        Expect.notEqual b a "the output of the changed unit differs"

                    Expect.notEqual bad.Digest good.Digest (sprintf "step %d: the digest stays changed" i)

            testCase "the digest commits to the cache: equal KvCache digests after the same units, on two executors"
            <| fun () ->
                let job = jobOf tinySpec
                let chains = plannedChains job
                let jobId = ServingJob.id job
                use a = executor "cpu-f32" (executorOptions job)
                use b = executor "cpu-f32" { executorOptions job with Threads = 1 }

                for KeyValue(chain, units) in chains do
                    use x = a.Open(jobId, chain)
                    use y = b.Open(jobId, chain)

                    for work, input in units do
                        Expect.equal (y.Apply(work, input)) (x.Apply(work, input)) "outputs at 2 and 1 threads"

                    Expect.equal y.Digest x.Digest "digests"

                    Expect.equal
                        (QwenStageExecutor.cacheDigest b y)
                        (QwenStageExecutor.cacheDigest a x)
                        "KvCache.digest"

                    Expect.isSome (QwenStageExecutor.cacheDigest a x) "a cache is held"

            testCase "the executor refuses units of another job, model, stage, class or position, and bad inputs"
            <| fun () ->
                let job = jobOf tinySpec
                let chains = plannedChains job
                let jobId = ServingJob.id job
                let work0, input0 = chains[stageChain 0 0].Head
                let work1, input1 = chains[stageChain 0 0][1]

                use ex =
                    executor
                        "cpu-f32"
                        { executorOptions job with
                            Hosted = [ 0 ]
                        }

                Expect.equal (ex :> IStageExecutor).CanHost [ 0u ] "CanHost lists only the hosted stage"
                let refused (label: string) (f: unit -> unit) = Expect.throwsT<StageException> f label

                use s = ex.Open(jobId, stageChain 0 0)
                refused "another job" (fun () -> s.Apply({ work0 with Job = ContentId.zero }, input0) |> ignore)
                refused "another chain" (fun () -> s.Apply({ work0 with Lane = 1u }, input0) |> ignore)

                refused "another class" (fun () -> s.Apply({ work0 with Class = FakeJob.classId }, input0) |> ignore)

                refused "input bytes that are not the unit's" (fun () -> s.Apply(work0, input1) |> ignore)

                let otherModel =
                    { StageArgs.decode work0.Args with
                        Model = ContentId.zero
                    }

                refused "another model" (fun () ->
                    s.Apply(
                        { work0 with
                            Args = StageArgs.encode otherModel
                        },
                        input0
                    )
                    |> ignore)

                refused "a position the chain does not hold" (fun () -> s.Apply(work1, input1) |> ignore)

                let stage1Work, stage1Input = chains[stageChain 0 1].Head
                use s1 = ex.Open(jobId, stageChain 0 1)
                refused "a stage not hosted" (fun () -> s1.Apply(stage1Work, stage1Input) |> ignore)

                // Nothing refused changed the state: the real units still apply in order.
                let digest = s.Digest
                s.Apply(work0, input0) |> ignore
                Expect.notEqual s.Digest digest "the first unit applies after the refusals"
                s.Apply(work1, input1) |> ignore
                s.Dispose()

                Expect.throwsT<ObjectDisposedException> (fun () -> s.Apply(work0, input0) |> ignore) "a disposed state"

            testCase "a unit beyond MaxContext is refused before the chain holds a cache"
            <| fun () ->
                let job = jobOf tinySpec
                let chains = plannedChains job
                let jobId = ServingJob.id job

                use ex =
                    executor
                        "cpu-f32"
                        { executorOptions job with
                            MaxContext = 12
                        }

                use s = ex.Open(jobId, stageChain 0 0)

                Expect.throwsT<StageException>
                    (fun () -> s.Apply(chains[stageChain 0 0].Head) |> ignore)
                    "p0's 13-row prefill does not fit 12 positions"

                Expect.equal ex.LiveChains 0 "no cache was allocated"

            testCase "MaxChains bounds the caches alive at once; disposing a state frees its cache"
            <| fun () ->
                let job = jobOf tinySpec
                let chains = plannedChains job
                let jobId = ServingJob.id job

                use ex =
                    executor
                        "cpu-f32"
                        { executorOptions job with
                            MaxChains = 2
                        }

                let first (lane: int) = chains[stageChain lane 0].Head
                let a = ex.Open(jobId, stageChain 0 0)
                let b = ex.Open(jobId, stageChain 1 0)
                let c = ex.Open(jobId, stageChain 2 0)
                a.Apply(first 0) |> ignore
                b.Apply(first 1) |> ignore
                Expect.equal ex.LiveChains 2 "two caches"
                Expect.throwsT<StageException> (fun () -> c.Apply(first 2) |> ignore) "a third chain is refused"
                a.Dispose()
                Expect.equal ex.LiveChains 1 "a's cache is released"
                c.Apply(first 2) |> ignore
                Expect.equal ex.LiveChains 2 "c holds one now"
                b.Dispose()
                c.Dispose()
                Expect.equal ex.LiveChains 0 "every cache released"

            testCase "chain states opened and disposed on another thread while units run stay consistent"
            <| fun () ->
                let job = jobOf tinySpec
                let chains = plannedChains job
                let jobId = ServingJob.id job
                use ex = executor "cpu-f32" (executorOptions job)
                let units = chains[stageChain 2 0]
                let reference = ex.Open(jobId, stageChain 2 0)

                let expected =
                    [
                        for work, input in units do
                            reference.Apply(work, input) |> ignore
                            yield reference.Digest
                    ]

                reference.Dispose()
                use stop = new CancellationTokenSource()

                // The "actor thread": opens a state, runs its first unit, disposes it, over and over.
                let churn =
                    Task.Run(fun () ->
                        let work, input = chains[stageChain 0 0].Head

                        while not stop.IsCancellationRequested do
                            let s = ex.Open(jobId, stageChain 0 0)
                            s.Apply(work, input) |> ignore
                            s.Dispose()

                            let t = ex.Open(jobId, stageChain 1 0)
                            t.Dispose())

                for _ in 1..20 do
                    use s = ex.Open(jobId, stageChain 2 0)

                    let got =
                        [
                            for work, input in units do
                                s.Apply(work, input) |> ignore
                                yield s.Digest
                        ]

                    Expect.equal got expected "the same digests while other states come and go"

                stop.Cancel()
                churn.Wait()
                Expect.isTrue (ex.LiveChains <= 1) "released caches are returned"

            testCase "a cluster in one process (InProc transports, two nodes, one stage each) equals the plan-order run"
            <| fun () ->
                let job = jobOf tinySpec
                let r = referenceOf tinySpec
                let hub = InProcHub()
                let workload = QwenWorkload job

                let coordinator =
                    Coordinator(PipelineConfig.coordinator PipelineConfig.split2, workload, MemoryBlobStore())

                use coordHost = new ActorHost(coordinator, hub.Connect 0UL, SystemClock.Instance)

                coordHost.Start()

                let nodes =
                    [
                        for i, stage in [ 1, 0; 2, 1 ] ->
                            let ex =
                                executor
                                    "cpu-f32"
                                    { executorOptions job with
                                        Hosted = [ stage ]
                                    }

                            let node = Node(NodeConfig.create (uint64 i) ("n" + Invariant.i32 i) (uint64 i), ex)

                            let host = new ActorHost(node, hub.Connect(uint64 i), SystemClock.Instance)
                            host.Start()
                            ex, host
                    ]

                try
                    Expect.isTrue (coordinator.Completion.Wait LivenessMs) "finished within 60 s"
                    let o = ServingOutcome.ofRun coordinator.Completion.Result workload.Result
                    Expect.isEmpty (ServingOutcome.differences r o) (ServingOutcome.describe o)
                finally
                    for ex, host in nodes do
                        (host :> IDisposable).Dispose()
                        ex.Dispose()
        ]
