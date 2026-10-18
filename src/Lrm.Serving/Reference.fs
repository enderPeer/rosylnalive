namespace Lrm.Serving

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Cluster
open Lrm.Models.Qwen

/// A workload run without a cluster: every unit in plan order on one executor that hosts every stage.
type PlanOrderRun =
    {
        Job: ContentId
        /// RunDigest.compute over the accepted units: what every cluster run of the job in this class must equal.
        RunDigest: ContentId
        /// The final head and digest of every chain, sorted by (lane, stage).
        Chains: ChainHead list
        /// RunDigest.replicas over Chains.
        ReplicaDigest: ContentId
        Units: int
        /// Every unit with its output id and state digest, in plan order.
        Accepted: AcceptedUnit list
    }

[<RequireQualifiedAccess>]
module PlanOrder =
    /// Runs every unit of `workload` in plan order (FIFO, as FakeReference does) on `executor`, which must host every
    /// stage. The chain states are disposed at the end.
    let run (workload: IWorkload) (executor: IStageExecutor) : PlanOrderRun =
        let states = Dictionary<ChainKey, IChainState * UnitId ref>()
        let queue = Queue<PlannedUnit>(workload.Start())
        let accepted = ResizeArray<AcceptedUnit>()

        try
            while queue.Count > 0 do
                let p = queue.Dequeue()
                let chain = WorkUnit.chain p.Unit

                let state, head =
                    match states.TryGetValue chain with
                    | true, s -> s
                    | _ ->
                        let s = executor.Open(workload.Job, chain), ref UnitId.none
                        states[chain] <- s
                        s

                let id = WorkUnit.id p.Unit

                if head.Value <> p.Unit.Prev then
                    failwithf "plan-order run: unit %s does not follow its chain head" id.Value.Short

                let output = state.Apply(p.Unit, p.Input)
                head.Value <- id

                accepted.Add
                    {
                        Unit = p.Unit
                        Id = id
                        Output = BlobId.ofBytes output
                        StateDigest = state.Digest
                    }

                for next in workload.Accept(p.Unit, output) do
                    queue.Enqueue next

            if not workload.IsComplete then
                failwith "plan-order run: the workload planned no more units but is not complete"

            let chains =
                states
                |> Seq.map (fun kv ->
                    {
                        Chain = kv.Key
                        Head = (snd kv.Value).Value
                        Digest = (fst kv.Value).Digest
                    })
                |> Seq.sortBy (fun h -> h.Chain.Lane, h.Chain.Stage)
                |> List.ofSeq

            {
                Job = workload.Job
                RunDigest = RunDigest.compute workload.Job accepted
                Chains = chains
                ReplicaDigest = RunDigest.replicas chains
                Units = accepted.Count
                Accepted = List.ofSeq accepted
            }
        finally
            for s in states.Values do
                (fst s).Dispose()

/// The oracles of a serving job (StagePipeline, "The reference"): the pipeline in plan order on one executor (whose
/// digests every cluster run must equal) and the whole model as one stage in one process on the canonical schedule
/// (whose logits and greedy ids the pipeline must equal bit for bit).
type QwenReference =
    {
        Job: ServingJob
        JobId: ContentId
        /// The pipeline in plan order.
        Pipeline: PlanOrderRun
        /// Its tokens, head outputs and logits digest.
        PipelineResult: ServingResult
        /// The single-process whole-model run.
        Single: ServingResult
    }

[<RequireQualifiedAccess>]
module QwenReference =
    /// The whole model as one stage on `ops` (one process, no cluster): per lane, a prefill of the prompt with the
    /// head on its last row, then single-token decode steps, each feeding back the argmax (ops.Argmax) of the previous
    /// step, stopped by the same rule as the workload. Every step's head output is its token and logits row, as the
    /// pipeline's head stage writes it. The ops' plan must include the whole model as a stage.
    let single (ops: IForwardOps<'t>) (model: Qwen3Model) (job: ServingJob) : ServingResult =
        let c = model.Config
        let stage = Stage.whole c

        if ServingJob.dtypeOf ops.ExecutionClass <> ops.DType then
            invalidArg "ops" "the ops' dtype is not their class's payload dtype"

        let lanes = ResizeArray<byte[] list * LaneEnd>()

        for lane in 0 .. job.Prompts.Length - 1 do
            let prompt = job.Prompts[lane]
            use cache = KvCache.create ops model stage (ServingJob.positions job lane)
            let outputs = ResizeArray<byte[]>()

            let step (s: int) (pos0: int) (ids: int[]) =
                let logits =
                    Forward.run
                        ops
                        model
                        cache
                        pos0
                        ids
                        ({
                            Head = HeadRows.Last
                            Taps = None
                            Step = Some s
                        }
                        : ForwardOptions<'t>)

                let token = (ops.Argmax logits)[0]
                let output = StagePayload.ofHead token ((ops.Read logits).ToBytes())
                outputs.Add output
                output

            let mutable output = step 0 0 prompt
            let mutable e = LaneRule.after job 0 output
            let mutable s = 1

            while e = LaneEnd.Running do
                let token = int (StagePayload.token output)
                output <- step s (prompt.Length + s - 1) [| token |]
                e <- LaneRule.after job s output
                s <- s + 1

            lanes.Add((List.ofSeq outputs, e))

        let lanes = List.ofSeq lanes

        ServingResult.ofOutputs (lanes |> List.map fst) (lanes |> List.map snd)

    /// Both oracles of `job` on a CPU backend (cpu-f32 or cpu-f64): the pipeline in plan order on one executor that
    /// hosts every stage, and the single-process whole-model run. `threads` CPU threads (the bits are the same for any
    /// count).
    let cpu (weights: Qwen3Weights) (job: ServingJob) (threads: int) : QwenReference =
        let c = weights.Config
        let backend = job.Class.Backend

        if job.Class <> ExecutionClass.forCpu backend then
            invalidArg "job" "the job's class is not this host's CPU class"

        let options =
            { StageExecutorOptions.defaults c with
                Boundaries = job.Boundaries
                Hosted = StageLayout.stages c.Layers job.Boundaries |> List.map _.Index
                MaxContext = max 1 (ServingJob.maxPositions job)
                MaxChains = 2 * job.Prompts.Length * (job.Boundaries.Length + 1)
                Threads = threads
            }

        let name = ExecutionBackend.name backend

        let factory =
            match QwenStageExecutor.tryBackend name with
            | Some b -> b
            | None -> invalidArg "job" (sprintf "no serving backend %s" name)

        let workload = QwenWorkload job

        let pipeline =
            use executor = factory.Create weights options
            PlanOrder.run workload executor

        let plan =
            { OpsOptions.defaults c with
                Threads = threads
                MaxContext = max 1 (ServingJob.maxPositions job)
                MaxRows = max 1 (ServingJob.maxRows job)
            }

        let single =
            match backend with
            | ExecutionBackend.CpuF64 ->
                use ops = CpuOps.f64 weights plan
                single ops weights.Model job
            | _ ->
                use ops = CpuOps.f32 weights plan
                single ops weights.Model job

        {
            Job = job
            JobId = ServingJob.id job
            Pipeline = pipeline
            PipelineResult = workload.Result
            Single = single
        }
