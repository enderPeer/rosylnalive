namespace Lrm.Cluster

open System
open System.Buffers.Binary
open System.Collections.Generic
open Lrm.Core

/// A fake pipeline job (ClusterProtocol contract, "Fake 3-stage pipeline"): Lanes independent sequences run Steps
/// autoregressive steps through Stages stages. Stage 0 at step 0 reads the lane's prompt; stage s+1 reads stage s's
/// output; stage 0 at step t+1 reads the last stage's output at step t (a 4-byte "token"). Every stage keeps a chain
/// state per lane, as a KV cache would, so the output of a unit depends on the whole history of its chain.
type FakeJob =
    {
        Seed: uint64
        Lanes: int
        Steps: int
        Stages: int
        /// Size of each lane's prompt in bytes.
        PromptBytes: int
        /// Output size of every stage but the last (the "residual"); the last stage outputs 4 bytes.
        HiddenBytes: int
    }

/// The fake stage function: SHA-256 only, so its results are the same on every host.
[<RequireQualifiedAccess>]
module FakeStage =
    [<Literal>]
    let OutTag = "lrm.fake.out/1"

    [<Literal>]
    let StateTag = "lrm.fake.state/1"

    [<Literal>]
    let InitTag = "lrm.fake.init/1"

    [<Literal>]
    let ExpandTag = "lrm.fake.expand/1"

    /// `length` bytes from a seed: the 32-byte blocks Canonical.hash "lrm.fake.expand/1" (seed, u32 block index) for
    /// blocks 0, 1, ..., concatenated and truncated.
    let expand (seed: ContentId) (length: int) : byte[] =
        if length < 0 then
            invalidArg "length" "negative length"

        let out = Array.zeroCreate<byte> length
        let mutable offset = 0
        let mutable block = 0u

        while offset < length do
            let index = block

            let h =
                Canonical.hash ExpandTag (fun w ->
                    w.Id seed
                    w.U32 index)

            let n = min 32 (length - offset)
            Array.blit (h.ToBytes()) 0 out offset n
            offset <- offset + n
            block <- block + 1u

        out

    /// The digest of an empty chain: Canonical.hash "lrm.fake.init/1" (job, lane, stage).
    let initialDigest (job: ContentId) (chain: ChainKey) =
        Canonical.hash InitTag (fun w ->
            w.Id job
            w.U32 chain.Lane
            w.U32 chain.Stage)

    /// The output size a fake unit asks for: its Args are one u32.
    let outputBytes (u: WorkUnit) =
        let r = CanonicalReader u.Args
        let n = r.U32()
        r.ExpectEnd()

        if n > uint32 WorkUnit.MaxData then
            raise (FormatException(sprintf "fake output size %d is too large" n))

        int n

    /// One step: h = Canonical.hash "lrm.fake.out/1" (job, lane, stage, step, digest, input bytes); the output is
    /// expand h (output size); the new digest is Canonical.hash "lrm.fake.state/1" (digest, step, input blob id,
    /// output blob id).
    let apply (digest: ContentId) (u: WorkUnit) (input: byte[]) : byte[] * ContentId =
        let h =
            Canonical.hash OutTag (fun w ->
                w.Id u.Job
                w.U32 u.Lane
                w.U32 u.Stage
                w.U32 u.Step
                w.Id digest
                w.Bytes input)

        let output = expand h (outputBytes u)
        let outputId = BlobId.ofBytes output

        let next =
            Canonical.hash StateTag (fun w ->
                w.Id digest
                w.U32 u.Step
                w.Id u.Input.Value
                w.Id outputId.Value)

        output, next

[<RequireQualifiedAccess>]
module FakeJob =
    [<Literal>]
    let JobTag = "lrm.fake.job/1"

    [<Literal>]
    let PromptTag = "lrm.fake.prompt/1"

    /// The job the chaos runs use: 3 lanes, 8 steps, 3 stages.
    let standard =
        {
            Seed = 1UL
            Lanes = 3
            Steps = 8
            Stages = 3
            PromptBytes = 64
            HiddenBytes = 256
        }

    /// The execution class of the fake executor. It names no real device and no runtime, because the fake stage
    /// uses only SHA-256 and integer operations, which give the same bits everywhere.
    let executionClass: ExecutionClass =
        {
            Backend = ExecutionBackend.CpuF32
            DeviceId = "fake"
            Driver = "lrm.fake/1"
            ShaderSetHash = None
            Arch = "any"
        }

    /// The class id of the fake executor.
    let classId = ExecutionClass.id executionClass

    /// The job when its sizes are in range.
    let validate (job: FakeJob) : Result<FakeJob, string> =
        if job.Lanes < 1 || job.Lanes > 4096 then
            Error "Lanes must be 1 to 4096"
        elif job.Steps < 1 || job.Steps > 1_000_000 then
            Error "Steps must be 1 to 1000000"
        elif job.Stages < 1 || job.Stages > 64 then
            Error "Stages must be 1 to 64"
        elif job.PromptBytes < 0 || job.PromptBytes > WorkUnit.MaxData then
            Error "PromptBytes out of range"
        elif job.HiddenBytes < 0 || job.HiddenBytes > WorkUnit.MaxData then
            Error "HiddenBytes out of range"
        else
            Ok job

    let private check (job: FakeJob) =
        match validate job with
        | Ok j -> j
        | Error e -> invalidArg "job" e

    /// Canonical.hash "lrm.fake.job/1" (Seed u64, then Lanes, Steps, Stages, PromptBytes, HiddenBytes as u32).
    let id (job: FakeJob) =
        let job = check job

        Canonical.hash JobTag (fun w ->
            w.U64 job.Seed
            w.U32(uint32 job.Lanes)
            w.U32(uint32 job.Steps)
            w.U32(uint32 job.Stages)
            w.U32(uint32 job.PromptBytes)
            w.U32(uint32 job.HiddenBytes))

    /// The prompt of a lane: expand (Canonical.hash "lrm.fake.prompt/1" (job id, lane)) PromptBytes.
    let prompt (job: FakeJob) (lane: int) =
        let jobId = id job

        FakeStage.expand
            (Canonical.hash PromptTag (fun w ->
                w.Id jobId
                w.U32(uint32 lane)))
            job.PromptBytes

    /// The Args of a fake unit: its output size as one u32 (HiddenBytes, or 4 at the last stage).
    let args (job: FakeJob) (stage: int) =
        let n = if stage = job.Stages - 1 then 4 else job.HiddenBytes
        Canonical.encode (fun w -> w.U32(uint32 n))

/// A fake chain state: just the running digest.
[<Sealed>]
type FakeChainState(job: ContentId, chain: ChainKey, corrupt: WorkUnit -> bool) =
    let mutable digest = FakeStage.initialDigest job chain

    interface IChainState with
        member _.Digest = digest

        member _.Apply(work: WorkUnit, input: byte[]) =
            let output, next = FakeStage.apply digest work input
            digest <- next

            if corrupt work && output.Length > 0 then
                // A faulty device: the output is wrong, the digest honest. Divergence detection must catch it.
                output[0] <- output[0] ^^^ 0xFFuy

            output

        member _.Dispose() = ()

/// The fake executor. `corrupt` marks units whose output this executor gets wrong (tripwire tests); `delay` runs
/// before each Apply, on the executing thread, to give real processes a cost.
[<Sealed>]
type FakeExecutor(canHost: uint32 list, corrupt: WorkUnit -> bool, delay: WorkUnit -> unit) =
    /// An honest executor that can host the given stages.
    new(canHost: uint32 list) = FakeExecutor(canHost, (fun _ -> false), ignore)

    interface IStageExecutor with
        member _.Class = FakeJob.executionClass
        member _.CanHost = canHost

        member _.Open(job: ContentId, chain: ChainKey) =
            let inner = new FakeChainState(job, chain, corrupt) :> IChainState

            { new IChainState with
                member _.Digest = inner.Digest

                member _.Apply(work, input) =
                    delay work
                    inner.Apply(work, input)

                member _.Dispose() = inner.Dispose()
            }

/// The fake job's plan (IWorkload).
[<Sealed>]
type FakeWorkload(job: FakeJob) =
    let jobId = FakeJob.id job
    let stages = job.Stages
    let lastPlanned = Dictionary<ChainKey, UnitId>()
    let tokens = Array.init job.Lanes (fun _ -> ResizeArray<uint32>())
    let mutable lanesLeft = job.Lanes

    let planned (lane: int) (stage: int) (step: int) (input: byte[]) : PlannedUnit =
        let chain =
            {
                Lane = uint32 lane
                Stage = uint32 stage
            }

        let prev =
            match lastPlanned.TryGetValue chain with
            | true, p -> p
            | _ -> UnitId.none

        let u =
            WorkUnit.fwdStage
                jobId
                FakeJob.classId
                (uint32 lane)
                (uint32 stage)
                (uint32 step)
                prev
                (BlobId.ofBytes input)
                (FakeJob.args job stage)

        lastPlanned[chain] <- WorkUnit.id u
        { Unit = u; Input = input }

    /// The tokens of each lane accepted so far.
    member _.Tokens = tokens |> Array.map List.ofSeq

    interface IWorkload with
        member _.Job = jobId
        member _.Class = FakeJob.classId
        member _.Stages = uint32 stages

        member _.Start() =
            [ for lane in 0 .. job.Lanes - 1 -> planned lane 0 0 (FakeJob.prompt job lane) ]

        member _.Accept(work: WorkUnit, output: byte[]) =
            let lane = int work.Lane
            let stage = int work.Stage
            let step = int work.Step

            if stage < stages - 1 then
                [ planned lane (stage + 1) step output ]
            else
                tokens[lane].Add(BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan output))

                if step + 1 < job.Steps then
                    [ planned lane 0 (step + 1) output ]
                else
                    lanesLeft <- lanesLeft - 1
                    []

        member _.IsComplete = lanesLeft = 0

/// The result of running a fake job directly, without a cluster: the oracle every cluster run must match.
type FakeReference =
    {
        Job: ContentId
        RunDigest: ContentId
        /// The final head and digest of every chain, sorted by (lane, stage).
        Chains: ChainHead list
        ReplicaDigest: ContentId
        Tokens: uint32 list list
        Units: int
    }

[<RequireQualifiedAccess>]
module FakeReference =
    /// Runs every unit of the job in plan order on one honest executor.
    let run (job: FakeJob) : FakeReference =
        let workload = FakeWorkload job
        let w = workload :> IWorkload
        let executor = FakeExecutor([ 0u .. uint32 job.Stages - 1u ]) :> IStageExecutor
        let states = Dictionary<ChainKey, IChainState * UnitId ref>()
        let queue = Queue<PlannedUnit>(w.Start())
        let accepted = ResizeArray<AcceptedUnit>()

        while queue.Count > 0 do
            let p = queue.Dequeue()
            let chain = WorkUnit.chain p.Unit

            let state, head =
                match states.TryGetValue chain with
                | true, s -> s
                | _ ->
                    let s = executor.Open(w.Job, chain), ref UnitId.none
                    states[chain] <- s
                    s

            let id = WorkUnit.id p.Unit

            if head.Value <> p.Unit.Prev then
                failwithf "reference run: unit %s does not follow its chain head" id.Value.Short

            let output = state.Apply(p.Unit, p.Input)
            head.Value <- id

            accepted.Add
                {
                    Unit = p.Unit
                    Id = id
                    Output = BlobId.ofBytes output
                    StateDigest = state.Digest
                }

            for next in w.Accept(p.Unit, output) do
                queue.Enqueue next

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
            Job = w.Job
            RunDigest = RunDigest.compute w.Job accepted
            Chains = chains
            ReplicaDigest = RunDigest.replicas chains
            Tokens = List.ofArray workload.Tokens
            Units = accepted.Count
        }
