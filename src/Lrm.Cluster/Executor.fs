namespace Lrm.Cluster

open System
open System.Collections.Generic
open Lrm.Core

/// The state of one chain on one node, for example a stage's KV cache for one sequence (ClusterProtocol contract,
/// "Executor interface"). A node calls Apply only with the unit whose Prev is the chain's head, so the state always
/// follows the hash chain; replay is Apply over the logged units in order.
type IChainState =
    inherit IDisposable

    /// The digest of the state after the last Apply, or the initial digest. It must be a deterministic function of
    /// the units applied (within one execution class), and it must change whenever an output changes, because
    /// replica digests are compared to detect divergence.
    abstract Digest: ContentId

    /// Applies a unit to the state and returns the unit's output bytes.
    abstract Apply: work: WorkUnit * input: byte[] -> byte[]

/// What a node executes: the stages it can host and a fresh chain state per (job, chain). C2's QwenStageExecutor
/// implements this for the Qwen pipeline; FakeExecutor is the 3-stage test executor.
type IStageExecutor =
    /// The execution class of everything this executor computes.
    abstract Class: ExecutionClass
    /// The stages this executor can host.
    abstract CanHost: uint32 list
    /// A fresh state for one chain of a job, at no head.
    abstract Open: job: ContentId * chain: ChainKey -> IChainState

/// A unit the workload wants run, with its input bytes (BlobId.ofBytes Input = Unit.Input).
type PlannedUnit = { Unit: WorkUnit; Input: byte[] }

/// The coordinator's source of units: a deterministic plan that unlocks units as results are accepted.
type IWorkload =
    /// The job id every unit carries.
    abstract Job: ContentId
    /// The execution class every unit carries.
    abstract Class: ExecutionClassId
    /// The number of pipeline stages (at least 1).
    abstract Stages: uint32
    /// The units that can start at once.
    abstract Start: unit -> PlannedUnit list
    /// Called exactly once per unit, with its first accepted output; returns the units that become ready.
    abstract Accept: work: WorkUnit * output: byte[] -> PlannedUnit list
    /// True when every unit of the job has been accepted.
    abstract IsComplete: bool

/// One accepted unit, as the run digest sees it.
type AcceptedUnit =
    {
        Unit: WorkUnit
        Id: UnitId
        Output: BlobId
        StateDigest: ContentId
    }

/// Digests of a whole run (ClusterProtocol contract). They hold only content (unit ids, output ids, state digests),
/// never times, nodes or schedules, so a chaos run and a fault-free run of one job in one class give the same bits.
[<RequireQualifiedAccess>]
module RunDigest =
    /// The tag of the run digest.
    [<Literal>]
    let Tag = "lrm.cluster.run/1"

    /// The tag of the replica digest.
    [<Literal>]
    let ReplicasTag = "lrm.cluster.replicas/1"

    /// Canonical.hash "lrm.cluster.run/1": the job id (32 raw bytes), then a sequence of the accepted units sorted by
    /// (lane, step, stage), each written as lane, step, stage (u32), unit id, output blob id and state digest (32 raw
    /// bytes each).
    let compute (job: ContentId) (units: AcceptedUnit seq) : ContentId =
        let sorted =
            units
            |> Seq.sortBy (fun a -> a.Unit.Lane, a.Unit.Step, a.Unit.Stage)
            |> Array.ofSeq

        Canonical.hash Tag (fun w ->
            w.Id job

            w.Seq(
                sorted,
                fun a ->
                    w.U32 a.Unit.Lane
                    w.U32 a.Unit.Step
                    w.U32 a.Unit.Stage
                    w.Id a.Id.Value
                    w.Id a.Output.Value
                    w.Id a.StateDigest
            ))

    /// Canonical.hash "lrm.cluster.replicas/1": a sequence of chain heads sorted by (lane, stage), each written as
    /// lane, stage (u32), head unit id and state digest (32 raw bytes each). Two replicas that hold the same chains at
    /// the same heads have the same replica digest.
    let replicas (heads: ChainHead seq) : ContentId =
        let sorted =
            heads |> Seq.sortBy (fun h -> h.Chain.Lane, h.Chain.Stage) |> Array.ofSeq

        Canonical.hash ReplicasTag (fun w ->
            w.Seq(
                sorted,
                fun h ->
                    w.U32 h.Chain.Lane
                    w.U32 h.Chain.Stage
                    w.Id h.Head.Value
                    w.Id h.Digest
            ))

/// The Ready-gated deadline rule (ClusterProtocol contract).
type DeadlinePolicy =
    {
        /// The deadline until MinSamples completion times exist: 30 s.
        InitialNs: int64
        /// The clamp bounds: 250 ms and 30 s.
        MinNs: int64
        MaxNs: int64
        /// deadline = clamp(Factor * p95 + MarginNs, MinNs, MaxNs): 4 and 50 ms.
        Factor: int64
        MarginNs: int64
        /// Samples needed before the adaptive rule applies: 8.
        MinSamples: int
        /// The most recent samples kept per stage: 64.
        Window: int
    }

[<RequireQualifiedAccess>]
module DeadlinePolicy =
    /// The contract values.
    let standard =
        {
            InitialNs = Ns.seconds 30L
            MinNs = Ns.ms 250L
            MaxNs = Ns.seconds 30L
            Factor = 4L
            MarginNs = Ns.ms 50L
            MinSamples = 8
            Window = 64
        }

/// Completion times of one stage and the deadline they imply. Samples run from the moment a deadline starts (the
/// node is Ready and positioned) to the accepted result; p95 is the nearest-rank 95th percentile of the window.
[<Sealed>]
type DeadlineEstimator(policy: DeadlinePolicy) =
    let samples = Queue<int64>()

    /// The number of samples in the window.
    member _.Count = samples.Count

    /// Adds one completion time in ns (negative values count as 0).
    member _.Add(ns: int64) =
        samples.Enqueue(max 0L ns)

        while samples.Count > policy.Window do
            samples.Dequeue() |> ignore

    /// The nearest-rank 95th percentile of the window: the ceil(0.95 n)-th smallest sample; 0 when empty.
    member _.P95 =
        if samples.Count = 0 then
            0L
        else
            let sorted = samples.ToArray()
            Array.sortInPlace sorted
            let n = sorted.Length
            let rank = (95 * n + 99) / 100
            sorted[rank - 1]

    /// The deadline in ns: InitialNs until MinSamples samples exist, then clamp(Factor * p95 + MarginNs, MinNs, MaxNs).
    member e.Current =
        if samples.Count < policy.MinSamples then
            policy.InitialNs
        else
            let raw = policy.Factor * e.P95 + policy.MarginNs
            min policy.MaxNs (max policy.MinNs raw)
