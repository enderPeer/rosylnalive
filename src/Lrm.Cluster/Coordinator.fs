namespace Lrm.Cluster

open System
open System.Collections.Generic
open System.Threading.Tasks
open Lrm.Core

/// Coordinator settings (ClusterProtocol contract values in CoordinatorConfig.standard).
type CoordinatorConfig =
    {
        /// Replicas per stage: every unit of a stage is assigned to all of them.
        Replicas: int
        /// Members needed before the run starts.
        MinNodes: int
        /// The heartbeat interval nodes are told to use: 100 ms.
        HeartbeatNs: int64
        /// The lease a heartbeat grants: 1 s.
        LeaseNs: int64
        /// How often the coordinator checks leases and deadlines: 50 ms.
        TickNs: int64
        /// The Ready-gated deadline rule.
        Deadline: DeadlinePolicy
        /// An assignment the node's heartbeat does not list is re-sent after this long: 500 ms.
        ResendNs: int64
        /// After the last unit, how long to wait for the remaining replicas to confirm the final states: 2 s.
        DrainNs: int64
        /// The divergence tripwire: stop the run at the first divergence.
        HaltOnDivergence: bool
        /// The largest Log batch, in encoded entry bytes (unit, input, output id, digest): 16 MiB. A batch holds at
        /// least one entry, and one entry always fits in a frame.
        LogBatchBytes: int
    }

[<RequireQualifiedAccess>]
module CoordinatorConfig =
    /// The contract values, with 2 replicas per stage.
    let standard =
        {
            Replicas = 2
            MinNodes = 1
            HeartbeatNs = Ns.ms 100L
            LeaseNs = Ns.seconds 1L
            TickNs = Ns.ms 50L
            Deadline = DeadlinePolicy.standard
            ResendNs = Ns.ms 500L
            DrainNs = Ns.seconds 2L
            HaltOnDivergence = true
            LogBatchBytes = 16 * 1024 * 1024
        }

/// Where a divergence was seen.
[<RequireQualifiedAccess>]
type DivergenceSource =
    /// Two results for one unit differ.
    | Result
    /// A replayed chain's digest differs from the accepted digest at the same head.
    | Replay

/// Two replicas of one class disagree about one unit: bit identity within a class is broken.
type Divergence =
    {
        Unit: UnitId
        Lane: uint32
        Stage: uint32
        Step: uint32
        FirstNode: uint64
        FirstOutput: BlobId
        FirstDigest: ContentId
        OtherNode: uint64
        /// None for a replay divergence (Ready carries only the digest).
        OtherOutput: BlobId option
        OtherDigest: ContentId
        Source: DivergenceSource
    }

/// How a run ended.
[<RequireQualifiedAccess>]
type RunStatus =
    /// Every unit was accepted, and every live replica that was due to confirm a chain's final state did.
    | Completed
    /// Every unit was accepted with no divergence, but when the drain ended some live replicas had still not
    /// confirmed a chain's final state (RunOutcome.Unconfirmed > 0): the cross-check is incomplete.
    | Unverified
    /// The divergence tripwire fired.
    | Diverged
    | Aborted of reason: string

[<RequireQualifiedAccess>]
module RunStatus =
    /// The status as the cluster.run event and the CLI print it: completed, unverified, diverged or aborted.
    let name (s: RunStatus) =
        match s with
        | RunStatus.Completed -> "completed"
        | RunStatus.Unverified -> "unverified"
        | RunStatus.Diverged -> "diverged"
        | RunStatus.Aborted _ -> "aborted"

/// The final state of one chain and the members that confirmed it.
type ChainFinal =
    {
        Final: ChainHead
        /// Members whose result or replay matched the final unit (the winner included), in id order.
        ConfirmedBy: uint64 list
        /// Live replicas that were still computing the final unit when the drain ended (laggards), in id order.
        PendingBy: uint64 list
        /// Live replicas that were due to confirm the final unit, were asked again, and never answered, in id order.
        UnconfirmedBy: uint64 list
    }

/// The result of a run.
type RunOutcome =
    {
        Status: RunStatus
        Job: ContentId
        /// RunDigest.compute over the accepted units: equal across every fault-free and chaos run of one job.
        RunDigest: ContentId
        /// RunDigest.replicas over the final chain heads.
        ReplicaDigest: ContentId
        Units: int
        /// The final epoch.
        Epoch: uint64
        Reissues: int
        Duplicates: int
        Divergences: Divergence list
        /// Fetch requests served (replays).
        Replays: int
        /// Frames rejected: stale epoch, unknown member, malformed.
        Rejected: int
        /// Accepted units that a second member confirmed, by an equal result or by a replay that recomputed them.
        CrossChecked: int
        /// Final chain states that were not verified when the run ended: fewer than two members computed them (one
        /// with Replicas = 1), or a live replica that was assigned the final unit on time neither confirmed it nor
        /// was still computing it. 0 when the status is Completed.
        Unconfirmed: int
        Chains: ChainFinal list
    }

[<Sealed; AllowNullLiteral>]
type internal MemberState
    (id: uint64, name: string, incarnation: uint64, serial: int64, cls: ExecutionClassId, canHost: Set<uint32>) =
    member _.Id = id
    member _.Name = name
    member _.Incarnation = incarnation
    /// Distinguishes memberships: every join, rejoin and restart gets a new serial, so obligations of an earlier
    /// membership (assignments, expected confirmations) end with it.
    member _.Serial = serial
    member _.Class = cls
    member _.CanHost = canHost
    member val LeaseUntil = 0L with get, set
    member val Ready = false with get, set
    /// The units the member's last current-epoch heartbeat listed as queued or running.
    member val Active = Set.empty<UnitId> with get, set

[<Sealed; AllowNullLiteral>]
type internal AssignmentState(node: uint64, serial: int64, now: int64) =
    member _.Node = node
    /// The membership (MemberState.Serial) the assignment and its deadline belong to.
    member val Serial = serial with get, set
    member val AssignedAt = now with get, set
    member val LastSentAt = now with get, set
    member val DeadlineStart = ValueOption<int64>.None with get, set
    member val DeadlineAt = Int64.MaxValue with get, set
    member val Misses = 0 with get, set

[<Sealed; AllowNullLiteral>]
type internal PendingState(index: int64, work: WorkUnit, id: UnitId, input: byte[]) =
    member _.Index = index
    member _.Unit = work
    member _.Id = id
    member _.Input = input
    member _.Chain = WorkUnit.chain work
    member val Assignments = SortedDictionary<uint64, AssignmentState>()

[<Sealed; AllowNullLiteral>]
type internal AcceptedState
    (work: WorkUnit, id: UnitId, output: BlobId, digest: ContentId, node: uint64, incarnation: uint64, at: int64) =
    member _.Unit = work
    member _.Id = id
    member _.Output = output
    member _.Digest = digest
    /// The member whose result was accepted, and its incarnation.
    member _.Node = node
    member _.NodeIncarnation = incarnation
    member _.AcceptedAt = at
    /// (member, membership serial) pairs that were assigned the unit and had not missed its deadline when it was
    /// accepted: the replicas whose confirmation of a final unit the drain waits for.
    member val Expected = Set.empty<uint64 * int64> with get, set
    /// (member, membership serial) pairs that were assigned the unit but had missed its deadline: asked again in
    /// the drain, never waited for.
    member val Late = Set.empty<uint64 * int64> with get, set
    /// Members whose result or chain Ready matched (the winner included).
    member val Confirmed = Set.empty<uint64> with get, set
    /// (member, membership serial) pairs the drain asked to compute the unit because too few members had.
    member val Verifiers = Set.empty<uint64 * int64> with get, set
    /// When the drain last asked the unconfirmed replicas again.
    member val LastRequestAt = at with get, set

/// The coordinator (ClusterProtocol contract): membership epochs, leases, stage replicas, assignment, Ready-gated
/// deadlines, reissue of lagging work, first-result-wins with divergence detection, the unit log and replay, and the
/// run digests. It is member 0 and a single-threaded actor; all its decisions iterate ordered collections, so in the
/// simulator it is deterministic.
[<Sealed>]
type Coordinator(config: CoordinatorConfig, workload: IWorkload, blobs: IBlobStore) =
    let tickTag = 1UL
    /// The coordinator's name in telemetry; no node may take it.
    let coordinatorName = "coord"
    let members = SortedDictionary<uint64, MemberState>()
    let roles = SortedDictionary<uint32, ResizeArray<uint64>>()
    let pending = SortedDictionary<int64, PendingState>()
    let pendingById = Dictionary<UnitId, PendingState>()
    let accepted = Dictionary<UnitId, AcceptedState>()
    let chains = SortedDictionary<ChainKey, ResizeArray<UnitId>>()
    /// (node, chain) pairs whose replay is in progress: the node asked for the chain's log and has not reported the
    /// chain Ready since. Deadlines of those assignments are paused.
    let replaying = HashSet<struct (uint64 * ChainKey)>()
    /// (node, chain) -> the last step before the entries the coordinator served the node for replay (-1 from the
    /// start). A replay checks every entry it applies, so a matching chain Ready confirms all of them.
    let replayServed = Dictionary<struct (uint64 * ChainKey), int>()
    let estimators = Dictionary<uint32, DeadlineEstimator>()
    let lagCounts = Dictionary<struct (uint64 * uint32), int>()
    let unavailable = HashSet<UnitId>()
    let divergences = ResizeArray<Divergence>()

    let completion =
        TaskCompletionSource<RunOutcome>(TaskCreationOptions.RunContinuationsAsynchronously)

    let mutable epoch = 0UL
    let mutable seq = 0UL
    let mutable nextIndex = 0L
    let mutable started = false
    let mutable outcome: RunOutcome option = None
    let mutable drainUntil = Int64.MaxValue
    let mutable reissues = 0
    let mutable duplicates = 0
    let mutable replays = 0
    let mutable rejected = 0
    let mutable memberships = 0L

    [<VolatileField>]
    let mutable acceptedCount = 0

    let mutable tel = Telemetry.noop
    let mutable ctx = Unchecked.defaultof<IActorContext>

    do
        if config.Replicas < 1 then
            invalidArg "config" "Replicas must be at least 1"

        if workload.Stages < 1u then
            invalidArg "workload" "a workload has at least one stage"

        // A Log payload is 14 bytes plus its entries, so a batch of several entries must leave room for those.
        if config.LogBatchBytes < 1 || config.LogBatchBytes > FrameCodec.MaxPayload - 14 then
            invalidArg "config" "LogBatchBytes must be 1 to FrameCodec.MaxPayload - 14"

    let stageList = [ for s in 0 .. int workload.Stages - 1 -> uint32 s ]

    let nameOf (node: uint64) =
        match members.TryGetValue node with
        | true, m -> m.Name
        | _ -> "id" + Invariant.u64 node

    let nodeAttr (node: uint64) =
        Attr.str TelemetryKeys.Node (nameOf node)

    let stageAttr (stage: uint32) =
        Attr.int TelemetryKeys.Stage (int stage)

    let unitAttrs (u: WorkUnit) =
        [
            Attr.int "lane" (int u.Lane)
            stageAttr u.Stage
            Attr.int "step" (int u.Step)
        ]

    let send (dest: uint64) (flags: uint32) (msg: Message) =
        seq <- seq + 1UL

        let frame =
            Message.toFrame
                {
                    Epoch = epoch
                    Sender = 0UL
                    Seq = seq
                    Flags = flags
                    Message = msg
                }

        if tel.Enabled then
            tel.Meter(
                ClusterNames.BytesSent,
                float frame.Payload.Length,
                TelemetryUnits.Bytes,
                [ Attr.str "kind" (MessageKind.name frame.Kind) ]
            )

        ctx.Send(dest, frame)

    let estimator (stage: uint32) =
        match estimators.TryGetValue stage with
        | true, e -> e
        | _ ->
            let e = DeadlineEstimator config.Deadline
            estimators[stage] <- e
            e

    let replicasOf (stage: uint32) =
        match roles.TryGetValue stage with
        | true, r -> r
        | _ ->
            let r = ResizeArray<uint64>()
            roles[stage] <- r
            r

    let rolesHeld (node: uint64) =
        roles.Values |> Seq.sumBy (fun r -> if r.Contains node then 1 else 0)

    /// (node, stage) pairs a node refused with WrongClass; cleared when the node leaves.
    let refused = HashSet<struct (uint64 * uint32)>()

    let eligible (m: MemberState) (stage: uint32) =
        m.Class = workload.Class
        && m.CanHost.Contains stage
        && not (refused.Contains(struct (m.Id, stage)))

    let startDeadline (now: int64) (p: PendingState) (a: AssignmentState) =
        if a.DeadlineStart.IsNone then
            a.DeadlineStart <- ValueSome now
            a.DeadlineAt <- now + (estimator p.Unit.Stage).Current

    /// Deadlines count only once the node is Ready, and not while it replays the unit's chain.
    let evaluate (now: int64) (p: PendingState) (a: AssignmentState) =
        match members.TryGetValue a.Node with
        | true, m when m.Ready && not (replaying.Contains(struct (a.Node, p.Chain))) -> startDeadline now p a
        | _ -> ()

    /// The node replays `chain`: pause its deadlines on that chain until it reports the chain Ready.
    let pause (node: uint64) (chain: ChainKey) =
        if replaying.Add(struct (node, chain)) then
            for p in pending.Values do
                if p.Chain = chain then
                    match p.Assignments.TryGetValue node with
                    | true, a ->
                        a.DeadlineStart <- ValueNone
                        a.DeadlineAt <- Int64.MaxValue
                    | _ -> ()

    /// The node holds `chain` again (Ready or a result): its deadlines on that chain run.
    let resume (now: int64) (node: uint64) (chain: ChainKey) =
        if replaying.Remove(struct (node, chain)) then
            for p in pending.Values do
                if p.Chain = chain then
                    match p.Assignments.TryGetValue node with
                    | true, a -> evaluate now p a
                    | _ -> ()

    let sendUnit (work: WorkUnit) (input: byte[]) (node: uint64) (flags: uint32) =
        send node flags (Message.Assign { Work = work; WorkInput = input })

        if tel.Enabled then
            tel.Counter(
                ClusterNames.Assign,
                1.0,
                TelemetryUnits.Count,
                [
                    stageAttr work.Stage
                    nodeAttr node
                    Attr.bool "retransmit" (flags <> FrameFlags.Empty)
                ]
            )

    let sendAssign (p: PendingState) (node: uint64) (flags: uint32) = sendUnit p.Unit p.Input node flags

    let serialOf (node: uint64) =
        match members.TryGetValue node with
        | true, m -> m.Serial
        | _ -> 0L

    /// True while `node` is still a member in the membership with this serial.
    let liveAs (node: uint64) (serial: int64) =
        match members.TryGetValue node with
        | true, m -> m.Serial = serial
        | _ -> false

    let assignTo (now: int64) (p: PendingState) (node: uint64) =
        if not (p.Assignments.ContainsKey node) then
            let a = AssignmentState(node, serialOf node, now)
            p.Assignments[node] <- a
            sendAssign p node FrameFlags.Empty
            evaluate now p a

    let plan (now: int64) (pu: PlannedUnit) =
        let id = WorkUnit.id pu.Unit

        if pu.Unit.Class <> workload.Class || pu.Unit.Job <> workload.Job then
            invalidOp "the workload planned a unit of another job or class"

        if pu.Unit.Stage >= workload.Stages then
            invalidOp (sprintf "the workload planned stage %d of %d" pu.Unit.Stage workload.Stages)

        if not (accepted.ContainsKey id || pendingById.ContainsKey id) then
            if blobs.Put pu.Input <> pu.Unit.Input then
                invalidOp "a planned unit's input does not match its Input blob id"

            let p = PendingState(nextIndex, pu.Unit, id, pu.Input)
            nextIndex <- nextIndex + 1L
            pending[p.Index] <- p
            pendingById[id] <- p

            for node in List.ofSeq (replicasOf pu.Unit.Stage) do
                assignTo now p node

    /// Keeps the surviving replicas of every stage and fills missing slots from eligible members: fewest roles first,
    /// then Ready ones, then the lowest id.
    let fillRoles () =
        for s in stageList do
            let r = replicasOf s
            r.RemoveAll(fun n -> not (members.ContainsKey n)) |> ignore
            let mutable searching = true

            while searching && r.Count < config.Replicas do
                let candidate =
                    members.Values
                    |> Seq.filter (fun m -> eligible m s && not (r.Contains m.Id))
                    |> Seq.sortBy (fun m -> rolesHeld m.Id, (if m.Ready then 0 else 1), m.Id)
                    |> Seq.tryHead

                match candidate with
                | Some m -> r.Add m.Id
                | None -> searching <- false

    let membershipMessage () =
        Message.Membership
            {
                Members =
                    [
                        for m in members.Values ->
                            {
                                MemberId = m.Id
                                MemberName = m.Name
                                MemberClass = m.Class
                            }
                    ]
                Roles =
                    [
                        for kv in roles ->
                            {
                                RoleStage = kv.Key
                                Nodes = List.ofSeq kv.Value
                            }
                    ]
            }

    let broadcastMembership () =
        let msg = membershipMessage ()

        for m in List.ofSeq members.Values do
            send m.Id FrameFlags.Empty msg

    /// After a membership change: drop assignments to departed members, re-send every pending assignment under the
    /// new epoch, and assign pending units to new replicas. A member that stayed keeps its deadlines, so churn
    /// elsewhere never postpones the reissue of its lagging work; only a restarted member (a new incarnation, with no
    /// state) starts its assignments over, from its Ready.
    let reassignAll (now: int64) =
        for p in List.ofSeq pending.Values do
            for node in List.ofSeq p.Assignments.Keys do
                if not (members.ContainsKey node) then
                    p.Assignments.Remove node |> ignore

            for a in List.ofSeq p.Assignments.Values do
                let serial = serialOf a.Node

                if a.Serial <> serial then
                    a.Serial <- serial
                    a.AssignedAt <- now
                    a.DeadlineStart <- ValueNone
                    a.DeadlineAt <- Int64.MaxValue
                    a.Misses <- 0

                a.LastSentAt <- now
                sendAssign p a.Node FrameFlags.Retransmit
                evaluate now p a

            for node in List.ofSeq (replicasOf p.Unit.Stage) do
                assignTo now p node

    let membershipChanged (now: int64) (reason: string) =
        if started then
            fillRoles ()

        broadcastMembership ()

        if started then
            reassignAll now

        if tel.Enabled then
            tel.Gauge(
                ClusterNames.Epoch,
                float epoch,
                TelemetryUnits.NoUnit,
                [ Attr.str "reason" reason; Attr.int "members" members.Count ]
            )

    let removeMember (node: uint64) (reason: string) =
        if tel.Enabled then
            tel.Event(ClusterNames.Leave, [ nodeAttr node; Attr.str "reason" reason ])

        members.Remove node |> ignore

        for r in roles.Values do
            r.Remove node |> ignore

        replaying.RemoveWhere(fun (struct (n, _)) -> n = node) |> ignore

        for key in List.ofSeq replayServed.Keys do
            let struct (n, _) = key

            if n = node then
                replayServed.Remove key |> ignore

        refused.RemoveWhere(fun (struct (n, _)) -> n = node) |> ignore

    let markReady (now: int64) (m: MemberState) =
        if not m.Ready then
            m.Ready <- true

            if tel.Enabled then
                tel.Event(ClusterNames.Ready, [ nodeAttr m.Id; Attr.str "scope" "node" ])

            for p in pending.Values do
                match p.Assignments.TryGetValue m.Id with
                | true, a -> evaluate now p a
                | _ -> ()

    let finalUnits () =
        [
            for kv in chains do
                if kv.Value.Count > 0 then
                    yield kv.Key, accepted[kv.Value[kv.Value.Count - 1]]
        ]

    /// Independent computations every final chain state needs before the run counts as verified: two, or one when
    /// the run has one replica per stage.
    let required = min config.Replicas 2

    /// Of the given (member, membership serial) pairs, the members still in that membership that can still run the
    /// unit's stage and have not confirmed the unit, in id order.
    let unconfirmedOf (acc: AcceptedState) (pairs: Set<uint64 * int64>) =
        pairs
        |> Seq.filter (fun (n, serial) ->
            not (acc.Confirmed.Contains n)
            && liveAs n serial
            && not (refused.Contains(struct (n, acc.Unit.Stage))))
        |> Seq.map fst
        |> List.ofSeq

    /// True when the member's last heartbeat lists the unit as queued or running: it is still computing it.
    let working (acc: AcceptedState) (node: uint64) =
        match members.TryGetValue node with
        | true, m -> m.Active.Contains acc.Id
        | _ -> false

    /// A final unit needs nothing more: enough members computed it, and every live on-time replica confirmed it.
    let settled (acc: AcceptedState) =
        acc.Confirmed.Count >= required && (unconfirmedOf acc acc.Expected).IsEmpty

    /// A final unit is verified at the end of the drain: enough members computed it, and every live on-time replica
    /// confirmed it or is still computing it (a laggard, reported in ChainFinal.PendingBy).
    let verified (acc: AcceptedState) =
        acc.Confirmed.Count >= required
        && unconfirmedOf acc acc.Expected |> List.forall (working acc)

    /// How often a member's assignments had missed their deadline when another replica won, over all stages.
    let lagOf (node: uint64) =
        stageList
        |> List.sumBy (fun s ->
            match lagCounts.TryGetValue(struct (node, s)) with
            | true, c -> c
            | _ -> 0)

    /// The member asked to compute a final unit because too few had: Ready, eligible, not yet asked; the least
    /// lagging first, then the fewest roles, then the lowest id. A verifier that lacks the chain replays it from
    /// the unit log first, and replay checks every logged output and digest it recomputes.
    let pickVerifier (acc: AcceptedState) =
        let asked = Set.unionMany [ acc.Expected; acc.Late; acc.Verifiers ] |> Set.map fst

        members.Values
        |> Seq.filter (fun m ->
            m.Ready
            && eligible m acc.Unit.Stage
            && not (acc.Confirmed.Contains m.Id)
            && not (asked.Contains m.Id))
        |> Seq.sortBy (fun m -> lagOf m.Id, rolesHeld m.Id, m.Id)
        |> Seq.tryHead

    let finish (now: int64) (status: RunStatus) =
        if outcome.IsNone then
            let finals =
                [
                    for chain, acc in finalUnits () ->
                        {
                            Final =
                                {
                                    Chain = chain
                                    Head = acc.Id
                                    Digest = acc.Digest
                                }
                            ConfirmedBy = Set.toList acc.Confirmed
                            PendingBy = unconfirmedOf acc acc.Expected |> List.filter (working acc)
                            UnconfirmedBy = unconfirmedOf acc acc.Expected |> List.filter (not << working acc)
                        }
                ]

            let units =
                accepted.Values
                |> Seq.map (fun a ->
                    {
                        Unit = a.Unit
                        Id = a.Id
                        Output = a.Output
                        StateDigest = a.Digest
                    })

            let o =
                {
                    Status = status
                    Job = workload.Job
                    RunDigest = RunDigest.compute workload.Job units
                    ReplicaDigest = RunDigest.replicas (finals |> List.map _.Final)
                    Units = accepted.Count
                    Epoch = epoch
                    Reissues = reissues
                    Duplicates = duplicates
                    Divergences = List.ofSeq divergences
                    Replays = replays
                    Rejected = rejected
                    CrossChecked = accepted.Values |> Seq.filter (fun a -> a.Confirmed.Count >= 2) |> Seq.length
                    Unconfirmed = finalUnits () |> List.filter (fun (_, acc) -> not (verified acc)) |> List.length
                    Chains = finals
                }

            outcome <- Some o
            let statusText = RunStatus.name status

            for m in List.ofSeq members.Values do
                send
                    m.Id
                    FrameFlags.Empty
                    (Message.Notice
                        {
                            Code = NoticeCode.Shutdown
                            Value = 0UL
                            Detail = statusText
                        })

            if tel.Enabled then
                tel.Event(
                    ClusterNames.Run,
                    [
                        Attr.str "status" statusText
                        Attr.int "units" o.Units
                        Attr.int64 "epoch" (int64 o.Epoch)
                        Attr.int "reissues" o.Reissues
                        Attr.int "duplicates" o.Duplicates
                        Attr.int "divergences" o.Divergences.Length
                        Attr.int "replays" o.Replays
                        Attr.int "cross_checked" o.CrossChecked
                        Attr.int "unconfirmed" o.Unconfirmed
                        Attr.str "run_digest" o.RunDigest.Hex
                        Attr.int64 "end_ns" now
                    ]
                )

            completion.TrySetResult o |> ignore

    let diverge (now: int64) (d: Divergence) =
        // A frame duplicated by the network reports the same divergence twice: record it once.
        if not (divergences.Contains d) then
            divergences.Add d

            // The other node's earlier confirmation of the unit (for example from a replay range) no longer holds.
            match accepted.TryGetValue d.Unit with
            | true, acc when d.OtherNode <> acc.Node -> acc.Confirmed <- acc.Confirmed.Remove d.OtherNode
            | _ -> ()

            if tel.Enabled then
                tel.Event(
                    ClusterNames.Divergence,
                    [
                        Attr.str "unit" d.Unit.Value.Hex
                        Attr.int "lane" (int d.Lane)
                        stageAttr d.Stage
                        Attr.int "step" (int d.Step)
                        Attr.str "first_node" (nameOf d.FirstNode)
                        Attr.str "other_node" (nameOf d.OtherNode)
                        Attr.str
                            "source"
                            (match d.Source with
                             | DivergenceSource.Result -> "result"
                             | DivergenceSource.Replay -> "replay")
                    ]
                )

            if config.HaltOnDivergence then
                finish now RunStatus.Diverged

    /// The end of the run (the drain): once the workload is complete and nothing is pending, wait until every final
    /// unit is settled, for at most DrainNs. Every ResendNs it asks again the replicas that owe a confirmation (a node
    /// answers a re-sent Assign of its chain head from its cache), and when too few members have computed a final
    /// unit (its other replicas left), it asks one more member to compute it.
    let checkDone (now: int64) =
        if started && outcome.IsNone && workload.IsComplete && pending.Count = 0 then
            if drainUntil = Int64.MaxValue then
                drainUntil <- now + config.DrainNs

            let finals = finalUnits ()
            let allSettled = finals |> List.forall (fun (_, acc) -> settled acc)
            let timedOut = now >= drainUntil

            if divergences.Count > 0 && (allSettled || timedOut) then
                finish now RunStatus.Diverged
            elif allSettled then
                finish now RunStatus.Completed
            elif timedOut then
                finish
                    now
                    (if finals |> List.forall (fun (_, acc) -> verified acc) then
                         RunStatus.Completed
                     else
                         RunStatus.Unverified)
            else
                for _, acc in finals do
                    if not (settled acc) && now - acc.LastRequestAt >= config.ResendNs then
                        acc.LastRequestAt <- now

                        let ask =
                            unconfirmedOf acc (Set.unionMany [ acc.Expected; acc.Late; acc.Verifiers ])

                        // Too few computations: engage one more member each round, rather than wait for a replica
                        // or a verifier that is behind.
                        let extra =
                            if acc.Confirmed.Count < required then
                                match pickVerifier acc with
                                | Some m ->
                                    acc.Verifiers <- acc.Verifiers.Add(m.Id, m.Serial)
                                    [ m.Id ]
                                | None -> []
                            else
                                []

                        if not (ask.IsEmpty && extra.IsEmpty) then
                            match blobs.TryGet acc.Unit.Input with
                            | Some input ->
                                for n in ask @ extra do
                                    sendUnit acc.Unit input n FrameFlags.Retransmit
                            | None -> invalidOp (sprintf "input blob of unit %s is missing" acc.Id.Value.Short)

    let maybeStart (now: int64) =
        if not started && members.Count >= config.MinNodes then
            started <- true
            fillRoles ()
            broadcastMembership ()

            for pu in workload.Start() do
                plan now pu

    let reject (node: uint64) (reason: string) (kind: MessageKind) =
        rejected <- rejected + 1

        if tel.Enabled then
            tel.Counter(
                ClusterNames.FrameRejected,
                1.0,
                TelemetryUnits.Count,
                [
                    Attr.str "reason" reason
                    Attr.str "kind" (MessageKind.name kind)
                    nodeAttr node
                ]
            )

    let notice (node: uint64) (code: NoticeCode) (value: uint64) (detail: string) =
        send
            node
            FrameFlags.Empty
            (Message.Notice
                {
                    Code = code
                    Value = value
                    Detail = detail
                })

    let welcome (node: uint64) (echo: uint64) =
        send
            node
            FrameFlags.Empty
            (Message.Welcome
                {
                    HeartbeatMs = Ns.toMs config.HeartbeatNs
                    LeaseMs = Ns.toMs config.LeaseNs
                    EchoSeq = echo
                })

    let onHello (now: int64) (env: Envelope) (h: HelloMsg) =
        let node = env.Sender

        if node = 0UL then
            reject node "reserved_id" MessageKind.Hello
        elif not (SafeName.isValid h.Name) || h.Name = coordinatorName then
            // Names label the node in Membership and telemetry: they must be safe names, and "coord" is ours.
            reject node "bad_name" MessageKind.Hello
        elif members.Values |> Seq.exists (fun m -> m.Name = h.Name && m.Id <> node) then
            // Two ids with one name would merge in the per-node charts.
            reject node "duplicate_name" MessageKind.Hello
        else
            match members.TryGetValue node with
            | true, m when m.Incarnation = h.Incarnation ->
                // A resync of a current member (its view of the lease ran out first): no membership change.
                m.LeaseUntil <- now + config.LeaseNs
                welcome node env.Seq
                send node FrameFlags.Empty (membershipMessage ())
            | found, _ ->
                if found then
                    removeMember node "restart"

                memberships <- memberships + 1L

                let m =
                    MemberState(
                        node,
                        h.Name,
                        h.Incarnation,
                        memberships,
                        h.Class,
                        Set.ofList h.CanHost,
                        LeaseUntil = now + config.LeaseNs
                    )

                members[node] <- m

                if tel.Enabled then
                    tel.Event(
                        ClusterNames.Join,
                        [
                            nodeAttr node
                            Attr.str "incarnation" (Invariant.u64 h.Incarnation)
                            Attr.str "reason" (if found then "restart" else "join")
                        ]
                    )

                epoch <- epoch + 1UL
                welcome node env.Seq
                membershipChanged now (if found then "restart" else "join")
                maybeStart now

    let onHeartbeat (now: int64) (env: Envelope) (hb: HeartbeatMsg) =
        match members.TryGetValue env.Sender with
        | false, _ ->
            reject env.Sender "unknown_member" MessageKind.Heartbeat
            notice env.Sender NoticeCode.UnknownMember epoch "send Hello"
        | true, m ->
            // Any heartbeat from a member is proof of life; a stale one's content is rejected.
            m.LeaseUntil <- now + config.LeaseNs

            send
                m.Id
                FrameFlags.Empty
                (Message.Lease
                    {
                        DurationMs = Ns.toMs config.LeaseNs
                        LeaseEcho = env.Seq
                    })

            if env.Epoch <> epoch then
                reject m.Id "stale_epoch" MessageKind.Heartbeat
                notice m.Id NoticeCode.StaleEpoch epoch "stale epoch"
                send m.Id FrameFlags.Empty (membershipMessage ())
            else
                if hb.IsReady then
                    markReady now m

                m.Active <- hb.Active |> List.map fst |> Set.ofList
                let active = HashSet<UnitId>()

                for id, phase in hb.Active do
                    active.Add id |> ignore

                    match pendingById.TryGetValue id with
                    | true, p when p.Assignments.ContainsKey m.Id ->
                        match phase with
                        | UnitPhase.Fetching
                        | UnitPhase.Replaying -> pause m.Id p.Chain
                        | UnitPhase.Positioned -> resume now m.Id p.Chain
                        | _ -> ()
                    | _ -> ()

                for p in pending.Values do
                    match p.Assignments.TryGetValue m.Id with
                    | true, a when not (active.Contains p.Id) && now - a.LastSentAt >= config.ResendNs ->
                        a.LastSentAt <- now
                        sendAssign p m.Id FrameFlags.Retransmit
                    | _ -> ()

    let onReady (now: int64) (m: MemberState) (r: ReadyMsg) =
        match r.ReadyChain with
        | None -> markReady now m
        | Some chain ->
            markReady now m

            if tel.Enabled then
                tel.Event(
                    ClusterNames.Ready,
                    [
                        nodeAttr m.Id
                        Attr.str "scope" "chain"
                        Attr.int "lane" (int chain.Lane)
                        stageAttr chain.Stage
                    ]
                )

            // The node holds the chain at Head with this digest and Head's output, as it computed them (a replay
            // recomputes every unit it applies, checking each against the log): compare them with the accepted unit.
            let agrees =
                match accepted.TryGetValue r.ReadyHead with
                | true, acc when WorkUnit.chain acc.Unit = chain ->
                    let outputGiven = not (BlobId.isNone r.ReadyOutput)

                    if acc.Digest <> r.ReadyDigest || (outputGiven && acc.Output <> r.ReadyOutput) then
                        replayServed.Remove(struct (m.Id, chain)) |> ignore

                        diverge
                            now
                            {
                                Unit = acc.Id
                                Lane = acc.Unit.Lane
                                Stage = acc.Unit.Stage
                                Step = acc.Unit.Step
                                FirstNode = acc.Node
                                FirstOutput = acc.Output
                                FirstDigest = acc.Digest
                                OtherNode = m.Id
                                OtherOutput = (if outputGiven then Some r.ReadyOutput else None)
                                OtherDigest = r.ReadyDigest
                                Source = DivergenceSource.Replay
                            }

                        false
                    else
                        if outputGiven then
                            acc.Confirmed <- acc.Confirmed.Add m.Id

                            // The replay that brought the chain here recomputed and checked every entry served.
                            match replayServed.TryGetValue(struct (m.Id, chain)), chains.TryGetValue chain with
                            | (true, from), (true, list) ->
                                replayServed.Remove(struct (m.Id, chain)) |> ignore

                                for i in from + 1 .. min (int acc.Unit.Step) (list.Count - 1) do
                                    let e = accepted[list[i]]
                                    e.Confirmed <- e.Confirmed.Add m.Id
                            | _ -> ()

                        true
                | _ -> true

            if agrees then
                resume now m.Id chain
                checkDone now

    let demoteLaggards (stage: uint32) =
        let r = replicasOf stage

        if r.Count > config.Replicas then
            let laggards =
                r
                |> Seq.map (fun n ->
                    n,
                    (match lagCounts.TryGetValue(struct (n, stage)) with
                     | true, c -> c
                     | _ -> 0))
                |> Seq.filter (fun (_, c) -> c >= 3)
                |> Seq.sortBy (fun (n, c) -> -c, n)
                |> List.ofSeq

            for n, _ in laggards do
                if r.Count > config.Replicas then
                    r.Remove n |> ignore
                    lagCounts[struct (n, stage)] <- 0

                    if tel.Enabled then
                        tel.Event(ClusterNames.Demote, [ stageAttr stage; nodeAttr n ])

    /// Another result for an accepted unit. Equal: it confirms the unit, unless it is the accepted result itself again
    /// (the same member and incarnation: a network duplicate, or a re-answer from the chain head). Different, from
    /// whoever (a node keeps its id across restarts): a divergence.
    let checkResult (now: int64) (m: MemberState) (acc: AcceptedState) (outputId: BlobId) (digest: ContentId) =
        if acc.Output = outputId && acc.Digest = digest then
            if m.Id <> acc.Node || m.Incarnation <> acc.NodeIncarnation then
                duplicates <- duplicates + 1
                acc.Confirmed <- acc.Confirmed.Add m.Id

                if tel.Enabled then
                    tel.Counter(
                        ClusterNames.Duplicate,
                        1.0,
                        TelemetryUnits.Count,
                        [ stageAttr acc.Unit.Stage; nodeAttr m.Id ]
                    )

                    tel.Histogram(
                        ClusterNames.Lag,
                        float (now - acc.AcceptedAt),
                        TelemetryUnits.Nanoseconds,
                        [ stageAttr acc.Unit.Stage; nodeAttr m.Id ]
                    )
        else
            diverge
                now
                {
                    Unit = acc.Id
                    Lane = acc.Unit.Lane
                    Stage = acc.Unit.Stage
                    Step = acc.Unit.Step
                    FirstNode = acc.Node
                    FirstOutput = acc.Output
                    FirstDigest = acc.Digest
                    OtherNode = m.Id
                    OtherOutput = Some outputId
                    OtherDigest = digest
                    Source = DivergenceSource.Result
                }

    let onResult (now: int64) (m: MemberState) (r: ResultMsg) =
        let outputId = BlobId.ofBytes r.ResultOutput

        match pendingById.TryGetValue r.ResultUnit with
        | true, p ->
            // First result wins.
            replaying.Remove(struct (m.Id, p.Chain)) |> ignore
            blobs.Put r.ResultOutput |> ignore
            let stage = p.Unit.Stage

            let winner =
                match p.Assignments.TryGetValue m.Id with
                | true, a -> Some a
                | _ -> None

            let deadlineStart =
                winner |> Option.bind (fun a -> ValueOption.toOption a.DeadlineStart)

            // The Gantt chart's start: the deadline start, else the assignment, else (no assignment) the result.
            let start =
                match deadlineStart, winner with
                | Some s, _ -> s
                | None, Some a -> a.AssignedAt
                | None, None -> now

            let latency = now - start

            // A deadline sample runs from the winning assignment's deadline start; without one (the deadline was
            // paused, or the winner was not assigned) the time includes cold start or replay, so it is no sample.
            match deadlineStart with
            | Some s ->
                let est = estimator stage
                let before = est.Current
                est.Add(now - s)

                if tel.Enabled && est.Current <> before then
                    tel.Gauge(
                        ClusterNames.Deadline,
                        float est.Current,
                        TelemetryUnits.Nanoseconds,
                        [ stageAttr stage; Attr.int "samples" est.Count ]
                    )
            | None -> ()

            let acc =
                AcceptedState(p.Unit, p.Id, outputId, r.ResultDigest, m.Id, m.Incarnation, now)

            let assigned =
                p.Assignments.Values
                |> Seq.filter (fun a -> liveAs a.Node a.Serial)
                |> List.ofSeq

            acc.Expected <-
                assigned
                |> Seq.filter (fun a -> a.Misses = 0)
                |> Seq.map (fun a -> a.Node, a.Serial)
                |> Set.ofSeq

            acc.Late <-
                assigned
                |> Seq.filter (fun a -> a.Misses > 0)
                |> Seq.map (fun a -> a.Node, a.Serial)
                |> Set.ofSeq

            acc.Confirmed <- Set.singleton m.Id
            accepted[p.Id] <- acc

            let list =
                match chains.TryGetValue p.Chain with
                | true, l -> l
                | _ ->
                    let l = ResizeArray<UnitId>()
                    chains[p.Chain] <- l
                    l

            if list.Count <> int p.Unit.Step then
                invalidOp (
                    sprintf "chain (%d, %d) accepted step %d out of order" p.Chain.Lane p.Chain.Stage p.Unit.Step
                )

            list.Add p.Id
            pending.Remove p.Index |> ignore
            pendingById.Remove p.Id |> ignore
            acceptedCount <- acceptedCount + 1

            for a in p.Assignments.Values do
                let key = struct (a.Node, stage)

                if a.Node <> m.Id && a.Misses > 0 then
                    lagCounts[key] <-
                        (match lagCounts.TryGetValue key with
                         | true, c -> c
                         | _ -> 0)
                        + 1
                else
                    lagCounts[key] <- 0

            demoteLaggards stage

            if tel.Enabled then
                tel.Event(
                    ClusterNames.UnitDone,
                    unitAttrs p.Unit
                    @ [
                        nodeAttr m.Id
                        Attr.int64 "start_ns" start
                        Attr.int64 "end_ns" now
                        Attr.int64 "latency_ns" latency
                    ]
                )

                if deadlineStart.IsSome then
                    tel.Histogram(
                        ClusterNames.UnitLatency,
                        float latency,
                        TelemetryUnits.Nanoseconds,
                        [ stageAttr stage; nodeAttr m.Id ]
                    )

            for next in workload.Accept(p.Unit, r.ResultOutput) do
                plan now next

            checkDone now
        | _ ->
            match accepted.TryGetValue r.ResultUnit with
            | true, acc ->
                resume now m.Id (WorkUnit.chain acc.Unit)
                checkResult now m acc outputId r.ResultDigest
                checkDone now
            | _ -> reject m.Id "unknown_unit" MessageKind.Result

    /// A Result of a stale epoch is rejected, but results are content-addressed: if its unit is already accepted, it
    /// is still compared with it (a confirmation or a divergence), since that does not depend on the epoch.
    let onStaleResult (now: int64) (m: MemberState) (r: ResultMsg) =
        match accepted.TryGetValue r.ResultUnit with
        | true, acc ->
            checkResult now m acc (BlobId.ofBytes r.ResultOutput) r.ResultDigest
            checkDone now
        | _ -> ()

    let onFetch (m: MemberState) (f: FetchMsg) =
        replays <- replays + 1
        pause m.Id f.FetchChain

        let empty reset =
            send
                m.Id
                FrameFlags.Empty
                (Message.Log
                    {
                        LogChain = f.FetchChain
                        Reset = reset
                        Final = true
                        Entries = []
                    })

        match chains.TryGetValue f.FetchChain, accepted.TryGetValue f.To with
        | (true, list), (true, target) when WorkUnit.chain target.Unit = f.FetchChain ->
            let iTo = int target.Unit.Step

            let reset, iFrom =
                if UnitId.isNone f.From then
                    false, -1
                else
                    match accepted.TryGetValue f.From with
                    | true, from when WorkUnit.chain from.Unit = f.FetchChain -> false, int from.Unit.Step
                    | _ -> true, -1

            if iFrom >= iTo then
                empty reset
            else
                let entries =
                    [
                        for i in iFrom + 1 .. iTo do
                            let acc = accepted[list[i]]

                            let input =
                                match blobs.TryGet acc.Unit.Input with
                                | Some b -> b
                                | None -> invalidOp (sprintf "input blob of unit %s is missing" acc.Id.Value.Short)

                            yield
                                {
                                    Unit = acc.Unit
                                    Input = input
                                    Output = acc.Output
                                    StateDigest = acc.Digest
                                }
                    ]

                // Batches of at most LogBatchBytes of encoded entries (at least one entry each; one entry always fits
                // in a frame, since inputs are at most WorkUnit.MaxData and Args at most WorkUnit.MaxArgs).
                let batches = ResizeArray<LogEntry list>()
                let current = ResizeArray<LogEntry>()
                let mutable size = 0L

                for e in entries do
                    let n = int64 (Message.logEntrySize e)

                    if current.Count > 0 && size + n > int64 config.LogBatchBytes then
                        batches.Add(List.ofSeq current)
                        current.Clear()
                        size <- 0L

                    current.Add e
                    size <- size + n

                batches.Add(List.ofSeq current)
                replayServed[struct (m.Id, f.FetchChain)] <- iFrom

                batches
                |> Seq.iteri (fun i batch ->
                    send
                        m.Id
                        FrameFlags.Empty
                        (Message.Log
                            {
                                LogChain = f.FetchChain
                                Reset = reset && i = 0
                                Final = i = batches.Count - 1
                                Entries = batch
                            }))

                if tel.Enabled then
                    tel.Counter(
                        ClusterNames.ReplayServed,
                        float entries.Length,
                        TelemetryUnits.Count,
                        [
                            Attr.int "lane" (int f.FetchChain.Lane)
                            stageAttr f.FetchChain.Stage
                            nodeAttr m.Id
                        ]
                    )
        | _ -> empty true

    let reissue (now: int64) (p: PendingState) =
        let candidate =
            members.Values
            |> Seq.filter (fun m -> m.Ready && eligible m p.Unit.Stage && not (p.Assignments.ContainsKey m.Id))
            |> Seq.sortBy (fun m -> rolesHeld m.Id, m.Id)
            |> Seq.tryHead

        match candidate with
        | Some m ->
            reissues <- reissues + 1
            let r = replicasOf p.Unit.Stage

            if not (r.Contains m.Id) then
                r.Add m.Id

            if tel.Enabled then
                tel.Counter(ClusterNames.Reissue, 1.0, TelemetryUnits.Count, unitAttrs p.Unit @ [ nodeAttr m.Id ])

            assignTo now p m.Id
        | None ->
            if unavailable.Add p.Id && tel.Enabled then
                tel.Event(ClusterNames.ReissueUnavailable, unitAttrs p.Unit)

    let onTick (now: int64) =
        let expired =
            members.Values
            |> Seq.filter (fun m -> m.LeaseUntil < now)
            |> Seq.map _.Id
            |> List.ofSeq

        if not expired.IsEmpty then
            for node in expired do
                removeMember node "lease"

            epoch <- epoch + 1UL
            membershipChanged now "lease"

        for p in List.ofSeq pending.Values do
            if outcome.IsNone then
                let mutable live = 0
                let mutable missed = false

                for a in List.ofSeq p.Assignments.Values do
                    match a.DeadlineStart with
                    | ValueNone -> live <- live + 1
                    | ValueSome _ when now < a.DeadlineAt ->
                        if a.Misses = 0 then
                            live <- live + 1
                    | ValueSome _ ->
                        // Missed: re-send (the Assign or the Result may be lost) and re-arm for another period.
                        a.Misses <- a.Misses + 1
                        a.DeadlineAt <- now + (estimator p.Unit.Stage).Current
                        a.LastSentAt <- now
                        missed <- true
                        sendAssign p a.Node FrameFlags.Retransmit

                        if tel.Enabled then
                            tel.Counter(
                                ClusterNames.DeadlineMiss,
                                1.0,
                                TelemetryUnits.Count,
                                [ stageAttr p.Unit.Stage; nodeAttr a.Node ]
                            )

                if live = 0 && (missed || p.Assignments.Count > 0) then
                    reissue now p

        checkDone now

    /// Completes with the outcome when the run finishes (from any thread).
    member _.Completion: Task<RunOutcome> = completion.Task

    /// A multi-line snapshot of the coordinator's state for liveness diagnostics. Safe to call from any thread, but
    /// off the actor thread the snapshot may be inconsistent.
    member _.Describe() : string =
        try
            let b = Text.StringBuilder()
            let line (s: string) = b.AppendLine s |> ignore
            let now = if isNull (box ctx) then 0L else ctx.Now

            let ms (ns: int64) =
                if ns = Int64.MaxValue then
                    "-"
                else
                    Invariant.i64 ((ns - now) / Ns.PerMs) + "ms"

            line (
                sprintf
                    "coordinator: epoch %d, started %b, accepted %d, pending %d, complete %b, drain %s, outcome %b"
                    epoch
                    started
                    accepted.Count
                    pending.Count
                    workload.IsComplete
                    (ms drainUntil)
                    outcome.IsSome
            )

            for m in List.ofSeq members.Values do
                line (sprintf "  member %s id %d ready %b lease %s" m.Name m.Id m.Ready (ms m.LeaseUntil))

            for kv in List.ofSeq roles do
                line (sprintf "  stage %d replicas %A" kv.Key (List.ofSeq kv.Value))

            for struct (n, c) in List.ofSeq replaying do
                line (sprintf "  replaying node %d chain (%d, %d)" n c.Lane c.Stage)

            if drainUntil <> Int64.MaxValue then
                for chain, acc in finalUnits () do
                    if not (settled acc) then
                        line (
                            sprintf
                                "  final (%d, %d) confirmed by %A, owed by %A, verifiers %A"
                                chain.Lane
                                chain.Stage
                                (Set.toList acc.Confirmed)
                                (unconfirmedOf acc acc.Expected)
                                (Set.toList acc.Verifiers |> List.map fst)
                        )

            for p in List.ofSeq pending.Values do
                line (
                    sprintf
                        "  pending #%d lane %d stage %d step %d %s"
                        p.Index
                        p.Unit.Lane
                        p.Unit.Stage
                        p.Unit.Step
                        p.Id.Value.Short
                )

                for a in List.ofSeq p.Assignments.Values do
                    line (
                        sprintf
                            "    node %d assigned %s sent %s deadline %s misses %d"
                            a.Node
                            (ms a.AssignedAt)
                            (ms a.LastSentAt)
                            (match a.DeadlineStart with
                             | ValueSome _ -> ms a.DeadlineAt
                             | ValueNone -> "paused")
                            a.Misses
                    )

            b.ToString()
        with e ->
            "coordinator: snapshot failed: " + e.Message

    /// The outcome, once the run has finished.
    member _.Outcome = outcome

    /// Units accepted so far (safe to read from any thread).
    member _.AcceptedCount = acceptedCount

    /// The current epoch (actor thread only).
    member _.Epoch = epoch

    /// Stops the run with RunStatus.Aborted (actor thread only).
    member _.Abort(now: int64, reason: string) = finish now (RunStatus.Aborted reason)

    interface IActor with
        member _.Id = 0UL

        member _.Start(c: IActorContext) =
            ctx <- c

            tel <-
                (Telemetry.current ())
                    .Child(
                        [
                            Attr.str TelemetryKeys.Component "cluster.coordinator"
                            Attr.str TelemetryKeys.Node coordinatorName
                        ]
                    )

            c.SetTimer(c.Now + config.TickNs, tickTag)

        member _.OnFrame(c: IActorContext, frame: Frame) =
            match outcome with
            | Some o ->
                // The run is over: whoever still talks to us missed the Shutdown (it is sent once and may be lost).
                if frame.Sender <> 0UL then
                    notice frame.Sender NoticeCode.Shutdown 0UL (RunStatus.name o.Status)
            | None ->
                let now = c.Now

                if frame.Sender = 0UL then
                    // Member 0 is the coordinator itself: never answer, or a frame would loop back to us.
                    reject 0UL "reserved_id" frame.Kind
                else
                    match Message.tryOfFrame frame with
                    | Error _ -> reject frame.Sender "malformed" frame.Kind
                    | Ok env ->
                        match env.Message with
                        | Message.Hello h -> onHello now env h
                        | Message.Heartbeat hb -> onHeartbeat now env hb
                        | msg ->
                            match members.TryGetValue env.Sender with
                            | false, _ ->
                                reject env.Sender "unknown_member" frame.Kind
                                notice env.Sender NoticeCode.UnknownMember epoch "send Hello"
                            | true, m when env.Epoch <> epoch ->
                                reject m.Id "stale_epoch" frame.Kind
                                notice m.Id NoticeCode.StaleEpoch epoch "stale epoch"
                                send m.Id FrameFlags.Empty (membershipMessage ())

                                match msg with
                                | Message.Result r -> onStaleResult now m r
                                | _ -> ()
                            | true, m ->
                                match msg with
                                | Message.Result r -> onResult now m r
                                | Message.Ready r -> onReady now m r
                                | Message.Fetch f -> onFetch m f
                                | Message.Notice n when n.Code = NoticeCode.Leave ->
                                    removeMember m.Id "leave"
                                    epoch <- epoch + 1UL
                                    membershipChanged now "leave"
                                | Message.Notice n when
                                    n.Code = NoticeCode.WrongClass && n.Value <= uint64 UInt32.MaxValue
                                    ->
                                    // The node cannot run units of this stage: never give it the stage again, and move
                                    // its pending work of the stage to replicas that can.
                                    let stage = uint32 n.Value

                                    if refused.Add(struct (m.Id, stage)) then
                                        (replicasOf stage).Remove m.Id |> ignore

                                        for p in List.ofSeq pending.Values do
                                            if p.Unit.Stage = stage then
                                                p.Assignments.Remove m.Id |> ignore

                                        fillRoles ()

                                        for p in List.ofSeq pending.Values do
                                            if p.Unit.Stage = stage then
                                                for node in List.ofSeq (replicasOf stage) do
                                                    assignTo now p node
                                | _ -> reject m.Id "unexpected_kind" frame.Kind

        member _.OnTimer(c: IActorContext, tag: uint64) =
            if tag = tickTag && outcome.IsNone then
                onTick c.Now

                if outcome.IsNone then
                    c.SetTimer(c.Now + config.TickNs, tickTag)

        member _.OnWorkDone(_, _, _) = ()
