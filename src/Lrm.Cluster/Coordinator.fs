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
        /// The largest Log batch, in input bytes: 16 MiB.
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
    | Completed
    /// The divergence tripwire fired.
    | Diverged
    | Aborted of reason: string

/// The final state of one chain and the members that confirmed it.
type ChainFinal =
    {
        Final: ChainHead
        ConfirmedBy: uint64 list
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
        Chains: ChainFinal list
    }

[<Sealed; AllowNullLiteral>]
type internal MemberState(id: uint64, name: string, incarnation: uint64, cls: ExecutionClassId, canHost: Set<uint32>) =
    member _.Id = id
    member _.Name = name
    member _.Incarnation = incarnation
    member _.Class = cls
    member _.CanHost = canHost
    member val LeaseUntil = 0L with get, set
    member val Ready = false with get, set

[<Sealed; AllowNullLiteral>]
type internal AssignmentState(node: uint64, now: int64) =
    member _.Node = node
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
type internal AcceptedState(work: WorkUnit, id: UnitId, output: BlobId, digest: ContentId, node: uint64, at: int64) =
    member _.Unit = work
    member _.Id = id
    member _.Output = output
    member _.Digest = digest
    member _.Node = node
    member _.AcceptedAt = at
    /// Members that were assigned the unit when it was accepted.
    member val Expected = Set.empty<uint64> with get, set
    /// Members whose result matched.
    member val Confirmed = Set.empty<uint64> with get, set

/// The coordinator (ClusterProtocol contract): membership epochs, leases, stage replicas, assignment, Ready-gated
/// deadlines, reissue of lagging work, first-result-wins with divergence detection, the unit log and replay, and the
/// run digests. It is member 0 and a single-threaded actor; all its decisions iterate ordered collections, so in the
/// simulator it is deterministic.
[<Sealed>]
type Coordinator(config: CoordinatorConfig, workload: IWorkload, blobs: IBlobStore) =
    let tickTag = 1UL
    let members = SortedDictionary<uint64, MemberState>()
    let roles = SortedDictionary<uint32, ResizeArray<uint64>>()
    let pending = SortedDictionary<int64, PendingState>()
    let pendingById = Dictionary<UnitId, PendingState>()
    let accepted = Dictionary<UnitId, AcceptedState>()
    let chains = SortedDictionary<ChainKey, ResizeArray<UnitId>>()
    /// (node, chain) pairs whose replay is in progress: the node asked for the chain's log and has not reported the
    /// chain Ready since. Deadlines of those assignments are paused.
    let replaying = HashSet<struct (uint64 * ChainKey)>()
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

    [<VolatileField>]
    let mutable acceptedCount = 0

    let mutable tel = Telemetry.noop
    let mutable ctx = Unchecked.defaultof<IActorContext>

    do
        if config.Replicas < 1 then
            invalidArg "config" "Replicas must be at least 1"

        if workload.Stages < 1u then
            invalidArg "workload" "a workload has at least one stage"

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

    let sendAssign (p: PendingState) (node: uint64) (flags: uint32) =
        send node flags (Message.Assign { Work = p.Unit; WorkInput = p.Input })

        if tel.Enabled then
            tel.Counter(
                ClusterNames.Assign,
                1.0,
                TelemetryUnits.Count,
                [
                    stageAttr p.Unit.Stage
                    nodeAttr node
                    Attr.bool "retransmit" (flags <> FrameFlags.Empty)
                ]
            )

    let assignTo (now: int64) (p: PendingState) (node: uint64) =
        if not (p.Assignments.ContainsKey node) then
            let a = AssignmentState(node, now)
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
    /// new epoch (deadlines restart from Ready), and assign pending units to new replicas.
    let reassignAll (now: int64) =
        for p in List.ofSeq pending.Values do
            for node in List.ofSeq p.Assignments.Keys do
                if not (members.ContainsKey node) then
                    p.Assignments.Remove node |> ignore

            for a in List.ofSeq p.Assignments.Values do
                a.AssignedAt <- now
                a.LastSentAt <- now
                a.DeadlineStart <- ValueNone
                a.DeadlineAt <- Int64.MaxValue
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
                    Chains = finals
                }

            outcome <- Some o

            let statusText =
                match status with
                | RunStatus.Completed -> "completed"
                | RunStatus.Diverged -> "diverged"
                | RunStatus.Aborted _ -> "aborted"

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
                        Attr.str "run_digest" o.RunDigest.Hex
                        Attr.int64 "end_ns" now
                    ]
                )

            completion.TrySetResult o |> ignore

    let diverge (now: int64) (d: Divergence) =
        divergences.Add d

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

    let checkDone (now: int64) =
        if started && outcome.IsNone && workload.IsComplete && pending.Count = 0 then
            if drainUntil = Int64.MaxValue then
                drainUntil <- now + config.DrainNs

            let confirmed =
                finalUnits ()
                |> List.forall (fun (_, acc) ->
                    acc.Expected
                    |> Set.forall (fun n -> acc.Confirmed.Contains n || not (members.ContainsKey n)))

            if confirmed || now >= drainUntil then
                finish
                    now
                    (if divergences.Count > 0 then
                         RunStatus.Diverged
                     else
                         RunStatus.Completed)

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

                let m =
                    MemberState(
                        node,
                        h.Name,
                        h.Incarnation,
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

            if not (UnitId.isNone r.ReadyHead) then
                match accepted.TryGetValue r.ReadyHead with
                | true, acc when acc.Digest <> r.ReadyDigest ->
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
                            OtherOutput = None
                            OtherDigest = r.ReadyDigest
                            Source = DivergenceSource.Replay
                        }
                | _ -> ()

            resume now m.Id chain

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

    let onResult (now: int64) (m: MemberState) (r: ResultMsg) =
        let outputId = BlobId.ofBytes r.ResultOutput

        match pendingById.TryGetValue r.ResultUnit with
        | true, p ->
            // First result wins.
            replaying.Remove(struct (m.Id, p.Chain)) |> ignore
            blobs.Put r.ResultOutput |> ignore
            let stage = p.Unit.Stage

            let start =
                match p.Assignments.TryGetValue m.Id with
                | true, a ->
                    match a.DeadlineStart with
                    | ValueSome s -> s
                    | ValueNone -> a.AssignedAt
                | _ -> now

            let latency = now - start
            let est = estimator stage
            let before = est.Current
            est.Add latency

            if tel.Enabled && est.Current <> before then
                tel.Gauge(
                    ClusterNames.Deadline,
                    float est.Current,
                    TelemetryUnits.Nanoseconds,
                    [ stageAttr stage; Attr.int "samples" est.Count ]
                )

            let acc = AcceptedState(p.Unit, p.Id, outputId, r.ResultDigest, m.Id, now)

            acc.Expected <- p.Assignments.Keys |> Seq.filter (fun n -> members.ContainsKey n) |> Set.ofSeq

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

                if acc.Output = outputId && acc.Digest = r.ResultDigest then
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
                elif m.Id <> acc.Node then
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
                            OtherDigest = r.ResultDigest
                            Source = DivergenceSource.Result
                        }

                checkDone now
            | _ -> reject m.Id "unknown_unit" MessageKind.Result

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

                // Batches of at most LogBatchBytes of input (at least one entry each).
                let batches = ResizeArray<LogEntry list>()
                let current = ResizeArray<LogEntry>()
                let mutable size = 0

                for e in entries do
                    if current.Count > 0 && size + e.Input.Length > config.LogBatchBytes then
                        batches.Add(List.ofSeq current)
                        current.Clear()
                        size <- 0

                    current.Add e
                    size <- size + e.Input.Length

                batches.Add(List.ofSeq current)

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
                            Attr.str TelemetryKeys.Node "coord"
                        ]
                    )

            c.SetTimer(c.Now + config.TickNs, tickTag)

        member _.OnFrame(c: IActorContext, frame: Frame) =
            match outcome with
            | Some o ->
                // The run is over: whoever still talks to us missed the Shutdown (it is sent once and may be lost).
                if frame.Sender <> 0UL then
                    let status =
                        match o.Status with
                        | RunStatus.Completed -> "completed"
                        | RunStatus.Diverged -> "diverged"
                        | RunStatus.Aborted _ -> "aborted"

                    notice frame.Sender NoticeCode.Shutdown 0UL status
            | None ->
                let now = c.Now

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
                        | true, m ->
                            match msg with
                            | Message.Result r -> onResult now m r
                            | Message.Ready r -> onReady now m r
                            | Message.Fetch f -> onFetch m f
                            | Message.Notice n when n.Code = NoticeCode.Leave ->
                                removeMember m.Id "leave"
                                epoch <- epoch + 1UL
                                membershipChanged now "leave"
                            | Message.Notice n when n.Code = NoticeCode.WrongClass && n.Value <= uint64 UInt32.MaxValue ->
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
