namespace Lrm.Cluster

open System
open System.Collections.Generic
open System.Threading.Tasks
open Lrm.Core

/// Node settings.
type NodeConfig =
    {
        /// The member id (1 or more; 0 is the coordinator).
        Id: uint64
        /// The node's name, a SafeName such as "n1".
        Name: string
        /// Distinguishes restarts of one id.
        Incarnation: uint64
        /// The heartbeat interval until Welcome says otherwise: 100 ms.
        HeartbeatNs: int64
        /// How long to wait for a Log before asking again: 1 s.
        FetchRetryNs: int64
        /// Time until the executor is loaded and the node is Ready (0 = ready at once).
        ColdStartNs: int64
    }

[<RequireQualifiedAccess>]
module NodeConfig =
    /// A node with the contract timings that is ready at once.
    let create (id: uint64) (name: string) (incarnation: uint64) =
        if id = 0UL then
            invalidArg "id" "member id 0 is the coordinator"

        {
            Id = id
            Name = SafeName.check "node name" name
            Incarnation = incarnation
            HeartbeatNs = Ns.ms 100L
            FetchRetryNs = Ns.seconds 1L
            ColdStartNs = 0L
        }

/// What a node reports about itself at the end of a run.
type NodeReport =
    {
        Id: uint64
        Name: string
        Incarnation: uint64
        /// Every chain the node holds, sorted by (lane, stage).
        Chains: ChainHead list
        /// RunDigest.replicas over Chains.
        ReplicaDigest: ContentId
        Executed: int
        Replayed: int
    }

[<Sealed; AllowNullLiteral>]
type internal ChainSlot(job: ContentId, chain: ChainKey, state: IChainState) =
    member _.Job = job
    member _.Chain = chain
    member _.State = state
    member val Head = UnitId.none with get, set
    member val HeadDigest = state.Digest with get, set
    member val LastOutput: byte[] = [||] with get, set
    member val Applied = HashSet<UnitId>()
    /// A Fetch is outstanding; the slot's queued units wait.
    member val Waiting = false with get, set
    member val FetchTarget = UnitId.none with get, set
    member val FetchSentAt = 0L with get, set
    /// Log entries received and not yet replayed.
    member val PendingLog = ResizeArray<LogEntry>()

type internal WorkItem =
    {
        Id: UnitId
        Unit: WorkUnit
        Input: byte[]
        Slot: ChainSlot
    }

type internal Busy =
    | Idle
    | Executing of WorkItem
    | Replaying of ChainSlot

type internal WorkOutcome =
    | Executed of output: byte[] * digest: ContentId
    | Replayed of applied: UnitId list * head: UnitId * digest: ContentId * lastOutput: byte[] * broken: bool

/// A node (ClusterProtocol contract): joins with Hello, heartbeats, runs assigned units one at a time on its
/// executor, replays a chain from the coordinator's unit log before executing a unit whose Prev it does not hold,
/// answers re-sent assignments from its chain heads, and rejoins when its lease runs out.
[<Sealed>]
type Node(config: NodeConfig, executor: IStageExecutor) =
    let heartbeatTag = 1UL
    let readyTag = 2UL
    let classId = ExecutionClass.id executor.Class
    let canHost = Set.ofList executor.CanHost
    let slots = Dictionary<ChainKey, ChainSlot>()
    let queue = ResizeArray<WorkItem>()
    let sentAt = Dictionary<uint64, int64>()
    let sentOrder = Queue<uint64>()

    let stopped =
        TaskCompletionSource<NodeReport>(TaskCreationOptions.RunContinuationsAsynchronously)

    let mutable busy = Idle
    let mutable token = 0UL
    let mutable isMember = false
    let mutable epoch = 0UL
    let mutable leaseUntil = 0L
    let mutable heartbeatNs = config.HeartbeatNs
    let mutable ready = false
    let mutable halted = false
    let mutable seq = 0UL
    let mutable executed = 0
    let mutable replayed = 0
    let mutable tel = Telemetry.noop
    let mutable ctx = Unchecked.defaultof<IActorContext>

    let sortedSlots () =
        slots.Values |> Seq.sortBy (fun s -> s.Chain.Lane, s.Chain.Stage) |> List.ofSeq

    let send (flags: uint32) (msg: Message) =
        seq <- seq + 1UL

        ctx.Send(
            0UL,
            Message.toFrame
                {
                    Epoch = epoch
                    Sender = config.Id
                    Seq = seq
                    Flags = flags
                    Message = msg
                }
        )

        seq

    let remember (s: uint64) (now: int64) =
        sentAt[s] <- now
        sentOrder.Enqueue s

        while sentOrder.Count > 128 do
            sentAt.Remove(sentOrder.Dequeue()) |> ignore

    let heads () =
        [
            for s in sortedSlots () do
                if not (UnitId.isNone s.Head) then
                    yield
                        {
                            Chain = s.Chain
                            Head = s.Head
                            Digest = s.HeadDigest
                        }
        ]

    let report () =
        let chains = heads ()

        {
            Id = config.Id
            Name = config.Name
            Incarnation = config.Incarnation
            Chains = chains
            ReplicaDigest = RunDigest.replicas chains
            Executed = executed
            Replayed = replayed
        }

    let hello (now: int64) =
        let s =
            send
                FrameFlags.Empty
                (Message.Hello
                    {
                        Name = config.Name
                        Incarnation = config.Incarnation
                        Class = classId
                        CanHost = executor.CanHost
                        Heads = heads ()
                    })

        remember s now

    let sendFetch (now: int64) (slot: ChainSlot) =
        slot.FetchSentAt <- now

        send
            FrameFlags.Empty
            (Message.Fetch
                {
                    FetchChain = slot.Chain
                    From = slot.Head
                    To = slot.FetchTarget
                })
        |> ignore

    let adopt (now: int64) (e: uint64) =
        if e > epoch then
            epoch <- e

            // Frames of the old epoch are rejected, so ask again for logs still outstanding.
            for slot in sortedSlots () do
                if slot.Waiting then
                    sendFetch now slot

    let sendResult (id: UnitId) (output: byte[]) (digest: ContentId) =
        send
            FrameFlags.Empty
            (Message.Result
                {
                    ResultUnit = id
                    ResultOutput = output
                    ResultDigest = digest
                })
        |> ignore

    let sendReady (chain: ChainKey option) (head: UnitId) (digest: ContentId) =
        send
            FrameFlags.Empty
            (Message.Ready
                {
                    ReadyChain = chain
                    ReadyHead = head
                    ReadyDigest = digest
                })
        |> ignore

    let phaseOf (item: WorkItem) =
        match busy with
        | Replaying s when obj.ReferenceEquals(s, item.Slot) -> UnitPhase.Replaying
        | _ ->
            if item.Slot.Waiting || item.Slot.PendingLog.Count > 0 then
                UnitPhase.Fetching
            elif item.Slot.Head = item.Unit.Prev then
                UnitPhase.Positioned
            else
                UnitPhase.Queued

    let heartbeat (now: int64) =
        let active =
            [
                match busy with
                | Executing w -> yield w.Id, UnitPhase.Positioned
                | _ -> ()
                for item in queue -> item.Id, phaseOf item
            ]

        let s =
            send FrameFlags.Empty (Message.Heartbeat { IsReady = ready; Active = active })

        remember s now

    let slotFor (u: WorkUnit) =
        let key = WorkUnit.chain u

        match slots.TryGetValue key with
        | true, s when s.Job = u.Job -> s
        | found, old ->
            if found then
                old.State.Dispose()

            let s = ChainSlot(u.Job, key, executor.Open(u.Job, key))
            slots[key] <- s
            s

    let resetSlot (slot: ChainSlot) =
        let fresh = ChainSlot(slot.Job, slot.Chain, executor.Open(slot.Job, slot.Chain))
        fresh.Waiting <- slot.Waiting
        fresh.FetchTarget <- slot.FetchTarget
        fresh.FetchSentAt <- slot.FetchSentAt
        slot.State.Dispose()
        slots[slot.Chain] <- fresh

        // Queued items point at the old slot: move them.
        for i in 0 .. queue.Count - 1 do
            if obj.ReferenceEquals(queue[i].Slot, slot) then
                queue[i] <- { queue[i] with Slot = fresh }

        fresh

    let startExecute (item: WorkItem) =
        token <- token + 1UL
        busy <- Executing item
        let state = item.Slot.State
        let work = item.Unit
        let input = item.Input

        ctx.Execute(
            {
                CostStage = work.Stage
                CostUnits = 1
            },
            (fun () ->
                let output = state.Apply(work, input)
                box (Executed(output, state.Digest))),
            token
        )

    let startReplay (slot: ChainSlot) =
        let entries = List.ofSeq slot.PendingLog
        slot.PendingLog.Clear()
        token <- token + 1UL
        busy <- Replaying slot
        let state = slot.State
        let startHead = slot.Head
        let startOutput = slot.LastOutput
        let skip = HashSet<UnitId>(slot.Applied)

        ctx.Execute(
            {
                CostStage = slot.Chain.Stage
                CostUnits = entries.Length
            },
            (fun () ->
                let applied = ResizeArray<UnitId>()
                let mutable head = startHead
                let mutable last = startOutput
                let mutable broken = false

                for e in entries do
                    let id = WorkUnit.id e.Unit

                    if not broken && not (skip.Contains id) then
                        if e.Unit.Prev = head then
                            last <- state.Apply(e.Unit, e.Input)
                            head <- id
                            applied.Add id
                        else
                            broken <- true

                box (Replayed(List.ofSeq applied, head, state.Digest, last, broken))),
            token
        )

    /// Starts the next piece of work when idle: a received log first, then the first queued unit that can run.
    let rec pump (now: int64) =
        match busy with
        | Idle when ready && not halted ->
            match sortedSlots () |> List.tryFind (fun s -> s.PendingLog.Count > 0) with
            | Some slot -> startReplay slot
            | None ->
                let mutable i = 0
                let mutable started = false

                while not started && i < queue.Count do
                    let item = queue[i]
                    let slot = item.Slot

                    if slot.Applied.Contains item.Id then
                        queue.RemoveAt i

                        if slot.Head = item.Id then
                            sendResult item.Id slot.LastOutput slot.HeadDigest
                    elif slot.Waiting then
                        i <- i + 1
                    elif slot.Head = item.Unit.Prev then
                        queue.RemoveAt i
                        startExecute item
                        started <- true
                    else
                        // Replay of the unit log before execution.
                        slot.Waiting <- true
                        slot.FetchTarget <- item.Unit.Prev
                        sendFetch now slot
                        i <- i + 1
        | _ -> ()

    let onAssign (now: int64) (a: AssignMsg) =
        let u = a.Work
        let id = WorkUnit.id u

        if u.Class <> classId || not (canHost.Contains u.Stage) then
            send
                FrameFlags.Empty
                (Message.Notice
                    {
                        Code = NoticeCode.WrongClass
                        Value = uint64 u.Stage
                        Detail = "cannot run this unit"
                    })
            |> ignore
        else
            let slot = slotFor u

            if slot.Applied.Contains id then
                if slot.Head = id then
                    sendResult id slot.LastOutput slot.HeadDigest
            else
                let inProgress =
                    match busy with
                    | Executing w -> w.Id = id
                    | _ -> false

                if not inProgress && not (queue.Exists(fun w -> w.Id = id)) then
                    queue.Add
                        {
                            Id = id
                            Unit = u
                            Input = a.WorkInput
                            Slot = slot
                        }

                    pump now

    let onLog (now: int64) (log: LogMsg) =
        match slots.TryGetValue log.LogChain with
        | true, slot0 when slot0.Waiting ->
            let replayingThis =
                match busy with
                | Replaying s -> obj.ReferenceEquals(s, slot0)
                | _ -> false

            if replayingThis then
                // A later batch of the same reply: replay it after the current one. A stale Reset is ignored.
                if not log.Reset then
                    slot0.PendingLog.AddRange log.Entries
            else
                let slot =
                    if log.Reset then
                        slot0.PendingLog.Clear()
                        resetSlot slot0
                    else
                        slot0

                slot.PendingLog.AddRange log.Entries

                if slot.PendingLog.Count = 0 && log.Final then
                    if slot.Head = slot.FetchTarget then
                        slot.Waiting <- false
                        sendReady (Some slot.Chain) slot.Head slot.HeadDigest
                    else
                        // The log cannot reach the target from our head: start over from the beginning.
                        let fresh = resetSlot slot
                        sendFetch now fresh

                pump now
        | _ -> ()

    let onWorkDone (now: int64) (t: uint64) (result: obj) =
        if t = token then
            match result with
            | :? exn as e -> raise (InvalidOperationException(sprintf "node %s: executor failed" config.Name, e))
            | _ -> ()

            match busy, result with
            | Executing item, (:? WorkOutcome as o) ->
                match o with
                | Executed(output, digest) ->
                    let slot = item.Slot
                    slot.Head <- item.Id
                    slot.HeadDigest <- digest
                    slot.LastOutput <- output
                    slot.Applied.Add item.Id |> ignore
                    executed <- executed + 1
                    busy <- Idle
                    sendResult item.Id output digest
                | Replayed _ -> invalidOp "replay result for an execution"
            | Replaying slot, (:? WorkOutcome as o) ->
                match o with
                | Replayed(applied, head, digest, lastOutput, broken) ->
                    for id in applied do
                        slot.Applied.Add id |> ignore

                    slot.Head <- head
                    slot.HeadDigest <- digest
                    slot.LastOutput <- lastOutput
                    replayed <- replayed + applied.Length
                    busy <- Idle

                    if tel.Enabled && not applied.IsEmpty then
                        tel.Counter(
                            ClusterNames.ReplayUnits,
                            float applied.Length,
                            TelemetryUnits.Count,
                            [
                                Attr.int "lane" (int slot.Chain.Lane)
                                Attr.int TelemetryKeys.Stage (int slot.Chain.Stage)
                            ]
                        )

                    if broken then
                        slot.PendingLog.Clear()
                        let fresh = resetSlot slot
                        fresh.Waiting <- true
                        sendFetch now fresh
                    elif slot.Waiting && slot.Head = slot.FetchTarget then
                        slot.Waiting <- false
                        slot.PendingLog.Clear()
                        sendReady (Some slot.Chain) slot.Head slot.HeadDigest
                | Executed _ -> invalidOp "execution result for a replay"
            | _ -> ()

            pump now

    /// Completes with the node's report when the coordinator shuts the run down (from any thread).
    member _.Stopped: Task<NodeReport> = stopped.Task

    /// The node's report (actor thread only).
    member _.Report() = report ()

    /// True while the node is a member of the current epoch, as far as it knows.
    member _.IsMember = isMember

    /// A multi-line snapshot of the node's state for liveness diagnostics. Safe to call from any thread, but off the
    /// actor thread the snapshot may be inconsistent.
    member _.Describe() : string =
        try
            let b = Text.StringBuilder()
            let line (s: string) = b.AppendLine s |> ignore

            let busyText =
                match busy with
                | Idle -> "idle"
                | Executing w -> sprintf "executing %d/%d/%d" w.Unit.Lane w.Unit.Stage w.Unit.Step
                | Replaying s -> sprintf "replaying (%d, %d)" s.Chain.Lane s.Chain.Stage

            line (
                sprintf
                    "node %s: member %b, epoch %d, ready %b, halted %b, %s, queue %d, executed %d, replayed %d"
                    config.Name
                    isMember
                    epoch
                    ready
                    halted
                    busyText
                    queue.Count
                    executed
                    replayed
            )

            for w in List.ofSeq queue do
                line (
                    sprintf
                        "  queued %d/%d/%d %s phase %A"
                        w.Unit.Lane
                        w.Unit.Stage
                        w.Unit.Step
                        w.Id.Value.Short
                        (phaseOf w)
                )

            for s in sortedSlots () do
                line (
                    sprintf
                        "  chain (%d, %d) head %s waiting %b target %s log %d"
                        s.Chain.Lane
                        s.Chain.Stage
                        s.Head.Value.Short
                        s.Waiting
                        s.FetchTarget.Value.Short
                        s.PendingLog.Count
                )

            b.ToString()
        with e ->
            sprintf "node %s: snapshot failed: %s" config.Name e.Message

    interface IActor with
        member _.Id = config.Id

        member _.Start(c: IActorContext) =
            ctx <- c

            tel <-
                (Telemetry.current ())
                    .Child(
                        [
                            Attr.str TelemetryKeys.Component "cluster.node"
                            Attr.str TelemetryKeys.Node config.Name
                        ]
                    )

            let now = c.Now

            if config.ColdStartNs <= 0L then
                ready <- true
            else
                c.SetTimer(now + config.ColdStartNs, readyTag)

            hello now
            c.SetTimer(now + heartbeatNs, heartbeatTag)

        member _.OnTimer(c: IActorContext, tag: uint64) =
            let now = c.Now

            if not halted then
                if tag = heartbeatTag then
                    if not isMember then
                        hello now
                    elif now > leaseUntil then
                        isMember <- false

                        if tel.Enabled then
                            tel.Event(ClusterNames.LeaseLost, [])

                        hello now
                    else
                        heartbeat now

                    for slot in sortedSlots () do
                        if slot.Waiting && now - slot.FetchSentAt >= config.FetchRetryNs then
                            sendFetch now slot

                    c.SetTimer(now + heartbeatNs, heartbeatTag)
                elif tag = readyTag then
                    ready <- true

                    if isMember then
                        sendReady None UnitId.none ContentId.zero

                    pump now

        member _.OnFrame(c: IActorContext, frame: Frame) =
            let now = c.Now

            if not halted && frame.Sender = 0UL then
                match Message.tryOfFrame frame with
                | Error _ ->
                    if tel.Enabled then
                        tel.Counter(
                            ClusterNames.FrameRejected,
                            1.0,
                            TelemetryUnits.Count,
                            [ Attr.str "reason" "malformed"; Attr.str "kind" (MessageKind.name frame.Kind) ]
                        )
                | Ok env ->
                    match env.Message with
                    | Message.Welcome w ->
                        adopt now env.Epoch
                        let wasMember = isMember
                        isMember <- true
                        heartbeatNs <- max (Ns.ms 1L) (Ns.ms (int64 w.HeartbeatMs))

                        let from =
                            match sentAt.TryGetValue w.EchoSeq with
                            | true, t -> t
                            | _ -> now

                        leaseUntil <- max leaseUntil (from + Ns.ms (int64 w.LeaseMs))

                        if ready && not wasMember then
                            sendReady None UnitId.none ContentId.zero
                    | Message.Lease l ->
                        adopt now env.Epoch

                        match sentAt.TryGetValue l.LeaseEcho with
                        | true, t -> leaseUntil <- max leaseUntil (t + Ns.ms (int64 l.DurationMs))
                        | _ -> ()
                    | Message.Membership m ->
                        if env.Epoch >= epoch then
                            adopt now env.Epoch

                            if isMember && not (m.Members |> List.exists (fun x -> x.MemberId = config.Id)) then
                                // Expelled (our lease ran out at the coordinator): rejoin.
                                isMember <- false
                                hello now
                    | Message.Assign a ->
                        if env.Epoch >= epoch then
                            adopt now env.Epoch
                            onAssign now a
                    | Message.Log log ->
                        if env.Epoch >= epoch then
                            adopt now env.Epoch
                            onLog now log
                    | Message.Notice n ->
                        match n.Code with
                        | NoticeCode.StaleEpoch -> adopt now n.Value
                        | NoticeCode.UnknownMember ->
                            adopt now n.Value

                            if isMember then
                                isMember <- false

                            hello now
                        | NoticeCode.Shutdown ->
                            halted <- true
                            stopped.TrySetResult(report ()) |> ignore
                        | _ -> ()
                    | _ -> ()

        member _.OnWorkDone(c: IActorContext, t: uint64, result: obj) =
            if not halted then
                onWorkDone c.Now t result
