/// The coordinator driven frame by frame through a scripted context: epochs, stale-epoch rejection, unknown members,
/// Ready-gated deadlines, reissue, first-result-wins and the divergence tripwire.
module Lrm.Cluster.Tests.CoordinatorTests

open System
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Cluster.Tests.Common

/// An IActorContext that records what the actor sends, on a settable clock.
type private Script() =
    let sent = ResizeArray<uint64 * Frame>()
    member val Now = 0L with get, set

    /// Everything sent since the last call, decoded.
    member _.Take() : (uint64 * Envelope) list =
        let xs =
            [
                for dest, f in sent ->
                    match Message.tryOfFrame f with
                    | Ok e -> dest, e
                    | Error e -> failwithf "the coordinator sent a malformed frame: %s" e
            ]

        sent.Clear()
        xs

    interface IActorContext with
        member s.Now = s.Now
        member _.Send(dest, frame) = sent.Add(dest, frame)
        member _.SetTimer(_, _) = ()

        member _.Execute(_, _, _) =
            failwith "the coordinator does no executor work"

/// A coordinator over a small fake job with a scripted context.
type private Rig(replicas: int, minNodes: int, tweak: CoordinatorConfig -> CoordinatorConfig) =
    let script = Script()

    let coordinator =
        Coordinator(
            tweak
                { CoordinatorConfig.standard with
                    Replicas = replicas
                    MinNodes = minNodes
                },
            FakeWorkload Vectors.smallJob,
            MemoryBlobStore()
        )

    let actor = coordinator :> IActor
    let seqs = Collections.Generic.Dictionary<uint64, uint64>()
    do actor.Start script
    new(replicas: int, minNodes: int) = Rig(replicas, minNodes, id)
    member _.Coordinator = coordinator
    member _.Script = script

    member _.Now
        with get () = script.Now
        and set v = script.Now <- v

    /// Delivers a message from node `id` with the given epoch.
    member _.From (id: uint64) (epoch: uint64) (m: Message) =
        let s =
            match seqs.TryGetValue id with
            | true, s -> s + 1UL
            | _ -> 1UL

        seqs[id] <- s

        actor.OnFrame(
            script,
            Message.toFrame
                {
                    Epoch = epoch
                    Sender = id
                    Seq = s
                    Flags = FrameFlags.Empty
                    Message = m
                }
        )

    member r.HelloAs (id: uint64) (name: string) (incarnation: uint64) =
        r.From
            id
            0UL
            (Message.Hello
                {
                    Name = name
                    Incarnation = incarnation
                    Class = FakeJob.classId
                    CanHost = [ 0u; 1u; 2u ]
                    Heads = []
                })

    member r.Hello(id: uint64) =
        r.HelloAs id ("n" + Invariant.u64 id) 1UL

    member r.Heartbeat (id: uint64) (epoch: uint64) (ready: bool) =
        r.From id epoch (Message.Heartbeat { IsReady = ready; Active = [] })

    member r.Ready (id: uint64) (epoch: uint64) =
        r.From
            id
            epoch
            (Message.Ready
                {
                    ReadyChain = None
                    ReadyHead = UnitId.none
                    ReadyDigest = ContentId.zero
                    ReadyOutput = BlobId.none
                })

    member _.Tick() = actor.OnTimer(script, 1UL)

let private assigns (sent: (uint64 * Envelope) list) =
    sent
    |> List.choose (fun (dest, e) ->
        match e.Message with
        | Message.Assign a -> Some(dest, a.Work)
        | _ -> None)

let private notices (sent: (uint64 * Envelope) list) =
    sent
    |> List.choose (fun (dest, e) ->
        match e.Message with
        | Message.Notice n -> Some(dest, n.Code, n.Value)
        | _ -> None)

/// The honest output of a fake unit at the start of its chain (all units here are step 0).
let private outputOf (u: WorkUnit) (input: byte[]) =
    let state =
        (FakeExecutor([ 0u; 1u; 2u ]) :> IStageExecutor).Open(u.Job, WorkUnit.chain u)

    let output = state.Apply(u, input)
    output, state.Digest

let private result (u: WorkUnit) (output: byte[]) (digest: ContentId) =
    Message.Result
        {
            ResultUnit = WorkUnit.id u
            ResultOutput = output
            ResultDigest = digest
        }

/// Honest nodes for a Rig: each computes the units assigned to it in chain order, and remembers every result so
/// that it can answer a re-sent Assign as a node answers from its chain head.
type private Honest() =
    let executor = FakeExecutor([ 0u; 1u; 2u ]) :> IStageExecutor
    let states = Collections.Generic.Dictionary<uint64 * ChainKey, IChainState>()
    let results = Collections.Generic.Dictionary<uint64 * UnitId, byte[] * ContentId>()

    /// The result of `node` for an assigned unit: computed once, then repeated.
    member _.Compute (node: uint64) (a: AssignMsg) =
        let key = node, WorkUnit.id a.Work

        match results.TryGetValue key with
        | true, r -> r
        | _ ->
            let chain = WorkUnit.chain a.Work

            let state =
                match states.TryGetValue((node, chain)) with
                | true, st -> st
                | _ ->
                    let st = executor.Open(a.Work.Job, chain)
                    states[(node, chain)] <- st
                    st

            let r = state.Apply(a.Work, a.WorkInput), state.Digest
            results[key] <- r
            r

/// Plays honest nodes against a Rig until the coordinator sends nothing more: every Assign to one of `computes` is
/// computed (in chain order, as a node would), and its Result is delivered when `deliver node unit` says so. Returns
/// everything the coordinator sent.
let private drive (rig: Rig) (honest: Honest) (computes: uint64 list) (deliver: uint64 -> WorkUnit -> bool) =
    let all = ResizeArray<uint64 * Envelope>()
    let mutable go = true

    while go do
        let sent = rig.Script.Take()
        all.AddRange sent
        go <- false

        for dest, e in sent do
            match e.Message with
            | Message.Assign a when List.contains dest computes ->
                let output, digest = honest.Compute dest a

                if deliver dest a.Work then
                    rig.From dest rig.Coordinator.Epoch (result a.Work output digest)
                    go <- true
            | _ -> ()

    List.ofSeq all

/// True for the last unit of a chain of the small job.
let private isFinal (u: WorkUnit) = int u.Step = Vectors.smallJob.Steps - 1

let tests =
    testList
        "coordinator"
        [
            testCase "each join is a new epoch; Welcome and Membership carry it"
            <| fun () ->
                let rig = Rig(2, 2)
                rig.Hello 1UL
                let first = rig.Script.Take()
                Expect.equal rig.Coordinator.Epoch 1UL "epoch after the first join"

                let welcome =
                    first
                    |> List.pick (fun (d, e) ->
                        match e.Message with
                        | Message.Welcome w -> Some(d, e.Epoch, w)
                        | _ -> None)

                Expect.equal
                    welcome
                    (1UL,
                     1UL,
                     {
                         HeartbeatMs = 100u
                         LeaseMs = 1000u
                         EchoSeq = 1UL
                     })
                    "welcome"

                Expect.isEmpty (assigns first) "no work before MinNodes members"
                rig.Hello 2UL
                let second = rig.Script.Take()
                Expect.equal rig.Coordinator.Epoch 2UL "epoch after the second join"

                let work = assigns second
                // Two lanes start at stage 0; both replicas of stage 0 get both units.
                Expect.equal (work |> List.map fst |> List.sort) [ 1UL; 1UL; 2UL; 2UL ] "assigned to both replicas"
                Expect.isTrue (second |> List.forall (fun (_, e) -> e.Epoch = 2UL)) "every frame carries epoch 2"

            testCase "a frame from a stale epoch is rejected with StaleEpoch and the membership"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                let work = assigns (rig.Script.Take())
                let _, u = work.Head
                let input = FakeJob.prompt Vectors.smallJob (int u.Lane)
                let output, digest = outputOf u input
                rig.From 1UL 0UL (result u output digest)
                let sent = rig.Script.Take()
                Expect.equal (notices sent) [ 1UL, NoticeCode.StaleEpoch, 1UL ] "stale notice with the current epoch"

                Expect.isTrue
                    (sent
                     |> List.exists (fun (_, e) ->
                         match e.Message with
                         | Message.Membership _ -> true
                         | _ -> false))
                    "membership re-sent"

                Expect.equal rig.Coordinator.AcceptedCount 0 "stale result not accepted"
                rig.From 1UL 1UL (result u output digest)
                Expect.equal rig.Coordinator.AcceptedCount 1 "current-epoch result accepted"

            testCase "a stale heartbeat renews the lease but its content is rejected"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Script.Take() |> ignore
                rig.Heartbeat 1UL 0UL true
                let sent = rig.Script.Take()

                Expect.isTrue
                    (sent
                     |> List.exists (fun (_, e) ->
                         match e.Message with
                         | Message.Lease l -> l.LeaseEcho = 2UL
                         | _ -> false))
                    "lease granted, echoing the heartbeat seq"

                Expect.equal (notices sent) [ 1UL, NoticeCode.StaleEpoch, 1UL ] "stale notice"

            testCase "frames from non-members get UnknownMember"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Script.Take() |> ignore
                rig.Ready 9UL 1UL
                Expect.equal (notices (rig.Script.Take())) [ 9UL, NoticeCode.UnknownMember, 1UL ] "unknown"
                rig.Heartbeat 9UL 1UL true
                Expect.equal (notices (rig.Script.Take())) [ 9UL, NoticeCode.UnknownMember, 1UL ] "unknown heartbeat"

            testCase "a member whose lease runs out is removed in a new epoch and its work moves"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Hello 2UL
                rig.Script.Take() |> ignore
                rig.Ready 2UL 2UL

                // Node 2 heartbeats; node 1 goes silent.
                for t in 1..12 do
                    rig.Now <- int64 t * Ns.ms 100L
                    rig.Heartbeat 2UL 2UL true
                    rig.Tick()

                Expect.equal rig.Coordinator.Epoch 3UL "lease expiry bumps the epoch"
                let moved = assigns (rig.Script.Take())
                Expect.isTrue (moved |> List.exists (fun (d, _) -> d = 2UL)) "work re-assigned to node 2"
                Expect.isFalse (moved |> List.exists (fun (d, _) -> d = 1UL)) "nothing for the departed node"

            testCase "deadlines count only after Ready: 30 s initially, then the work is reissued"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Hello 2UL
                rig.Script.Take() |> ignore
                rig.Ready 2UL 2UL

                let advanceTo (seconds: int) =
                    let target = Ns.seconds (int64 seconds)

                    while rig.Now < target do
                        rig.Now <- rig.Now + Ns.ms 500L
                        rig.Heartbeat 1UL 2UL false
                        rig.Heartbeat 2UL 2UL true
                        rig.Tick()

                // Node 1 holds stage 0 but never says Ready: a minute passes with no reissue.
                advanceTo 60

                Expect.isFalse
                    (assigns (rig.Script.Take()) |> List.exists (fun (d, _) -> d = 2UL))
                    "no reissue before Ready"

                // Ready at 60 s: the 30 s initial deadline starts now.
                rig.Ready 1UL 2UL
                advanceTo 89
                Expect.isFalse (assigns (rig.Script.Take()) |> List.exists (fun (d, _) -> d = 2UL)) "not yet at 89 s"
                advanceTo 91
                let late = assigns (rig.Script.Take())
                Expect.isTrue (late |> List.exists (fun (d, _) -> d = 2UL)) "reissued to node 2 after 30 s"
                Expect.isTrue (late |> List.exists (fun (d, _) -> d = 1UL)) "and re-sent to node 1"

            testCase "first result wins; an equal duplicate confirms it"
            <| fun () ->
                let rig = Rig(2, 2)
                rig.Hello 1UL
                rig.Hello 2UL
                let _, u = (assigns (rig.Script.Take())).Head
                let output, digest = outputOf u (FakeJob.prompt Vectors.smallJob (int u.Lane))
                rig.From 1UL 2UL (result u output digest)
                rig.From 2UL 2UL (result u output digest)
                rig.Coordinator.Abort(rig.Now, "test")
                let o = rig.Coordinator.Outcome.Value
                Expect.equal o.Units 1 "one unit accepted"
                Expect.equal o.Duplicates 1 "one duplicate"
                Expect.isEmpty o.Divergences "no divergence"

            testCase "a differing duplicate raises cluster.divergence and the tripwire stops the run"
            <| fun () ->
                let sink = InMemorySink()
                use _ = Telemetry.install (Telemetry.ofSink sink)
                let rig = Rig(2, 2)
                rig.Hello 1UL
                rig.Hello 2UL
                let _, u = (assigns (rig.Script.Take())).Head
                let output, digest = outputOf u (FakeJob.prompt Vectors.smallJob (int u.Lane))
                rig.From 1UL 2UL (result u output digest)
                let bad = Array.copy output
                bad[0] <- bad[0] ^^^ 1uy
                rig.From 2UL 2UL (result u bad digest)
                let o = rig.Coordinator.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                let d = o.Divergences |> List.exactlyOne
                Expect.equal (d.Unit, d.FirstNode, d.OtherNode) (WorkUnit.id u, 1UL, 2UL) "who disagreed"
                Expect.equal d.OtherOutput (Some(BlobId.ofBytes bad)) "the other output"

                Expect.isTrue
                    (sink.Records |> List.exists (fun r -> r.Name = ClusterNames.Divergence))
                    "cluster.divergence emitted"

                Expect.equal
                    (notices (rig.Script.Take()) |> List.map (fun (d, c, _) -> d, c))
                    [ 1UL, NoticeCode.Shutdown; 2UL, NoticeCode.Shutdown ]
                    "every member is shut down"

            testCase "after the run, every frame is answered with Shutdown (a lost Shutdown is repaired)"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Script.Take() |> ignore
                rig.Coordinator.Abort(rig.Now, "test")
                rig.Script.Take() |> ignore
                rig.Heartbeat 1UL 1UL true
                Expect.equal (notices (rig.Script.Take())) [ 1UL, NoticeCode.Shutdown, 0UL ] "heartbeat answered"
                rig.Hello 5UL
                Expect.equal (notices (rig.Script.Take())) [ 5UL, NoticeCode.Shutdown, 0UL ] "even a newcomer"

            testCase "a node that answers WrongClass loses the stage and its work moves"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Hello 2UL
                let first = assigns (rig.Script.Take())
                Expect.isTrue (first |> List.forall (fun (d, _) -> d = 1UL)) "stage 0 starts on node 1"

                rig.From
                    1UL
                    2UL
                    (Message.Notice
                        {
                            Code = NoticeCode.WrongClass
                            Value = 0UL
                            Detail = "cannot run this unit"
                        })

                let moved = assigns (rig.Script.Take())
                Expect.equal (moved |> List.map fst) [ 2UL; 2UL ] "both stage-0 units go to node 2"

            testCase "a replayed chain whose digest differs is a divergence"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                let _, u = (assigns (rig.Script.Take())).Head
                let output, digest = outputOf u (FakeJob.prompt Vectors.smallJob (int u.Lane))
                rig.From 1UL 1UL (result u output digest)

                rig.From
                    1UL
                    1UL
                    (Message.Ready
                        {
                            ReadyChain = Some(WorkUnit.chain u)
                            ReadyHead = WorkUnit.id u
                            ReadyDigest = Hash.sha256Utf8 "wrong"
                            ReadyOutput = BlobId.ofBytes output
                        })

                let o = rig.Coordinator.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                Expect.equal (o.Divergences.Head.Source) DivergenceSource.Replay "replay divergence"

            testCase "Fetch returns the accepted chain up to the requested unit"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                let _, u = (assigns (rig.Script.Take())).Head
                let input = FakeJob.prompt Vectors.smallJob (int u.Lane)
                let output, digest = outputOf u input
                rig.From 1UL 1UL (result u output digest)
                rig.Script.Take() |> ignore

                rig.From
                    1UL
                    1UL
                    (Message.Fetch
                        {
                            FetchChain = WorkUnit.chain u
                            From = UnitId.none
                            To = WorkUnit.id u
                        })

                let logs =
                    rig.Script.Take()
                    |> List.choose (fun (_, e) ->
                        match e.Message with
                        | Message.Log l -> Some l
                        | _ -> None)

                let log = logs |> List.exactlyOne
                Expect.isTrue log.Final "final"
                Expect.isFalse log.Reset "no reset from the start"

                Expect.equal
                    log.Entries
                    [
                        {
                            Unit = u
                            Input = input
                            Output = BlobId.ofBytes output
                            StateDigest = digest
                        }
                    ]
                    "the one accepted unit"

            testCase "a network duplicate of the accepted result checks nothing; a later different one is a divergence"
            <| fun () ->
                let rig = Rig(2, 2)
                rig.Hello 1UL
                rig.Hello 2UL
                let _, u = (assigns (rig.Script.Take())).Head
                let output, digest = outputOf u (FakeJob.prompt Vectors.smallJob (int u.Lane))
                rig.From 1UL 2UL (result u output digest)
                // The same Result again (the network duplicated it, or node 1 answered a re-sent Assign).
                rig.From 1UL 2UL (result u output digest)
                rig.From 2UL 2UL (result u output digest)
                let bad = Array.copy output
                bad[0] <- bad[0] ^^^ 1uy
                // The accepted node itself now reports something else (a node keeps its id across restarts).
                rig.From 1UL 2UL (result u bad digest)
                let o = rig.Coordinator.Outcome.Value
                Expect.equal o.Duplicates 1 "only node 2's result confirms the unit"
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                let d = o.Divergences |> List.exactlyOne
                Expect.equal (d.FirstNode, d.OtherNode, d.Source) (1UL, 1UL, DivergenceSource.Result) "the same node"

            testCase "a stale-epoch result for an accepted unit is rejected but still compared with it"
            <| fun () ->
                let setup () =
                    let rig = Rig(2, 2)
                    rig.Hello 1UL
                    rig.Hello 2UL
                    let _, u = (assigns (rig.Script.Take())).Head
                    let output, digest = outputOf u (FakeJob.prompt Vectors.smallJob (int u.Lane))
                    rig.From 1UL 2UL (result u output digest)
                    rig.Script.Take() |> ignore
                    rig, u, output, digest

                // Equal: a confirmation, though the frame is rejected as stale.
                let rig, u, output, digest = setup ()
                rig.From 2UL 1UL (result u output digest)
                Expect.equal (notices (rig.Script.Take())) [ 2UL, NoticeCode.StaleEpoch, 2UL ] "stale notice"
                rig.Coordinator.Abort(rig.Now, "test")
                let o = rig.Coordinator.Outcome.Value
                Expect.equal (o.Duplicates, o.Rejected) (1, 1) "confirmed, and counted as rejected"
                Expect.equal (o.Chains |> List.map _.ConfirmedBy) [ [ 1UL; 2UL ] ] "confirmed by both"

                // Different: a divergence.
                let rig, u, output, digest = setup ()
                let bad = Array.copy output
                bad[0] <- bad[0] ^^^ 1uy
                rig.From 2UL 1UL (result u bad digest)
                let o = rig.Coordinator.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                Expect.equal (o.Divergences |> List.exactlyOne).OtherNode 2UL "the stale sender"

            testCase "Hello names are unique safe names other than coord; frames from member 0 are never answered"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.HelloAs 1UL "N1" 1UL
                rig.HelloAs 2UL "coord" 1UL
                Expect.isEmpty (rig.Script.Take()) "no Welcome for a bad name"
                Expect.equal rig.Coordinator.Epoch 0UL "no member yet"
                rig.HelloAs 3UL "n3" 1UL
                rig.Script.Take() |> ignore
                Expect.equal rig.Coordinator.Epoch 1UL "n3 joined"
                rig.HelloAs 4UL "n3" 1UL
                Expect.isEmpty (rig.Script.Take()) "a second id with the same name is refused"
                rig.HelloAs 3UL "n3" 2UL
                rig.Script.Take() |> ignore
                Expect.equal rig.Coordinator.Epoch 2UL "a restart keeps its name"
                // Member 0 is the coordinator: an answer would be addressed to itself.
                rig.Ready 0UL 2UL
                rig.Heartbeat 0UL 2UL true
                rig.HelloAs 0UL "n0" 1UL
                Expect.isEmpty (rig.Script.Take()) "nothing sent"
                rig.Coordinator.Abort(rig.Now, "test")
                Expect.equal rig.Coordinator.Outcome.Value.Rejected 6 "six rejected frames"

            testCase "a chain Ready is checked against the accepted unit: a match confirms, another output diverges"
            <| fun () ->
                let setup () =
                    let rig = Rig(1, 2)
                    rig.Hello 1UL
                    rig.Hello 2UL
                    let dest, u = (assigns (rig.Script.Take())).Head
                    let output, digest = outputOf u (FakeJob.prompt Vectors.smallJob (int u.Lane))
                    rig.From dest 2UL (result u output digest)
                    let other = if dest = 1UL then 2UL else 1UL

                    let ready (o: BlobId) =
                        Message.Ready
                            {
                                ReadyChain = Some(WorkUnit.chain u)
                                ReadyHead = WorkUnit.id u
                                ReadyDigest = digest
                                ReadyOutput = o
                            }

                    rig, dest, other, ready, BlobId.ofBytes output

                let rig, dest, other, ready, outputId = setup ()
                rig.From other 2UL (ready outputId)
                rig.Coordinator.Abort(rig.Now, "test")
                let o = rig.Coordinator.Outcome.Value

                Expect.equal
                    (o.Chains |> List.map _.ConfirmedBy)
                    [ List.sort [ dest; other ] ]
                    "confirmed by the replay"

                let rig, dest, other, ready, outputId = setup ()
                let wrong = BlobId.ofBytes "wrong"B
                rig.From other 2UL (ready wrong)
                let o = rig.Coordinator.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                let d = o.Divergences |> List.exactlyOne

                Expect.equal
                    (d.Source, d.FirstNode, d.OtherNode, d.FirstOutput, d.OtherOutput)
                    (DivergenceSource.Replay, dest, other, outputId, Some wrong)
                    "a replay divergence on the output alone"

            testCase "the drain asks again: a replica that never answers leaves the run Unverified"
            <| fun () ->
                let run (answerInDrain: bool) =
                    let rig = Rig(2, 2)
                    let honest = Honest()
                    rig.Hello 1UL
                    rig.Hello 2UL
                    // Both compute everything; node 2's results for the final units are lost.
                    drive rig honest [ 1UL; 2UL ] (fun node u -> node = 1UL || not (isFinal u))
                    |> ignore

                    Expect.equal rig.Coordinator.AcceptedCount 18 "every unit accepted"
                    let asked = ResizeArray<WorkUnit>()

                    for t in 1..50 do
                        if rig.Coordinator.Outcome.IsNone then
                            rig.Now <- int64 t * Ns.ms 50L
                            rig.Heartbeat 1UL rig.Coordinator.Epoch true
                            rig.Heartbeat 2UL rig.Coordinator.Epoch true
                            rig.Tick()
                            let sent = drive rig honest [ 2UL ] (fun _ _ -> answerInDrain)

                            for d, u in assigns sent do
                                if d = 2UL then
                                    asked.Add u

                    rig.Coordinator.Outcome.Value, List.ofSeq asked

                let o, asked = run false
                Expect.equal o.Status RunStatus.Unverified "unverified"
                Expect.equal o.RunDigest (reference Vectors.smallJob).RunDigest "the digest is still right"
                Expect.equal o.Unconfirmed 6 "every chain's final state"

                for c in o.Chains do
                    Expect.equal (c.ConfirmedBy, c.UnconfirmedBy) ([ 1UL ], [ 2UL ]) "node 2 never answered"

                Expect.isTrue (asked |> List.forall isFinal) "only final units are asked again"
                Expect.isGreaterThanOrEqual asked.Length 18 "every final unit, three times"

                let o, _ = run true
                Expect.equal o.Status RunStatus.Completed "node 2 answered from its chain heads"
                Expect.equal o.Unconfirmed 0 "nothing unconfirmed"

                for c in o.Chains do
                    Expect.equal c.ConfirmedBy [ 1UL; 2UL ] "both replicas"

            testCase "a deadline sample needs a started deadline"
            <| fun () ->
                let sink = InMemorySink()
                use _ = Telemetry.install (Telemetry.ofSink sink)
                let rig = Rig(1, 1)
                let honest = Honest()
                rig.Hello 1UL
                rig.Now <- Ns.ms 300L
                // Node 1 never said Ready: its deadlines never started, so its first result is no sample.
                let _, u = (assigns (rig.Script.Take())).Head

                let a =
                    {
                        Work = u
                        WorkInput = FakeJob.prompt Vectors.smallJob (int u.Lane)
                    }

                let output, digest = honest.Compute 1UL a
                rig.From 1UL 1UL (result u output digest)

                let latency () =
                    sink.Records
                    |> List.filter (fun x -> x.Name = ClusterNames.UnitLatency)
                    |> List.length

                Expect.equal (latency ()) 0 "no sample without a deadline start"

                Expect.isTrue
                    (sink.Records |> List.exists (fun x -> x.Name = ClusterNames.UnitDone))
                    "the unit is still on the Gantt chart"

                rig.Ready 1UL 1UL
                rig.Now <- Ns.ms 400L
                drive rig honest [ 1UL ] (fun _ _ -> true) |> ignore
                Expect.isGreaterThan (latency ()) 0 "samples once Ready"

            testCase "Log batches are bounded by their encoded size, not just their inputs"
            <| fun () ->
                let rig = Rig(1, 1, (fun c -> { c with LogBatchBytes = 300 }))
                let honest = Honest()
                rig.Hello 1UL
                // Lane 1 never finishes, so the run is still on when the Fetch comes.
                let sent = drive rig honest [ 1UL ] (fun _ u -> not (u.Lane = 1u && isFinal u))

                let chain =
                    [
                        for _, u in assigns sent do
                            if u.Lane = 0u && u.Stage = 0u then
                                yield u
                    ]
                    |> List.distinctBy WorkUnit.id
                    |> List.sortBy _.Step

                Expect.equal chain.Length 3 "chain (0, 0) has three units"

                rig.From
                    1UL
                    1UL
                    (Message.Fetch
                        {
                            FetchChain = { Lane = 0u; Stage = 0u }
                            From = UnitId.none
                            To = WorkUnit.id (List.last chain)
                        })

                let logs =
                    rig.Script.Take()
                    |> List.choose (fun (_, e) ->
                        match e.Message with
                        | Message.Log l -> Some l
                        | _ -> None)

                let entries = logs |> List.collect _.Entries
                Expect.equal (entries |> List.map _.Unit) chain "every unit, in order"
                Expect.isLessThan (entries |> List.sumBy _.Input.Length) 300 "the inputs alone would fit one batch"
                Expect.equal logs.Length 3 "but each encoded entry takes most of a batch"

                for l in logs do
                    Expect.isTrue
                        (l.Entries.Length = 1 || (l.Entries |> List.sumBy Message.logEntrySize) <= 300)
                        "within the bound"

                Expect.equal (logs |> List.map _.Final) [ false; false; true ] "the last batch is final"

                Expect.throws
                    (fun () ->
                        Coordinator(
                            { CoordinatorConfig.standard with
                                LogBatchBytes = FrameCodec.MaxPayload
                            },
                            FakeWorkload Vectors.smallJob,
                            MemoryBlobStore()
                        )
                        |> ignore)
                    "a batch must leave room for the Log header"

            testCase "deadlines survive membership changes, so churn never postpones a reissue"
            <| fun () ->
                let rig = Rig(1, 1)
                rig.Hello 1UL
                rig.Ready 1UL 1UL

                let stuck =
                    assigns (rig.Script.Take()) |> List.map (snd >> WorkUnit.id) |> Set.ofList

                let joined = ResizeArray<uint64>([ 1UL ])
                let moved = ResizeArray<uint64 * UnitId>()

                // Node 1 holds the work and never answers. A node joins every 5 s, each one a new epoch.
                for t in 1..62 do
                    rig.Now <- int64 t * Ns.ms 500L

                    if t % 10 = 0 then
                        let id = uint64 (t / 10 + 1)
                        rig.Hello id
                        rig.Ready id rig.Coordinator.Epoch
                        joined.Add id

                    for n in joined do
                        rig.Heartbeat n rig.Coordinator.Epoch true

                    rig.Tick()

                    for d, u in assigns (rig.Script.Take()) do
                        if d <> 1UL && stuck.Contains(WorkUnit.id u) then
                            moved.Add(d, WorkUnit.id u)

                Expect.isGreaterThan rig.Coordinator.Epoch 6UL "six joins"
                Expect.isNonEmpty moved "the stuck work was reissued 30 s after Ready, despite the joins"
        ]
