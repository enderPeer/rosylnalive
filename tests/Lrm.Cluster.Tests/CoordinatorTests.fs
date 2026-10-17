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
type private Rig(replicas: int, minNodes: int) =
    let script = Script()

    let coordinator =
        Coordinator(
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

    member r.Hello(id: uint64) =
        r.From
            id
            0UL
            (Message.Hello
                {
                    Name = "n" + Invariant.u64 id
                    Incarnation = 1UL
                    Class = FakeJob.classId
                    CanHost = [ 0u; 1u; 2u ]
                    Heads = []
                })

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
        ]
