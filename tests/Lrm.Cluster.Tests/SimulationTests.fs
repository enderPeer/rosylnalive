/// The C1 acceptance gates in the deterministic simulator: 200 chaos seeds bit-identical with 2 replicas per stage, a
/// stable trace hash over 100 runs, loss of both replicas recovered by replay, the divergence tripwire, Ready-gated
/// deadlines, reissue of lagging work, rejoin after a partition, and telemetry that changes nothing.
module Lrm.Cluster.Tests.SimulationTests

open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Cluster.Tests.Common

let private standardRef () = reference FakeJob.standard

/// Seven nodes ready at 10 ms on a clean LAN; roles are n1, n2 (stage 0), n3, n4 (stage 1), n5, n6 (stage 2) and the
/// spare n7.
let private seven (seed: uint64) (faults: Fault list) =
    { Simulation.clean seed 7 with
        Faults = faults
        Coordinator =
            { CoordinatorConfig.standard with
                MinNodes = 7
            }
    }

let private expectReference (what: string) (r: SimResult) =
    match r.Outcome with
    | None -> failtestf "%s: %s" what (describe r)
    | Some o ->
        Expect.equal o.Status RunStatus.Completed (sprintf "%s: %s" what (describe r))
        Expect.equal o.RunDigest (standardRef ()).RunDigest (sprintf "%s: run digest (%s)" what (describe r))
        Expect.equal o.ReplicaDigest (standardRef ()).ReplicaDigest (sprintf "%s: replica digest" what)
        Expect.isEmpty o.Divergences (sprintf "%s: divergences" what)

let tests =
    testList
        "simulation"
        [
            testCase "a fault-free run equals the reference, with every unit confirmed by both replicas"
            <| fun () ->
                let r = Simulation.run (seven 1UL [])
                expectReference "fault-free" r
                let o = r.Outcome.Value
                Expect.equal o.Units 72 "units"
                Expect.equal o.Duplicates 72 "every unit duplicated exactly once"
                Expect.equal o.Reissues 0 "no reissues"
                Expect.equal o.Epoch 7UL "one epoch per join"

                for c in o.Chains do
                    Expect.equal c.ConfirmedBy.Length 2 (sprintf "chain %A confirmed by both replicas" c.Final.Chain)

            testCase "200 chaos seeds are bit-identical with 2 replicas per stage"
            <| fun () ->
                let results = Simulation.runAll [ for s in 1UL .. 200UL -> Simulation.standard s ]
                let reference = standardRef ()
                let finals = reference.Chains |> List.map (fun c -> c.Chain, c) |> Map.ofList

                let bad =
                    results
                    |> List.filter (fun r ->
                        match r.Outcome with
                        | Some o -> o.Status <> RunStatus.Completed || o.RunDigest <> reference.RunDigest
                        | None -> true)

                Expect.isEmpty (bad |> List.map describe) "every seed matches the reference"

                for r in results do
                    let o = r.Outcome.Value

                    for c in o.Chains do
                        // Completed means verified: at least two members computed every chain's final state.
                        Expect.isGreaterThanOrEqual
                            c.ConfirmedBy.Length
                            2
                            (sprintf "seed %d chain %A confirmations" r.Seed c.Final.Chain)

                        Expect.isEmpty c.UnconfirmedBy (sprintf "seed %d chain %A unanswered" r.Seed c.Final.Chain)

                    // The replicas themselves, from the nodes' own reports: every live node that holds a chain at its
                    // final unit holds the reference state, and every chain survives on some live node.
                    for n in r.Nodes do
                        for h in n.Chains do
                            if h.Head = finals[h.Chain].Head then
                                Expect.equal
                                    h.Digest
                                    finals[h.Chain].Digest
                                    (sprintf "seed %d: %s holds chain %A" r.Seed n.Name h.Chain)

                    for KeyValue(chain, f) in finals do
                        let held =
                            r.Nodes
                            |> List.exists (fun n ->
                                n.Chains |> List.exists (fun h -> h.Chain = chain && h.Head = f.Head))

                        Expect.isTrue held (sprintf "seed %d: chain %A is held at its final unit" r.Seed chain)

                // Cross-check coverage: accepted units that a second member's result or replay confirmed.
                let units = results |> List.sumBy (fun r -> r.Outcome.Value.Units)
                let checkedUnits = results |> List.sumBy (fun r -> r.Outcome.Value.CrossChecked)

                Expect.isGreaterThanOrEqual
                    (float checkedUnits / float units)
                    0.95
                    (sprintf "%d of %d units cross-checked" checkedUnits units)

                // The seeds really were chaotic.
                Expect.isGreaterThan (results |> List.sumBy (fun r -> r.Stats.Kills)) 100 "kills"
                Expect.isGreaterThan (results |> List.sumBy (fun r -> r.Stats.Restarts)) 30 "restarts"
                Expect.isGreaterThan (results |> List.sumBy (fun r -> r.Stats.FramesDropped)) 1000L "drops"
                Expect.isGreaterThan (results |> List.sumBy (fun r -> r.Stats.FramesDuplicated)) 1000L "duplicates"
                Expect.isGreaterThan (results |> List.sumBy (fun r -> r.Outcome.Value.Replays)) 100 "replays"
                Expect.isGreaterThan (results |> List.sumBy (fun r -> r.Outcome.Value.Reissues)) 5 "reissues"
                Expect.isGreaterThan (results |> List.map (fun r -> r.Outcome.Value.Epoch) |> List.max) 8UL "epochs"

            testCase "200 chaos seeds with one corrupt replica each: a wrong result never completes"
            <| fun () ->
                // Each seed adds Corrupt(node, stage, step) drawn from its own stream to the standard chaos. Whenever
                // the corruption reaches an accepted result, a duplicate, a replay or the drain must catch it.
                let corrupt (seed: uint64) =
                    let r = DetRng(seed, DetRng.streamOf "lrm.cluster.test.corrupt")
                    let node = 1UL + uint64 (r.NextBelow 7u)
                    let stage = r.NextBelow 3u
                    let step = r.NextBelow 8u
                    Fault.Corrupt(node, stage, step)

                let results =
                    Simulation.runAll
                        [
                            for s in 1UL .. 200UL ->
                                let c = Simulation.standard s

                                { c with
                                    Faults = corrupt s :: c.Faults
                                }
                        ]

                let expected = (standardRef ()).RunDigest

                let silent =
                    results
                    |> List.filter (fun r ->
                        match r.Outcome with
                        | Some o -> o.Status <> RunStatus.Diverged && o.RunDigest <> expected
                        | None -> true)

                Expect.isEmpty (silent |> List.map describe) "no seed ends without the tripwire and with a wrong digest"

                let diverged =
                    results |> List.filter (fun r -> r.Outcome.Value.Status = RunStatus.Diverged)

                Expect.isGreaterThan diverged.Length 40 "the tripwire fired in many seeds"

            testCase "the trace hash is stable over 100 runs of one seed, and is the known answer"
            <| fun () ->
                let results = Simulation.runAll [ for _ in 1..100 -> Simulation.standard 7UL ]
                let traces = results |> List.map _.TraceHash |> List.distinct
                Expect.equal traces.Length 1 "one trace hash"
                Expect.equal (results |> List.map digestOf |> List.distinct) [ (standardRef ()).RunDigest ] "digest"

                // Pinned, so drift across processes, machines and runtimes (an iteration order, a timer) shows.
                Expect.equal traces.Head.Hex Vectors.seed7Trace "the seed-7 trace hash (cluster.md)"
                Expect.equal results.Head.Stats Vectors.seed7Stats "the seed-7 stats (cluster.md)"

                let other = Simulation.run (Simulation.standard 8UL)
                Expect.notEqual other.TraceHash traces.Head "another seed has another schedule"

            testCase "loss of both replicas of a stage is recovered by replay"
            <| fun () ->
                // n3 and n4 hold stage 1; both die at 150 ms, mid-run, and never come back.
                let r =
                    Simulation.run (seven 2UL [ Fault.Kill(3UL, Ns.ms 150L); Fault.Kill(4UL, Ns.ms 150L) ])

                expectReference "both replicas lost" r
                let o = r.Outcome.Value
                Expect.equal r.Stats.Kills 2 "two kills"
                Expect.isGreaterThan o.Replays 0 "the new replicas replayed the unit log"
                let replayed = r.Nodes |> List.sumBy _.Replayed
                Expect.isGreaterThan replayed 0 "units replayed"
                let spare = r.Nodes |> List.find (fun n -> n.Id = 7UL)
                Expect.isTrue (spare.Chains |> List.exists (fun c -> c.Chain.Stage = 1u)) "the spare took stage 1"

                // Every chain a survivor holds at the final head has the reference digest.
                let finals = (standardRef ()).Chains |> List.map (fun c -> c.Chain, c) |> Map.ofList

                for n in r.Nodes do
                    for c in n.Chains do
                        let f = finals[c.Chain]

                        if c.Head = f.Head then
                            Expect.equal c.Digest f.Digest (sprintf "%s chain %A" n.Name c.Chain)

            testCase "loss of both replicas with no spare: the survivors take the stage over"
            <| fun () ->
                let config =
                    { Simulation.clean 3UL 6 with
                        Faults = [ Fault.Kill(5UL, Ns.ms 120L); Fault.Kill(6UL, Ns.ms 120L) ]
                    }

                let r = Simulation.run config
                expectReference "no spare" r
                Expect.isGreaterThan r.Outcome.Value.Replays 0 "replays"

            testCase "a killed node that restarts rejoins as a new incarnation and the run still matches"
            <| fun () ->
                // 50 ms per unit makes the run last about 2 s, so the restart at 600 ms joins a live run.
                let r =
                    Simulation.run
                        { seven 4UL [ Fault.Kill(1UL, Ns.ms 100L); Fault.Restart(1UL, Ns.ms 600L) ] with
                            StageCostNs = [ Ns.ms 50L ]
                        }

                expectReference "kill and restart" r
                Expect.equal r.Stats.Restarts 1 "restarted"
                Expect.isTrue (r.Nodes |> List.exists (fun n -> n.Id = 1UL && n.Incarnation = 2UL)) "new incarnation"

            testCase "the divergence tripwire fires on a corrupt replica"
            <| fun () ->
                let sink = InMemorySink()

                let r =
                    Simulation.run
                        { seven 5UL [ Fault.Corrupt(3UL, 1u, 2u) ] with
                            Sink = Some sink
                        }

                let o = r.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                let d = o.Divergences |> List.exactlyOne
                Expect.equal (d.Stage, d.Step) (1u, 2u) "the corrupted unit"
                Expect.isTrue (d.FirstNode = 3UL || d.OtherNode = 3UL) "the corrupt node is named"
                Expect.notEqual o.RunDigest (standardRef ()).RunDigest "the run stopped short"

                let events = sink.Records |> List.filter (fun x -> x.Name = ClusterNames.Divergence)
                Expect.equal events.Length 1 "one cluster.divergence event"

            testCase "with the tripwire off, a corrupt replica is reported and the run completes"
            <| fun () ->
                let config = seven 6UL [ Fault.Corrupt(4UL, 1u, 0u) ]

                let r =
                    Simulation.run
                        { config with
                            Coordinator =
                                { config.Coordinator with
                                    HaltOnDivergence = false
                                }
                        }

                let o = r.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "still marked diverged"
                Expect.isNonEmpty o.Divergences "reported"

            testCase "deadlines count only after Ready: 40 s cold starts cause no reissue"
            <| fun () ->
                let config =
                    { seven 7UL [] with
                        Nodes = Simulation.nodes FakeJob.standard 7 (fun i -> Ns.seconds 40L + Ns.ms (int64 i))
                    }

                let r = Simulation.run config
                expectReference "cold start" r
                Expect.equal r.Outcome.Value.Reissues 0 "no reissue storm"
                Expect.isGreaterThan r.Stats.VirtualNs (Ns.seconds 40L) "the run waited for Ready"

            testCase "lagging work is recomputed elsewhere and the first result wins"
            <| fun () ->
                // From 200 ms (after stage 1 has its 8 deadline samples) both stage-1 replicas run 200 times slower
                // for 20 s: the work is recomputed elsewhere.
                let lag n =
                    Fault.Lag(n, Ns.ms 200L, Ns.seconds 20L, 200)

                let r = Simulation.run (seven 8UL [ lag 3UL; lag 4UL ])

                expectReference "lag" r
                Expect.isGreaterThan r.Outcome.Value.Reissues 0 "reissued"
                Expect.isLessThan r.Stats.VirtualNs (Ns.seconds 20L) "finished before the laggards recovered"

            testCase "a corrupt result accepted first is caught when a takeover node replays its chain"
            <| fun () ->
                // n4 dies before it is Ready, so n3 alone computes stage 1 (and gets step 2 wrong) until n4's lease
                // runs out; then the spare n7 takes the stage over and replays the chain from the unit log. 50 ms per
                // unit keeps the run going past the lease. Both kinds of corruption are caught by that replay.
                for fault in [ Fault.CorruptOutput(3UL, 1u, 2u); Fault.Corrupt(3UL, 1u, 2u) ] do
                    let sink = InMemorySink()

                    let r =
                        Simulation.run
                            { seven 21UL [ Fault.Kill(4UL, Ns.ms 5L); fault ] with
                                StageCostNs = [ Ns.ms 50L ]
                                Sink = Some sink
                            }

                    let o = r.Outcome.Value
                    Expect.equal o.Status RunStatus.Diverged (sprintf "%A: tripwire (%s)" fault (describe r))
                    let d = o.Divergences.Head
                    Expect.equal d.Source DivergenceSource.Replay (sprintf "%A: found by the replay" fault)
                    Expect.equal (d.Stage, d.FirstNode, d.OtherNode) (1u, 3UL, 7UL) (sprintf "%A: who disagreed" fault)
                    Expect.isSome d.OtherOutput (sprintf "%A: the replayed output is reported" fault)

                    let events = sink.Records |> List.filter (fun x -> x.Name = ClusterNames.Divergence)

                    Expect.equal events.Length 1 (sprintf "%A: one cluster.divergence event" fault)

                    Expect.isTrue
                        (events.Head.Attrs |> List.contains (Attr.str "source" "replay"))
                        (sprintf "%A: source=replay" fault)

            testCase "a wrong output with an honest digest is caught by the duplicate comparison"
            <| fun () ->
                let r = Simulation.run (seven 22UL [ Fault.CorruptOutput(5UL, 2u, 3u) ])
                let o = r.Outcome.Value
                Expect.equal o.Status RunStatus.Diverged "tripwire"
                let d = o.Divergences |> List.exactlyOne
                Expect.equal (d.Source, d.Stage, d.Step) (DivergenceSource.Result, 2u, 3u) "the corrupted unit"
                Expect.isTrue (d.FirstNode = 5UL || d.OtherNode = 5UL) "the corrupt node is named"

            testCase "a final unit whose other replica died is computed again by another member"
            <| fun () ->
                // n6 dies at 285 ms, while the last units of stage 2 run: n5 alone computed them. The drain asks n6
                // again until its lease runs out, then asks the spare n7, which replays each chain and computes the
                // final unit, so every final state is still checked by two members.
                let r = Simulation.run (seven 13UL [ Fault.Kill(6UL, Ns.ms 285L) ])
                expectReference "final replica lost" r
                let o = r.Outcome.Value
                Expect.isGreaterThan o.Replays 0 "the verifier replayed the unit log"

                for c in o.Chains do
                    if c.Final.Chain.Stage = 2u then
                        Expect.equal
                            c.ConfirmedBy
                            [ 5UL; 7UL ]
                            (sprintf "chain %A confirmed by n5 and n7" c.Final.Chain)

                    Expect.isGreaterThanOrEqual c.ConfirmedBy.Length 2 (sprintf "chain %A" c.Final.Chain)

            testCase "churn does not postpone the reissue of stuck work: a flapping spare and stuck replicas"
            <| fun () ->
                // Both stage-1 replicas are stuck from the start, before the stage has 8 samples, so the deadline is
                // the initial 30 s. The spare is cut off for 1.2 s of every 1.5 s, past its lease, so the membership
                // changes about every 0.75 s. Deadlines of members that stay are kept across those epochs, so the
                // stuck work is reissued after 30 s and the run ends.
                let stuck n =
                    Fault.Lag(n, 0L, Ns.seconds 1000L, 100000)

                let flaps =
                    [
                        for k in 0L .. 100L ->
                            Fault.Partition(7UL, Ns.ms 100L + k * Ns.ms 1500L, Ns.ms 1300L + k * Ns.ms 1500L)
                    ]

                let r = Simulation.run (seven 23UL (stuck 3UL :: stuck 4UL :: flaps))
                expectReference "flapping spare" r
                let o = r.Outcome.Value
                Expect.isGreaterThan o.Reissues 0 "reissued"
                Expect.isGreaterThan o.Epoch 30UL "the membership churned"
                Expect.isLessThan r.Stats.VirtualNs (Ns.seconds 60L) "done soon after the 30 s deadline"

            testCase "a partitioned node is expelled, rejoins, and the run still matches"
            <| fun () ->
                let sink = InMemorySink()

                // 50 ms per unit makes the run last about 2 s; n2 is cut off from 100 ms to 1.3 s, past its 1 s lease.
                let r =
                    Simulation.run
                        { seven 9UL [ Fault.Partition(2UL, Ns.ms 100L, Ns.ms 1300L) ] with
                            Sink = Some sink
                            StageCostNs = [ Ns.ms 50L ]
                        }

                expectReference "partition" r

                let leaves =
                    sink.Records
                    |> List.filter (fun x -> x.Name = ClusterNames.Leave)
                    |> List.length

                let joins =
                    sink.Records |> List.filter (fun x -> x.Name = ClusterNames.Join) |> List.length

                Expect.equal leaves 1 "expelled once (lease)"
                Expect.equal joins 8 "seven joins and one rejoin"

            testCase "every frame is lost 20% of the time and the run still matches"
            <| fun () ->
                let config = seven 10UL []

                let r =
                    Simulation.run
                        { config with
                            Chaos =
                                { ChaosConfig.standard with
                                    DropPermille = 200
                                    DuplicatePermille = 100
                                }
                        }

                expectReference "heavy loss" r

            testCase "telemetry changes nothing: the same trace and digest with and without a sink"
            <| fun () ->
                let config = Simulation.standard 11UL
                let quiet = Simulation.run config
                let sink = InMemorySink()
                let loud = Simulation.run { config with Sink = Some sink }
                Expect.equal loud.TraceHash quiet.TraceHash "trace"
                Expect.equal (digestOf loud) (digestOf quiet) "digest"
                Expect.isGreaterThan sink.Count 100 "records were written"

            testCase "the cluster emits its reserved telemetry names, on virtual time"
            <| fun () ->
                let sink = InMemorySink()

                // n5 dies; from 200 ms both stage-1 replicas lag, so their work is reissued.
                let lag n =
                    Fault.Lag(n, Ns.ms 200L, Ns.seconds 20L, 200)

                let r =
                    Simulation.run
                        { seven 12UL [ Fault.Kill(5UL, Ns.ms 150L); lag 3UL; lag 4UL ] with
                            Sink = Some sink
                        }

                expectReference "telemetry run" r
                let names = sink.Records |> List.map _.Name |> Set.ofList

                for name in
                    [
                        ClusterNames.Epoch
                        ClusterNames.Join
                        ClusterNames.Leave
                        ClusterNames.Ready
                        ClusterNames.Assign
                        ClusterNames.UnitDone
                        ClusterNames.UnitLatency
                        ClusterNames.Lag
                        ClusterNames.Deadline
                        ClusterNames.DeadlineMiss
                        ClusterNames.Reissue
                        ClusterNames.Duplicate
                        ClusterNames.ReplayServed
                        ClusterNames.ReplayUnits
                        ClusterNames.Chaos
                        ClusterNames.Run
                        ClusterNames.BytesSent
                    ] do
                    Expect.isTrue (names.Contains name) (sprintf "%s emitted" name)

                Expect.isTrue (names |> Set.forall TelemetryNames.isReserved) "every name is cluster.*"
                let last = sink.Records |> List.map _.TimeNs |> List.max
                Expect.equal last r.Stats.VirtualNs "timestamps are virtual time"
        ]
