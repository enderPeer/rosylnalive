/// A real cluster run log (ReportTemplates, "cluster"): one simulated chaos run written through T1's RunContext, so
/// `lrm report <runId>` renders the cluster report from C1's own records (cluster.unit.done, cluster.member.leave,
/// cluster.reissue, cluster.run and the rest, docs/contracts/cluster.md "Telemetry"). The records are timed on the
/// simulator's virtual clock; the run.node header and run.end trailer on the wall clock.
module Lrm.Cluster.Tests.RunLogTests

open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Cluster

let tests =
    testList
        "runlog"
        [
            testCase "a simulated chaos run writes a cluster run log that the cluster report reads"
            <| fun () ->
                // Seven nodes: n1, n2 (stage 0), n3, n4 (stage 1), n5, n6 (stage 2) and the spare n7. n5 dies at
                // 150 ms; from 200 ms both stage-1 replicas run 200 times slower, so their work is reissued.
                let lag n =
                    Fault.Lag(n, Ns.ms 200L, Ns.seconds 20L, 200)

                use run =
                    RunContext.Create(
                        { RunOptions.create "cluster" "Simulated chaos run: n5 killed, stage 1 lagging (C1 simulator)" with
                            ExecutionClass = Some FakeJob.executionClass
                            GitSha = Lrm.Testing.Env.commit ()
                            Params =
                                [
                                    "source", "tests/Lrm.Cluster.Tests (C1 deterministic simulator)"
                                    "job", "FakeJob.standard: 4 lanes, 3 stages, 6 steps, 2 replicas per stage"
                                    "nodes", "7 (n1 to n6 and the spare n7), ready at 10 ms, clean LAN"
                                    "faults", "kill n5 at 150 ms; n3 and n4 lag 200x from 200 ms"
                                    "seed", "12"
                                    "time", "virtual ns"
                                ]
                        }
                    )

                let config =
                    { Simulation.clean 12UL 7 with
                        Faults = [ Fault.Kill(5UL, Ns.ms 150L); lag 3UL; lag 4UL ]
                        Coordinator =
                            { CoordinatorConfig.standard with
                                MinNodes = 7
                            }
                        Sink = Some(run.Sink :> ITelemetrySink)
                    }

                let r = Simulation.run config

                let completed = r.Outcome |> Option.exists (fun o -> o.Status = RunStatus.Completed)

                run.Finish(
                    if completed then
                        Lrm.Telemetry.RunStatus.Ok
                    else
                        Lrm.Telemetry.RunStatus.Failed
                )

                printfn "cluster run log: %s" run.Directory
                Expect.isTrue completed (sprintf "the run completes: %A" r.Failure)

                let log = RunLogReader.loadOrFail run.Directory
                Expect.isEmpty log.Problems "a clean log"
                Expect.equal log.Manifest.Kind "cluster" "kind"

                for name in
                    [
                        ClusterNames.UnitDone
                        ClusterNames.Leave
                        ClusterNames.Reissue
                        ClusterNames.Deadline
                        ClusterNames.Run
                    ] do
                    Expect.isNonEmpty (log.Named name) (name + " is in the run log")
        ]
