/// Telemetry of the pipeline: telemetry changes no value, and a real multi-process run writes one cluster run log
/// (the coordinator creates it, every node process joins it, a restarted process as <name>-r<attempt>) that the
/// cluster template renders with its stage Gantt chart from C1's records.
module Lrm.Serving.Tests.RunLogTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Cluster
open Lrm.Serving
open Lrm.Serving.Tests.Common

/// The variable that keeps the cluster run of this test under LRM_DATA/runs (for `lrm report`).
[<Literal>]
let KeepVariable = "LRM_WRITE_CLUSTER_RUN"

let tests =
    testList
        "runlog"
        [
            testCase "telemetry changes no value: the plan-order pipeline with and without a strict sink"
            <| fun () ->
                let job = jobOf tinySpec

                let runOnce () =
                    use ex = executor "cpu-f32" (executorOptions job)
                    PlanOrder.run (QwenWorkload job) ex

                let quiet = runOnce ()
                let sink = InMemorySink()

                let recorded =
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    runOnce ()

                Expect.equal recorded.RunDigest quiet.RunDigest "run digest"
                Expect.equal recorded.ReplicaDigest quiet.ReplicaDigest "replica digest"

                let spans =
                    sink.Records |> List.filter (fun r -> r.Name = "model.forward") |> List.length

                Expect.equal spans 96 "one model.forward span per unit"

            testCase "a multi-process run writes a cluster run log; the cluster template renders its Gantt chart"
            <| fun () ->
                let keep = Environment.GetEnvironmentVariable KeepVariable = "1"
                let root = if keep then None else Some(tempDir "runs")

                try
                    let s =
                        { ServingHarness.clean tinySpec Topology.Split2 21UL with
                            Name = "split2-runlog-21"
                            Kills =
                                [
                                    {
                                        Node = 2
                                        AfterUnits = 40
                                        RestartAfterMs = Some 400
                                    }
                                ]
                            ProxyFaults = [ Fault.Sever(1UL, Ns.ms 1500L) ]
                        }

                    let mode =
                        { HarnessMode.processes workerStart with
                            RunLog =
                                Some
                                    "Tiny model on two node processes behind lrm chaos-proxy: n2 killed after 40 units and restarted (C2a)"
                            RunsRoot = root
                        }

                    let r = ServingHarness.run mode s
                    expectIdentical (referenceOf tinySpec) r
                    let dir = r.RunDir.Value
                    printfn "cluster run log: %s" dir
                    let log = RunLogReader.loadOrFail dir
                    Expect.equal log.Manifest.Kind "cluster" "kind"
                    Expect.equal log.Manifest.Status Lrm.Telemetry.RunStatus.Ok "the creator finished the run as ok"

                    Expect.equal
                        (log.Nodes |> List.map (fun n -> n.Node) |> List.sort)
                        [ "coord"; "n1"; "n2"; "n2-r1" ]
                        "the coordinator, both nodes and the restarted n2"

                    let restarted = log.Nodes |> List.find (fun n -> n.Node = "n2-r1")
                    Expect.equal restarted.RestartOf "n2" "n2-r1 restarts n2"

                    // The killed process wrote no trailer; nothing else may be wrong with the log.
                    let unexpected =
                        log.Nodes
                        |> List.collect (fun n ->
                            n.Problems
                            |> List.filter (fun p -> not (n.Node = "n2" && (p.Contains "trailer" || p.Contains "cut")))
                            |> List.map (fun p -> n.Node + ": " + p))

                    Expect.isEmpty unexpected "no other problem"

                    for name in
                        [
                            ClusterNames.UnitDone
                            ClusterNames.Leave
                            ClusterNames.Join
                            ClusterNames.Assign
                            ClusterNames.Chaos
                            ClusterNames.ReplayServed
                            ClusterNames.ReplayUnits
                            ClusterNames.Run
                            "model.forward"
                        ] do
                        Expect.isNonEmpty (log.Named name) (name + " is in the run log")

                    let stages =
                        log.Named "model.forward"
                        |> List.choose (fun e -> e.TryInt TelemetryKeys.Stage |> Option.map (fun s -> e.Node, s))
                        |> List.distinct
                        |> List.sort

                    Expect.contains stages ("n1", 0L) "n1 ran stage 0"
                    Expect.contains stages ("n2-r1", 1L) "the restarted n2 ran stage 1"

                    let checks = log.Named "golden.compare"
                    Expect.equal checks.Length 4 "the four comparisons with the oracles"
                    Expect.isTrue (checks |> List.forall (fun e -> e.Value = 1.0)) "all passed"

                    let out = Reports.render "cluster" log
                    Expect.stringContains out.Markdown "**PASS.**" "the verdict"
                    let _, gantt = out.Svgs |> List.find (fun (f, _) -> f.Contains "gantt")
                    Expect.isFalse (gantt.Contains "this run has no") "the Gantt chart has data"
                    Expect.stringContains gantt "n2 killed" "the kill marker"
                    Expect.stringContains gantt "n2 restarted" "the restart marker"
                finally
                    root |> Option.iter deleteDir
        ]
