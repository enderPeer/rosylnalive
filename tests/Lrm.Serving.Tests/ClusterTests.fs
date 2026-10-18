/// The pipeline on a real cluster: C1's coordinator and node actors, TCP on 127.0.0.1 through C1's ChaosProxy, on
/// real time. The 50 proxy seeds run the nodes as actors in this process (each seed has its own coordinator, proxy and
/// nodes); ProcessTests runs the same scenarios with node, proxy and coordinator processes. Every run waits at most
/// 60 s and asserts digests and ids only.
module Lrm.Serving.Tests.ClusterTests

open System
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Serving
open Lrm.Serving.Tests.Common

/// The sweep's runs at once: LRM_SERVING_SWEEP_PARALLEL, else 4 (each run is mostly timers and network waits).
let private degree () =
    match
        Lrm.Testing.Env.tryGet "LRM_SERVING_SWEEP_PARALLEL"
        |> Option.bind Invariant.tryParseI32
    with
    | Some n when n >= 1 -> n
    | _ -> 4

/// The proxy seeds of the CI sweep.
let seeds = [ 1UL .. 50UL ]

let tests =
    testList
        "cluster"
        [
            testCase "a clean split (two nodes, frame chaos only) is bit-identical to the single-process run"
            <| fun () ->
                let s = ServingHarness.clean tinySpec Topology.Split2 1UL
                let r = ServingHarness.run HarnessMode.inProcess s
                expectIdentical (referenceOf tinySpec) r
                Expect.equal r.Outcome.Value.Units 96 "every unit"
                Expect.equal r.Outcome.Value.Replays 0 "no replay without faults"

            testCase
                "50 proxy seeds: kills, restarts, partitions, severs and frame chaos are bit-identical to one process"
            <| fun () ->
                let reference = referenceOf tinySpec

                let scenarios =
                    seeds
                    |> List.map (fun s -> ServingHarness.chaos tinySpec (Topology.ofSeed s) s 200)

                let results =
                    ServingHarness.sweep
                        (fun _ -> HarnessMode.inProcess)
                        (degree ())
                        (fun r -> printfn "  %s" (ServingHarness.describe r (ServingHarness.verdict reference r)))
                        scenarios

                let bad =
                    results
                    |> List.filter (fun r -> not (ServingHarness.verdict reference r).IsEmpty)

                for r in bad do
                    printfn "FAILED %s\n%s" r.Scenario.Name (String.concat "\n" (ServingHarness.verdict reference r))
                    printfn "%s" r.Diagnostics

                Expect.isEmpty (bad |> List.map _.Scenario.Name) "every seed is bit-identical"

                let outcomes = results |> List.map _.Outcome.Value

                let byTopology t =
                    results |> List.filter (fun r -> r.Scenario.Topology = t)

                for t in [ Topology.Split2; Topology.Replicated3; Topology.BothLost5 ] do
                    Expect.isGreaterThanOrEqual (byTopology t).Length 16 (Topology.name t + " seeds")

                Expect.isTrue (results |> List.forall (fun r -> r.Kills >= 1)) "every seed kills a node"

                Expect.isGreaterThanOrEqual
                    (results |> List.sumBy _.Restarts)
                    33
                    "restarts (every split2 and bothlost5 seed restarts a node)"

                Expect.isTrue
                    (outcomes |> List.forall (fun o -> o.Replays >= 1))
                    "every seed rebuilds a chain by replay"

                Expect.isTrue
                    (byTopology Topology.BothLost5 |> List.forall (fun r -> r.Kills = 2))
                    "both replicas of stage 1 are lost in every bothlost5 seed"

                Expect.isGreaterThan
                    (results
                     |> List.sumBy (fun r -> r.Proxy.Value.Dropped + r.Proxy.Value.Duplicated))
                    0L
                    "the proxy dropped and duplicated frames"

                Expect.isGreaterThan
                    (results |> List.sumBy (fun r -> r.Proxy.Value.Severed))
                    0L
                    "the proxy severed connections"

                Expect.isTrue
                    (scenarios
                     |> List.exists (fun s -> s.ProxyFaults |> List.exists (fun f -> f.IsPartition)))
                    "some seeds partition a node"

            testCase "cpu-f64: three seeds, one per topology, are bit-identical to the single-process run"
            <| fun () ->
                let reference = referenceOf tinySpec64

                for seed in [ 1UL; 2UL; 3UL ] do
                    let s = ServingHarness.chaos tinySpec64 (Topology.ofSeed seed) seed 200
                    expectIdentical reference (ServingHarness.run HarnessMode.inProcess s)

            testCase "nodes with fast rebuild on run the same bits (replay through C1's node stays exact)"
            <| fun () ->
                let s =
                    { ServingHarness.chaos tinySpec Topology.Split2 4UL 200 with
                        FastRebuild = true
                    }

                expectIdentical (referenceOf tinySpec) (ServingHarness.run HarnessMode.inProcess s)
        ]
