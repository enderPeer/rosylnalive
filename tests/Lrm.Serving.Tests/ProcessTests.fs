/// The multi-process harness on localhost: `node` processes (this test exe as a worker), C1's standalone
/// `chaos-proxy` process, and the coordinator in this process or as a `coord run` process. Kills are real process
/// kills; a restart is a new process with a new incarnation and no state. 60 s per run, digests and ids only.
module Lrm.Serving.Tests.ProcessTests

open System
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Serving
open Lrm.Serving.Tests.Common

let private processes = HarnessMode.processes workerStart

let private kill (node: int) (after: int) (restart: int option) =
    {
        Node = node
        AfterUnits = after
        RestartAfterMs = restart
    }

/// A topology's run with explicit faults (times from the proxy's start; node processes start in about a second).
let private scenario (t: Topology) (seed: uint64) (kills: HarnessKill list) (proxyFaults: Fault list) =
    { ServingHarness.clean tinySpec t seed with
        Name = Invariant.format $"{Topology.name t}-processes-{seed}"
        Kills = kills
        ProxyFaults = proxyFaults
    }

let private expectExits (r: HarnessResult) =
    for n in r.Nodes do
        if n.Killed then
            Expect.isNone n.ExitCode (n.Name + " was killed")
        else
            Expect.equal n.ExitCode (Some 0) (sprintf "%s exits cleanly: %s" n.Name n.Errors)

let tests =
    testList
        "processes"
        [
            testCase "two node processes, one stage each: kill and restart a node, sever the other's connection"
            <| fun () ->
                let s =
                    scenario Topology.Split2 11UL [ kill 2 30 (Some 400) ] [ Fault.Sever(1UL, Ns.ms 1500L) ]

                let r = ServingHarness.run processes s
                expectIdentical (referenceOf tinySpec) r
                expectExits r
                Expect.equal (r.Kills, r.Restarts) (1, 1) "one kill, one restart"
                Expect.isGreaterThanOrEqual r.Outcome.Value.Replays 1 "the restarted node replays its chains"

                let restarted = r.Nodes |> List.filter (fun n -> n.Name = "n2" && not n.Killed)
                Expect.equal restarted.Length 1 "n2 came back as one new incarnation"
                Expect.isGreaterThan restarted.Head.Report.Value.Replayed 0 "and replayed"

            testCase "three node processes, two replicas per stage: a killed node fails over with no catch-up"
            <| fun () ->
                let s = scenario Topology.Replicated3 12UL [ kill 1 40 None ] []
                let r = ServingHarness.run processes s
                expectIdentical (referenceOf tinySpec) r
                expectExits r
                Expect.isGreaterThan r.Outcome.Value.CrossChecked 90 "units cross-checked by a second replica"

            testCase "both replicas of the last stage killed: the spare process takes over by replay, one comes back"
            <| fun () ->
                let s =
                    scenario
                        Topology.BothLost5
                        13UL
                        [ kill 3 20 None; kill 4 30 (Some 300) ]
                        [ Fault.Partition(1UL, Ns.ms 1200L, Ns.ms 1700L) ]

                let r = ServingHarness.run processes s
                expectIdentical (referenceOf tinySpec) r
                expectExits r
                let spare = r.Nodes |> List.find (fun n -> n.Name = "n5")
                Expect.isGreaterThan spare.Report.Value.Replayed 0 "the spare replayed the lost chains"

            testCase
                "every piece a process: coord run, chaos-proxy and two nodes, with a kill, a restart and a partition"
            <| fun () ->
                let s =
                    scenario
                        Topology.Split2
                        14UL
                        [ kill 1 25 (Some 300) ]
                        [ Fault.Partition(2UL, Ns.ms 1200L, Ns.ms 1800L) ]

                let r =
                    ServingHarness.run
                        { processes with
                            CoordinatorProcess = true
                        }
                        s

                expectIdentical (referenceOf tinySpec) r
                expectExits r

            testCase "lrm node refuses bad options before it connects"
            <| fun () ->
                let bad args =
                    match ServingNode.parse args with
                    | Error _ -> ()
                    | Ok o ->
                        // Options that parse but fit no model are refused when the node opens the model.
                        match ServingNode.executorOptions (tinyWeights ()).Config o with
                        | Error _ -> ()
                        | Ok _ -> failwithf "accepted %A" args

                let ok = [ "--connect"; "127.0.0.1:1"; "--id"; "1"; "--name"; "n1" ]
                bad [ "--id"; "1"; "--name"; "n1" ]
                bad [ "--connect"; "127.0.0.1:1"; "--id"; "0"; "--name"; "n1" ]
                bad [ "--connect"; "127.0.0.1:1"; "--id"; "1"; "--name"; "coord" ]
                bad (ok @ [ "--layers"; "0-5" ])
                bad (ok @ [ "--split"; "2" ])
                bad (ok @ [ "--unknown"; "x" ])
                Expect.isOk (ServingNode.parse (ok @ [ "--layers"; "1-1"; "--fast-rebuild" ])) "a good one"
        ]
