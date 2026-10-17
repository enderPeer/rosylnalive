/// Cluster chaos runs over the fake 3-stage executor (ClusterProtocol contract). Owner: C1.
///
///     lrm chaos sim [--seeds 200] [--from 1]           every seed of the simulator against the reference digest
///     lrm chaos trace [--seed 7] [--runs 100]          the trace hash of one seed, run many times
///     lrm chaos local [--seed 1] [--nodes 3] [--kill 2] [--kill-after N] [--cost-ms 2] [--timeout-ms 60000]
///                                                      node processes behind a ChaosProxy, one killed part-way
///     lrm chaos node --connect host:port --id N --name nN [...]   one node process (started by `local`)
///     lrm chaos-proxy --listen PORT --target HOST:PORT [--seed N] [...]   the frame-aware TCP chaos proxy
module Lrm.Cli.Commands.Chaos

open System
open Lrm.Core
open Lrm.Cluster

let private usage () =
    eprintfn "usage: lrm chaos sim [--seeds N] [--from S]"
    eprintfn "       lrm chaos trace [--seed S] [--runs N]"

    eprintfn
        "       lrm chaos local [--seed S] [--nodes N] [--kill I|none] [--kill-after U] [--cost-ms C] [--timeout-ms T]"

    eprintfn "                       [--chaos none|lan|proxy|standard] [--drop P] [--duplicate P] [--reorder P] ..."
    eprintfn "       lrm chaos node --connect HOST:PORT --id N --name NAME [--cost-ms C] [--stages 0,1,2]"
    2

let private get (opts: (string * string) list) (key: string) =
    opts |> List.tryFindBack (fun (k, _) -> k = key) |> Option.map snd

let private intOr (opts: (string * string) list) (key: string) (fallback: int) =
    get opts key
    |> Option.bind Invariant.tryParseI32
    |> Option.defaultValue fallback

let private u64Or (opts: (string * string) list) (key: string) (fallback: uint64) =
    get opts key
    |> Option.bind Invariant.tryParseU64
    |> Option.defaultValue fallback

let private sim (opts: (string * string) list) =
    let count = intOr opts "seeds" 200
    let first = u64Or opts "from" 1UL
    let reference = FakeReference.run FakeJob.standard
    let clock = Diagnostics.Stopwatch.StartNew()

    let results =
        Simulation.runAll [ for i in 0 .. count - 1 -> Simulation.standard (first + uint64 i) ]

    let mutable bad = 0

    for r in results do
        match r.Outcome with
        | Some o when o.Status = RunStatus.Completed && o.RunDigest = reference.RunDigest -> ()
        | Some o ->
            bad <- bad + 1
            printfn "seed %d: %A, digest %s, %d divergence(s)" r.Seed o.Status o.RunDigest.Hex o.Divergences.Length
        | None ->
            bad <- bad + 1
            printfn "seed %d: did not finish: %A" r.Seed r.Failure

    let sum (f: SimResult -> int64) = results |> List.sumBy f

    printfn
        "lrm chaos sim: %d seed(s) from %d, %d bit-identical to the reference %s, in %d ms"
        count
        first
        (count - bad)
        reference.RunDigest.Hex
        clock.ElapsedMilliseconds

    printfn
        "  %d kills, %d restarts, %d frames dropped, %d duplicated, %d reissues, %d replays, max epoch %d"
        (sum (fun r -> int64 r.Stats.Kills))
        (sum (fun r -> int64 r.Stats.Restarts))
        (sum _.Stats.FramesDropped)
        (sum _.Stats.FramesDuplicated)
        (sum (fun r -> r.Outcome |> Option.map (fun o -> int64 o.Reissues) |> Option.defaultValue 0L))
        (sum (fun r -> r.Outcome |> Option.map (fun o -> int64 o.Replays) |> Option.defaultValue 0L))
        (results
         |> List.map (fun r -> r.Outcome |> Option.map _.Epoch |> Option.defaultValue 0UL)
         |> List.max)

    if bad = 0 then 0 else 1

let private trace (opts: (string * string) list) =
    let seed = u64Or opts "seed" 7UL
    let runs = intOr opts "runs" 100
    let results = Simulation.runAll [ for _ in 1..runs -> Simulation.standard seed ]
    let traces = results |> List.map _.TraceHash |> List.distinct

    printfn
        "lrm chaos trace: seed %d, %d run(s), %d distinct trace hash(es): %s"
        seed
        runs
        traces.Length
        (traces |> List.map _.Hex |> String.concat ", ")

    if traces.Length = 1 then 0 else 1

let private local (opts: (string * string) list) =
    match Harness.parseChaos opts ChaosConfig.proxy with
    | Error e ->
        eprintfn "lrm chaos local: %s" e
        2
    | Ok chaos ->
        let standard = Harness.standardLocalRun (u64Or opts "seed" 1UL)
        let nodes = intOr opts "nodes" standard.Nodes
        let units = standard.Job.Lanes * standard.Job.Steps * standard.Job.Stages

        let kill =
            match get opts "kill" with
            | Some "none" -> None
            | k ->
                let index = k |> Option.bind Invariant.tryParseI32 |> Option.defaultValue 2
                Some(min index nodes, intOr opts "kill-after" (units / 3))

        let run =
            { standard with
                Nodes = nodes
                Chaos = chaos
                Kill = kill
                CostMs = intOr opts "cost-ms" standard.CostMs
                TimeoutMs = intOr opts "timeout-ms" standard.TimeoutMs
            }

        let start = Harness.selfStartInfo [ "chaos" ] []
        let r = Harness.runLocal start run

        for n in r.Nodes do
            let report =
                match n.Report with
                | Some rep ->
                    sprintf
                        "replica digest %s, %d executed, %d replayed"
                        rep.ReplicaDigest.Short
                        rep.Executed
                        rep.Replayed
                | None -> "no report"

            printfn "  %s: %s, exit %A, %s" n.Name (if n.Killed then "killed" else "alive") n.ExitCode report

            if n.Errors <> "" then
                printfn "    stderr: %s" n.Errors

        printfn
            "  proxy: %d connection(s), %d forwarded, %d dropped, %d duplicated"
            r.Proxy.Connections
            r.Proxy.Forwarded
            r.Proxy.Dropped
            r.Proxy.Duplicated

        match r.Outcome with
        | Some o ->
            let same =
                o.RunDigest = r.Reference.RunDigest
                && o.ReplicaDigest = r.Reference.ReplicaDigest

            printfn
                "lrm chaos local: %A in %d ms, run digest %s (%s), epoch %d, %d reissues, %d replays"
                o.Status
                r.ElapsedMs
                o.RunDigest.Hex
                (if same then
                     "matches the reference"
                 else
                     "DIFFERS from the reference " + r.Reference.RunDigest.Hex)
                o.Epoch
                o.Reissues
                o.Replays

            if same && o.Status = RunStatus.Completed then 0 else 1
        | None ->
            printfn "lrm chaos local: %A" r.Failure
            1

let private run (args: string[]) : int =
    match List.ofArray args with
    | "node" :: _ -> Harness.workerMain (List.ofArray args)
    | sub :: rest ->
        match Harness.options rest with
        | Error e ->
            eprintfn "lrm chaos: %s" e
            usage ()
        | Ok opts ->
            match sub with
            | "sim" -> sim opts
            | "trace" -> trace opts
            | "local" -> local opts
            | _ -> usage ()
    | [] -> usage ()

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "chaos"
            Summary = "Cluster chaos on the fake 3-stage executor: sim, trace, local, node."
            Run = run
        |}
        {|
            Name = "chaos-proxy"
            Summary = "Frame-aware TCP chaos proxy between cluster nodes and a coordinator."
            Run = fun args -> Harness.runProxy (List.ofArray args)
        |}
    ]
