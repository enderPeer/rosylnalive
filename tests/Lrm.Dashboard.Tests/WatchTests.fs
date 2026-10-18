module Lrm.Dashboard.Tests.WatchTests

open System
open System.Diagnostics
open System.IO
open System.Threading
open System.Threading.Tasks
open Expecto
open Lrm.Telemetry
open Lrm.Dashboard
open Lrm.Dashboard.Tests.Common

/// Polls `condition` every 20 ms for at most 60 s (CI liveness tests use 60 s timeouts).
let private waitFor (what: string) (condition: unit -> bool) =
    let clock = Stopwatch.StartNew()

    while not (condition ()) do
        if clock.Elapsed > TimeSpan.FromSeconds 60.0 then
            failtestf "timed out waiting for %s" what

        Thread.Sleep 20

let private read (path: string) =
    try
        if File.Exists path then File.ReadAllText path else ""
    with :? IOException ->
        ""

let tests =
    testList
        "cpu.dashboard.watch"
        [
            testCase "watch writes the page, rewrites it when a run appears or grows, and stops when cancelled"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                let out = Path.Combine(tmp.Path, "dashboard.html")
                let o = DashboardOptions.over [ RunsRoot.runs tmp.Path runs ]
                let writes = ref 0
                use stop = new CancellationTokenSource()

                let task =
                    Task.Run(fun () ->
                        Dashboard.watch o out (TimeSpan.FromMilliseconds 50.0) stop.Token (fun _ ->
                            Interlocked.Increment writes |> ignore))

                // The file is renamed into place before onWrite runs, so wait for both.
                waitFor "the first write" (fun () -> writes.Value = 1 && (read out).Contains "id=\"run-r-a\"")
                let first = read out
                Expect.stringContains first "<meta http-equiv=\"refresh\"" "watch mode reloads"

                // Unchanged logs: no rewrite.
                Thread.Sleep 300
                Expect.equal writes.Value 1 "no rewrite while nothing changes"

                writeRun runs "r-b" "parity" "B" (unixNs 2026 10 18 11 0) RunStatus.Ok (parityBody 0.6)
                |> ignore

                // A poll can see the new run half written (its manifest says running) and the next one finished.
                waitFor "the new run, finished" (fun () ->
                    let page = read out

                    writes.Value >= 2
                    && page.Contains "id=\"run-r-b\""
                    && not (page.Contains "<span class=\"d-st d-st-running\">"))

                let settled = writes.Value
                Thread.Sleep 300
                Expect.equal writes.Value settled "no rewrite once the logs are still again"

                stop.Cancel()
                Expect.isTrue (task.Wait(TimeSpan.FromSeconds 60.0)) "stopped"
                Expect.isFalse (File.Exists(out + ".tmp")) "no temporary file left"

            testCase "the fingerprint changes when a run appears, an events file grows or a manifest changes"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                let roots = [ RunsRoot.runs tmp.Path runs ]
                let empty = Dashboard.fingerprint roots

                let dir =
                    writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)

                let one = Dashboard.fingerprint roots
                Expect.notEqual one empty "a run appeared"
                Expect.equal (Dashboard.fingerprint roots) one "stable while nothing changes"
                File.AppendAllText(Path.Combine(dir, "events.n0.jsonl"), "\n")
                let grown = Dashboard.fingerprint roots
                Expect.notEqual grown one "an events file grew"
                File.AppendAllText(Path.Combine(dir, "manifest.json"), " ")
                Expect.notEqual (Dashboard.fingerprint roots) grown "the manifest changed"
                File.WriteAllText(Path.Combine(dir, "artifacts.txt"), "x")
                Expect.notEqual (Dashboard.fingerprint roots) one "still changed"
        ]
