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

/// The page as it is now, or "" while it is missing or being replaced (Windows reports a file that a rename is
/// replacing as access denied).
let private read (path: string) =
    try
        if File.Exists path then File.ReadAllText path else ""
    with
    | :? IOException
    | :? UnauthorizedAccessException -> ""

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
                Expect.stringContains first "<html lang=\"en\" data-watch=\"3\">" "watch mode reloads"

                // Unchanged logs: no rewrite.
                Thread.Sleep 300
                Expect.equal writes.Value 1 "no rewrite while nothing changes"

                // The new run is written beside the root and moved in whole. A run written in place races with the
                // watch's reads: T1's RunManifest.read holds the manifest without delete sharing, so the run's own
                // Finish (a rename over the manifest) can fail on Windows while a rebuild reads it (routed to T1).
                let dir =
                    writeRun
                        (tmp.Sub "staging")
                        "r-b"
                        "parity"
                        "B"
                        (unixNs 2026 10 18 11 0)
                        RunStatus.Ok
                        (parityBody 0.6)

                Directory.Move(dir, Path.Combine(runs, "r-b"))

                waitFor "the new run" (fun () -> writes.Value >= 2 && (read out).Contains "id=\"run-r-b\"")

                // The run's log grows by a line that is cut off: the page shows the problem.
                let grown = writes.Value
                File.AppendAllText(Path.Combine(runs, "r-b", "events.n0.jsonl"), "{\"v\":1")

                waitFor "the grown log" (fun () ->
                    writes.Value > grown && (read out).Contains "<span>Log problems</span><b>1</b>")

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

            testCase "watch outlives its runs folder going away, and shows the runs when it comes back"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                let out = Path.Combine(tmp.Path, "dashboard.html")
                let o = DashboardOptions.over [ RunsRoot.runs tmp.Path runs ]
                use stop = new CancellationTokenSource()

                let task =
                    Task.Run(fun () -> Dashboard.watch o out (TimeSpan.FromMilliseconds 20.0) stop.Token ignore)

                waitFor "the first write" (fun () -> (read out).Contains "id=\"run-r-a\"")

                // The watch reads the folder every 20 ms, so a delete can meet an open file: retry it.
                waitFor "the folder deleted" (fun () ->
                    try
                        Directory.Delete(runs, true)
                        true
                    with
                    | :? IOException
                    | :? UnauthorizedAccessException -> false)

                waitFor "the page without runs" (fun () -> (read out).Contains "No runs yet.")
                Expect.isFalse task.IsCompleted "still watching"

                // Written elsewhere and moved in whole, as in the test above.
                let staged =
                    writeRun
                        (tmp.Sub "staging")
                        "r-b"
                        "parity"
                        "B"
                        (unixNs 2026 10 18 11 0)
                        RunStatus.Ok
                        (parityBody 0.6)

                Directory.CreateDirectory runs |> ignore
                Directory.Move(staged, Path.Combine(runs, "r-b"))

                waitFor "the run in the new folder" (fun () -> (read out).Contains "id=\"run-r-b\"")
                stop.Cancel()
                Expect.isTrue (task.Wait(TimeSpan.FromSeconds 60.0)) "stopped"
                Expect.isNull task.Exception "no error ended the watch"

            testCase "the fingerprint of a root that is missing or a file is just its path"
            <| fun () ->
                use tmp = new TempDir()
                let file = Path.Combine(tmp.Path, "not-a-folder")
                File.WriteAllText(file, "x")
                let missing = Path.Combine(tmp.Path, "missing")

                Expect.equal
                    (Dashboard.fingerprint [ RunsRoot.runs tmp.Path file; RunsRoot.runs tmp.Path missing ])
                    (file + "\n" + missing + "\n")
                    "no runs, no error"
        ]
