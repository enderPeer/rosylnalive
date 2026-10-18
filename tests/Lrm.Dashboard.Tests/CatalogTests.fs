module Lrm.Dashboard.Tests.CatalogTests

open System.IO
open Expecto
open Lrm.Telemetry
open Lrm.Dashboard
open Lrm.Dashboard.Tests.Common

/// Three parity runs and one tokenizer run under `root`, started 10:00, 11:00, 12:00 and 10:30 on 2026-10-18.
let private fourRuns (root: string) =
    writeRun root "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
    |> ignore

    writeRun root "r-b" "parity" "B" (unixNs 2026 10 18 11 0) RunStatus.Ok (parityBody 0.6)
    |> ignore

    writeRun root "r-c" "parity" "C" (unixNs 2026 10 18 12 0) RunStatus.Failed (parityBody 3.0)
    |> ignore

    writeRun root "r-t" "tokenizer" "T" (unixNs 2026 10 18 10 30) RunStatus.Ok recordsBody
    |> ignore

let private root (path: string) = RunsRoot.runs path path

let tests =
    testList
        "cpu.dashboard.catalog"
        [
            testCase "runs are listed newest first, and folders that are not runs are listed with a reason"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                fourRuns runs
                Directory.CreateDirectory(Path.Combine(runs, "no-manifest")) |> ignore
                Directory.CreateDirectory(Path.Combine(runs, "bad-manifest")) |> ignore
                File.WriteAllText(Path.Combine(runs, "bad-manifest", "manifest.json"), "{\"schema\": \"lrm.run/0\"}")
                File.WriteAllText(Path.Combine(runs, "stray.txt"), "not a folder")

                let found, bad = Catalog.scan [ root runs ]
                Expect.equal (found |> List.map _.Id) [ "r-c"; "r-b"; "r-t"; "r-a" ] "newest first"
                Expect.equal (found |> List.map _.Nodes) [ 1; 1; 1; 1 ] "one events file each"
                Expect.isTrue (found |> List.forall (fun r -> r.EventBytes > 0L)) "events sizes"
                Expect.equal (bad |> List.map _.Folder) [ "bad-manifest"; "no-manifest" ] "unreadable, by name"
                Expect.equal (bad |> List.find (fun u -> u.Folder = "no-manifest")).Reason "no manifest.json" "reason"

                for u in bad do
                    Expect.isFalse (u.Reason.Contains tmp.Path) "a reason never holds a path"

            testCase "a run id under an earlier root shadows the same id under a later one"
            <| fun () ->
                use tmp = new TempDir()
                let one = tmp.Sub "one"
                let two = tmp.Sub "two"

                writeRun one "r-a" "parity" "first" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                writeRun two "r-a" "parity" "second" (unixNs 2026 10 18 11 0) RunStatus.Ok (parityBody 0.5)
                |> ignore

                let found, bad = Catalog.scan [ root one; root two ]
                Expect.equal (found |> List.map (fun r -> r.Manifest.Title)) [ "first" ] "the first root wins"
                Expect.equal (bad |> List.map (fun u -> u.Root, u.Folder)) [ 1, "r-a" ] "the shadowed folder"
                Expect.stringContains bad.Head.Reason "already listed" "reason"

            testCase "a missing root holds no runs"
            <| fun () ->
                use tmp = new TempDir()
                let found, bad = Catalog.scan [ root (Path.Combine(tmp.Path, "absent")) ]
                Expect.isEmpty found "no runs"
                Expect.isEmpty bad "nothing unreadable"

            testCase "a run folder that disappears before it is read gets a view that says so"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"

                let dir =
                    writeRun runs "r-a" "parity" "A" (unixNs 2026 10 18 10 0) RunStatus.Ok (parityBody 0.5)

                let entry =
                    match Catalog.read 0 dir with
                    | Ok e -> e
                    | Error u -> failtestf "unreadable: %s" u.Reason

                Directory.Delete(dir, true)
                let v = RunView.load (root runs) entry
                Expect.isSome v.Error "a view that says so"
                Expect.stringContains v.Error.Value "cannot be read" "reason"
                Expect.isFalse (v.Error.Value.Contains runs) "no path in the reason"

            testCase "durations come from the manifest; a running run has none"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                fourRuns runs
                let found, _ = Catalog.scan [ root runs ]

                for r in found do
                    Expect.isSome r.DurationNs (r.Id + " finished")
                    Expect.isGreaterThan r.DurationNs.Value 0L (r.Id + " positive")

            testCase "views: the newest PerKind runs of each kind, every included run, every run of a ViewAll root"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                fourRuns runs
                let found, _ = Catalog.scan [ root runs ]

                let o =
                    { DashboardOptions.over [ root runs ] with
                        PerKind = 1
                    }

                let views, none = Dashboard.select o found
                Expect.equal (views |> List.map _.Id) [ "r-c"; "r-t" ] "newest of each kind"
                Expect.equal (none.TryFind "r-a") (Some NoView.Older) "older"

                let views, _ = Dashboard.select { o with Include = [ "r-a" ] } found
                Expect.equal (views |> List.map _.Id) [ "r-c"; "r-t"; "r-a" ] "included"

                let all =
                    { o with
                        Roots = [ { root runs with ViewAll = true } ]
                    }

                let views, none = Dashboard.select all found
                Expect.equal views.Length 4 "view all"
                Expect.isEmpty none "none left out"

            testCase "views: runs with logs over MaxLogBytes get none unless included"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                fourRuns runs
                let found, _ = Catalog.scan [ root runs ]

                let o =
                    { DashboardOptions.over [ root runs ] with
                        MaxLogBytes = 10L
                    }

                let views, none = Dashboard.select o found
                Expect.isEmpty views "every log is larger than 10 bytes"

                match none.TryFind "r-c" with
                | Some(NoView.TooLarge b) ->
                    Expect.equal b (found |> List.find (fun r -> r.Id = "r-c")).EventBytes "size"
                | other -> failtestf "expected TooLarge, got %A" other

                let views, _ = Dashboard.select { o with Include = [ "r-c" ] } found
                Expect.equal (views |> List.map _.Id) [ "r-c" ] "an included run gets a view at any size"

            testCase "comparisons: every pair among the newest ComparePerKind views of a kind, older first"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                fourRuns runs
                let o = DashboardOptions.over [ root runs ]

                let pairs (o: DashboardOptions) =
                    (Dashboard.build o).Model.Pairs
                    |> List.map (fun (x, y) -> x.Entry.Id, y.Entry.Id)

                Expect.equal (pairs o) [ "r-b", "r-c"; "r-a", "r-c"; "r-a", "r-b" ] "three parity pairs"
                Expect.equal (pairs { o with ComparePerKind = 2 }) [ "r-b", "r-c" ] "capped"

            testCase "comparisons never mix synthetic and measured runs"
            <| fun () ->
                use tmp = new TempDir()
                let runs = tmp.Sub "runs"
                let synthetic = tmp.Sub "synthetic"
                fourRuns runs
                writeSynthetic synthetic

                let o =
                    DashboardOptions.over [ root runs; RunsRoot.samples tmp.Path tmp.Path synthetic ]

                let model = (Dashboard.build o).Model
                Expect.equal model.Views.Length 13 "four measured and nine synthetic views"

                for x, y in model.Pairs do
                    Expect.equal x.Entry.Manifest.Synthetic y.Entry.Manifest.Synthetic "same side"
                    Expect.equal x.Entry.Kind y.Entry.Kind "same kind"
        ]
