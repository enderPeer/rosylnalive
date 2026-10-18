/// Build the self-contained HTML dashboard over run logs (ADR T2-0001). Owner: T2.
///
///     lrm dashboard [--runs <dir>]... [--synthetic [--samples <dir>]] [--out <file.html>] [--watch [--interval <s>]]
///                   [--per-kind <n>] [--run <runId>]... [--max-log-mb <n>] [--write-reports]
module Lrm.Cli.Commands.Dashboard

open System
open System.IO
open System.Threading
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Dashboard

let private usage () =
    eprintfn "usage: lrm dashboard [--runs <dir>]... [--synthetic [--samples <dir>]] [--out <file.html>]"
    eprintfn "                     [--watch [--interval <s>]] [--per-kind <n>] [--run <runId>]... [--max-log-mb <n>]"
    eprintfn "                     [--write-reports]"
    eprintfn ""
    eprintfn "  --runs           a folder of run folders; repeat for several (default: LRM_DATA/runs, unless"
    eprintfn "                   --synthetic is given alone)"
    eprintfn "  --synthetic      add the synthetic run of every report template (T1), each with a view; their"
    eprintfn "                   report links go to the committed samples"
    eprintfn "  --samples        the folder of the committed sample reports (default: docs/reports/samples)"
    eprintfn "  --out            the HTML file (default: LRM_DATA/dashboard.html)"
    eprintfn "  --watch          rewrite the file whenever the run logs change; the page reloads itself"
    eprintfn "  --interval       watch mode: seconds between polls and between page reloads (default 3)"
    eprintfn "  --per-kind       how many of the newest runs of each kind get a view (default 3)"
    eprintfn "  --run            a run id that always gets a view; repeat for several"
    eprintfn "  --max-log-mb     runs with larger event logs get no view unless named with --run (default 64)"
    eprintfn "  --write-reports  also render the Markdown report of every run with a view into <run>/report,"
    eprintfn "                   so the report links resolve"
    2

type private Options =
    {
        Runs: string list
        Synthetic: bool
        Samples: string option
        Out: string option
        Watch: bool
        Interval: int
        PerKind: int
        Include: string list
        MaxLogMb: int
        WriteReports: bool
    }

let private positive (flag: string) (v: string) =
    match Invariant.tryParseI32 v with
    | Some n when n >= 1 -> Ok n
    | _ -> Error(sprintf "%s needs a whole number of at least 1, not '%s'" flag v)

let rec private parse (o: Options) (args: string list) : Result<Options, string> =
    match args with
    | [] -> Ok o
    | "--runs" :: v :: rest -> parse { o with Runs = o.Runs @ [ v ] } rest
    | "--synthetic" :: rest -> parse { o with Synthetic = true } rest
    | "--samples" :: v :: rest -> parse { o with Samples = Some v } rest
    | "--out" :: v :: rest -> parse { o with Out = Some v } rest
    | "--watch" :: rest -> parse { o with Watch = true } rest
    | "--write-reports" :: rest -> parse { o with WriteReports = true } rest
    | "--run" :: v :: rest ->
        match RunId.tryParse v with
        | Some id ->
            parse
                { o with
                    Include = o.Include @ [ id.Value ]
                }
                rest
        | None -> Error(sprintf "'%s' is not a run id" v)
    | "--interval" :: v :: rest ->
        positive "--interval" v
        |> Result.bind (fun n -> parse { o with Interval = n } rest)
    | "--per-kind" :: v :: rest ->
        positive "--per-kind" v
        |> Result.bind (fun n -> parse { o with PerKind = n } rest)
    | "--max-log-mb" :: v :: rest ->
        positive "--max-log-mb" v
        |> Result.bind (fun n -> parse { o with MaxLogMb = n } rest)
    | flag :: _ when flag.StartsWith("--", StringComparison.Ordinal) ->
        Error(sprintf "unknown or incomplete option '%s'" flag)
    | extra :: _ -> Error(sprintf "unexpected argument '%s'" extra)

let private describe (o: DashboardOutput) =
    let m = o.Model

    sprintf
        "%d runs, %d with a view, %d comparisons, %s"
        m.Runs.Length
        m.Views.Length
        m.Pairs.Length
        (Fmt.bytes (float o.Bytes.Length))

/// The runs roots: every --runs folder, then the synthetic runs, whose report links point at the committed samples.
let private rootsOf (runDirs: string list) (synthetic: string option) (samples: string) (outDir: string) =
    (runDirs |> List.map (RunsRoot.runs outDir))
    @ (synthetic |> Option.map (RunsRoot.samples outDir samples) |> Option.toList)

let private build (o: Options) (out: string) (runDirs: string list) (synthetic: string option) =
    let samples =
        Path.GetFullPath(o.Samples |> Option.defaultValue (Path.Combine("docs", "reports", "samples")))

    match synthetic with
    | Some dir ->
        Directory.CreateDirectory dir |> ignore

        for kind in Synthetic.kinds do
            Synthetic.write kind dir |> ignore
    | None -> ()

    let roots = rootsOf runDirs synthetic samples (Path.GetDirectoryName out)

    let options =
        { DashboardOptions.over roots with
            Watch = o.Watch
            IntervalSeconds = o.Interval
            PerKind = o.PerKind
            Include = o.Include
            MaxLogBytes = int64 o.MaxLogMb * 1024L * 1024L
        }

    let reports (output: DashboardOutput) =
        if o.WriteReports then
            for md in Dashboard.writeReports output do
                printfn "wrote %s" md

    if o.Watch then
        use stop = new CancellationTokenSource()

        Console.CancelKeyPress.Add(fun e ->
            e.Cancel <- true
            stop.Cancel())

        printfn
            "lrm dashboard: watching %s every %d s; Ctrl+C stops"
            (String.concat ", " (roots |> List.map _.Path))
            o.Interval

        Dashboard.watch options out (TimeSpan.FromSeconds(float o.Interval)) stop.Token (fun output ->
            reports output
            printfn "wrote %s (%s)" out (describe output))
    else
        let output = Dashboard.build options
        Dashboard.write out output
        reports output
        printfn "wrote %s (%s)" out (describe output)

        if not output.Model.Unreadable.IsEmpty then
            eprintfn
                "lrm dashboard: %d folders are not readable runs; the runs index lists them"
                output.Model.Unreadable.Length

let private run (argv: string[]) : int =
    let empty =
        {
            Runs = []
            Synthetic = false
            Samples = None
            Out = None
            Watch = false
            Interval = 3
            PerKind = 3
            Include = []
            MaxLogMb = 64
            WriteReports = false
        }

    match parse empty (List.ofArray argv) with
    | Error e ->
        eprintfn "lrm dashboard: %s" e
        usage ()
    | Ok o ->
        let out =
            Path.GetFullPath(o.Out |> Option.defaultWith (fun () -> LrmPaths.under "dashboard.html"))

        let runDirs =
            match o.Runs with
            | [] when o.Synthetic -> []
            | [] -> [ LrmPaths.runs () ]
            | dirs -> dirs |> List.map Path.GetFullPath

        match runDirs |> List.filter (fun d -> not (Directory.Exists d)) with
        | _ :: _ as missing ->
            for d in missing do
                eprintfn "lrm dashboard: no runs folder %s" d

            1
        | [] ->
            // The synthetic runs go to a fresh temporary folder, deleted when the command ends.
            let synthetic =
                if o.Synthetic then
                    Some(Path.Combine(Path.GetTempPath(), "lrm-dashboard-synthetic", Guid.NewGuid().ToString("N")))
                else
                    None

            try
                try
                    build o out runDirs synthetic
                    0
                with
                | :? IOException as e ->
                    eprintfn "lrm dashboard: %s" e.Message
                    1
                | :? UnauthorizedAccessException as e ->
                    eprintfn "lrm dashboard: %s" e.Message
                    1
            finally
                match synthetic with
                | Some dir when Directory.Exists dir ->
                    try
                        Directory.Delete(dir, true)
                    with _ ->
                        ()
                | _ -> ()

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "dashboard"
            Summary = "Build the self-contained HTML dashboard over run logs."
            Run = run
        |}
    ]
