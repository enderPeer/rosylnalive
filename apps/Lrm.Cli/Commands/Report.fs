/// Render Markdown and SVG reports from run logs (ReportTemplates contract). Owner: T1.
///
///     lrm report <runId|path> [--template <name>] [--out <dir>]
///     lrm report --synthetic <name|all> [--out <dir>] [--runs <dir>]
///     lrm report --list
module Lrm.Cli.Commands.Report

open System
open System.IO
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports

let private usage () =
    eprintfn "usage: lrm report <runId|path> [--template <name>] [--out <dir>]"
    eprintfn "       lrm report --synthetic <name|all> [--out <dir>] [--runs <dir>]"
    eprintfn "       lrm report --list"
    eprintfn ""
    eprintfn "  <runId|path>   a run folder, or a run id under LRM_DATA/runs"
    eprintfn "  --template     the template to render (default: the run's kind)"
    eprintfn "  --out          the output folder (default: <run>/report)"
    eprintfn "  --synthetic    write synthetic runs and render their reports, for reviewing the visuals"
    eprintfn "                 (default --out: docs/reports/samples under the current folder)"
    eprintfn "  --runs         where synthetic runs are written (default: a fresh temporary folder)"
    eprintfn "  --list         list the templates"
    2

type private Options =
    {
        Target: string option
        Template: string option
        Out: string option
        Synthetic: string option
        Runs: string option
        List: bool
    }

let rec private parse (o: Options) (args: string list) : Result<Options, string> =
    match args with
    | [] -> Ok o
    | "--template" :: v :: rest -> parse { o with Template = Some v } rest
    | "--out" :: v :: rest -> parse { o with Out = Some v } rest
    | "--synthetic" :: v :: rest -> parse { o with Synthetic = Some v } rest
    | "--runs" :: v :: rest -> parse { o with Runs = Some v } rest
    | "--list" :: rest -> parse { o with List = true } rest
    | flag :: _ when flag.StartsWith("--", StringComparison.Ordinal) ->
        Error(sprintf "unknown or incomplete option '%s'" flag)
    | target :: rest when o.Target.IsNone -> parse { o with Target = Some target } rest
    | extra :: _ -> Error(sprintf "unexpected argument '%s'" extra)

/// A run folder from a path or a run id.
let private resolveRun (target: string) =
    if Directory.Exists target then
        Some(Path.GetFullPath target)
    else
        match RunId.tryParse target with
        | Some id when Directory.Exists(LrmPaths.run id) -> Some(LrmPaths.run id)
        | _ -> None

let private printWritten (paths: string list) =
    match paths with
    | md :: svgs -> printfn "wrote %s (+%d charts)" md svgs.Length
    | [] -> ()

let private renderRun (o: Options) (target: string) =
    match resolveRun target with
    | None ->
        eprintfn "lrm report: no run folder or run id '%s' (runs live in %s)" target (LrmPaths.runs ())
        1
    | Some dir ->
        match RunLogReader.load dir with
        | Error e ->
            eprintfn "lrm report: %s" e
            1
        | Ok log ->
            let template =
                match o.Template with
                | Some name -> Reports.tryFind name
                | None -> Reports.defaultTemplate log

            match template with
            | None ->
                eprintfn
                    "lrm report: no template for '%s'; pass --template (one of %s)"
                    (defaultArg o.Template log.Manifest.Kind)
                    (String.concat ", " Reports.names)

                2
            | Some t ->
                let out = o.Out |> Option.defaultValue (Path.Combine(dir, RunLogNames.ReportDir))

                printWritten (Reports.write out (Reports.render t.Name log))

                for p in log.Problems do
                    eprintfn "lrm report: log problem: %s" p

                0

let private renderSynthetic (o: Options) (which: string) =
    if which <> "all" && not (List.contains which Synthetic.kinds) then
        eprintfn "lrm report: no synthetic generator '%s' (all, %s)" which (String.concat ", " Synthetic.kinds)
        2
    else
        let runs =
            o.Runs
            |> Option.defaultWith (fun () ->
                Path.Combine(Path.GetTempPath(), "lrm-synthetic-runs", Guid.NewGuid().ToString("N")))

        let out = o.Out |> Option.defaultValue (Path.Combine("docs", "reports", "samples"))

        if which = "all" then
            for x in Reports.renderSamples runs out do
                printfn "wrote %s (+%d charts)" (Path.Combine(out, x.Template + ".md")) x.Svgs.Length

            printfn "wrote %s" (Path.Combine(out, "README.md"))
        else
            let dir = Synthetic.write which runs
            printWritten (Reports.write out (Reports.render which (RunLogReader.loadOrFail dir)))

        printfn "synthetic runs are in %s" runs
        0

let private run (argv: string[]) : int =
    let empty =
        {
            Target = None
            Template = None
            Out = None
            Synthetic = None
            Runs = None
            List = false
        }

    match parse empty (List.ofArray argv) with
    | Error e ->
        eprintfn "lrm report: %s" e
        usage ()
    | Ok o when o.List ->
        for t in Reports.templates do
            printfn "%-10s phase %d  %s" t.Name t.Phase t.Description

        0
    | Ok o ->
        try
            match o.Synthetic, o.Target with
            | Some which, None -> renderSynthetic o which
            | None, Some target -> renderRun o target
            | _ -> usage ()
        with
        | TemplateError msg ->
            eprintfn "lrm report: template error: %s" msg
            1
        | :? IOException as e ->
            eprintfn "lrm report: %s" e.Message
            1
        | :? UnauthorizedAccessException as e ->
            eprintfn "lrm report: %s" e.Message
            1

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "report"
            Summary = "Render Markdown and SVG reports from run logs."
            Run = run
        |}
    ]
