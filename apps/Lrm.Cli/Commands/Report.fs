/// Render Markdown and SVG reports from run logs.
/// Owner: T1. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Report

let private run (_: string[]) : int =
    eprintfn "lrm report: not implemented yet (owner T1)"
    2

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
