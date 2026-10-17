/// Run every Gate 1 check and write the summary report.
/// Owner: I3. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Gate1

let private run (_: string[]) : int =
    eprintfn "lrm gate1: not implemented yet (owner I3)"
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
            Name = "gate1"
            Summary = "Run every Gate 1 check and write the summary report."
            Run = run
        |}
    ]
