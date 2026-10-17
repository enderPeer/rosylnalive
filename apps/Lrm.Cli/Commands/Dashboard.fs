/// Build the self-contained HTML dashboard over run logs.
/// Owner: T2. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Dashboard

let private run (_: string[]) : int =
    eprintfn "lrm dashboard: not implemented yet (owner T2)"
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
            Name = "dashboard"
            Summary = "Build the self-contained HTML dashboard over run logs."
            Run = run
        |}
    ]
