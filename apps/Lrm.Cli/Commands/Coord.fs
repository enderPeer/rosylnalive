/// Run the pipeline coordinator.
/// Owner: C2a. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Coord

let private run (_: string[]) : int =
    eprintfn "lrm coord: not implemented yet (owner C2a)"
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
            Name = "coord"
            Summary = "Run the pipeline coordinator."
            Run = run
        |}
    ]
