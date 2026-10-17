/// Find the first layer and tap where two backends diverge.
/// Owner: I1. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Bisect

let private run (_: string[]) : int =
    eprintfn "lrm bisect: not implemented yet (owner I1)"
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
            Name = "bisect"
            Summary = "Find the first layer and tap where two backends diverge."
            Run = run
        |}
    ]
