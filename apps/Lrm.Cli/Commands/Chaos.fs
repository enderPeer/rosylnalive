/// Cluster chaos runs (fault injection over seeds).
/// Owner: C1. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Chaos

let private run (_: string[]) : int =
    eprintfn "lrm chaos: not implemented yet (owner C1)"
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
            Name = "chaos"
            Summary = "Cluster chaos runs (fault injection over seeds)."
            Run = run
        |}
    ]
