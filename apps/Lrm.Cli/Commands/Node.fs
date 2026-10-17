/// Run a pipeline stage node.
/// Owner: C2a. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Node

let private run (_: string[]) : int =
    eprintfn "lrm node: not implemented yet (owner C2a)"
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
            Name = "node"
            Summary = "Run a pipeline stage node."
            Run = run
        |}
    ]
