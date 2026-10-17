/// Greedy generation.
/// Owner: M2a. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Generate

let private run (_: string[]) : int =
    eprintfn "lrm generate: not implemented yet (owner M2a)"
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
            Name = "generate"
            Summary = "Greedy generation."
            Run = run
        |}
    ]
