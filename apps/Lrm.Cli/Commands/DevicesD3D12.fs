/// List D3D12 adapters and the SM 6.6 / binding tier 3 verdict.
/// Owner: R3. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.DevicesD3D12

let private run (_: string[]) : int =
    eprintfn "lrm devices-d3d12: not implemented yet (owner R3)"
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
            Name = "devices-d3d12"
            Summary = "List D3D12 adapters and the SM 6.6 / binding tier 3 verdict."
            Run = run
        |}
    ]
