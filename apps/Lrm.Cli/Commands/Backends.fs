/// Forward-ops registry (cpu-f32, cpu-f64); GpuBackends.fs registers GPU backends into it.
/// Owner: M2a. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.Backends

/// Commands this module adds to `lrm` (none yet). Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    []
