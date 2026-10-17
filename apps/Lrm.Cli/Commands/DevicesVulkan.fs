/// List Vulkan devices and the VK_EXT_descriptor_heap verdict.
/// Owner: R2a. Stub created by S0a; the owner replaces the body.
module Lrm.Cli.Commands.DevicesVulkan

let private run (_: string[]) : int =
    eprintfn "lrm devices-vulkan: not implemented yet (owner R2a)"
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
            Name = "devices-vulkan"
            Summary = "List Vulkan devices and the VK_EXT_descriptor_heap verdict."
            Run = run
        |}
    ]
