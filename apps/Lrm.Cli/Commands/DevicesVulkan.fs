/// List Vulkan devices and the VK_EXT_descriptor_heap verdict (VulkanHeapRecipe section 1).
/// Owner: R2.
module Lrm.Cli.Commands.DevicesVulkan

open Lrm.Gpu.Vulkan.Interop

/// Exit codes: 0 at least one device is accepted; 1 no Vulkan loader; 2 usage error; 3 no device is accepted;
/// 4 the binding's struct layouts differ from vk.xml.
let private run (args: string[]) : int =
    let json =
        match args with
        | [||] -> Some false
        | [| "--json" |] -> Some true
        | _ -> None

    match json with
    | None ->
        eprintfn "usage: lrm devices-vulkan [--json]"
        2
    | Some json ->
        match LayoutCheck.run () with
        | _ :: _ as errors ->
            eprintfn "lrm devices-vulkan: the binding's struct layouts differ from vk.xml:"

            for e in errors do
                eprintfn "  %s" e

            4
        | [] ->
            match VulkanLoader.TryOpen() with
            | None ->
                eprintfn "lrm devices-vulkan: no Vulkan loader (vulkan-1.dll / libvulkan.so.1) found"
                1
            | Some loader ->
                use loader = loader

                try
                    let instanceVersion, reports = DeviceCheck.checkAll loader

                    if json then
                        printfn "%s" (DeviceCheck.toJson instanceVersion reports)
                    else
                        printfn "Vulkan devices (loader %s):" (VkVersion.toString instanceVersion)

                        for r in reports do
                            for line in DeviceCheck.format r do
                                printfn "  %s" line

                    if reports |> List.exists _.Accepted then 0 else 3
                with :? VulkanException as e ->
                    eprintfn "lrm devices-vulkan: %s" e.Message
                    1

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
