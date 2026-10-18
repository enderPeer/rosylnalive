/// VkDeviceCheck: checks the binding's struct layouts against vk.xml, then lists every Vulkan device with the
/// VK_EXT_descriptor_heap verdict (VulkanHeapRecipe section 1).
///
///     dotnet artifacts/bin/VkDeviceCheck/release/VkDeviceCheck.dll [--json]
///
/// Exit codes: 0 at least one device accepted; 1 no Vulkan loader; 2 layout mismatch; 3 no device accepted.
module VkDeviceCheck.Program

open Lrm.Gpu.Vulkan.Interop

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()
    let json = argv |> Array.contains "--json"

    match LayoutCheck.run () with
    | [] ->
        if not json then
            printfn
                "struct layouts: %d structs and unions match vk.xml (VK_HEADER_VERSION %d)"
                VkMeta.Structs.Length
                VkMeta.HeaderVersion

        match VulkanLoader.TryOpen() with
        | None ->
            eprintfn "no Vulkan loader (vulkan-1.dll / libvulkan.so.1) found"
            1
        | Some loader ->
            use loader = loader
            let instanceVersion, reports = DeviceCheck.checkAll loader

            if json then
                printfn "%s" (DeviceCheck.toJson instanceVersion reports)
            else
                printfn "loader instance version: %s" (VkVersion.toString instanceVersion)
                printfn "physical devices: %d" reports.Length

                for r in reports do
                    printfn ""

                    for line in DeviceCheck.format r do
                        printfn "%s" line

                let accepted = reports |> List.filter _.Accepted |> List.length
                printfn ""
                printfn "%d of %d device(s) accepted" accepted reports.Length

            if reports |> List.exists _.Accepted then 0 else 3
    | errors ->
        printfn "struct layouts: MISMATCH"

        for e in errors do
            printfn "  %s" e

        2
