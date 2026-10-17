open Lrm.Gpu.Vulkan
open Lrm.Gpu.Vulkan.DeviceCheck

let deviceTypeName =
    function
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU -> "integrated GPU"
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU -> "discrete GPU"
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU -> "virtual GPU"
    | VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_CPU -> "CPU"
    | _ -> "other"

let yesNo b = if b then "yes" else "no"

let printOptional (o: OptionalFeatures) =
    printfn "  optional:"
    printfn "    descriptorHeapCaptureReplay       %s" (yesNo o.DescriptorHeapCaptureReplay)
    printfn "    VK_KHR_shader_untyped_pointers    %s" (yesNo o.ShaderUntypedPointers)
    printfn "    VK_KHR_cooperative_matrix         %s" (yesNo o.CooperativeMatrix)
    printfn "    shaderFloat16                     %s" (yesNo o.ShaderFloat16)
    printfn "    shaderInt8                        %s" (yesNo o.ShaderInt8)

let printHeap (h: DescriptorHeapLimits) =
    printfn "  descriptor heap:"
    printfn "    maxPushDataSize                   %d" h.MaxPushDataSize
    printfn "    maxResourceHeapSize               %d" h.MaxResourceHeapSize
    printfn "    resourceHeapAlignment             %d" h.ResourceHeapAlignment
    printfn "    minResourceHeapReservedRange      %d" h.MinResourceHeapReservedRange
    printfn "    bufferDescriptorSize / align      %d / %d" h.BufferDescriptorSize h.BufferDescriptorAlignment
    printfn "    imageDescriptorSize / align       %d / %d" h.ImageDescriptorSize h.ImageDescriptorAlignment
    printfn "    samplerDescriptorSize / align     %d / %d" h.SamplerDescriptorSize h.SamplerDescriptorAlignment
    printfn "    maxSamplerHeapSize                %d" h.MaxSamplerHeapSize

[<EntryPoint>]
let main _ =
    match LayoutCheck.run () with
    | [] -> printfn "struct layouts: %d structs match vk.xml" VkXmlLayout.expected.Length
    | errors ->
        printfn "struct layouts: MISMATCH"
        errors |> List.iter (printfn "  %s")
        exit 2

    match VulkanLoader.TryOpen() with
    | None ->
        printfn "no Vulkan loader (vulkan-1.dll / libvulkan.so.1) found"
        1
    | Some loader ->
        use loader = loader
        let instanceVersion, reports = checkAll loader
        printfn "loader instance version: %s" (VkVersion.toString instanceVersion)
        printfn "physical devices: %d" reports.Length
        for r in reports do
            let id = r.Identity
            printfn ""
            printfn "%s (%s, vendor 0x%04x device 0x%04x)" id.Name (deviceTypeName id.Type) id.VendorId id.DeviceId
            printfn "  Vulkan %s, driver %s %s" (VkVersion.toString id.ApiVersion) id.DriverName id.DriverInfo
            match r.Verdict with
            | Accepted caps ->
                printfn "  verdict: ACCEPTED"
                printHeap caps.DescriptorHeap
            | Rejected reasons ->
                printfn "  verdict: REJECTED"
                for reason in reasons do
                    printfn "    - %s" reason
            printOptional r.Optional
        let accepted = reports |> List.filter (fun r -> r.Verdict.IsAccepted) |> List.length
        printfn ""
        printfn "%d of %d device(s) accepted" accepted reports.Length
        if accepted > 0 then 0 else 3
