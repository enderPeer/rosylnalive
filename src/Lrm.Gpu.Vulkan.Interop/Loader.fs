namespace Lrm.Gpu.Vulkan.Interop

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop

/// What an instance is created with.
type InstanceOptions =
    {
        ApplicationName: string
        /// VkApplicationInfo.apiVersion: the highest core version the application uses.
        ApiVersion: uint32
        /// Instance extensions to enable. Each must be offered by the loader (VulkanLoader.InstanceExtensions).
        Extensions: string list
        Layers: string list
    }

module InstanceOptions =
    /// Vulkan 1.4, no layers, and VK_EXT_debug_utils when `debugUtils` is true.
    let create (applicationName: string) (debugUtils: bool) =
        {
            ApplicationName = applicationName
            ApiVersion = VkVersion.VK_API_VERSION_1_4
            Extensions =
                if debugUtils then
                    [ VkConst.VK_EXT_DEBUG_UTILS_EXTENSION_NAME ]
                else
                    []
            Layers = []
        }

/// Arrays of null-terminated UTF-8 strings for ppEnabledExtensionNames and the like, pinned until disposed.
type internal PinnedStrings(strings: string list) =
    let bytes = strings |> List.map VkText.utf8z |> Array.ofList
    let handles = bytes |> Array.map (fun b -> GCHandle.Alloc(b, GCHandleType.Pinned))
    let pointers = handles |> Array.map (fun h -> h.AddrOfPinnedObject())
    let table = GCHandle.Alloc(pointers, GCHandleType.Pinned)

    member _.Count = uint32 pointers.Length

    member _.Pointer =
        if pointers.Length = 0 then
            0n
        else
            table.AddrOfPinnedObject()

    interface IDisposable with
        member _.Dispose() =
            table.Free()

            for h in handles do
                h.Free()

/// The Vulkan loader (vulkan-1.dll, libvulkan.so.1). Only vkGetInstanceProcAddr is bound by symbol; every other
/// command comes through it (global and instance commands) or through vkGetDeviceProcAddr (device commands).
type VulkanLoader private (lib: nativeint, getInstanceProcAddr: PFN_vkGetInstanceProcAddr) =
    static let candidates =
        if OperatingSystem.IsWindows() then
            [ "vulkan-1.dll" ]
        else
            [ "libvulkan.so.1"; "libvulkan.so" ]

    let procAddr (instance: VkInstance) (name: string) =
        let b = VkText.utf8z name
        use p = fixed b
        getInstanceProcAddr.Invoke(instance, NativePtr.toNativeInt p)

    let globalCommands = VkGlobalCommands(procAddr VkInstance.Null)

    /// Returns None when no Vulkan loader is installed.
    static member TryOpen() : VulkanLoader option =
        candidates
        |> List.tryPick (fun name ->
            match NativeLibrary.TryLoad name with
            | true, h -> Some h
            | _ -> None)
        |> Option.bind (fun h ->
            match NativeLibrary.TryGetExport(h, "vkGetInstanceProcAddr") with
            | true, fp ->
                Some(new VulkanLoader(h, Marshal.GetDelegateForFunctionPointer<PFN_vkGetInstanceProcAddr> fp))
            | _ ->
                NativeLibrary.Free h
                None)

    /// vkGetInstanceProcAddr(instance, name); 0n when the command is not available.
    member _.ProcAddr(instance: VkInstance, name: string) = procAddr instance name

    member _.Global = globalCommands

    /// The highest instance-level version the loader supports (1.0 without vkEnumerateInstanceVersion).
    member _.InstanceVersion =
        if isNull globalCommands.vkEnumerateInstanceVersion then
            VkVersion.VK_API_VERSION_1_0
        else
            let mutable v = 0u
            globalCommands.EnumerateInstanceVersion(&v)
            v

    /// Instance extensions the loader and its drivers offer, with their spec versions.
    member _.InstanceExtensions() : Map<string, uint32> =
        let rec query () =
            let mutable count = 0u
            globalCommands.EnumerateInstanceExtensionProperties(0n, &count, 0n) |> ignore
            let props = Array.zeroCreate<VkExtensionProperties>(int count)
            use p = fixed props

            match globalCommands.EnumerateInstanceExtensionProperties(0n, &count, NativePtr.toNativeInt p) with
            | VkResult.VK_INCOMPLETE -> query ()
            | _ ->
                props
                |> Array.truncate (int count)
                |> Array.map (fun e -> e.extensionName.AsString, e.specVersion)
                |> Map.ofArray

        query ()

    /// Instance layers (validation layers and the like), with their spec versions.
    member _.InstanceLayers() : Map<string, uint32> =
        let rec query () =
            let mutable count = 0u
            globalCommands.EnumerateInstanceLayerProperties(&count, 0n) |> ignore
            let props = Array.zeroCreate<VkLayerProperties>(int count)
            use p = fixed props

            match globalCommands.EnumerateInstanceLayerProperties(&count, NativePtr.toNativeInt p) with
            | VkResult.VK_INCOMPLETE -> query ()
            | _ ->
                props
                |> Array.truncate (int count)
                |> Array.map (fun e -> e.layerName.AsString, e.specVersion)
                |> Map.ofArray

        query ()

    /// Creates an instance. Every struct of the create info lives in this frame for the call.
    member this.CreateInstance(o: InstanceOptions) : Instance =
        let appName = VkText.utf8z o.ApplicationName
        let engineName = VkText.utf8z "LRM"
        use pApp = fixed appName
        use pEngine = fixed engineName
        use extensions = new PinnedStrings(o.Extensions)
        use layers = new PinnedStrings(o.Layers)
        let mutable app = VkApplicationInfo.Create()
        app.pApplicationName <- NativePtr.toNativeInt pApp
        app.applicationVersion <- 1u
        app.pEngineName <- NativePtr.toNativeInt pEngine
        app.engineVersion <- 1u
        app.apiVersion <- o.ApiVersion
        let mutable ci = VkInstanceCreateInfo.Create()
        ci.pApplicationInfo <- NativePtr.toNativeInt &&app
        ci.enabledExtensionCount <- extensions.Count
        ci.ppEnabledExtensionNames <- extensions.Pointer
        ci.enabledLayerCount <- layers.Count
        ci.ppEnabledLayerNames <- layers.Pointer
        let mutable handle = VkInstance.Null
        globalCommands.CreateInstance(&ci, &handle)
        new Instance(this, handle, o)

    interface IDisposable with
        member _.Dispose() = NativeLibrary.Free lib

/// A VkInstance with its instance-level dispatch table. Disposing destroys the instance.
and [<Sealed>] Instance internal (loader: VulkanLoader, handle: VkInstance, options: InstanceOptions) =
    let commands = VkInstanceCommands(fun name -> loader.ProcAddr(handle, name))
    let mutable disposed = false

    member _.Handle = handle
    member _.Loader = loader
    member _.Commands = commands
    member _.ApiVersion = options.ApiVersion
    member _.EnabledExtensions = options.Extensions

    /// True when the instance enabled the extension `name`.
    member _.HasExtension(name: string) = List.contains name options.Extensions

    member _.PhysicalDevices() : VkPhysicalDevice[] =
        let mutable count = 0u
        commands.EnumeratePhysicalDevices(handle, &count, 0n) |> ignore
        let devices = Array.zeroCreate<VkPhysicalDevice>(int count)

        if count > 0u then
            use p = fixed devices

            commands.EnumeratePhysicalDevices(handle, &count, NativePtr.toNativeInt p)
            |> ignore

        Array.truncate (int count) devices

    /// Device extensions and their spec versions.
    member _.DeviceExtensions(device: VkPhysicalDevice) : Map<string, uint32> =
        let rec query () =
            let mutable count = 0u
            commands.EnumerateDeviceExtensionProperties(device, 0n, &count, 0n) |> ignore
            let props = Array.zeroCreate<VkExtensionProperties>(int count)
            use p = fixed props

            match commands.EnumerateDeviceExtensionProperties(device, 0n, &count, NativePtr.toNativeInt p) with
            | VkResult.VK_INCOMPLETE -> query () // the list grew between the two calls
            | _ ->
                props
                |> Array.truncate (int count)
                |> Array.map (fun e -> e.extensionName.AsString, e.specVersion)
                |> Map.ofArray

        query ()

    member _.Properties(device: VkPhysicalDevice) : VkPhysicalDeviceProperties =
        let mutable p = VkPhysicalDeviceProperties()
        commands.GetPhysicalDeviceProperties(device, &p)
        p

    member _.QueueFamilies(device: VkPhysicalDevice) : VkQueueFamilyProperties[] =
        let mutable count = 0u
        commands.GetPhysicalDeviceQueueFamilyProperties(device, &count, 0n)
        let families = Array.zeroCreate<VkQueueFamilyProperties>(int count)
        use p = fixed families
        commands.GetPhysicalDeviceQueueFamilyProperties(device, &count, NativePtr.toNativeInt p)
        Array.truncate (int count) families

    member _.MemoryProperties(device: VkPhysicalDevice) : VkPhysicalDeviceMemoryProperties =
        let mutable p = VkPhysicalDeviceMemoryProperties()
        commands.GetPhysicalDeviceMemoryProperties(device, &p)
        p

    /// The memory properties with VK_EXT_memory_budget's heapBudget and heapUsage, when the device supports the
    /// extension (a physical-device query needs it supported, not enabled).
    member this.MemoryBudget
        (device: VkPhysicalDevice)
        : (VkPhysicalDeviceMemoryProperties * VkPhysicalDeviceMemoryBudgetPropertiesEXT) option =
        if not ((this.DeviceExtensions device).ContainsKey VkConst.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) then
            None
        else
            let mutable budget = VkPhysicalDeviceMemoryBudgetPropertiesEXT.Create()
            let mutable p2 = VkPhysicalDeviceMemoryProperties2.Create()
            p2.pNext <- NativePtr.toNativeInt &&budget
            commands.GetPhysicalDeviceMemoryProperties2(device, &p2)
            Some(p2.memoryProperties, budget)

    /// VK_KHR_cooperative_matrix: every supported (M, N, K, types, scope) combination; empty without the extension.
    member this.CooperativeMatrixProperties(device: VkPhysicalDevice) : VkCooperativeMatrixPropertiesKHR[] =
        if
            isNull commands.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
            || not ((this.DeviceExtensions device).ContainsKey VkConst.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)
        then
            [||]
        else
            let mutable count = 0u

            commands.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &count, 0n)
            |> ignore

            let props =
                Array.init (int count) (fun _ -> VkCooperativeMatrixPropertiesKHR.Create())

            use p = fixed props

            commands.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &count, NativePtr.toNativeInt p)
            |> ignore

            Array.truncate (int count) props

    /// VK_KHR_calibrated_timestamps: the time domains vkGetCalibratedTimestampsKHR can sample; empty without it.
    member this.CalibrateableTimeDomains(device: VkPhysicalDevice) : VkTimeDomainKHR[] =
        if
            isNull commands.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
            || not ((this.DeviceExtensions device).ContainsKey VkConst.VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
        then
            [||]
        else
            let mutable count = 0u

            commands.GetPhysicalDeviceCalibrateableTimeDomainsKHR(device, &count, 0n)
            |> ignore

            let domains = Array.zeroCreate<VkTimeDomainKHR>(int count)
            use p = fixed domains

            commands.GetPhysicalDeviceCalibrateableTimeDomainsKHR(device, &count, NativePtr.toNativeInt p)
            |> ignore

            Array.truncate (int count) domains

    /// Active tools (validation layers, profilers), Vulkan 1.3.
    member _.ToolProperties(device: VkPhysicalDevice) : (string * string) list =
        let mutable count = 0u
        commands.GetPhysicalDeviceToolProperties(device, &count, 0n) |> ignore

        let props =
            Array.init (int count) (fun _ -> VkPhysicalDeviceToolProperties.Create())

        use p = fixed props

        commands.GetPhysicalDeviceToolProperties(device, &count, NativePtr.toNativeInt p)
        |> ignore

        props
        |> Array.truncate (int count)
        |> Array.map (fun t -> t.name.AsString, t.version.AsString)
        |> List.ofArray

    /// vkGetPhysicalDeviceDescriptorSizeEXT (VK_EXT_descriptor_heap); None without the extension.
    member _.DescriptorSize(device: VkPhysicalDevice, descriptorType: VkDescriptorType) : uint64 option =
        if isNull commands.vkGetPhysicalDeviceDescriptorSizeEXT then
            None
        else
            Some(commands.GetPhysicalDeviceDescriptorSizeEXT(device, descriptorType))

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                commands.DestroyInstance(handle)
