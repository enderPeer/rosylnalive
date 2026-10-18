namespace Lrm.Gpu.Vulkan.Interop

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.CompilerServices
open Microsoft.FSharp.NativeInterop
open Lrm.Core

/// Which structs a FeatureChain links. The recipe's structs are linked only when the device's version or extensions
/// make them valid, so a query never relies on undefined behaviour.
type FeatureParts =
    {
        Vulkan12: bool
        Vulkan13: bool
        Maintenance5: bool
        DescriptorHeap: bool
        /// VK_EXT_device_fault's VkPhysicalDeviceFaultFeaturesEXT.
        Fault: bool
        /// VK_KHR_shader_untyped_pointers (queried, never enabled).
        UntypedPointers: bool
        /// VK_KHR_cooperative_matrix (queried, never enabled).
        CooperativeMatrix: bool
    }

/// The feature chain of VulkanHeapRecipe section 1, as one value that lives in the caller's frame: a byref-like
/// struct cannot be boxed, captured or stored on the heap, so its pNext pointers never outlive the frame (the
/// VulkanInteropApi contract). The device check queries features over the same chain that Device.create enables:
///
///     VkPhysicalDeviceFeatures2 (shaderStorageBufferArrayDynamicIndexing) -> Vulkan12Features (bufferDeviceAddress,
///     timelineSemaphore, runtimeDescriptorArray) -> Vulkan13Features (synchronization2) -> Maintenance5Features
///     (maintenance5) -> DescriptorHeapFeaturesEXT (descriptorHeap) [-> FaultFeaturesEXT (deviceFault)]
///
/// VkPhysicalDeviceVulkan14Features is never linked: it must not appear with Maintenance5Features
/// (VUID-VkDeviceCreateInfo-pNext-10360).
[<Struct; IsByRefLike; NoEquality; NoComparison>]
type FeatureChain =
    val mutable Features2: VkPhysicalDeviceFeatures2
    val mutable Vulkan12: VkPhysicalDeviceVulkan12Features
    val mutable Vulkan13: VkPhysicalDeviceVulkan13Features
    val mutable Maintenance5: VkPhysicalDeviceMaintenance5Features
    val mutable DescriptorHeap: VkPhysicalDeviceDescriptorHeapFeaturesEXT
    val mutable Fault: VkPhysicalDeviceFaultFeaturesEXT
    val mutable UntypedPointers: VkPhysicalDeviceShaderUntypedPointersFeaturesKHR
    val mutable CooperativeMatrix: VkPhysicalDeviceCooperativeMatrixFeaturesKHR

module FeatureChain =
    let inline private addr (x: byref<'T>) : nativeint =
        NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte>(Unsafe.AsPointer(&x)))

    /// Sets every sType, links the selected structs in recipe order and returns the head (the Features2 struct).
    /// `chain` must be a local of the calling frame.
    let link (chain: byref<FeatureChain>) (parts: FeatureParts) : nativeint =
        chain.Features2.sType <- VkPhysicalDeviceFeatures2.StructureType
        chain.Vulkan12.sType <- VkPhysicalDeviceVulkan12Features.StructureType
        chain.Vulkan13.sType <- VkPhysicalDeviceVulkan13Features.StructureType
        chain.Maintenance5.sType <- VkPhysicalDeviceMaintenance5Features.StructureType
        chain.DescriptorHeap.sType <- VkPhysicalDeviceDescriptorHeapFeaturesEXT.StructureType
        chain.Fault.sType <- VkPhysicalDeviceFaultFeaturesEXT.StructureType
        chain.UntypedPointers.sType <- VkPhysicalDeviceShaderUntypedPointersFeaturesKHR.StructureType
        chain.CooperativeMatrix.sType <- VkPhysicalDeviceCooperativeMatrixFeaturesKHR.StructureType

        // Built back to front, so the chain reads in recipe order.
        let mutable head = 0n

        if parts.CooperativeMatrix then
            chain.CooperativeMatrix.pNext <- head
            head <- addr &chain.CooperativeMatrix

        if parts.UntypedPointers then
            chain.UntypedPointers.pNext <- head
            head <- addr &chain.UntypedPointers

        if parts.Fault then
            chain.Fault.pNext <- head
            head <- addr &chain.Fault

        if parts.DescriptorHeap then
            chain.DescriptorHeap.pNext <- head
            head <- addr &chain.DescriptorHeap

        if parts.Maintenance5 then
            chain.Maintenance5.pNext <- head
            head <- addr &chain.Maintenance5

        if parts.Vulkan13 then
            chain.Vulkan13.pNext <- head
            head <- addr &chain.Vulkan13

        if parts.Vulkan12 then
            chain.Vulkan12.pNext <- head
            head <- addr &chain.Vulkan12

        chain.Features2.pNext <- head
        addr &chain.Features2

    /// Sets the recipe's five features (and deviceFault when `fault`) to VK_TRUE, for device creation.
    let enableRecipe (chain: byref<FeatureChain>) (fault: bool) =
        chain.Features2.features.shaderStorageBufferArrayDynamicIndexing <- VkConst.VK_TRUE
        chain.Vulkan12.bufferDeviceAddress <- VkConst.VK_TRUE
        chain.Vulkan12.timelineSemaphore <- VkConst.VK_TRUE
        chain.Vulkan12.runtimeDescriptorArray <- VkConst.VK_TRUE
        chain.Vulkan13.synchronization2 <- VkConst.VK_TRUE
        chain.Maintenance5.maintenance5 <- VkConst.VK_TRUE
        chain.DescriptorHeap.descriptorHeap <- VkConst.VK_TRUE

        if fault then
            chain.Fault.deviceFault <- VkConst.VK_TRUE

    /// The parts a device creation links: the recipe's five structs, plus the fault struct when enabled.
    let forCreate (fault: bool) =
        {
            Vulkan12 = true
            Vulkan13 = true
            Maintenance5 = true
            DescriptorHeap = true
            Fault = fault
            UntypedPointers = false
            CooperativeMatrix = false
        }

    /// The parts a query may link on a device with core version `api` and these extensions.
    let forQuery (api: uint32) (extensions: Map<string, uint32>) =
        {
            Vulkan12 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_2 api
            Vulkan13 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_3 api
            Maintenance5 =
                VkVersion.atLeast VkVersion.VK_API_VERSION_1_4 api
                || extensions.ContainsKey "VK_KHR_maintenance5"
            DescriptorHeap = extensions.ContainsKey VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME
            Fault = extensions.ContainsKey VkConst.VK_EXT_DEVICE_FAULT_EXTENSION_NAME
            UntypedPointers = extensions.ContainsKey VkConst.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME
            CooperativeMatrix = extensions.ContainsKey VkConst.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
        }

/// Which structs a PropertiesChain links.
type PropertyParts =
    {
        Vulkan11: bool
        Vulkan12: bool
        Vulkan13: bool
        Vulkan14: bool
        DescriptorHeap: bool
        CooperativeMatrix: bool
    }

/// The properties the device check reads, in one frame like FeatureChain: VkPhysicalDeviceProperties2 ->
/// Vulkan11Properties (UUIDs) -> Vulkan12Properties (driver, float controls) -> Vulkan13Properties ->
/// Vulkan14Properties -> DescriptorHeapPropertiesEXT -> CooperativeMatrixPropertiesKHR. The per-version structs
/// stand in for the promoted ones (driver, ID, float controls), which are not linked beside them.
[<Struct; IsByRefLike; NoEquality; NoComparison>]
type PropertiesChain =
    val mutable Properties2: VkPhysicalDeviceProperties2
    val mutable Vulkan11: VkPhysicalDeviceVulkan11Properties
    val mutable Vulkan12: VkPhysicalDeviceVulkan12Properties
    val mutable Vulkan13: VkPhysicalDeviceVulkan13Properties
    val mutable Vulkan14: VkPhysicalDeviceVulkan14Properties
    val mutable DescriptorHeap: VkPhysicalDeviceDescriptorHeapPropertiesEXT
    val mutable CooperativeMatrix: VkPhysicalDeviceCooperativeMatrixPropertiesKHR

module PropertiesChain =
    let inline private addr (x: byref<'T>) : nativeint =
        NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte>(Unsafe.AsPointer(&x)))

    let link (chain: byref<PropertiesChain>) (parts: PropertyParts) : nativeint =
        chain.Properties2.sType <- VkPhysicalDeviceProperties2.StructureType
        chain.Vulkan11.sType <- VkPhysicalDeviceVulkan11Properties.StructureType
        chain.Vulkan12.sType <- VkPhysicalDeviceVulkan12Properties.StructureType
        chain.Vulkan13.sType <- VkPhysicalDeviceVulkan13Properties.StructureType
        chain.Vulkan14.sType <- VkPhysicalDeviceVulkan14Properties.StructureType
        chain.DescriptorHeap.sType <- VkPhysicalDeviceDescriptorHeapPropertiesEXT.StructureType
        chain.CooperativeMatrix.sType <- VkPhysicalDeviceCooperativeMatrixPropertiesKHR.StructureType
        let mutable head = 0n

        if parts.CooperativeMatrix then
            chain.CooperativeMatrix.pNext <- head
            head <- addr &chain.CooperativeMatrix

        if parts.DescriptorHeap then
            chain.DescriptorHeap.pNext <- head
            head <- addr &chain.DescriptorHeap

        if parts.Vulkan14 then
            chain.Vulkan14.pNext <- head
            head <- addr &chain.Vulkan14

        if parts.Vulkan13 then
            chain.Vulkan13.pNext <- head
            head <- addr &chain.Vulkan13

        if parts.Vulkan12 then
            chain.Vulkan12.pNext <- head
            head <- addr &chain.Vulkan12

        if parts.Vulkan11 then
            chain.Vulkan11.pNext <- head
            head <- addr &chain.Vulkan11

        chain.Properties2.pNext <- head
        addr &chain.Properties2

    let forQuery (api: uint32) (extensions: Map<string, uint32>) =
        {
            Vulkan11 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_2 api
            Vulkan12 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_2 api
            Vulkan13 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_3 api
            Vulkan14 = VkVersion.atLeast VkVersion.VK_API_VERSION_1_4 api
            DescriptorHeap = extensions.ContainsKey VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME
            CooperativeMatrix = extensions.ContainsKey VkConst.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
        }

/// How Device.create sets up a device beyond VulkanHeapRecipe section 1.
type DeviceOptions =
    {
        /// Also enable VK_EXT_device_fault with deviceFault, when the device supports both. The recipe enables
        /// VK_EXT_descriptor_heap only; this is the one addition V1's fault diagnostics need.
        DeviceFault: bool
        /// Also enable VK_EXT_memory_budget when supported. It has no feature bits; budget queries need it only
        /// supported, so this only matters for tools that require it enabled.
        MemoryBudget: bool
        /// Further device extensions without feature bits, such as VK_KHR_calibrated_timestamps, for tools and
        /// tests. Each must be supported.
        Extensions: string list
        /// The queue family; None picks the recipe's: the first family with COMPUTE, without GRAPHICS and with
        /// timestampValidBits > 0.
        QueueFamily: uint32 option
    }

module DeviceOptions =
    /// Exactly VulkanHeapRecipe section 1: one extension, the five features.
    let recipe =
        {
            DeviceFault = false
            MemoryBudget = false
            Extensions = []
            QueueFamily = None
        }

    /// The recipe plus VK_EXT_device_fault and VK_EXT_memory_budget where supported.
    let diagnostics =
        {
            DeviceFault = true
            MemoryBudget = true
            Extensions = []
            QueueFamily = None
        }

/// The recipe's queue family choice.
module QueueFamily =
    /// The first family with COMPUTE, without GRAPHICS and with timestampValidBits > 0 (VulkanHeapRecipe section 1).
    let recipe (families: VkQueueFamilyProperties[]) : uint32 option =
        families
        |> Array.tryFindIndex (fun f ->
            f.queueFlags.HasFlag VkQueueFlagBits.VK_QUEUE_COMPUTE_BIT
            && not (f.queueFlags.HasFlag VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT)
            && f.timestampValidBits > 0u)
        |> Option.map uint32

/// A logical device with its device-level dispatch table and its queue. Disposing waits for the device to go idle
/// and destroys it.
[<Sealed>]
type Device
    internal
    (
        instance: Instance,
        physical: VkPhysicalDevice,
        handle: VkDevice,
        commands: VkDeviceCommands,
        family: uint32,
        queue: VkQueue,
        apiVersion: uint32,
        extensions: string list
    ) =
    let mutable disposed = false

    member _.Instance = instance
    member _.PhysicalDevice = physical
    member _.Handle = handle
    member _.Commands = commands
    member _.QueueFamily = family
    member _.Queue = queue
    /// The device's core version (VkPhysicalDeviceProperties.apiVersion).
    member _.ApiVersion = apiVersion
    member _.EnabledExtensions = extensions

    /// True when a provider is enabled on this device: a core version it supports (and the instance asked for), or
    /// an enabled device or instance extension.
    member _.Provides(p: VkProvider) =
        match p with
        | VkProvider.Core(a, b) ->
            VkVersion.covers apiVersion (a, b)
            && VkVersion.covers instance.ApiVersion (a, b)
        | VkProvider.Extension(n, true) -> instance.HasExtension n
        | VkProvider.Extension(n, _) -> List.contains n extensions

    /// Bound device commands whose provider is enabled but whose entry point is NULL. Empty on a conformant driver:
    /// the acceptance check "every dispatch entry is non-null".
    member this.MissingEnabledCommands: string list =
        let missing = set commands.Missing

        VkMeta.Commands
        |> List.filter (fun c ->
            c.Level = VkCommandLevel.Device
            && missing.Contains c.Name
            && c.Providers |> List.exists this.Provides)
        |> List.map _.Name

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true

                // A lost device fails the wait; it is destroyed all the same.
                try
                    commands.DeviceWaitIdle(handle)
                with :? VulkanException ->
                    ()

                commands.DestroyDevice(handle)

module Device =
    /// Queries deviceFault over a chain that holds VkPhysicalDeviceFaultFeaturesEXT.
    let private supportsDeviceFault (instance: Instance) (physical: VkPhysicalDevice) api available =
        let mutable q = FeatureChain()
        FeatureChain.link &q (FeatureChain.forQuery api available) |> ignore
        instance.Commands.GetPhysicalDeviceFeatures2(physical, &q.Features2)
        q.Fault.deviceFault <> VkConst.VK_FALSE

    /// Creates the recipe's logical device on `physical`: Vulkan 1.4, VK_EXT_descriptor_heap (plus what `options`
    /// adds), the five features over FeatureChain, and one queue of the recipe's family. Raises VulkanException
    /// when vkCreateDevice fails and InvalidOperationException when the device lacks the extension or a family.
    let create (instance: Instance) (physical: VkPhysicalDevice) (options: DeviceOptions) : Device =
        use span =
            Telemetry.span "gpu.device.create" [ Attr.str TelemetryKeys.Backend "vulkan" ]

        let props = instance.Properties physical
        let available = instance.DeviceExtensions physical

        if not (available.ContainsKey VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME) then
            invalidOp (sprintf "%s does not support VK_EXT_descriptor_heap" props.deviceName.AsString)

        let family =
            match options.QueueFamily with
            | Some f -> f
            | None ->
                match QueueFamily.recipe (instance.QueueFamilies physical) with
                | Some f -> f
                | None ->
                    invalidOp (
                        sprintf
                            "%s has no queue family with COMPUTE, without GRAPHICS and with timestamps"
                            props.deviceName.AsString
                    )

        let fault =
            options.DeviceFault
            && available.ContainsKey VkConst.VK_EXT_DEVICE_FAULT_EXTENSION_NAME
            && supportsDeviceFault instance physical props.apiVersion available

        let extensions =
            [
                VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME
                if fault then
                    VkConst.VK_EXT_DEVICE_FAULT_EXTENSION_NAME
                if
                    options.MemoryBudget
                    && available.ContainsKey VkConst.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
                then
                    VkConst.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
                for x in options.Extensions do
                    if not (available.ContainsKey x) then
                        invalidOp (sprintf "%s does not support %s" props.deviceName.AsString x)

                    x
            ]
            |> List.distinct

        let mutable chain = FeatureChain()
        let head = FeatureChain.link &chain (FeatureChain.forCreate fault)
        FeatureChain.enableRecipe &chain fault

        match VkChain.problems "VkDeviceCreateInfo" head with
        | [] -> ()
        | ps -> invalidOp (sprintf "the device feature chain is invalid: %s" (String.concat "; " ps))

        let mutable priority = 1.0f
        let mutable queueInfo = VkDeviceQueueCreateInfo.Create()
        queueInfo.queueFamilyIndex <- family
        queueInfo.queueCount <- 1u
        queueInfo.pQueuePriorities <- NativePtr.toNativeInt &&priority
        use names = new PinnedStrings(extensions)
        let mutable ci = VkDeviceCreateInfo.Create()
        ci.pNext <- head
        ci.queueCreateInfoCount <- 1u
        ci.pQueueCreateInfos <- NativePtr.toNativeInt &&queueInfo
        ci.enabledExtensionCount <- names.Count
        ci.ppEnabledExtensionNames <- names.Pointer
        ci.pEnabledFeatures <- 0n // the chain holds a VkPhysicalDeviceFeatures2 (VUID-VkDeviceCreateInfo-pNext-00373)
        let mutable handle = VkDevice.Null
        instance.Commands.CreateDevice(physical, &ci, &handle)

        // Device commands of instance extensions (VK_EXT_debug_utils) fall back to vkGetInstanceProcAddr.
        let instanceExtensionCommands =
            VkMeta.Commands
            |> List.filter (fun c ->
                c.Level = VkCommandLevel.Device
                && c.Providers
                   |> List.exists (function
                       | VkProvider.Extension(n, true) -> instance.HasExtension n
                       | _ -> false))
            |> List.map _.Name
            |> set

        let load (name: string) =
            let b = VkText.utf8z name
            use p = fixed b
            let fp = instance.Commands.GetDeviceProcAddr(handle, NativePtr.toNativeInt p)

            if fp = 0n && instanceExtensionCommands.Contains name then
                instance.Loader.ProcAddr(instance.Handle, name)
            else
                fp

        let commands = VkDeviceCommands(load)
        let mutable queue = VkQueue.Null
        commands.GetDeviceQueue(handle, family, 0u, &queue)

        let device =
            new Device(instance, physical, handle, commands, family, queue, props.apiVersion, extensions)

        span.Add(Attr.str "device" props.deviceName.AsString)
        span.Add(Attr.int "queue_family" (int family))
        span.Add(Attr.str "extensions" (String.concat "," extensions))
        span.Add(Attr.int "commands_missing" device.MissingEnabledCommands.Length)
        device
