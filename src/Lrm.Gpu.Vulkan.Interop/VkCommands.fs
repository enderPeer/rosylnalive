// Entry points. Only vkGetInstanceProcAddr is exported-symbol bound (via NativeLibrary);
// everything else is fetched through vkGetInstanceProcAddr / vkGetDeviceProcAddr as the loader intends.
//
// Calls currently go through delegates from Marshal.GetDelegateForFunctionPointer. That is fine for
// queries; hot command-buffer paths (vkCmdPushDataEXT, dispatch) will want raw calli thunks later.
namespace Lrm.Gpu.Vulkan

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open System.Text
open Microsoft.FSharp.NativeInterop

// ---- PFN_* (signatures from vk.xml <command> entries; VKAPI_CALL == platform default) ----

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkGetInstanceProcAddr = delegate of instance: VkInstance * pName: nativeptr<byte> -> nativeint

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkGetDeviceProcAddr = delegate of device: VkDevice * pName: nativeptr<byte> -> nativeint

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkEnumerateInstanceVersion = delegate of pApiVersion: nativeptr<uint32> -> VkResult

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkCreateInstance =
    delegate of
        pCreateInfo: nativeptr<VkInstanceCreateInfo> * pAllocator: nativeint * pInstance: nativeptr<VkInstance> ->
            VkResult

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkDestroyInstance = delegate of instance: VkInstance * pAllocator: nativeint -> unit

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkEnumeratePhysicalDevices =
    delegate of
        instance: VkInstance * pPhysicalDeviceCount: nativeptr<uint32> * pPhysicalDevices: nativeptr<VkPhysicalDevice> ->
            VkResult

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkGetPhysicalDeviceProperties =
    delegate of physicalDevice: VkPhysicalDevice * pProperties: nativeptr<VkPhysicalDeviceProperties> -> unit

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkGetPhysicalDeviceProperties2 =
    delegate of physicalDevice: VkPhysicalDevice * pProperties: nativeptr<VkPhysicalDeviceProperties2> -> unit

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkGetPhysicalDeviceFeatures2 =
    delegate of physicalDevice: VkPhysicalDevice * pFeatures: nativeptr<VkPhysicalDeviceFeatures2> -> unit

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_vkEnumerateDeviceExtensionProperties =
    delegate of
        physicalDevice: VkPhysicalDevice *
        pLayerName: nativeptr<byte> *
        pPropertyCount: nativeptr<uint32> *
        pProperties: nativeptr<VkExtensionProperties> ->
            VkResult

exception VulkanException of call: string * result: VkResult with
    override this.Message = sprintf "%s failed: %A" this.call this.result

module internal CStr =
    /// Null-terminated UTF-8 bytes; callers pin with `use p = fixed bytes`.
    let bytes (s: string) = Encoding.UTF8.GetBytes(s + "\000")

module internal Check =
    let ok call (r: VkResult) =
        if r <> VkResult.VK_SUCCESS then
            raise (VulkanException(call, r))

/// The Vulkan loader (vulkan-1.dll / libvulkan.so.1) and the global (instance-less) commands.
type VulkanLoader private (lib: nativeint, getInstanceProcAddr: PFN_vkGetInstanceProcAddr) =
    static let candidates =
        if OperatingSystem.IsWindows() then
            [ "vulkan-1.dll" ]
        else
            [ "libvulkan.so.1"; "libvulkan.so" ]

    let load (instance: VkInstance) (name: string) : 'T voption =
        let b = CStr.bytes name
        use p = fixed b
        let fp = getInstanceProcAddr.Invoke(instance, p)

        if fp = 0n then
            ValueNone
        else
            ValueSome(Marshal.GetDelegateForFunctionPointer<'T> fp)

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
            | _ -> None)

    member _.LoadInstanceProc<'T when 'T :> Delegate>(instance, name) : 'T voption = load instance name

    member this.LoadInstanceProcOrFail<'T when 'T :> Delegate>(instance, name) : 'T =
        match load instance name with
        | ValueSome d -> d
        | ValueNone -> failwithf "vkGetInstanceProcAddr returned NULL for %s" name

    /// Highest instance-level API version the loader supports (1.0 if vkEnumerateInstanceVersion is absent).
    member this.InstanceVersion =
        match load (VkInstance 0n) "vkEnumerateInstanceVersion" with
        | ValueNone -> VkVersion.VK_API_VERSION_1_0
        | ValueSome(f: PFN_vkEnumerateInstanceVersion) ->
            let mutable v = 0u
            Check.ok "vkEnumerateInstanceVersion" (f.Invoke(&&v))
            v

    member this.CreateInstance(applicationName: string, apiVersion: uint32) : Instance =
        let create =
            this.LoadInstanceProcOrFail<PFN_vkCreateInstance>(VkInstance 0n, "vkCreateInstance")

        let appName = CStr.bytes applicationName
        let engineName = CStr.bytes "LRM"
        use pApp = fixed appName
        use pEngine = fixed engineName
        let mutable app = VkApplicationInfo()
        app.sType <- VkStructureType.VK_STRUCTURE_TYPE_APPLICATION_INFO
        app.pApplicationName <- NativePtr.toNativeInt pApp
        app.applicationVersion <- 1u
        app.pEngineName <- NativePtr.toNativeInt pEngine
        app.engineVersion <- 1u
        app.apiVersion <- apiVersion
        let mutable ci = VkInstanceCreateInfo()
        ci.sType <- VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        ci.pApplicationInfo <- NativePtr.toNativeInt &&app
        let mutable handle = VkInstance 0n
        Check.ok "vkCreateInstance" (create.Invoke(&&ci, 0n, &&handle))
        new Instance(this, handle)

    interface IDisposable with
        member _.Dispose() = NativeLibrary.Free lib

/// A VkInstance plus the instance-level commands loaded for it.
and Instance internal (loader: VulkanLoader, handle: VkInstance) =
    let get name =
        loader.LoadInstanceProcOrFail(handle, name)

    let destroy: PFN_vkDestroyInstance = get "vkDestroyInstance"
    let mutable disposed = false

    member _.Handle = handle
    member val EnumeratePhysicalDevices: PFN_vkEnumeratePhysicalDevices = get "vkEnumeratePhysicalDevices"
    member val GetPhysicalDeviceProperties: PFN_vkGetPhysicalDeviceProperties = get "vkGetPhysicalDeviceProperties"
    member val GetPhysicalDeviceProperties2: PFN_vkGetPhysicalDeviceProperties2 = get "vkGetPhysicalDeviceProperties2"
    member val GetPhysicalDeviceFeatures2: PFN_vkGetPhysicalDeviceFeatures2 = get "vkGetPhysicalDeviceFeatures2"

    member val EnumerateDeviceExtensionProperties: PFN_vkEnumerateDeviceExtensionProperties =
        get "vkEnumerateDeviceExtensionProperties"

    /// For loading device-level commands once a VkDevice exists (not needed by the device check).
    member val GetDeviceProcAddr: PFN_vkGetDeviceProcAddr = get "vkGetDeviceProcAddr"

    member this.PhysicalDevices: VkPhysicalDevice[] =
        let mutable count = 0u
        Check.ok "vkEnumeratePhysicalDevices" (this.EnumeratePhysicalDevices.Invoke(handle, &&count, NativePtr.nullPtr))
        let devices = Array.zeroCreate<VkPhysicalDevice>(int count)

        if count > 0u then
            use p = fixed devices
            let r = this.EnumeratePhysicalDevices.Invoke(handle, &&count, p)

            if r <> VkResult.VK_SUCCESS && r <> VkResult.VK_INCOMPLETE then
                raise (VulkanException("vkEnumeratePhysicalDevices", r))

        Array.truncate (int count) devices

    member this.DeviceExtensions(device: VkPhysicalDevice) : Map<string, uint32> =
        let rec query () =
            let mutable count = 0u

            Check.ok
                "vkEnumerateDeviceExtensionProperties"
                (this.EnumerateDeviceExtensionProperties.Invoke(device, NativePtr.nullPtr, &&count, NativePtr.nullPtr))

            let props = Array.zeroCreate<VkExtensionProperties>(int count)

            let r =
                if count = 0u then
                    VkResult.VK_SUCCESS
                else
                    use p = fixed props
                    this.EnumerateDeviceExtensionProperties.Invoke(device, NativePtr.nullPtr, &&count, p)

            match r with
            | VkResult.VK_SUCCESS ->
                props
                |> Array.truncate (int count)
                |> Array.map (fun e -> e.extensionName.AsString, e.specVersion)
                |> Map.ofArray
            | VkResult.VK_INCOMPLETE -> query () // list grew between the two calls
            | r -> raise (VulkanException("vkEnumerateDeviceExtensionProperties", r))

        query ()

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                destroy.Invoke(handle, 0n)
