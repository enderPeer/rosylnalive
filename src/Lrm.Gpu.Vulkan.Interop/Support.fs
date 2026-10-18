namespace Lrm.Gpu.Vulkan.Interop

open System
open System.Runtime.InteropServices

/// A command returned a VkResult outside its success codes (vk.xml `successcodes`).
exception VulkanException of command: string * result: VkResult with
    override this.Message =
        sprintf "%s failed: %A (%d)" this.command this.result (int this.result)

/// A command's entry point was NULL when its dispatch table was loaded: its core version or extension is not
/// enabled on this instance or device.
exception VulkanMissingCommandException of command: string with
    override this.Message =
        sprintf "%s is not loaded: its core version or extension is not enabled" this.command

/// How a dispatch table loads one delegate.
[<RequireQualifiedAccess>]
module VkLoad =
    /// The delegate for `name`, or null (recorded in `missing`) when `load` returns NULL. Called once per command
    /// when a table is built, never per call.
    let fn<'D when 'D :> Delegate and 'D: null>
        (missing: ResizeArray<string>)
        (load: string -> nativeint)
        (name: string)
        : 'D =
        let p = load name

        if p = 0n then
            missing.Add name
            null
        else
            Marshal.GetDelegateForFunctionPointer<'D> p

/// The dispatch level of a command (vk.xml: its first parameter).
[<RequireQualifiedAccess>]
type VkCommandLevel =
    /// No dispatchable handle: vkCreateInstance, vkEnumerateInstance*.
    | Global
    /// A VkInstance or VkPhysicalDevice (and vkGetDeviceProcAddr).
    | Instance
    /// A VkDevice, VkQueue or VkCommandBuffer.
    | Device

/// What makes a command, enum value or struct available.
[<RequireQualifiedAccess>]
type VkProvider =
    /// A core version (vk.xml VK_BASE_VERSION_x_y, VK_COMPUTE_VERSION_x_y, VK_GRAPHICS_VERSION_x_y or VK_VERSION_x_y).
    | Core of major: int * minor: int
    /// An extension; `instance` is true for an instance extension.
    | Extension of name: string * instance: bool

type VkExtensionMeta =
    {
        Name: string
        /// The registry number (enum offsets are 1000000000 + (number - 1) * 1000 + offset).
        Number: int
        Instance: bool
    }

type VkStructMeta =
    {
        Name: string
        Type: Type
        IsUnion: bool
        /// The value of the sType member, when the struct has one.
        SType: VkStructureType voption
        /// vk.xml `structextends`: the structs whose pNext chain may hold this one.
        Extends: string list
        ReturnedOnly: bool
        /// The struct may appear more than once in one chain.
        AllowDuplicate: bool
        /// The generator's C layout (x64): size, alignment, and (member, offset, size) in declaration order.
        Size: int
        Align: int
        Fields: (string * int * int) list
    }

type VkCommandMeta =
    {
        Name: string
        Level: VkCommandLevel
        /// The bound core versions and extensions that provide the command (vk.xml `require` blocks).
        Providers: VkProvider list
        /// vk.xml `successcodes`, for commands that return VkResult.
        SuccessCodes: VkResult list
        Delegate: Type
    }
