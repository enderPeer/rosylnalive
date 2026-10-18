namespace Lrm.Gpu.Vulkan.Interop

open System

// Markers on the generated items (Vk.g.fs). The ABI oracle (tools/abi-oracle/vulkan) finds every bound item through
// them and asserts its runtime layout against the Vulkan headers with cl.exe.

/// A generated struct and the C struct it mirrors.
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)>]
type VkStructAttribute(name: string) =
    inherit Attribute()
    member _.Name = name

/// A generated union (explicit layout, every member at offset 0) and the C union it mirrors.
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)>]
type VkUnionAttribute(name: string) =
    inherit Attribute()
    member _.Name = name

/// A generated enum or FlagBits type; its cases have the C names and values.
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Enum, AllowMultiple = false)>]
type VkEnumAttribute(name: string) =
    inherit Attribute()
    member _.Name = name

/// A generated handle: a struct over one nativeint (dispatchable) or uint64 (non-dispatchable).
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)>]
type VkHandleAttribute(name: string, dispatchable: bool) =
    inherit Attribute()
    member _.Name = name
    member _.Dispatchable = dispatchable

/// A generated inline array that stands for a C array member such as char deviceName[256].
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)>]
type VkArrayAttribute(element: string, count: int) =
    inherit Attribute()
    member _.Element = element
    member _.Count = count

/// The unmanaged delegate of a command; its C function type is PFN_<name> in vulkan_core.h.
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Delegate, AllowMultiple = false)>]
type VkCommandAttribute(name: string) =
    inherit Attribute()
    member _.Name = name

/// The unmanaged delegate of a vk.xml funcpointer type such as PFN_vkDebugUtilsMessengerCallbackEXT.
[<Sealed; AllowNullLiteral; AttributeUsage(AttributeTargets.Delegate, AllowMultiple = false)>]
type VkFuncPointerAttribute(name: string) =
    inherit Attribute()
    member _.Name = name
