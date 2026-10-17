// Hand-written bindings to the Vulkan C API (vulkan_core.h): only what the device check needs.
// Names, member order and enum values follow the Khronos registry vk.xml (VK_HEADER_VERSION 365).
// Layouts assume a 64-bit process; the console app verifies them against vk.xml-derived offsets.
namespace Lrm.Gpu.Vulkan

open System
open System.Runtime.InteropServices
open System.Text

type VkBool32 = uint32

[<RequireQualifiedAccess>]
module VkConst =
    [<Literal>]
    let VK_TRUE = 1u

    [<Literal>]
    let VK_FALSE = 0u

    [<Literal>]
    let VK_MAX_EXTENSION_NAME_SIZE = 256

    [<Literal>]
    let VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256

    [<Literal>]
    let VK_MAX_DRIVER_NAME_SIZE = 256

    [<Literal>]
    let VK_MAX_DRIVER_INFO_SIZE = 256

    [<Literal>]
    let VK_UUID_SIZE = 16

    [<Literal>]
    let VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME = "VK_EXT_descriptor_heap"

    [<Literal>]
    let VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME = "VK_KHR_shader_untyped_pointers"

    [<Literal>]
    let VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME = "VK_KHR_cooperative_matrix"

/// VK_MAKE_API_VERSION and the matching decoders.
[<RequireQualifiedAccess>]
module VkVersion =
    let make (variant: uint32) (major: uint32) (minor: uint32) (patch: uint32) =
        (variant <<< 29) ||| (major <<< 22) ||| (minor <<< 12) ||| patch

    let variant (v: uint32) = v >>> 29
    let major (v: uint32) = (v >>> 22) &&& 0x7Fu
    let minor (v: uint32) = (v >>> 12) &&& 0x3FFu
    let patch (v: uint32) = v &&& 0xFFFu
    let VK_API_VERSION_1_0 = make 0u 1u 0u 0u
    let VK_API_VERSION_1_1 = make 0u 1u 1u 0u
    let VK_API_VERSION_1_2 = make 0u 1u 2u 0u
    let VK_API_VERSION_1_3 = make 0u 1u 3u 0u
    let VK_API_VERSION_1_4 = make 0u 1u 4u 0u

    /// Compares major.minor only, as the spec does for core-version gating.
    let atLeast (required: uint32) (actual: uint32) =
        (major actual, minor actual) >= (major required, minor required)

    let toString (v: uint32) =
        sprintf "%d.%d.%d" (major v) (minor v) (patch v)

type VkStructureType =
    | VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
    | VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT = 1000135008
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT = 1000135009
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES = 1000196000
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR = 1000387000
    | VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR = 1000506000

type VkResult =
    | VK_SUCCESS = 0
    | VK_NOT_READY = 1
    | VK_TIMEOUT = 2
    | VK_EVENT_SET = 3
    | VK_EVENT_RESET = 4
    | VK_INCOMPLETE = 5
    | VK_ERROR_OUT_OF_HOST_MEMORY = -1
    | VK_ERROR_OUT_OF_DEVICE_MEMORY = -2
    | VK_ERROR_INITIALIZATION_FAILED = -3
    | VK_ERROR_DEVICE_LOST = -4
    | VK_ERROR_MEMORY_MAP_FAILED = -5
    | VK_ERROR_LAYER_NOT_PRESENT = -6
    | VK_ERROR_EXTENSION_NOT_PRESENT = -7
    | VK_ERROR_FEATURE_NOT_PRESENT = -8
    | VK_ERROR_INCOMPATIBLE_DRIVER = -9
    | VK_ERROR_TOO_MANY_OBJECTS = -10
    | VK_ERROR_FORMAT_NOT_SUPPORTED = -11
    | VK_ERROR_FRAGMENTED_POOL = -12
    | VK_ERROR_UNKNOWN = -13

type VkPhysicalDeviceType =
    | VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
    | VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
    | VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
    | VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
    | VK_PHYSICAL_DEVICE_TYPE_CPU = 4

/// VkDriverId is an open enum (vendors keep being added), so it stays a raw int32.
type VkDriverId = int32

// ---- Dispatchable handles (pointer-sized) ----

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkInstance =
    val Handle: nativeint
    new(h) = { Handle = h }

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkPhysicalDevice =
    val Handle: nativeint
    new(h) = { Handle = h }

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkDevice =
    val Handle: nativeint
    new(h) = { Handle = h }

// ---- Fixed-size C arrays: blittable stand-ins for T[N] struct members ----

module internal FixedText =
    let decode (bytes: ReadOnlySpan<byte>) =
        let n = bytes.IndexOf 0uy
        Encoding.UTF8.GetString(if n < 0 then bytes else bytes.Slice(0, n))

/// char[256]: VK_MAX_EXTENSION_NAME_SIZE, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE, VK_MAX_DRIVER_*_SIZE.
[<Struct; StructLayout(LayoutKind.Sequential, Size = 256)>]
type FixedBytes256 =
    val private e0: byte

    member this.AsString =
        let mutable copy = this
        FixedText.decode (MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(&copy, 1)))

/// uint8_t[VK_UUID_SIZE]
[<Struct; StructLayout(LayoutKind.Sequential, Size = 16)>]
type FixedBytes16 =
    val private e0: byte

    member this.ToArray() =
        let mutable copy = this
        MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(&copy, 1)).ToArray()

/// uint32_t[3]
[<Struct; StructLayout(LayoutKind.Sequential, Size = 12)>]
type FixedUInt32x3 =
    val private e0: uint32

/// uint32_t[2]
[<Struct; StructLayout(LayoutKind.Sequential, Size = 8)>]
type FixedUInt32x2 =
    val private e0: uint32

/// float[2]
[<Struct; StructLayout(LayoutKind.Sequential, Size = 8)>]
type FixedFloat32x2 =
    val private e0: float32

// ---- Instance creation ----

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkApplicationInfo =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable pApplicationName: nativeint
    val mutable applicationVersion: uint32
    val mutable pEngineName: nativeint
    val mutable engineVersion: uint32
    val mutable apiVersion: uint32

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkInstanceCreateInfo =
    val mutable sType: VkStructureType
    val mutable pNext: nativeint
    val mutable flags: uint32
    val mutable pApplicationInfo: nativeint
    val mutable enabledLayerCount: uint32
    val mutable ppEnabledLayerNames: nativeint
    val mutable enabledExtensionCount: uint32
    val mutable ppEnabledExtensionNames: nativeint

[<Struct; StructLayout(LayoutKind.Sequential)>]
type VkExtensionProperties =
    val mutable extensionName: FixedBytes256
    val mutable specVersion: uint32
