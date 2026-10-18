namespace Lrm.Gpu.Vulkan.Interop

open System
open System.Runtime.InteropServices
open System.Text

/// Strings at the C boundary: null-terminated UTF-8 in both directions.
[<RequireQualifiedAccess>]
module VkText =
    /// The string in a char[N] member: UTF-8 up to the first zero byte.
    let ofBytes (bytes: ReadOnlySpan<byte>) =
        let n = bytes.IndexOf 0uy
        Encoding.UTF8.GetString(if n < 0 then bytes else bytes.Slice(0, n))

    /// A null-terminated UTF-8 string at `p`, or "" for NULL.
    let ofPointer (p: nativeint) =
        if p = 0n then "" else Marshal.PtrToStringUTF8 p

    /// Null-terminated UTF-8 bytes for a `const char*`; pin them for the call with `use p = fixed bytes`.
    let utf8z (s: string) = Encoding.UTF8.GetBytes(s + "\000")

[<AutoOpen>]
module VkTextExtensions =
    type Array256_byte with
        /// The string in a char[256] member (deviceName, extensionName, driverName, description and the like).
        member this.AsString =
            let mutable copy = this
            VkText.ofBytes (MemoryMarshal.CreateReadOnlySpan(&copy.e0, 256))

/// VK_MAKE_API_VERSION and its decoders.
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

    /// True when a core version (major, minor) is at most `actual`.
    let covers (actual: uint32) (major': int, minor': int) =
        atLeast (make 0u (uint32 major') (uint32 minor') 0u) actual

    let toString (v: uint32) =
        sprintf "%d.%d.%d" (major v) (minor v) (patch v)
