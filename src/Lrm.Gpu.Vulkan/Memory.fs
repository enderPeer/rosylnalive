namespace Lrm.Gpu.Vulkan

open System
open Lrm.Gpu.Vulkan.Interop

/// The buffers the backend creates, each with its usage and memory rule (VulkanHeapRecipe section 2).
[<RequireQualifiedAccess>]
type BufferKind =
    /// An arena page, the tensor table (slot 0) or the dummy page behind unused slots: STORAGE_BUFFER |
    /// SHADER_DEVICE_ADDRESS | TRANSFER_SRC | TRANSFER_DST in DEVICE_LOCAL memory that is not HOST_VISIBLE.
    | Page
    /// The resource heap: DESCRIPTOR_HEAP | SHADER_DEVICE_ADDRESS in DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory,
    /// mapped persistently and written in place.
    | ResourceHeap
    /// The fallback resource heap, only when no DEVICE_LOCAL | HOST_VISIBLE type exists (or when a test forces it):
    /// DESCRIPTOR_HEAP | SHADER_DEVICE_ADDRESS | TRANSFER_DST in DEVICE_LOCAL memory, filled by a copy.
    | ResourceHeapCopied
    /// Upload staging (and the fallback heap's descriptor staging): TRANSFER_SRC in HOST_VISIBLE | HOST_COHERENT memory,
    /// preferably not HOST_CACHED.
    | Upload
    /// Readback staging: TRANSFER_DST in HOST_VISIBLE | HOST_COHERENT memory, HOST_CACHED when a type has it.
    | Readback

[<RequireQualifiedAccess>]
module BufferKind =
    let all =
        [
            BufferKind.Page
            BufferKind.ResourceHeap
            BufferKind.ResourceHeapCopied
            BufferKind.Upload
            BufferKind.Readback
        ]

    /// The name used in telemetry (`gpu.memory_type`'s `kind`) and in the run manifest.
    let name (k: BufferKind) =
        match k with
        | BufferKind.Page -> "page"
        | BufferKind.ResourceHeap -> "resource_heap"
        | BufferKind.ResourceHeapCopied -> "resource_heap_copied"
        | BufferKind.Upload -> "upload_staging"
        | BufferKind.Readback -> "readback_staging"

    /// The buffer usage (VkBufferUsageFlags2, chained through VkBufferUsageFlags2CreateInfo).
    let usage (k: BufferKind) : VkBufferUsageFlags2 =
        match k with
        | BufferKind.Page ->
            VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
            ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
            ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
            ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
        | BufferKind.ResourceHeap ->
            VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT
            ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
        | BufferKind.ResourceHeapCopied ->
            VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT
            ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
            ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
        | BufferKind.Upload -> VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
        | BufferKind.Readback -> VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_DST_BIT

    /// True when the buffer is mapped persistently.
    let mapped (k: BufferKind) =
        match k with
        | BufferKind.ResourceHeap
        | BufferKind.Upload
        | BufferKind.Readback -> true
        | BufferKind.Page
        | BufferKind.ResourceHeapCopied -> false

    /// True when the buffer has a device address (its memory is allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT).
    let deviceAddress (k: BufferKind) =
        (usage k).HasFlag VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT

/// Memory types chosen by property flags among a buffer's memoryTypeBits, never by index (VulkanHeapRecipe section 2).
[<RequireQualifiedAccess>]
module MemoryTypes =
    let private none: VkMemoryPropertyFlags = LanguagePrimitives.EnumOfValue 0u

    let private deviceLocal =
        VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

    let private hostVisible =
        VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT

    let private hostCoherent =
        VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT

    let private hostCached = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_CACHED_BIT

    let private lazily =
        VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT

    let private protectedBit = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_PROTECTED_BIT

    /// The tiers of (required, forbidden) flags for a kind, best first. A type matches a tier when it has every
    /// required flag and none of the forbidden ones; LAZILY_ALLOCATED and PROTECTED memory is never used.
    let tiers (k: BufferKind) : (VkMemoryPropertyFlags * VkMemoryPropertyFlags) list =
        let never = lazily ||| protectedBit

        match k with
        | BufferKind.Page -> [ deviceLocal, hostVisible ||| never ]
        | BufferKind.ResourceHeap -> [ deviceLocal ||| hostVisible ||| hostCoherent, never ]
        | BufferKind.ResourceHeapCopied -> [ deviceLocal, hostVisible ||| never; deviceLocal, never ]
        | BufferKind.Upload ->
            [
                hostVisible ||| hostCoherent, hostCached ||| deviceLocal ||| never
                hostVisible ||| hostCoherent, hostCached ||| never
                hostVisible ||| hostCoherent, never
            ]
        | BufferKind.Readback ->
            [
                hostVisible ||| hostCoherent ||| hostCached, deviceLocal ||| never
                hostVisible ||| hostCoherent ||| hostCached, never
                hostVisible ||| hostCoherent, never
            ]

    /// The first memory type (index order) of the first tier that matches, among the types `bits` allows.
    let choose (types: VkMemoryPropertyFlags[]) (bits: uint32) (k: BufferKind) : int option =
        let matches (required: VkMemoryPropertyFlags, forbidden: VkMemoryPropertyFlags) i =
            i < 32
            && (bits &&& (1u <<< i)) <> 0u
            && (types[i] &&& required) = required
            && (types[i] &&& forbidden) = none

        tiers k
        |> List.tryPick (fun tier -> [ 0 .. types.Length - 1 ] |> List.tryFind (matches tier))

    /// True when the device has a type for the resource heap in place (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT);
    /// otherwise the heap takes the copy path (VulkanHeapRecipe section 2, fallback).
    let hasHeapInPlace (types: VkMemoryPropertyFlags[]) =
        (choose types UInt32.MaxValue BufferKind.ResourceHeap).IsSome

    /// The flags as the manifest and telemetry write them: "device_local|host_visible|...", or "none".
    let flagsText (f: VkMemoryPropertyFlags) =
        [
            deviceLocal, "device_local"
            hostVisible, "host_visible"
            hostCoherent, "host_coherent"
            hostCached, "host_cached"
            lazily, "lazily_allocated"
            protectedBit, "protected"
        ]
        |> List.filter (fun (b, _) -> f.HasFlag b)
        |> List.map snd
        |> function
            | [] -> "none"
            | xs -> String.concat "|" xs
