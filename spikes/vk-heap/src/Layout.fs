/// Struct layouts checked at run time. Each struct the spike hand-writes is listed here a second time, as the member
/// list of its C declaration, transcribed from the Vulkan reference pages (docs.vulkan.org/refpages, VK_EXT_descriptor_heap
/// revision 1; core structs from vulkan_core.h as published in the same specification). A small C layout calculator
/// computes the x64 offsets and sizes from those member types, and `check` compares them with the CLR layout of the
/// F# struct (names, order, offsets, size, blittability). Two independent transcriptions have to agree.
module VkHeapSpike.Layout

open System
open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

/// C member types on x64 (LLP64 Windows and LP64 Linux agree on every type used here).
type CType =
    | U8
    | U16
    | U32 // uint32_t, int32_t, VkBool32, VkFlags, every enum, float
    | U64 // uint64_t, VkDeviceSize, VkDeviceAddress, VkFlags64, non-dispatchable handles
    | Ptr // pointers, dispatchable handles, size_t
    | S of string // a nested struct or union, by name
    | Arr of CType * int

type CDecl =
    | Struct of (string * CType) list
    | Union of (string * CType) list

/// The C declarations, member by member. Sources: the reference page of each type.
let decls: (string * CDecl) list =
    let hdr = [ "sType", U32; "pNext", Ptr ]

    [
        // ---- core ----
        "VkExtent3D", Struct [ "width", U32; "height", U32; "depth", U32 ]
        "VkQueueFamilyProperties",
        Struct
            [
                "queueFlags", U32
                "queueCount", U32
                "timestampValidBits", U32
                "minImageTransferGranularity", S "VkExtent3D"
            ]
        "VkMemoryType", Struct [ "propertyFlags", U32; "heapIndex", U32 ]
        "VkMemoryHeap", Struct [ "size", U64; "flags", U32 ]
        "VkPhysicalDeviceMemoryProperties",
        Struct
            [
                "memoryTypeCount", U32
                "memoryTypes", Arr(S "VkMemoryType", 32)
                "memoryHeapCount", U32
                "memoryHeaps", Arr(S "VkMemoryHeap", 16)
            ]
        "VkDeviceQueueCreateInfo",
        Struct(
            hdr
            @ [
                "flags", U32
                "queueFamilyIndex", U32
                "queueCount", U32
                "pQueuePriorities", Ptr
            ]
        )
        "VkDeviceCreateInfo",
        Struct(
            hdr
            @ [
                "flags", U32
                "queueCreateInfoCount", U32
                "pQueueCreateInfos", Ptr
                "enabledLayerCount", U32
                "ppEnabledLayerNames", Ptr
                "enabledExtensionCount", U32
                "ppEnabledExtensionNames", Ptr
                "pEnabledFeatures", Ptr
            ]
        )
        "VkPhysicalDeviceMaintenance5Features", Struct(hdr @ [ "maintenance5", U32 ])
        "VkBufferCreateInfo",
        Struct(
            hdr
            @ [
                "flags", U32
                "size", U64
                "usage", U32
                "sharingMode", U32
                "queueFamilyIndexCount", U32
                "pQueueFamilyIndices", Ptr
            ]
        )
        "VkMemoryRequirements", Struct [ "size", U64; "alignment", U64; "memoryTypeBits", U32 ]
        "VkMemoryAllocateInfo", Struct(hdr @ [ "allocationSize", U64; "memoryTypeIndex", U32 ])
        "VkMemoryAllocateFlagsInfo", Struct(hdr @ [ "flags", U32; "deviceMask", U32 ])
        "VkBufferDeviceAddressInfo", Struct(hdr @ [ "buffer", U64 ])
        "VkMappedMemoryRange", Struct(hdr @ [ "memory", U64; "offset", U64; "size", U64 ])
        "VkShaderModuleCreateInfo", Struct(hdr @ [ "flags", U32; "codeSize", Ptr; "pCode", Ptr ])
        "VkPipelineShaderStageCreateInfo",
        Struct(
            hdr
            @ [
                "flags", U32
                "stage", U32
                "module", U64
                "pName", Ptr
                "pSpecializationInfo", Ptr
            ]
        )
        "VkComputePipelineCreateInfo",
        Struct(
            hdr
            @ [
                "flags", U32
                "stage", S "VkPipelineShaderStageCreateInfo"
                "layout", U64
                "basePipelineHandle", U64
                "basePipelineIndex", U32
            ]
        )
        "VkPipelineCreateFlags2CreateInfo", Struct(hdr @ [ "flags", U64 ])
        "VkCommandPoolCreateInfo", Struct(hdr @ [ "flags", U32; "queueFamilyIndex", U32 ])
        "VkCommandBufferAllocateInfo",
        Struct(hdr @ [ "commandPool", U64; "level", U32; "commandBufferCount", U32 ])
        "VkCommandBufferBeginInfo", Struct(hdr @ [ "flags", U32; "pInheritanceInfo", Ptr ])
        "VkBufferCopy", Struct [ "srcOffset", U64; "dstOffset", U64; "size", U64 ]
        "VkMemoryBarrier2",
        Struct(
            hdr
            @ [
                "srcStageMask", U64
                "srcAccessMask", U64
                "dstStageMask", U64
                "dstAccessMask", U64
            ]
        )
        "VkDependencyInfo",
        Struct(
            hdr
            @ [
                "dependencyFlags", U32
                "memoryBarrierCount", U32
                "pMemoryBarriers", Ptr
                "bufferMemoryBarrierCount", U32
                "pBufferMemoryBarriers", Ptr
                "imageMemoryBarrierCount", U32
                "pImageMemoryBarriers", Ptr
            ]
        )
        "VkSemaphoreTypeCreateInfo", Struct(hdr @ [ "semaphoreType", U32; "initialValue", U64 ])
        "VkSemaphoreCreateInfo", Struct(hdr @ [ "flags", U32 ])
        "VkSemaphoreSubmitInfo",
        Struct(hdr @ [ "semaphore", U64; "value", U64; "stageMask", U64; "deviceIndex", U32 ])
        "VkCommandBufferSubmitInfo", Struct(hdr @ [ "commandBuffer", Ptr; "deviceMask", U32 ])
        "VkSubmitInfo2",
        Struct(
            hdr
            @ [
                "flags", U32
                "waitSemaphoreInfoCount", U32
                "pWaitSemaphoreInfos", Ptr
                "commandBufferInfoCount", U32
                "pCommandBufferInfos", Ptr
                "signalSemaphoreInfoCount", U32
                "pSignalSemaphoreInfos", Ptr
            ]
        )
        "VkSemaphoreWaitInfo",
        Struct(hdr @ [ "flags", U32; "semaphoreCount", U32; "pSemaphores", Ptr; "pValues", Ptr ])
        "VkQueryPoolCreateInfo",
        Struct(hdr @ [ "flags", U32; "queryType", U32; "queryCount", U32; "pipelineStatistics", U32 ])
        // ---- VK_EXT_descriptor_heap ----
        "VkDeviceAddressRangeEXT", Struct [ "address", U64; "size", U64 ] // typedef of VkDeviceAddressRangeKHR
        "VkHostAddressRangeEXT", Struct [ "address", Ptr; "size", Ptr ]
        "VkHostAddressRangeConstEXT", Struct [ "address", Ptr; "size", Ptr ]
        "VkResourceDescriptorDataEXT",
        Union [ "pImage", Ptr; "pTexelBuffer", Ptr; "pAddressRange", Ptr; "pTensorARM", Ptr ]
        "VkResourceDescriptorInfoEXT", Struct(hdr @ [ "type", U32; "data", S "VkResourceDescriptorDataEXT" ])
        "VkBindHeapInfoEXT",
        Struct(
            hdr
            @ [
                "heapRange", S "VkDeviceAddressRangeEXT"
                "reservedRangeOffset", U64
                "reservedRangeSize", U64
            ]
        )
        "VkPushDataInfoEXT", Struct(hdr @ [ "offset", U32; "data", S "VkHostAddressRangeConstEXT" ])
        "VkDescriptorMappingSourceConstantOffsetEXT",
        Struct
            [
                "heapOffset", U32
                "heapArrayStride", U32
                "pEmbeddedSampler", Ptr
                "samplerHeapOffset", U32
                "samplerHeapArrayStride", U32
            ]
        "VkDescriptorMappingSourcePushIndexEXT",
        Struct
            [
                "heapOffset", U32
                "pushOffset", U32
                "heapIndexStride", U32
                "heapArrayStride", U32
                "pEmbeddedSampler", Ptr
                "useCombinedImageSamplerIndex", U32
                "samplerHeapOffset", U32
                "samplerPushOffset", U32
                "samplerHeapIndexStride", U32
                "samplerHeapArrayStride", U32
            ]
        "VkDescriptorMappingSourceIndirectIndexEXT",
        Struct
            [
                "heapOffset", U32
                "pushOffset", U32
                "addressOffset", U32
                "heapIndexStride", U32
                "heapArrayStride", U32
                "pEmbeddedSampler", Ptr
                "useCombinedImageSamplerIndex", U32
                "samplerHeapOffset", U32
                "samplerPushOffset", U32
                "samplerAddressOffset", U32
                "samplerHeapIndexStride", U32
                "samplerHeapArrayStride", U32
            ]
        "VkDescriptorMappingSourceIndirectIndexArrayEXT",
        Struct
            [
                "heapOffset", U32
                "pushOffset", U32
                "addressOffset", U32
                "heapIndexStride", U32
                "pEmbeddedSampler", Ptr
                "useCombinedImageSamplerIndex", U32
                "samplerHeapOffset", U32
                "samplerPushOffset", U32
                "samplerAddressOffset", U32
                "samplerHeapIndexStride", U32
            ]
        "VkDescriptorMappingSourceHeapDataEXT", Struct [ "heapOffset", U32; "pushOffset", U32 ]
        "VkDescriptorMappingSourceIndirectAddressEXT", Struct [ "pushOffset", U32; "addressOffset", U32 ]
        "VkDescriptorMappingSourceShaderRecordIndexEXT",
        Struct
            [
                "heapOffset", U32
                "shaderRecordOffset", U32
                "heapIndexStride", U32
                "heapArrayStride", U32
                "pEmbeddedSampler", Ptr
                "useCombinedImageSamplerIndex", U32
                "samplerHeapOffset", U32
                "samplerShaderRecordOffset", U32
                "samplerHeapIndexStride", U32
                "samplerHeapArrayStride", U32
            ]
        "VkDescriptorMappingSourceDataEXT",
        Union
            [
                "constantOffset", S "VkDescriptorMappingSourceConstantOffsetEXT"
                "pushIndex", S "VkDescriptorMappingSourcePushIndexEXT"
                "indirectIndex", S "VkDescriptorMappingSourceIndirectIndexEXT"
                "indirectIndexArray", S "VkDescriptorMappingSourceIndirectIndexArrayEXT"
                "heapData", S "VkDescriptorMappingSourceHeapDataEXT"
                "pushDataOffset", U32
                "pushAddressOffset", U32
                "indirectAddress", S "VkDescriptorMappingSourceIndirectAddressEXT"
                "shaderRecordIndex", S "VkDescriptorMappingSourceShaderRecordIndexEXT"
                "shaderRecordDataOffset", U32
                "shaderRecordAddressOffset", U32
            ]
        "VkDescriptorSetAndBindingMappingEXT",
        Struct(
            hdr
            @ [
                "descriptorSet", U32
                "firstBinding", U32
                "bindingCount", U32
                "resourceMask", U32
                "source", U32
                "sourceData", S "VkDescriptorMappingSourceDataEXT"
            ]
        )
        "VkShaderDescriptorSetAndBindingMappingInfoEXT", Struct(hdr @ [ "mappingCount", U32; "pMappings", Ptr ])
    ]

let private table = dict decls

/// (size, alignment, member offsets) of a C type on x64.
let rec layoutOf (t: CType) : int * int * (string * int) list =
    match t with
    | U8 -> 1, 1, []
    | U16 -> 2, 2, []
    | U32 -> 4, 4, []
    | U64
    | Ptr -> 8, 8, []
    | Arr(e, n) ->
        let s, a, _ = layoutOf e
        s * n, a, []
    | S name ->
        match table.TryGetValue name with
        | false, _ -> failwithf "no C declaration for %s" name
        | true, Struct members ->
            let mutable off = 0
            let mutable align = 1

            let offsets =
                [
                    for (m, mt) in members do
                        let s, a, _ = layoutOf mt
                        off <- (off + a - 1) / a * a
                        yield m, off
                        off <- off + s
                        align <- max align a
                ]

            (off + align - 1) / align * align, align, offsets
        | true, Union members ->
            let parts = members |> List.map (fun (m, mt) -> m, layoutOf mt)
            let align = parts |> List.map (fun (_, (_, a, _)) -> a) |> List.max
            let size = parts |> List.map (fun (_, (s, _, _)) -> s) |> List.max
            (size + align - 1) / align * align, align, members |> List.map (fun (m, _) -> m, 0)

/// The C size of a declared type.
let sizeOfDecl name =
    let s, _, _ = layoutOf (S name)
    s

/// Mismatches between the F# struct `t` and the C declaration `name`; empty when they agree. For a union, the F#
/// type may declare a subset of the members (all at offset 0) but must have the full size.
let check (t: Type) (name: string) : string list =
    let size, align, offsets = layoutOf (S name)
    let isUnion =
        match table[name] with
        | Union _ -> true
        | Struct _ -> false

    [
        let managed = RuntimeHelpers.SizeOf t.TypeHandle
        let marshalled = Marshal.SizeOf t

        if managed <> size || marshalled <> size then
            sprintf "%s: sizeof managed %d, marshalled %d, C %d" name managed marshalled size

        match
            (try
                Ok(GCHandle.Alloc(Activator.CreateInstance t, GCHandleType.Pinned))
             with e ->
                 Error e.Message)
        with
        | Ok h -> h.Free()
        | Error msg -> sprintf "%s: not blittable (%s)" name msg

        let fields =
            t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
            |> Array.map (fun f -> f.Name, int (Marshal.OffsetOf(t, f.Name)))
            |> Array.sortBy snd
            |> List.ofArray

        if isUnion then
            for (f, off) in fields do
                match offsets |> List.tryFind (fun (m, _) -> m = f) with
                | None -> sprintf "%s: union member %s is not in the C declaration" name f
                | Some _ when off <> 0 -> sprintf "%s.%s: union member at offset %d" name f off
                | Some _ -> ()
        elif List.map fst fields <> List.map fst offsets then
            sprintf "%s: members %A, C %A" name (List.map fst fields) (List.map fst offsets)
        else
            for ((f, off), (_, expected)) in List.zip fields offsets do
                if off <> expected then
                    sprintf "%s.%s: offset %d, C %d" name f off expected
    // Alignment is covered by the sizes and by the offsets of nested structs in their parents.
    ]

/// The alignment of a declared type (for the generated C cross-check).
let alignOfDecl name =
    let _, a, _ = layoutOf (S name)
    a

/// Known answers for the calculator, computed by hand from the x64 C rules (and cross-checked by cl.exe, see
/// `cSource`).
let selfTest () : string list =
    [
        for (name, size) in
            [
                "VkPhysicalDeviceMemoryProperties", 520
                "VkQueueFamilyProperties", 24
                "VkDescriptorMappingSourceDataEXT", 56
                "VkDescriptorSetAndBindingMappingEXT", 96
                "VkBindHeapInfoEXT", 48
                "VkPushDataInfoEXT", 40
                "VkResourceDescriptorInfoEXT", 32
                "VkComputePipelineCreateInfo", 96
            ] do
            let s = sizeOfDecl name

            if s <> size then
                yield sprintf "calculator: %s is %d bytes, expected %d" name s size
    ]

/// A C translation unit that re-declares every type in `decls` and static_asserts the calculator's sizes, alignments
/// and offsets, so a C compiler (cl.exe) checks the calculator itself.
let cSource () : string =
    let rec cty (t: CType) =
        match t with
        | U8 -> "unsigned char"
        | U16 -> "unsigned short"
        | U32 -> "unsigned int"
        | U64 -> "unsigned long long"
        | Ptr -> "void*"
        | S n -> n
        | Arr _ -> failwith "arrays are written by the member printer"

    let sb = Text.StringBuilder()
    let line (s: string) = sb.Append(s).Append('\n') |> ignore
    line "/* Generated by spikes/vk-heap (Layout.cSource). Compile: cl /nologo /c /W4 /WX layout_check.c */"
    line "#include <stddef.h>"

    for (name, d) in decls do
        let kw, members =
            match d with
            | Struct ms -> "struct", ms
            | Union ms -> "union", ms

        line (sprintf "typedef %s %s {" kw name)

        for (m, t) in members do
            match t with
            | Arr(e, n) -> line (sprintf "    %s %s[%d];" (cty e) m n)
            | t -> line (sprintf "    %s %s;" (cty t) m)

        line (sprintf "} %s;" name)
        let size, align, offsets = layoutOf (S name)
        line (sprintf "_Static_assert(sizeof(%s) == %d, \"%s size\");" name size name)
        line (sprintf "_Static_assert(_Alignof(%s) == %d, \"%s align\");" name align name)

        for (m, off) in offsets do
            line (sprintf "_Static_assert(offsetof(%s, %s) == %d, \"%s.%s\");" name m off name m)

    sb.ToString()
