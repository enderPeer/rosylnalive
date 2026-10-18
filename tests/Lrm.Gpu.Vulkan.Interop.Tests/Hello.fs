/// Hello-compute through the generated binding alone: VulkanHeapRecipe sections 2 to 6 on the recipe's device, with
/// one args_echo dispatch whose ArgBlock is KAT-4. It exercises buffers with usage flags 2 and device addresses,
/// memory types chosen by flags, the descriptor heap (vkWriteResourceDescriptorsEXT, vkCmdBindResourceHeapEXT,
/// vkCmdPushDataEXT), a heap pipeline with the set/binding mapping, sync2 barriers, a timeline semaphore and
/// timestamps. The output region is poisoned before each run (recipe section 8), so a run that writes nothing fails.
/// This is the interop's smoke test, not a backend: V1 owns the backend.
module Lrm.Gpu.Vulkan.Interop.Tests.Hello

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open System.Security.Cryptography
open System.Text
open Microsoft.FSharp.NativeInterop
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop

/// KAT-4 of VulkanHeapRecipe: args_echo writes these 32 ArgBlock words, in order, to its output. Word 0 is
/// lrm_groups_x = 1 and word 1 the output token 3.
let kat4: uint32[] =
    [|
        0x00000001u
        0x00000003u
        0xffffffffu
        0x7fc00001u
        0x80000000u
        0x00000001u
        0xb519c20eu
        0x7fd49868u
        0x12508679u
        0x4c777b2cu
        0x3775418cu
        0x6b0e50d0u
        0x34a1269au
        0x7f79bf4au
        0x9a8da36du
        0x2f6087d0u
        0x33f8d54au
        0x40511b2bu
        0xd98077b5u
        0xac105150u
        0xdfd2c9fdu
        0xd5b8cf0eu
        0x12ee2896u
        0x43de8e57u
        0x28dca649u
        0x65c6007eu
        0x610e2eeau
        0xae4d91c4u
        0xd2a32be2u
        0x86dc79beu
        0xc1d53a9cu
        0x97b2b7a7u
    |]

/// The SHA-256 of KAT-4's output bytes (VulkanHeapRecipe section 9).
[<Literal>]
let Kat4Sha256 = "74f5e898a05c7e2063639a11cbdaed2cd75d077e98c5c27296542f577afc3531"

/// The output token of KAT-4.
[<Literal>]
let OutputToken = 3

/// Bytes per region of the one page buffer: the table (slot 0), the output (slot 1) and the dummy page (slots 2..63).
[<Literal>]
let RegionBytes = 65536UL

type Buffer =
    {
        Handle: VkBuffer
        Memory: VkDeviceMemory
        Size: uint64
        Address: uint64
        Mapped: nativeint
        MemoryType: int
    }

type Result =
    {
        Output: byte[]
        /// Output words still equal to their poison.
        PoisonLeft: int
        GpuStartNs: int64
        GpuNs: int64
        MemoryTypes: int list
        HeapAddress: uint64
    }

let private noMemoryFlags: VkMemoryPropertyFlags = LanguagePrimitives.EnumOfValue 0u

/// The first memory type among `bits` that has every `required` flag and none of `forbidden` (by flags, never by
/// index: VulkanHeapRecipe section 2).
let memoryType (props: VkPhysicalDeviceMemoryProperties) (bits: uint32) required forbidden =
    let types = props.memoryTypes.ToArray()

    [ 0 .. int props.memoryTypeCount - 1 ]
    |> List.tryFind (fun i ->
        (bits &&& (1u <<< i)) <> 0u
        && (types[i].propertyFlags &&& required) = required
        && (types[i].propertyFlags &&& forbidden) = noMemoryFlags)

let createBuffer
    (d: Device)
    (props: VkPhysicalDeviceMemoryProperties)
    (size: uint64)
    (usage: VkBufferUsageFlags2)
    (required: VkMemoryPropertyFlags)
    (forbidden: VkMemoryPropertyFlags)
    (map: bool)
    : Buffer =
    let c = d.Commands
    let mutable usage2 = VkBufferUsageFlags2CreateInfo.Create()
    usage2.usage <- usage
    let mutable ci = VkBufferCreateInfo.Create()
    ci.pNext <- NativePtr.toNativeInt &&usage2 // VkBufferCreateInfo.usage is ignored when flags 2 are chained
    ci.size <- size
    ci.sharingMode <- VkSharingMode.VK_SHARING_MODE_EXCLUSIVE
    let mutable buffer = VkBuffer.Null
    c.CreateBuffer(d.Handle, &ci, &buffer)
    let mutable req = VkMemoryRequirements()
    c.GetBufferMemoryRequirements(d.Handle, buffer, &req)

    let index =
        match memoryType props req.memoryTypeBits required forbidden with
        | Some i -> i
        | None -> failwithf "no memory type with %A and without %A" required forbidden

    let deviceAddress =
        usage.HasFlag VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT

    let mutable allocFlags = VkMemoryAllocateFlagsInfo.Create()
    allocFlags.flags <- VkMemoryAllocateFlagBits.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    let mutable ai = VkMemoryAllocateInfo.Create()

    if deviceAddress then
        ai.pNext <- NativePtr.toNativeInt &&allocFlags

    ai.allocationSize <- req.size
    ai.memoryTypeIndex <- uint32 index
    let mutable memory = VkDeviceMemory.Null
    c.AllocateMemory(d.Handle, &ai, &memory)
    c.BindBufferMemory(d.Handle, buffer, memory, 0UL)

    let address =
        if deviceAddress then
            let mutable info = VkBufferDeviceAddressInfo.Create()
            info.buffer <- buffer
            c.GetBufferDeviceAddress(d.Handle, &info)
        else
            0UL

    let mapped =
        if map then
            let mutable p = 0n
            c.MapMemory(d.Handle, memory, 0UL, VkConst.VK_WHOLE_SIZE, 0u, &p)
            p
        else
            0n

    {
        Handle = buffer
        Memory = memory
        Size = size
        Address = address
        Mapped = mapped
        MemoryType = index
    }

let destroyBuffer (d: Device) (b: Buffer) =
    d.Commands.DestroyBuffer(d.Handle, b.Handle)
    d.Commands.FreeMemory(d.Handle, b.Memory)

/// The poison of one run: signalling-NaN patterns from SHA-256("vulkan-interop|args_echo|<run>"), never equal to
/// the expected word.
let poison (run: int) : uint32[] =
    let h =
        SHA256.HashData(Encoding.UTF8.GetBytes(sprintf "vulkan-interop|args_echo|%d" run))

    Array.init 32 (fun i ->
        let bits = BitConverter.ToUInt32(h, (i * 4) % 32) ^^^ (uint32 i * 0x9E3779B9u)

        let p = 0x7F800001u ||| (bits &&& 0x003FFFFEu)
        if p = kat4[i] then p ^^^ 0x2u else p)

let private memoryBarrier (src: VkPipelineStageFlags2) srcAccess (dst: VkPipelineStageFlags2) dstAccess =
    let mutable b = VkMemoryBarrier2.Create()
    b.srcStageMask <- src
    b.srcAccessMask <- srcAccess
    b.dstStageMask <- dst
    b.dstAccessMask <- dstAccess
    b

let private barrier (d: Device) (cmd: VkCommandBuffer) (b: VkMemoryBarrier2) =
    let mutable mb = b
    let mutable dep = VkDependencyInfo.Create()
    dep.memoryBarrierCount <- 1u
    dep.pMemoryBarriers <- NativePtr.toNativeInt &&mb
    d.Commands.CmdPipelineBarrier2(cmd, &dep)

/// One run on `d`: creates every object, records, submits, waits, reads back and destroys everything. `dispatch`
/// false is the negative control (recipe section 8): the same job without its dispatch.
let run
    (d: Device)
    (heap: DeviceCheck.DescriptorHeapLimits)
    (timestampValidBits: uint32)
    (period: float32)
    (runIndex: int)
    (dispatch: bool)
    : Result =
    let c = d.Commands
    let dev = d.Handle
    let props = d.Instance.MemoryProperties d.PhysicalDevice
    let g = DeviceCheck.geometry heap
    let deviceLocal = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    let hostVisible = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    let hostCoherent = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    let hostCached = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_CACHED_BIT

    // ---- buffers (recipe section 2) ----
    let page =
        createBuffer
            d
            props
            (3UL * RegionBytes)
            (VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
             ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
             ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
             ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_DST_BIT)
            deviceLocal
            hostVisible
            false

    let heapBuffer =
        createBuffer
            d
            props
            (g.HeapRangeSize + heap.ResourceHeapAlignment)
            (VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT
             ||| VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT)
            (deviceLocal ||| hostVisible ||| hostCoherent)
            noMemoryFlags
            true

    let upload =
        createBuffer
            d
            props
            4096UL
            VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
            (hostVisible ||| hostCoherent)
            hostCached
            true

    let readback =
        createBuffer
            d
            props
            4096UL
            VkBufferUsageFlagBits2.VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
            (hostVisible ||| hostCoherent)
            noMemoryFlags
            true

    let objects = ResizeArray<unit -> unit>()

    try
        // ---- the heap (sections 3 and 4): 64 slots written once, in one call ----
        let align = heap.ResourceHeapAlignment
        let heapAddress = (heapBuffer.Address + align - 1UL) / align * align
        let hostHeap = heapBuffer.Mapped + nativeint (heapAddress - heapBuffer.Address)

        let ranges =
            Array.init DeviceCheck.HeapSlots (fun i ->
                let mutable r = VkDeviceAddressRangeEXT()

                r.address <-
                    page.Address
                    + (match i with
                       | 0 -> 0UL
                       | 1 -> RegionBytes
                       | _ -> 2UL * RegionBytes)

                r.size <- RegionBytes
                r)

        use pRanges = fixed ranges

        let infos =
            Array.init DeviceCheck.HeapSlots (fun i ->
                let mutable x = VkResourceDescriptorInfoEXT.Create()
                x.``type`` <- VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                x.data.pAddressRange <- NativePtr.toNativeInt (NativePtr.add pRanges i)
                x)

        let destinations =
            Array.init DeviceCheck.HeapSlots (fun i ->
                let mutable h = VkHostAddressRangeEXT()
                h.address <- hostHeap + nativeint (uint64 i * g.SlotBytes)
                h.size <- unativeint g.SlotBytes
                h)

        c.WriteResourceDescriptorsEXT(dev, ReadOnlySpan infos, ReadOnlySpan destinations)

        // ---- staging: the table rows (token 3 is the output) and the output's poison ----
        let row = TensorRow.create 1 0L DType.U32 [ 32L ] TensorFlags.None 0u

        let tableBytes = Array.zeroCreate<byte>((OutputToken + 1) * 64)
        row.WriteTo(Span(tableBytes, OutputToken * 64, 64))
        let poisonWords = poison runIndex
        Marshal.Copy(tableBytes, 0, upload.Mapped, tableBytes.Length)
        Marshal.Copy(Common.bytesOf poisonWords, 0, upload.Mapped + nativeint tableBytes.Length, 128)

        // ---- the pipeline (section 5) ----
        let blob =
            match ShaderSet.tryBlob "args_echo" ShaderTarget.Spirv (Common.shaders ()) with
            | Some b -> b
            | None -> failwith "Lrm.Shaders has no SPIR-V args_echo"

        let code = blob.Bytes
        use pCode = fixed code
        let mutable smci = VkShaderModuleCreateInfo.Create()
        smci.codeSize <- unativeint code.Length
        smci.pCode <- NativePtr.toNativeInt pCode
        let mutable shader = VkShaderModule.Null
        c.CreateShaderModule(dev, &smci, &shader)
        objects.Add(fun () -> c.DestroyShaderModule(dev, shader))

        let mutable mapping = VkDescriptorSetAndBindingMappingEXT.Create()
        mapping.descriptorSet <- 0u
        mapping.firstBinding <- 0u
        mapping.bindingCount <- 1u

        mapping.resourceMask <-
            VkSpirvResourceTypeFlagBitsEXT.VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT
            ||| VkSpirvResourceTypeFlagBitsEXT.VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT

        mapping.source <- VkDescriptorMappingSourceEXT.VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT
        mapping.sourceData.constantOffset.heapOffset <- 0u
        mapping.sourceData.constantOffset.heapArrayStride <- uint32 g.SlotBytes
        let mutable mappingInfo = VkShaderDescriptorSetAndBindingMappingInfoEXT.Create()
        mappingInfo.mappingCount <- 1u
        mappingInfo.pMappings <- NativePtr.toNativeInt &&mapping
        let entry = VkText.utf8z "main"
        use pEntry = fixed entry
        let mutable flags2 = VkPipelineCreateFlags2CreateInfo.Create()
        flags2.flags <- VkPipelineCreateFlagBits2.VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT
        let mutable stage = VkPipelineShaderStageCreateInfo.Create()
        stage.pNext <- NativePtr.toNativeInt &&mappingInfo
        stage.stage <- VkShaderStageFlagBits.VK_SHADER_STAGE_COMPUTE_BIT
        stage.``module`` <- shader
        stage.pName <- NativePtr.toNativeInt pEntry
        let mutable cpci = VkComputePipelineCreateInfo.Create()
        cpci.pNext <- NativePtr.toNativeInt &&flags2
        cpci.stage <- stage
        cpci.layout <- VkPipelineLayout.Null
        cpci.basePipelineIndex <- -1
        let mutable pipeline = VkPipeline.Null

        c.CreateComputePipelines(
            dev,
            VkPipelineCache.Null,
            MemoryMarshal.CreateReadOnlySpan(&cpci, 1),
            MemoryMarshal.CreateSpan(&pipeline, 1)
        )
        |> ignore

        objects.Add(fun () -> c.DestroyPipeline(dev, pipeline))

        // ---- command pool, command buffer, timestamps, timeline semaphore ----
        let mutable poolInfo = VkCommandPoolCreateInfo.Create()
        poolInfo.flags <- VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
        poolInfo.queueFamilyIndex <- d.QueueFamily
        let mutable pool = VkCommandPool.Null
        c.CreateCommandPool(dev, &poolInfo, &pool)
        // The pool goes first at teardown, before the heap buffer (recipe section 4).
        objects.Insert(0, (fun () -> c.DestroyCommandPool(dev, pool)))
        let mutable cbai = VkCommandBufferAllocateInfo.Create()
        cbai.commandPool <- pool
        cbai.level <- VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY
        cbai.commandBufferCount <- 1u
        let mutable cmd = VkCommandBuffer.Null
        c.AllocateCommandBuffers(dev, &cbai, NativePtr.toNativeInt &&cmd)
        let mutable qpci = VkQueryPoolCreateInfo.Create()
        qpci.queryType <- VkQueryType.VK_QUERY_TYPE_TIMESTAMP
        qpci.queryCount <- 2u
        let mutable queries = VkQueryPool.Null
        c.CreateQueryPool(dev, &qpci, &queries)
        objects.Add(fun () -> c.DestroyQueryPool(dev, queries))
        let mutable timeline = VkSemaphoreTypeCreateInfo.Create()
        timeline.semaphoreType <- VkSemaphoreType.VK_SEMAPHORE_TYPE_TIMELINE
        timeline.initialValue <- 0UL
        let mutable sci = VkSemaphoreCreateInfo.Create()
        sci.pNext <- NativePtr.toNativeInt &&timeline
        let mutable semaphore = VkSemaphore.Null
        c.CreateSemaphore(dev, &sci, &semaphore)
        objects.Add(fun () -> c.DestroySemaphore(dev, semaphore))

        // ---- record (section 6) ----
        let mutable begin' = VkCommandBufferBeginInfo.Create()
        begin'.flags <- VkCommandBufferUsageFlagBits.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        c.BeginCommandBuffer(cmd, &begin')
        c.CmdResetQueryPool(cmd, queries, 0u, 2u)
        let mutable tableCopy = VkBufferCopy()
        tableCopy.size <- uint64 tableBytes.Length
        let mutable poisonCopy = VkBufferCopy()
        poisonCopy.srcOffset <- uint64 tableBytes.Length
        poisonCopy.dstOffset <- RegionBytes
        poisonCopy.size <- 128UL
        c.CmdCopyBuffer(cmd, upload.Handle, page.Handle, ReadOnlySpan [| tableCopy; poisonCopy |])

        barrier
            d
            cmd
            (memoryBarrier
                VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COPY_BIT
                VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_WRITE_BIT
                VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                (VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                 ||| VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT))

        let mutable bind = VkBindHeapInfoEXT.Create()
        bind.heapRange.address <- heapAddress
        bind.heapRange.size <- g.HeapRangeSize
        bind.reservedRangeOffset <- g.ReservedRangeOffset
        bind.reservedRangeSize <- g.ReservedRangeSize
        c.CmdBindResourceHeapEXT(cmd, &bind)
        c.CmdBindPipeline(cmd, VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        use pArgs = fixed kat4
        let mutable push = VkPushDataInfoEXT.Create()
        push.offset <- 0u
        push.data.address <- NativePtr.toNativeInt pArgs
        push.data.size <- unativeint 128
        c.CmdPushDataEXT(cmd, &push)
        c.CmdWriteTimestamp2(cmd, VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, 0u)

        if dispatch then
            c.CmdDispatch(cmd, 1u, 1u, 1u)

        c.CmdWriteTimestamp2(cmd, VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, 1u)

        barrier
            d
            cmd
            (memoryBarrier
                VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                (VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                 ||| VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COPY_BIT)
                (VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                 ||| VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                 ||| VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_READ_BIT))

        let mutable outCopy = VkBufferCopy()
        outCopy.srcOffset <- RegionBytes
        outCopy.size <- 128UL
        c.CmdCopyBuffer(cmd, page.Handle, readback.Handle, MemoryMarshal.CreateReadOnlySpan(&outCopy, 1))

        barrier
            d
            cmd
            (memoryBarrier
                VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COPY_BIT
                VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_WRITE_BIT
                VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_HOST_BIT
                VkAccessFlagBits2.VK_ACCESS_2_HOST_READ_BIT)

        c.EndCommandBuffer(cmd)

        // ---- submit and wait ----
        let mutable cbsi = VkCommandBufferSubmitInfo.Create()
        cbsi.commandBuffer <- cmd
        let mutable signal = VkSemaphoreSubmitInfo.Create()
        signal.semaphore <- semaphore
        signal.value <- 1UL
        signal.stageMask <- VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        let mutable submit = VkSubmitInfo2.Create()
        submit.commandBufferInfoCount <- 1u
        submit.pCommandBufferInfos <- NativePtr.toNativeInt &&cbsi
        submit.signalSemaphoreInfoCount <- 1u
        submit.pSignalSemaphoreInfos <- NativePtr.toNativeInt &&signal
        c.QueueSubmit2(d.Queue, MemoryMarshal.CreateReadOnlySpan(&submit, 1), VkFence.Null)
        let mutable value = 1UL
        let mutable wait = VkSemaphoreWaitInfo.Create()
        wait.semaphoreCount <- 1u
        wait.pSemaphores <- NativePtr.toNativeInt &&semaphore
        wait.pValues <- NativePtr.toNativeInt &&value

        match c.WaitSemaphores(dev, &wait, 10_000_000_000UL) with
        | VkResult.VK_SUCCESS -> ()
        | r -> failwithf "the submission did not complete within 10 s: %A" r

        // ---- read back ----
        let output = Array.zeroCreate<byte> 128
        Marshal.Copy(readback.Mapped, output, 0, 128)
        let words = Common.wordsOf output
        let left = Array.fold2 (fun n w p -> if w = p then n + 1 else n) 0 words poisonWords
        let stamps = Array.zeroCreate<byte> 16

        c.GetQueryPoolResults(
            dev,
            queries,
            0u,
            2u,
            Span stamps,
            8UL,
            VkQueryResultFlagBits.VK_QUERY_RESULT_64_BIT
            ||| VkQueryResultFlagBits.VK_QUERY_RESULT_WAIT_BIT
        )
        |> ignore

        let mask =
            if timestampValidBits >= 64u then
                UInt64.MaxValue
            else
                (1UL <<< int timestampValidBits) - 1UL

        let t0 = BitConverter.ToUInt64(stamps, 0) &&& mask
        let t1 = BitConverter.ToUInt64(stamps, 8) &&& mask
        let ns (t: uint64) = int64 (float t * float period)

        GpuTelemetry.bytesMoved ExecutionBackend.Vulkan "upload" (int64 (tableBytes.Length + 128))
        GpuTelemetry.bytesMoved ExecutionBackend.Vulkan "download" 128L

        {
            Output = output
            PoisonLeft = left
            GpuStartNs = ns t0
            GpuNs = ns ((t1 - t0) &&& mask)
            MemoryTypes =
                [
                    page.MemoryType
                    heapBuffer.MemoryType
                    upload.MemoryType
                    readback.MemoryType
                ]
            HeapAddress = heapAddress
        }
    finally
        // Every command buffer that bound the heap is complete (the wait) and reset (its pool goes first) before
        // the heap is destroyed (recipe section 4).
        c.DeviceWaitIdle(dev)

        for destroy in objects do
            destroy ()

        destroyBuffer d readback
        destroyBuffer d upload
        destroyBuffer d heapBuffer
        destroyBuffer d page
