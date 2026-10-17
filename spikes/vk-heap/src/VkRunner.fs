/// The Vulkan side of the spike, end to end on VK_EXT_descriptor_heap:
/// - a logical device on a compute-only queue family with descriptorHeap, bufferDeviceAddress, timelineSemaphore,
///   runtimeDescriptorArray, synchronization2 and maintenance5;
/// - arena pages as device-local buffers with device addresses;
/// - one resource heap buffer, written once with vkWriteResourceDescriptorsEXT, the reserved range at its end;
/// - pipelines created with VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT, a NULL layout and a set 0 binding 0 -> heap
///   mapping (HEAP_WITH_CONSTANT_OFFSET, heapOffset 0, heapArrayStride = bufferDescriptorSize);
/// - per dispatch: vkCmdBindResourceHeapEXT (once per command buffer), vkCmdPushDataEXT with 128 bytes at offset 0,
///   vkCmdDispatch, timestamps; completion through a timeline semaphore.
module VkHeapSpike.VkRunner

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open Lrm.Core
open Lrm.Gpu.Vulkan
open VkHeapSpike.Native
open VkHeapSpike.Vk
open VkHeapSpike.Heap

/// Where the resource heap buffer lives and how the descriptors get there.
type HeapPlacement =
    /// DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (the BAR window): vkWriteResourceDescriptorsEXT writes in place.
    | Bar
    /// HOST_VISIBLE | HOST_COHERENT system memory: written in place, read by the GPU over PCIe.
    | HostMemory
    /// DEVICE_LOCAL only: written to host memory, uploaded with vkCmdCopyBuffer and a TRANSFER_WRITE ->
    /// RESOURCE_HEAP_READ barrier.
    | CopyToDeviceLocal

let placementName =
    function
    | Bar -> "bar"
    | HostMemory -> "host"
    | CopyToDeviceLocal -> "copy"

type MemoryTypeInfo =
    {
        Index: int
        Flags: uint32
        Heap: int
        HeapBytes: uint64
        HeapFlags: uint32
    }

let memFlagNames (f: uint32) =
    [
        if f &&& Flags.MemDeviceLocal <> 0u then "DEVICE_LOCAL"
        if f &&& Flags.MemHostVisible <> 0u then "HOST_VISIBLE"
        if f &&& Flags.MemHostCoherent <> 0u then "HOST_COHERENT"
        if f &&& Flags.MemHostCached <> 0u then "HOST_CACHED"
        if f &&& ~~~0xFu <> 0u then sprintf "0x%x" (f &&& ~~~0xFu)
    ]
    |> String.concat "|"

type Buffer =
    {
        Handle: uint64
        Memory: uint64
        Bytes: uint64
        Address: uint64
        /// Host pointer of the whole allocation when mapped, else 0.
        Mapped: nativeint
        MemoryType: int
        Alignment: uint64
    }

/// What the spike observed on the device (for the ADR and telemetry).
type DeviceFacts =
    {
        Name: string
        Driver: string
        ApiVersion: string
        QueueFamily: int
        QueueFamilyFlags: uint32
        TimestampValidBits: uint32
        TimestampPeriodNs: float
        MemoryTypes: MemoryTypeInfo list
        Heap: DeviceCheck.DescriptorHeapLimits
        StorageBufferDescriptorSize: uint64
        MinStorageBufferOffsetAlignment: uint64
        MaxStorageBufferRange: uint32
        PageMemoryType: int
        UploadMemoryType: int
        ReadbackMemoryType: int
        /// Memory type of the heap buffer per placement, when the device has one.
        HeapMemoryTypes: (HeapPlacement * int) list
    }

let private loadInstanceProc<'T when 'T :> Delegate> (loader: VulkanLoader) (instance: Instance) (name: string) : 'T =
    loader.LoadInstanceProcOrFail<'T>(instance.Handle, name)

let private stagingBytes = 64UL * 1024UL * 1024UL
let private timeoutNs = 10_000_000_000UL

/// Device-wide state shared by every configuration: device, queue, pages, staging buffers, command buffer, timeline
/// semaphore and timestamp pool.
type VulkanContext() =
    let loader =
        match VulkanLoader.TryOpen() with
        | Some l -> l
        | None -> failwith "no Vulkan loader (vulkan-1.dll)"

    let instance = loader.CreateInstance("lrm-r1-vk-heap-spike", VkVersion.VK_API_VERSION_1_4)

    let physical, caps =
        instance.PhysicalDevices
        |> Array.tryPick (fun pd ->
            match (DeviceCheck.check instance pd).Verdict with
            | DeviceCheck.Accepted c -> Some(pd, c)
            | DeviceCheck.Rejected _ -> None)
        |> Option.defaultWith (fun () -> failwith "no Vulkan device passes the device check (VK_EXT_descriptor_heap)")

    let pd = physical.Handle

    let props =
        let mutable p = VkPhysicalDeviceProperties()
        instance.GetPhysicalDeviceProperties.Invoke(physical, &&p)
        p

    let queueFamilies =
        let f = loadInstanceProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties> loader instance "vkGetPhysicalDeviceQueueFamilyProperties"
        use s = new Scratch()
        let pCount = s.Alloc 4
        f.Invoke(pd, pCount, 0n)
        let n = read<uint32> pCount
        let arr = s.Alloc(int n * sizeof<VkQueueFamilyProperties>)
        f.Invoke(pd, pCount, arr)
        Array.init (int n) (readAt<VkQueueFamilyProperties> arr)

    /// A compute-only family (no graphics) with timestamps; V1 uses the same rule.
    let family =
        queueFamilies
        |> Array.tryFindIndex (fun q ->
            q.queueFlags &&& Flags.QueueCompute <> 0u
            && q.queueFlags &&& Flags.QueueGraphics = 0u
            && q.timestampValidBits > 0u)
        |> Option.defaultWith (fun () -> failwith "no compute-only queue family with timestamps")

    let memProps =
        let f = loadInstanceProc<PFN_vkGetPhysicalDeviceMemoryProperties> loader instance "vkGetPhysicalDeviceMemoryProperties"
        use s = new Scratch()
        let p = s.Alloc(sizeof<VkPhysicalDeviceMemoryProperties>)
        f.Invoke(pd, p)
        let count = read<uint32> p
        let heapCount = Marshal.ReadInt32(p, 260)

        let heaps =
            Array.init heapCount (fun h -> Marshal.ReadInt64(p, 264 + 16 * h) |> uint64, Marshal.ReadInt32(p, 272 + 16 * h) |> uint32)

        [
            for t in 0 .. int count - 1 do
                let flags = uint32 (Marshal.ReadInt32(p, 4 + 8 * t))
                let heap = Marshal.ReadInt32(p, 8 + 8 * t)

                {
                    Index = t
                    Flags = flags
                    Heap = heap
                    HeapBytes = fst heaps[heap]
                    HeapFlags = snd heaps[heap]
                }
        ]

    let descriptorSize =
        let f = loadInstanceProc<PFN_vkGetPhysicalDeviceDescriptorSizeEXT> loader instance "vkGetPhysicalDeviceDescriptorSizeEXT"
        f.Invoke(pd, Enum.DescriptorTypeStorageBuffer)

    // ---- logical device ----
    let device =
        use s = new Scratch()
        // Check support first, with the same chain the device is created with.
        let mutable v12 = VkPhysicalDeviceVulkan12Features()
        v12.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        let mutable v13 = VkPhysicalDeviceVulkan13Features()
        v13.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
        let mutable m5 = VkPhysicalDeviceMaintenance5Features()
        m5.sType <- SType.PhysicalDeviceMaintenance5Features
        let mutable hf = VkPhysicalDeviceDescriptorHeapFeaturesEXT()
        hf.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT
        v12.pNext <- NativePtr.toNativeInt &&v13
        v13.pNext <- NativePtr.toNativeInt &&m5
        m5.pNext <- NativePtr.toNativeInt &&hf
        let mutable f2 = VkPhysicalDeviceFeatures2()
        f2.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
        f2.pNext <- NativePtr.toNativeInt &&v12
        instance.GetPhysicalDeviceFeatures2.Invoke(physical, &&f2)

        let missing =
            [
                if v12.bufferDeviceAddress = 0u then "bufferDeviceAddress"
                if v12.timelineSemaphore = 0u then "timelineSemaphore"
                if v12.runtimeDescriptorArray = 0u then "runtimeDescriptorArray"
                if v13.synchronization2 = 0u then "synchronization2"
                if m5.maintenance5 = 0u then "maintenance5"
                if hf.descriptorHeap = 0u then "descriptorHeap"
            ]

        if not missing.IsEmpty then
            failwithf "device lacks features: %s" (String.concat ", " missing)

        // Enable exactly what the recipe needs.
        let mutable e12 = VkPhysicalDeviceVulkan12Features()
        e12.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        e12.bufferDeviceAddress <- 1u
        e12.timelineSemaphore <- 1u
        e12.runtimeDescriptorArray <- 1u
        let mutable e13 = VkPhysicalDeviceVulkan13Features()
        e13.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
        e13.synchronization2 <- 1u
        let mutable em5 = VkPhysicalDeviceMaintenance5Features()
        em5.sType <- SType.PhysicalDeviceMaintenance5Features
        em5.maintenance5 <- 1u
        let mutable ehf = VkPhysicalDeviceDescriptorHeapFeaturesEXT()
        ehf.sType <- VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT
        ehf.descriptorHeap <- 1u
        e12.pNext <- NativePtr.toNativeInt &&e13
        e13.pNext <- NativePtr.toNativeInt &&em5
        em5.pNext <- NativePtr.toNativeInt &&ehf

        let prio = s.Struct 1.0f
        let mutable q = VkDeviceQueueCreateInfo()
        q.sType <- SType.DeviceQueueCreateInfo
        q.queueFamilyIndex <- uint32 family
        q.queueCount <- 1u
        q.pQueuePriorities <- prio
        let exts = [ VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME ]
        let mutable ci = VkDeviceCreateInfo()
        ci.sType <- SType.DeviceCreateInfo
        ci.pNext <- NativePtr.toNativeInt &&e12
        ci.queueCreateInfoCount <- 1u
        ci.pQueueCreateInfos <- s.Struct q
        ci.enabledExtensionCount <- uint32 exts.Length
        ci.ppEnabledExtensionNames <- s.Utf8Array exts
        let out = s.Alloc 8
        ok "vkCreateDevice" ((loadInstanceProc<PFN_vkCreateDevice> loader instance "vkCreateDevice").Invoke(pd, s.Struct ci, 0n, out))
        read<nativeint> out

    let fns = DeviceFns(instance, device)

    let queue =
        use s = new Scratch()
        let out = s.Alloc 8
        fns.GetDeviceQueue.Invoke(device, uint32 family, 0u, out)
        read<nativeint> out

    let pickType (bits: uint32) (want: uint32) (avoid: uint32) =
        memProps
        |> List.filter (fun m ->
            (bits >>> m.Index) &&& 1u = 1u
            && m.Flags &&& want = want
            && m.Flags &&& avoid = 0u)
        |> List.tryHead

    let buffers = ResizeArray<Buffer>()

    let createBuffer (bytes: uint64) (usage: uint32) (want: uint32) (avoid: uint32) (map: bool) : Buffer option =
        use s = new Scratch()
        let mutable bci = VkBufferCreateInfo()
        bci.sType <- SType.BufferCreateInfo
        bci.size <- bytes
        bci.usage <- usage
        bci.sharingMode <- Enum.SharingExclusive
        let out = s.Alloc 8
        ok "vkCreateBuffer" (fns.CreateBuffer.Invoke(device, s.Struct bci, 0n, out))
        let handle = read<uint64> out
        let req = s.Alloc(sizeof<VkMemoryRequirements>)
        fns.GetBufferMemoryRequirements.Invoke(device, handle, req)
        let r = read<VkMemoryRequirements> req

        match pickType r.memoryTypeBits want avoid with
        | None ->
            fns.DestroyBuffer.Invoke(device, handle, 0n)
            None
        | Some mt ->
            let deviceAddress = usage &&& Flags.BufShaderDeviceAddress <> 0u
            let mutable flagsInfo = VkMemoryAllocateFlagsInfo()
            flagsInfo.sType <- SType.MemoryAllocateFlagsInfo
            flagsInfo.flags <- Flags.AllocDeviceAddress
            let mutable ai = VkMemoryAllocateInfo()
            ai.sType <- SType.MemoryAllocateInfo
            ai.pNext <- (if deviceAddress then s.Struct flagsInfo else 0n)
            ai.allocationSize <- r.size
            ai.memoryTypeIndex <- uint32 mt.Index
            ok "vkAllocateMemory" (fns.AllocateMemory.Invoke(device, s.Struct ai, 0n, out))
            let memory = read<uint64> out
            ok "vkBindBufferMemory" (fns.BindBufferMemory.Invoke(device, handle, memory, 0UL))

            let address =
                if deviceAddress then
                    let mutable bi = VkBufferDeviceAddressInfo()
                    bi.sType <- SType.BufferDeviceAddressInfo
                    bi.buffer <- handle
                    fns.GetBufferDeviceAddress.Invoke(device, s.Struct bi)
                else
                    0UL

            let mapped =
                if map then
                    ok "vkMapMemory" (fns.MapMemory.Invoke(device, memory, 0UL, VK_WHOLE_SIZE, 0u, out))
                    read<nativeint> out
                else
                    0n

            let b =
                {
                    Handle = handle
                    Memory = memory
                    Bytes = bytes
                    Address = address
                    Mapped = mapped
                    MemoryType = mt.Index
                    Alignment = r.alignment
                }

            buffers.Add b
            Some b

    let destroyBuffer (b: Buffer) =
        fns.DestroyBuffer.Invoke(device, b.Handle, 0n)
        fns.FreeMemory.Invoke(device, b.Memory, 0n)
        buffers.Remove b |> ignore

    let need what =
        function
        | Some b -> b
        | None -> failwithf "no memory type for %s" what

    let pageUsage =
        Flags.BufStorage ||| Flags.BufShaderDeviceAddress ||| Flags.BufTransferSrc ||| Flags.BufTransferDst

    /// slot -> buffer; slots without their own page share the dummy page.
    let pages =
        let dummy =
            createBuffer (uint64 DummyBytes) pageUsage Flags.MemDeviceLocal Flags.MemHostVisible false
            |> need "pages"

        Array.init SlotCount (fun slot ->
            match pageBytes slot with
            | Some bytes ->
                createBuffer (uint64 bytes) pageUsage Flags.MemDeviceLocal Flags.MemHostVisible false
                |> need "pages"
            | None -> dummy)

    let upload =
        createBuffer
            stagingBytes
            Flags.BufTransferSrc
            (Flags.MemHostVisible ||| Flags.MemHostCoherent)
            (Flags.MemDeviceLocal ||| Flags.MemHostCached)
            true
        |> Option.orElseWith (fun () ->
            createBuffer stagingBytes Flags.BufTransferSrc (Flags.MemHostVisible ||| Flags.MemHostCoherent) 0u true)
        |> need "upload staging"

    let readback =
        createBuffer
            stagingBytes
            Flags.BufTransferDst
            (Flags.MemHostVisible ||| Flags.MemHostCoherent ||| Flags.MemHostCached)
            Flags.MemDeviceLocal
            true
        |> Option.orElseWith (fun () ->
            createBuffer stagingBytes Flags.BufTransferDst (Flags.MemHostVisible ||| Flags.MemHostCoherent) 0u true)
        |> need "readback staging"

    let pool =
        use s = new Scratch()
        let mutable ci = VkCommandPoolCreateInfo()
        ci.sType <- SType.CommandPoolCreateInfo
        ci.flags <- Flags.PoolTransient
        ci.queueFamilyIndex <- uint32 family
        let out = s.Alloc 8
        ok "vkCreateCommandPool" (fns.CreateCommandPool.Invoke(device, s.Struct ci, 0n, out))
        read<uint64> out

    let cmd =
        use s = new Scratch()
        let mutable ai = VkCommandBufferAllocateInfo()
        ai.sType <- SType.CommandBufferAllocateInfo
        ai.commandPool <- pool
        ai.level <- Enum.CommandBufferLevelPrimary
        ai.commandBufferCount <- 1u
        let out = s.Alloc 8
        ok "vkAllocateCommandBuffers" (fns.AllocateCommandBuffers.Invoke(device, s.Struct ai, out))
        read<nativeint> out

    let timeline =
        use s = new Scratch()
        let mutable ti = VkSemaphoreTypeCreateInfo()
        ti.sType <- SType.SemaphoreTypeCreateInfo
        ti.semaphoreType <- Enum.SemaphoreTypeTimeline
        ti.initialValue <- 0UL
        let mutable ci = VkSemaphoreCreateInfo()
        ci.sType <- SType.SemaphoreCreateInfo
        ci.pNext <- s.Struct ti
        let out = s.Alloc 8
        ok "vkCreateSemaphore" (fns.CreateSemaphore.Invoke(device, s.Struct ci, 0n, out))
        read<uint64> out

    let maxQueries = 64u

    let queries =
        use s = new Scratch()
        let mutable ci = VkQueryPoolCreateInfo()
        ci.sType <- SType.QueryPoolCreateInfo
        ci.queryType <- Enum.QueryTypeTimestamp
        ci.queryCount <- maxQueries
        let out = s.Alloc 8
        ok "vkCreateQueryPool" (fns.CreateQueryPool.Invoke(device, s.Struct ci, 0n, out))
        read<uint64> out

    let mutable timelineValue = 0UL
    let mutable disposed = false

    let barrier (s: Scratch) (srcStage: uint64) (srcAccess: uint64) (dstStage: uint64) (dstAccess: uint64) =
        let mutable mb = VkMemoryBarrier2()
        mb.sType <- SType.MemoryBarrier2
        mb.srcStageMask <- srcStage
        mb.srcAccessMask <- srcAccess
        mb.dstStageMask <- dstStage
        mb.dstAccessMask <- dstAccess
        let mutable di = VkDependencyInfo()
        di.sType <- SType.DependencyInfo
        di.memoryBarrierCount <- 1u
        di.pMemoryBarriers <- s.Struct mb
        fns.CmdPipelineBarrier2.Invoke(cmd, s.Struct di)

    let facts =
        let heapType placement =
            match placement with
            | Bar -> pickType UInt32.MaxValue (Flags.MemDeviceLocal ||| Flags.MemHostVisible ||| Flags.MemHostCoherent) 0u
            | HostMemory ->
                pickType UInt32.MaxValue (Flags.MemHostVisible ||| Flags.MemHostCoherent) Flags.MemDeviceLocal
            | CopyToDeviceLocal -> pickType UInt32.MaxValue Flags.MemDeviceLocal Flags.MemHostVisible

        {
            Name = caps.Identity.Name
            Driver = sprintf "%s %s" caps.Identity.DriverName caps.Identity.DriverInfo
            ApiVersion = VkVersion.toString caps.Identity.ApiVersion
            QueueFamily = family
            QueueFamilyFlags = queueFamilies[family].queueFlags
            TimestampValidBits = queueFamilies[family].timestampValidBits
            TimestampPeriodNs = float props.limits.timestampPeriod
            MemoryTypes = memProps
            Heap = caps.DescriptorHeap
            StorageBufferDescriptorSize = descriptorSize
            MinStorageBufferOffsetAlignment = props.limits.minStorageBufferOffsetAlignment
            MaxStorageBufferRange = props.limits.maxStorageBufferRange
            PageMemoryType = pages[1].MemoryType
            UploadMemoryType = upload.MemoryType
            ReadbackMemoryType = readback.MemoryType
            HeapMemoryTypes =
                [ Bar; HostMemory; CopyToDeviceLocal ]
                |> List.choose (fun p -> heapType p |> Option.map (fun m -> p, m.Index))
        }

    member _.Facts = facts
    member _.Fns = fns
    member _.Device = device
    member _.Pages = pages
    member _.UploadStaging = upload
    member _.Cmd = cmd
    member _.CreateBuffer(bytes, usage, want, avoid, map) = createBuffer bytes usage want avoid map
    member _.DestroyBuffer b = destroyBuffer b
    member _.Barrier(s, a, b, c, d) = barrier s a b c d

    /// Records with `record`, submits once, signals the timeline and waits for it on the host.
    member _.Submit(record: Scratch -> unit) =
        use s = new Scratch()
        ok "vkResetCommandPool" (fns.ResetCommandPool.Invoke(device, pool, 0u))
        let mutable bi = VkCommandBufferBeginInfo()
        bi.sType <- SType.CommandBufferBeginInfo
        bi.flags <- Flags.CmdOneTimeSubmit
        ok "vkBeginCommandBuffer" (fns.BeginCommandBuffer.Invoke(cmd, s.Struct bi))
        record s
        ok "vkEndCommandBuffer" (fns.EndCommandBuffer.Invoke cmd)
        timelineValue <- timelineValue + 1UL
        let mutable cbi = VkCommandBufferSubmitInfo()
        cbi.sType <- SType.CommandBufferSubmitInfo
        cbi.commandBuffer <- cmd
        let mutable ssi = VkSemaphoreSubmitInfo()
        ssi.sType <- SType.SemaphoreSubmitInfo
        ssi.semaphore <- timeline
        ssi.value <- timelineValue
        ssi.stageMask <- Flags.Stage2AllCommands
        let mutable si = VkSubmitInfo2()
        si.sType <- SType.SubmitInfo2
        si.commandBufferInfoCount <- 1u
        si.pCommandBufferInfos <- s.Struct cbi
        si.signalSemaphoreInfoCount <- 1u
        si.pSignalSemaphoreInfos <- s.Struct ssi
        ok "vkQueueSubmit2" (fns.QueueSubmit2.Invoke(queue, 1u, s.Struct si, 0UL))
        let mutable wi = VkSemaphoreWaitInfo()
        wi.sType <- SType.SemaphoreWaitInfo
        wi.semaphoreCount <- 1u
        wi.pSemaphores <- s.Struct timeline
        wi.pValues <- s.Struct timelineValue
        ok "vkWaitSemaphores (timeline)" (fns.WaitSemaphores.Invoke(device, s.Struct wi, timeoutNs))

    member _.TimelineValue = timelineValue

    /// Runs one job against a heap that is already bound by `bindHeap` (recorded first in the command buffer).
    member this.RunJob(bindHeap: Scratch -> unit, pipelines: Map<string, uint64>, job: Job) : JobResult =
        let nq = uint32 (2 * job.Dispatches.Length)

        if nq > maxQueries then
            failwith "too many dispatches for the timestamp pool"

        // Stage uploads.
        let mutable at = 0UL

        let uploads =
            [
                for u in job.Uploads do
                    if at + uint64 u.Data.Length > stagingBytes then
                        failwith "upload staging too small"

                    copyIn (upload.Mapped + nativeint at) u.Data
                    yield u, at
                    at <- alignUp (at + uint64 u.Data.Length) 256UL
            ]

        let mutable rat = 0UL

        let readbacks =
            [
                for r in job.Readbacks do
                    if rat + uint64 r.Length > stagingBytes then
                        failwith "readback staging too small"

                    yield r, rat
                    rat <- alignUp (rat + uint64 r.Length) 256UL
            ]

        this.Submit(fun s ->
            if nq > 0u then
                fns.CmdResetQueryPool.Invoke(cmd, queries, 0u, nq)

            for (u, off) in uploads do
                let mutable c = VkBufferCopy()
                c.srcOffset <- off
                c.dstOffset <- uint64 u.Offset
                c.size <- uint64 u.Data.Length
                fns.CmdCopyBuffer.Invoke(cmd, upload.Handle, pages[u.Slot].Handle, 1u, s.Struct c)

            barrier
                s
                Flags.Stage2Copy
                Flags.Access2TransferWrite
                Flags.Stage2ComputeShader
                (Flags.Access2ShaderStorageRead ||| Flags.Access2ShaderStorageWrite)

            bindHeap s

            job.Dispatches
            |> List.iteri (fun k d ->
                if d.Args.Length <> 32 || d.Args[0] <> d.GroupsX then
                    failwith "ArgBlock must be 32 dwords with dword0 = groups_x"

                fns.CmdBindPipeline.Invoke(cmd, Enum.BindPointCompute, pipelines[d.Kernel])
                let a = s.Array d.Args
                let mutable pi = VkPushDataInfoEXT()
                pi.sType <- SType.PushDataInfoEXT
                pi.offset <- 0u
                let mutable data = VkHostAddressRangeConstEXT()
                data.address <- a
                data.size <- 128un
                pi.data <- data
                fns.CmdPushDataEXT.Invoke(cmd, s.Struct pi)
                fns.CmdWriteTimestamp2.Invoke(cmd, Flags.Stage2AllCommands, queries, uint32 (2 * k))
                fns.CmdDispatch.Invoke(cmd, d.GroupsX, d.GroupsY, 1u)
                fns.CmdWriteTimestamp2.Invoke(cmd, Flags.Stage2AllCommands, queries, uint32 (2 * k + 1))

                barrier
                    s
                    Flags.Stage2ComputeShader
                    Flags.Access2ShaderStorageWrite
                    (Flags.Stage2ComputeShader ||| Flags.Stage2Copy)
                    (Flags.Access2ShaderStorageRead
                     ||| Flags.Access2ShaderStorageWrite
                     ||| Flags.Access2TransferRead))

            for (r, off) in readbacks do
                let mutable c = VkBufferCopy()
                c.srcOffset <- uint64 r.Offset
                c.dstOffset <- off
                c.size <- uint64 r.Length
                fns.CmdCopyBuffer.Invoke(cmd, pages[r.Slot].Handle, readback.Handle, 1u, s.Struct c)

            barrier s Flags.Stage2Copy Flags.Access2TransferWrite Flags.Stage2Host Flags.Access2HostRead)

        let outputs =
            [ for (r, off) in readbacks -> copyOut (readback.Mapped + nativeint off) r.Length ]

        let times =
            if nq = 0u then
                []
            else
                use s = new Scratch()
                let data = s.Alloc(int nq * 8)

                ok
                    "vkGetQueryPoolResults"
                    (fns.GetQueryPoolResults.Invoke(
                        device,
                        queries,
                        0u,
                        nq,
                        unativeint (int nq * 8),
                        data,
                        8UL,
                        Flags.Query64 ||| Flags.QueryWait
                    ))

                let mask =
                    if facts.TimestampValidBits >= 64u then
                        UInt64.MaxValue
                    else
                        (1UL <<< int facts.TimestampValidBits) - 1UL

                [
                    for k in 0 .. job.Dispatches.Length - 1 do
                        let t0 = readAt<uint64> data (2 * k) &&& mask
                        let t1 = readAt<uint64> data (2 * k + 1) &&& mask
                        float ((t1 - t0) &&& mask) * facts.TimestampPeriodNs, int64 (float t0 * facts.TimestampPeriodNs)
                ]

        {
            Outputs = outputs
            DispatchGpuNs = List.map fst times
            DispatchGpuStartNs = List.map snd times
        }

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                fns.DeviceWaitIdle.Invoke device |> ignore
                fns.DestroyQueryPool.Invoke(device, queries, 0n)
                fns.DestroySemaphore.Invoke(device, timeline, 0n)
                fns.DestroyCommandPool.Invoke(device, pool, 0n)

                for b in List.ofSeq buffers do
                    destroyBuffer b

                fns.DestroyDevice.Invoke(device, 0n)
                (instance :> IDisposable).Dispose()
                (loader :> IDisposable).Dispose()

/// Heap geometry: slot i at byte HeapOffset + i * Stride, the reserved range after the slots, aligned to 64 bytes (a
/// multiple of bufferDescriptorAlignment and imageDescriptorAlignment on the RTX 4080). The recipe is HeapOffset 0 and
/// Stride = bufferDescriptorSize (see `geometry`); other values are mapping controls.
type HeapGeometry =
    {
        HeapOffset: uint64
        Stride: uint64
        /// bufferDescriptorSize: the bytes vkWriteResourceDescriptorsEXT writes per slot.
        DescriptorBytes: uint64
        /// Bytes from the heap start to the end of the last slot.
        SlotBytes: uint64
        ReservedOffset: uint64
        ReservedSize: uint64
        HeapBytes: uint64
    }

let geometryWith (h: DeviceCheck.DescriptorHeapLimits) (heapOffset: uint64) (stride: uint64) =
    let slotBytes = heapOffset + uint64 SlotCount * stride

    let align =
        [ 64UL; h.BufferDescriptorAlignment; h.ImageDescriptorAlignment ]
        |> List.max

    let reservedOffset = alignUp slotBytes align
    let reservedSize = h.MinResourceHeapReservedRange

    {
        HeapOffset = heapOffset
        Stride = stride
        DescriptorBytes = h.BufferDescriptorSize
        SlotBytes = slotBytes
        ReservedOffset = reservedOffset
        ReservedSize = reservedSize
        HeapBytes = reservedOffset + reservedSize
    }

/// The recipe's geometry: heapOffset 0, stride bufferDescriptorSize.
let geometry (h: DeviceCheck.DescriptorHeapLimits) = geometryWith h 0UL h.BufferDescriptorSize

/// The one mapping of the recipe: set 0, binding 0, one binding, every storage-buffer variable (read-only and
/// read-write), HEAP_WITH_CONSTANT_OFFSET with heapOffset and heapArrayStride (0 and bufferDescriptorSize in the
/// recipe), no embedded sampler. The ContractKAT in docs/contracts/vulkan-heap-recipe.md is its byte image.
let recipeMapping (heapOffset: uint64) (stride: uint64) =
    let mutable co = VkDescriptorMappingSourceConstantOffsetEXT()
    co.heapOffset <- uint32 heapOffset
    co.heapArrayStride <- uint32 stride
    let mutable m = VkDescriptorSetAndBindingMappingEXT()
    m.sType <- SType.DescriptorSetAndBindingMappingEXT
    m.descriptorSet <- 0u
    m.firstBinding <- 0u
    m.bindingCount <- 1u
    m.resourceMask <- Flags.SpirvReadOnlyStorageBuffer ||| Flags.SpirvReadWriteStorageBuffer
    m.source <- Enum.MappingSourceHeapWithConstantOffset
    let mutable sd = VkDescriptorMappingSourceDataEXT()
    sd.constantOffset <- co
    m.sourceData <- sd
    m

/// The raw bytes of an unmanaged struct (for known-answer checks).
let structBytes<'T when 'T: unmanaged> (value: 'T) =
    let b = Array.zeroCreate<byte> sizeof<'T>
    let h = GCHandle.Alloc(b, GCHandleType.Pinned)

    try
        NativePtr.write (NativePtr.ofNativeInt<'T> (h.AddrOfPinnedObject())) value
    finally
        h.Free()

    b

/// What a configuration observed while it ran.
type HeapObservation =
    {
        HeapAddress: uint64
        HeapMemoryType: int
        BufferAlignment: uint64
        /// Bytes of the reserved range the driver changed (canary check; only for host-visible placements).
        ReservedBytesChanged: int option
        /// The 16 descriptor bytes of slot 1, as vkWriteResourceDescriptorsEXT wrote them.
        Slot1Descriptor: byte[]
    }

/// A mapping other than the recipe's, to show the driver honours heapOffset and heapArrayStride.
type MappingControl =
    {
        HeapOffset: uint64
        Stride: uint64
        Suffix: string
    }

/// One configuration: a binding variant and a heap placement over a shared context, with the recipe's mapping unless
/// a control is given.
type VulkanConfig
    (
        ctx: VulkanContext,
        shaders: Shaders.ShaderSet,
        variant: Shaders.Variant,
        placement: HeapPlacement,
        control: MappingControl option
    ) =
    let fns = ctx.Fns
    let device = ctx.Device
    let facts = ctx.Facts

    let geo =
        match control with
        | None -> geometry facts.Heap
        | Some c -> geometryWith facts.Heap c.HeapOffset c.Stride

    let label =
        sprintf
            "vulkan/%s/%s%s"
            (Shaders.variantName variant)
            (placementName placement)
            (control |> Option.map (fun c -> "+" + c.Suffix) |> Option.defaultValue "")

    do
        if facts.StorageBufferDescriptorSize <> facts.Heap.BufferDescriptorSize then
            failwithf
                "vkGetPhysicalDeviceDescriptorSizeEXT(STORAGE_BUFFER) = %d but bufferDescriptorSize = %d"
                facts.StorageBufferDescriptorSize
                facts.Heap.BufferDescriptorSize

        if geo.Stride % facts.Heap.BufferDescriptorAlignment <> 0UL
           || geo.HeapOffset % facts.Heap.BufferDescriptorAlignment <> 0UL then
            failwith "heapOffset and heapArrayStride must be multiples of bufferDescriptorAlignment"

        if geo.HeapBytes > facts.Heap.MaxResourceHeapSize then
            failwith "heap larger than maxResourceHeapSize"

    let heapUsage =
        Flags.BufDescriptorHeapEXT ||| Flags.BufShaderDeviceAddress ||| Flags.BufTransferDst

    // Extra bytes so the heap start can be aligned up to resourceHeapAlignment inside the buffer.
    let slack = max facts.Heap.ResourceHeapAlignment 64UL

    let heapBuffer =
        let bytes = geo.HeapBytes + slack

        let b =
            match placement with
            | Bar ->
                ctx.CreateBuffer(
                    bytes,
                    heapUsage,
                    Flags.MemDeviceLocal ||| Flags.MemHostVisible ||| Flags.MemHostCoherent,
                    0u,
                    true
                )
            | HostMemory ->
                ctx.CreateBuffer(
                    bytes,
                    heapUsage,
                    Flags.MemHostVisible ||| Flags.MemHostCoherent,
                    Flags.MemDeviceLocal,
                    true
                )
            | CopyToDeviceLocal -> ctx.CreateBuffer(bytes, heapUsage, Flags.MemDeviceLocal, Flags.MemHostVisible, false)

        match b with
        | Some b -> b
        | None -> failwithf "%s: no memory type for the heap buffer" label

    let heapAddress = alignUp heapBuffer.Address facts.Heap.ResourceHeapAlignment
    let heapOffsetInBuffer = heapAddress - heapBuffer.Address

    /// Writes all 64 descriptors to `dst` (host memory laid out like the heap) in one call.
    let writeDescriptors (dst: nativeint) =
        use s = new Scratch()
        let ranges = s.Alloc(SlotCount * sizeof<VkDeviceAddressRangeEXT>)
        let infos = s.Alloc(SlotCount * sizeof<VkResourceDescriptorInfoEXT>)
        let hosts = s.Alloc(SlotCount * sizeof<VkHostAddressRangeEXT>)

        for slot in 0 .. SlotCount - 1 do
            let page = ctx.Pages[slot]

            if page.Address % facts.MinStorageBufferOffsetAlignment <> 0UL then
                failwithf "slot %d address 0x%x is not aligned to minStorageBufferOffsetAlignment" slot page.Address

            let pRange = ranges + nativeint (slot * sizeof<VkDeviceAddressRangeEXT>)
            let mutable range = VkDeviceAddressRangeEXT()
            range.address <- page.Address
            range.size <- page.Bytes
            NativePtr.write (NativePtr.ofNativeInt<VkDeviceAddressRangeEXT> pRange) range
            let mutable info = VkResourceDescriptorInfoEXT()
            info.sType <- SType.ResourceDescriptorInfoEXT
            info.``type`` <- Enum.DescriptorTypeStorageBuffer
            let mutable data = VkResourceDescriptorDataEXT()
            data.pAddressRange <- pRange
            info.data <- data
            NativePtr.set (NativePtr.ofNativeInt<VkResourceDescriptorInfoEXT> infos) slot info

            let mutable host = VkHostAddressRangeEXT()
            host.address <- dst + nativeint (geo.HeapOffset + uint64 slot * geo.Stride)
            host.size <- unativeint geo.DescriptorBytes
            NativePtr.set (NativePtr.ofNativeInt<VkHostAddressRangeEXT> hosts) slot host

        ok "vkWriteResourceDescriptorsEXT" (fns.WriteResourceDescriptorsEXT.Invoke(device, uint32 SlotCount, infos, hosts))

    let canary = 0xA5uy

    let slot1Descriptor =
        match placement with
        | Bar
        | HostMemory ->
            let heapHost = heapBuffer.Mapped + nativeint heapOffsetInBuffer
            // Canary in the reserved range, written before the heap is ever bound (allowed: nothing has it bound).
            for i in 0 .. int geo.ReservedSize - 1 do
                Marshal.WriteByte(heapHost + nativeint geo.ReservedOffset + nativeint i, canary)

            writeDescriptors heapHost
            copyOut (heapHost + nativeint (geo.HeapOffset + geo.Stride)) (int geo.DescriptorBytes)
        | CopyToDeviceLocal ->
            // Descriptors go to host memory, then through the upload staging buffer into the device-local heap.
            let host = Array.zeroCreate<byte> (int geo.SlotBytes)
            let h = GCHandle.Alloc(host, GCHandleType.Pinned)

            try
                writeDescriptors (h.AddrOfPinnedObject())
            finally
                h.Free()

            copyIn ctx.UploadStaging.Mapped host

            ctx.Submit(fun s ->
                let mutable c = VkBufferCopy()
                c.srcOffset <- 0UL
                c.dstOffset <- heapOffsetInBuffer
                c.size <- uint64 host.Length
                fns.CmdCopyBuffer.Invoke(ctx.Cmd, ctx.UploadStaging.Handle, heapBuffer.Handle, 1u, s.Struct c)
                // Pipeline-barrier scopes extend to later submissions on the queue, so this covers every later job.
                ctx.Barrier(
                    s,
                    Flags.Stage2Copy,
                    Flags.Access2TransferWrite,
                    Flags.Stage2ComputeShader,
                    Flags.Access2ResourceHeapReadEXT
                ))

            Array.sub host (int (geo.HeapOffset + geo.Stride)) (int geo.DescriptorBytes)

    let mapping (s: Scratch) =
        let mutable mi = VkShaderDescriptorSetAndBindingMappingInfoEXT()
        mi.sType <- SType.ShaderDescriptorSetAndBindingMappingInfoEXT
        mi.mappingCount <- 1u
        mi.pMappings <- s.Struct(recipeMapping geo.HeapOffset geo.Stride)
        s.Struct mi

    let pipelines =
        [
            for k in Shaders.kernels do
                use s = new Scratch()
                let spv = shaders.Spirv[(variant, k)]
                let code = s.Alloc spv.Length
                copyIn code spv
                let mutable mci = VkShaderModuleCreateInfo()
                mci.sType <- SType.ShaderModuleCreateInfo
                mci.codeSize <- unativeint spv.Length
                mci.pCode <- code
                let out = s.Alloc 8
                ok "vkCreateShaderModule" (fns.CreateShaderModule.Invoke(device, s.Struct mci, 0n, out))
                let shaderModule = read<uint64> out
                let mutable stage = VkPipelineShaderStageCreateInfo()
                stage.sType <- SType.PipelineShaderStageCreateInfo
                stage.pNext <- mapping s
                stage.stage <- Flags.StageCompute
                stage.``module`` <- shaderModule
                stage.pName <- s.Utf8 "main"
                let mutable f2 = VkPipelineCreateFlags2CreateInfo()
                f2.sType <- SType.PipelineCreateFlags2CreateInfo
                f2.flags <- Flags.Pipeline2DescriptorHeapEXT
                let mutable ci = VkComputePipelineCreateInfo()
                ci.sType <- SType.ComputePipelineCreateInfo
                ci.pNext <- s.Struct f2
                ci.stage <- stage
                ci.layout <- 0UL // VK_NULL_HANDLE: the mapping replaces the layout
                ci.basePipelineIndex <- -1

                let r = fns.CreateComputePipelines.Invoke(device, 0UL, 1u, s.Struct ci, 0n, out)
                fns.DestroyShaderModule.Invoke(device, shaderModule, 0n)
                ok (sprintf "vkCreateComputePipelines(%s, %s)" k label) r
                yield k, read<uint64> out
        ]
        |> Map.ofList

    let bindHeap (s: Scratch) =
        let mutable bi = VkBindHeapInfoEXT()
        bi.sType <- SType.BindHeapInfoEXT
        let mutable range = VkDeviceAddressRangeEXT()
        range.address <- heapAddress
        range.size <- geo.HeapBytes
        bi.heapRange <- range
        bi.reservedRangeOffset <- geo.ReservedOffset
        bi.reservedRangeSize <- geo.ReservedSize
        ctx.Fns.CmdBindResourceHeapEXT.Invoke(ctx.Cmd, s.Struct bi)

    let mutable disposed = false

    member _.Label = label
    member _.Geometry = geo

    /// Reads the reserved range back (host-visible placements only) and counts bytes that are not the canary.
    member _.Observe() : HeapObservation =
        let changed =
            match placement with
            | Bar
            | HostMemory ->
                let heapHost = heapBuffer.Mapped + nativeint heapOffsetInBuffer
                let bytes = copyOut (heapHost + nativeint geo.ReservedOffset) (int geo.ReservedSize)
                Some(bytes |> Array.filter (fun b -> b <> canary) |> Array.length)
            | CopyToDeviceLocal -> None

        {
            HeapAddress = heapAddress
            HeapMemoryType = heapBuffer.MemoryType
            BufferAlignment = heapBuffer.Alignment
            ReservedBytesChanged = changed
            Slot1Descriptor = slot1Descriptor
        }

    interface IGpu with
        member _.Label = label
        member _.Run job = ctx.RunJob(bindHeap, pipelines, job)

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                fns.DeviceWaitIdle.Invoke device |> ignore

                for KeyValue(_, p) in pipelines do
                    fns.DestroyPipeline.Invoke(device, p, 0n)

                ctx.DestroyBuffer heapBuffer
