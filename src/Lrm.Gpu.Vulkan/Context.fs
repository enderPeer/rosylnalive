namespace Lrm.Gpu.Vulkan

#nowarn "9"
#nowarn "51"

open System
open System.Collections.Generic
open System.Runtime.InteropServices
open System.Threading
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop
open Microsoft.FSharp.NativeInterop

/// The Vulkan objects the backends of this process have created and not yet destroyed (instances, devices, buffers,
/// memory, pools, pipelines, semaphores): the leak check's counter, as Com.liveReferences is for D3D12.
[<RequireQualifiedAccess>]
module VulkanObjects =
    let mutable private live = 0

    /// Objects alive now.
    let count () = Volatile.Read &live

    let internal created () = Interlocked.Increment &live |> ignore
    let internal destroyed () = Interlocked.Decrement &live |> ignore

/// A buffer with its own memory allocation.
type internal NativeBuffer =
    {
        Handle: VkBuffer
        Memory: VkDeviceMemory
        Bytes: int64
        /// vkGetBufferDeviceAddress, or 0 for buffers without SHADER_DEVICE_ADDRESS usage.
        Address: uint64
        /// The persistent mapping, or 0n.
        Mapped: nativeint
        MemoryType: int
        Kind: BufferKind
        Name: string
    }

/// gpu.buffer_memory: a Gauge in bytes of the device memory the backend holds outside the heap's arenas (mem.arena),
/// per `buffer`: `upload_staging` and `readback_staging` (two buffers each), `resource_heap` (the descriptor heap
/// buffer), `dummy_page` (the page behind unused slots) and `probe_heap` (the denormal probe's transient heap, back to
/// 0 once it is released). Emitted whenever that amount changes.
[<RequireQualifiedAccess>]
module internal BufferMemory =
    [<Literal>]
    let Name = "gpu.buffer_memory"

    let report (buffer: string) (bytes: int64) =
        Telemetry.gauge
            Name
            (float bytes)
            TelemetryUnits.Bytes
            [
                Attr.str TelemetryKeys.Backend (ExecutionBackend.name ExecutionBackend.Vulkan)
                Attr.str "buffer" buffer
            ]

/// Host transfers through the staging buffers (GpuBackendInterface: Upload, UploadFrom and Download wait for all
/// submitted work, then transfer, in chunks of at most 64 MiB).
[<RequireQualifiedAccess>]
module Transfer =
    /// The largest chunk, which is also the size of each of the two upload and two readback staging buffers.
    [<Literal>]
    let ChunkBytes = 67108864L

    /// The staging size for a transfer of `bytes`: the chunk size, or less for a small transfer (64 KiB steps).
    let stagingBytes (bytes: int64) =
        min ChunkBytes (max Heap.PageGranularity (Heap.alignUp Heap.PageGranularity bytes))

    /// The chunks of a transfer: (offset from the start, size) pairs of at most ChunkBytes.
    let chunks (bytes: int64) =
        [
            let mutable at = 0L

            while at < bytes do
                let n = min ChunkBytes (bytes - at)
                yield at, n
                at <- at + n
        ]

/// The resource heap's binding (VkBindHeapInfoEXT, VulkanHeapRecipe sections 3 and 6.1).
type internal HeapRange =
    {
        /// heapRange.address: the heap buffer's address aligned up to resourceHeapAlignment.
        Address: uint64
        /// heapRange.size.
        Bytes: uint64
        ReservedOffset: uint64
        ReservedBytes: uint64
    }

/// What a command buffer binds and how the encoder's targets resolve to buffers.
type internal Bindings =
    {
        Heap: HeapRange option
        Pipeline: KernelId -> VkPipeline
        Buffer: Target -> NativeBuffer
    }

/// One command pool and its primary command buffer, with the timestamp query pool of the submission that uses it.
/// A frame is reused only after its timeline value has completed, and its pool is reset before it records again.
[<Sealed; AllowNullLiteral>]
type internal Frame(index: int, pool: VkCommandPool, command: VkCommandBuffer) =
    let mutable queries = VkQueryPool.Null
    let mutable capacity = 0

    member _.Index = index
    member _.Pool = pool
    member _.Command = command
    member _.Queries = queries
    member _.QueryCapacity = capacity

    /// The timeline value of the submission that last used the frame (0 before its first use).
    member val Value = 0UL with get, set

    /// True once the frame's command buffer has recorded since its pool was last reset.
    member val Recorded = false with get, set

    /// Runs when the frame's submission has completed, before the frame is reused (reads its timestamps).
    member val OnRetire: (Frame -> unit) option = None with get, set

    /// Makes room for `count` timestamp queries. Only called on a frame that is not in flight.
    member _.EnsureQueries(c: VkDeviceCommands, device: VkDevice, count: int, name: string -> unit) =
        if count > capacity then
            if not queries.IsNull then
                c.DestroyQueryPool(device, queries)
                VulkanObjects.destroyed ()
                queries <- VkQueryPool.Null

            let n = max 64 (int (Numerics.BitOperations.RoundUpToPowerOf2(uint32 count)))
            let mutable ci = VkQueryPoolCreateInfo.Create()
            ci.queryType <- VkQueryType.VK_QUERY_TYPE_TIMESTAMP
            ci.queryCount <- uint32 n
            let mutable q = VkQueryPool.Null
            c.CreateQueryPool(device, &ci, &q)
            VulkanObjects.created ()
            queries <- q
            capacity <- n
            name (sprintf "lrm.vulkan.frame%d.timestamps" index)

    /// The first `count` timestamps, masked to `mask` (timestampValidBits).
    member _.ReadStamps(c: VkDeviceCommands, device: VkDevice, count: int, mask: uint64) : uint64[] =
        if count = 0 then
            [||]
        else
            let out = Array.zeroCreate<uint64> count

            c.GetQueryPoolResults(
                device,
                queries,
                0u,
                uint32 count,
                MemoryMarshal.AsBytes(Span out),
                8UL,
                VkQueryResultFlagBits.VK_QUERY_RESULT_64_BIT
                ||| VkQueryResultFlagBits.VK_QUERY_RESULT_WAIT_BIT
            )
            |> ignore

            for i in 0 .. count - 1 do
                out[i] <- out[i] &&& mask

            out

    member _.Destroy(c: VkDeviceCommands, device: VkDevice) =
        if not queries.IsNull then
            c.DestroyQueryPool(device, queries)
            VulkanObjects.destroyed ()
            queries <- VkQueryPool.Null

        c.DestroyCommandPool(device, pool)
        VulkanObjects.destroyed ()

/// A staging buffer and the timeline value of its last use.
[<Sealed>]
type internal Staging(buffer: NativeBuffer) =
    member _.Buffer = buffer
    member _.Bytes = buffer.Bytes
    member _.Mapped = buffer.Mapped
    member val LastValue = 0UL with get, set

/// The instance, the recipe's device, its compute-only queue and timeline semaphore, the pool of frames and the staging
/// buffers. Every submission signals the timeline (vkQueueSubmit2 at ALL_COMMANDS), every wait is bounded, and a lost
/// device raises VulkanDeviceLostException with VK_EXT_device_fault's report.
[<Sealed>]
type internal DeviceContext
    (
        instance: Instance,
        device: Device,
        caps: DeviceCheck.Capabilities,
        messenger: DebugMessenger option,
        waitTimeout: TimeSpan
    ) =
    let c = device.Commands
    let dev = device.Handle
    let debugUtils = instance.HasExtension VkConst.VK_EXT_DEBUG_UTILS_EXTENSION_NAME

    let memoryTypes =
        let p = instance.MemoryProperties device.PhysicalDevice
        let types = p.memoryTypes.ToArray()
        Array.init (int p.memoryTypeCount) (fun i -> types[i].propertyFlags)

    let memoryHeaps =
        let p = instance.MemoryProperties device.PhysicalDevice
        let types = p.memoryTypes.ToArray()
        Array.init (int p.memoryTypeCount) (fun i -> int types[i].heapIndex)

    let timestampMask =
        if caps.TimestampValidBits >= 64u then
            UInt64.MaxValue
        else
            (1UL <<< int caps.TimestampValidBits) - 1UL

    let period = float caps.TimestampPeriod
    let mutable semaphore = VkSemaphore.Null
    let mutable nextValue = 0UL
    let free = Stack<Frame>()
    let inFlight = Queue<Frame>()
    let allFrames = ResizeArray<Frame>()
    let uploads = ResizeArray<Staging>()
    let readbacks = ResizeArray<Staging>()
    let choices = Dictionary<BufferKind, int>()
    let mutable disposed = false

    let name (objectType: VkObjectType) (handle: uint64) (text: string) =
        if debugUtils then
            DebugUtils.setName device objectType handle text

    do
        let mutable timeline = VkSemaphoreTypeCreateInfo.Create()
        timeline.semaphoreType <- VkSemaphoreType.VK_SEMAPHORE_TYPE_TIMELINE
        timeline.initialValue <- 0UL
        let mutable sci = VkSemaphoreCreateInfo.Create()
        sci.pNext <- NativePtr.toNativeInt &&timeline
        let mutable s = VkSemaphore.Null
        c.CreateSemaphore(dev, &sci, &s)
        VulkanObjects.created ()
        semaphore <- s
        name VkObjectType.VK_OBJECT_TYPE_SEMAPHORE s.Handle "lrm.vulkan.timeline"
        name VkObjectType.VK_OBJECT_TYPE_QUEUE (uint64 device.Queue.Handle) "lrm.vulkan.queue"

    member _.Instance = instance
    member _.Device = device
    member _.Commands = c
    member _.Handle = dev
    member _.Capabilities = caps
    member _.Messenger = messenger

    /// True when the instance has VK_EXT_debug_utils, so objects carry debug names.
    member _.DebugUtils = debugUtils

    /// The memory types' property flags, by index.
    member _.MemoryTypes = memoryTypes

    /// The memory type chosen for each kind of buffer created so far (by flags, VulkanHeapRecipe section 2).
    member _.MemoryChoices: (BufferKind * int) list =
        BufferKind.all
        |> List.choose (fun k ->
            match choices.TryGetValue k with
            | true, i -> Some(k, i)
            | _ -> None)

    member _.TimestampMask = timestampMask

    /// timestampPeriod: ns per tick.
    member _.TimestampPeriod = period

    /// Frames created so far: the pool's size, which stays flat once the in-flight depth stops growing.
    member _.FramesCreated = allFrames.Count

    /// The last timeline value signalled.
    member _.LastValue = nextValue

    /// GPU ticks (masked) to ns.
    member _.TicksToNs(ticks: uint64) : int64 =
        let t = ticks &&& timestampMask

        if period = 1.0 then
            int64 t
        else
            int64 (Math.Round(float t * period))

    /// Names a Vulkan object when VK_EXT_debug_utils is enabled.
    member _.Name(objectType: VkObjectType, handle: uint64, text: string) = name objectType handle text

    /// VK_EXT_device_fault's report now.
    member _.Fault() : DeviceFaultQuery =
        try
            DeviceFault.query device
        with _ ->
            DeviceFaultQuery.NoFault VkResult.VK_ERROR_UNKNOWN

    /// The exception for a device lost at `call`, with the fault report.
    member x.Lost(call: string, result: VkResult) =
        VulkanDeviceLostException(call, result, x.Fault())

    /// The timeline's completed value.
    member x.Completed: uint64 =
        let mutable v = 0UL

        try
            c.GetSemaphoreCounterValue(dev, semaphore, &v)
        with VulkanException(_, VkResult.VK_ERROR_DEVICE_LOST) ->
            raise (x.Lost("vkGetSemaphoreCounterValue", VkResult.VK_ERROR_DEVICE_LOST))

        v

    /// Retires every frame whose submission has completed (running its OnRetire), so it can be reused.
    member x.Recycle() =
        if inFlight.Count > 0 then
            let completed = x.Completed

            while inFlight.Count > 0 && inFlight.Peek().Value <= completed do
                let f = inFlight.Dequeue()

                match f.OnRetire with
                | Some retire ->
                    f.OnRetire <- None
                    retire f
                | None -> ()

                free.Push f

    /// Blocks until timeline value `value` has completed, at most the wait timeout, then retires completed frames.
    /// Raises VulkanDeviceLostException, with the fault report, when the device is lost or the wait timed out.
    member x.WaitFor(value: uint64) =
        if value > nextValue then
            invalidArg "value" (sprintf "timeline value %d was never signalled (last %d)" value nextValue)

        if value > 0UL then
            let mutable s = semaphore
            let mutable v = value
            let mutable wait = VkSemaphoreWaitInfo.Create()
            wait.semaphoreCount <- 1u
            wait.pSemaphores <- NativePtr.toNativeInt &&s
            wait.pValues <- NativePtr.toNativeInt &&v
            let timeoutNs = uint64 waitTimeout.Ticks * 100UL

            let r =
                try
                    c.WaitSemaphores(dev, &wait, timeoutNs)
                with VulkanException(_, VkResult.VK_ERROR_DEVICE_LOST) ->
                    raise (x.Lost(sprintf "timeline wait for %d" value, VkResult.VK_ERROR_DEVICE_LOST))

            match r with
            | VkResult.VK_SUCCESS -> ()
            | VkResult.VK_TIMEOUT ->
                raise (
                    x.Lost(
                        sprintf "timeline wait for %d (completed %d after %O)" value x.Completed waitTimeout,
                        VkResult.VK_TIMEOUT
                    )
                )
            | other -> raise (VulkanException("vkWaitSemaphores", other))

        x.Recycle()

    /// Waits for everything submitted so far.
    member x.WaitIdle() = x.WaitFor nextValue

    /// Waits for everything, then resets every frame's command pool, so no command buffer still holds a heap binding
    /// (VulkanHeapRecipe section 4: before a heap is destroyed, every command buffer that bound one is complete and
    /// reset).
    member x.ResetFrames() =
        x.WaitIdle()

        for f in free do
            if f.Recorded then
                c.ResetCommandPool(dev, f.Pool, LanguagePrimitives.EnumOfValue 0u)
                f.Recorded <- false

    member private _.AcquireFrame() : Frame =
        if free.Count > 0 then
            free.Pop()
        else
            let index = allFrames.Count
            let mutable pi = VkCommandPoolCreateInfo.Create()
            pi.flags <- VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
            pi.queueFamilyIndex <- device.QueueFamily
            let mutable pool = VkCommandPool.Null
            c.CreateCommandPool(dev, &pi, &pool)
            VulkanObjects.created ()
            let mutable ai = VkCommandBufferAllocateInfo.Create()
            ai.commandPool <- pool
            ai.level <- VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY
            ai.commandBufferCount <- 1u
            let mutable cmd = VkCommandBuffer.Null
            c.AllocateCommandBuffers(dev, &ai, NativePtr.toNativeInt &&cmd)
            name VkObjectType.VK_OBJECT_TYPE_COMMAND_POOL pool.Handle (sprintf "lrm.vulkan.frame%d.pool" index)

            name
                VkObjectType.VK_OBJECT_TYPE_COMMAND_BUFFER
                (uint64 cmd.Handle)
                (sprintf "lrm.vulkan.frame%d.commands" index)

            let f = new Frame(index, pool, cmd)
            allFrames.Add f
            f

    /// Records `ops` into a frame (resetting its pool first), submits it with vkQueueSubmit2 signalling the next
    /// timeline value, and returns the frame, whose Value is the submission's, with `onRetire` to run once it has
    /// completed.
    member x.Execute
        (bindings: Bindings option)
        (queries: int)
        (ops: NativeOp list)
        (onRetire: (Frame -> unit) option)
        : Frame =
        if disposed then
            raise (ObjectDisposedException "VulkanBackend")

        x.Recycle()
        let frame = x.AcquireFrame()

        try
            x.Record(frame, bindings, queries, ops)
        with _ ->
            // Never submitted: the pool is reset before the frame records again.
            free.Push frame
            reraise ()

        let value = nextValue + 1UL
        let mutable cbsi = VkCommandBufferSubmitInfo.Create()
        cbsi.commandBuffer <- frame.Command
        let mutable signal = VkSemaphoreSubmitInfo.Create()
        signal.semaphore <- semaphore
        signal.value <- value
        signal.stageMask <- VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        let mutable submit = VkSubmitInfo2.Create()
        submit.commandBufferInfoCount <- 1u
        submit.pCommandBufferInfos <- NativePtr.toNativeInt &&cbsi
        submit.signalSemaphoreInfoCount <- 1u
        submit.pSignalSemaphoreInfos <- NativePtr.toNativeInt &&signal

        try
            c.QueueSubmit2(device.Queue, MemoryMarshal.CreateReadOnlySpan(&submit, 1), VkFence.Null)
        with
        | VulkanException(_, VkResult.VK_ERROR_DEVICE_LOST) ->
            free.Push frame
            raise (x.Lost("vkQueueSubmit2", VkResult.VK_ERROR_DEVICE_LOST))
        | _ ->
            free.Push frame
            reraise ()

        nextValue <- value
        frame.Value <- value
        frame.OnRetire <- onRetire
        inFlight.Enqueue frame
        frame

    /// Resets a frame's pool and records `ops` into its command buffer.
    member private _.Record(frame: Frame, bindings: Bindings option, queries: int, ops: NativeOp list) =
        if frame.Recorded then
            c.ResetCommandPool(dev, frame.Pool, LanguagePrimitives.EnumOfValue 0u)
            frame.Recorded <- false

        if queries > 0 then
            frame.EnsureQueries(
                c,
                dev,
                queries,
                fun n -> name VkObjectType.VK_OBJECT_TYPE_QUERY_POOL frame.Queries.Handle n
            )

        let cmd = frame.Command
        let mutable begin' = VkCommandBufferBeginInfo.Create()
        begin'.flags <- VkCommandBufferUsageFlagBits.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        c.BeginCommandBuffer(cmd, &begin')
        frame.Recorded <- true

        let buffer (t: Target) : NativeBuffer =
            match t with
            | Target.Upload i -> uploads[i].Buffer
            | Target.Readback i -> readbacks[i].Buffer
            | _ ->
                match bindings with
                | Some b -> b.Buffer t
                | None -> invalidOp (sprintf "%O needs heap bindings" t)

        for op in ops do
            match op with
            | NativeOp.ResetQueries n -> c.CmdResetQueryPool(cmd, frame.Queries, 0u, uint32 n)
            | NativeOp.BindHeap ->
                match bindings |> Option.bind _.Heap with
                | Some h ->
                    let mutable bind = VkBindHeapInfoEXT.Create()
                    bind.heapRange.address <- h.Address
                    bind.heapRange.size <- h.Bytes
                    bind.reservedRangeOffset <- h.ReservedOffset
                    bind.reservedRangeSize <- h.ReservedBytes
                    c.CmdBindResourceHeapEXT(cmd, &bind)
                | None -> invalidOp "a dispatch needs the resource heap"
            | NativeOp.Barrier k ->
                let mutable mb = BarrierKind.barrier k
                let mutable dep = VkDependencyInfo.Create()
                dep.memoryBarrierCount <- 1u
                dep.pMemoryBarriers <- NativePtr.toNativeInt &&mb
                c.CmdPipelineBarrier2(cmd, &dep)
            | NativeOp.Pipeline k ->
                match bindings with
                | Some b -> c.CmdBindPipeline(cmd, VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_COMPUTE, b.Pipeline k)
                | None -> invalidOp "a dispatch needs the heap bindings"
            | NativeOp.PushData words ->
                use p = fixed words
                let mutable push = VkPushDataInfoEXT.Create()
                push.offset <- SpirvBinding.PushDataOffset
                push.data.address <- NativePtr.toNativeInt p
                push.data.size <- unativeint (4 * words.Length)
                c.CmdPushDataEXT(cmd, &push)
            | NativeOp.Dispatch g -> c.CmdDispatch(cmd, uint32 g.X, uint32 g.Y, uint32 g.Z)
            | NativeOp.Copy(d, dOff, s, sOff, n) ->
                let mutable region = VkBufferCopy()
                region.srcOffset <- uint64 sOff
                region.dstOffset <- uint64 dOff
                region.size <- uint64 n
                c.CmdCopyBuffer(cmd, (buffer s).Handle, (buffer d).Handle, MemoryMarshal.CreateReadOnlySpan(&region, 1))
            | NativeOp.Timestamp q ->
                c.CmdWriteTimestamp2(
                    cmd,
                    VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    frame.Queries,
                    uint32 q
                )

        c.EndCommandBuffer(cmd)

    /// Creates a buffer of `bytes` of `kind` with its own memory, chosen by flags among its memoryTypeBits (VulkanHeapRecipe
    /// section 2): VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT for buffers with device addresses, mapped persistently for the
    /// host-visible kinds, and named for debuggers. The first choice of each kind is a gpu.memory_type event.
    member _.CreateBuffer(kind: BufferKind, bytes: int64, label: string) : NativeBuffer =
        let mutable usage2 = VkBufferUsageFlags2CreateInfo.Create()
        usage2.usage <- BufferKind.usage kind
        let mutable ci = VkBufferCreateInfo.Create()
        // VkBufferCreateInfo.usage is ignored when VkBufferUsageFlags2CreateInfo is chained (maintenance5).
        ci.pNext <- NativePtr.toNativeInt &&usage2
        ci.size <- uint64 bytes
        ci.sharingMode <- VkSharingMode.VK_SHARING_MODE_EXCLUSIVE
        let mutable buffer = VkBuffer.Null
        c.CreateBuffer(dev, &ci, &buffer)
        VulkanObjects.created ()

        try
            let mutable req = VkMemoryRequirements()
            c.GetBufferMemoryRequirements(dev, buffer, &req)

            let index =
                match MemoryTypes.choose memoryTypes req.memoryTypeBits kind with
                | Some i -> i
                | None ->
                    invalidOp (
                        sprintf
                            "vulkan: no memory type for %s among 0x%x (%s)"
                            (BufferKind.name kind)
                            req.memoryTypeBits
                            (memoryTypes |> Array.map MemoryTypes.flagsText |> String.concat ", ")
                    )

            if not (choices.ContainsKey kind) then
                choices[kind] <- index

                Telemetry.event
                    "gpu.memory_type"
                    [
                        Attr.str TelemetryKeys.Backend (ExecutionBackend.name ExecutionBackend.Vulkan)
                        Attr.str "kind" (BufferKind.name kind)
                        Attr.int "index" index
                        Attr.int "heap" memoryHeaps[index]
                        Attr.str "flags" (MemoryTypes.flagsText memoryTypes[index])
                    ]

            let mutable allocFlags = VkMemoryAllocateFlagsInfo.Create()
            allocFlags.flags <- VkMemoryAllocateFlagBits.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
            let mutable ai = VkMemoryAllocateInfo.Create()

            if BufferKind.deviceAddress kind then
                ai.pNext <- NativePtr.toNativeInt &&allocFlags

            ai.allocationSize <- req.size
            ai.memoryTypeIndex <- uint32 index
            let mutable memory = VkDeviceMemory.Null
            c.AllocateMemory(dev, &ai, &memory)
            VulkanObjects.created ()

            try
                c.BindBufferMemory(dev, buffer, memory, 0UL)

                let address =
                    if BufferKind.deviceAddress kind then
                        let mutable info = VkBufferDeviceAddressInfo.Create()
                        info.buffer <- buffer
                        c.GetBufferDeviceAddress(dev, &info)
                    else
                        0UL

                let mapped =
                    if BufferKind.mapped kind then
                        let mutable p = 0n
                        c.MapMemory(dev, memory, 0UL, VkConst.VK_WHOLE_SIZE, 0u, &p)
                        p
                    else
                        0n

                name VkObjectType.VK_OBJECT_TYPE_BUFFER buffer.Handle label
                name VkObjectType.VK_OBJECT_TYPE_DEVICE_MEMORY memory.Handle (label + ".memory")

                {
                    Handle = buffer
                    Memory = memory
                    Bytes = bytes
                    Address = address
                    Mapped = mapped
                    MemoryType = index
                    Kind = kind
                    Name = label
                }
            with _ ->
                c.FreeMemory(dev, memory)
                VulkanObjects.destroyed ()
                reraise ()
        with _ ->
            c.DestroyBuffer(dev, buffer)
            VulkanObjects.destroyed ()
            reraise ()

    /// Destroys a buffer and frees its memory (unmapped first). The caller makes sure no pending work uses it.
    member _.DestroyBuffer(b: NativeBuffer) =
        if not b.Handle.IsNull then
            if b.Mapped <> 0n then
                c.UnmapMemory(dev, b.Memory)

            c.DestroyBuffer(dev, b.Handle)
            VulkanObjects.destroyed ()
            c.FreeMemory(dev, b.Memory)
            VulkanObjects.destroyed ()

    /// The two upload staging buffers, each at least `bytes` (at most Transfer.ChunkBytes). Only called with no transfer
    /// in flight (transfers wait for all submitted work first).
    member x.UploadStaging(bytes: int64) : Staging list =
        if uploads.Count = 0 || uploads[0].Bytes < bytes then
            for s in uploads do
                x.DestroyBuffer s.Buffer

            uploads.Clear()

            for i in 0..1 do
                uploads.Add(Staging(x.CreateBuffer(BufferKind.Upload, bytes, sprintf "lrm.vulkan.upload%d" i)))

            BufferMemory.report "upload_staging" (2L * bytes)

        List.ofSeq uploads

    /// The two readback staging buffers, each at least `bytes`.
    member x.ReadbackStaging(bytes: int64) : Staging list =
        if readbacks.Count = 0 || readbacks[0].Bytes < bytes then
            for s in readbacks do
                x.DestroyBuffer s.Buffer

            readbacks.Clear()

            for i in 0..1 do
                readbacks.Add(Staging(x.CreateBuffer(BufferKind.Readback, bytes, sprintf "lrm.vulkan.readback%d" i)))

            BufferMemory.report "readback_staging" (2L * bytes)

        List.ofSeq readbacks

    /// Creates the compute pipeline of one SPIR-V blob (VulkanHeapRecipe section 5): VkPipelineCreateFlags2CreateInfo with
    /// VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT, a NULL layout, and the stage's set 0 binding 0 mapping to the heap
    /// with HEAP_WITH_CONSTANT_OFFSET at stride `stride` (SpirvBinding). The shader module is destroyed once the
    /// pipeline exists. The chain lives in this frame for the call.
    member _.CreatePipeline(kernel: string, code: byte[], stride: uint32) : VkPipeline =
        use pCode = fixed code
        let mutable smci = VkShaderModuleCreateInfo.Create()
        smci.codeSize <- unativeint code.Length
        smci.pCode <- NativePtr.toNativeInt pCode
        let mutable shader = VkShaderModule.Null
        c.CreateShaderModule(dev, &smci, &shader)
        VulkanObjects.created ()

        try
            let mutable mapping = HeapMapping.create stride
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

            match
                c.CreateComputePipelines(
                    dev,
                    VkPipelineCache.Null,
                    MemoryMarshal.CreateReadOnlySpan(&cpci, 1),
                    MemoryMarshal.CreateSpan(&pipeline, 1)
                )
            with
            | VkResult.VK_SUCCESS -> ()
            | r -> raise (VulkanException("vkCreateComputePipelines", r))

            VulkanObjects.created ()
            name VkObjectType.VK_OBJECT_TYPE_PIPELINE pipeline.Handle ("lrm.vulkan.pipeline." + kernel)
            pipeline
        finally
            c.DestroyShaderModule(dev, shader)
            VulkanObjects.destroyed ()

    member _.DestroyPipeline(p: VkPipeline) =
        if not p.IsNull then
            c.DestroyPipeline(dev, p)
            VulkanObjects.destroyed ()

    member x.Dispose() =
        if not disposed then
            try
                try
                    x.WaitIdle()
                with _ ->
                    ()
            finally
                disposed <- true

                while inFlight.Count > 0 do
                    free.Push(inFlight.Dequeue())

                for f in allFrames do
                    f.Destroy(c, dev)

                allFrames.Clear()
                free.Clear()

                for s in uploads do
                    x.DestroyBuffer s.Buffer

                for s in readbacks do
                    x.DestroyBuffer s.Buffer

                uploads.Clear()
                readbacks.Clear()

                if not semaphore.IsNull then
                    c.DestroySemaphore(dev, semaphore)
                    VulkanObjects.destroyed ()
                    semaphore <- VkSemaphore.Null

                (device :> IDisposable).Dispose()
                VulkanObjects.destroyed ()

                match messenger with
                | Some m -> (m :> IDisposable).Dispose()
                | None -> ()

                (instance :> IDisposable).Dispose()
                VulkanObjects.destroyed ()
