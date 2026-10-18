namespace Lrm.Gpu.Vulkan

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop

/// A buffer the encoder names.
[<RequireQualifiedAccess; Struct>]
type Target =
    /// A heap slot: 0 is the tensor table, 1..63 the pages.
    | Page of slot: int
    /// Upload staging buffer i.
    | Upload of upload: int
    /// Readback staging buffer i.
    | Readback of readback: int
    /// The fallback heap's host-written descriptor staging buffer (VulkanHeapRecipe section 2, fallback).
    | HeapStaging
    /// The fallback heap's DEVICE_LOCAL buffer.
    | HeapBuffer

    override t.ToString() =
        match t with
        | Page s -> sprintf "page %d" s
        | Upload i -> sprintf "upload %d" i
        | Readback i -> sprintf "readback %d" i
        | HeapStaging -> "heap staging"
        | HeapBuffer -> "heap buffer"

/// The memory barriers the backend records, each one VkMemoryBarrier2 in one vkCmdPipelineBarrier2
/// (VulkanHeapRecipe section 6; synchronization2).
[<RequireQualifiedAccess>]
type BarrierKind =
    /// First in every list that dispatches or copies: every earlier write in submission order (earlier lists, uploads)
    /// before this list's dispatches and copies. Vulkan gives no implicit memory dependency between submissions, so
    /// this keeps D3D12's behaviour, where one list sees everything the queue ran before it.
    | ListStart
    /// ICommandList.Barrier, the recipe's dispatch barrier (section 6.3): compute writes before compute and copy work.
    | Dispatch
    /// Before a Copy: earlier compute and copy writes before the copy reads and writes (Copy is a full barrier).
    | BeforeCopy
    /// After a copy into pages: the recipe's upload barrier (section 6.4), widened to later copies.
    | AfterCopy
    /// After a readback copy: the recipe's host barrier (section 6.4); the host reads after the timeline wait.
    | ToHost
    /// Before a timed dispatch: an execution dependency on everything earlier, so its timestamps bracket it alone.
    | Serialize
    /// After the fallback heap's descriptor copy: COPY / TRANSFER_WRITE before COMPUTE / RESOURCE_HEAP_READ.
    | HeapUpload

[<RequireQualifiedAccess>]
module BarrierKind =
    let private compute =
        VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT

    let private copy = VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COPY_BIT
    let private host = VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_HOST_BIT

    let private allCommands =
        VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT

    let private storageRead = VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    let private storageWrite = VkAccessFlagBits2.VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    let private transferRead = VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_READ_BIT
    let private transferWrite = VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_WRITE_BIT
    let private hostRead = VkAccessFlagBits2.VK_ACCESS_2_HOST_READ_BIT
    let private heapRead = VkAccessFlagBits2.VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT
    let private noAccess: VkAccessFlags2 = LanguagePrimitives.EnumOfValue 0UL

    /// (srcStageMask, srcAccessMask, dstStageMask, dstAccessMask).
    let masks (k: BarrierKind) : VkPipelineStageFlags2 * VkAccessFlags2 * VkPipelineStageFlags2 * VkAccessFlags2 =
        let everything = storageRead ||| storageWrite ||| transferRead ||| transferWrite

        match k with
        | BarrierKind.ListStart -> compute ||| copy, storageWrite ||| transferWrite, compute ||| copy, everything
        | BarrierKind.Dispatch -> compute, storageWrite, compute ||| copy, storageRead ||| storageWrite ||| transferRead
        | BarrierKind.BeforeCopy ->
            compute ||| copy, storageWrite ||| transferWrite, copy, transferRead ||| transferWrite
        | BarrierKind.AfterCopy -> copy, transferWrite, compute ||| copy, everything
        | BarrierKind.ToHost -> copy, transferWrite, host, hostRead
        | BarrierKind.Serialize -> allCommands, noAccess, allCommands, noAccess
        | BarrierKind.HeapUpload -> copy, transferWrite, compute, heapRead

    /// The VkMemoryBarrier2 of a kind.
    let barrier (k: BarrierKind) : VkMemoryBarrier2 =
        let srcStage, srcAccess, dstStage, dstAccess = masks k
        let mutable b = VkMemoryBarrier2.Create()
        b.srcStageMask <- srcStage
        b.srcAccessMask <- srcAccess
        b.dstStageMask <- dstStage
        b.dstAccessMask <- dstAccess
        b

/// One native command-buffer call, in recording order.
[<RequireQualifiedAccess>]
type NativeOp =
    /// vkCmdResetQueryPool(0, count) on the frame's timestamp pool: first in a list that writes timestamps.
    | ResetQueries of count: int
    /// vkCmdBindResourceHeapEXT with the heap's VkBindHeapInfoEXT: once per command buffer, before its first dispatch.
    | BindHeap
    /// vkCmdPipelineBarrier2 with one VkMemoryBarrier2.
    | Barrier of BarrierKind
    /// vkCmdBindPipeline(COMPUTE) with the kernel's heap pipeline.
    | Pipeline of kernel: KernelId
    /// vkCmdPushDataEXT { offset 0, the 32 words (128 bytes) }.
    | PushData of words: uint32[]
    /// vkCmdDispatch(x, y, z) of the physical grid.
    | Dispatch of grid: PhysicalGrid
    /// vkCmdCopyBuffer(source, destination, { sourceOffset, destinationOffset, bytes }).
    | Copy of destination: Target * destinationOffset: int64 * source: Target * sourceOffset: int64 * bytes: int64
    /// vkCmdWriteTimestamp2(ALL_COMMANDS, the frame's pool, query).
    | Timestamp of query: int

/// A command as an ICommandList recorded it: checked when recorded (Dispatch.prepare, Dispatch.checkCopy, the timer
/// rules), encoded at Submit.
[<RequireQualifiedAccess>]
type Command =
    | Dispatch of PreparedDispatch
    | Barrier
    | Copy of source: BufferRange * destination: BufferRange
    | BeginTimer of timer: int * name: string * attrs: Attr list
    | EndTimer of timer: int

/// A timed region of an encoded list: the two timestamp queries that bracket it.
type TimerSlot =
    {
        Name: string
        Attrs: Attr list
        Begin: int
        End: int
    }

/// An encoded command list.
type Encoded =
    {
        Ops: NativeOp list
        /// Timestamp queries the list writes (0..Queries-1), reset at its start.
        Queries: int
        /// Timers in the order they were opened.
        Timers: TimerSlot list
        /// gpu.dispatch timings, one per dispatch when dispatches are timed, in dispatch order.
        DispatchTimers: TimerSlot list
        Dispatches: int
        /// The byte count of every Copy command, in order (gpu.bytes_moved, direction copy).
        Copies: int64 list
    }

    /// True when the list dispatches kernels, so it binds the resource heap first.
    member e.Dispatching = e.Dispatches > 0

/// The command encoder: a pure function from recorded commands to native calls (VulkanHeapRecipe section 6). The cpu
/// tier checks it with Plan.verify; the backend replays the ops into a VkCommandBuffer.
[<RequireQualifiedAccess>]
module Plan =
    /// The gpu.dispatch attributes of each catalog kernel, one list per kernel shared by all its timed dispatches.
    let private dispatchAttrs: Attr list[] =
        Kernels.all |> Array.map (fun k -> [ Attr.str TelemetryKeys.Kernel k.Name ])

    let private attrsOf (d: PreparedDispatch) =
        let i = d.Kernel.Value

        if i >= 0 && i < dispatchAttrs.Length then
            dispatchAttrs[i]
        else
            [ Attr.str TelemetryKeys.Kernel d.Spec.Name ]

    /// Encodes one command list:
    /// - a list that dispatches or copies starts with the ListStart barrier, and a dispatching list then binds the
    ///   resource heap once (vkCmdBindResourceHeapEXT), before its first dispatch;
    /// - a dispatch binds the pipeline when the kernel changes, pushes the whole 128-byte ArgBlock at offset 0 (after
    ///   every pipeline bind, since each dispatch pushes its own) and dispatches the physical grid;
    /// - Barrier is the recipe's dispatch barrier (a global memory barrier between compute stages);
    /// - Copy is a full barrier on both sides: BeforeCopy, vkCmdCopyBuffer, AfterCopy. A copy within one page is one
    ///   vkCmdCopyBuffer with the same source and destination buffer (checkCopy refuses overlapping ranges);
    /// - timers and (with timeDispatches) every dispatch are bracketed by ALL_COMMANDS timestamps; a timed dispatch is
    ///   preceded by a Serialize barrier, so its interval covers that dispatch alone;
    /// - a list that writes timestamps resets its queries first.
    let encode (timeDispatches: bool) (commands: Command list) : Encoded =
        let ops = ResizeArray<NativeOp>()
        let mutable queries = 0
        let mutable pipeline: KernelId option = None
        let mutable heapBound = false
        let opened = Dictionary<int, string * Attr list * int>()
        let timers = SortedDictionary<int, TimerSlot>()
        let dispatchTimers = ResizeArray<TimerSlot>()
        let copies = ResizeArray<int64>()
        let mutable dispatches = 0

        let works =
            commands
            |> List.exists (function
                | Command.Dispatch _
                | Command.Copy _ -> true
                | _ -> false)

        if works then
            ops.Add(NativeOp.Barrier BarrierKind.ListStart)

        let query () =
            let q = queries
            queries <- queries + 1
            ops.Add(NativeOp.Timestamp q)
            q

        for c in commands do
            match c with
            | Command.Dispatch d ->
                if not heapBound then
                    ops.Add NativeOp.BindHeap
                    heapBound <- true

                if pipeline <> Some d.Kernel then
                    ops.Add(NativeOp.Pipeline d.Kernel)
                    pipeline <- Some d.Kernel

                ops.Add(NativeOp.PushData(d.Args.ToWords()))

                if timeDispatches then
                    ops.Add(NativeOp.Barrier BarrierKind.Serialize)
                    let q0 = query ()
                    ops.Add(NativeOp.Dispatch d.Physical)
                    let q1 = query ()

                    dispatchTimers.Add
                        {
                            Name = GpuTelemetry.Dispatch
                            Attrs = attrsOf d
                            Begin = q0
                            End = q1
                        }
                else
                    ops.Add(NativeOp.Dispatch d.Physical)

                dispatches <- dispatches + 1
            | Command.Barrier -> ops.Add(NativeOp.Barrier BarrierKind.Dispatch)
            | Command.Copy(src, dst) ->
                copies.Add src.Bytes
                ops.Add(NativeOp.Barrier BarrierKind.BeforeCopy)
                ops.Add(NativeOp.Copy(Target.Page dst.Slot, dst.Offset, Target.Page src.Slot, src.Offset, src.Bytes))
                ops.Add(NativeOp.Barrier BarrierKind.AfterCopy)
            | Command.BeginTimer(id, name, attrs) -> opened[id] <- (name, attrs, query ())
            | Command.EndTimer id ->
                let name, attrs, q0 = opened[id]
                let q1 = query ()

                timers[id] <-
                    {
                        Name = name
                        Attrs = attrs
                        Begin = q0
                        End = q1
                    }

        if queries > 0 then
            ops.Insert(0, NativeOp.ResetQueries queries)

        {
            Ops = List.ofSeq ops
            Queries = queries
            Timers = List.ofSeq timers.Values
            DispatchTimers = List.ofSeq dispatchTimers
            Dispatches = dispatches
            Copies = List.ofSeq copies
        }

    /// An upload chunk: staging buffer `staging` (from byte 0) into `slot` at `offset`, ordered after everything
    /// earlier on the queue and before later dispatches and copies (the recipe's upload barrier).
    let upload (slot: int) (offset: int64) (staging: int) (bytes: int64) : NativeOp list =
        [
            NativeOp.Barrier BarrierKind.ListStart
            NativeOp.Copy(Target.Page slot, offset, Target.Upload staging, 0L, bytes)
            NativeOp.Barrier BarrierKind.AfterCopy
        ]

    /// A download chunk: `slot` at `offset` into readback staging buffer `staging` (from byte 0), then the recipe's
    /// host barrier.
    let download (slot: int) (offset: int64) (staging: int) (bytes: int64) : NativeOp list =
        [
            NativeOp.Barrier BarrierKind.ListStart
            NativeOp.Copy(Target.Readback staging, 0L, Target.Page slot, offset, bytes)
            NativeOp.Barrier BarrierKind.ToHost
        ]

    /// The fallback heap's fill: the host-written descriptors (from byte 0 of the staging buffer) into the heap buffer
    /// at `offset`, then COPY / TRANSFER_WRITE before COMPUTE / RESOURCE_HEAP_READ (VulkanHeapRecipe section 2).
    let heapUpload (offset: int64) (bytes: int64) : NativeOp list =
        [
            NativeOp.Copy(Target.HeapBuffer, offset, Target.HeapStaging, 0L, bytes)
            NativeOp.Barrier BarrierKind.HeapUpload
        ]

    let private has (flags: 'T) (bit: 'T) =
        let f = LanguagePrimitives.EnumToValue flags
        let b = LanguagePrimitives.EnumToValue bit
        (f &&& b) = b

    /// Checks an op list against the rules it must keep (`dispatches`: the dispatches expected):
    /// - ResetQueries comes first when the list writes timestamps, once, and every query below its count is written
    ///   at most once and only after it;
    /// - BindHeap comes once, before the first dispatch, and never in a list without dispatches;
    /// - every dispatch follows a pipeline bind and a 32-word PushData recorded after that bind and after the previous
    ///   dispatch;
    /// - a list that dispatches or copies a page starts with the ListStart barrier;
    /// - every copy is preceded by a barrier whose second scope holds COPY with TRANSFER_READ and TRANSFER_WRITE (or by
    ///   another copy of the same group), and the group is followed by a barrier whose first scope holds COPY with
    ///   TRANSFER_WRITE;
    /// - copies move a positive size, never from a readback or heap buffer or into an upload or heap staging buffer,
    ///   and never between overlapping ranges of one buffer.
    /// Error names the first broken rule.
    let verify (dispatches: int) (ops: NativeOp list) : Result<unit, string> =
        let a = Array.ofList ops
        let mutable error: string option = None

        let fail (s: string) =
            if error.IsNone then
                error <- Some s

        let resets =
            a
            |> Array.indexed
            |> Array.choose (fun (i, op) ->
                match op with
                | NativeOp.ResetQueries n -> Some(i, n)
                | _ -> None)

        let stamps =
            a
            |> Array.choose (function
                | NativeOp.Timestamp q -> Some q
                | _ -> None)

        match resets with
        | [||] when stamps.Length > 0 -> fail "the list writes timestamps but never resets its queries"
        | [||] -> ()
        | [| 0, n |] ->
            if n <> stamps.Length then
                fail (sprintf "%d queries are reset but %d timestamps are written" n stamps.Length)

            if stamps |> Array.exists (fun q -> q < 0 || q >= n) then
                fail "a timestamp query is outside the reset range"

            if (Array.distinct stamps).Length <> stamps.Length then
                fail "a timestamp query is written twice"
        | [| i, _ |] -> fail (sprintf "op %d: ResetQueries must be the first op" i)
        | _ -> fail "ResetQueries appears more than once"

        let works =
            a
            |> Array.exists (function
                | NativeOp.Dispatch _ -> true
                | NativeOp.Copy(Target.Page _, _, _, _, _)
                | NativeOp.Copy(_, _, Target.Page _, _, _) -> true
                | _ -> false)

        if works then
            let first =
                a
                |> Array.tryFind (function
                    | NativeOp.ResetQueries _ -> false
                    | _ -> true)

            match first with
            | Some(NativeOp.Barrier BarrierKind.ListStart) -> ()
            | _ -> fail "a list that dispatches or copies pages must start with the ListStart barrier"

        let mutable bound = false
        let mutable pipeline = false
        let mutable pushed = false
        let mutable seen = 0

        for i in 0 .. a.Length - 1 do
            match a[i] with
            | NativeOp.BindHeap ->
                if bound then
                    fail (sprintf "op %d: the heap is bound twice in one command buffer" i)

                if seen > 0 then
                    fail (sprintf "op %d: the heap is bound after a dispatch" i)

                bound <- true
            | NativeOp.Pipeline _ ->
                pipeline <- true
                pushed <- false
            | NativeOp.PushData w ->
                if w.Length <> ArgBlock.WordCount then
                    fail (sprintf "op %d: push data of %d words, not %d" i w.Length ArgBlock.WordCount)

                pushed <- true
            | NativeOp.Dispatch _ ->
                if not bound then
                    fail (sprintf "op %d: dispatch before the heap is bound" i)

                if not pipeline then
                    fail (sprintf "op %d: dispatch without a pipeline" i)

                if not pushed then
                    fail (sprintf "op %d: dispatch without push data since the last pipeline bind or dispatch" i)

                pushed <- false
                seen <- seen + 1
            | NativeOp.Copy(d, dOff, s, sOff, n) ->
                if n <= 0L then
                    fail (sprintf "op %d: empty copy" i)

                match s with
                | Target.Readback _
                | Target.HeapBuffer -> fail (sprintf "op %d: %O is never a copy source" i s)
                | _ -> ()

                match d with
                | Target.Upload _
                | Target.HeapStaging -> fail (sprintf "op %d: %O is never a copy destination" i d)
                | _ -> ()

                if obj.Equals(s, d) && sOff < dOff + n && dOff < sOff + n then
                    fail (sprintf "op %d: %O is copied onto an overlapping range of itself" i s)

                let rec before j =
                    if j < 0 then
                        None
                    else
                        match a[j] with
                        | NativeOp.Copy _ -> before (j - 1)
                        | op -> Some op

                let rec after j =
                    if j >= a.Length then
                        None
                    else
                        match a[j] with
                        | NativeOp.Copy _ -> after (j + 1)
                        | op -> Some op

                match before (i - 1) with
                | Some(NativeOp.Barrier k) ->
                    let _, _, dstStage, dstAccess = BarrierKind.masks k

                    if
                        not (has dstStage VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COPY_BIT)
                        || not (has dstAccess VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_READ_BIT)
                        || not (has dstAccess VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_WRITE_BIT)
                    then
                        fail (sprintf "op %d: the barrier before the copy (%A) does not cover it" i k)
                | None when d = Target.HeapBuffer -> () // the fallback heap's first and only fill
                | _ -> fail (sprintf "op %d: a copy without a barrier before it" i)

                match after (i + 1) with
                | Some(NativeOp.Barrier k) ->
                    let srcStage, srcAccess, _, _ = BarrierKind.masks k

                    if
                        not (has srcStage VkPipelineStageFlagBits2.VK_PIPELINE_STAGE_2_COPY_BIT)
                        || not (has srcAccess VkAccessFlagBits2.VK_ACCESS_2_TRANSFER_WRITE_BIT)
                    then
                        fail (sprintf "op %d: the barrier after the copy (%A) does not make it visible" i k)
                | _ -> fail (sprintf "op %d: a copy without a barrier after it" i)
            | NativeOp.Barrier _
            | NativeOp.ResetQueries _
            | NativeOp.Timestamp _ -> ()

        if bound && seen = 0 then
            fail "the heap is bound in a list without dispatches"

        if seen <> dispatches then
            fail (sprintf "%d dispatches expected, %d found" dispatches seen)

        match error with
        | None -> Ok()
        | Some e -> Error e

/// The heap pipeline's set/binding → heap mapping (VulkanHeapRecipe section 5; its bytes are KAT-1): set 0, binding 0,
/// one binding, READ_ONLY | READ_WRITE storage buffers, HEAP_WITH_CONSTANT_OFFSET at heapOffset 0 with heapArrayStride
/// bufferDescriptorSize, no embedded sampler, from R4b's SpirvBinding values.
[<RequireQualifiedAccess>]
module HeapMapping =
    let create (bufferDescriptorSize: uint32) : VkDescriptorSetAndBindingMappingEXT =
        let mutable mapping = VkDescriptorSetAndBindingMappingEXT.Create()
        mapping.descriptorSet <- SpirvBinding.DescriptorSet
        mapping.firstBinding <- SpirvBinding.FirstBinding
        mapping.bindingCount <- SpirvBinding.BindingCount
        mapping.resourceMask <- LanguagePrimitives.EnumOfValue SpirvBinding.ResourceMask
        mapping.source <- LanguagePrimitives.EnumOfValue(int SpirvBinding.MappingSourceHeapWithConstantOffset)
        mapping.sourceData.constantOffset.heapOffset <- SpirvBinding.HeapOffset
        mapping.sourceData.constantOffset.heapArrayStride <- SpirvBinding.heapArrayStride bufferDescriptorSize
        mapping
