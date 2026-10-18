/// cpu tier: the command encoder and its rules (VulkanHeapRecipe section 6), the barrier masks, memory types chosen by
/// flags, the device mapping and rejection texts, the heap geometry and mapping known answers (KAT-1, KAT-2), the
/// selftests against D1's bytes, KernelBounds and CpuRef, the SPIR-V guard, the device-lost text, transfer chunking,
/// the hello arguments and the tail-call scan. No GPU is touched.
module Lrm.Gpu.Vulkan.Tests.CpuTests

#nowarn "9"
#nowarn "51"

open System
open System.Reflection
open System.Runtime.InteropServices
open Expecto
open Lrm.Gpu.Vulkan.Interop
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan
open Lrm.Gpu.Vulkan.Tests.Common
open Lrm.Tensor
open Lrm.Testing
open Microsoft.FSharp.NativeInterop

/// A recording backend with the suite's heap and table, so dispatches are prepared by the shared rules.
let private recording () =
    let b = new RecordingBackend()
    let ib = b :> IGpuBackend
    let layout, table = SelfTests.Suite.Plan()
    let heap = ib.CreateHeap layout
    heap.WriteTable table
    ib.LoadKernels(shaders ()) |> ignore
    b, ib, table

let private tok (table: SealedTable) name =
    match table.TryFind name with
    | Some t -> t
    | None -> failtestf "no tensor %s" name

let private prepared (table: SealedTable) (kernel: KernelId) (args: ArgBlock) =
    Command.Dispatch(Dispatch.prepare table kernel args)

let private dispatchesOf (cmds: Command list) =
    cmds
    |> List.filter (function
        | Command.Dispatch _ -> true
        | _ -> false)
    |> List.length

let private verifyOk (cmds: Command list) (enc: Encoded) =
    match Plan.verify (dispatchesOf cmds) enc.Ops with
    | Ok() -> ()
    | Error e -> failtestf "the encoding breaks a rule: %s\n%A" e enc.Ops

let private stages (k: BarrierKind) =
    let s, sa, d, da = BarrierKind.masks k
    uint64 s, uint64 sa, uint64 d, uint64 da

let private planTests =
    let _, _, table = recording ()

    let scale =
        prepared
            table
            Kernels.ScaleF32.Id
            (Kernels.ScaleF32.encode
                {
                    X = tok table "scale.x"
                    Y = tok table "scale.y"
                    N = 7u
                    A = 2.0f
                })

    let groups =
        prepared
            table
            Kernels.GroupsEcho.Id
            (Kernels.GroupsEcho.encode
                {
                    Y = tok table "groups_echo.y.folded"
                    Groups = uint32 SelfTests.FoldedGroups
                })

    let range slot offset bytes : BufferRange =
        {
            Slot = slot
            Offset = offset
            Bytes = bytes
        }

    let words (c: Command) =
        match c with
        | Command.Dispatch d -> d.Args.ToWords()
        | _ -> [||]

    testList
        "plan"
        [
            testCase
                "a dispatch list starts with ListStart, binds the heap once, then the pipeline, 128 bytes of push data and the dispatch"
            <| fun () ->
                let enc = Plan.encode false [ scale ]
                verifyOk [ scale ] enc

                Expect.equal
                    enc.Ops
                    [
                        NativeOp.Barrier BarrierKind.ListStart
                        NativeOp.BindHeap
                        NativeOp.Pipeline Kernels.ScaleF32.Id
                        NativeOp.PushData(words scale)
                        NativeOp.Dispatch { X = 1; Y = 1; Z = 1 }
                    ]
                    "ops"

                Expect.equal (words scale).Length 32 "the whole ArgBlock"
                Expect.equal enc.Queries 0 "no queries"
                Expect.equal enc.Dispatches 1 "one dispatch"

            testCase "word 0 is the physical x: a folded grid of 131,077 groups is 43,693 x 3"
            <| fun () ->
                let enc = Plan.encode false [ groups ]

                let dispatched =
                    enc.Ops
                    |> List.choose (function
                        | NativeOp.Dispatch g -> Some g
                        | _ -> None)

                Expect.equal dispatched [ { X = 43693; Y = 3; Z = 1 } ] "folded grid"

                let pushed =
                    enc.Ops
                    |> List.pick (function
                        | NativeOp.PushData w -> Some w
                        | _ -> None)

                Expect.equal pushed[0] 43693u "lrm_groups_x"

            testCase
                "the pipeline is bound once for consecutive dispatches of one kernel, and every dispatch pushes its block"
            <| fun () ->
                let cmds = [ scale; scale; groups; scale ]
                let enc = Plan.encode false cmds
                verifyOk cmds enc

                let count f = enc.Ops |> List.filter f |> List.length

                Expect.equal
                    (count (function
                        | NativeOp.Pipeline _ -> true
                        | _ -> false))
                    3
                    "pipeline binds on kernel changes"

                Expect.equal
                    (count (function
                        | NativeOp.PushData _ -> true
                        | _ -> false))
                    4
                    "push data per dispatch"

                Expect.equal (count ((=) NativeOp.BindHeap)) 1 "one heap bind per command buffer"

            testCase "Barrier is the recipe's dispatch barrier: compute writes before compute and copy reads"
            <| fun () ->
                let enc = Plan.encode false [ scale; Command.Barrier; scale ]
                verifyOk [ scale; Command.Barrier; scale ] enc
                Expect.contains enc.Ops (NativeOp.Barrier BarrierKind.Dispatch) "the barrier"
                // src COMPUTE_SHADER / SHADER_STORAGE_WRITE; dst COMPUTE_SHADER | COPY /
                // SHADER_STORAGE_READ | SHADER_STORAGE_WRITE | TRANSFER_READ (VulkanHeapRecipe section 6.3).
                Expect.equal
                    (stages BarrierKind.Dispatch)
                    (0x800UL, 0x400000000UL, 0x100000800UL, 0x600000800UL)
                    "masks"

            testCase
                "a copy is a full barrier: BeforeCopy, one vkCmdCopyBuffer, AfterCopy; within a page it is one buffer"
            <| fun () ->
                let across = Command.Copy(range 1 0L 256L, range 2 1024L 256L)
                let within = Command.Copy(range 3 0L 512L, range 3 4096L 512L)
                let enc = Plan.encode false [ across; within ]
                verifyOk [ across; within ] enc

                Expect.equal
                    enc.Ops
                    [
                        NativeOp.Barrier BarrierKind.ListStart
                        NativeOp.Barrier BarrierKind.BeforeCopy
                        NativeOp.Copy(Target.Page 2, 1024L, Target.Page 1, 0L, 256L)
                        NativeOp.Barrier BarrierKind.AfterCopy
                        NativeOp.Barrier BarrierKind.BeforeCopy
                        NativeOp.Copy(Target.Page 3, 4096L, Target.Page 3, 0L, 512L)
                        NativeOp.Barrier BarrierKind.AfterCopy
                    ]
                    "ops"

                Expect.equal enc.Copies [ 256L; 512L ] "copy bytes for gpu.bytes_moved"

                Expect.equal
                    (stages BarrierKind.BeforeCopy)
                    (0x100000800UL, 0x400001000UL, 0x100000000UL, 0x1800UL)
                    "before: compute and copy writes before the copy's reads and writes"

                Expect.equal
                    (stages BarrierKind.AfterCopy)
                    (0x100000000UL, 0x1000UL, 0x100000800UL, 0x600001800UL)
                    "after: the copy's writes before later compute and copy work"

            testCase "timers and timed dispatches are bracketed by ALL_COMMANDS timestamps after a Serialize barrier"
            <| fun () ->
                let cmds =
                    [ Command.BeginTimer(0, "gpu.outer", []); scale; Command.EndTimer 0; groups ]

                let enc = Plan.encode true cmds
                verifyOk cmds enc
                Expect.equal enc.Queries 6 "two timer queries and two per dispatch"
                Expect.equal enc.Ops.Head (NativeOp.ResetQueries 6) "the queries are reset first"

                Expect.equal
                    (enc.Timers |> List.map (fun t -> t.Name, t.Begin, t.End))
                    [ "gpu.outer", 0, 3 ]
                    "the timer brackets the timed dispatch"

                Expect.equal
                    (enc.DispatchTimers |> List.map (fun t -> t.Begin, t.End))
                    [ 1, 2; 4, 5 ]
                    "dispatch timings"

                let serialized =
                    enc.Ops
                    |> List.pairwise
                    |> List.filter (fun (a, b) ->
                        a = NativeOp.Barrier BarrierKind.Serialize
                        && (match b with
                            | NativeOp.Timestamp _ -> true
                            | _ -> false))
                    |> List.length

                Expect.equal serialized 2 "each timed dispatch runs alone"
                Expect.equal (stages BarrierKind.Serialize) (0x10000UL, 0UL, 0x10000UL, 0UL) "an execution dependency"

                let untimed = Plan.encode false cmds
                Expect.equal untimed.Queries 2 "untimed: the timer only"
                Expect.isEmpty untimed.DispatchTimers "no dispatch timings"

            testCase "an empty list encodes to nothing"
            <| fun () ->
                let enc = Plan.encode true []
                Expect.isEmpty enc.Ops "no ops"
                verifyOk [] enc

            testCase "uploads and downloads: ListStart, the copy, then the upload barrier or the host barrier"
            <| fun () ->
                Expect.equal
                    (Plan.upload 3 64L 1 4096L)
                    [
                        NativeOp.Barrier BarrierKind.ListStart
                        NativeOp.Copy(Target.Page 3, 64L, Target.Upload 1, 0L, 4096L)
                        NativeOp.Barrier BarrierKind.AfterCopy
                    ]
                    "upload"

                Expect.equal
                    (Plan.download 2 128L 0 64L)
                    [
                        NativeOp.Barrier BarrierKind.ListStart
                        NativeOp.Copy(Target.Readback 0, 0L, Target.Page 2, 128L, 64L)
                        NativeOp.Barrier BarrierKind.ToHost
                    ]
                    "download"

                // COPY / TRANSFER_WRITE before HOST / HOST_READ (VulkanHeapRecipe section 6.4).
                Expect.equal (stages BarrierKind.ToHost) (0x100000000UL, 0x1000UL, 0x4000UL, 0x2000UL) "host barrier"

                for ops in [ Plan.upload 3 64L 1 4096L; Plan.download 2 128L 0 64L ] do
                    match Plan.verify 0 ops with
                    | Ok() -> ()
                    | Error e -> failtestf "%s" e

            testCase "the fallback heap fill ends with COPY to RESOURCE_HEAP_READ"
            <| fun () ->
                let ops = Plan.heapUpload 64L 1024L

                Expect.equal
                    ops
                    [
                        NativeOp.Copy(Target.HeapBuffer, 64L, Target.HeapStaging, 0L, 1024L)
                        NativeOp.Barrier BarrierKind.HeapUpload
                    ]
                    "ops"

                // VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT is 0x0400000000000000 (VulkanHeapRecipe section 2).
                Expect.equal
                    (stages BarrierKind.HeapUpload)
                    (0x100000000UL, 0x1000UL, 0x800UL, 0x0400000000000000UL)
                    "masks"

                match Plan.verify 0 ops with
                | Ok() -> ()
                | Error e -> failtestf "%s" e

            testCase "verify catches broken encodings"
            <| fun () ->
                let d = NativeOp.Dispatch { X = 1; Y = 1; Z = 1 }
                let push = NativeOp.PushData(Array.zeroCreate 32)
                let p = NativeOp.Pipeline Kernels.ScaleF32.Id
                let start = NativeOp.Barrier BarrierKind.ListStart
                let copy = NativeOp.Copy(Target.Page 2, 0L, Target.Page 1, 0L, 4L)

                let bad =
                    [
                        [ start; p; push; d ], 1, "before the heap is bound"
                        [ start; NativeOp.BindHeap; push; d ], 1, "without a pipeline"
                        [ start; NativeOp.BindHeap; p; d ], 1, "without push data"
                        [ start; NativeOp.BindHeap; p; push; d; d ], 2, "without push data"
                        [ start; NativeOp.BindHeap; p; NativeOp.PushData [| 1u |]; d ], 1, "push data of 1 words"
                        [ start; NativeOp.BindHeap; p; push; d; NativeOp.BindHeap ], 1, "bound twice"
                        [ NativeOp.BindHeap; p; push; d ], 1, "must start with the ListStart"
                        [ start; NativeOp.BindHeap ], 0, "without dispatches"
                        [ start; NativeOp.BindHeap; p; push; d ], 2, "2 dispatches expected"
                        [ NativeOp.Timestamp 0 ], 0, "never resets"
                        [
                            NativeOp.Barrier BarrierKind.Dispatch
                            NativeOp.ResetQueries 1
                            NativeOp.Timestamp 0
                        ],
                        0,
                        "must be the first op"
                        [ NativeOp.ResetQueries 1; NativeOp.Timestamp 0; NativeOp.Timestamp 0 ],
                        0,
                        "queries are reset but"
                        [ start; copy ], 0, "without a barrier after it"
                        [ start; copy; NativeOp.Barrier BarrierKind.Dispatch ], 0, "does not make it visible"
                        [
                            start
                            NativeOp.Barrier BarrierKind.Dispatch
                            copy
                            NativeOp.Barrier BarrierKind.AfterCopy
                        ],
                        0,
                        "does not cover it"
                        [
                            start
                            NativeOp.Barrier BarrierKind.BeforeCopy
                            NativeOp.Copy(Target.Upload 0, 0L, Target.Page 1, 0L, 4L)
                            NativeOp.Barrier BarrierKind.AfterCopy
                        ],
                        0,
                        "never a copy destination"
                        [
                            start
                            NativeOp.Barrier BarrierKind.BeforeCopy
                            NativeOp.Copy(Target.Page 1, 4L, Target.Page 1, 0L, 8L)
                            NativeOp.Barrier BarrierKind.AfterCopy
                        ],
                        0,
                        "overlapping range of itself"
                    ]

                for ops, n, text in bad do
                    match Plan.verify n ops with
                    | Ok() -> failtestf "accepted %A" ops
                    | Error e -> Expect.stringContains e text (sprintf "the broken rule of %A" ops)

            Props.testProperty "random command lists keep every rule" (fun (picks: (byte * byte * byte) list) ->
                let cmds =
                    picks
                    |> List.truncate 40
                    |> List.map (fun (a, b, c) ->
                        match int a % 6 with
                        | 0 -> scale
                        | 1 -> groups
                        | 2 -> Command.Barrier
                        | 3 ->
                            let s = 1 + int b % 3
                            let d = 1 + int c % 3
                            Command.Copy(range s 0L 256L, range d 1024L 256L)
                        | 4 -> Command.BeginTimer(int b, "gpu.t", [])
                        | _ -> Command.Copy(range 2 0L 512L, range 3 4096L 512L))

                // Timers in the property are opened and never closed; drop them unless closed in order.
                let cmds =
                    cmds
                    |> List.filter (function
                        | Command.BeginTimer _ -> false
                        | _ -> true)

                let enc = Plan.encode (int (List.length picks) % 2 = 0) cmds

                match Plan.verify (dispatchesOf cmds) enc.Ops with
                | Ok() -> true
                | Error e -> failwith e)
        ]

/// The RTX 4080's memory types under driver 616.92, as the backend reports them (DeviceInfo.Details memory_types).
let private rtx4080Types: VkMemoryPropertyFlags[] =
    let dl = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    let hv = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    let hc = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    let cached = VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_CACHED_BIT

    [|
        LanguagePrimitives.EnumOfValue 0u
        dl
        hv ||| hc
        hv ||| hc ||| cached
        dl ||| hv ||| hc
    |]

let private memoryTests =
    testList
        "memory"
        [
            testCase "the RTX 4080's types give pages 1, the heap 4, upload staging 2 and readback staging 3 (R1's ADR)"
            <| fun () ->
                let all = UInt32.MaxValue
                let pick k = MemoryTypes.choose rtx4080Types all k
                Expect.equal (pick BufferKind.Page) (Some 1) "pages: DEVICE_LOCAL, not HOST_VISIBLE"
                Expect.equal (pick BufferKind.ResourceHeap) (Some 4) "heap: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT"
                Expect.equal (pick BufferKind.Upload) (Some 2) "upload: HOST_VISIBLE | HOST_COHERENT, not cached"
                Expect.equal (pick BufferKind.Readback) (Some 3) "readback: HOST_CACHED"
                Expect.equal (pick BufferKind.ResourceHeapCopied) (Some 1) "the copy path's heap: DEVICE_LOCAL"
                Expect.isTrue (MemoryTypes.hasHeapInPlace rtx4080Types) "the BAR type exists"

            testCase "without a BAR type the heap takes the copy path; memoryTypeBits restricts every choice"
            <| fun () ->
                let noBar = rtx4080Types[0..3]
                Expect.isFalse (MemoryTypes.hasHeapInPlace noBar) "no in-place heap"
                Expect.equal (MemoryTypes.choose noBar UInt32.MaxValue BufferKind.ResourceHeap) None "no BAR type"
                Expect.equal (MemoryTypes.choose noBar UInt32.MaxValue BufferKind.ResourceHeapCopied) (Some 1) "copied"
                Expect.equal (MemoryTypes.choose rtx4080Types 0b10000u BufferKind.Upload) (Some 4) "only type 4 allowed"
                Expect.equal (MemoryTypes.choose rtx4080Types 0b00100u BufferKind.Readback) (Some 2) "uncached fallback"
                Expect.equal (MemoryTypes.choose rtx4080Types 0b00100u BufferKind.Page) None "no device-local type"

            testCase "LAZILY_ALLOCATED and PROTECTED memory is never chosen"
            <| fun () ->
                let lazyDl =
                    VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    ||| VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT

                let protectedDl =
                    VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    ||| VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_PROTECTED_BIT

                Expect.equal (MemoryTypes.choose [| lazyDl; protectedDl |] UInt32.MaxValue BufferKind.Page) None "none"

            testCase "buffer usages follow the recipe (DESCRIPTOR_HEAP_BIT_EXT is 0x10000000)"
            <| fun () ->
                Expect.equal (uint64 (BufferKind.usage BufferKind.Page)) 0x20023UL "STORAGE | SDA | SRC | DST"
                Expect.equal (uint64 (BufferKind.usage BufferKind.ResourceHeap)) 0x10020000UL "DESCRIPTOR_HEAP | SDA"
                Expect.equal (uint64 (BufferKind.usage BufferKind.ResourceHeapCopied)) 0x10020002UL "plus TRANSFER_DST"
                Expect.equal (uint64 (BufferKind.usage BufferKind.Upload)) 0x1UL "TRANSFER_SRC"
                Expect.equal (uint64 (BufferKind.usage BufferKind.Readback)) 0x2UL "TRANSFER_DST"
                Expect.isTrue (BufferKind.deviceAddress BufferKind.Page) "pages have device addresses"
                Expect.isFalse (BufferKind.deviceAddress BufferKind.Upload) "staging has none"
                Expect.equal (MemoryTypes.flagsText rtx4080Types[4]) "device_local|host_visible|host_coherent" "text"
                Expect.equal (MemoryTypes.flagsText rtx4080Types[0]) "none" "no flags"
        ]

/// What the RTX 4080 reports (driver 616.92), as the baseline of the device tests (R2's rule tests use the same).
let private rtx4080 () : DeviceCheck.Capabilities =
    {
        Identity =
            {
                Name = "NVIDIA GeForce RTX 4080"
                Type = VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                ApiVersion = VkVersion.make 0u 1u 4u 351u
                DriverVersion = (616u <<< 22) ||| (92u <<< 14)
                Driver = "616.92"
                VendorId = 0x10DEu
                DeviceId = 0x2704u
                DriverId = enum 4
                DriverName = "NVIDIA"
                DriverInfo = "616.92"
                DeviceUuid = "00112233445566778899aabbccddeeff"
                DriverUuid = ""
            }
        Extensions =
            Map.ofList
                [
                    VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, 1u
                    VkConst.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, 1u
                ]
        DescriptorHeap =
            Some
                {
                    SamplerHeapAlignment = 64UL
                    ResourceHeapAlignment = 64UL
                    MaxSamplerHeapSize = 0UL
                    MaxResourceHeapSize = 33554432UL
                    MinSamplerHeapReservedRange = 0UL
                    MinSamplerHeapReservedRangeWithEmbedded = 0UL
                    MinResourceHeapReservedRange = 96768UL
                    SamplerDescriptorSize = 32UL
                    ImageDescriptorSize = 32UL
                    BufferDescriptorSize = 16UL
                    SamplerDescriptorAlignment = 32UL
                    ImageDescriptorAlignment = 32UL
                    BufferDescriptorAlignment = 8UL
                    MaxPushDataSize = 256UL
                    ImageCaptureReplayOpaqueDataSize = 0UL
                    MaxDescriptorHeapEmbeddedSamplers = 0u
                    SamplerYcbcrConversionCount = 0u
                    SparseDescriptorHeaps = false
                    ProtectedDescriptorHeaps = false
                }
        StorageBufferDescriptorSize = Some 16UL
        Features =
            {
                ShaderStorageBufferArrayDynamicIndexing = true
                BufferDeviceAddress = true
                TimelineSemaphore = true
                RuntimeDescriptorArray = true
                Synchronization2 = true
                Maintenance5 = true
                DescriptorHeap = true
            }
        Optional =
            {
                DescriptorHeapCaptureReplay = true
                ShaderUntypedPointers = true
                CooperativeMatrix = true
                ShaderFloat16 = true
                ShaderInt8 = true
                DeviceFault = true
                DeviceFaultVendorBinary = true
                MemoryBudget = true
                CalibratedTimestamps = true
                ShaderDenormPreserveFloat32 = false
                ShaderDenormFlushToZeroFloat32 = false
            }
        QueueFamily = Some 2u
        TimestampValidBits = 64u
        TimestampPeriod = 1.0f
        MaxStorageBufferRange = UInt32.MaxValue
        MinStorageBufferOffsetAlignment = 16UL
        MaxComputeSharedMemorySize = 49152u
        MaxComputeWorkGroupInvocations = 1024u
        MemoryTypes =
            rtx4080Types
            |> Array.mapi (fun i f ->
                ({
                    Index = i
                    Flags = f
                    HeapIndex = if i = 1 || i = 4 then 0 else 1
                }
                : DeviceCheck.MemoryTypeInfo))
            |> List.ofArray
        MemoryHeaps =
            [
                ({
                    Index = 0
                    Size = 16827547648UL
                    DeviceLocal = true
                    Budget = None
                    Usage = None
                }
                : DeviceCheck.MemoryHeapInfo)
                ({
                    Index = 1
                    Size = 34000000000UL
                    DeviceLocal = false
                    Budget = None
                    Usage = None
                }
                : DeviceCheck.MemoryHeapInfo)
            ]
        CooperativeMatrixShapes = []
        TimeDomains = []
    }

let private report (c: DeviceCheck.Capabilities) : DeviceCheck.DeviceReport =
    {
        Capabilities = c
        Reasons = DeviceCheck.reasons c
    }

let private deviceTests =
    testList
        "devices"
        [
            testCase "an accepted RTX 4080 report becomes the vulkan DeviceInfo of the fill rule"
            <| fun () ->
                let c = PhysicalDevices.candidate 0 (report (rtx4080 ()))
                Expect.isTrue c.Accepted "accepted"
                let i = c.Info
                Expect.equal i.Backend ExecutionBackend.Vulkan "backend"
                Expect.equal i.Driver "616.92" "Driver: driverInfo (GpuClass.vulkanDriver)"
                Expect.equal i.MemoryBytes 16827547648L "the largest device-local heap"
                Expect.equal (PhysicalDevices.uuidOf i) (Some "00112233445566778899aabbccddeeff") "deviceUUID"

                Expect.equal
                    (i.Details |> List.map fst)
                    [
                        "api"
                        "driver_name"
                        "driver_info"
                        "driver_version"
                        "device_uuid"
                        "driver_uuid"
                        "device_type"
                        "queue_family"
                        "timestamp_valid_bits"
                        "timestamp_period_ns"
                        "max_push_data_size"
                        "buffer_descriptor_size"
                        "resource_heap_alignment"
                        "max_resource_heap_size"
                        "min_resource_heap_reserved_range"
                        "min_storage_buffer_offset_alignment"
                        "max_storage_buffer_range"
                        "shader_denorm_preserve_float32"
                        "shader_denorm_flush_to_zero_float32"
                        "device_fault"
                        "memory_budget"
                        "memory_types"
                    ]
                    "detail keys in order"

                Expect.equal
                    (i.Details |> List.find (fst >> (=) "memory_types") |> snd)
                    "0:none 1:device_local 2:host_visible|host_coherent 3:host_visible|host_coherent|host_cached 4:device_local|host_visible|host_coherent"
                    "memory types"

                let cls = GpuClass.fill i (shaders ())
                Expect.equal cls.Backend ExecutionBackend.Vulkan "class backend"
                Expect.equal cls.DeviceId "pci:10de:2704" "DeviceId"
                Expect.equal cls.Driver "616.92" "class driver"

                Expect.equal
                    cls.ShaderSetHash
                    (Some(ShaderSet.setHash ShaderTarget.Spirv (shaders ())))
                    "the SPIR-V ShaderSetHash"

            testCase "rejections use the Requirements texts for the interface's three requirements"
            <| fun () ->
                let c = rtx4080 ()
                let heap = c.DescriptorHeap.Value

                let rejected (x: DeviceCheck.Capabilities) =
                    (PhysicalDevices.candidate 0 (report x)).Rejections

                Expect.contains
                    (rejected { c with DescriptorHeap = None })
                    Requirements.VulkanDescriptorHeap
                    "no descriptor heap"

                Expect.isFalse
                    (rejected { c with DescriptorHeap = None }
                     |> List.contains "VK_EXT_descriptor_heap not supported")
                    "DeviceCheck's text is replaced"

                Expect.contains
                    (rejected
                        { c with
                            DescriptorHeap =
                                Some
                                    { heap with
                                        BufferDescriptorSize = 32UL
                                        BufferDescriptorAlignment = 32UL
                                    }
                            StorageBufferDescriptorSize = Some 32UL
                        })
                    Requirements.VulkanDescriptorSize
                    "32-byte descriptors"

                let small =
                    rejected
                        { c with
                            DescriptorHeap = Some { heap with MaxPushDataSize = 64UL }
                        }

                Expect.contains small Requirements.ArgumentBytes "64 bytes of push data"

                Expect.isFalse
                    (small |> List.exists (fun r -> r.StartsWith "maxPushDataSize"))
                    "DeviceCheck's push text is replaced"

                Expect.contains
                    (rejected
                        { c with
                            DescriptorHeap = Some { heap with MaxPushDataSize = 128UL }
                        })
                    "maxPushDataSize 128 < 256"
                    "DeviceCheck's floor stays (ADR R2-0001)"

                Expect.contains
                    (rejected { c with QueueFamily = None })
                    "no queue family with COMPUTE, without GRAPHICS and with timestamps"
                    "other reasons are DeviceCheck's"

            testCase "the heap geometry of the RTX 4080 limits is KAT-2"
            <| fun () ->
                let g =
                    Heap.vulkanGeometry (PhysicalDevices.heapLimits (rtx4080 ()).DescriptorHeap.Value)

                Expect.equal g.SlotStride 16L "slot i at 16 * i"
                Expect.equal g.SlotAreaBytes 1024L "slot area"
                Expect.equal g.ReservedRangeOffset 1024L "reservedRangeOffset"
                Expect.equal g.ReservedRangeBytes 96768L "reservedRangeSize"
                Expect.equal g.HeapBytes 97792L "heapRange.size"

            testCase "the heap mapping's 96 bytes are KAT-1"
            <| fun () ->
                let mutable m = HeapMapping.create 16u
                let bytes = Array.zeroCreate<byte> 96
                Marshal.Copy(NativePtr.toNativeInt &&m, bytes, 0, 96)

                Expect.equal
                    (Convert.ToHexStringLower bytes)
                    ("5dd99c3b000000000000000000000000000000000000000001000000c0000000"
                     + "0000000000000000000000001000000000000000000000000000000000000000"
                     + "0000000000000000000000000000000000000000000000000000000000000000")
                    "VulkanHeapRecipe KAT-1"

                Expect.equal
                    (Runtime.CompilerServices.Unsafe.SizeOf<VkDescriptorSetAndBindingMappingEXT>())
                    96
                    "96 bytes"
        ]

let private selftestTests =
    testList
        "selftests"
        [
            testCase "V1's selftest inputs, layout and table are D1's bytes, so the words compare across backends"
            <| fun () ->
                Expect.equal SelfTests.Seed Lrm.Gpu.D3D12.SelfTests.Seed "seed"
                Expect.equal SelfTests.scaleInputs Lrm.Gpu.D3D12.SelfTests.scaleInputs "scale_f32 inputs"
                Expect.equal SelfTests.inPlaceInputs Lrm.Gpu.D3D12.SelfTests.inPlaceInputs "in-place inputs"
                Expect.equal SelfTests.argsWords Lrm.Gpu.D3D12.SelfTests.argsWords "args_echo words"
                Expect.equal SelfTests.bf16Inputs Lrm.Gpu.D3D12.SelfTests.bf16Inputs "bf16 inputs"

                Expect.equal
                    (SelfTests.ScaleN, SelfTests.ScaleA, SelfTests.InPlaceN, SelfTests.FoldedGroups, SelfTests.Spare)
                    (Lrm.Gpu.D3D12.SelfTests.ScaleN,
                     Lrm.Gpu.D3D12.SelfTests.ScaleA,
                     Lrm.Gpu.D3D12.SelfTests.InPlaceN,
                     Lrm.Gpu.D3D12.SelfTests.FoldedGroups,
                     Lrm.Gpu.D3D12.SelfTests.Spare)
                    "sizes"

                let vLayout, vTable = SelfTests.Suite.Plan()
                let dLayout, dTable = Lrm.Gpu.D3D12.SelfTests.Suite.Plan()
                Expect.equal vLayout dLayout "layout"
                Expect.equal (vTable.Image.ToArray()) (dTable.Image.ToArray()) "the sealed table image"

                for mode in [ Fp32DenormalMode.Preserve; Fp32DenormalMode.Flush ] do
                    Expect.equal
                        (SelfTests.scaleInputs
                         |> Array.map (SelfTests.scaleExpected mode SelfTests.ScaleA))
                        (Lrm.Gpu.D3D12.SelfTests.scaleInputs
                         |> Array.map (Lrm.Gpu.D3D12.SelfTests.scaleExpected mode Lrm.Gpu.D3D12.SelfTests.ScaleA))
                        (sprintf "expectations in %A" mode)

                for case in 0..8 do
                    for i in [ 0; 1; 100; 4098 ] do
                        Expect.equal
                            (SelfTests.poison true case 1 i (ValueSome 7u))
                            (Lrm.Gpu.D3D12.SelfTests.poison true case 1 i (ValueSome 7u))
                            "poison"

            testCase "scale_f32's expectations are CpuRef's multiply, checked by KernelBounds in each denormal mode"
            <| fun () ->
                for mode in [ Fp32DenormalMode.Preserve; Fp32DenormalMode.Flush ] do
                    let expected =
                        SelfTests.scaleInputs
                        |> Array.map (SelfTests.scaleExpected mode SelfTests.ScaleA)

                    let reference =
                        SelfTests.scaleInputs
                        |> Array.map (fun x ->
                            let f (w: uint32) =
                                if mode = Fp32DenormalMode.Flush then
                                    Fp32Denormals.flushBits w
                                else
                                    w

                            f (
                                BitConverter.SingleToUInt32Bits(
                                    BitConverter.UInt32BitsToSingle(f x) * SelfTests.ScaleA
                                )
                            ))

                    let c =
                        KernelBounds.checkExact
                            "scale_f32"
                            4
                            (ReadOnlySpan(bytesOf expected))
                            (ReadOnlySpan(bytesOf reference))

                    Expect.isTrue c.Passed (sprintf "%A: %g differ" mode c.Measured)

                Expect.equal
                    (Fp32Denormals.probeInputs
                     |> Array.map (SelfTests.scaleExpected Fp32DenormalMode.Flush Fp32Denormals.ProbeScale))
                    (Fp32Denormals.expected Fp32DenormalMode.Flush)
                    "the flush expectation of the probe's inputs is the probe's flush column"

            testCase "bf16_unpack's expectation is CpuRef.widen over every pattern"
            <| fun () ->
                let input = HostTensor.bf16 (Shape.vector SelfTests.Bf16N) SelfTests.bf16Inputs
                let widened: HostTensor<float32> = Lrm.Tensor.CpuReference.CpuRef.widen input

                let reference = widened.Data |> Array.map BitConverter.SingleToUInt32Bits
                let expected = SelfTests.bf16Inputs |> Array.map Bf16.widenBits

                let k =
                    KernelBounds.checkExact
                        "bf16_unpack"
                        4
                        (ReadOnlySpan(bytesOf expected))
                        (ReadOnlySpan(bytesOf reference))

                Expect.isTrue k.Passed (sprintf "%g of 65536 differ" k.Measured)
        ]

let private miscTests =
    testList
        "misc"
        [
            testCase "the SPIR-V guard refuses a module without its final OpFunctionEnd, which the inspector accepts"
            <| fun () ->
                let set = shaders ()

                for b in ShaderSet.blobsFor ShaderTarget.Spirv set do
                    Expect.isNone (SpirvGuard.problem b.Bytes) (b.Kernel + " passes")

                let scale = (ShaderSet.tryBlob "scale_f32" ShaderTarget.Spirv set).Value
                let spec = Kernels.all[(Kernels.tryFind "scale_f32").Value.Value]

                for cut in [ 4; 8 ] do
                    let truncated = scale.Bytes[.. scale.Bytes.Length - 1 - cut]
                    // A finding for R4b: the inspector reads these as valid modules.
                    Expect.isTrue
                        (SpirvInspector.inspect spec truncated).Passed
                        (sprintf "the inspector accepts -%d" cut)

                    Expect.isSome (SpirvGuard.problem truncated) (sprintf "the guard refuses -%d" cut)

                Expect.isSome (SpirvGuard.problem scale.Bytes[.. scale.Bytes.Length - 3]) "not whole words"

            testCase "a device-lost exception carries the call, the result and VK_EXT_device_fault's report"
            <| fun () ->
                let fault =
                    DeviceFaultQuery.Fault
                        {
                            Description = "page fault"
                            Addresses =
                                [
                                    VkDeviceFaultAddressTypeKHR.VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_KHR,
                                    0x1000UL,
                                    0x0FFFUL
                                ]
                            Vendor = [ "mmu", 7UL, 9UL ]
                            VendorBinaryBytes = 32UL
                        }

                let e =
                    VulkanDeviceLostException("vkQueueSubmit2", VkResult.VK_ERROR_DEVICE_LOST, fault)

                Expect.equal e.Call "vkQueueSubmit2" "call"
                Expect.equal e.Result VkResult.VK_ERROR_DEVICE_LOST "result"
                Expect.stringContains e.Message "device fault: page fault" "description"
                Expect.stringContains e.Message "0x0000000000001000" "address"
                Expect.stringContains e.Message "vendor mmu: code 0x0000000000000007" "vendor"
                Expect.stringContains e.Message "vendor binary: 32 bytes" "binary"

                Expect.stringContains
                    (FaultText.format (DeviceFaultQuery.NoFault VkResult.VK_ERROR_UNKNOWN))
                    "no fault"
                    "no fault"

                Expect.stringContains (FaultText.format DeviceFaultQuery.NotEnabled) "not enabled" "not enabled"

            testCase "transfers are cut into chunks of at most 64 MiB, staged in 64 KiB steps"
            <| fun () ->
                Expect.equal (Transfer.chunks 0L) [] "nothing"
                Expect.equal (Transfer.chunks 28L) [ 0L, 28L ] "small"

                Expect.equal
                    (Transfer.chunks (Transfer.ChunkBytes * 2L + 8L))
                    [
                        0L, Transfer.ChunkBytes
                        Transfer.ChunkBytes, Transfer.ChunkBytes
                        2L * Transfer.ChunkBytes, 8L
                    ]
                    "two full chunks and a tail"

                Expect.equal (Transfer.stagingBytes 28L) 65536L "a small transfer stages 64 KiB"
                Expect.equal (Transfer.stagingBytes (1L <<< 40)) Transfer.ChunkBytes "a large one 64 MiB"

            testCase "hello-vulkan arguments"
            <| fun () ->
                Expect.equal (Hello.parseArgs [||]) (Ok None) "none"
                Expect.equal (Hello.parseArgs [| "--index"; "1" |]) (Ok(Some 1)) "index"
                Expect.isError (Hello.parseArgs [| "--index"; "x" |]) "bad index"
                Expect.isError (Hello.parseArgs [| "--json" |]) "unknown"

            testCase "Lrm.Gpu.Vulkan and this exe build without tail calls (ADR R3-0001)"
            <| fun () ->
                let asm = typeof<VulkanBackend>.Assembly

                if TailCalls.builtWithoutOptimization asm then
                    skiptest TailCalls.UnoptimizedBuild

                for a in [ asm; Assembly.GetExecutingAssembly() ] do
                    Expect.isEmpty (TailCalls.scan a.Location) (a.GetName().Name + " has no tail. prefixes")

            testCase "the provider names the vulkan backend"
            <| fun () -> Expect.equal Vulkan.provider.Backend ExecutionBackend.Vulkan "backend"
        ]

let tests =
    testList "vulkan.cpu" [ planTests; memoryTests; deviceTests; selftestTests; miscTests ]
