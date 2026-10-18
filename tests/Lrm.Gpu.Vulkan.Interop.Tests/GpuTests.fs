/// The gpu tier (vulkan mutex): the device check on the real device, the recipe's logical device created with the
/// full feature chain and destroyed cleanly, every dispatch entry loaded, hello-compute twice bit-identically with a
/// negative control, and the diagnostics extensions (debug utils, memory budget, device fault, cooperative-matrix
/// properties, calibrated timestamps).
module Lrm.Gpu.Vulkan.Interop.Tests.GpuTests

#nowarn "9"
#nowarn "51"

open System
open Expecto
open Microsoft.FSharp.NativeInterop
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop
open Lrm.Gpu.Vulkan.Interop.Tests.Common
open Lrm.Testing

/// Creates an instance and the recipe's device on the first accepted physical device, runs `f`, and destroys both.
let private withDevice
    (debugUtils: bool)
    (options: DeviceOptions)
    (f: Instance -> DeviceCheck.DeviceReport -> Device -> 'a)
    =
    use instance = createInstance debugUtils
    let physical, report = accepted instance
    use device = Device.create instance physical options
    f instance report device

let private heapOf (r: DeviceCheck.DeviceReport) =
    match r.Capabilities.DescriptorHeap with
    | Some h -> h
    | None -> failtest "the accepted device has no descriptor-heap limits"

/// Hello-compute on a fresh device, run `run` of a pair; returns the output bytes and emits the dispatch interval.
let private hello (runIndex: int) (dispatch: bool) =
    withDevice false DeviceOptions.recipe (fun _ report device ->
        let c = report.Capabilities

        let r =
            Hello.run device (heapOf report) c.TimestampValidBits c.TimestampPeriod runIndex dispatch

        if dispatch then
            Telemetry.gpuInterval
                GpuTelemetry.Dispatch
                r.GpuStartNs
                r.GpuNs
                [
                    Attr.str TelemetryKeys.Backend "vulkan"
                    Attr.str TelemetryKeys.Kernel "args_echo"
                    Attr.str TelemetryKeys.Case "kat4"
                    Attr.int "run" runIndex
                ]

            GpuTelemetry.dispatches ExecutionBackend.Vulkan 1

        r)

let tests =
    testList
        "vulkan interop (gpu)"
        [
            gpuTest "the RTX 4080 is accepted by the device check"
            <| fun () ->
                recordDevice ()
                use loader = (VulkanLoader.TryOpen()).Value
                let _, reports = DeviceCheck.checkAll loader
                let accepted = reports |> List.filter _.Accepted
                Expect.isNonEmpty accepted "at least one device is accepted"
                let c = accepted.Head.Capabilities
                Expect.stringContains c.Identity.Name "RTX 4080" "the development machine's GPU"

                match c.DescriptorHeap with
                | Some h ->
                    Expect.isGreaterThanOrEqual h.MaxPushDataSize DeviceCheck.RequiredPushDataSize "push data floor"
                    let g = DeviceCheck.geometry h
                    Expect.equal g.HeapRangeSize 97792UL "KAT-2 on the device's own limits"
                | None -> failtest "no descriptor heap"

                Expect.equal c.StorageBufferDescriptorSize (Some 16UL) "vkGetPhysicalDeviceDescriptorSizeEXT"
                Expect.equal c.QueueFamily (Some 2u) "the recipe's queue family"

            gpuTest "the recipe's device: full feature chain, every enabled dispatch entry loaded, destroyed cleanly"
            <| fun () ->
                // Twice: creation and destruction leave nothing behind that a second device would trip over.
                for round in 1..2 do
                    withDevice false DeviceOptions.recipe (fun _ _ device ->
                        Expect.equal
                            device.EnabledExtensions
                            [ VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME ]
                            "the recipe enables one extension"

                        Expect.isEmpty device.MissingEnabledCommands "every enabled device command is non-null"

                        let expected =
                            VkMeta.Commands
                            |> List.filter (fun c ->
                                c.Level = VkCommandLevel.Device && c.Providers |> List.exists device.Provides)

                        let loaded = expected.Length - device.MissingEnabledCommands.Length

                        Telemetry.check
                            "gpu.device.dispatch_table"
                            device.MissingEnabledCommands.IsEmpty
                            [
                                Attr.str TelemetryKeys.Backend "vulkan"
                                Attr.int "round" round
                                Attr.int "loaded" loaded
                                Attr.int "expected" expected.Length
                                Attr.int "bound" VkDeviceCommands.Names.Length
                            ]

                        Expect.isFalse device.Queue.IsNull "the queue")

            gpuTest "the instance table loads every bound instance command of Vulkan 1.4"
            <| fun () ->
                use instance = createInstance true

                let missing =
                    VkMeta.Commands
                    |> List.filter (fun c ->
                        c.Level = VkCommandLevel.Instance
                        && (instance.Commands.Missing |> List.contains c.Name)
                        && c.Providers
                           |> List.exists (function
                               | VkProvider.Core(a, b) -> VkVersion.covers instance.ApiVersion (a, b)
                               | VkProvider.Extension(n, true) -> instance.HasExtension n
                               | VkProvider.Extension _ -> false))

                Expect.isEmpty missing "every instance command whose provider is enabled is loaded"

            gpuTest "recording 3,000 wrapper calls into a real command buffer allocates no managed memory"
            <| fun () ->
                withDevice false DeviceOptions.recipe (fun _ _ device ->
                    let d = device.Commands
                    let args = Hello.kat4
                    use p = fixed args
                    let mutable info = VkPushDataInfoEXT.Create()
                    info.data.address <- NativePtr.toNativeInt p
                    info.data.size <- unativeint 128
                    let mutable mb = VkMemoryBarrier2.Create()
                    let mutable dep = VkDependencyInfo.Create()
                    dep.memoryBarrierCount <- 1u
                    dep.pMemoryBarriers <- NativePtr.toNativeInt &&mb

                    // No dispatch: recording one without a bound pipeline and heap is invalid usage, which the
                    // driver need not survive. Push data and barriers are valid on their own.
                    let record (cmd: VkCommandBuffer) =
                        for _ in 1..1500 do
                            d.CmdPushDataEXT(cmd, &info)
                            d.CmdPipelineBarrier2(cmd, &dep)

                    let allocated =
                        Hello.withRecording device (fun cmd ->
                            record cmd // stubs are generated on first use
                            let before = GC.GetAllocatedBytesForCurrentThread()
                            record cmd
                            GC.GetAllocatedBytesForCurrentThread() - before)

                    Telemetry.check
                        "gpu.interop.allocations"
                        (allocated = 0L)
                        [
                            Attr.str TelemetryKeys.Backend "vulkan"
                            Attr.int64 "bytes" allocated
                            Attr.int "calls" 3000
                        ]

                    Expect.equal allocated 0L "the push-data and barrier wrappers allocate nothing")

            gpuTest "hello-compute: args_echo (KAT-4) twice bit-identically, no poison left"
            <| fun () ->
                let runIndex = ref 0

                let output, _ =
                    Determinism.runTwice "args_echo kat4" (fun () ->
                        runIndex.Value <- runIndex.Value + 1
                        let r = hello runIndex.Value true
                        Expect.equal r.PoisonLeft 0 (sprintf "run %d wrote every output word" runIndex.Value)
                        r.Output)

                let digest = sha256Hex output

                Telemetry.check
                    "gpu.interop.hello"
                    (digest = Hello.Kat4Sha256)
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.str TelemetryKeys.Kernel "args_echo"
                        Attr.str "sha256" digest
                    ]

                Expect.equal (wordsOf output) Hello.kat4 "the output is the ArgBlock, word for word"
                Expect.equal digest Hello.Kat4Sha256 "KAT-4's SHA-256"

            gpuTest "hello-compute negative control: without the dispatch every output word keeps its poison"
            <| fun () ->
                let r = hello 99 false
                Expect.equal r.PoisonLeft 32 "a run that writes nothing fails the poison check"

            gpuTest "debug utils: messenger, object names and labels on a recipe device"
            <| fun () ->
                withDevice true DeviceOptions.recipe (fun instance _ device ->
                    Expect.isTrue
                        (instance.HasExtension VkConst.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
                        "the loader offers VK_EXT_debug_utils"

                    let messages = Collections.Concurrent.ConcurrentQueue<DebugMessage>()

                    use messenger =
                        new DebugMessenger(
                            instance,
                            VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                            ||| VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                            messages.Enqueue
                        )

                    Expect.isFalse messenger.Handle.IsNull "the messenger"

                    DebugUtils.setName
                        device
                        VkObjectType.VK_OBJECT_TYPE_DEVICE
                        (uint64 device.Handle.Handle)
                        "lrm-device"

                    DebugUtils.setName
                        device
                        VkObjectType.VK_OBJECT_TYPE_QUEUE
                        (uint64 device.Queue.Handle)
                        "lrm-queue"

                    DebugUtils.beginQueueLabel device device.Queue "lrm-label" [| 0.2f; 0.4f; 0.6f; 1.0f |]
                    DebugUtils.endQueueLabel device device.Queue
                    Expect.equal messenger.Dropped 0 "no sink raised"

                    Telemetry.check
                        "gpu.debug.utils"
                        true
                        [ Attr.str TelemetryKeys.Backend "vulkan"; Attr.int "messages" messages.Count ])

            gpuTest "diagnostics device: device fault and memory budget enabled; a healthy device reports no fault"
            <| fun () ->
                withDevice false DeviceOptions.diagnostics (fun instance report device ->
                    Expect.contains device.EnabledExtensions VkConst.VK_EXT_DEVICE_FAULT_EXTENSION_NAME "device fault"

                    Expect.contains
                        device.EnabledExtensions
                        VkConst.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
                        "memory budget"

                    Expect.isEmpty device.MissingEnabledCommands "vkGetDeviceFaultInfoEXT and the rest are loaded"

                    // A healthy device has no fault to describe: either an empty report or the driver's "nothing
                    // here" result (NVIDIA 616.92: VK_ERROR_UNKNOWN). Provoking a device loss is out of bounds on a
                    // shared GPU, so the described-fault path is exercised by the cpu tier's layouts only.
                    let outcome =
                        match DeviceFault.query device with
                        | DeviceFaultQuery.Fault f ->
                            Expect.isEmpty f.Addresses "no faulting address on a healthy device"
                            "empty report"
                        | DeviceFaultQuery.NoFault r -> string r
                        | DeviceFaultQuery.NotEnabled -> failtest "VK_EXT_device_fault is enabled, so the query runs"

                    Telemetry.check
                        "gpu.device.fault_query"
                        true
                        [ Attr.str TelemetryKeys.Backend "vulkan"; Attr.str "outcome" outcome ]

                    match instance.MemoryBudget device.PhysicalDevice with
                    | Some(memory, budget) ->
                        let heaps = memory.memoryHeaps.ToArray()
                        let budgets = budget.heapBudget.ToArray()
                        let usage = budget.heapUsage.ToArray()

                        for i in 0 .. int memory.memoryHeapCount - 1 do
                            Expect.isGreaterThan budgets[i] 0UL (sprintf "heap %d has a budget" i)
                            Expect.isLessThanOrEqual usage[i] budgets[i] (sprintf "heap %d usage within budget" i)

                            Telemetry.gauge
                                "gpu.device.memory"
                                (float budgets[i])
                                TelemetryUnits.Bytes
                                [
                                    Attr.str TelemetryKeys.Backend "vulkan"
                                    Attr.str "pool" (sprintf "heap%d" i)
                                    Attr.str "kind" "budget"
                                    Attr.int64 "heap_bytes" (int64 heaps[i].size)
                                ]
                    | None -> failtest "VK_EXT_memory_budget is supported"

                    Expect.isTrue report.Capabilities.Optional.DeviceFault "deviceFault is reported")

            gpuTest "cooperative-matrix properties and calibrated timestamps"
            <| fun () ->
                let options =
                    { DeviceOptions.recipe with
                        Extensions = [ VkConst.VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME ]
                    }

                withDevice false options (fun instance report device ->
                    let shapes = instance.CooperativeMatrixProperties device.PhysicalDevice
                    Expect.isNonEmpty shapes "the RTX 4080 reports cooperative-matrix shapes"

                    for s in shapes do
                        Expect.isGreaterThan s.MSize 0u "M > 0"

                    let domains = instance.CalibrateableTimeDomains device.PhysicalDevice
                    Expect.contains domains VkTimeDomainKHR.VK_TIME_DOMAIN_DEVICE_KHR "the device domain"
                    Expect.isEmpty device.MissingEnabledCommands "vkGetCalibratedTimestampsKHR is loaded"

                    let infos =
                        [|
                            VkTimeDomainKHR.VK_TIME_DOMAIN_DEVICE_KHR
                            domains |> Array.find (fun d -> d <> VkTimeDomainKHR.VK_TIME_DOMAIN_DEVICE_KHR)
                        |]
                        |> Array.map (fun d ->
                            let mutable i = VkCalibratedTimestampInfoKHR.Create()
                            i.timeDomain <- d
                            i)

                    let stamps = Array.zeroCreate<uint64> infos.Length
                    let mutable deviation = 0UL

                    device.Commands.GetCalibratedTimestampsKHR(
                        device.Handle,
                        ReadOnlySpan infos,
                        Span stamps,
                        &deviation
                    )

                    Expect.isGreaterThan stamps[0] 0UL "a device timestamp"
                    Expect.isGreaterThan stamps[1] 0UL "a host timestamp"
                    Expect.isGreaterThan deviation 0UL "a maximum deviation"

                    Telemetry.gauge
                        "gpu.device.calibration_deviation"
                        (float deviation)
                        "ns"
                        [ Attr.str TelemetryKeys.Backend "vulkan" ]

                    Telemetry.check
                        "gpu.device.cooperative_matrix"
                        true
                        [
                            Attr.str TelemetryKeys.Backend "vulkan"
                            Attr.int "shapes" shapes.Length
                            Attr.int "time_domains" domains.Length
                            Attr.bool "denorm_preserve_f32" report.Capabilities.Optional.ShaderDenormPreserveFloat32
                        ])
        ]
