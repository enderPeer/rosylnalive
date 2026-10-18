/// The bench tier (local, informational): the cost of generated wrapper calls into the real driver, recorded into a
/// command buffer that is never submitted.
module Lrm.Gpu.Vulkan.Interop.Tests.BenchTests

#nowarn "9"
#nowarn "51"

open System
open System.Diagnostics
open Expecto
open Lrm.Core
open Lrm.Gpu.Vulkan.Interop
open Lrm.Gpu.Vulkan.Interop.Tests.Common

let tests =
    testList
        "vulkan interop (bench)"
        [
            testCase "wrapper calls into the driver: vkCmdPushDataEXT and vkCmdPipelineBarrier2 (inref)"
            <| fun () ->
                use instance = createInstance false
                let physical, _ = accepted instance
                use device = Device.create instance physical DeviceOptions.recipe
                let d = device.Commands
                let args = Hello.kat4
                let n = 20_000

                let perCall (record: VkCommandBuffer -> unit) =
                    Hello.withRecording device (fun cmd ->
                        record cmd // warm up
                        let clock = Stopwatch.StartNew()
                        record cmd
                        clock.Elapsed.TotalNanoseconds / float n)

                let push (cmd: VkCommandBuffer) =
                    use p = fixed args
                    let mutable info = VkPushDataInfoEXT.Create()
                    info.data.address <- Microsoft.FSharp.NativeInterop.NativePtr.toNativeInt p
                    info.data.size <- unativeint 128

                    for _ in 1..n do
                        d.CmdPushDataEXT(cmd, &info)

                // A dispatch needs a bound pipeline and heap (recording one without them is invalid usage, which
                // the driver may not survive), so the second command is the barrier every dispatch is followed by.
                let barrier (cmd: VkCommandBuffer) =
                    let mutable mb = VkMemoryBarrier2.Create()
                    let mutable dep = VkDependencyInfo.Create()
                    dep.memoryBarrierCount <- 1u
                    dep.pMemoryBarriers <- Microsoft.FSharp.NativeInterop.NativePtr.toNativeInt &&mb

                    for _ in 1..n do
                        d.CmdPipelineBarrier2(cmd, &dep)

                for (name, record) in [ "vkCmdPushDataEXT", push; "vkCmdPipelineBarrier2", barrier ] do
                    let ns = min (perCall record) (perCall record)

                    Telemetry.histogram
                        "gpu.interop.call"
                        ns
                        TelemetryUnits.Nanoseconds
                        [ Attr.str TelemetryKeys.Backend "vulkan"; Attr.str "command" name ]

                    printfn "%s through the generated wrapper into the driver: %.1f ns per call" name ns
        ]
