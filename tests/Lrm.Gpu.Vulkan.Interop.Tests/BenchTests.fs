/// The bench tier (local, informational): the cost of one generated wrapper call on a fake entry point.
module Lrm.Gpu.Vulkan.Interop.Tests.BenchTests

open System
open System.Diagnostics
open Expecto
open Lrm.Core
open Lrm.Gpu.Vulkan.Interop
open Lrm.Gpu.Vulkan.Interop.Tests.Fakes

let tests =
    testList
        "vulkan interop (bench)"
        [
            testCase "one wrapper call (inref, a no-op entry point)"
            <| fun () ->
                let fake = FakeLoader()
                fake.Add("vkCmdPushDataEXT", PFN_vkCmdPushDataEXT(fun _ _ -> ()))
                let d = VkDeviceCommands(fake.Load)
                let mutable info = VkPushDataInfoEXT.Create()
                let cmd = VkCommandBuffer 1n
                let n = 1_000_000

                let measure () =
                    let clock = Stopwatch.StartNew()

                    for _ in 1..n do
                        d.CmdPushDataEXT(cmd, &info)

                    clock.Elapsed.TotalNanoseconds / float n

                measure () |> ignore
                let ns = min (measure ()) (measure ())

                Telemetry.histogram
                    "gpu.interop.call"
                    ns
                    TelemetryUnits.Nanoseconds
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.str "command" "vkCmdPushDataEXT"
                    ]

                printfn "vkCmdPushDataEXT wrapper on a managed fake: %.1f ns per call" ns
        ]
