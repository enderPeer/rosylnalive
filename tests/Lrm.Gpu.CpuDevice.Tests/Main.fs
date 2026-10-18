module Lrm.Gpu.CpuDevice.Tests.Main

open System.Reflection
open Expecto
open Lrm.Core
open Lrm.Gpu.CpuDevice
open Lrm.Testing

/// CpuDevice is in the backends group, so it builds without tail calls (Directory.Build.props, ADR R3-0001).
let private tailCalls =
    testCase "Lrm.Gpu.CpuDevice and this exe build without tail calls"
    <| fun () ->
        if TailCalls.builtWithoutOptimization typeof<ContentId>.Assembly then
            skiptest TailCalls.UnoptimizedBuild

        for asm in [ typeof<CpuDeviceBackend>.Assembly; Assembly.GetExecutingAssembly() ] do
            Expect.isEmpty (TailCalls.scan asm.Location) (asm.GetName().Name + " has no tail. prefixes")

/// Root test lists per tier (TestTiers contract). Everything here is cpu tier: CpuDevice needs no GPU.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu =
            [
                InterfaceTests.tests
                DispatchRuleTests.tests
                EmulatorTests.tests
                AbiTests.tests
                tailCalls
            ]
    }

[<EntryPoint>]
let main argv = Runner.main argv tiers
