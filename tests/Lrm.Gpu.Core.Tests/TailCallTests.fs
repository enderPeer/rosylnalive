/// The root build default Tailcalls=false for GPU code (Directory.Build.props, ADR R3-0001): the GPU core, the
/// shader set and this test exe carry no IL tail. prefix, while Lrm.Core, outside the GPU groups, still has tail calls.
module Lrm.Gpu.Core.Tests.TailCallTests

open System.Reflection
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Testing

let tests =
    testList
        "tailcalls"
        [
            testCase "Lrm.Gpu.Core, Lrm.Shaders and this exe build without tail calls"
            <| fun () ->
                for asm in
                    [
                        typeof<ArgBlock>.Assembly
                        Assembly.Load(AssemblyName "Lrm.Shaders")
                        Assembly.GetExecutingAssembly()
                    ] do
                    Expect.isEmpty (TailCalls.scan asm.Location) (asm.GetName().Name + " has no tail. prefixes")

            testCase "the control: Lrm.Core, outside the GPU groups, keeps its tail calls"
            <| fun () ->
                let sites = TailCalls.scan typeof<ContentId>.Assembly.Location
                Expect.isNonEmpty sites "Lrm.Core is built with tail calls, so the scan sees them"
        ]
