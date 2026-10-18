/// The conformance suites: every discovered case on CpuDevice (cpu tier, CI) and on each GPU backend whose provider
/// exists (gpu tier, local), each writing a run log of kind kernels that `lrm report` renders.
module Lrm.Kernels.Conformance.Suites

open System
open System.Collections.Concurrent
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Telemetry
open Lrm.Testing

/// Every case of this exe (Discovery.cases over Cases/**).
let allCases = lazy (Discovery.cases typeof<KernelCase>.Assembly)

/// One backend's conformance run: the backend, created on first use, its run log and its session.
type BackendRun(label: string, create: unit -> IGpuBackend, title: string, record: IGpuBackend -> unit) =
    let state =
        lazy
            (let backend = create ()
             let kernels = backend.LoadKernels(Lrm.Shaders.Shaders.embedded ())
             let cls = backend.ExecutionClass
             record backend

             let run =
                 RunContext.Create(
                     { RunOptions.create "kernels" title with
                         ExecutionClass = Some cls
                         GitSha = Env.commit ()
                         Params =
                             [
                                 "source", "tests/Lrm.Kernels.Conformance (ConformanceHarness)"
                                 "backend", ExecutionBackend.name cls.Backend
                                 "device", backend.Info.Name
                                 "cases", Invariant.i32 allCases.Value.Length
                             ]
                     }
                 )

             printfn "conformance run log (%s): %s" label run.Directory
             use _ = run.Install()

             let session =
                 Session.Create(backend, kernels, allCases.Value, HarnessOptions.Default)

             run, session)

    member _.Label = label
    member _.Started = state.IsValueCreated

    /// The run log and session (created on first use).
    member _.State = state.Value

    /// Finishes the run log: ok when the exe passed, failed otherwise; frees the backend.
    member _.Finish(exitCode: int) =
        if state.IsValueCreated then
            let run, session = state.Value
            run.Finish(if exitCode = 0 then RunStatus.Ok else RunStatus.Failed)
            (session :> IDisposable).Dispose()

/// Every backend run started in this process, finished at exit (Main).
let runs = ConcurrentQueue<BackendRun>()

/// The tests of one backend: each case (skipped when the backend lacks its kernel), then the negative control.
let suite (run: BackendRun) : Test =
    runs.Enqueue run

    let caseTest (c: KernelCase) =
        testCase c.Id
        <| fun () ->
            let log, session = run.State
            use _ = log.Install()

            if not (session.Contains c) then
                skiptestf "%s has no %s kernel yet (its source is missing)" session.BackendName c.Kernel

            let o = session.Run c

            if not o.Passed then
                failtestf "%s on %s: %s" c.Id o.Backend o.Detail

    testSequenced
    <| testList
        ("conformance." + run.Label)
        [
            yield! allCases.Value |> List.map caseTest
            testCase "negative control: with the dispatches removed, every case fails"
            <| fun () ->
                let _, session = run.State

                let passing =
                    allCases.Value
                    |> List.filter session.Contains
                    |> List.filter (fun c -> (session.Control c).Passed)
                    |> List.map _.Id

                Expect.isEmpty passing "cases that pass without their dispatch prove nothing"
        ]

/// The cpu-tier suite: CpuDevice, at most 4 emulation threads.
let cpuDevice () =
    let cpu =
        { CpuDeviceOptions.Default with
            Threads = min 4 Environment.ProcessorCount
        }

    BackendRun(
        "cpu-device",
        (fun () ->
            CpuDevice.create
                cpu
                { DeviceOptions.For ExecutionBackend.CpuDevice with
                    TimeDispatches = true
                }),
        "Conformance on CpuDevice (every case, bit-exact to CpuRef F32)",
        ignore
    )
    |> suite

/// The gpu-tier suites: one per GPU backend whose provider exists (D1 and V1 add them), on its first accepted device.
let gpu () : (Backend * Test list) list =
    [
        for backend, testing in
            [
                ExecutionBackend.D3D12, Backend.D3D12
                ExecutionBackend.Vulkan, Backend.Vulkan
            ] do
            match Discovery.provider backend with
            | None -> ()
            | Some provider ->
                let registry = BackendRegistry()
                registry.Register provider
                let name = ExecutionBackend.name backend

                let create () =
                    match
                        registry.TryCreate
                            { DeviceOptions.For backend with
                                TimeDispatches = true
                            }
                    with
                    | Ok b -> b
                    | Error e -> failwithf "no %s device: %s" name e

                let record (b: IGpuBackend) =
                    let c = b.ExecutionClass
                    Devices.record testing b.Info.Name c.Driver (c.ShaderSetHash |> Option.map _.Hex)

                testing, [ BackendRun(name, create, "Conformance on " + name, record) |> suite ]
    ]
