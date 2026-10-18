/// The executor's telemetry (GpuExecutor, "Telemetry"): a gpu.dispatch interval per dispatch with kernel, layer, step
/// and phase (CpuDevice's emulation time), the mem.arena gauges of every arena with its lowercase name, the
/// gpu.exec.create span, and the forward's model.* spans, all in a real run log the inference and kernels templates
/// read; ops created before a run's telemetry still give the run its arenas; telemetry changes no value.
module Lrm.Gpu.Exec.Tests.TelemetryTests

open System
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Gpu.Core
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Gpu.Exec.Tests.Common

let private greedyOptions () =
    { tinyOptions () with
        Taps = TapFilter.Off
        Caches = 1
        MaxContext = 13 + 7
        MaxRows = 13
    }

/// Runs `f` inside a fresh run log (kind inference) and returns the log.
let private withRun (f: unit -> unit) =
    let root = tempDir "runs"

    try
        let dir =
            use run =
                RunContext.Create(
                    { RunOptions.create "inference" "X1a telemetry test" with
                        ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuDevice)
                        RunsRoot = Some root
                    }
                )

            do
                use _ = run.Install()
                f ()

            run.Finish RunStatus.Ok
            run.Directory

        RunLogReader.loadOrFail dir
    finally
        deleteDir root

let private named (name: string) (kind: TelemetryKind) (log: RunLog) =
    log.Named name |> List.filter (fun e -> e.Kind = kind)

let tests =
    testList
        "telemetry"
        [
            testCase "a greedy run on cpu-device: gpu.dispatch per kernel, layer and step; mem.arena per arena; spans"
            <| fun () ->
                let ids = promptIds "p0"
                let dispatches = ref 0L

                let log =
                    withRun (fun () ->
                        use ops = gpuWith GpuExecOptions.Default (greedyOptions ())

                        Greedy.run (ops :> IForwardOps<GpuTensor>) (tinyModel ()) ids (GreedyOptions.parity 8)
                        |> ignore

                        dispatches.Value <- ops.Dispatches)

                Expect.isEmpty log.Problems "a clean log"
                let gpu = named GpuTelemetry.Dispatch TelemetryKind.GpuInterval log
                Expect.equal (int64 gpu.Length) dispatches.Value "one gpu.dispatch per dispatch"

                Expect.equal
                    (gpu
                     |> List.choose (fun e -> e.TryStr TelemetryKeys.Kernel)
                     |> List.distinct
                     |> List.sort)
                    (ExecPlan.kernels |> List.map (fun k -> Kernels.all[k.Value].Name) |> List.sort)
                    "every kernel the executor dispatches"

                Expect.equal
                    (gpu
                     |> List.choose (fun e -> e.TryInt TelemetryKeys.Layer)
                     |> List.distinct
                     |> List.sort)
                    [ 0L; 1L ]
                    "both layers"

                Expect.equal
                    (gpu |> List.choose (fun e -> e.TryInt "step") |> List.distinct |> List.sort)
                    [ 0L .. 7L ]
                    "steps 0 to 7"

                Expect.equal
                    (gpu |> List.choose (fun e -> e.TryStr "phase") |> List.distinct |> List.sort)
                    [ "decode"; "prefill" ]
                    "phases"

                Expect.isTrue
                    (gpu |> List.forall (fun e -> e.TryStr TelemetryKeys.Backend = Some "cpu-device"))
                    "the backend on every interval"

                // A layer's dispatches: 19 ops, as 19 dispatches in a decode step.
                let decodeLayer0 =
                    gpu
                    |> List.filter (fun e -> e.TryInt "step" = Some 3L && e.TryInt TelemetryKeys.Layer = Some 0L)

                Expect.equal decodeLayer0.Length 19 "19 dispatches in layer 0 of a decode step"

                let arenas =
                    named TelemetryNames.MemArena TelemetryKind.Gauge log
                    |> List.choose (fun e -> e.TryStr TelemetryKeys.Arena)
                    |> List.distinct
                    |> List.sort

                Expect.equal
                    arenas
                    ([ "weights"; "rope"; "kv_cache"; "activations"; "scratch"; "logits"; "table" ]
                     |> List.sort)
                    "mem.arena, lowercase"

                let create =
                    named GpuForwardOps.CreateSpan TelemetryKind.Span log |> List.exactlyOne

                Expect.equal (create.TryStr TelemetryKeys.Component) (Some GpuForwardOps.Component) "the component"
                Expect.equal (named Forward.ForwardSpan TelemetryKind.Span log).Length 8 "8 model.forward spans"
                Expect.equal (named Forward.LayerSpan TelemetryKind.Span log).Length 16 "16 model.layer spans"

            testCase "with Taps planned, the taps arena too; ops made before the run still give it its arenas"
            <| fun () ->
                use ops = gpuWith GpuExecOptions.Default (tinyOptions ())
                let ids = promptIds "p0"

                let log =
                    withRun (fun () ->
                        let model = tinyModel ()
                        let g = ops :> IForwardOps<GpuTensor>
                        use cache = KvCache.create g model (Stage.whole model.Config) ids.Length

                        Forward.run g model cache 0 ids (ForwardOptions.withTaps (TapStore(TapFilter.All)))
                        |> ignore)

                let arenas =
                    named TelemetryNames.MemArena TelemetryKind.Gauge log
                    |> List.choose (fun e -> e.TryStr TelemetryKeys.Arena)
                    |> Set.ofList

                Expect.isTrue (arenas.Contains "taps") "the taps arena"
                Expect.equal arenas.Count 8 "seven arenas and the table"
                Expect.isEmpty (named GpuForwardOps.CreateSpan TelemetryKind.Span log) "created before the run"

            testCase "telemetry changes no value: no-op facade, in-memory sink, and no dispatch timers"
            <| fun () ->
                let ids = promptIds "p2"

                let run (exec: GpuExecOptions) (sink: ITelemetrySink option) =
                    use _ =
                        match sink with
                        | Some s -> Telemetry.install (Telemetry.ofSink s)
                        | None -> Telemetry.install Telemetry.noop

                    use ops = gpuWith exec (tinyOptions ())
                    ParityTests.schedule (ops :> IForwardOps<GpuTensor>) ids 6

                let a = run GpuExecOptions.Default None
                let sink = InMemorySink()
                let b = run GpuExecOptions.Default (Some sink)

                let c =
                    run
                        { GpuExecOptions.Default with
                            TimeDispatches = false
                        }
                        (Some(InMemorySink()))

                Expect.isNonEmpty (sink.Records |> List.filter (fun r -> r.Name = GpuTelemetry.Dispatch)) "timed"

                for label, x, y in ParityTests.pairsOf a b @ ParityTests.pairsOf a c do
                    expectSameBits label x y
        ]
