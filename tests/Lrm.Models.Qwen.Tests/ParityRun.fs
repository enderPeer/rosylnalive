/// The parity run logs of the M2 tests (ReportTemplates, parity). The acceptance tests report through the facade, and
/// this module also writes their records to a real run log: one run per descriptor (RunSpec), created by the first
/// test that installs it, with one node per CPU class: n0 (the creator) is cpu-f64, n1 is cpu-f32. Main finishes every
/// run when the exe ends. The cpu tier writes M2a's tiny-model run (`tiny`); the model tier writes M2b's full-model
/// gate1-cpu run (GateCpuTests).
///
/// Where a run goes: with LRM_WRITE_PARITY_RUN=1 it is kept under LRM_DATA/runs/<runId>, so `lrm report <runId>`
/// renders the parity report from it (docs/reports/parity/README.md). Otherwise it goes to a temporary folder that
/// `finish` deletes, so routine test runs by every agent on the shared machine leave nothing in LRM_DATA.
module Lrm.Models.Qwen.Tests.ParityRun

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Models.Qwen.Tests.Common

/// The variable that keeps the parity run under LRM_DATA/runs.
[<Literal>]
let KeepVariable = "LRM_WRITE_PARITY_RUN"

/// True when this process keeps its parity run (LRM_WRITE_PARITY_RUN=1).
let keeps () =
    Environment.GetEnvironmentVariable KeepVariable = "1"

/// Every span on both facades, ended together.
type private TeeSpan(a: ISpan, b: ISpan) =
    interface ISpan with
        member _.Id = a.Id

        member _.Add(attr: Attr) =
            a.Add attr
            b.Add attr

    interface IDisposable with
        member _.Dispose() =
            b.Dispose()
            a.Dispose()

/// A facade that reports every record to both `a` and `b`. Each keeps its own span ids, so neither log's span tree
/// mixes with the other's.
type private Tee(a: ITelemetry, b: ITelemetry) =
    interface ITelemetry with
        member _.Enabled = a.Enabled || b.Enabled

        member _.Span(name, attrs) =
            new TeeSpan(a.Span(name, attrs), b.Span(name, attrs)) :> ISpan

        member _.Counter(name, delta, unit, attrs) =
            a.Counter(name, delta, unit, attrs)
            b.Counter(name, delta, unit, attrs)

        member _.Gauge(name, value, unit, attrs) =
            a.Gauge(name, value, unit, attrs)
            b.Gauge(name, value, unit, attrs)

        member _.Histogram(name, value, unit, attrs) =
            a.Histogram(name, value, unit, attrs)
            b.Histogram(name, value, unit, attrs)

        member _.Event(name, attrs) =
            a.Event(name, attrs)
            b.Event(name, attrs)

        member _.GpuInterval(name, start, duration, attrs) =
            a.GpuInterval(name, start, duration, attrs)
            b.GpuInterval(name, start, duration, attrs)

        member _.Meter(name, amount, unit, attrs) =
            a.Meter(name, amount, unit, attrs)
            b.Meter(name, amount, unit, attrs)

        member _.Check(name, passed, attrs) =
            a.Check(name, passed, attrs)
            b.Check(name, passed, attrs)

        member _.Child(attrs) =
            Tee(a.Child attrs, b.Child attrs) :> ITelemetry

/// What a parity run is: its key (one run per key and process), title and parameters, and whether CpuRef's per-op
/// `cpuref.*` spans stay out of the log.
type RunSpec =
    {
        Key: string
        Title: string
        Params: (string * string) list
        /// Leave the per-op `cpuref.*` spans out (CpuRefSemantics' op timing). The full-model run would otherwise
        /// hold about half a million of them per node; the model.forward and model.layer spans and every meter stay.
        DropOpSpans: bool
    }

/// M2a's tiny-model acceptance run (cpu tier).
let tinySpec =
    {
        Key = "tiny"
        Title = "Tiny model: the Qwen forward (M2a) on cpu-f64 and cpu-f32"
        Params =
            [
                "golden", "goldens/tiny/g1"
                "model", "goldens/tiny/g1/model (local:tiny-g1)"
                "source", "tests/Lrm.Models.Qwen.Tests (M2a tiny-model acceptance)"
                "nodes", "n0 cpu-f64, n1 cpu-f32"
                "threads", "8"
            ]
        DropOpSpans = false
    }

/// A facade that leaves the spans named `cpuref.*` out and passes every other record on.
type private WithoutOpSpans(inner: ITelemetry) =
    interface ITelemetry with
        member _.Enabled = inner.Enabled

        member _.Span(name, attrs) =
            if name.StartsWith("cpuref.", StringComparison.Ordinal) then
                Telemetry.noop.Span(name, attrs)
            else
                inner.Span(name, attrs)

        member _.Counter(name, delta, unit, attrs) = inner.Counter(name, delta, unit, attrs)
        member _.Gauge(name, value, unit, attrs) = inner.Gauge(name, value, unit, attrs)

        member _.Histogram(name, value, unit, attrs) =
            inner.Histogram(name, value, unit, attrs)

        member _.Event(name, attrs) = inner.Event(name, attrs)

        member _.GpuInterval(name, start, duration, attrs) =
            inner.GpuInterval(name, start, duration, attrs)

        member _.Meter(name, amount, unit, attrs) = inner.Meter(name, amount, unit, attrs)
        member _.Check(name, passed, attrs) = inner.Check(name, passed, attrs)

        member _.Child(attrs) =
            WithoutOpSpans(inner.Child attrs) :> ITelemetry

type private Runs =
    {
        F64: RunContext
        F32: RunContext
        Spec: RunSpec
        /// The temporary runs folder that finish deletes, when the run is not kept.
        Temp: string option
    }

let private gate = obj ()

let private runs =
    Collections.Generic.Dictionary<string, Runs>(StringComparer.Ordinal)

let private create (spec: RunSpec) =
    let temp = if keeps () then None else Some(tempDir "parity-runs")

    let f64 =
        RunContext.Create(
            { RunOptions.create "parity" spec.Title with
                ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuF64)
                RunsRoot = temp
                GitSha = Lrm.Testing.Env.commit ()
                Params = spec.Params
            }
        )

    let f32 =
        RunContext.Join(
            f64.Directory,
            { JoinOptions.create (NodeId.parse "n1") with
                ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuF32)
            }
        )

    if temp.IsNone then
        printfn "parity run log (%s): %s" spec.Key f64.Directory

    {
        F64 = f64
        F32 = f32
        Spec = spec
        Temp = temp
    }

let private get (spec: RunSpec) =
    lock gate (fun () ->
        match runs.TryGetValue spec.Key with
        | true, r when r.Spec = spec -> r
        | true, _ -> invalidOp (sprintf "parity run '%s' was created with another descriptor" spec.Key)
        | _ ->
            let r = create spec
            runs[spec.Key] <- r
            r)

/// The run-log facade of a CPU class (cpu-f64 or cpu-f32) in the run of `spec`.
let telemetryFor (spec: RunSpec) (backend: ExecutionBackend) =
    let r = get spec

    let t =
        match backend with
        | ExecutionBackend.CpuF64 -> r.F64.Telemetry
        | ExecutionBackend.CpuF32 -> r.F32.Telemetry
        | b -> invalidArg "backend" (sprintf "the parity run has no node for %s" (ExecutionBackend.name b))

    if spec.DropOpSpans then
        WithoutOpSpans t :> ITelemetry
    else
        t

/// The run-log facade of a CPU class in the tiny-model run.
let telemetryOf (backend: ExecutionBackend) = telemetryFor tinySpec backend

/// Makes the run-log facade of `backend` in the run of `spec` ambient until disposed.
let installFor (spec: RunSpec) (backend: ExecutionBackend) =
    Telemetry.install (telemetryFor spec backend)

/// Makes the run-log facade of `backend` in the tiny-model run ambient until disposed.
let install (backend: ExecutionBackend) = installFor tinySpec backend

/// Makes a facade ambient that reports to both `other` (for example a strict in-memory sink's) and the tiny-model run.
let installWith (backend: ExecutionBackend) (other: ITelemetry) =
    Telemetry.install (Tee(other, telemetryOf backend))

/// The run folder of `spec`'s run, if a test created it.
let directoryOf (spec: RunSpec) =
    lock gate (fun () ->
        match runs.TryGetValue spec.Key with
        | true, r -> Some r.F64.Directory
        | _ -> None)

/// Finishes every run a test created: ok when the exe passed, failed otherwise.
let finish (exitCode: int) =
    lock gate (fun () ->
        let status = if exitCode = 0 then RunStatus.Ok else RunStatus.Failed

        for r in List.ofSeq runs.Values do
            // Finish closes each node's events file; the creator (n0) also writes the final manifest.
            r.F32.Finish status
            r.F64.Finish status
            r.Temp |> Option.iter deleteDir

        runs.Clear())
