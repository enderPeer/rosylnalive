/// Helpers shared by the Lrm.Serving tests.
module Lrm.Serving.Tests.Common

open System
open System.IO
open Lrm.Core
open Lrm.Cluster
open Lrm.Models.Qwen
open Lrm.Serving
open Lrm.Testing

/// The tiny model folder of this repo, goldens/tiny/g1/model.
let tinyDir () =
    Path.Combine(Env.repoRoot (), "goldens", "tiny", "g1", "model")

/// The tiny weights, opened once per process through the harness cache (shared by the in-process nodes).
let tinyWeights () = ServingHarness.weights (tinyDir ())

/// The tiny model's standard serving job (JobSpec.tiny): p0, p1 and p2 for 16 steps on cpu-f32, split at layer 1, with
/// the model given by its path so the tests do not depend on the current folder.
let tinySpec =
    { JobSpec.tiny with
        Model = tinyDir ()
        Prompts =
            [
                for p in [ "p0"; "p1"; "p2" ] ->
                    PromptSpec.IdsFrom(
                        Path.Combine(
                            Env.repoRoot (),
                            "goldens",
                            "tiny",
                            "g1",
                            Invariant.format $"fwd_f32.{p}.safetensors"
                        )
                    )
            ]
    }

/// The same job on cpu-f64.
let tinySpec64 = { tinySpec with Backend = "cpu-f64" }

/// The ServingJob of a spec.
let jobOf (spec: JobSpec) = snd (ServingHarness.job spec)

/// The oracles of a spec's job (computed once per process).
let referenceOf (spec: JobSpec) =
    ServingHarness.reference (tinyWeights ()) (jobOf spec) 4

/// The golden greedy ids of a tiny prompt for an oracle (torch32 or torch64p).
let goldenGreedy (prompt: string) (oracle: string) =
    use g =
        Lrm.Tensor.SafetensorsFile.Open(Path.Combine(Env.repoRoot (), "goldens", "tiny", "g1", "greedy.safetensors"))

    (g.ReadI64(Invariant.format $"{prompt}.ids.{oracle}")).Data |> Array.map int

/// Executor options for a tiny job: every stage of its split, its positions, room for every chain, 2 threads.
let executorOptions (job: ServingJob) =
    let c = (tinyWeights ()).Config

    { StageExecutorOptions.defaults c with
        Boundaries = job.Boundaries
        Hosted = ServingJob.stages job |> List.map _.Index
        MaxContext = ServingJob.maxPositions job
        MaxChains = 4 * job.Prompts.Length * (job.Boundaries.Length + 1)
        Threads = 2
    }

/// An executor of a backend over the tiny weights.
let executor (backend: string) (options: StageExecutorOptions) =
    (QwenStageExecutor.tryBackend backend).Value.Create (tinyWeights ()) options

/// Every unit of the job in plan order with its input bytes, grouped by chain, in step order: what a node's replay
/// applies (the coordinator's unit log).
let plannedChains (job: ServingJob) : Map<ChainKey, (WorkUnit * byte[]) list> =
    let workload = QwenWorkload job :> IWorkload
    let queue = Collections.Generic.Queue<PlannedUnit>(workload.Start())
    let all = ResizeArray<PlannedUnit>()
    use ex = executor (ExecutionBackend.name job.Class.Backend) (executorOptions job)
    let states = Collections.Generic.Dictionary<ChainKey, IChainState>()

    try
        while queue.Count > 0 do
            let p = queue.Dequeue()
            all.Add p
            let chain = WorkUnit.chain p.Unit

            let state =
                match states.TryGetValue chain with
                | true, s -> s
                | _ ->
                    let s = ex.Open(workload.Job, chain)
                    states[chain] <- s
                    s

            for next in workload.Accept(p.Unit, state.Apply(p.Unit, p.Input)) do
                queue.Enqueue next
    finally
        for s in states.Values do
            s.Dispose()

    all
    |> Seq.groupBy (fun p -> WorkUnit.chain p.Unit)
    |> Seq.map (fun (k, ps) ->
        k,
        ps
        |> Seq.sortBy (fun p -> p.Unit.Step)
        |> Seq.map (fun p -> p.Unit, p.Input)
        |> List.ofSeq)
    |> Map.ofSeq

/// The process start info of a serving worker: this test exe again, as a worker (it never takes a test mutex).
let workerStart = Harness.selfStartInfo [ "--serving-worker" ] Env.workerEnvironment

/// The CI liveness limit of every real-time run (TestTiers: 60 s timeouts, digest-only assertions).
[<Literal>]
let LivenessMs = 60_000

/// Fails with the run's verdict, its node errors and the coordinator's state unless the run is bit-identical.
let expectIdentical (reference: QwenReference) (r: HarnessResult) =
    match ServingHarness.verdict reference r with
    | [] -> ()
    | problems ->
        let nodes =
            r.Nodes
            |> List.map (fun n ->
                sprintf
                    "  %s inc %d killed %b exit %A %s\n%s"
                    n.Name
                    n.Incarnation
                    n.Killed
                    n.ExitCode
                    n.Errors
                    n.State)

        failwithf
            "%s\n%s\n%s\n%s"
            (ServingHarness.describe r problems)
            (String.concat "\n" problems)
            (String.concat "\n" nodes)
            r.Diagnostics

/// A fresh, empty temporary folder; deleted by the caller.
let tempDir (label: string) =
    let d =
        Path.Combine(Path.GetTempPath(), "lrm-c2a-tests", label + "-" + Guid.NewGuid().ToString("N").Substring(0, 12))

    Directory.CreateDirectory d |> ignore
    d

/// Deletes a temporary folder, ignoring failures.
let deleteDir (d: string) =
    try
        Directory.Delete(d, true)
    with _ ->
        ()
