/// The run log of X1a's acceptance (ReportTemplates, parity): the acceptance tests report their golden.compare
/// records — cpu-device against the torch32 goldens (pair gpu-vs-torch32) and the bit-identity self-tests — through
/// the facade, and `within` also writes them to one run with one node, n0, of class cpu-device. The first test that
/// reports creates the run; Main finishes it when the exe ends.
///
/// Where the run goes: with LRM_WRITE_EXEC_RUN=1 it is kept under LRM_DATA/runs/<runId>, so `lrm report <runId>`
/// renders the parity report from it (tests/Lrm.Gpu.Exec.Tests/reports/README.md). Otherwise it goes to a temporary
/// folder that `finish` deletes, so routine test runs leave nothing in LRM_DATA.
module Lrm.Gpu.Exec.Tests.AcceptanceRun

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Testing
open Lrm.Gpu.Exec.Tests.Common

/// The variable that keeps the acceptance run under LRM_DATA/runs.
[<Literal>]
let KeepVariable = "LRM_WRITE_EXEC_RUN"

/// True when this process keeps its acceptance run.
let keeps () =
    Environment.GetEnvironmentVariable KeepVariable = "1"

let private gate = obj ()
let mutable private run: (RunContext * string option) option = None

let private ensure () =
    lock gate (fun () ->
        match run with
        | Some(r, _) -> r
        | None ->
            let temp = if keeps () then None else Some(tempDir "acceptance-run")

            let r =
                RunContext.Create(
                    { RunOptions.create
                          "parity"
                          "Tiny model: the GPU executor (X1a) on cpu-device against cpu-f32 and torch32" with
                        ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuDevice)
                        GitSha = Env.commit ()
                        RunsRoot = temp
                        Params =
                            [
                                "backend", "cpu-device"
                                "model", "goldens/tiny/g1/model"
                                "threads", Invariant.i32 Threads
                                "tests", "tests/Lrm.Gpu.Exec.Tests (cpu tier, acceptance)"
                            ]
                    }
                )

            run <- Some(r, temp)
            r)

/// Runs f with the acceptance run's facade ambient, so its records also go to the run log.
let within (f: unit -> unit) =
    let r = ensure ()
    use _ = r.Install()
    f ()

/// Finishes the run (ok when the exe passed) and prints where it is, or deletes its temporary folder.
let finish (exitCode: int) =
    lock gate (fun () ->
        match run with
        | None -> ()
        | Some(r, temp) ->
            r.Finish(if exitCode = 0 then RunStatus.Ok else RunStatus.Failed)
            (r :> IDisposable).Dispose()

            match temp with
            | Some d -> deleteDir d
            | None -> printfn "acceptance run log: %s (render it with: lrm report %s)" r.Directory r.RunId.Value

            run <- None)
