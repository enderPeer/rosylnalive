/// Helpers shared by the Lrm.Gpu.Exec tests: the tiny model, the executor on a CpuDevice, CpuOps F32 to compare it
/// with, and the bit comparisons.
module Lrm.Gpu.Exec.Tests.Common

open System
open System.IO
open Lrm.Core
open Lrm.Tensor
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Testing

/// The threads the tests compute with (the plan's limit for CPU jobs on the shared machine).
[<Literal>]
let Threads = 8

/// The repo root of this test run.
let repoRoot () = Env.repoRoot ()

/// The tiny golden set, opened once per process.
let tiny = lazy (Goldens.Open(Goldens.TinyDir(repoRoot ())))

/// The tiny weights, opened and verified once per process. Never disposed: tests share the mapping.
let tinyWeights =
    lazy (Qwen3Weights.Open(Path.Combine(Goldens.TinyDir(repoRoot ()), "model")))

/// The tiny model description.
let tinyModel () = tinyWeights.Value.Model

/// The plan of most tests: 8 threads, the tiny model's 512 positions and rows, the whole model, every tap and two
/// caches alive at once (the plan of M2a's tests).
let tinyOptions () =
    { OpsOptions.defaults tinyWeights.Value.Config with
        Threads = Threads
        Taps = TapFilter.All
        Caches = 2
    }

/// The same plan over the stages of a split at the boundaries.
let splitOptions (boundaries: int list) =
    { tinyOptions () with
        Stages = Stage.split tinyWeights.Value.Config boundaries
    }

/// A CpuDevice with the default model (preserve, CpuRef's NaN, barrier checks on) and `threads` workers.
let cpuDevice (threads: int) =
    CpuDevice.create
        { CpuDeviceOptions.Default with
            Threads = threads
        }
        (DeviceOptions.For ExecutionBackend.CpuDevice)

/// The executor on a fresh CpuDevice with `exec` and the embedded shader set.
let gpuWith (exec: GpuExecOptions) (options: OpsOptions) =
    GpuForwardOps.createWith exec (cpuDevice Threads) (Lrm.Shaders.Shaders.embedded ()) tinyWeights.Value options

/// The executor on a fresh CpuDevice, as the registry's cpu-device creates it.
let gpu (options: OpsOptions) =
    gpuWith GpuExecOptions.Default options :> IForwardOps<GpuTensor>

/// cpu-f32 on the tiny model.
let cpu (options: OpsOptions) = CpuOps.f32 tinyWeights.Value options

/// The ids of a tiny golden prompt (from its fwd_f32 file).
let promptIds (prompt: string) =
    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"fwd_f32.{prompt}.safetensors")

    (f.ReadI64 "input_ids").Data |> Array.map int

/// The 64 golden greedy ids of a prompt for an oracle (torch32 or torch64p).
let goldenGreedy (prompt: string) (oracle: string) =
    use g = tiny.Force().OpenSafetensors "greedy.safetensors"
    (g.ReadI64(Invariant.format $"{prompt}.ids.{oracle}")).Data |> Array.map int

/// The argmax of the last row of x (Greedy's pick).
let lastArgmax (ops: IForwardOps<'t>) (x: 't) =
    let a = ops.Argmax x
    int a[a.Length - 1]

/// A fresh, empty temporary folder; deleted by the caller.
let tempDir (label: string) =
    let d =
        Path.Combine(Path.GetTempPath(), "lrm-x1a-tests", label + "-" + Guid.NewGuid().ToString("N").Substring(0, 12))

    Directory.CreateDirectory d |> ignore
    d

/// Deletes a temporary folder, ignoring failures.
let deleteDir (d: string) =
    try
        Directory.Delete(d, true)
    with _ ->
        ()

/// Fails unless both tensors hold the same bytes, naming the first differing element.
let expectSameBits (label: string) (a: IHostTensor) (b: IHostTensor) =
    if not (a.Desc.Shape.Equals b.Desc.Shape) then
        failwithf "%s: shapes differ: %O and %O" label a.Desc.Shape b.Desc.Shape

    if a.Desc.DType <> b.Desc.DType then
        failwithf "%s: dtypes differ: %s and %s" label (DType.name a.Desc.DType) (DType.name b.Desc.DType)

    match Determinism.firstDifference (a.ToBytes()) (b.ToBytes()) with
    | None -> ()
    | Some i -> failwithf "%s: bits differ at element %d" label (i / a.Desc.ElementSize)

/// Rows lo .. hi - 1 of an F32 [rows, cols] host tensor (a copy).
let rowsOf (t: IHostTensor) (lo: int) (hi: int) : IHostTensor =
    let cols = int t.Desc.Shape.Cols
    let bytes = t.ToBytes()

    HostTensor.ofBytes<float32>
        DType.F32
        (Shape.matrix (hi - lo) cols)
        (ReadOnlySpan(bytes, lo * cols * 4, (hi - lo) * cols * 4))
    :> IHostTensor

/// Runs f, which must raise an 'e, and returns the exception.
let throws<'e when 'e :> exn> (label: string) (f: unit -> unit) : 'e =
    let raised =
        try
            f ()
            None
        with :? 'e as e ->
            Some e

    match raised with
    | Some e -> e
    | None -> failwithf "%s: expected %s, nothing was raised" label typeof<'e>.Name

/// Fails unless f raises a ForwardException whose message contains `text`.
let refused (label: string) (text: string) (f: unit -> unit) =
    let e = throws<ForwardException> label f

    if not (e.Message.Contains text) then
        failwithf "%s: the message '%s' does not contain '%s'" label e.Message text
