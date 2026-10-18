/// Helpers shared by the Lrm.Models.Qwen tests.
module Lrm.Models.Qwen.Tests.Common

open System
open System.IO
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen
open Lrm.Testing

/// The repo root of this test run.
let repoRoot () = Env.repoRoot ()

/// The tiny golden set, opened once per process (the manifest is validated; files are verified when read).
let tiny = lazy (Goldens.Open(Goldens.TinyDir(repoRoot ())))

/// The tiny model folder, goldens/tiny/g1/model.
let tinyModelDir () =
    Path.Combine(Goldens.TinyDir(repoRoot ()), "model")

/// The tiny weights, opened and verified once per process. Never disposed: tests share the mapping.
let tinyWeights = lazy (Qwen3Weights.Open(tinyModelDir ()))

/// The threads the tests compute with (the plan's limit for CPU jobs on the shared machine).
[<Literal>]
let Threads = 8

/// Options for the tiny model: 8 threads, its 512 positions.
let tinyOptions () =
    {
        Threads = Threads
        MaxContext = tinyWeights.Value.Config.MaxPositionEmbeddings
    }

/// cpu-f32 ops on the tiny model.
let tinyF32 () =
    CpuOps.f32 tinyWeights.Value (tinyOptions ())

/// cpu-f64 ops on the tiny model.
let tinyF64 () =
    CpuOps.f64 tinyWeights.Value (tinyOptions ())

/// The tiny model description.
let tinyModel () = tinyWeights.Value.Model

/// The ids of a tiny golden prompt (from its fwd_f32 file).
let promptIds (prompt: string) =
    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"fwd_f32.{prompt}.safetensors")

    (f.ReadI64 "input_ids").Data |> Array.map int

/// The 64 golden greedy ids of a prompt for an oracle (torch32 or torch64p).
let goldenGreedy (prompt: string) (oracle: string) =
    use g = tiny.Force().OpenSafetensors "greedy.safetensors"
    (g.ReadI64(Invariant.format $"{prompt}.ids.{oracle}")).Data |> Array.map int

/// A fresh, empty temporary folder; deleted by the caller.
let tempDir (label: string) =
    let d =
        Path.Combine(Path.GetTempPath(), "lrm-m2a-tests", label + "-" + Guid.NewGuid().ToString("N").Substring(0, 12))

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

/// Rows lo .. hi - 1 of a [rows, cols] host tensor (a copy, same dtype).
let rowsOf (t: IHostTensor) (lo: int) (hi: int) : IHostTensor =
    let cols = int t.Desc.Shape.Cols
    let size = t.Desc.ElementSize
    let bytes = t.ToBytes()

    match t.Desc.DType with
    | DType.F32 ->
        HostTensor.ofBytes<float32>
            DType.F32
            (Shape.matrix (hi - lo) cols)
            (ReadOnlySpan(bytes, lo * cols * size, (hi - lo) * cols * size))
        :> IHostTensor
    | DType.F64 ->
        HostTensor.ofBytes<float>
            DType.F64
            (Shape.matrix (hi - lo) cols)
            (ReadOnlySpan(bytes, lo * cols * size, (hi - lo) * cols * size))
        :> IHostTensor
    | d -> invalidArg "t" (sprintf "rowsOf: %s" (DType.name d))

/// Copies a model folder into a temporary folder, for tests that tamper with it.
let copyModel (src: string) (label: string) =
    let d = tempDir label

    for f in Directory.GetFiles src do
        File.Copy(f, Path.Combine(d, Path.GetFileName f))

    d

/// Rewrites a folder's SHA256SUMS over the given files (binary-mode lines, sorted by name).
let writeSums (dir: string) (names: string list) =
    let lines =
        names
        |> List.sortWith (fun a b -> String.CompareOrdinal(a, b))
        |> List.map (fun n -> (Hash.sha256File (Path.Combine(dir, n))).Hex + " *" + n + "\n")

    File.WriteAllText(Path.Combine(dir, "SHA256SUMS"), String.Concat lines)

/// The tensors of a safetensors file, as writer inputs.
let tensorsOf (path: string) =
    use f = SafetensorsFile.Open path

    [
        for e in f.Entries -> SafetensorsTensor.ofBytes e.Name e.Desc.DType e.Desc.Shape (f.Span(e.Name).ToArray())
    ],
    f.Metadata |> List.ofArray

/// Runs f, which must raise an 'e, and returns the exception (Expecto's throwsT returns unit).
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
