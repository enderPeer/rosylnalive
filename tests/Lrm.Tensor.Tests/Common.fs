/// Helpers shared by the Lrm.Tensor tests.
module Lrm.Tensor.Tests.Common

open System
open System.IO
open System.Numerics
open Lrm.Core
open Lrm.Tensor
open Lrm.Testing

/// The repo root of this test run.
let repoRoot () = Env.repoRoot ()

/// The tiny golden set, opened once per process (the manifest is validated; files are verified when read).
let tiny = lazy (Goldens.Open(Goldens.TinyDir(repoRoot ())))

/// A fresh, empty temporary folder under the system temp path; deleted by the caller.
let tempDir (label: string) =
    let d =
        Path.Combine(Path.GetTempPath(), "lrm-o2-tests", label + "-" + Guid.NewGuid().ToString("N").Substring(0, 12))

    Directory.CreateDirectory d |> ignore
    d

/// Runs `f` with a strict in-memory telemetry sink installed, and returns its result and the records.
let withSink (f: unit -> 'a) : 'a * TelemetryRecord list =
    let sink = InMemorySink()
    use _ = Telemetry.install (Telemetry.ofSink sink)
    let r = f ()
    r, sink.Records

/// The float attribute `key` of a record.
let floatAttr (key: string) (r: TelemetryRecord) =
    r.Attrs
    |> List.pick (fun a ->
        match a.Key, a.Value with
        | k, AttrValue.Float v when k = key -> Some v
        | _ -> None)

/// The string attribute `key` of a record.
let strAttr (key: string) (r: TelemetryRecord) =
    r.Attrs
    |> List.tryPick (fun a ->
        match a.Key, a.Value with
        | k, AttrValue.Str v when k = key -> Some v
        | _ -> None)

/// Random float32 values in [-scale, scale) from DetRng(seed, stream).
let randomF32 (seed: uint64) (stream: string) (n: int) (scale: float32) =
    let rng = DetRng(seed, DetRng.streamOf stream)
    Array.init n (fun _ -> (rng.NextF32() * 2.0f - 1.0f) * scale)

/// Random float values in [-scale, scale).
let randomF64 (seed: uint64) (stream: string) (n: int) (scale: float) =
    let rng = DetRng(seed, DetRng.streamOf stream)
    Array.init n (fun _ -> (rng.NextF64() * 2.0 - 1.0) * scale)

/// Random finite bf16 bits of values in [-scale, scale).
let randomBf16 (seed: uint64) (stream: string) (n: int) (scale: float32) =
    randomF32 seed stream n scale |> Array.map Bf16.narrowRne

/// Random values of 'T (float32 or float) in [-scale, scale): the float64 draws rounded once.
let random<'T when 'T :> IFloatingPointIeee754<'T>> (seed: uint64) (stream: string) (n: int) (scale: float) : 'T[] =
    randomF64 seed stream n scale |> Array.map (fun v -> 'T.CreateTruncating v)

/// The bytes of a tensor's elements.
let bytesOf (t: #IHostTensor) = t.ToBytes()

/// Fails unless both tensors hold the same bytes, naming the first differing element.
let expectSameBits (label: string) (a: #IHostTensor) (b: #IHostTensor) =
    let x = a.ToBytes()
    let y = b.ToBytes()

    if not (a.Desc.Shape.Equals b.Desc.Shape) then
        failwithf "%s: shapes differ: %O and %O" label a.Desc.Shape b.Desc.Shape

    match Determinism.firstDifference x y with
    | None -> ()
    | Some i -> failwithf "%s: bits differ at element %d" label (i / a.Desc.ElementSize)

/// Row r of a row-major tensor as a [1, cols] tensor (a copy).
let rowOf<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (t: HostTensor<'T>)
    (r: int)
    =
    HostTensor.create t.DType (Shape.matrix 1 t.Cols) (Array.sub t.Data (r * t.Cols) t.Cols)

/// Rows lo .. hi - 1 as a [hi - lo, cols] tensor (a copy).
let rowsOf<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (t: HostTensor<'T>)
    (lo: int)
    (hi: int)
    =
    HostTensor.create t.DType (Shape.matrix (hi - lo) t.Cols) (Array.sub t.Data (lo * t.Cols) ((hi - lo) * t.Cols))
