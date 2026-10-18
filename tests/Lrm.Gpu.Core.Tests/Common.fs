/// Helpers shared by the Lrm.Gpu.Core tests.
module Lrm.Gpu.Core.Tests.Common

open System
open Expecto
open Lrm.Core
open Lrm.Gpu.Core

/// Lowercase hex.
let hex (bytes: byte[]) = Convert.ToHexStringLower bytes

/// Bytes from hex that may contain spaces.
let unhex (text: string) =
    Convert.FromHexString(text.Replace(" ", ""))

/// Requires `f` to raise GpuContractException whose message contains `fragment`.
let raisesContract (fragment: string) (f: unit -> 'a) =
    try
        f () |> ignore
        failtestf "expected GpuContractException containing '%s', but nothing was raised" fragment
    with
    | :? GpuContractException as e ->
        if not (e.Message.Contains fragment) then
            failtestf "expected a message containing '%s', got '%s'" fragment e.Message
    | :? AssertException -> reraise ()
    | e -> failtestf "expected GpuContractException containing '%s', got %s: %s" fragment (e.GetType().Name) e.Message

/// A small layout: one page each of weights (64 KiB), activations (256 KiB) and scratch (64 KiB): slots 1, 2, 3.
let smallLayout () =
    HeapLayout.ofPages
        [
            ArenaKind.Weights, 65536L
            ArenaKind.Activations, 262144L
            ArenaKind.Scratch, 65536L
        ]

/// Runs `f` with an in-memory telemetry sink installed and returns its records.
let withSink (f: unit -> unit) =
    let sink = InMemorySink()
    use _ = Telemetry.install (Telemetry.ofSinkWithClock (fun () -> 0L) sink)
    f ()
    sink.Records
