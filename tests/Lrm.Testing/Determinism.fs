/// Run-twice checks (TestTiers contract: GPU work runs twice with a byte comparison; timing uses the faster run).
module Lrm.Testing.Determinism

open System
open System.Diagnostics

/// Index of the first differing byte, or None when equal.
let firstDifference (a: byte[]) (b: byte[]) =
    let n = min a.Length b.Length
    let i = a.AsSpan(0, n).CommonPrefixLength(ReadOnlySpan(b, 0, n))

    if i < n then Some i
    elif a.Length <> b.Length then Some n
    else None

/// Runs `f` twice and requires byte-identical results. Returns the bytes and the faster of the two wall times.
let runTwice (label: string) (f: unit -> byte[]) : byte[] * TimeSpan =
    let clock = Stopwatch.StartNew()
    let first = f ()
    let t1 = clock.Elapsed
    clock.Restart()
    let second = f ()
    let t2 = clock.Elapsed

    match firstDifference first second with
    | None -> first, min t1 t2
    | Some i ->
        failwithf
            "%s: run 1 and run 2 differ (lengths %d and %d, first difference at byte %d)"
            label
            first.Length
            second.Length
            i

/// An Expecto test case that runs `f` twice and requires identical bytes.
let testTwice (name: string) (f: unit -> byte[]) =
    Expecto.Tests.testCase name (fun () -> runTwice name f |> ignore)
