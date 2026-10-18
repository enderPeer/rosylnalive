/// Helpers shared by the Lrm.Telemetry and Lrm.Reports tests.
module Lrm.Telemetry.Tests.Common

open System
open System.Globalization
open System.IO
open Lrm.Core

/// Runs `f` with the current culture and UI culture set to `name`, then restores both. Checks that the culture is
/// really active (ICU present), so a culture test cannot pass vacuously.
let withCulture (name: string) (f: unit -> 'a) : 'a =
    let culture = CultureInfo.CurrentCulture
    let ui = CultureInfo.CurrentUICulture

    try
        CultureInfo.CurrentCulture <- CultureInfo.GetCultureInfo name
        CultureInfo.CurrentUICulture <- CultureInfo.GetCultureInfo name

        if name = "de-DE" && (1.5).ToString() <> "1,5" then
            failwith "de-DE is not active (no ICU?)"

        f ()
    finally
        CultureInfo.CurrentCulture <- culture
        CultureInfo.CurrentUICulture <- ui

/// A fresh empty folder under the temp path; deleted when the result is disposed.
type TempDir() =
    let path =
        Path.Combine(Path.GetTempPath(), "lrm-telemetry-tests", Guid.NewGuid().ToString("N"))

    do Directory.CreateDirectory path |> ignore
    member _.Path = path

    interface IDisposable with
        member _.Dispose() =
            try
                Directory.Delete(path, true)
            with _ ->
                ()

/// A clock that advances by `step` ns on every read, from `start`.
let steppingClock (start: int64) (step: int64) =
    let t = ref (start - step)

    fun () ->
        t.Value <- t.Value + step
        t.Value

/// A record with defaults for every field.
let record (kind: TelemetryKind) (name: string) =
    {
        Kind = kind
        Name = name
        TimeNs = 1_760_000_000_000_000_000L
        DurationNs = 0L
        Value = 0.0
        Unit = ""
        GpuStartNs = 0L
        Attrs = []
        SpanId = 0L
        ParentId = 0L
    }

/// The illustrative GPU class of the ExecutionClass contract's vector X2.
let x2Class =
    {
        Backend = ExecutionBackend.Vulkan
        DeviceId = "pci:10de:2704"
        Driver = "616.92"
        ShaderSetHash = Some(Hash.sha256Utf8 "abc")
        Arch = "win-x64"
    }

/// Every file under `dir` as (relative path with forward slashes, bytes), sorted by path.
let filesUnder (dir: string) =
    Directory.GetFiles(dir, "*", SearchOption.AllDirectories)
    |> Array.map (fun f -> Path.GetRelativePath(dir, f).Replace('\\', '/'), File.ReadAllBytes f)
    |> Array.sortBy fst
    |> List.ofArray
