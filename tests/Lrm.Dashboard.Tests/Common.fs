/// Helpers shared by the dashboard tests: temporary folders, cultures, and small run logs written through the real
/// RunContext on a virtual clock, so every test run gives the same bytes.
module Lrm.Dashboard.Tests.Common

open System
open System.Globalization
open System.IO
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Dashboard

/// Runs `f` with the current culture and UI culture set to `name`, then restores both. Checks that the culture is
/// really active, so a culture test cannot pass vacuously.
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

/// A fresh empty folder under the temp path; deleted when disposed.
type TempDir() =
    let path =
        Path.Combine(Path.GetTempPath(), "lrm-dashboard-tests", Guid.NewGuid().ToString("N"))

    do Directory.CreateDirectory path |> ignore
    member _.Path = path

    /// A sub-folder, created.
    member _.Sub(name: string) =
        let p = Path.Combine(path, name)
        Directory.CreateDirectory p |> ignore
        p

    interface IDisposable with
        member _.Dispose() =
            try
                Directory.Delete(path, true)
            with _ ->
                ()

/// Unix ns of a UTC time.
let unixNs (year: int) (month: int) (day: int) (hour: int) (minute: int) =
    DateTimeOffset(year, month, day, hour, minute, 0, TimeSpan.Zero).ToUnixTimeMilliseconds()
    * 1_000_000L

/// A clock that advances by `step` ns on every read, from `start`.
let steppingClock (start: int64) (step: int64) =
    let t = ref (start - step)

    fun () ->
        t.Value <- t.Value + step
        t.Value

/// A CPU execution class for test runs.
let cpuClass =
    {
        Backend = ExecutionBackend.CpuF32
        DeviceId = "cpu"
        Driver = "dotnet-10"
        ShaderSetHash = None
        Arch = "win-x64"
    }

/// Writes a run under `root` with the real RunContext on a virtual clock that starts at `startNs` and advances
/// 1 ms per read; `body` reports through the ambient facade. Returns the run folder.
let writeRun (root: string) (id: string) (kind: string) (title: string) (startNs: int64) (status: RunStatus) body =
    let options =
        { RunOptions.create kind title with
            RunId = Some(RunId.parse id)
            RunsRoot = Some root
            Clock = Some(steppingClock startNs 1_000_000L)
            FlushInterval = TimeSpan.Zero
            GitSha = Some "0123abc"
            ExecutionClass = Some cpuClass
        }

    use run = RunContext.Create options

    do
        use _ = run.Install()
        body ()

    run.Finish status
    run.Directory

/// A small parity run: two pairs over two layers, the cpu32 pair at `scale` of its bound.
let parityBody (scale: float) () =
    for pair, bound in [ "cpu64-vs-torch64p", 1e-12; "cpu32-vs-torch32", 1e-5 ] do
        for layer in 0..1 do
            for tap in [ "q_proj"; "mlp.down.out" ] do
                use _ = Telemetry.span "model.layer" [ Attr.int "layer" layer ]
                Telemetry.meter "model.flops" 2048.0 TelemetryUnits.Flops [ Attr.int "layer" layer ]
                let measured = bound * scale * float (layer + 1) / 2.0

                Telemetry.check
                    "golden.compare"
                    (measured <= bound)
                    [
                        Attr.str "pair" pair
                        Attr.str "case" "p0"
                        Attr.str "tap" (Invariant.format $"L{layer}.{tap}")
                        Attr.str "metric" "rel_l2"
                        Attr.float "measured" measured
                        Attr.float "bound" bound
                    ]

/// A run of a kind no report template reads: checks, a span, a counter and a gauge.
let recordsBody () =
    for i in 0..9 do
        Telemetry.check "tokenizer.case" (i <> 3) [ Attr.str "case" (Invariant.format $"c{i}") ]

    use _ = Telemetry.span "tokenizer.load" []
    Telemetry.counter "tokenizer.merges" 151387.0 TelemetryUnits.Count []
    Telemetry.gauge "tokenizer.cache" 4096.0 TelemetryUnits.Bytes []

/// T1's synthetic runs of every template, written under `root`.
let writeSynthetic (root: string) =
    for kind in Synthetic.kinds do
        Synthetic.write kind root |> ignore

/// The committed samples dashboard: its path, and the folder of the committed sample reports it links to.
let samplesHtml () =
    Path.Combine(Lrm.Testing.Env.repoRoot (), "tests", "Lrm.Dashboard.Tests", "reports", "samples.html")

let samplesReports () =
    Path.Combine(Lrm.Testing.Env.repoRoot (), "docs", "reports", "samples")

/// The options `lrm dashboard --synthetic --out <samples.html>` uses, over synthetic runs in `runs`.
let samplesOptions (runs: string) =
    DashboardOptions.over
        [
            RunsRoot.samples (Path.GetDirectoryName(samplesHtml ())) (samplesReports ()) runs
        ]

/// Every value of attribute `name` in the page, in order.
let attributeValues (name: string) (html: string) =
    let pattern = " " + name + "=\""

    [
        let mutable i = html.IndexOf(pattern, StringComparison.Ordinal)

        while i >= 0 do
            let start = i + pattern.Length
            let stop = html.IndexOf('"', start)
            yield html.Substring(start, stop - start)
            i <- html.IndexOf(pattern, stop, StringComparison.Ordinal)
    ]

/// The number of times `part` occurs in `text`.
let occurrences (part: string) (text: string) =
    let mutable n = 0
    let mutable i = text.IndexOf(part, StringComparison.Ordinal)

    while i >= 0 do
        n <- n + 1
        i <- text.IndexOf(part, i + part.Length, StringComparison.Ordinal)

    n
