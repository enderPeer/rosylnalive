// Renders K2's matmul bench report (templates/matmul.md) from a MatmulBench run log, with T1's template engine and
// charts (Lrm.Reports), until the kernels template charts the gpu.matmul.* records. A pure function of the run log:
// the same log gives the same bytes.
//
//   dotnet build Lrm.slnx -c Release
//   dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/report.fsx <run id or folder> <out dir>
//
// It writes <out dir>/matmul.md and <out dir>/charts/matmul-*.svg; `lrm report <run>` renders the kernels report of
// the same run beside it.
#I "../../../../artifacts/bin/Lrm.Cli/release"
#r "Lrm.Core.dll"
#r "Lrm.Telemetry.dll"
#r "Lrm.Reports.dll"

open System
open System.IO
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders

Process.init ()

let shapeOrder =
    [ "q_proj"; "kv_proj"; "o_proj"; "gate_up_proj"; "down_proj"; "lm_head" ]

let budgetNs = 100_000_000.0

type Point =
    {
        Shape: string
        M: int64
        N: int64
        K: int64
        TimeNs: int64
        Flops: float
        Bytes: float
    }

let i64 (e: RunEvent) (key: string) = defaultArg (e.TryInt key) 0L
let str (e: RunEvent) (key: string) = defaultArg (e.TryStr key) ""
let byShape (shape: string) = Common.orderIn shapeOrder shape

let build (log: RunLog) : Scope =
    let bandwidth =
        log.Named "gpu.matmul.bandwidth"
        |> List.map (fun e -> (str e "shape", i64 e "m"), e.Value)
        |> Map.ofList

    let points =
        log.Named "gpu.matmul.throughput"
        |> List.map (fun e ->
            let shape = str e "shape"
            let m = i64 e "m"

            {
                Shape = shape
                M = m
                N = i64 e "n"
                K = i64 e "k"
                TimeNs = i64 e "time_ns"
                Flops = e.Value
                Bytes = bandwidth.TryFind((shape, m)) |> Option.defaultValue nan
            })
        |> List.sortBy (fun p -> byShape p.Shape, p.M)

    let shapes = points |> List.map _.Shape |> Common.sortIn shapeOrder

    let series (value: Point -> float) =
        shapes
        |> List.map (fun s ->
            {
                Name = s
                Color = Common.colorIn shapeOrder s
                Points =
                    points
                    |> List.filter (fun p -> p.Shape = s)
                    |> List.map (fun p -> Math.Log10(float p.M), value p)
            })

    let line (title: string) (subtitle: string) (yLabel: string) (yFormat: NumFormat) (yLog: bool) value refs =
        if points.IsEmpty then
            Chart.Empty(title, "no gpu.matmul.throughput gauges in this run")
        else
            Chart.Line
                {
                    Title = title
                    Subtitle = subtitle
                    XLabel = "m, rows of x (log)"
                    YLabel = yLabel
                    XFormat = NumFormat.Pow10
                    YFormat = yFormat
                    YLog = yLog
                    YFromZero = not yLog
                    IntegerX = false
                    Series = series value
                    RefLines = refs
                    Bands = []
                }

    let invariance = log.Named "gpu.matmul.batch_invariance"
    let exact = log.Named "gpu.matmul.cpuref_exact"
    let probes = log.Named "gpu.matmul.probe"
    let passedOf (e: RunEvent) = e.Value = 1.0

    let denormals =
        match log.Named "gpu.fp32_denormals" |> List.tryLast with
        | Some e -> str e "mode"
        | None -> "not measured"

    let slowest = points |> List.sortByDescending _.TimeNs |> List.tryHead
    let fastest = points |> List.sortByDescending _.Flops |> List.tryHead

    let lmDecode = points |> List.tryFind (fun p -> p.Shape = "lm_head" && p.M = 1L)

    let ok =
        not invariance.IsEmpty
        && invariance |> List.forall passedOf
        && points |> List.forall (fun p -> float p.TimeNs < budgetNs)

    let verdict =
        if invariance.IsEmpty then
            "**INCOMPLETE.** The run has no batch-invariance checks."
        elif ok then
            Invariant.format
                $"**PASS.** Every row alone equals its rows in the batched calls on all {invariance.Length} shapes, and every dispatch is under the 100 ms budget."
        else
            "**FAIL.** A row alone differs from a batched call, or a dispatch is over the 100 ms budget."

    let reps =
        log.Manifest.Params
        |> List.tryFind (fun (k, _) -> k = "reps")
        |> Option.map snd
        |> Option.defaultValue "?"

    let checkRows =
        invariance
        |> List.sortBy (fun e -> byShape (str e "shape"))
        |> List.map (fun e ->
            let shape = str e "shape"

            let ref = exact |> List.tryFind (fun x -> str x "shape" = shape)

            [
                shape
                Invariant.i64 (i64 e "n")
                Invariant.i64 (i64 e "k")
                str e "ms"
                Invariant.i64 (i64 e "rows")
                (if passedOf e then
                     "✓ identical"
                 else
                     "✗ " + Invariant.i64 (i64 e "differences") + " differ")
                (match ref with
                 | Some x when passedOf x -> "✓ bit-exact"
                 | Some x -> Invariant.i64 (i64 x "differences") + " differ"
                 | None -> "not run")
            ])

    let probeRows =
        probes
        |> List.map (fun e ->
            [
                str e "probe"
                "`" + str e "result" + "`"
                "`" + str e "expected" + "`"
                "`" + str e "other" + "`"
                str e "detail"
            ])

    Common.runScope log
    |> Scope.value "verdict" verdict
    |> Scope.value "points" (Invariant.i32 points.Length)
    |> Scope.value "reps" reps
    |> Scope.value
        "peak"
        (match fastest with
         | Some p -> Fmt.siWith "FLOP/s" p.Flops
         | None -> "n/a")
    |> Scope.value
        "peak_at"
        (match fastest with
         | Some p -> Invariant.format $"{p.Shape}, m = {p.M}"
         | None -> "n/a")
    |> Scope.value
        "lm_head_decode"
        (match lmDecode with
         | Some p ->
             Fmt.duration (float p.TimeNs)
             + ", "
             + Fmt.siWith "B/s" p.Bytes
             + " of weight and activation traffic"
         | None -> "not measured")
    |> Scope.value
        "slowest"
        (match slowest with
         | Some p -> Invariant.format $"{Fmt.duration (float p.TimeNs)} ({p.Shape}, m = {p.M})"
         | None -> "n/a")
    |> Scope.value
        "slowest_budget"
        (match slowest with
         | Some p -> Fmt.pct (float p.TimeNs / budgetNs)
         | None -> "n/a")
    |> Scope.value "denormals" denormals
    |> Scope.chart
        "throughput"
        (line
            "Throughput by m"
            "FLOP/s (2mnk over the faster dispatch) per Qwen3-0.6B weight shape"
            "FLOP/s"
            NumFormat.Si
            false
            _.Flops
            [])
    |> Scope.chart
        "bandwidth"
        (line
            "Memory traffic by m"
            "The FLOP registry's minimum bytes over the faster dispatch: decode (small m) is bound by the weights"
            "B/s"
            NumFormat.Si
            false
            _.Bytes
            [])
    |> Scope.chart
        "time"
        (line
            "Dispatch time by m"
            "The faster dispatch, against the 100 ms budget of every dispatch"
            "time (log)"
            NumFormat.Duration
            true
            (fun p -> float p.TimeNs)
            [
                {
                    Value = budgetNs
                    Label = "100 ms budget"
                }
            ])
    |> Scope.table
        "points"
        (Common.table
            [
                "Shape", Common.L
                "m", Common.R
                "n", Common.R
                "k", Common.R
                "Time", Common.R
                "FLOP/s", Common.R
                "Traffic", Common.R
                "Of budget", Common.R
            ]
            (points
             |> List.map (fun p ->
                 [
                     p.Shape
                     Fmt.int p.M
                     Fmt.int p.N
                     Fmt.int p.K
                     Fmt.duration (float p.TimeNs)
                     Fmt.siWith "FLOP/s" p.Flops
                     Fmt.siWith "B/s" p.Bytes
                     Fmt.pct (float p.TimeNs / budgetNs)
                 ])))
    |> Scope.table
        "invariance"
        (Common.table
            [
                "Shape", Common.L
                "n", Common.R
                "k", Common.R
                "Calls (m)", Common.L
                "Rows", Common.R
                "Rows alone vs batched", Common.L
                "vs CpuRef F32", Common.L
            ]
            checkRows)
    |> Scope.table
        "probes"
        (Common.table
            [
                "Probe", Common.L
                "Result", Common.L
                "CpuRef F32", Common.L
                "Other choice", Common.L
                "What it shows", Common.L
            ]
            probeRows)

let template: ReportTemplate =
    {
        Name = "matmul"
        Description = "K2's matmul_bf16w throughput, batch invariance across calls and arithmetic probes"
        Phase = 1
        Build = build
    }

let run, outDir =
    match fsi.CommandLineArgs |> Array.tail with
    | [| run; out |] -> run, out
    | _ -> failwith "usage: report.fsx <run id or folder> <out dir>"

let runDir =
    if Directory.Exists run then
        run
    else
        match RunId.tryParse run with
        | Some id -> LrmPaths.run id
        | None -> failwithf "no run %s" run

let log =
    match RunLogReader.load runDir with
    | Ok l -> l
    | Error e -> failwithf "cannot read %s: %s" runDir e

let text =
    File.ReadAllText(Path.Combine(__SOURCE_DIRECTORY__, "templates", "matmul.md")).Replace("\r\n", "\n")

for path in Reports.write outDir (Reports.renderWith template text log) do
    printfn "wrote %s" path
