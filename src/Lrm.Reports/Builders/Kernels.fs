/// The kernels report: the conformance matrix and GPU time by kernel, layer and dispatch (ReportTemplates contract,
/// "kernels"). Reads conformance.case checks and gpu.dispatch GPU intervals.
module Lrm.Reports.Builders.Kernels

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Kernel families in colour order. A kernel belongs to the first family whose prefix it starts with.
let families =
    [
        "matmul", [ "matmul" ]
        "attention", [ "attention" ]
        "norm", [ "rmsnorm" ]
        "rope", [ "rope" ]
        "elementwise", [ "silu"; "add"; "scale"; "mul" ]
        "memory", [ "embed"; "copy"; "bf16_unpack"; "gather" ]
        "reduce", [ "argmax"; "softmax"; "sum" ]
    ]

/// The family of a kernel name, or "other".
let familyOf (kernel: string) =
    families
    |> List.tryFind (fun (_, prefixes) ->
        prefixes
        |> List.exists (fun p -> kernel.StartsWith(p, StringComparison.Ordinal)))
    |> Option.map fst
    |> Option.defaultValue "other"

let private familyNames = families |> List.map fst

let private familyColor (f: string) = colorIn familyNames f

/// The dispatch time budget of the KernelAbiAndCatalog contract.
let budgetNs = 100e6

/// Builds the kernels scope.
let build (log: RunLog) : Scope =
    let cases =
        log.Named TelemetryNames.ConformanceCase
        |> List.filter (fun e -> e.Kind = TelemetryKind.Check)

    let dispatches =
        log.Named "gpu.dispatch"
        |> List.filter (fun e -> e.Kind = TelemetryKind.GpuInterval)

    let backendOfEvent (e: RunEvent) =
        match e.TryStr TelemetryKeys.Backend with
        | Some b -> b
        | None -> backendOf log e.Node

    let kernelOf (e: RunEvent) =
        defaultArg (e.TryStr TelemetryKeys.Kernel) "(unnamed)"

    let caseBackends = cases |> List.map backendOfEvent |> sortIn backendOrder

    let caseKernels =
        cases
        |> List.map kernelOf
        |> List.distinct
        |> List.sortBy (fun k -> orderIn familyNames (familyOf k), k)

    let failed = cases |> List.filter (fun e -> e.Value <> 1.0)

    let matrix =
        table
            (("Kernel", L) :: ("Family", L) :: (caseBackends |> List.map (fun b -> b, L)))
            [
                for k in caseKernels ->
                    k
                    :: familyOf k
                    :: [
                        for b in caseBackends ->
                            let xs = cases |> List.filter (fun e -> kernelOf e = k && backendOfEvent e = b)
                            let ok = xs |> List.filter (fun e -> e.Value = 1.0)

                            if xs.IsEmpty then
                                "n/a"
                            else
                                Invariant.format
                                    $"{(if ok.Length = xs.Length then '✓' else '✗')} {ok.Length}/{xs.Length}"
                    ]
            ]

    let failRows =
        failed
        |> List.map (fun e ->
            [
                kernelOf e
                defaultArg (e.TryStr TelemetryKeys.Case) ""
                backendOfEvent e
                e.TryFloat "measured" |> Option.map (Fmt.sci 2) |> Option.defaultValue ""
                e.TryFloat "bound" |> Option.map (Fmt.sci 2) |> Option.defaultValue ""
                e.TryText "detail" |> Option.defaultValue ""
            ])

    let dispatchBackends = dispatches |> List.map backendOfEvent |> sortIn backendOrder
    let kernels = dispatches |> List.map kernelOf |> List.distinct

    let totalBy (pred: RunEvent -> bool) =
        dispatches |> List.filter pred |> List.sumBy (fun e -> float e.DurationNs)

    let kernelsByTime =
        kernels |> List.sortBy (fun k -> -(totalBy (fun e -> kernelOf e = k)), k)

    let byKernel =
        if dispatches.IsEmpty then
            noData "GPU time by kernel" "gpu.dispatch"
        else
            Chart.Bar
                {
                    Title = "GPU time by kernel"
                    Subtitle = "Sum of gpu.dispatch durations over the run"
                    CategoryLabel = "kernel"
                    ValueLabel = "GPU time"
                    ValueFormat = NumFormat.Duration
                    Categories = kernelsByTime
                    Series =
                        dispatchBackends
                        |> List.map (fun b ->
                            {
                                Name = b
                                Color = colorIn backendOrder b
                                Values =
                                    kernelsByTime
                                    |> List.map (fun k -> totalBy (fun e -> kernelOf e = k && backendOfEvent e = b))
                            })
                    Stacked = false
                    Horizontal = true
                }

    // The timelines show the last step of the run on every backend, each lane starting at its first dispatch of the
    // step: the whole step, then a zoom on the embedding and the first two layers.
    let lastStep =
        dispatches
        |> List.choose (fun e -> e.TryInt "step")
        |> List.distinct
        |> List.sort
        |> List.tryLast

    let inStep (e: RunEvent) =
        match lastStep with
        | Some s -> e.TryInt "step" = Some s
        | None -> true

    let stepText =
        match lastStep with
        | Some s -> ", step " + Invariant.i64 s
        | None -> ""

    let timelineOf (title: string) (subtitle: string) (keep: RunEvent -> bool) =
        if dispatches.IsEmpty then
            noData title "gpu.dispatch"
        else
            let lanes = dispatchBackends

            let intervals =
                lanes
                |> List.mapi (fun lane b ->
                    let xs = dispatches |> List.filter (fun e -> inStep e && backendOfEvent e = b)

                    let t0 =
                        if xs.IsEmpty then
                            0L
                        else
                            xs |> List.map _.GpuStartNs |> List.min

                    xs
                    |> List.filter keep
                    |> List.map (fun e ->
                        let k = kernelOf e

                        {
                            Lane = lane
                            Start = float (e.GpuStartNs - t0)
                            Length = float e.DurationNs
                            Color = familyColor (familyOf k)
                            Tip =
                                k
                                + (match e.TryInt TelemetryKeys.Layer with
                                   | Some l -> ", layer " + Invariant.i64 l
                                   | None -> "")
                                + ": "
                                + Fmt.duration (float e.DurationNs)
                        }))
                |> List.concat

            let used = intervals |> List.map _.Color |> List.distinct

            Chart.Timeline
                {
                    Title = title + stepText
                    Subtitle = subtitle
                    XLabel = "GPU time"
                    XFormat = NumFormat.Duration
                    Lanes = lanes
                    Intervals = intervals
                    Marks = []
                    Legend =
                        (familyNames @ [ "other" ])
                        |> List.map (fun f -> f, familyColor f)
                        |> List.filter (fun (_, c) -> List.contains c used)
                }

    let timeline =
        timelineOf
            "Kernel timeline"
            "Each dispatch at its GPU start, relative to the lane's first dispatch; colour is the kernel family"
            (fun _ -> true)

    let timelineZoom =
        timelineOf "Kernel timeline, embedding and layers 0 to 1" "The start of the same step, close up" (fun e ->
            match e.TryInt TelemetryKeys.Layer with
            | Some l -> l <= 1L
            | None -> familyOf (kernelOf e) = "memory")

    let layerStack =
        let layered =
            dispatches |> List.filter (fun e -> (e.TryInt TelemetryKeys.Layer).IsSome)

        match dispatchBackends with
        | first :: _ when not layered.IsEmpty ->
            let xs = layered |> List.filter (fun e -> backendOfEvent e = first)

            let layerIds =
                xs
                |> List.choose (fun e -> e.TryInt TelemetryKeys.Layer)
                |> List.distinct
                |> List.sort

            let fams =
                xs |> List.map (kernelOf >> familyOf) |> sortIn (familyNames @ [ "other" ])

            Chart.Bar
                {
                    Title = "GPU time per layer by kernel family, " + first
                    Subtitle = "Sum over all steps"
                    CategoryLabel = "layer"
                    ValueLabel = "GPU time"
                    ValueFormat = NumFormat.Duration
                    Categories = layerIds |> List.map Invariant.i64
                    Series =
                        fams
                        |> List.map (fun f ->
                            {
                                Name = f
                                Color = familyColor f
                                Values =
                                    layerIds
                                    |> List.map (fun l ->
                                        xs
                                        |> List.filter (fun e ->
                                            e.TryInt TelemetryKeys.Layer = Some l && familyOf (kernelOf e) = f)
                                        |> List.sumBy (fun e -> float e.DurationNs))
                            })
                    Stacked = true
                    Horizontal = false
                }
        | _ -> noData "GPU time per layer" "per-layer gpu.dispatch"

    let durations = dispatches |> List.map (fun e -> float e.DurationNs)

    let dispatchHist =
        if dispatches.IsEmpty then
            noData "Dispatch durations" "gpu.dispatch"
        else
            Chart.Histogram
                {
                    Title = "Dispatch durations"
                    Subtitle =
                        Invariant.format
                            $"{dispatches.Length} dispatches on a log scale; the longest is {Fmt.duration (maxOr 0.0 durations)} (budget 100 ms)"
                    XLabel = "duration (ns)"
                    XFormat = NumFormat.Pow10
                    Bins = bins 24 true durations
                    Color = Color.Cat 0
                    Marks = []
                }

    let slowest =
        dispatches
        |> List.sortBy (fun e -> -e.DurationNs, e.Node, e.Seq)
        |> List.truncate 10
        |> List.map (fun e ->
            [
                kernelOf e
                backendOfEvent e
                e.TryText TelemetryKeys.Layer |> Option.defaultValue ""
                e.TryText "step" |> Option.defaultValue ""
                Fmt.duration (float e.DurationNs)
                Fmt.pct (float e.DurationNs / budgetNs)
            ])

    let overBudget = dispatches |> List.filter (fun e -> float e.DurationNs >= budgetNs)

    runScope log
    |> Scope.value
        "verdict"
        (if cases.IsEmpty then
             "No conformance cases were recorded."
         elif failed.IsEmpty then
             Invariant.format $"**PASS.** All {cases.Length} conformance cases pass."
         else
             Invariant.format $"**FAIL.** {failed.Length} of {cases.Length} conformance cases fail.")
    |> Scope.value "cases" (Fmt.int (int64 cases.Length))
    |> Scope.value "failed" (Fmt.int (int64 failed.Length))
    |> Scope.value "dispatches" (Fmt.int (int64 dispatches.Length))
    |> Scope.value "gpu_total" (Fmt.duration (durations |> List.sum))
    |> Scope.value
        "budget"
        (if overBudget.IsEmpty then
             "Every dispatch is under the 100 ms budget."
         else
             Invariant.format $"**{overBudget.Length} dispatches exceed the 100 ms budget.**")
    |> Scope.table "matrix" matrix
    |> Scope.table
        "failures"
        (table [ "Kernel", L; "Case", L; "Backend", L; "Measured", R; "Bound", R; "Detail", L ] failRows)
    |> Scope.table
        "slowest"
        (table
            [
                "Kernel", L
                "Backend", L
                "Layer", R
                "Step", R
                "Duration", R
                "Of budget", R
            ]
            slowest)
    |> Scope.chart "gpu_by_kernel" byKernel
    |> Scope.chart "timeline" timeline
    |> Scope.chart "timeline_zoom" timelineZoom
    |> Scope.chart "layer_stack" layerStack
    |> Scope.chart "dispatch_hist" dispatchHist
