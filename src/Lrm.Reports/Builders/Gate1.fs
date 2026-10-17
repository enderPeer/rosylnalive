/// The Gate 1 report: every gate check by group and backend, and the logits agreement per gate prompt
/// (ReportTemplates contract, "gate1"). Reads gate1.check checks and logits golden.compare checks.
module Lrm.Reports.Builders.Gate1

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Check groups in the order lrm gate1 runs them.
let groups =
    [
        "environment"
        "tokenizer"
        "cpuref"
        "conformance"
        "backend"
        "cluster"
        "determinism"
    ]

/// Builds the gate1 scope.
let build (log: RunLog) : Scope =
    let checks =
        log.Named "gate1.check" |> List.filter (fun e -> e.Kind = TelemetryKind.Check)

    let groupOf (e: RunEvent) = defaultArg (e.TryStr "group") "other"

    let backendOfCheck (e: RunEvent) =
        defaultArg (e.TryStr TelemetryKeys.Backend) ""

    let passed = checks |> List.filter (fun e -> e.Value = 1.0)
    let ok = not checks.IsEmpty && passed.Length = checks.Length
    let groupNames = checks |> List.map groupOf |> sortIn groups

    let checkRows =
        checks
        |> List.sortBy (fun e -> orderIn groups (groupOf e), e.Node, e.Seq)
        |> List.map (fun e ->
            [
                groupOf e
                defaultArg (e.TryStr "check") e.Name
                (match backendOfCheck e with
                 | "" -> "all"
                 | b -> b)
                mark (e.Value = 1.0)
                defaultArg (e.TryStr "detail") ""
            ])

    let byGroup =
        if checks.IsEmpty then
            noData "Checks by group" "gate1.check"
        else
            Chart.Bar
                {
                    Title = "Gate checks by group"
                    Subtitle = "Passed and failed checks per group"
                    CategoryLabel = "group"
                    ValueLabel = "checks"
                    ValueFormat = NumFormat.Int
                    Categories = groupNames
                    Series =
                        [
                            "passed", Color.Good, (fun (e: RunEvent) -> e.Value = 1.0)
                            "failed", Color.Critical, (fun (e: RunEvent) -> e.Value <> 1.0)
                        ]
                        |> List.map (fun (name, color, pred) ->
                            {
                                Name = name
                                Color = color
                                Values =
                                    groupNames
                                    |> List.map (fun g ->
                                        checks
                                        |> List.filter (fun e -> groupOf e = g && pred e)
                                        |> List.length
                                        |> float)
                            })
                    Stacked = true
                    Horizontal = true
                }

    let logits =
        log.Named TelemetryNames.GoldenCompare
        |> List.filter (fun e ->
            e.Kind = TelemetryKind.Check
            && e.TryStr "tap" = Some "logits"
            && e.TryStr "metric" = Some "rel_l2")

    let prompts =
        logits
        |> List.choose (fun e -> e.TryStr TelemetryKeys.Case)
        |> List.distinct
        |> List.sort

    let logitBackends =
        logits
        |> List.map (fun e -> defaultArg (e.TryStr TelemetryKeys.Backend) "")
        |> sortIn backendOrder

    let logitsChart =
        if logits.IsEmpty then
            noData "Logits rel_l2 per gate prompt" "logits rel_l2 golden.compare"
        else
            Chart.ErrorBars
                {
                    Title = "Logits rel_l2 per gate prompt"
                    Subtitle = "Dot: worst step of the prompt; whiskers: best to worst step. Line: the bound."
                    XLabel = "prompt"
                    YLabel = "rel_l2 vs torch32 (log)"
                    YFormat = NumFormat.Sci
                    YLog = true
                    Categories = prompts
                    Series =
                        logitBackends
                        |> List.map (fun b ->
                            {
                                Name = b
                                Color = colorIn backendOrder b
                                Spreads =
                                    prompts
                                    |> List.mapi (fun i p ->
                                        let xs =
                                            logits
                                            |> List.filter (fun e ->
                                                e.TryStr TelemetryKeys.Case = Some p
                                                && defaultArg (e.TryStr TelemetryKeys.Backend) "" = b)
                                            |> List.choose (fun e -> e.TryFloat "measured")

                                        {
                                            Index = i
                                            Low = minOr nan xs
                                            Mid = maxOr nan xs
                                            High = maxOr nan xs
                                        })
                            })
                    RefLines =
                        [
                            let b = logits |> List.choose (fun e -> e.TryFloat "bound") |> maxOr nan

                            if Double.IsFinite b then
                                {
                                    Value = b
                                    Label = "bound " + Fmt.sci 1 b
                                }
                        ]
                }

    let matrixBackends =
        checks
        |> List.map backendOfCheck
        |> List.filter (fun b -> b <> "")
        |> sortIn backendOrder

    let matrix =
        table
            (("Group", L) :: ("All", L) :: (matrixBackends |> List.map (fun b -> b, L)))
            [
                for g in groupNames ->
                    let cell (pred: RunEvent -> bool) =
                        let xs = checks |> List.filter (fun e -> groupOf e = g && pred e)

                        if xs.IsEmpty then
                            "–"
                        else
                            let good = xs |> List.filter (fun e -> e.Value = 1.0)

                            (if good.Length = xs.Length then "✓ " else "✗ ")
                            + Invariant.format $"{good.Length}/{xs.Length}"

                    g
                    :: cell (fun e -> backendOfCheck e = "")
                    :: [ for b in matrixBackends -> cell (fun e -> backendOfCheck e = b) ]
            ]

    runScope log
    |> Scope.value
        "verdict"
        (if checks.IsEmpty then
             "No gate checks were recorded."
         elif ok then
             Invariant.format $"**Gate 1: PASS.** {checks.Length} of {checks.Length} checks pass."
         else
             Invariant.format $"**Gate 1: FAIL.** {checks.Length - passed.Length} of {checks.Length} checks fail.")
    |> Scope.value "passed" (Fmt.int (int64 passed.Length))
    |> Scope.value "total" (Fmt.int (int64 checks.Length))
    |> Scope.table "matrix" matrix
    |> Scope.table "checks" (table [ "Group", L; "Check", L; "Backend", L; "Result", L; "Detail", L ] checkRows)
    |> Scope.chart "by_group" byGroup
    |> Scope.chart "logits" logitsChart
