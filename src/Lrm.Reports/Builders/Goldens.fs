/// The goldens report: one run of a golden tool (ReportTemplates contract, "goldens"; ReferenceOracles, "Telemetry";
/// GoldenSet). Reads the goldens.* records of tools/goldens: the committed size of each set against its budget and per
/// file, the byte-identical regeneration and the double run, the oracle self-tests, and for the tokenizer set the cases
/// per category and kind and the agreement of transformers' tokenizer.
module Lrm.Reports.Builders.Goldens

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports
open Lrm.Reports.Builders.Common

/// Builds the goldens scope.
let build (log: RunLog) : Scope =
    let records (name: string) (kind: TelemetryKind) =
        log.Named name |> List.filter (fun e -> e.Kind = kind)

    let setBytes = records "goldens.set_bytes" TelemetryKind.Gauge
    let fileBytes = records "goldens.file_bytes" TelemetryKind.Gauge
    let regen = records "goldens.regen_identical" TelemetryKind.Check
    let doubleRun = records "goldens.double_run_identical" TelemetryKind.Check

    let setOf (e: RunEvent) = defaultArg (e.TryStr "set") "?"
    let budgetOf (e: RunEvent) = e.TryFloat "budget"

    let overBudget =
        setBytes
        |> List.filter (fun e ->
            match budgetOf e with
            | Some b -> e.Value > b
            | None -> false)

    // Every goldens.* check, and the golden.compare self-tests (a `check` and no `tap`).
    let checks =
        [
            yield! log.Prefixed "goldens." |> List.filter (fun e -> e.Kind = TelemetryKind.Check)
            yield!
                log.Named TelemetryNames.GoldenCompare
                |> List.filter (fun e ->
                    e.Kind = TelemetryKind.Check
                    && (e.TryStr "tap").IsNone
                    && (e.TryStr "check").IsSome)
        ]
        |> List.sortBy (fun e -> e.Node, e.Seq)

    let failed = checks |> List.filter (fun e -> e.Value <> 1.0)

    let checkDetail (e: RunEvent) =
        e.Attrs
        |> List.filter (fun a -> a.Key <> TelemetryKeys.Component && a.Key <> "tool" && a.Key <> "direction")
        |> List.map (fun a -> a.Key + "=" + defaultArg (e.TryText a.Key) "")
        |> String.concat " "

    let verdictText =
        if setBytes.IsEmpty && checks.IsEmpty then
            "No golden-set records were found."
        elif failed.IsEmpty && overBudget.IsEmpty then
            let passing = plural checks.Length "check passes" "checks pass"
            Invariant.format $"**PASS.** {passing}, and every set is within its budget."
        else
            let parts =
                [
                    if not failed.IsEmpty then
                        Invariant.format
                            $"{Fmt.int (int64 failed.Length)} of {Fmt.int (int64 checks.Length)} checks fail"
                    if not overBudget.IsEmpty then
                        plural overBudget.Length "set is over its budget" "sets are over their budgets"
                ]

            "**FAIL.** " + String.concat "; " parts + "."

    let regenText =
        match regen with
        | [] -> "not checked (the tool wrote the set; `--check` compares a regeneration with the committed bytes)"
        | xs when xs |> List.forall (fun e -> e.Value = 1.0) -> "byte-identical to the committed files"
        | _ -> "**differs** from the committed files"

    let doubleText =
        match doubleRun with
        | [] -> "not recorded"
        | xs ->
            xs
            |> List.map (fun e ->
                let threads = e.TryText "threads" |> Option.defaultValue "?"

                (if e.Value = 1.0 then "identical" else "**different**")
                + " at "
                + threads
                + " thread"
                + (if threads = "1" then "" else "s"))
            |> String.concat ", "

    let setRows =
        [
            for e in setBytes ->
                let budget = budgetOf e

                [
                    setOf e
                    Fmt.bytes e.Value
                    (budget |> Option.map Fmt.bytes |> Option.defaultValue "n/a")
                    (budget
                     |> Option.map (fun b -> Fmt.pct (e.Value / b))
                     |> Option.defaultValue "n/a")
                    (match budget with
                     | Some b -> mark (e.Value <= b)
                     | None -> "n/a")
                ]
        ]

    let budgetChart =
        if setBytes.IsEmpty then
            noData "Committed bytes against the budget" "goldens.set_bytes gauge"
        else
            let sets = setBytes |> List.map setOf

            Chart.Bar
                {
                    Title = "Committed bytes against the budget"
                    Subtitle = "Per golden set (GoldenSet: tiny 5 MiB, tokenizer 2 MiB, qwen3-0.6b 10 MiB)"
                    CategoryLabel = "set"
                    ValueLabel = "bytes"
                    ValueFormat = NumFormat.Bytes
                    Categories = sets
                    Series =
                        [
                            {
                                Name = "committed"
                                Color = Color.Cat 0
                                Values = setBytes |> List.map _.Value
                            }
                            {
                                Name = "budget"
                                Color = Color.Neutral
                                Values = setBytes |> List.map (fun e -> budgetOf e |> Option.defaultValue nan)
                            }
                        ]
                    Stacked = false
                    Horizontal = true
                }

    let fileOf (e: RunEvent) = defaultArg (e.TryStr "file") "?"

    let filesChart =
        if fileBytes.IsEmpty then
            noData "Committed bytes per file" "goldens.file_bytes gauge"
        else
            Chart.Bar
                {
                    Title = "Committed bytes per file"
                    Subtitle = "Every file of the set; a committed golden file is at most 2 MiB"
                    CategoryLabel = "file"
                    ValueLabel = "bytes"
                    ValueFormat = NumFormat.Bytes
                    Categories = fileBytes |> List.map fileOf
                    Series =
                        [
                            {
                                Name = "bytes"
                                Color = Color.Cat 0
                                Values = fileBytes |> List.map _.Value
                            }
                        ]
                    Stacked = false
                    Horizontal = true
                }

    let fileRows =
        [
            for e in fileBytes -> [ fileOf e; Fmt.bytes e.Value; Fmt.int (int64 e.Value) ]
        ]

    let checkRows =
        [
            for e in checks ->
                let measured =
                    match e.TryFloat "measured", e.TryFloat "bound" with
                    | Some m, Some b -> Fmt.sci 2 m + " ≤ " + Fmt.sci 2 b
                    | Some m, None -> Fmt.sci 2 m
                    | _ -> ""

                [ e.Name; checkDetail e; measured; mark (e.Value = 1.0) ]
        ]

    let phaseRows =
        [
            for e in log.Prefixed "goldens." |> List.filter (fun e -> e.Kind = TelemetryKind.Span) ->
                [
                    e.Name
                    (e.TryStr "set" |> Option.defaultValue "")
                    Fmt.duration (float e.DurationNs)
                ]
        ]

    let corpus =
        let cats = records "goldens.tokenizer.cases" TelemetryKind.Gauge

        if cats.IsEmpty then
            []
        else
            let catOf (e: RunEvent) = defaultArg (e.TryStr "category") "?"
            let total = cats |> List.sumBy _.Value
            let kinds = records "goldens.tokenizer.case_kinds" TelemetryKind.Gauge

            let transformers =
                records "goldens.tokenizer.transformers" TelemetryKind.Check |> List.tryLast

            let agreement =
                match transformers with
                | None -> "Agreement with transformers' tokenizer was not recorded."
                | Some e ->
                    let agree = e.TryFloat "agree" |> Option.defaultValue nan
                    let differ = e.TryFloat "differ" |> Option.defaultValue nan

                    let cls =
                        e.TryStr "tokenizer_class" |> Option.defaultValue "transformers' tokenizer"

                    if e.Value = 1.0 then
                        Invariant.format
                            $"transformers' {cls} agrees with tokenizers on all {Fmt.int (int64 agree)} encode cases it was given."
                    else
                        Invariant.format
                            $"transformers' {cls} **differs** from tokenizers on {Fmt.int (int64 differ)} cases (agrees on {Fmt.int (int64 agree)})."

            let chart =
                Chart.Bar
                    {
                        Title = "Tokenizer cases per category"
                        Subtitle =
                            Invariant.format
                                $"{Fmt.int (int64 total)} cases in {Fmt.int (int64 cats.Length)} categories"
                        CategoryLabel = "category"
                        ValueLabel = "cases"
                        ValueFormat = NumFormat.Int
                        Categories = cats |> List.map catOf
                        Series =
                            [
                                {
                                    Name = "cases"
                                    Color = Color.Cat 0
                                    Values = cats |> List.map _.Value
                                }
                            ]
                        Stacked = false
                        Horizontal = true
                    }

            [
                Scope.empty
                |> Scope.value
                    "summary"
                    (Invariant.format $"{Fmt.int (int64 total)} cases in {Fmt.int (int64 cats.Length)} categories. "
                     + agreement)
                |> Scope.chart "categories" chart
                |> Scope.table
                    "kinds"
                    (table
                        [ "Kind", L; "Cases", R ]
                        [
                            for e in kinds -> [ defaultArg (e.TryStr "kind") "?"; Fmt.int (int64 e.Value) ]
                        ])
            ]

    runScope log
    |> Scope.value "verdict" verdictText
    |> Scope.value
        "sets"
        (if setBytes.IsEmpty then
             "none recorded"
         else
             setBytes
             |> List.map (fun e ->
                 setOf e
                 + " "
                 + Fmt.bytes e.Value
                 + (match budgetOf e with
                    | Some b -> " of " + Fmt.bytes b
                    | None -> ""))
             |> String.concat ", ")
    |> Scope.value "regen" regenText
    |> Scope.value "double_run" doubleText
    |> Scope.value
        "checks"
        (Invariant.format $"{Fmt.int (int64 (checks.Length - failed.Length))} of {Fmt.int (int64 checks.Length)} pass")
    |> Scope.chart "budget" budgetChart
    |> Scope.chart "files" filesChart
    |> Scope.table "sets" (table [ "Set", L; "Committed", R; "Budget", R; "Share", R; "Within budget", L ] setRows)
    |> Scope.table "files" (table [ "File", L; "Size", R; "Bytes", R ] fileRows)
    |> Scope.table "checks" (table [ "Check", L; "Detail", L; "Measured", R; "Result", L ] checkRows)
    |> Scope.table "phases" (table [ "Phase", L; "Set", L; "Duration", R ] phaseRows)
    |> Scope.list "corpus" corpus
