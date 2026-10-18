namespace Lrm.Dashboard

open System
open Lrm.Core
open Lrm.Telemetry
open Lrm.Reports

/// One run with a view: its report, rendered by the same template engine and builders as `lrm report` (or the
/// dashboard's records template for a kind without one), and its metering totals. The run log itself is not kept.
type RunView =
    {
        Entry: RunEntry
        /// The T1 template the report comes from, or None for the records template.
        Template: string option
        /// The report Markdown, byte-identical to what `lrm report` writes for the run (ReportTemplates contract).
        Markdown: string
        /// Every chart the Markdown links to: (file name under charts/, chart).
        Charts: (string * Chart) list
        /// The builder's top-level values other than run.*, by name (ordinal).
        Values: (string * string) list
        /// Resource use summed over every attribution key (RunLog contract, "Metering").
        Totals: MeterTotals
        /// Data records read.
        Records: int
        /// Every RunLog problem, prefixed by its node.
        Problems: string list
        /// The href of the Markdown report, when the run's root gives one.
        ReportHref: string option
        /// Set when the run log could not be read or rendered: the reason, in fixed words.
        Error: string option
    }

    /// The template name the charts are prefixed with.
    member v.TemplateName = defaultArg v.Template Records.Name

[<RequireQualifiedAccess>]
module RunView =
    /// The element id of a chart of a run: r-<run id>-<file stem>. Ids are unique on the page because run ids are
    /// unique and a report's chart files are.
    let chartId (runId: string) (file: string) =
        "r-" + runId + "-" + IO.Path.GetFileNameWithoutExtension file

    /// The chart's key within its report: the file stem without the template prefix ("rel_l2_by_layer",
    /// "pairs-0-heatmap"), the same for every run of a template.
    let chartKey (template: string) (file: string) =
        let stem = IO.Path.GetFileNameWithoutExtension file
        let prefix = template + "-"

        if stem.StartsWith(prefix, StringComparison.Ordinal) then
            stem.Substring prefix.Length
        else
            stem

    /// The view of a run whose log could not be read or rendered.
    let failed (root: RunsRoot) (entry: RunEntry) (reason: string) =
        {
            Entry = entry
            Template = Catalog.templateOf entry.Kind |> Option.map _.Name
            Markdown = ""
            Charts = []
            Values = []
            Totals = MeterTotals.Zero
            Records = 0
            Problems = []
            ReportHref = None
            Error = Some reason
        }

    /// Builds the view of a run from its log: the template's builder, the template text and the metering totals.
    let ofLog (root: RunsRoot) (entry: RunEntry) (log: RunLog) : RunView =
        let template = Catalog.templateOf entry.Kind

        let name, text, build =
            match template with
            | Some t -> t.Name, Reports.templateText t.Name, t.Build
            | None -> Records.Name, Records.templateText.Force(), Records.build

        let scope = build log
        let rendered = Template.render name (Template.parse name text) scope
        // The same trimming as Reports.renderWith, so the Markdown is the report's to the byte.
        let markdown = rendered.Markdown.TrimEnd('\n', ' ') + "\n"

        let totals =
            Metering.attribute log.Events
            |> List.fold (fun acc (_, t) -> acc + t) MeterTotals.Zero

        {
            Entry = entry
            Template = template |> Option.map _.Name
            Markdown = markdown
            Charts = rendered.Charts
            Values =
                scope.Values
                |> Map.toList
                |> List.filter (fun (k, _) -> not (k.StartsWith("run.", StringComparison.Ordinal)))
            Totals = totals
            Records = log.Events.Length
            Problems = log.Problems
            ReportHref = template |> Option.bind (fun t -> root.ReportHref entry.Dir t.Name)
            Error = None
        }

    /// Reads and builds the view of a run. A log that cannot be read, or a template that does not fit its data, gives
    /// a view that says so instead of stopping the dashboard.
    let load (root: RunsRoot) (entry: RunEntry) : RunView =
        use span =
            Telemetry.span "dashboard.view" [ Attr.str TelemetryKeys.Component "dashboard"; Attr.str "kind" entry.Kind ]

        match RunLogReader.load entry.Dir with
        | Error e -> failed root entry ("the run log cannot be read: " + e.Replace(entry.Dir, entry.Id))
        | Ok log ->
            try
                let v = ofLog root entry log
                span.Add(Attr.int "records" v.Records)
                span.Add(Attr.int "charts" v.Charts.Length)
                v
            with
            | TemplateError msg -> failed root entry ("template error: " + msg)
            | e -> failed root entry ("the report could not be built (" + e.GetType().Name + ")")
