namespace Lrm.Reports

open System
open System.IO
open System.Reflection
open System.Text
open Lrm.Core
open Lrm.Telemetry

/// One report template: its Markdown file (embedded from src/Lrm.Reports/templates/<name>.md) and its builder.
type ReportTemplate =
    {
        /// The template name, also the run kind that selects it by default.
        Name: string
        /// One line for `lrm report --list` and the samples index.
        Description: string
        /// The roadmap phase whose runs feed it.
        Phase: int
        /// The builder: a pure function of the run log.
        Build: RunLog -> Scope
    }

/// The report registry and renderer (ReportTemplates contract).
[<RequireQualifiedAccess>]
module Reports =
    /// Every template, in report order.
    let templates: ReportTemplate list =
        [
            {
                Name = "parity"
                Description = "Agreement with the goldens, tap by tap and layer by layer"
                Phase = 1
                Build = Builders.Parity.build
            }
            {
                Name = "inference"
                Description = "Prefill and decode speed, per-layer time, memory per arena and metering"
                Phase = 1
                Build = Builders.Inference.build
            }
            {
                Name = "kernels"
                Description = "The conformance matrix and GPU time by kernel, layer and dispatch"
                Phase = 1
                Build = Builders.Kernels.build
            }
            {
                Name = "cluster"
                Description = "The stage Gantt chart, node loss and reissue, step latency and per-node load"
                Phase = 1
                Build = Builders.Cluster.build
            }
            {
                Name = "gate1"
                Description = "Every Gate 1 check by group and backend, and logits agreement per gate prompt"
                Phase = 1
                Build = Builders.Gate1.build
            }
            {
                Name = "training"
                Description = "Loss per curriculum stage, gradient and activation statistics per layer"
                Phase = 2
                Build = Builders.Training.build
            }
            {
                Name = "routing"
                Description = "Roslyn expert usage and router entropy per layer"
                Phase = 3
                Build = Builders.Routing.build
            }
            {
                Name = "thinking"
                Description = "Roslyn calls made from inside the model, as a trace with outcomes"
                Phase = 3
                Build = Builders.Thinking.build
            }
        ]

    /// The template with this name.
    let tryFind (name: string) =
        templates |> List.tryFind (fun t -> Invariant.equals t.Name name)

    /// The names of every template.
    let names = templates |> List.map _.Name

    type private Marker = class end

    /// The text of an embedded template (templates/<name>.md), with LF line ends.
    let templateText (name: string) =
        let asm = typeof<Marker>.Assembly
        use stream = asm.GetManifestResourceStream("templates/" + name + ".md")

        if isNull stream then
            invalidArg "name" (sprintf "no embedded template templates/%s.md" name)

        use reader = new StreamReader(stream, UTF8Encoding(false, true))
        reader.ReadToEnd().Replace("\r\n", "\n")

    /// One rendered report: the Markdown and every chart's SVG text, by file name under charts/.
    type Output =
        {
            Template: string
            Markdown: string
            Svgs: (string * string) list
        }

    /// Renders `template` over a run log, in memory. Pure: the same log gives the same bytes under any culture.
    let renderWith (template: ReportTemplate) (text: string) (log: RunLog) : Output =
        use _ = Telemetry.span "report.render" [ Attr.str "template" template.Name ]
        let nodes = Template.parse template.Name text
        let rendered = Template.render template.Name nodes (template.Build log)

        let markdown =
            let md = rendered.Markdown.TrimEnd('\n', ' ')
            md + "\n"

        {
            Template = template.Name
            Markdown = markdown
            Svgs =
                rendered.Charts
                |> List.map (fun (file, chart) -> file, ChartSvg.render (Path.GetFileNameWithoutExtension file) chart)
        }

    /// Renders the embedded template `name` over a run log.
    let render (name: string) (log: RunLog) : Output =
        match tryFind name with
        | Some t -> renderWith t (templateText name) log
        | None -> invalidArg "name" (sprintf "unknown template '%s' (known: %s)" name (String.concat ", " names))

    let private utf8 = UTF8Encoding(false)

    /// Writes <outDir>/<template>.md and <outDir>/charts/<file>.svg; returns the paths written, Markdown first. The
    /// render replaces the previous render of the same template: its chart files (charts/<template>-*.svg) are deleted
    /// first, so the folder holds what this run log renders to and nothing left over from an earlier render.
    let write (outDir: string) (o: Output) : string list =
        let charts = Path.Combine(outDir, "charts")
        Directory.CreateDirectory charts |> ignore
        let prefix = o.Template + "-"

        for stale in Directory.GetFiles charts do
            let name = Path.GetFileName stale

            if
                name.StartsWith(prefix, StringComparison.Ordinal)
                && name.EndsWith(".svg", StringComparison.Ordinal)
            then
                File.Delete stale

        let md = Path.Combine(outDir, o.Template + ".md")
        File.WriteAllText(md, o.Markdown, utf8)

        md
        :: [
            for file, svg in o.Svgs ->
                let path = Path.Combine(charts, file)
                File.WriteAllText(path, svg, utf8)
                path
        ]

    /// The template a run uses by default: the one named by its kind.
    let defaultTemplate (log: RunLog) = tryFind log.Manifest.Kind

    /// The index page of a samples folder: one row per template with its description and a link.
    let samplesIndex (outputs: Output list) =
        let sb = StringBuilder()

        sb.Append(
            "# Sample reports\n\n"
            + "Rendered by `lrm report --synthetic all --out docs/reports/samples` from synthetic run logs, so the report "
            + "visuals can be reviewed before real runs exist. Every number here is made up; each report says so in its "
            + "banner. The files are regenerated, never edited: a cpu-tier test fails when they differ from a fresh "
            + "render.\n\n"
            + "| Report | Phase | Shows | Charts |\n| --- | ---: | --- | ---: |\n"
        )
        |> ignore

        for o in outputs do
            match tryFind o.Template with
            | Some t ->
                sb.Append(
                    Invariant.format $"| [{t.Name}]({t.Name}.md) | {t.Phase} | {t.Description} | {o.Svgs.Length} |\n"
                )
                |> ignore
            | None -> ()

        sb.ToString()

    /// Writes the synthetic run of every template under `runsRoot`, renders each one and writes the reports and the
    /// index (README.md) to `outDir`. Returns the outputs in template order.
    let renderSamples (runsRoot: string) (outDir: string) : Output list =
        let outputs =
            Synthetic.kinds
            |> List.map (fun kind ->
                let dir = Synthetic.write kind runsRoot
                render kind (RunLogReader.loadOrFail dir))

        Directory.CreateDirectory outDir |> ignore

        for o in outputs do
            write outDir o |> ignore

        File.WriteAllText(Path.Combine(outDir, "README.md"), samplesIndex outputs, utf8)
        outputs
