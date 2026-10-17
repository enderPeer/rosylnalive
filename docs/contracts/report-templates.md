# ReportTemplates contract

Owner: T1. Status: v1, proposed with T1's PR; it changes only through a `contract-change` PR once it is tagged. Consumers: the parity report (M2b), the per-backend Gate 1 reports (I1), the cluster reports (C2, I2), the Gate 1 summary (I3), the dashboard (T2), and Phase 2 and 3 training, routing and thinking runs. Code: `src/Lrm.Reports`, `apps/Lrm.Cli/Commands/Report.fs`. Tests: `tests/Lrm.Telemetry.Tests` (cpu tier). Samples: `docs/reports/samples/`.

Reports are visual first: Markdown generated from templates, with SVG charts beside the tables that hold the same numbers. They are generated from run logs only (RunLog contract), never written by hand, and the same run log always renders to the same bytes.

## Pipeline

```
RunLog --builder (pure)--> Scope --template--> Markdown + charts --ChartSvg--> SVG files
```

1. `RunLogReader.load` reads the run folder.
2. The template's **builder** turns the RunLog into a `Scope`: named values, tables, charts and lists. A builder is a pure function of the RunLog. It reads no clock, no environment and no culture, and it never fails on missing records: a chart without data becomes a placeholder that names the records it needed.
3. The **template**, `src/Lrm.Reports/templates/<name>.md` (embedded in the assembly as `templates/<name>.md`), is filled from the Scope.
4. Each chart is rendered to SVG by `ChartSvg.render`.

`Reports.render name log` does steps 2 to 4 in memory and returns the Markdown and every SVG's text; `Reports.write outDir output` writes them.

## lrm report

```
lrm report <runId|path> [--template <name>] [--out <dir>]
lrm report --synthetic <name|all> [--out <dir>] [--runs <dir>]
lrm report --list
```

- `<runId|path>` is a run folder, or a run id under `LRM_DATA/runs`.
- `--template` picks the template; by default it is the run's `kind`. A kind with no template is an error that lists the templates.
- `--out` defaults to `<run>/report`.
- `--synthetic` writes synthetic runs (below) under `--runs` (default: a fresh temporary folder) and renders them; with `all` it also writes the index `README.md`. Its `--out` defaults to `docs/reports/samples`.
- Exit codes: 0 rendered; 1 no such run, a bad manifest, a template error or an I/O error; 2 a usage error. Problems found in the log are printed to stderr and do not change the exit code; they are in the report.

## Output

```
<out>/<template>.md
<out>/charts/<template>-<chart>.svg                  a chart of the top-level scope
<out>/charts/<template>-<list>-<i>-<chart>.svg       a chart inside {{#each list}}, item i (from 0)
```

Charts are linked relatively, `![<chart title>](charts/<file>)`, so a report folder can be moved or committed as a whole. Dots in names become underscores in file names. Every file is UTF-8 without a BOM, with LF line ends; the Markdown ends with exactly one LF.

## Templates

| Placeholder | Fills in |
| --- | --- |
| `{{value NAME}}` | text, as the builder formatted it |
| `{{table NAME}}` | a GitHub-flavoured Markdown table |
| `{{chart NAME}}` | a chart link (above) |
| `{{#each NAME}}` ... `{{/each}}` | the body once per item of a list. Names resolve in the item first, then outwards |
| `{{! comment }}` | nothing |

- **Names** match `[a-z][a-z0-9_.]*`. White space inside the braces is ignored: `{{ chart loss }}`.
- **Standalone lines.** A line that holds only an `{{#each}}`, `{{/each}}` or comment tag (and spaces or tabs) disappears with its line break, so blocks leave no blank lines.
- **Strict.** An unknown directive, a bad name, an unclosed `{{`, an unclosed comment, an unclosed `{{#each}}`, a stray `{{/each}}` and a name the scope does not define each raise `TemplateError` with the template name and line. A missing name is never an empty string.
- **Tables.** Header cells and alignment (`---`, `---:`, `:---:`) come from the builder. In cells, a pipe is escaped with a backslash and line breaks become spaces. Short rows are padded with empty cells. A table with no rows renders as `_No rows._`.
- **Line ends.** Templates are read with CRLF turned into LF.

**Vector T1.** The template

```
{{! a comment line disappears }}
# {{value title}}
{{#each items}}
- {{value name}} in {{value title}} {{chart c}}
{{/each}}
{{table t}}
{{ chart c }}
```

named `x`, with `title` = `T`, `items` = [`name` = `a`; `name` = `b`], table `t` with columns A (left) and B (right) and rows [`x|y`, `1`] and [`multi` LF `line`], and chart `c` titled `C [1]`, renders to

```
# T
- a in T ![C 1](charts/x-items-0-c.svg)
- b in T ![C 1](charts/x-items-1-c.svg)
| A | B |
| --- | ---: |
| x\|y | 1 |
| multi line |  |
![C 1](charts/x-c.svg)
```

## Common values

Every builder starts from `runScope`, so every template can use:

| Name | Kind | Content |
| --- | --- | --- |
| `run.id`, `run.title`, `run.kind`, `run.status`, `run.created` | value | From the manifest |
| `run.class`, `run.driver` | value | The manifest's execution class (`ExecutionClass.describe`) and driver |
| `run.banner` | value | For synthetic runs a block quote saying the numbers are not measurements, else empty |
| `run.problems` | value | Every RunLog problem as a bullet list, or `None.` |
| `run.summary` | table | Run, title, kind, status, times, class and class id, driver, commit, nodes, records, synthetic |
| `run.nodes` | table | Per node: role, execution class, records, finish status, dropped plus invalid records |
| `run.params` | table | The manifest's params |

Every template ends with these, so a reader can always tell what produced a report.

## The templates and what they read

Records are named by the TelemetryFacade's reserved names where one exists. Attributes in **bold** are required for the record to count; the others refine it. A component that emits these records with these attributes gets its report without further work.

| Template | Phase | Reads | Values | Tables | Charts |
| --- | ---: | --- | --- | --- | --- |
| `parity` | 1 | `golden.compare` checks: **`pair`**, **`case`**, **`tap`** (an ActivationNames name), **`metric`** (`rel_l2`, `max_abs`, ..., or `greedy_match` with the matched and required steps), **`measured`**, **`bound`**, `layer` (else parsed from `L<i>.` in the tap) | `verdict`, `checks`, `failed`, `pairs`, `cases`, `worst`, `closest.title` | `cases`, `closest` | `rel_l2_by_layer` (error bars), `ratio_histogram`, and per pair (`{{#each pairs}}`: `name`, `verdict`, `checks`, `failed`, `worst`) a `heatmap` of worst measured/bound by tap kind and layer |
| `inference` | 1 | `model.forward` spans: **`phase`** (`prefill` or `decode`), `step`, `tokens`; `model.layer` spans: **`layer`**, `phase` (children of `model.forward`); `model.flops` and `model.tokens` meters; `mem.arena` gauges: **`arena`**; every span, gpu and meter record for metering | `prefill_rate`, `decode_rate`, `ttft`, `flops_per_token`, `achieved` | `phases`, `arenas`, `meters` | `decode_latency` (line per node), `latency_hist`, `layer_time` (per-layer, prefill and decode, per token), `arena_peak` |
| `kernels` | 1 | `conformance.case` checks: **`kernel`**, `case`, `backend` (else the node's class), `measured`, `bound`, `detail`; `gpu.dispatch` gpu records: **`kernel`**, `layer`, `step`, `backend` | `verdict`, `budget`, `cases`, `failed`, `dispatches`, `gpu_total` | `matrix` (kernel by backend), `failures`, `slowest` | `gpu_by_kernel`, `layer_stack` (stacked by kernel family), `timeline` (the last step, stacked kernel timeline), `timeline_zoom` (its start), `dispatch_hist` |
| `cluster` | 1 | `cluster.unit` spans on worker nodes: **`stage`**, `step`, `unit`, `status` (`ok`, `duplicate`, `reissued`, `lost`); `cluster.step` spans on the coordinator: `step`; `cluster.node_lost` (`target`), `cluster.reissue` and `cluster.divergence` events; `cluster.replica_digest` checks; `cluster.*` byte meters | `verdict`, `units`, `reissued`, `lost`, `divergences`, `digests` | `nodes`, `events`, `meters` | `gantt` (stage Gantt chart with node-loss markers), `step_latency`, `busy` |
| `gate1` | 1 | `gate1.check` checks: **`group`** (`environment`, `tokenizer`, `cpuref`, `conformance`, `backend`, `cluster`, `determinism`), **`check`**, `backend`, `detail`; `golden.compare` checks with `tap` = `logits` and `metric` = `rel_l2`: **`case`**, **`backend`**, `measured`, `bound` | `verdict`, `passed`, `total` | `matrix` (group by backend), `checks` | `by_group`, `logits` (per gate prompt and backend) |
| `training` | 2 | gauges `train.loss`: **`step`**, `split` (`train` or `eval`), `stage` (curriculum stage); `train.grad_norm` and `train.update_ratio`: **`step`**, **`layer`**; `train.act_rms`: **`step`**, **`layer`**, **`site`** | `final_loss`, `steps` | `stages` | `loss` (with stage bands), `grad_heatmap` (layer by step), `grad_spread` (per-layer error bars), `update_hist`, `act_rms` (per-layer, by site) |
| `routing` | 3 | `route.usage` counters (tokens routed): **`layer`**, **`expert`** (`none` for no expert), **`step`**; `route.entropy` gauges (nats): **`layer`**, **`step`** | `experts`, `layers`, `step` | `top` | `usage_heatmap` (expert by layer), `expert_mix` (stacked per layer), `entropy_by_layer` (per checkpoint), `entropy_over_steps` |
| `thinking` | 3 | `roslyn.call` spans: **`api`**, `layer`, `token`, `outcome` (`ok`, `diagnostic`, `timeout`, `error`), `detail`; `think.step` events: `kind`, `detail` | `calls`, `ok_rate`, `layers_calling`, `steps` | `apis`, `calls` (the first 40) | `trace` (calls on API lanes, coloured by outcome, thinking steps as markers), `by_api`, `latency` |

Names: `gate1.*`, `train.*`, `route.*`, `roslyn.*` and `think.*` are the prefixes these reports expect from I1 to I3 and from Phase 2 and 3. Arena names follow the heap layout in lowercase: `weights`, `rope`, `kv_cache`, `activations`, `scratch`, `logits`, `taps` (then `master`, `grad`, `adam_m`, `adam_v`, `roslyn_in`, `roslyn_out`).

**Adding a metric** (dev plan, visual-first rule): a PR that adds a metric also adds it to a template. Within T1's templates that is a builder change plus a placeholder; for new report kinds, T1 adds the template on request.

## Charts

| Kind | Record | Used for |
| --- | --- | --- |
| line | `LineChart` | metrics over steps, several series; with `IntegerX` it is the per-layer multi-series chart |
| bar | `BarChart` | grouped or stacked, horizontal bars or vertical columns |
| heatmap | `Heatmap` | layer by step, expert by layer, tap by layer; linear or log colour scale |
| histogram | `Histogram` | pre-binned counts; `Builders.Common.bins` bins samples, optionally in log10 space |
| timeline | `Timeline` | lanes of intervals with event markers: the stacked kernel timeline, the cluster stage Gantt chart, the Roslyn call trace |
| error bars | `ErrorBarChart` | per-layer spread: a dot at the middle value, whiskers from low to high, several series side by side |
| placeholder | `Chart.Empty` | a chart without data: its title and the reason |

**Form.** Every chart is 720 px wide with its own surface (8 px rounded card), a title (15 px, semibold), an optional subtitle, a legend for two or more series (never for one), a y-axis label above the plot, and recessive 1 px solid gridlines. Lines are 2 px with round joins; markers are 8 px dots with a 2 px surface ring, drawn when a series has at most 40 points; bars are at most 24 px thick with a 4 px rounded data end and a square base; touching bars and stacked segments are separated by 2 px of surface. Direct labels mark line ends only for 2 to 4 series whose ends are at least 14 px apart. Text never wears a series colour. A log axis has decade ticks; reference lines (bounds) are 1 px lines with a label.

**Palette.** Categorical slots in this fixed order, light and dark steps:

| Slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hue | blue | orange | aqua | yellow | magenta | green | violet | red |
| Light | `#2a78d6` | `#eb6834` | `#1baf7a` | `#eda100` | `#e87ba4` | `#008300` | `#6250d6` | `#e34948` |
| Dark | `#3987e5` | `#d95926` | `#199e70` | `#c98500` | `#d55181` | `#008300` | `#9085e9` | `#e66767` |

- **Colour follows the entity, not its rank.** Builders assign slots from fixed orders: backends in ExecutionClass order (cpu-f32, cpu-f64, cpu-device, vulkan, d3d12, d3d12-warp), ToleranceSpec pairs in contract order, kernel families (matmul, attention, norm, rope, elementwise, memory, reduce). A single series uses slot 0. A ninth entity, or one outside the order, is grey (`Neutral`, "other").
- **Status colours** (good `#0ca30c`, warning `#fab219`, serious `#ec835a`, critical `#d03b3b`) mark state only (pass and fail, unit outcomes, call outcomes) and always come with a legend or text label.
- **Sequential ramp** for heatmaps: 13 blue steps from `#cde2fb` to `#0d366b`; on the dark surface the same steps run the other way, so values near zero recede towards the surface. Missing cells are surface-grey.
- **Themes.** Each SVG embeds one `<style>` with the light colours and a `@media (prefers-color-scheme: dark)` block with the dark ones, so the chart follows the viewer's setting and reads on a light or a dark page either way. Surfaces: light `#fcfcfb`, dark `#1a1a19`.
- **Validated.** The categorical steps pass the colour-vision checks in both modes (worst adjacent CVD delta E 9.1 light and 8.4 dark, normal-vision 19.6 and 19.3). Three light steps (aqua, yellow, magenta) are below 3:1 on the light surface, which is why every chart has a legend and every report puts the numbers in a table beside the chart.

**Accessibility and hover.** The root is `<svg role="img" aria-labelledby="<id>-title <id>-desc">` with a `<title>` and a one-sentence `<desc>`; `id` is the file stem, so several charts can be inlined in one page (the dashboard). Marks carry `<title>` tooltips with their values.

**Determinism.**
- The same chart renders to the same bytes on every machine and under any culture.
- Coordinates are rounded to 0.01 and printed in invariant form, never as `-0`; non-finite values are never printed.
- Axis layout and the synthetic data use `DetMath` (logarithm, exponential, powers of ten from IEEE add, subtract, multiply and divide only) instead of `Math.Log`, `Math.Exp` and `Math.Pow`, whose last bits come from the platform's C runtime.
- Text widths are estimated from fixed per-character classes, not measured with a font.
- Sorting is ordinal, and iteration follows record order or a fixed order.
- No chart or report contains the time it was rendered.

## Numbers

`Fmt` prints every number in reports, in the invariant culture: `dp 2 3.14159` = `3.14`; `sci 2 0.000321` = `3.21e-04`; `num` picks separators below 10^6 (`1,234`), four significant digits (`0.5000`) or scientific outside [10^-3, 10^6) (`1.23e+06`); `int 1234567` = `1,234,567`; `si 1.25e9` = `1.25G`; `siWith "FLOP/s" 1.56e12` = `1.56 TFLOP/s`; `bytes` uses binary units (`1.50 GiB`); `duration` takes ns (`3.40 ms`, `800 ns`); `pct 0.1234` = `12.3%`. A value that rounds to zero has no sign, and non-finite values print as `NaN`, `inf` and `-inf`. Axis ticks pick one display unit per axis (`0.5 GiB`, `100 µs`, `40%`) and steps of 1, 2, 2.5 or 5 times a power of ten in it.

## Synthetic runs and samples

`Synthetic.write kind runsRoot` writes `runsRoot/synthetic-<kind>` for every template through the real `RunContext` and sink, on a virtual clock starting at 2026-10-18T09:00:00Z, with numbers from DetRng (seed `0x4C524D2D54315359`, stream `DetRng.streamOf "synthetic.<kind>"`) shaped with DetMath. The runs are plausible for Qwen3-0.6B and the RTX 4080 but made up; their manifests say `synthetic: true` and every report shows the banner. They include the cases a reviewer needs to see: a comparison at about 80% of its bound, a failing conformance case, a killed node with a reissue, a Roslyn call timeout and an error.

`docs/reports/samples/` holds every template rendered from its synthetic run plus a `README.md` index, regenerated with

```
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report --synthetic all --out docs/reports/samples
```

The files are never edited by hand. A cpu-tier test renders them afresh and fails when the committed folder differs in any byte, so the samples always show what the code renders.

## Verification

The cpu tier of `Lrm.Telemetry.Tests` checks:
- vector T1, every template error, CRLF templates and empty lists;
- that every chart kind, fed NaN, infinities and non-positive values on log axes, renders to SVG that parses as XML, with a title first, a description, a `viewBox`, and no `NaN` or `Infinity` in it;
- that charts are byte-identical when rendered twice and under de-DE and tr-TR, and carry both themes and every palette step;
- the `Fmt` vectors under the invariant culture, de-DE and tr-TR, and `DetMath` against the platform functions (relative error below 1e-15 for ln, 4e-15 for exp);
- that the synthetic runs are byte-identical when written again and under de-DE, and that each is a clean log of its kind;
- that every template renders its synthetic run identically twice and under de-DE and tr-TR, with the banner, no placeholder left, every chart linked and parsing, and data in every chart;
- that every template renders a run with no records, each chart a placeholder;
- that the committed samples are byte-identical to a fresh render;
- that every template is registered, embedded and parses, with a synthetic run for each.

The bench tier times rendering per template.
