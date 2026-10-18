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
4. Each chart is rendered to SVG by `ChartSvg.render`. It never throws: a chart that cannot be drawn becomes a placeholder that names the failure, so one bad series never stops a report.

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
- `--synthetic` writes synthetic runs (below) under `--runs` and renders them; with `all` it also writes the index `README.md`. Its `--out` defaults to `docs/reports/samples`. Without `--runs` the runs go to a fresh temporary folder that is deleted afterwards.
- Paths it prints are full paths.
- Exit codes: 0 rendered; 1 no such run, a bad manifest, a template error or an I/O error; 2 a usage error. Problems found in the log are printed to stderr and do not change the exit code; they are in the report.

## Output

```
<out>/<template>.md
<out>/charts/<template>-<chart>.svg                  a chart of the top-level scope
<out>/charts/<template>-<list>-<i>-<chart>.svg       a chart inside {{#each list}}, item i (from 0)
```

Charts are linked relatively, `![<chart title>](charts/<file>)`, so a report folder can be moved or committed as a whole. Dots in names become underscores in file names. Every file is UTF-8 without a BOM, with LF line ends; the Markdown ends with exactly one LF.

A render replaces the previous render of the same template: `Reports.write` first deletes `charts/<template>-*.svg`, so the folder holds what the run log renders to now and no chart left over from an earlier render or template version. Files of other templates are untouched.

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
| `parity` | 1 | `golden.compare` checks: **`pair`**, **`case`** (else `prompt`, the ComparatorsAndGoldenReader key; else `(<kind>)` such as `(gradient)`), **`tap`** (an ActivationNames name; else `tensor`), **`metric`** (`rel_l2`, `max_abs`, ..., or `greedy_match` with the matched and required steps; camelCase names such as ComparatorsAndGoldenReader's `relL2` and `maxAbs` read as their snake_case form), **`measured`**, **`bound`** (absent for a measurement such as an e_ref, which then has no ratio and is charted only by value), `layer` (a negative `layer`, ComparatorsAndGoldenReader's −1, marks a model-level tap with no layer; without the key it is parsed from `L<i>.` in the tap). **Oracle self-tests** (ReferenceOracles, "Telemetry"): `golden.compare` checks with a **`check`** and no `tap`, and `goldens.selftest` checks (**`check`**); they count in the verdict apart from the pairs' comparisons, in `worst`, the histogram and `closest`, and in their own table, never as a pair or a case. **Greedy margins** of golden tools: `goldens.greedy.min_margin` gauges (**`prompt`**, **`oracle`**), `goldens.greedy.min_margin_f32_ulps` gauges and `goldens.greedy.f32_argmax_agrees` checks (same keys) | `verdict`, `checks`, `failed` (the pairs' comparisons), `pairs`, `cases`, `worst`, `closest.title` | `cases`, `closest` (every failing comparison, at most 40, else the 15 closest to their bound); `{{#each selftests}}` (one item when the run has self-tests): `summary`, table `selftests`; `{{#each greedy}}` (one item when the run has margins): `summary`, table `margins` | `rel_l2_by_layer` (error bars), `ratio_histogram`, and per pair (`{{#each pairs}}`: `name`, `verdict`, `checks`, `failed`, `worst`) a `heatmap` of worst measured/bound by tap kind and layer, its scale ending at the bound, or, for a pair whose per-layer checks have no bound (e_ref measurements), of the measured rel_l2 by value; `margins` in `{{#each greedy}}` (the smallest margin per prompt and oracle) |
| `inference` | 1 | `model.forward` spans: **`phase`** (`prefill` or `decode`), `step`, `tokens`; `model.layer` spans: **`layer`**, `phase` (children of `model.forward`); `model.flops` and `model.tokens` meters; `mem.arena` gauges: **`arena`**; every span, gpu and meter record for metering | `prefill_rate`, `decode_rate`, `ttft`, `flops_per_token`, `achieved` | `phases`, `arenas`, `meters` | `decode_latency` (line per node), `latency_hist`, `layer_time` (per-layer, prefill and decode, per token), `arena_peak` |
| `kernels` | 1 | `conformance.case` checks: **`kernel`**, `case`, `backend` (else the node's class), `measured`, `bound`, `detail`; `gpu.dispatch` gpu records: **`kernel`**, `layer`, `step`, `backend`; `gpu.fp32_denormals` events: **`mode`**, `backend` (else the node's class), `blob` (a probe on another blob, such as a control), and the probe's words `inputs`, `results`, `expected_preserve`, `expected_flush` (GpuBackendInterface, `Fp32Denormals.recordProbe`) | `verdict`, `budget`, `cases`, `failed`, `dispatches`, `gpu_total` | `matrix` (kernel by backend), `failures`, `slowest`, `denormals` (the last measured mode per backend, node and blob, with what bit-exact means in it), `denormal_words` (the probe's bit table: each input, the preserve and flush results, and one column per measurement) | `gpu_by_kernel`, `layer_stack` (stacked by kernel family), `timeline` (the last step, stacked kernel timeline), `timeline_zoom` (its start), `dispatch_hist` |
| `cluster` | 1 | C1's records (ClusterProtocol, "Telemetry"), with the worker in `node`: `cluster.unit.done` events: **`lane`**, **`stage`**, **`step`**, **`node`**, **`start_ns`**, **`end_ns`** (the coordinator's clock; the record's own time minus `end_ns` aligns it with the run log's clock); `cluster.reissue` counters (`lane`, `stage`, `step`, `node`: the unit is drawn as reissued); `cluster.assign`, `cluster.duplicate` and `cluster.deadline.miss` counters (`stage`, `node`); `cluster.replay.*`, `cluster.frame.rejected` (`reason`) and `cluster.chaos` (`fault`, `node`) counters; `cluster.member.leave`, `cluster.demote` and `cluster.chaos` kill and restart as markers; `cluster.deadline` gauges (`stage`); `cluster.unit.latency` histograms; the last `cluster.run` event (`status`, `units`, `epoch`, `cross_checked`, `unconfirmed`, `run_digest`); every meter for metering | `verdict` (PASS when `status` is `completed` with no divergence; UNVERIFIED, FAIL or INCOMPLETE otherwise), `status`, `units`, `cross_checked`, `reissued`, `lost`, `divergences`, `epoch`, `digest` | `nodes` (per node and stage: backend, assigned, accepted, reissued to it, duplicates, deadline misses, accepted time), `counters`, `events` (every `cluster.*` event but `unit.done` and `ready`, and kills, at most 40), `meters` | `gantt` (accepted units per node and stage, coloured accepted or reissued, with markers), `step_latency` (per lane, first stage's assignment to last stage's acceptance), `deadlines` (per stage, as steps), `latency` (log histogram), `busy` |
| `goldens` | 1 | The `goldens.*` records of `tools/goldens` (ReferenceOracles, "Telemetry"): `goldens.set_bytes` gauges (**`set`**, `budget` in bytes), `goldens.file_bytes` gauges (**`file`**), `goldens.regen_identical` and `goldens.double_run_identical` (`set`, `threads`) checks, every other `goldens.*` check and the `golden.compare` self-tests; for a tokenizer run `goldens.tokenizer.cases` (**`category`**) and `goldens.tokenizer.case_kinds` (**`kind`**) gauges and the `goldens.tokenizer.transformers` check (`agree`, `differ`, `tokenizer_class`); `goldens.*` spans as phases | `verdict` (PASS when every check passes and every set is within its budget), `sets`, `regen`, `double_run`, `checks`; `{{#each corpus}}` (one item for a tokenizer run): `summary` | `sets`, `files`, `checks`, `phases`; `kinds` in `{{#each corpus}}` | `budget` (committed bytes against the budget per set), `files` (bytes per file); `categories` in `{{#each corpus}}` (cases per category) |
| `gate1` | 1 | `gate1.check` checks: **`group`** (`environment`, `tokenizer`, `cpuref`, `conformance`, `backend`, `cluster`, `determinism`), **`check`**, `backend`, `detail`; `golden.compare` checks with `tap` = `logits` and `metric` = `rel_l2` (or `relL2`): **`case`** (else `prompt`), **`backend`**, `measured`, `bound` | `verdict`, `passed`, `total` | `matrix` (group by backend), `checks` | `by_group`, `logits` (per gate prompt and backend) |
| `training` | 2 | gauges `train.loss`: **`step`**, `split` (`train` or `eval`), `stage` (curriculum stage); `train.lr`: **`step`**; `train.grad_norm` and `train.update_ratio`: **`step`**, **`layer`**; `train.act_rms`: **`step`**, **`layer`**, **`site`** | `final_loss`, `last_step`, `steps` (the count of training steps, with its noun) | `stages` | `loss` (with stage bands), `lr` (with stage bands), `grad_heatmap` (layer by step, columns binned past 120), `grad_spread` (per-layer error bars), `update_hist`, `act_rms` (per-layer, by site) |
| `routing` | 3 | `route.usage` counters (tokens routed): **`layer`**, **`expert`** (`none` for no expert), **`step`**; `route.entropy` gauges (nats): **`layer`**, **`step`** | `experts`, `layers`, `step` | `top` | `usage_heatmap` (expert by layer), `expert_mix` (stacked per layer), `entropy_by_layer` (per checkpoint), `entropy_over_steps` |
| `thinking` | 3 | `roslyn.call` spans: **`api`**, `layer`, `token`, `outcome` (`ok`, `diagnostic`, `timeout`, `error`), `detail`; `think.step` events: `kind`, `detail` | `calls`, `ok_rate`, `layers_calling`, `steps` | `apis`, `calls` (the first 40) | `trace` (calls on API lanes, coloured by outcome, thinking steps as markers), `by_api`, `latency` |

Names: `gate1.*`, `train.*`, `route.*`, `roslyn.*` and `think.*` are the prefixes these reports expect from I1 to I3 and from Phase 2 and 3. Arena names follow the heap layout in lowercase: `weights`, `rope`, `kv_cache`, `activations`, `scratch`, `logits`, `taps` (then `master`, `grad`, `adam_m`, `adam_v`, `roslyn_in`, `roslyn_out`).

**Adding a metric** (dev plan, visual-first rule): a PR that adds a metric also adds it to a template. Within T1's templates that is a builder change plus a placeholder; for new report kinds, T1 adds the template on request.

**NaN and infinity are results, not gaps.** A NaN or infinite measurement is the commonest GPU failure, so no builder drops it. In `parity` an invalid comparison sorts first in the failure table (NaN, then infinity, then finite ratios from the largest, then `greedy_match`), is the `worst` value, is a crossed critical cell in the heatmap, and is a cross in the NaN/inf strip of the error bars; the histogram counts it in its NaN/inf column. `gate1` logits do the same in their error bars.

## Charts

| Kind | Record | Used for |
| --- | --- | --- |
| line | `LineChart` | metrics over steps, several series; with `IntegerX` it is the per-layer multi-series chart |
| bar | `BarChart` | grouped or stacked, horizontal bars or vertical columns |
| heatmap | `Heatmap` | layer by step, expert by layer, tap by layer; linear or log colour scale |
| histogram | `Histogram` | pre-binned counts; `Builders.Common.bin` bins samples, optionally in log10 space, and counts what has no place on the axis |
| timeline | `Timeline` | lanes of intervals with event markers: the stacked kernel timeline, the cluster stage Gantt chart, the Roslyn call trace |
| error bars | `ErrorBarChart` | per-layer spread: a dot at the middle value, whiskers from low to high, several series side by side; `Builders.Common.spread` builds a point and counts its NaN and infinite values |
| placeholder | `Chart.Empty` | a chart without data: its title and the reason |

**Form.** Every chart is 720 px wide with its own surface (8 px rounded card), a title (15 px, semibold), an optional subtitle, a legend for two or more series (never for one), a y-axis label above the plot, and recessive 1 px solid gridlines. Lines are 2 px with round joins; markers are 8 px dots with a 2 px surface ring, drawn when a series has at most 40 points; bars are at most 24 px thick with a 4 px rounded data end and a square base; touching bars and stacked segments are separated by 2 px of surface. Direct labels mark line ends only for 2 to 4 series whose ends are at least 14 px apart. Text never wears a series colour. A log axis has decade ticks; reference lines (bounds) are 1 px lines with a label.

**Values that have no place on an axis** are drawn, never dropped:
- **Exact zeros on a log scale** are the best outcome of a comparison (bit-exact). Line and error-bar charts put them in a strip labelled `0` below the decades; a heatmap draws an outlined empty cell with the tooltip `0 (exact)` and a key entry; a histogram binned in log10 space counts them (`Binned.Zeros`) and draws them as a column labelled `0` left of the axis.
- **NaN and infinite values** are drawn in the critical colour: a heatmap cell (`Some nan` or an infinity) is crossed, an error-bar point with `Invalid > 0` gets a cross in a strip labelled `NaN/inf` above the decades, and a histogram's `Invalid` count (NaN, infinite, or negative in log10 space) is a column labelled `NaN/inf` at the right. A line series breaks at a non-finite point.
- **Missing heatmap cells** (`None`) are surface-grey, apart from both of the above, with a `no data` key entry when present.

**Heatmap scale and size.** A `Ceiling` (value, label) fixes the top of the colour scale, such as (1.0, `over bound`) for ratios to a tolerance bound: the ramp runs from the smallest value to the ceiling, so darker means closer to it, and cells above it are critical with that label in the key. More than 120 columns (`ChartSvg.MaxHeatColumns`) are binned into at most 120, each the `Aggregate` (`Max` or `Mean`) of its columns, and the subtitle says so (`columns bin 167 steps each (max)`). With `ColValues` (the numeric position of every column, such as its step) the column axis gets nice ticks instead of column labels.

**Long series.** A line series is drawn through the first, lowest, highest and last point of every pixel column, so a path never holds more than about four points per pixel and every extreme survives; markers are drawn only for series of at most 40 points. A 20,000-step training log renders its charts in about 0.5 MB.

**Timeline outcomes.** An interval in a status colour other than good (warning, serious, critical) is an outcome to notice: it is drawn at least 4 px wide, outlined, and on top of the event markers. Marker labels take two staggered rows above the plot; a label that fits in neither is left to the marker's tooltip.

**Extreme magnitudes.** Layout clamps magnitudes to 1e300 (and positive values on a log axis to at least 1e-300), and a span too small to divide (subnormal values) falls back to a unit span, so `±Double.MaxValue`, subnormals and counts near `Int32.MaxValue` render as charts inside their frame.

**Text.** Every string from a log is escaped for XML: control characters are dropped and characters XML cannot hold (U+FFFE, U+FFFF, unpaired surrogates) become U+FFFD. A label cut to fit never splits a surrogate pair.

**Palette.** Categorical slots in this fixed order, light and dark steps:

| Slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hue | blue | orange | aqua | yellow | magenta | green | violet | red |
| Light | `#2a78d6` | `#eb6834` | `#1baf7a` | `#eda100` | `#e87ba4` | `#008300` | `#6250d6` | `#e34948` |
| Dark | `#3987e5` | `#d95926` | `#199e70` | `#c98500` | `#d55181` | `#008300` | `#9085e9` | `#e66767` |

- **Colour follows the entity, not its rank.** Builders assign slots from fixed orders: backends in ExecutionClass order (cpu-f32, cpu-f64, cpu-device, vulkan, d3d12, d3d12-warp), ToleranceSpec pairs in contract order, kernel families (matmul, attention, norm, rope, elementwise, memory, reduce). A single series uses slot 0. A ninth entity, or one outside the order, is grey (`Neutral`, "other").
- **Status colours** (good `#0ca30c`, warning `#fab219`, serious `#ec835a`, critical `#d03b3b`) mark state only (pass and fail, unit outcomes, call outcomes) and always come with a legend or text label.
- **Sequential ramp** for heatmaps: 13 blue steps from `#cde2fb` to `#0d366b`; on the dark surface the same steps run the other way, so values near zero recede towards the surface. Missing cells are surface-grey.
- **Themes.** Each SVG embeds one `<style>` with the light colours and a `@media (prefers-color-scheme: dark)` block with the dark ones, so the chart follows the viewer's setting and reads on a light or a dark page either way. Surfaces: light `#fcfcfb`, dark `#1a1a19`. Axis ticks and labels are `#6f6e69` on light (4.98:1) and `#898781` on dark (4.85:1).
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

`Fmt` prints every number in reports, in the invariant culture: `dp 2 3.14159` = `3.14`; `sci 2 0.000321` = `3.21e-04`; `num` picks separators below 10^6 (`1,234`), four significant digits (`0.5000`) or scientific outside [10^-3, 10^6) (`1.23e+06`); `int 1234567` = `1,234,567`; `si 1.25e9` = `1.25G`; `siWith "FLOP/s" 1.56e12` = `1.56 TFLOP/s`; `bytes` uses binary units (`1.50 GiB`); `duration` takes ns (`3.40 ms`, `800 ns`); `pct 0.1234` = `12.3%`. A value that rounds to zero has no sign, and non-finite values print as `NaN`, `inf` and `-inf`. Linear axis ticks pick one display unit per axis (`0.5 GiB`, `100 µs`, `40%`) and steps of 1, 2, 2.5 or 5 times a power of ten in it. Decade axes (log scales, and histograms binned in log10 space with `NumFormat.Pow10` or `NumFormat.Decades`) tick each decade in one style per axis: plain decimals when every tick lies within 10^-2 to 10^3 (`0.01`, `0.1`, `1`, `10`), powers of ten otherwise (`10⁻⁶`, `10⁻³`, `10⁰`), and for `Decades Duration` each decade in its own unit (`100 ns`, `1 µs`, `10 µs`, `1 ms`). An axis over at most three decades also ticks 2 and 5 times each decade when its labels are plain or in units (`2 µs`, `5 µs`).

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
- that every chart kind, fed NaN, infinities and non-positive values on log axes, renders to SVG that parses as XML, with a title first, a description, a `viewBox`, no `NaN` printed as a number and no `Infinity`, and every positioned element and line point inside its frame;
- that `±Double.MaxValue`, subnormals and huge counts render as charts inside their frame;
- that exact zeros on log scales stay visible in every chart form, that NaN and infinite cells are crossed and apart from missing ones, and that a ratio heatmap's scale ends at its ceiling;
- that a heatmap of 1,000 columns is binned to 112 with nice step ticks, that a 20,000-point line keeps its extremes in under 3,000 path points, and that a 20,000-step, 28-layer training log renders under 1.5 MB;
- that the parity report lists NaN and infinite failures first and marks them in its heatmap, error bars and histogram;
- that SVG text holds only characters XML allows, and that cuts never split a surrogate pair;
- that a render replaces the charts of the previous render of its template;
- that timeline outcomes to notice are at least 4 px wide and drawn over the markers, and that marker labels stagger;
- that decade axes use one style and their units;
- that the samples exercise every column and series they show (work assigned to every cluster node and its backend, GPU time in inference metering, all six routing checkpoints, the learning rate);
- that charts are byte-identical when rendered twice and under de-DE and tr-TR, and carry both themes and every palette step;
- the `Fmt` vectors under the invariant culture, de-DE and tr-TR, and `DetMath` against the platform functions (relative error below 1e-15 for ln, 4e-15 for exp);
- that the synthetic runs are byte-identical when written again and under de-DE, and that each is a clean log of its kind;
- that every template renders its synthetic run identically twice and under de-DE and tr-TR, with the banner, no placeholder left, every chart linked and parsing, and data in every chart;
- that every template renders a run with no records, each chart a placeholder;
- that the committed samples are byte-identical to a fresh render;
- that every template is registered, embedded and parses, with a synthetic run for each.

The bench tier times rendering per template.
