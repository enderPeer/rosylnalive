# ADR T2-0001: The dashboard is one self-contained HTML file rendered from run logs by T1's templates

- **Status:** proposed (T2, 2026-10-18), with the T2 pull request.
- **Owner:** T2.
- **Scope:** `lrm dashboard` (`apps/Lrm.Cli/Commands/Dashboard.fs`), the `Lrm.Dashboard` library and what its page shows; what I3's `docs/reports/gate1/dashboard.html` will be.
- **Code:** `src/Lrm.Dashboard`. **Tests:** `tests/Lrm.Dashboard.Tests` (cpu tier). **Committed pages:** `tests/Lrm.Dashboard.Tests/reports/`.

## Context

Observability is visual first (CLAUDE.md): Markdown reports with SVG charts from templates, and an HTML dashboard over the same run logs. The plan gives T2 one acceptance line ("the snapshot is deterministic, and the page works with no network; the PR includes a screenshot of a real run") and no contract of its own; T2 consumes RunLog and ReportTemplates. Two of its choices bind others:

- I3 writes `docs/reports/gate1/dashboard.html` with it, so the page format, its inputs and its determinism are I3's to rely on.
- The page has to show the same charts as the reports, so it depends on T1's chart SVG and theme in a specific way, written down here so a change on T1's side is caught.

Constraints: the RunLog contract makes run logs the only input of reports and the dashboard. Nothing may be downloaded, so no chart library, font or CDN. The repo's size check holds every committed file to 1 MiB.

## Decision

1. **One command, one file.** `lrm dashboard [--runs <dir>]... [--synthetic [--samples <dir>]] [--out <file.html>] [--watch [--interval <s>]] [--per-kind <n>] [--run <runId>]... [--max-log-mb <n>] [--write-reports]` writes one HTML file with inline CSS, JavaScript and SVG. It makes no network request: no `src` attribute, `<link>`, `@import` or `url(`; every `href` is a fragment, a relative path or a `file:` URI; the only `http` text is the SVG namespace name. Defaults: `--runs LRM_DATA/runs` (none when `--synthetic` is given alone), `--out LRM_DATA/dashboard.html`, `--interval 3`, `--per-kind 3`, `--max-log-mb 64`. Exit codes: 0 written, 1 a missing runs folder or an I/O error, 2 a usage error.
2. **Inputs are run folders only**, read through T1's `RunManifest.parse` and `RunLogReader.load`. A folder without a readable manifest is listed as not readable, with a reason in fixed words that never holds a path. A run id seen under an earlier `--runs` root shadows the same id under a later one.
3. **Views** (switched by URL fragment; without JavaScript through `:target`):
   - **Runs** (`#runs`): every readable run, newest first, with start (UTC), run id, kind, status (a dot and the word), duration, node count and execution class; counts by status; a "Runs by kind and status" bar chart; a filter by kind, status and text (JavaScript).
   - **Kind** (`#kind-<kind>`): one per report template, Phase 2 and 3 included, plus one per other kind present. A placeholder says when a kind has no measured runs yet (Phase 2: training; Phase 3: routing and thinking). It shows counts, a "Duration of each run" chart (the 40 newest finished runs, by status), its comparisons and its 15 newest runs.
   - **Run** (`#run-<runId>`): the run's report, rendered from the same template text, builder and chart records as `lrm report` (see 4), preceded by its metering totals (wall self time, GPU time, FLOPs, bytes moved, tokens, metered time when non-zero; `Metering.attribute` summed over every key), its duration, nodes, records and log-problem count, a link to the Markdown report, and its comparisons. A kind without a T1 template gets the dashboard's records view (`src/Lrm.Dashboard/templates/records.md`): checks passed and failed per name, time per span name, every record name with what it adds up to, and the metering table.
   - **Compare** (`#compare`, `#cmp-<older>~<newer>`): two runs of one kind side by side: the run fields (status, duration, nodes, records, class, driver, commit, problems), the builder's top-level values, the metering totals with change and ratio, and every chart of A beside the same chart of B (paired by the chart file name minus the template prefix). Pairs are every pair among the newest 4 runs with a view of each kind; synthetic runs are paired only with synthetic runs. The chart pairs are clones of the run views' figures, made by the script when the view opens, so a comparison adds no chart bytes; without JavaScript each slot links to the chart.
4. **The charts are the reports' charts.** A run view's Markdown is `Template.render` over the template's own text and builder, trimmed like `Reports.renderWith`, so it is byte-identical to the report. Each chart is `ChartSvg.render ("r-" + runId + "-" + stem)` of the same `Chart`, so ids stay unique on one page, with its `<style>` element (exactly `"<style>" + Theme.style + "</style>\n"`) left out because the page carries `Theme.style` once. Replacing the id prefix back and putting the style back gives the report's SVG file byte for byte; a test checks this for every chart of every view.
5. **Themes.** Page colours are tokens on `:root`, with a dark set under `prefers-color-scheme: dark` (unless the toggle chose light) and under `[data-theme=dark]`. The toggle cycles auto, light and dark and is remembered in `localStorage` (guarded). Charts keep T1's own light and dark rules; the page adds T1's light rules under `:root[data-theme=light] .d-chart` and its dark rules under `:root[data-theme=dark] .d-chart`, both derived from `Theme.style` at build time, so the toggle wins over the viewer's preference in both directions.
6. **Phone width.** At 375 px nothing scrolls the page sideways: charts scale to the width (a click or Enter shows one at full size inside its own scroll box), tables scroll inside their box, the runs tables become cards with column names, and stat tiles take two columns.
7. **Determinism.** The same run folders give the same bytes, under any culture and any thread count: no clock, no environment, no absolute path except a `file:` report link for a run outside the page's folder, invariant formatting (`Fmt`, `Invariant`), ordinal sorting. UTF-8 without a byte order mark, LF line ends.
8. **Watch mode** (`--watch`) polls the runs folders every `--interval` seconds (folder names, and the size and write time of every manifest and events file), rebuilds when they change and writes only when the bytes change, through a temporary file and a rename. Only a watch-mode page carries `<meta http-equiv="refresh">` and `data-watch`; its script keeps the scroll position across reloads.
9. **Size guards.** Only the newest `--per-kind` runs of each kind get a view (plus every `--run` and every synthetic run); runs whose events files exceed `--max-log-mb` get none unless named with `--run` (the 400 MB tokenizer bench logs). Run logs are read on at most 4 threads.
10. **Markdown subset.** The renderer handles what the templates produce: ATX headings (shifted one level, ids `<view>-<slug>`), paragraphs, block quotes, bullet and numbered lists, GitHub tables with alignment, fenced code, rules, images alone on a line, inline code, bold, italic (an underscore inside a word is text), links and backslash escapes. Every source character is escaped; raw HTML never passes. A link to `http:` or `https:` stays text. Vector MD1 below pins it.
11. **Telemetry.** The build reports a `dashboard.render` span (component `dashboard`; `runs`, `views`, `comparisons`, `chars`) and one `dashboard.view` span per run read (`kind`, `records`, `charts`) through the ambient facade, so a host that records a run meters the dashboard's wall time with no further work.
12. **Committed pages.** `samples.html` is the dashboard over T1's synthetic runs (`lrm dashboard --synthetic --out tests/Lrm.Dashboard.Tests/reports/samples.html`, from the repo root). A cpu-tier test renders it afresh and fails when the committed file differs in any byte. `runs.html` is the dashboard over the real wave 1 and 2 runs in `LRM_DATA/runs` at the time of the PR (a view of the newest run of each kind and of four chosen wave 2 runs), regenerated by hand with the command in `tests/Lrm.Dashboard.Tests/reports/README.md`. They are two files because one page with both is over the 1 MiB file limit (1.02 MiB with one view per kind).

### Vector MD1

`Markdown.render { HeadingShift = 1; IdPrefix = "v"; Figure = (an image of charts/x-c.svg becomes <figure>alt</figure>) }` of

```
# Title *x*

> **Synthetic.** a `b|c` <tag>

**PASS.** rel_l2 of L3.q_proj and _em_ and a\*b
next line

- one
- two [link](#x) [net](https://example.com/a)

| A | B | C |
| --- | ---: | :---: |
| x\|y | 1 | `c` |
| short |

![Chart [1]](charts/x-c.svg)

![missing](charts/none.svg)

## Title *x*
```

is exactly

```html
<h2 id="v-title-x">Title <em>x</em></h2>
<blockquote>
<p><strong>Synthetic.</strong> a <code>b|c</code> &lt;tag&gt;</p>
</blockquote>
<p><strong>PASS.</strong> rel_l2 of L3.q_proj and <em>em</em> and a*b
next line</p>
<ul>
<li>one</li>
<li>two <a href="#x">link</a> net (<code>https://example.com/a</code>)</li>
</ul>
<div class="d-scroll"><table class="d-table">
<thead><tr><th>A</th><th class="d-r">B</th><th class="d-c">C</th></tr></thead>
<tbody>
<tr><td>x|y</td><td class="d-r">1</td><td class="d-c"><code>c</code></td></tr>
<tr><td>short</td><td class="d-r"></td><td class="d-c"></td></tr>
</tbody></table></div>
<figure>Chart [1]</figure>
<p class="d-muted">missing</p>
<h3 id="v-title-x-2">Title <em>x</em></h3>
```

## Evidence

On the development machine (Windows 11, .NET SDK 10.0.402), at the PR's head:

- `LRM_TEST_TIER=cpu dotnet artifacts/bin/Lrm.Dashboard.Tests/release/Lrm.Dashboard.Tests.dll`: 40 tests pass, among them vector MD1, byte-identical pages twice, under de-DE and tr-TR and on one thread, the committed samples page against a fresh render, every chart and report against `Reports.render`, unique ids with every fragment link, compare slot and label resolving, every inline SVG parsing as XML, the self-containment rules of decision 1, watch mode rewriting on a new run and not on an idle poll, and markup in titles and attributes staying text.
- Every view, loaded in a 375 px iframe in headless Edge 155 (the headless window cannot be narrower than 496 px): `scrollWidth` equals `clientWidth` in all 57 views of the dashboard over `LRM_DATA/runs` plus the synthetic runs, in all 28 of `runs.html` and in all 20 of `samples.html`.
- Screenshots of a real run's view (desktop and 375 px), the dark theme, and the toggle's light-on-dark and dark-on-light overrides were checked by eye; the PR shows them.
- Sizes: `samples.html` 684 KiB; `runs.html` 868 KiB; the real runs with one view per kind plus the synthetic runs 1.02 MiB, over the limit. The page over all 327 runs builds in about 2.5 s.

## Consequences

- **I3** renders `docs/reports/gate1/dashboard.html` with `lrm dashboard --runs <gate1 runs> --out docs/reports/gate1/dashboard.html` and can rely on decisions 1, 3 and 7.
- **T1:** a change to the chart `<style>` element, to `Theme.style`'s two blocks or to how `ChartSvg.render` uses its id shows up in this repo's cpu tier (the chart identity test and `Assets.chartCss`, which refuses a `Theme.style` without its dark block).
- **Reopened by:** a contract for the dashboard (if I3 or the integrator want one, these decisions move into `docs/contracts/dashboard.md`), or a raised file limit for the committed pages, which would let one page hold the synthetic and the real runs.

## Alternatives

- **Static compare pages with their own charts.** Every pair would carry its charts again: with three views per kind, about three times the chart bytes. Cloning the run views' figures costs nothing.
- **Charts as `<img src="data:...">`.** Isolates each SVG's style and keeps its bytes, but drops the marks' hover titles and adds base64 overhead.
- **A data blob rendered by JavaScript.** Smaller, but the page would need JavaScript for everything, the charts would be re-implemented in the browser, and the F# tests could not check what the reader sees.
- **Reading `artifacts/hygiene/summary.json` directly** (S0b's routed note): the RunLog contract makes run logs the only input of the dashboard, so the PR asks for `hygiene.ps1` to write a run log instead (kind `hygiene`), which the records view shows without changes.
