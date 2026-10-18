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

1. **One command, one file.** `lrm dashboard [--runs <dir>]... [--synthetic [--samples <dir>]] [--out <file.html>] [--watch [--interval <s>]] [--per-kind <n>] [--run <runId>]... [--max-log-mb <n>] [--until <utc time>] [--write-reports]` writes one HTML file with inline CSS, JavaScript and SVG. It makes no network request: no `src` attribute, `<link>`, `@import` or `url(`; every `href` passes `Html.isLocalHref`, which allows a fragment, a `file:///` URI not followed by another slash, or a relative path that names no scheme, does not start with a slash and holds no backslash, space or control character (a browser turns `\\host`, `/<tab>/host` and ` //host` into the network path `//host`, which a `file:` page opens as an SMB request); the page itself writes no `file:` URI and no absolute path (decision 7); the only `http` text is the SVG namespace name. `--until` keeps only the runs that started at or before a UTC time, so a committed snapshot stays the same while new runs are written. Defaults: `--runs LRM_DATA/runs` (none when `--synthetic` is given alone), `--out LRM_DATA/dashboard.html`, `--interval 3`, `--per-kind 3`, `--max-log-mb 64`, no `--until`. Exit codes: 0 written, 1 a missing runs folder or an I/O error, 2 a usage error.
2. **Inputs are run folders only**, read through T1's `RunManifest.parse` and `RunLogReader.load`. A folder without a readable manifest, or one whose files change or go away while it is listed, is listed as not readable, with a reason in fixed words that never holds a path. The dashboard opens a manifest with read, write and delete sharing, so a live run can still replace it with a rename while the dashboard reads it. A run id seen under an earlier `--runs` root shadows the same id under a later one.
3. **Views** (switched by URL fragment; without JavaScript through `:target`):
   - **Runs** (`#runs`): every readable run, newest first, with start (UTC), run id, kind, status (a dot and the word), duration, node count and execution class; counts by status; a "Runs by kind and status" bar chart; a filter by kind, status and text (JavaScript). The chart's count axis uses `NumFormat.Int` only where T1's tick step is whole (`Page.countFormat`: the largest count is at least 5 and its fifth is not in (2, 2.5]); otherwise `NumFormat.Si`, whose ticks print fractional steps as they are (Int rounds each label, which printed "0 0 0 1 1 1" for one run per kind). A cpu-tier test checks the labels for every count from 1 to 60.
   - **Kind** (`#kind-<kind>`): one per report template, Phase 2 and 3 included, plus one per other kind present. A placeholder says when a kind has no measured runs yet (Phase 2: training; Phase 3: routing and thinking). It shows counts, a "Duration of each run" chart (the 40 newest finished runs as dots on a log axis, coloured by status: one kind's runs last from a millisecond to hours, and a linear axis flattened all but the longest), its comparisons and its 15 newest runs.
   - **Run** (`#run-<runId>`): the run's report, rendered from the same template text, builder and chart records as `lrm report` (see 4), preceded by its duration, nodes and records, a tile for each metering total that is not zero (wall self time, GPU time, FLOPs, bytes moved, tokens, metered time; `Metering.attribute` summed over every key) with the zero ones named in one line, a log-problem tile when there are problems, where the Markdown report is, and its comparisons. A kind without a T1 template gets the dashboard's records view (`src/Lrm.Dashboard/templates/records.md`): checks passed and failed per name, time per span name, every record name with what it adds up to, and the metering table.
   - **Compare** (`#compare`, `#cmp-<older>~<newer>`): two runs of one kind side by side: the run fields (status, duration, nodes, records, class, driver, commit, problems), the builder's top-level values, the metering totals with change and ratio, and every chart of A beside the same chart of B (paired by the chart file name minus the template prefix). Pairs are every pair among the newest 4 runs with a view of each kind; synthetic runs are paired only with synthetic runs. The chart pairs are clones of the run views' figures, made by the script when the view opens, so a comparison adds no chart bytes; without JavaScript each slot links to the chart.
4. **The charts are the reports' charts.** A run view's Markdown is `Template.render` over the template's own text and builder, trimmed like `Reports.renderWith`, so it is byte-identical to the report. Each chart is `ChartSvg.render ("r-" + runId + "-" + stem)` of the same `Chart`, so ids stay unique on one page, with its `<style>` element (exactly `"<style>" + Theme.style + "</style>\n"`) left out because the page carries `Theme.style` once. Replacing the id prefix back and putting the style back gives the report's SVG file byte for byte; a test checks this for every chart of every view.
5. **Themes.** Page colours are tokens on `:root`, with a dark set under `prefers-color-scheme: dark` (unless the toggle chose light) and under `[data-theme=dark]`. Muted text is at least 4.5:1 against every surface it sits on (4.78 light, 4.80 dark, on the table-head surface). The toggle cycles auto, light and dark and is remembered in `localStorage` (guarded). Charts keep T1's own light and dark rules; the page adds T1's light rules under `:root[data-theme=light] .d-chart` and its dark rules under `:root[data-theme=dark] .d-chart`, both derived from `Theme.style` at build time, so the toggle wins over the viewer's preference in both directions.
6. **Phone width.** At 375 px nothing scrolls the page sideways: charts scale to the width (a click or Enter shows one at full size inside its own scroll box), tables scroll inside their box, the runs tables become cards with column names, stat tiles take three columns, and a nav pill or kind tag never gets wider than the page (a long kind name ends in an ellipsis; its `title` holds it all). Below 640 px wide, or 500 px high, the header is not sticky, and below 640 px wide the nav is one row of pills that scrolls sideways inside itself (the script brings the current pill into it), so a run view's verdict and first chart come before the fold.
7. **Determinism.** The same run folders give the same bytes, under any culture and any thread count: no clock, no environment, no absolute path, invariant formatting (`Fmt`, `Invariant`), ordinal sorting. UTF-8 without a byte order mark, LF line ends. A run inside the page's folder links its report with a relative href; a run outside it names the report as text, `<runs folder name>/<runId>/report/<template>.md` ("beside the run log"), because its only link would be an absolute `file:` URI, which would put the user's path into the page and make a committed page (I3's `docs/reports/gate1/dashboard.html`) differ from machine to machine.
8. **Watch mode** (`--watch`) polls the runs folders every `--interval` seconds (folder names, and the size and write time of every manifest and events file), rebuilds when they change and writes only when the bytes change, through a temporary file and a rename. An I/O error while the folders are listed or read, or while the page is written, is retried at the next poll and never ends the watch. A watch-mode page carries `data-watch="<interval>"`; its script reloads it with `location.reload()` every interval, which keeps the fragment of the view the reader navigated to. A `<meta http-equiv="refresh">` would reload the URL the page was opened with, and Chromium drops the fragment (every reload went back to the runs index, and a page opened with a fragment did not reload at all), so the meta refresh sits only inside `<noscript>`. The reload waits while the tab is hidden, a form field has focus, a chart is zoomed or text is selected (the live marker turns amber and says so), and the scroll position and the runs filter are kept across it (`sessionStorage`, guarded).
9. **Size guards.** Only the newest `--per-kind` runs of each kind get a view (plus every `--run` and every synthetic run); runs whose events files exceed `--max-log-mb` get none unless named with `--run` (the 400 MB tokenizer bench logs). Run logs are read on at most 4 threads.
10. **Markdown subset.** The renderer handles what the templates produce: ATX headings (shifted one level, ids `<view>-<slug>`), paragraphs, block quotes, bullet and numbered lists, GitHub tables with alignment, fenced code, rules, images alone on a line, inline code, bold, italic (an underscore inside a word is text), links and backslash escapes. Every source character is escaped; raw HTML never passes. A link whose target fails `Html.isLocalHref` (decision 1: any scheme but a local `file:///`, a network path, a leading slash, a backslash, a space or a control character) stays text: its label, then the target in code. Values from the run log (the title, attributes) reach this renderer through the templates, so this is what keeps a run log from putting a network link into the page. Vector MD1 below pins the rest.
11. **Telemetry.** The build reports a `dashboard.render` span (component `dashboard`; `runs`, `views`, `comparisons`, `chars`) and one `dashboard.view` span per run read (`kind`, `records`, `charts`) through the ambient facade, so a host that records a run meters the dashboard's wall time with no further work.
12. **Committed pages.** `samples.html` is the dashboard over T1's synthetic runs (`lrm dashboard --synthetic --out tests/Lrm.Dashboard.Tests/reports/samples.html`, from the repo root). A cpu-tier test renders it afresh and fails when the committed file differs in any byte. `runs.html` is the dashboard over the real wave 1 and 2 runs in `LRM_DATA/runs` that started by 2026-10-18 14:00 UTC (`--until`; a view of the newest run of each kind and of four chosen wave 2 runs), regenerated by hand with the command in `tests/Lrm.Dashboard.Tests/reports/README.md`. They are two files because one page with both is over the 1 MiB file limit.
13. **Accessibility.** The page title is the one `<h1>`; every view starts with an `<h2 tabindex="-1">`, and a switch to another view moves focus to it (a target inside a view keeps the browser's own focus). A skip link, the first Tab stop, moves focus to the current view's heading without changing the view (it is hidden without JavaScript, where its fragment would switch to the runs index). Every chart figure is a toggle button: `role="button"`, `aria-pressed`, `aria-label="Zoom chart: <chart title>"` and `aria-describedby` the chart's own description; Enter, Space or a click toggles it. A compare slot's clone gets every id and id reference suffixed.

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

On the development machine (Windows 11, .NET SDK 10.0.402, Edge 155), at the PR's head:

- `LRM_TEST_TIER=cpu dotnet artifacts/bin/Lrm.Dashboard.Tests/release/Lrm.Dashboard.Tests.dll`: 50 tests pass, among them vector MD1, byte-identical pages twice, under de-DE and tr-TR and on one thread, the committed samples page against a fresh render, every chart and report against `Reports.render`, unique ids with every fragment link, compare slot, `aria-labelledby` and `aria-describedby` resolving, every inline SVG parsing as XML, the self-containment rules of decision 1 (every href local, no `file:` URI, no absolute path), network-path link targets (`//host`, `\\host`, `/<tab>/host`, `file://host`) staying text from Markdown and from a run's title, the same bytes for a page written from another folder, `--until` keeping a snapshot's bytes after a later run, the count axis labels for every count from 1 to 60, the durations on a log axis, the long-kind pill, the h1, skip link, focusable headings and chart toggle buttons, the watch page with no meta refresh outside `<noscript>`, watch mode rewriting on a new run and a grown log and not on an idle poll, and outliving its runs folder being deleted and recreated. The suite passed 30 times in a row.
- Watch mode in headless Edge (CDP, a page written by `lrm dashboard --watch --interval 3`): after a click on the `#kind-tokenizer` pill, the page reloaded 3 times in 10 s and stayed on `#kind-tokenizer` (`performance.timeOrigin` changed; the meta refresh it replaced went back to the runs index every time). A page opened as `watch.html#kind-cluster` reloaded too and kept its fragment. The scroll position and the runs filter survived a reload; a focused search field and a zoomed chart each held the reload for 7 s (the live marker turned amber), and the reload followed once the field lost focus.
- Every view at 375 by 812 px (device emulation), in the light and the dark theme: `scrollWidth` equals `clientWidth` in all 28 views of `runs.html`, all 20 of `samples.html` and all 19 of a page whose runs include kind `a-very-long-kind-name-for-an-experimental-benchmark-suite-x`. The header is 100 px high and not sticky at 375 by 812 (was 200 px, sticky) and at 812 by 375; a run view's first report content starts at 496 to 607 px, above the fold.
- Focus: a mouse click on a pill moves focus to the view's heading with no visible ring; Enter on a pill moves it with the ring; the skip link is the first Tab stop and Enter on it focuses the current view's heading; Enter on a focused chart toggles its zoom and `aria-pressed`.
- Screenshots of a real run's view (desktop and 375 px), the runs index, the parity kind view with its log-axis durations, and the dark theme were checked by eye; the PR shows them.
- Sizes: `samples.html` 691 KiB; `runs.html` 886 KiB (396 runs that started by 14:00 UTC); the page over all runs grows with every run, so a committed page needs `--until`. The page over about 400 runs builds in about 2.7 s.

## Consequences

- **I3** renders `docs/reports/gate1/dashboard.html` with `lrm dashboard --runs <gate1 runs> --out docs/reports/gate1/dashboard.html` and can rely on decisions 1, 3 and 7: the committed page holds no local path and the same runs give the same bytes on any machine and from any clone. Its file must stay under the 1 MiB limit: `--per-kind 1`, `--until` and `--max-log-mb` bound it.
- **T1:** a change to the chart `<style>` element, to `Theme.style`'s two blocks or to how `ChartSvg.render` uses its id shows up in this repo's cpu tier (the chart identity test and `Assets.chartCss`, which refuses a `Theme.style` without its dark block). The count-axis test also fails if bar axes change their tick steps.
- **Reopened by:** a contract for the dashboard (if I3 or the integrator want one, these decisions move into `docs/contracts/dashboard.md`), a raised file limit for the committed pages, which would let one page hold the synthetic and the real runs, or integer tick steps for `NumFormat.Int` bar axes in T1, which would let `Page.countFormat` go.

## Alternatives

- **Static compare pages with their own charts.** Every pair would carry its charts again: with three views per kind, about three times the chart bytes. Cloning the run views' figures costs nothing.
- **Charts as `<img src="data:...">`.** Isolates each SVG's style and keeps its bytes, but drops the marks' hover titles and adds base64 overhead.
- **A data blob rendered by JavaScript.** Smaller, but the page would need JavaScript for everything, the charts would be re-implemented in the browser, and the F# tests could not check what the reader sees.
- **Reading `artifacts/hygiene/summary.json` directly** (S0b's routed note): the RunLog contract makes run logs the only input of the dashboard, so the PR asks for `hygiene.ps1` to write a run log instead (kind `hygiene`), which the records view shows without changes.
- **A meta refresh for watch mode** (the first version). It needs no script, but Chromium reloads the URL the page was opened with and drops the fragment, so the reader was sent back to the runs index every interval. It stays only as the `<noscript>` fallback.
- **`file:` links to reports outside the page's folder, or relative links that climb out of it.** Both work on the machine that rendered the page, but the first puts the user's path into the page and the second ties its bytes to where the clone and `LRM_DATA` sit; a committed page has to be the same everywhere. A `--link-base` option would let a reader choose, but nothing needs it yet.
- **A headline chart per kind from the builders' values** (verdict, rel_l2 against its bound, tok/s, loss across runs). Only runs with a view have builder values (`--per-kind`, 3 by default), and reading every run's log for each kind view would undo the size guards of decision 9 (the 400 MB tokenizer logs). The durations chart colours each run by status, which shows pass and fail across runs; a per-kind trend of a headline value belongs in a T1 builder that reads it from a summary record.
