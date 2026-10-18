# Dashboard pages

Rendered by `lrm dashboard` (ADR T2-0001). Each file is one self-contained HTML page: open it in a browser, no server and no network needed.

| File | What it shows | Regenerate (from the repo root) |
| --- | --- | --- |
| `samples.html` | T1's synthetic run of every report template, Phase 2 and 3 included. Every number is made up; each run says so in its banner. A cpu-tier test renders it afresh and fails when this file differs in any byte | `dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll dashboard --synthetic --out tests/Lrm.Dashboard.Tests/reports/samples.html` |
| `runs.html` | The real wave 1 and 2 runs in `LRM_DATA/runs` when it was rendered, two views per kind. Its report links are `file:` links into that folder, so they work on the machine that rendered it | `dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll dashboard --runs <LRM_DATA>/runs --per-kind 2 --out tests/Lrm.Dashboard.Tests/reports/runs.html` |
| `screenshots/` | The pages as a browser shows them, at desktop and phone width, light and dark | headless Edge, see the T2 pull request |

They are two pages because one page with both is over the repo's 1 MiB file limit. Locally, `lrm dashboard --synthetic --runs <LRM_DATA>/runs` puts both in one page, and `--watch` keeps it current while runs are written.
