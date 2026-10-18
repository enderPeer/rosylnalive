# Dashboard pages

Rendered by `lrm dashboard` (ADR T2-0001). Each file is one self-contained HTML page: open it in a browser, no server and no network needed.

| File | What it shows | Regenerate (from the repo root) |
| --- | --- | --- |
| `samples.html` | T1's synthetic run of every report template, Phase 2 and 3 included. Every number is made up; each run says so in its banner. A cpu-tier test renders it afresh and fails when this file differs in any byte | `dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll dashboard --synthetic --out tests/Lrm.Dashboard.Tests/reports/samples.html` |
| `runs.html` | The real wave 1 and 2 runs in `LRM_DATA/runs` on 2026-10-18: the index of every run, a view of the newest run of each kind, and four chosen wave 2 runs (D1's D3D12 conformance, R5's WARP conformance, M2a's tiny Qwen forward, G2's CpuRef calibration probe on Qwen3-0.6B). Its report links are `file:` links into that folder, so they work on the machine that rendered it | `dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll dashboard --runs <LRM_DATA>/runs --per-kind 1 --run 20261018-122158-d41fd8f4 --run 20261018-072521-2f9a4c0d --run 20261018-060450-cfb11171 --run 20261018-091611-7cb55719 --out tests/Lrm.Dashboard.Tests/reports/runs.html` |
| `screenshots/` | `runs.html` in headless Edge: the runs index, G2's calibration probe at desktop and at phone width (375 px), and the D3D12 against WARP comparison in the dark theme | see the T2 pull request |

They are two pages because one page with both is over the repo's 1 MiB file limit. Locally, `lrm dashboard --synthetic --runs <LRM_DATA>/runs` puts both in one page, and `--watch` keeps it current while runs are written.
