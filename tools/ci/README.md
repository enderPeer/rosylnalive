# tools/ci

CI scripts owned by S0. The rules they enforce are in `docs/contracts/ownership-and-contract-change.md` and `docs/contracts/project-graph.md`.

| Script | What it does |
| --- | --- |
| `gen-project-graph.ps1` | Validates `docs/project-graph.json` and writes `build/ProjectGraph.g.props`, `build/ProjectKinds.g.props` and `Lrm.slnx`. `-Check` writes nothing and fails on a rule violation or a stale output |
| `hygiene.ps1` | Runs the five hygiene checks below and writes `artifacts/hygiene/summary.{json,md}`. The `hygiene` CI job runs it with `-Ci` |
| `test-hygiene.ps1` | The checkers' self-test: every fixture fails as designed and every positive control passes. CI runs it before the checks |
| `Hygiene.psm1` | The check logic, as pure functions over plain inputs, plus the git input helpers |

## The five checks

| Check | Script | Fails when |
| --- | --- | --- |
| ownership | `check-ownership.ps1` | A tracked file is owned by no family or by several (`docs/ownership.json`). The branch name is not `ws/<id>-<slug>`, `ws/<id>-fix-<slug>` or `int/<slug>`, or `<id>` is no slice. A branch outside S0 changes a governance path (`docs/ownership.json`, `tools/ci/**`, `.github/**`). A `ws/` branch changes a path its family does not own, other than appending to a shared path at its end. In CI: an `int/` PR's author, or whoever set `ownership-exception`, is not an integrator |
| layering | `check-layering.ps1` | `gen-project-graph.ps1 -Check` fails: a layering rule, a hand-written reference or other non-file item in an fsproj, an unregistered fsproj, or stale generated files. Or the MSBuild extension scan fails: a `.lrm.targets` file imports, adds a reference item or sets a root or restore property; a Compile item leaves its project; or a nested `Directory.Build.*`, `Directory.Packages.props`, `global.json`, `NuGet.Config`, or any `Directory.Build.rsp`, `Directory.Solution.*` or `<project>.user` exists outside `spikes/` |
| sizes | `check-sizes.ps1` | A committed blob is over its limit (1 MiB, or 2 MiB under `goldens/`), a budget is exceeded (`goldens/tiny` 5 MiB, `goldens/qwen3-0.6b` 10 MiB, `goldens` 20 MiB, the tree 64 MiB), or a forbidden file is tracked, in any letter case (`tools/ci/hygiene.json`) |
| eol | `check-eol.ps1` | The index holds CRLF or mixed line endings, a file has no text attribute, or binary content is not declared binary in `.gitattributes` |
| contract-label | `check-contract-label.ps1` | After `contracts-v1`, a contract file changes without the `contract-change` label. A labelled PR changes no contract file, adds no CHANGELOG line, or leaves a touched contract text unnamed in the CHANGELOG |

**Which rules judge a branch.** `docs/ownership.json` and `tools/ci/hygiene.json` as they are at the merge base, not the branch's own copies; only map coverage uses the checked-out map. The details are in the contract's Trust section.

Each check script exits 0 when it passes, 1 on findings and 2 on a usage or git error, and writes its record with `-ResultPath`.

## Running locally

```bash
pwsh tools/ci/hygiene.ps1                                   # a branch based on main
pwsh tools/ci/hygiene.ps1 -Base origin/ws/S1-core           # a stacked branch: diff from its parent
pwsh tools/ci/hygiene.ps1 -Labels contract-change           # as if the PR had that label
pwsh tools/ci/hygiene.ps1 -Only ownership,contract-label    # some checks only
pwsh tools/ci/test-hygiene.ps1                              # the self-test (about 35 s)
```

The checks read committed trees and the index, so commit before running them. Without `-Labels`, the label rules are reported, not enforced.

## In CI

The `hygiene` job (ubuntu-latest) checks out the full history and tags, runs `test-hygiene.ps1`, then `hygiene.ps1 -Ci`:
- The diff runs from the merge base of the PR's base and head SHAs to the head.
- The tree checks see the merge commit, which is what `main` becomes.
- Labels are read live through `gh api`, falling back to the event payload. Adding or removing a label starts a new run.
- The PR author (for `int/` branches) and whoever last applied `ownership-exception` (the issue events) are checked against `integrators`.
- When the PR changes `tools/ci/**`, the base commit's `hygiene.ps1` runs too, on the same tree, and writes `artifacts/hygiene/base/`.

## Adding a fixture

Add a case to `fixtures/<check>/cases.json` with `expect: fail` and `match` regexes that pin the designed error (or `expect: pass` for a positive control). Then run `test-hygiene.ps1`. Ownership known answers live in `fixtures/ownership/vectors.json` and in the contract text's Vectors section, which change together.
