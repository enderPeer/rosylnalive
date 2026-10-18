# OwnershipAndContractChange contract

Owner: S0 (S0b). Status: v1. Consumers: every workstream and the integrator. Machine-readable form: `docs/ownership.json` (schema `lrm.ownership/1`). Enforced by the `hygiene` CI job: `tools/ci/check-ownership.ps1` and `tools/ci/check-contract-label.ps1`, both run by `tools/ci/hygiene.ps1`. Tests: `tools/ci/test-hygiene.ps1`.

This contract decides two things: who may change which path, and how a contract changes once it is frozen.

## Families and paths

- **Families own paths.** `families` in `docs/ownership.json` maps each workstream family (S0, S1, R1, R2, …) to its slices and its path globs.
  - Slices (ids with an a/b suffix, such as R4a and R4b) share their family's paths.
  - A glob with a leading `!` removes paths from that family. For example, R4 owns `src/Lrm.Shaders/**` except the three kernel folders, which K1, K2 and K3 own.
- **Exactly one owner.** Every tracked file is owned by exactly one family. A file that no family owns, or that two families own, fails the ownership check on every branch, `main` included.
- **No mid-plan handovers.** A family's globs change only through an integrator (`int/`) PR to `docs/ownership.json`. This is enforced: a branch is judged by the map at its merge base, and the map is a governance path (see Trust below).
- **Where a family writes its documents:**
  - contract texts in `docs/contracts/<name>.md` (the names are in `docs/contracts/README.md`);
  - ADRs in `docs/decisions/<family>-*.md`; the cross-cutting ADRs `docs/decisions/0000-*.md` and `0001-*.md` belong to S0;
  - generated reports under the family's own `docs/reports/` folder. Wave-exit reports (`docs/reports/waves/**`) belong to the integrator.

### Glob syntax

Paths are repo-relative, use `/`, and are compared case-sensitively, as git stores them. The one exception is the forbidden-file rules of the size check (`tools/ci/hygiene.json`), which ignore case: on Windows, `foo.DLL` is a binary and `.claude/Settings.json` is the allowlist.

| Pattern | Matches |
| --- | --- |
| `**/` | zero or more whole path segments |
| `**` elsewhere, for example at the end | anything, including `/` |
| `*` | any characters within one segment |
| `?` | one character other than `/` |
| anything else | itself |

Brackets, braces and backslashes are rejected, so a glob never silently means something else.

### Shared paths

`shared` lists paths that any family may change in one way, without owning them. The only mode is `append`: a branch of another family may add lines after the last line of the file as it is at the merge base, but must insert nothing earlier, delete or rewrite no line, and not delete the file. The check reads the positions from the `-U0` hunks of the diff.

| Path | Mode | Why |
| --- | --- | --- |
| `docs/contracts/CHANGELOG.md` | append | Every contract-change PR adds its own entry |

The owner (S0) and the integrator may edit a shared path freely.

## Branches

| Branch | Who | What it may change |
| --- | --- | --- |
| `ws/<id>-<slug>` | the agent of workstream `<id>` | the paths of `<id>`'s family, plus appends to shared paths |
| `ws/<id>-fix-<slug>` | a fix session for `<id>` | the same as `ws/<id>-<slug>` |
| `int/<slug>` | the integrator | any path; in CI the PR author must be listed under `integrators` |
| `main` | nobody pushes directly | changes arrive through reviewed PRs, so only coverage is checked |

The exact grammar:
- `<id>` is a slice listed in `docs/ownership.json`, matched case-sensitively: `S0b`, `R4a` and `M2b` are slices, but `s1` is not.
- `<slug>` is lowercase letters and digits joined by single hyphens: `[a-z0-9]+(-[a-z0-9]+)*`. An `int/` slug may also use dots.
- Any other branch name fails the ownership check.

## Trust: what judges a branch

A branch never chooses the rules it is checked by:
- **The merge base's policy.** The branch grammar, the out-of-path rule, the shared paths, the integrators, the contract files, the labels and the freeze tag come from `docs/ownership.json` as it is at the merge base. The size rules come from `tools/ci/hygiene.json` there. A change to either file takes effect for the PRs after it.
  - Coverage is the exception: it uses the checked-out map, because that map must describe the checked-out tree.
  - When the merge base has no `docs/ownership.json`, every `ws/` and `int/` branch fails: rebase onto `origin/main`.
  - When the merge base's map has no `contract` section (S0a's first map) or there is no `hygiene.json` yet, the checked-out one is used, and the summary says so.
- **Governance paths.** `docs/ownership.json`, `tools/ci/**` and `.github/**` change only on S0 branches (`ws/S0a-…`, `ws/S0b-…`) and `int/` branches. This holds whatever any map says, and the exception label does not lift it.
- **The base commit's checker.** On a pull request that changes `tools/ci/**`, the hygiene job also runs the base commit's `tools/ci/hygiene.ps1` on the same tree. A branch that edits the checks to switch them off is still judged by the old ones.
- **Generated and validator code.** The layering check runs the checked-out `gen-project-graph.ps1`, so an S0 PR that changes the generator is checked by its new version. The governance rule fails any other family's change to it.

## The ownership check

`check-ownership.ps1` runs these steps:

1. **Coverage.** Every file in the checked-out tree has exactly one owning family in the checked-out map.
2. **Branch.** The branch name resolves as in the table above, against the merge base's map.
3. **Diff.** On a `ws/` branch, the changed paths are those from the merge base of the PR's base and head to the head (`git diff --no-renames base...head`).
   - A rename counts as a deletion plus an addition, so both paths must be the family's.
   - Deleting another family's file is an out-of-path edit, like changing it.
4. **Governance paths.** Outside family S0, a change to a governance path is an error, whatever the map and the labels say.
5. **Out-of-path edits.** Every other changed path must be owned by the branch's family in the merge base's map, or be a shared path that the change only appends to, at the end. Anything else is an error that names the path, its owner and the branch.

An `int/` branch skips steps 4 and 5. Every `int/` run is listed as a warning in the hygiene summary, and in CI the PR author must be one of the `integrators` in the merge base's map.

**The exception label.** The integrator may accept a reviewed out-of-path edit by putting the label `ownership-exception` on the PR. An example is an agent's additive edit to `docs/project-graph.json` for its own projects.
- With the label, those paths are reported as warnings, not errors, and the hygiene summary still lists every one of them.
- Only the integrator sets this label. Agents never do. In CI the label counts only when whoever last applied it (the PR's issue events, read through `gh api`) is one of the `integrators`. If that cannot be read, the label does not count.
- The label never covers a governance path.
- Without known PR labels (outside a pull request), the exception never applies. Outside CI, who set the label and who opened an `int/` PR are not verified; the result says so as a warning.

**Limits of the author check.** `integrators` lists GitHub logins (`enderPeer`). Today every session, the integrator's and the agents', acts through that one account, so the login check separates nothing yet. What protects the plan is that agents never use `int/` branches or the exception label (AgentPermissions), and that the hygiene summary lists every use as a warning for the integrator to review. If agents later get their own account, the check starts to separate them without a code change.

## Contract files

Contract files are the paths matched by `contract.files` in `docs/ownership.json`:

| Glob | What it covers |
| --- | --- |
| `docs/contracts/*.md`, except `README.md` and `CHANGELOG.md` | every contract text |
| `src/Lrm.Core/**` | LrmCore, TelemetryFacade, ExecutionClass |
| `src/Lrm.Gpu.Core/**` | HeapLayoutAndTensorTable, KernelAbiAndCatalog, GpuBackendInterface, including the generated `Kernels.g.fs` |
| `src/Lrm.Shaders/hlsl/**/*.hlsli` | the shader include files, `lrm.hlsli` and the generated `lrm_args.g.hlsli` |
| `src/Lrm.Shaders/Lrm.Shaders.lrm.targets` and `src/Lrm.Shaders.Build/**` | the shader build (ShaderBuildAndSourceConvention) |
| `src/Lrm.Tensor/**` | the Lrm.Tensor API: HostTensorAndSafetensors, ComparatorsAndGoldenReader, the ToleranceSpec types and `KernelBounds.fs` |
| `src/Lrm.Cluster/Protocol*.fs` | ClusterProtocol |
| `**/*.schema.json` | manifest schemas |
| `goldens/**/tolerances.json` | ToleranceSpec numbers |

Kernels, the CPU reference, the cluster's internals and other implementation files are not contract files; their behaviour is pinned by the contracts' vectors and tests. An owner who needs a narrower or wider glob asks the integrator.

### Freeze and contract-change PRs

The tag `contracts-v1` freezes the contracts (`contract.freezeTag`). Before the tag exists, contract files are drafts and change freely. The tag freezes every contract file on `main` at that point, not only the contracts the plan names for it. After it exists, any change to a contract file, including a draft text that lands later, goes only through a **contract-change PR**, which has these properties:

1. **It carries the label `contract-change`.** The PR author sets it, and an agent may set it on its own PR.
2. **It adds a CHANGELOG entry.** The entry goes in `docs/contracts/CHANGELOG.md` and names every contract text the PR touches by its file name, for example `` `lrm-core.md` ``.
3. **It updates the known-answer vectors** of every contract whose encoding or semantics change. The vectors live in the contract text and its tests, and the integrator reviews them.
4. **A breaking change bumps the contract's version** (v1 to v2, and the `Status:` line of the text). It merges in one PR together with the consumer fixes, so `main` stays green. The consumer fixes sit in other families' paths, so that PR either is an `int/` branch, or is the owner's `ws/` branch carrying `ownership-exception`, which the integrator sets after reviewing the fixes.
5. **An additive change**, such as a new catalog kernel, a new optional field or a new reserved tag, keeps the major version. Its entry says "additive", and it gets same-day review.

A new contract text written after the freeze, such as a wave-2 workstream's first version, also goes through a contract-change PR. Its CHANGELOG entry records it as v1.

### The label check

`check-contract-label.ps1` fails:
- when the freeze tag exists, a contract file changed, and the PR lacks the `contract-change` label;
- when the label is set but no contract file changed;
- when the label is set but the PR adds no line to `docs/contracts/CHANGELOG.md`;
- when the label is set and a touched contract text is not named in the added CHANGELOG lines.

Labels are read when the job runs, through `gh api`, with the event payload as a fallback. Adding or removing a label starts a new CI run (`labeled` and `unlabeled` events), so the result always reflects the current labels. Outside a pull request the labels are unknown: the label rules are then reported as warnings and are not enforced.

The contract files, the label names and the freeze tag are those of the merge base's map, so a branch cannot drop its files from `contract.files` or rename the tag to escape the rule.

## The CHANGELOG

`docs/contracts/CHANGELOG.md` records every contract version, oldest first. Entries are appended at the end, under a `## YYYY-MM-DD` heading for the merge date; the ownership check fails another family's lines inserted anywhere else. The integrator corrects the heading date at merge when it differs from the date the entry was written. Each entry is one bullet with these parts:

```markdown
- **<Contract> v<N>** (`<file>.md`, <owner>, <PR>): <what changed, or "first version">. Vectors: <what changed, or none>. Consumers: <who must act, or none>.
```

Any family may append. Only S0 rewrites or removes lines, for example to fix a typo.

## Vectors

These known answers hold for `docs/ownership.json` at v1. `tools/ci/fixtures/ownership/vectors.json` holds the same values, and `test-hygiene.ps1` checks them on every hygiene run. Change them only together with this text.

**Owners.**

| Path | Family |
| --- | --- |
| `src/Lrm.Core/Canonical.fs` | S1 |
| `docs/contracts/lrm-core.md` | S1 |
| `docs/contracts/CHANGELOG.md` | S0 (shared, append) |
| `docs/decisions/0001-shader-toolchain.md` | S0 |
| `docs/decisions/R1-vk-heap-recipe.md` | R1 |
| `spikes/probes/probe/vkdevice.fsx` | S0 |
| `spikes/vk-heap/src/Main.fs` | R1 |
| `apps/Lrm.Cli/Program.fs` | S0 |
| `apps/Lrm.Cli/Commands/DevicesVulkan.fs` | R2 |
| `apps/Lrm.Cli/Commands/Report.fs` | T1 |
| `src/Lrm.Shaders/hlsl/lrm.hlsli` | R4 |
| `src/Lrm.Shaders/hlsl/kernels/basic/add.hlsl` | K1 |
| `tests/Lrm.Kernels.Conformance/Harness.fs` | E1 |
| `tests/Lrm.Kernels.Conformance/Cases/attention/Gqa.fs` | K3 |
| `tools/goldens/common.py` | O1 |
| `tools/goldens/qwen/make_qwen.py` | G2 |
| `goldens/qwen3-0.6b/g1/tolerances.json` | G2 |
| `docs/reports/gate1/backends/vulkan.md` | I1 |
| `docs/reports/gate1/summary/report.md` | I3 |
| `docs/reports/waves/wave-1.md` | S0 |

**Branches.**

| Branch | Resolves to |
| --- | --- |
| `ws/S0b-governance` | workstream, family S0 |
| `ws/R4b-spirv-inspector` | workstream, family R4 |
| `ws/O2-fix-kernel-bounds` | workstream (fix), family O2 |
| `int/status-2026-10-18` | integrator |
| `main` | main |
| `ws/Q7-new`, `ws/s1-core` | unknown workstream id: fails |
| `ws/S1_core`, `ws/S1-`, `ws/S1-Core`, `feature/x`, `int/` | invalid: fails |

**Contract files.**

| Path | Contract file |
| --- | --- |
| `src/Lrm.Gpu.Core/Kernels.g.fs` | yes |
| `src/Lrm.Shaders/hlsl/lrm.hlsli` | yes |
| `src/Lrm.Shaders/hlsl/kernels/basic/add.hlsl` | no |
| `src/Lrm.Tensor/KernelBounds.fs` | yes |
| `src/Lrm.Tensor.CpuReference/Ops.fs` | no |
| `src/Lrm.Cluster/ProtocolFrames.fs` | yes |
| `src/Lrm.Cluster/Coordinator.fs` | no |
| `docs/contracts/README.md`, `docs/contracts/CHANGELOG.md` | no |
| `docs/project-graph.json` | no |

**Globs.** `src/Lrm.Core/**` matches `src/Lrm.Core/sub/dir/x.fs` but not `src/Lrm.CoreX/Hash.fs`. `docs/contracts/*.md` does not match `docs/contracts/sub/x.md`. `**/*.safetensors` matches `model.safetensors` at the root. `README.md` does not match `readme.md`. The globs `src/{a,b}/**`, `src/[ab]/x.fs`, `src\x.fs` and the empty glob are rejected.

## Verification

`tools/ci/test-hygiene.ps1` runs before the checks in every hygiene job, and checks the following:
- the vectors above;
- 33 ownership and 10 contract-label fixtures, run through the pure functions. Each failing fixture must produce errors matching its designed messages. They cover the self-granted map, a merge base without a map, governance paths with and without the exception label, an insert into the middle of the CHANGELOG, and `int/` authors and exception actors who are and are not integrators;
- end-to-end fixture git repositories, checked by the real scripts in child processes:
  - a `ws/O1-…` fixture branch that edits `src/Lrm.Core/Hash.fs` fails the ownership check, and a `ws/S1-…` branch passes;
  - an append to the CHANGELOG passes; a rewrite of it fails, and so does a back-dated entry inserted into an older section;
  - a move into another family's path fails;
  - a branch that rewrites its own map to grant itself `docs/ownership.json` and `src/Lrm.Core/**`, and then edits `Hash.fs`, fails, with and without the exception label; through `hygiene.ps1 -Only ownership,contract-label` it fails both checks after the freeze;
  - a branch outside S0 that edits `tools/ci/Hygiene.psm1` fails;
  - contract changes behave as specified before and after a `contracts-v1` tag, with and without the label and the CHANGELOG entry, and a branch's own map cannot thaw the freeze.

Mutation runs show that the fixtures detect broken rules:
- The first run disabled the out-of-path rule and the CRLF rule, and 16 cases failed the self-test.
- After review, a second run broke eight rules one at a time. Each mutation made at least one case fail:

| Mutation | Cases that fail |
| --- | ---: |
| the branch's own map judges it | 4 |
| no governance paths | 6 |
| no MSBuild extension scan | 12 |
| CHANGELOG lines may be inserted anywhere | 2 |
| everyone counts as an integrator | 3 |
| forbidden-file rules are case-sensitive | 1 |
| the branch's own contract section applies | 1 |
| a merge base without a map is accepted | 1 |
