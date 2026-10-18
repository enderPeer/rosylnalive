# Contract changelog

Every contract version, oldest first (OwnershipAndContractChange). Append new entries at the end, under a `## YYYY-MM-DD` heading for the merge date, one bullet per contract version:

```markdown
- **<Contract> v<N>** (`<file>.md`, <owner>, <PR>): <what changed, or "first version">. Vectors: <what changed, or none>. Consumers: <who must act, or none>.
```

Any family may append at the end (the ownership check rejects lines inserted earlier); only S0 rewrites, removes or inserts lines elsewhere. After `contracts-v1`, a `contract-change` PR must add an entry naming every contract text it touches. The integrator corrects a heading's date at merge when it is not the merge date.

## 2026-10-17

- **ProjectGraph v1** (`project-graph.md`, S0a, PR #2): first version. `docs/project-graph.json` is the only source of ProjectReferences and PackageReferences; it covers the layering rules, the file-items-only fsproj shape, generated files and locked restore. Vectors: none. Consumers: every workstream.
- **TestTiers v1** (`test-tiers.md`, S0a, PR #2): first version. It covers the cpu, gpu, model and bench root lists; the mutexes `Local\LrmGpu.d3d12`, `Local\LrmGpu.vulkan` and `Local\LrmModelCpu`, taken in that order; tickets, the 120-minute wait and verify priority; run-twice GPU results; and the fixed FsCheck seed `0x4c524d2d46534b31,0x9e3779b97f4a7c15`. Vectors: the seed. Consumers: every test exe.
- **DataRootAndToolchain v1** (`data-root-and-toolchain.md`, S0a, PR #2): first version. It covers `LRM_DATA` and its single default (`dataRoot`), `toolchain.json` pins, `build/Toolchain.g.props`, bootstrap kinds and approval gating, and the DXC regression. Vectors: the DXC pins. Consumers: every workstream that reads toolchains, weights or data.
- **AgentPermissions v1** (`agent-permissions.md`, S0a, PR #2): first version. AP-0 was declined, so no allowlist is committed. Merge, tag and protection commands are the integrator's; agents push only their own branch. Vectors: none. Consumers: every agent session.
- **LrmCore v1** (`lrm-core.md`, S1, PR #3): first version. It covers ContentId and SHA-256; the canonical encoding and `Canonical.hash` (SHA-256 over tag, `0x00` and the payload); keyspaces and `lrm.blob/1`; DType codes 1 to 8; `Bf16` and `F16` conversions; SplitMix64, Philox4x32-10 and DetRng; ids; Invariant helpers; LrmPaths; and `Process.init`. Vectors: Canonical C1 to C4, bf16 and binary16 tables, SplitMix64, Philox4x32-10, and DetRng. Consumers: every project.
- **TelemetryFacade v1** (`telemetry-facade.md`, S1, PR #3): first version. It covers the ambient no-op-by-default `ITelemetry` with Span, Counter, Gauge, Histogram, Event, GpuInterval, Meter, Check and Child; the rules that telemetry never changes values and timestamps never enter hashes; and record fields, names, units and the reserved names. Vectors: none (behavioural tests). Consumers: every component, and T1's runtime.
- **ExecutionClass v1** (`execution-class.md`, S1, PR #3): first version. It covers the record `{Backend; DeviceId; Driver; ShaderSetHash option; Arch}`, frozen backend names, validation, the id `Canonical.hash "lrm.xclass/1"` and the CPU fill rule. R4a owns the GPU fill rule. Vectors: X1 (CPU) and X2 (GPU, illustrative). Consumers: M2, C1, C2, D1, V1, R5, X1, T1 and I1 to I3.

## 2026-10-18

- **OwnershipAndContractChange v1** (`ownership-and-contract-change.md`, S0b, PR #9): first version. It covers families, slices and globs; exactly one owner per file; the shared CHANGELOG, appended at the end only; the branch grammar; trust (a branch is judged by the merge base's map and size rules, the governance paths change only on S0 and `int/` branches, and the base commit's checker also runs); the ownership check, the `integrators` list and the integrator's `ownership-exception` label; contract files; the `contracts-v1` freeze of every contract file present; `contract-change` PRs and the label check; and the CHANGELOG format. Vectors: owners, branches, contract files and globs (`tools/ci/fixtures/ownership/vectors.json`). Consumers: every workstream and the integrator.
- **DataRootAndToolchain v1, additive** (`data-root-and-toolchain.md`, S0b, PR #9):
  - the `tokenizer` entry (11,422,654 bytes, SHA-256 `aeb13307…`), pending for CI runners;
  - verification of a pending file entry's pinned local copy;
  - `$(LrmTokenizerJson)` and `$(LrmTokenizerSha256)`;
  - the optional, manual `vulkanSdk` entry (AP-3);
  - the bootstrap self-test.

  Vectors: a pins table. Consumers: M1 and O1 (tokenizer path and hash), and V1 (SDK, if AP-3 is approved).
- **ProjectGraph v1, additive** (`project-graph.md`, S0b, PR #9): the hygiene job runs the same validator, and its layering check scans the MSBuild extension files. A `.lrm.targets` file imports nothing, adds no reference item and sets no property of the root build files or of restore and the guard; Compile items stay inside their project; nested `Directory.Build.*`, `Directory.Packages.props`, `global.json` and `NuGet.Config`, and any `Directory.Build.rsp`, `Directory.Solution.*` or `<project>.user`, are refused outside `spikes/`. Vectors: one failing edit per rule, and one failing file set per evasion. Consumers: owners of a `.lrm.targets` file (R3 and R4a pass unchanged).
- **AgentPermissions v1, additive** (`agent-permissions.md`, S0b, PR #9): the size check forbids `.claude/settings.json`, in any case. Agents may set `contract-change` on their own PRs; only the integrator sets `ownership-exception` and opens `int/` branches, which CI checks against `integrators`. Vectors: none. Consumers: every agent session.
