# LRM — working notes for agents

The LRM (Large Roslyn Model) is a transformer whose experts call Roslyn inside its layers, written in F# on our own D3D12/Vulkan stack. The design is in `docs/LRM-design.md`. Current status and next steps are in `docs/STATUS.md`; the multi-agent plan is in `docs/dev-plan.md`, and who owns which paths is in `docs/ownership.json`.

## Fixed decisions (do not re-litigate)

- **Language:** F# on .NET 10 (SDK 10.0.402) for everything in the system. No PyTorch or TorchSharp in the system. Python is allowed only for offline tools such as golden-output generation, run in an isolated venv.
- **GPU:** our own stack for Direct3D 12 **and** Vulkan, using a bindless descriptor-heap design.
  - Vulkan requires `VK_EXT_descriptor_heap`; reject devices without it.
  - Write our own F# P/Invoke bindings; no Silk.NET, TerraFX or Vortice.
  - Start with the set/binding → heap mapping path, not direct heap indexing.
- **Heap layout:** one resource heap, written once at startup, holding arena buffers plus a tensor table. Kernels receive 32-bit tensor tokens via push data or root constants. Kernel arguments are capped at 128 bytes (RTX 4080 `maxPushDataSize` is 256).
- **Oracle:** the F# CPU reference implementation is the specification for every GPU kernel.
- **Clusters:** designed for networked GPU clusters from the start, robust to lag and dropouts.
  - Work units are pure and content-addressed.
  - Lagging work is recomputed elsewhere.
  - Reductions run in fixed shard-ID order.
  - Chaos tests must produce bit-identical results.
- **Observability is a core requirement, visual first.**
  - Metrics, tracing and metering are built into every component.
  - Markdown reports are generated from templates using computed data, with SVG charts embedded.
  - An HTML dashboard runs over the same run logs.
  - Reports cover training per layer and per stage, loss, thinking, routing and inference.

## Roadmap

1. **Phase 1:** Qwen3-0.6B forward pass on our stack.
   - Gate: on both backends, logits match the reference goldens within tolerance and greedy decoding gives identical tokens.
   - The same forward pass must also run split across two nodes.
2. **Phase 2:** the backward pass and training, multi-node and fault tolerant.
   - Gate: gradients match, and a chaos run gives identical weights.
3. **Phase 3:** the LRM itself (Roslyn experts).

## Local resources

- Qwen3-0.6B weights: `C:\Users\end\dev\lrm-data\models\Qwen3-0.6B`. Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca`, bf16, with `SHA256SUMS` alongside. Keep model files out of git.
- Toolchain pins: `tools/bootstrap/toolchain.json`.
  - DXC is at `C:\Users\end\dev\lrm-data\toolchain\dxc\1.9.2609.5\build\native\bin\x64\dxc.exe`. Never use the Windows Kits dxc: it has no SPIR-V backend.
  - The golden-tool Python venv is at `C:\Users\end\dev\lrm-data\venv-golden`; run it with `python -I`. It produces test data only.
  - vk.xml and Vulkan-Headers 1.4.365 are under `C:\Users\end\dev\lrm-data\toolchain\vulkan\1.4.365` and `C:\Users\end\dev\lrm-data\toolchain\vulkan-headers\1.4.365`, verified against the upstream tags. CI runners fetch them and `tokenizer.json` (AP-1, approved 2026-10-18).
  - Not approved: the LunarG Vulkan SDK and validation layers (AP-3), and the Agility SDK (AP-4). Ask before any new download.
- **Testing uses many oracles.** The F# CPU reference is the spec. It is cross-checked against:
  - PyTorch/transformers fp32 goldens and HF `tokenizers`;
  - finite differences and torch autograd (Phase 2);
  - agreement between D3D12 and Vulkan;
  - Roslyn itself in Phase 3 (compile, ILVerify, test execution).
- GPU: one NVIDIA RTX 4080 16 GB, driver 616.92; 32 CPU threads. GitHub-hosted CI has no GPU, so GPU tests run locally and are tagged.

## Repo layout and build

- **Layout:** libraries in `src/`, apps in `apps/` (`Lrm.Cli`, `VkDeviceCheck`), Expecto test exes in `tests/`, scripts and generators in `tools/`, throwaway spikes in `spikes/` (never in the solution or CI). `Lrm.slnx` lists every project.
- **References come only from `docs/project-graph.json`.** `tools/ci/gen-project-graph.ps1` checks the layering rules and writes `build/ProjectGraph.g.props`, `build/ProjectKinds.g.props` and `Lrm.slnx`. An fsproj holds file items only; per-project build logic goes in `<Name>.lrm.targets` next to it. A hand-written reference or an unregistered project fails the build. NuGet lock files live in `build/locks/` and CI restores in locked mode. To change an edge or add a project, ask the integrator (S0). Contract texts are in `docs/contracts/<name>.md` (index in `docs/contracts/README.md`, every version in `docs/contracts/CHANGELOG.md`); the S0 ones are `project-graph`, `test-tiers`, `data-root-and-toolchain`, `agent-permissions` and `ownership-and-contract-change`.
- **Data root:** `LRM_DATA` (default `C:/Users/end/dev/lrm-data`, from `dataRoot` in `tools/bootstrap/toolchain.json`) holds toolchains, weights, goldens, run logs and test locks. Nothing under it is committed. `pwsh tools/bootstrap/bootstrap.ps1` verifies or installs every approved pin (DXC, vk.xml, Vulkan-Headers, the tokenizer); it downloads only with `-AllowDownload`. MSBuild reads the pins from `build/Toolchain.g.props` (for example `$(LrmDxcExe)`, `$(LrmTokenizerJson)`).
- **Hygiene** (a required check, with build-test): `pwsh tools/ci/hygiene.ps1` checks ownership (your branch may change only your family's paths in `docs/ownership.json`; `docs/contracts/CHANGELOG.md` is append-only for everyone), layering, file sizes, line endings (LF only) and the `contract-change` label (needed after `contracts-v1` for any contract file). On a stacked branch pass `-Base origin/<parent branch>`. Decisions that bind other workstreams go in an ADR (`docs/decisions/0000-adr-process.md`).
- **Build:** `dotnet build Lrm.slnx -c Release`. Warnings are errors; output goes to `artifacts/`. Apps run as `dotnet <dll>`, x64 only, and every exe calls `Lrm.Core.Process.init ()` first.
- **Tests:** `pwsh tools/test.ps1 -Tier cpu|gpu|model|bench`. Each test exe declares `TestTiers` (cpu, gpu per backend, model, bench) and calls `Runner.main` from `tests/Lrm.Testing`. GPU and model runs queue on the mutexes `Local\LrmGpu.d3d12`, `Local\LrmGpu.vulkan` and `Local\LrmModelCpu`, with tickets in `LRM_DATA/locks`. Use `Props.testProperty` (fixed FsCheck seed, `LRM_FSCHECK_SEED` overrides) and `Determinism.runTwice` for GPU results. CI runs only the cpu tier. `LRM_REQUIRED_BACKENDS` is enforced on the gpu tier always, and on the model tier with `-RequireBackends`.
- **Before pushing:** `dotnet fantomas .` (CI runs `fantomas --check`), a 0-warning Release build, the tiers you touched, and `pwsh tools/ci/hygiene.ps1` on the committed branch.
- **No `.claude/settings.json`** is committed (AP-0 was declined).

## Working rules

- **Git:**
  - Work on a branch per workstream; for parallel agents, use a worktree each.
  - Push after every green checkpoint, and at least once per working session.
  - Merge to `main` through PRs; `main` stays green.
  - Only the integrator merges, after CI and the agent reviews pass.
    - Merges use merge commits (`gh pr merge --merge`), so stacked branches keep their merge bases.
    - Before merging GPU or model work, the integrator builds the exact future tree locally and runs the gpu and model tiers.
    - A stacked branch moves with `git rebase --onto origin/main <old base head> <branch>`.
  - End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **File ownership:** stay inside the paths your workstream owns. The S0-owned files belong to the integrator: `docs/project-graph.json`, `build/**`, `.github/**`, `Directory.Build.*`, `docs/contracts/README.md`, `tools/ci/**` and `tools/bootstrap/**`.
  - A workstream that needs to edit one of these (a project-graph edge, a CI step) does it in a separate commit whose subject starts with `INT-REQUEST:`.
  - The integrator lifts that commit into an `int/` PR, merges it first, then re-stacks the workstream.
  - Appending your contract's line to `docs/contracts/CHANGELOG.md` is allowed in normal commits.
- **Labels:**
  - Since `contracts-v1`, a PR that adds or changes a contract file needs the `contract-change` label and a CHANGELOG entry.
  - `ownership-exception` is integrator-only.
  - Only the integrator sets labels; agents never create or set them.
- **The user** (enderPeer) sets direction at the concept level. Make routine engineering decisions yourself and report them in a line; don't ask about plumbing.
- **Updating STATUS:** after finishing a workstream or wave, update `docs/STATUS.md` with what changed, what is verified, and what is next.
