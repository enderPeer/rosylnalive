# Status

Last updated: 2026-10-18 (integrator, `int/wave1-fixes`). Phase 1 (Qwen3-0.6B forward pass) is in progress: waves 0 and 1 are built and reviewed, and nothing is merged to `main` yet, because the merges wait for the user.

## Where the work is

| PR | Branch | What | State |
| --- | --- | --- | --- |
| #1 | `int/dev-plan` | The multi-agent development plan (`docs/dev-plan.md`) | open, waiting for merge |
| #2, #3 | `ws/S0a-scaffold`, `ws/S1-core` | Wave 0: scaffold, build, test runner, CI; Lrm.Core | open, reviewed |
| #4 to #12 | `ws/O1-…` to `ws/R4b-…` | Wave 1: O1, R1, R3, C1, T1, S0b, R4a, O2, R4b | open, reviewed |
| #13 | `int/wave1` | Integration preview of #2 to #12 (not a merge unit) | draft |
| #14 | `int/wave1-fixes` | Cross-cutting fixes from the wave-1 requests, on top of #13 | open |

R2a and R2b (Vulkan interop generator) have not started: they need vk.xml and Vulkan-Headers, which wait for the user's approval (AP-1). V1 depends on R2a.

## Done and verified (on `int/wave1` plus `int/wave1-fixes`, locally)

| Item | Where | Verified |
| --- | --- | --- |
| Design document, with an independent review | `docs/LRM-design.md` | Reviewed before the plan |
| Qwen3-0.6B weights and tokenizer; DXC 1.9.2609.5; the golden venv | `LRM_DATA` (`C:/Users/end/dev/lrm-data`) | SHA-256 against Hugging Face and nuget.org; pins in `tools/bootstrap/toolchain.json` |
| Scaffold, project graph, test tiers, mutexes, CI | S0a, S0b | Release build with 0 warnings; codegen checks; locked restore; hygiene self-test (125 fixtures); hygiene checks |
| Lrm.Core: canonical encoding, hashes, DType and bf16, RNGs, telemetry facade, ExecutionClass | S1 | Known-answer vectors; bf16 exhaustive; de-DE identical; `run.*` now reserved for the run log |
| Golden tooling, torch32/torch64p oracles, tiny and tokenizer goldens | O1 | Self-tests pass; `make_tiny.py --check` regenerates byte-identically; the tools now write T1's RunLog format, so their runs render with `lrm report` |
| Tensors, safetensors, CpuRef F32/F64, comparators, KernelBounds, ToleranceSpec | O2 | Tiny model: F64 vs torch64p within 1e-12, F32 vs torch32 within 1e-6, greedy identical, 1/7/32 threads bit-equal, prefill = decode; parity report from a real run in `docs/reports/cpuref/tiny/` |
| Telemetry runtime, run logs, reports, samples | T1 | Byte-deterministic Markdown and SVG; samples in `docs/reports/samples/` |
| Cluster core: frames, transports, coordinator, chaos | C1 | 200 seeds bit-identical; replica-loss replay; 3-process kill test |
| D3D12 COM interop, ABI oracle, device check | R3 | Slots match the probe; cl.exe oracle passes; RTX 4080 ACCEPTED; no tail calls (now a repo-wide default for GPU code) |
| Vulkan descriptor-heap recipe | R1 | Bit-exact `scale`, table and bf16 on the RTX 4080; 32 push dwords echo; ADR `R1-heap-recipe.md`; VulkanHeapRecipe v1.1 names the typed argument blocks |
| GPU core contracts, catalog, shader build | R4a | DXIL and SPIR-V compile with the pinned DXC; ShaderSetHash reproducible; catalog diff-clean |
| SPIR-V binding and inspector | R4b | One variable at set 0 binding 0; push offsets equal the catalog; 16 inspector rules, 18 DXC fixtures, 66 mutations; typed blocks pass R1's spike on the device |
| fp32 denormals on D3D12 | integrator | The probe on the real `scale_f32` DXIL blob gives `preserve` on the RTX 4080, driver 616.92 (ADR `int-0001-denormals.md`) |
| Every wave-1 contract registered | `docs/contracts/README.md`, `CHANGELOG.md` | v1, or v1.1 for R4b's three and VulkanHeapRecipe |

## Next

1. **The user:** merge #1, then #2 to #12 in order and #14; then the integrator tags `contracts-v1` and `contracts-v1.1` (both after every wave-1 text is on `main`) and `wave-1-done`.
2. **The user:** AP-1 (vk.xml, Vulkan-Headers and the tokenizer download on CI runners), so that R2a, R2b and V1 can start and M1's corpus test can run in CI; the repo labels `contract-change` and `ownership-exception`; AP-2 (branch protection).
3. **Wave 2:** G2, M1, M2a, M2b, E1, D1 and R5 can start from `main`; V1 after R2a. Each starts from the routed notes in `LRM_DATA/spikes-staging/routed-notes.json`.
4. **Template slots.** Several wave-1 metrics (R3's device and interop checks, R4b's shader metrics, C1's `cluster.*` names, S0b's hygiene summary) have no template slot yet. Each is added, through an `int/` change to T1's templates, in the PR whose run log first carries the metric.

## Known gaps

- No Vulkan validation layers or D3D12 debug layer are installed (AP-3 is optional).
- The cluster template reads `cluster.unit` and `cluster.step` spans and `cluster.node_lost`; C1's coordinator emits `cluster.unit.done` events and `cluster.member.leave`. C2a's cluster report needs the two aligned.
- DXIL float arithmetic carries fast-math flags (no `-Gis`), so a NaN test written as `v != v` is folded away; kernels use `isnan()` (ADR int-0001).
- GPU NaN results do not keep their payloads (R1: `0x7FFFFFFF`), unlike the CPU reference.
- The hygiene job has not yet run on GitHub's Linux runner; every check was run locally.
- The Vulkan fp32 denormal mode is not measured yet (V1).
