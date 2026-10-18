# Status

Last updated: 2026-10-18.

Waves 0 and 1 of Phase 1 are merged into `main` (PRs #1–#12), and the wave 1 integration fixes are PR #21. Wave 2 is built on branches (PRs #15–#20) and is finishing its last two workstreams; its PRs are rebased onto `main` and merged in order as their reviews and CI pass. The plan is in `docs/dev-plan.md`.

## On main

| Workstream | PR | Verified |
| --- | --- | --- |
| Phase 1 multi-agent plan | #1 | Planning run `wf_320f4061-04c`: research, 3 competing plans, 2 judges, synthesis, red-team, final pass; corrected through wave 1 |
| S0a: scaffold, build, test runner, minimal CI | #2 | 43 projects generated from `docs/project-graph.json` with layering enforced; deterministic Release build with 0 warnings; tiers cpu, gpu, model and bench with per-backend GPU mutexes |
| S1: Lrm.Core | #3 | Known-answer tests; bf16 and f16 checked exhaustively; identical under de-DE; NFC canary |
| R1: Vulkan descriptor-heap spike | #5 | `VK_EXT_descriptor_heap` mapping path proven on the RTX 4080, bit-exact against CPU and D3D12 |
| O1: oracles and goldens | #4 | torch32 (spec) and torch64p (truth) with self-tests; tiny model and tokenizer corpus; byte-identical regeneration |
| O2: tensors, safetensors, CPU reference | #11 | Tiny model: CpuRef F64 vs torch64p about 1.5e-15; F32 vs torch32 below 1e-6; bit-equal at 1 and 32 threads |
| R4a: GPU contracts, kernel catalog, shader build | #10 | Reproducible ShaderSetHash; DXIL and SPIR-V from the pinned DXC only; known-answer vectors |
| C1: cluster core | #7 | 200 chaos seeds bit-identical; replica-loss replay; 3-process kill test |
| T1: telemetry, run logs, reports | #8 | Byte-deterministic Markdown and SVG; samples in `docs/reports/samples/` |
| R3: D3D12 COM interop | #6 | cl.exe ABI oracle against the SDK headers; RTX 4080 ACCEPTED |
| S0b: governance and hygiene | #9 | 124 checker fixtures behave as designed; hygiene job green |
| R4b: SPIR-V binding and inspector | #12 | Inspector passes every kernel; 18 bad-shader fixtures fail exactly as expected |

The merged tree was verified before merging: the local CI sequence, 527 cpu-tier tests and the D3D12 gpu tier, byte-identical to `main` afterwards.

## Ready to merge

- **#21, wave 1 integration fixes** (rebased onto `main`):
  - `Tailcalls=false` for GPU code;
  - VulkanHeapRecipe v1.1 (completes R4b);
  - every contract registered;
  - `run.*` reserved for the run log;
  - new CI steps for the D3D12 codegen diff, the cl.exe ABI oracle and the golden sets;
  - O1's tools write T1's run-log format;
  - real parity, goldens and cluster reports;
  - ADR int-0001: fp32 denormals are preserved on D3D12;
  - KernelBounds compares in the class's denormal mode.

  Local verification: 548 cpu tests and 9 gpu tests, shader checks, hygiene and golden sets all green.

## Wave 2 (workflow `wf_d63d9762-c8a`)

| Workstream | PR | State |
| --- | --- | --- |
| M2a: Qwen forward pass on the CPU (tiny model) | #15 | Reviewed, fixed and verified |
| D1: D3D12 runtime backend | #16 | Reviewed, fixed and verified |
| E1: CpuDevice and conformance harness | #17 | Reviewed, fixed and verified |
| M1: byte-level BPE tokenizer | #18 | Reviewed, fixed and verified |
| R5: WARP spike | #20 | Reviewed, fixed and verified |
| G2: full-model goldens and tolerances | #19 | Final fix-and-verify running |
| M2b: full-model CPU parity (gate1-cpu) | — | Starts after G2 |

These PRs are stacked on `int/wave1-fixes` (PR #14). Each one is rebased onto `main` before its merge.

## Next

1. Merge #21, then wave 2 in dependency order. Tag `contracts-v1`, `contracts-v1.1`, `wave-1-done`, then `hello-d3d12` and `gate1-cpu` when their gates pass.
2. **Wave 3:** K1–K3 (kernels), X1a and X1b (the GPU executor), C2a and C2b (the two-stage pipeline), T2 (the dashboard).
3. **Waves 4–5:** Gate 1 per backend, the two-node run, the single-command harness.
4. Phase 2 (backward pass, fault-tolerant training), then the Phase 3 LRM core.

## Findings worth knowing

- HF's "fp64" Qwen still runs its norms, softmax and RoPE in fp32. The patched oracle torch64p fixes this.
- `GenerationConfig(eos_token_id=None)` does not stop EOS in transformers 5.20.1 (ADR O1-0001).
- An `&&local` passed to native code in tail position is a use-after-return bug in F#, so tail calls are off for GPU code (ADR R3-0001).
- fp32 denormals are preserved on D3D12 with `-denorm preserve`, but the Vulkan mode is not measured yet (ADR int-0001).

## Waiting on the user

- **AP-1:** downloading vk.xml v1.4.365 and Vulkan-Headers v1.4.365, which R2a, R2b and V1 (the Vulkan backend) need. Also fetching `tokenizer.json` on CI runners, for M1's corpus test in CI.
- **Mono.Cecil 0.11.6:** a required dependency of the approved Expecto, already restored. Please confirm.
- **Repo labels** `contract-change` and `ownership-exception`: hygiene's label check needs them once contracts-v1 is tagged.
- **AP-2 (optional):** branch protection with required checks. The integrator enforces the rules by hand until then.
- **AP-3 (optional, for V1):** a copy-only install of the LunarG Vulkan SDK into `LRM_DATA`, for one validation-layer run.

## Known gaps

- There is no Vulkan runtime backend yet; it is blocked on AP-1.
- No validation layers are installed, and the D3D12 debug layer is not installed.
- GitHub CI has no GPU, so GPU tiers run locally and are pasted into PRs.
