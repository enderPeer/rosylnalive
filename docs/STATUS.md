# Status

Last updated: 2026-10-18.

Waves 0 and 1 of Phase 1 are merged into `main` (PRs #1–#12), with the wave 1 integration fixes (#21), and so are D1, the D3D12 backend (#16), and the AP-1 pins (#23). The rest of wave 2 merges as each PR's CI and reviews pass. Three workflows are running: wave 2's last workstreams, wave 3a (kernels and the dashboard) and the Vulkan side (R2, then V1). The plan is in `docs/dev-plan.md`.

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
| Wave 1 integration fixes | #21 | Tailcalls off for GPU code; VulkanHeapRecipe v1.1; contract registry; CI steps; real parity, goldens and cluster reports; ADR int-0001 (denormals preserved on D3D12) |
| Wave 2 project-graph edges and contract rows | #22 | Integrator-owned edits lifted out of D1, E1 and M1 |
| D1: D3D12 runtime backend | #16 | Selftests bit-identical twice; 1.2 GB round trip; soak; conformance harness on D3D12; future tree green (874 cpu and 55 gpu tests) |
| AP-1 pins: vk.xml and Vulkan-Headers 1.4.365; CI fetches them and tokenizer.json | #23 | vk.xml git blob and every header blob equal GitHub's tag records |

The merged tree was verified before merging: the local CI sequence, 527 cpu-tier tests and the D3D12 gpu tier, byte-identical to `main` afterwards.

## Wave 2 (workflow `wf_d63d9762-c8a`)

| Workstream | PR | State |
| --- | --- | --- |
| E1: CpuDevice and conformance harness | #17 | Verified; CI re-running with the `contract-change` label |
| M1: byte-level BPE tokenizer | #18 | Verified; stacked on E1, merges after it |
| R5: WARP spike | #20 | Verified; merges after E1 |
| M2a: Qwen forward pass on the CPU (tiny model) | #15 | Verified; re-stacked onto `main` once M2b has branched from it |
| G2: full-model goldens and tolerances | #19 | Final fix-and-verify running |
| M2b: full-model CPU parity (gate1-cpu) | — | Starts after G2 |

## Wave 3a (workflow `wf_097b5762-91c`) and Vulkan (workflow `wf_30631995-81a`)

- K1 (basic kernels), K2 (matmul), K3 (attention) and T2 (dashboard) are being implemented on `int/wave2-preview`, which is `main` plus E1, M1 and R5.
- R2 (vk.xml-driven generator, Vulkan interop, cl.exe ABI oracle) is being implemented; V1 (Vulkan runtime backend) follows on top of it.
- Wave 3b (X1a GPU executor, C2a two-stage pipeline) starts once M2a is on `main`.

## Next

1. Merge the rest of wave 2, then tag `hello-d3d12`, `wave-2-done` and `gate1-cpu` when their gates pass.
2. **Wave 3:** K1–K3 and T2 (running), the Vulkan side (running), X1a and C2a, then X1b and C2b on both GPUs.
3. **Waves 4–5:** Gate 1 per backend, the two-node run, the single-command harness.
4. Phase 2 (backward pass, fault-tolerant training), then the Phase 3 LRM core.

## Findings worth knowing

- HF's "fp64" Qwen still runs its norms, softmax and RoPE in fp32. The patched oracle torch64p fixes this.
- `GenerationConfig(eos_token_id=None)` does not stop EOS in transformers 5.20.1 (ADR O1-0001).
- An `&&local` passed to native code in tail position is a use-after-return bug in F#, so tail calls are off for GPU code (ADR R3-0001).
- fp32 denormals are preserved on D3D12 with `-denorm preserve`, but the Vulkan mode is not measured yet (ADR int-0001).

## Decided by the user (2026-10-18)

- The integrator merges each PR itself once CI and the agent reviews pass.
- AP-1: vk.xml and Vulkan-Headers 1.4.365 are downloaded, and CI runners may fetch them and `tokenizer.json`.
- The repo labels `contract-change` and `ownership-exception` exist, applied as the hygiene rules intend.

## Waiting on the user

- **Mono.Cecil 0.11.6:** a required dependency of the approved Expecto, already restored. Please confirm.
- **AP-2 (optional):** branch protection with required checks. The integrator enforces the rules by hand until then.
- **AP-3 (optional, for V1):** a copy-only install of the LunarG Vulkan SDK into `LRM_DATA`, for one validation-layer run.

## Known gaps

- The Vulkan runtime backend is in progress (R2, then V1).
- No validation layers are installed, and the D3D12 debug layer is not installed.
- GitHub CI has no GPU, so GPU tiers run locally and are pasted into PRs.
