# Status

Last updated: 2026-10-18. Wave 0 is built and reviewed, but not merged: merging is waiting on the user. Wave 1 is running on branches stacked on wave 0.

## Done

| Item | Where | Verified |
| --- | --- | --- |
| Design document, with an independent review applied | `docs/LRM-design.md` | Reviewed by a first-principles reviewer agent |
| F# Vulkan binding (query only) and device check | `gpu/vulkan` | Builds with no warnings; RTX 4080 accepted; 17 struct layouts match vk.xml |
| Qwen3-0.6B weights and tokenizer | `C:\Users\end\dev\lrm-data\models\Qwen3-0.6B` | SHA-256 of `model.safetensors` and `tokenizer.json` match Hugging Face |
| Public GitHub repo | https://github.com/enderPeer/rosylnalive | `main` pushed |
| Shader compiler: DXC 1.9.2609.5 from NuGet, pinned in `tools/bootstrap/toolchain.json` | `lrm-data\toolchain\dxc\1.9.2609.5` | SHA-512 matches nuget.org; Microsoft signature valid; a bindless test kernel compiles to DXIL and SPIR-V |
| Golden-tool venv: torch 2.14.1+cpu, transformers 5.20.1, tokenizers 0.23.3 (lock file `tools/goldens/requirements.lock`) | `lrm-data\venv-golden` | Qwen3 fp32 forward runs in 2.3 s; `lm_head` is bitwise equal to `embed_tokens` |
| Multi-agent Phase 1 plan | `docs/dev-plan.md` | Run `wf_320f4061-04c`: 3 research agents, 3 competing plans, 2 judges, synthesis, red-team (1 critical and 13 major issues, all fixed), final pass |
| Planning probes staged | `lrm-data\spikes-staging` | Step 0 of the plan; S0b commits them to `spikes/probes/` after review |

## Built, reviewed, not merged

| Workstream | PR | Verified |
| --- | --- | --- |
| S0a: scaffold, build, test runner, minimal CI | [#2](https://github.com/enderPeer/rosylnalive/pull/2) | Fresh-clone Release build with 0 warnings; 43 projects come from `docs/project-graph.json`, with layering enforced; VkDeviceCheck ACCEPTED from `apps/`; the cpu tier passes; GitHub CI green after a bootstrap fix (a closure-scope bug that only showed on the download path) |
| S1: Lrm.Core (hashing, canonical encoding, bf16/f16, Philox, telemetry facade, ExecutionClass) | [#3](https://github.com/enderPeer/rosylnalive/pull/3), stacked on #2 | Known-answer tests pass; bf16 and f16 checked exhaustively over every bit pattern; identical under de-DE; the NFC canary passes; two reviews approved |

## In progress

- **Wave 1** (workflow `wf_6818f11b-506`), stacked on `ws/S1-core`:
  - R4a (GPU contracts, catalog, shader build), O1 (oracles and goldens), T1 (telemetry, run logs, reports with charts), C1 (cluster core), R1 (Vulkan heap spike), R3 (D3D12 interop) and S0b (governance);
  - then O2 (the CPU reference, after O1) and R4b (the SPIR-V binding, after R4a and R1).
- R2a and R2b (vkgen v2 and the Vulkan ABI oracle) are waiting on approval to download vk.xml and Vulkan-Headers.

## Next

1. Merge waves 0 and 1 in order, once merging is decided.
2. **Waves 2–5:** goldens and tokenizer, backends, kernels, executor, two-node pipeline, dashboard, then Gate 1.
3. Phase 2 (backward pass, fault-tolerant training), then the Phase 3 LRM core.

## Waiting on the user

- **Merging (blocking):** auto mode does not let the integrator merge its own PRs. Either add an allow rule for `gh pr merge`, or merge the PRs yourself.
- **Mono.Cecil 0.11.6:** a required dependency of the approved Expecto. It was already fetched during restore; please confirm.

- **AP-1 (rest):** downloading vk.xml v1.4.365 and Vulkan-Headers v1.4.365, needed by R2a in wave 1. Also approval for CI runners to fetch `tokenizer.json`.
- **AP-2:** branch protection on `main` (non-blocking; the integrator enforces the rules by hand until then).

## Known gaps

- The Vulkan binding detects `VK_EXT_descriptor_heap` but cannot use it yet. It has no logical device, memory, buffers, heap writes, push data, pipelines or dispatch.
- No D3D12 binding exists yet.
- No Vulkan validation layers or D3D12 debug layer are installed.
- Golden reference outputs have not been generated yet; the tool environment is ready.
