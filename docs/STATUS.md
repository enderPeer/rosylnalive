# Status

Last updated: 2026-10-17. The Phase 1 plan is final (`docs/dev-plan.md`). Wave 0 (scaffold S0a, then Lrm.Core S1) is starting.

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

## In progress

- **Wave 0:** S0a (scaffold, build, test runner, minimal CI), then S1 (Lrm.Core).

## Next

1. **Wave 1:** S0b, R1 (Vulkan heap spike), R2a/R2b (vkgen v2 and ABI oracle), R3 (D3D12 interop), R4a/R4b (GPU contracts and shader build), O1/O2 (oracles and the CPU reference), T1 (telemetry and reports), C1 (cluster core).
2. **Waves 2–5:** goldens and tokenizer, backends, kernels, executor, two-node pipeline, dashboard, then Gate 1.
3. Phase 2 (backward pass, fault-tolerant training), then the Phase 3 LRM core.

## Waiting on the user

- **AP-1 (rest):** downloading vk.xml v1.4.365 and Vulkan-Headers v1.4.365, needed by R2a in wave 1. Also approval for CI runners to fetch `tokenizer.json`.
- **AP-2:** branch protection on `main` (non-blocking; the integrator enforces the rules by hand until then).

## Known gaps

- The Vulkan binding detects `VK_EXT_descriptor_heap` but cannot use it yet. It has no logical device, memory, buffers, heap writes, push data, pipelines or dispatch.
- No D3D12 binding exists yet.
- No Vulkan validation layers or D3D12 debug layer are installed.
- Golden reference outputs have not been generated yet; the tool environment is ready.
