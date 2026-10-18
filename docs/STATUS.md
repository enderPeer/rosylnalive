# Status

Last updated: 2026-10-18. Waves 0 and 1 are built, reviewed and combined on the integration branch `int/wave1` (draft PR #13, green locally on CPU and GPU). Nothing is merged to `main`; merging is waiting on the user. Wave 2 is running on the integration branch.

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
| O1: oracles (torch32 spec, torch64p truth), tiny model and tokenizer goldens | [#4](https://github.com/enderPeer/rosylnalive/pull/4) | Byte-identical regeneration; oracle self-tests; two reviews approved |
| R1: Vulkan descriptor-heap spike on the RTX 4080 | [#5](https://github.com/enderPeer/rosylnalive/pull/5) | Heap mapping proven bit-exact against CPU and D3D12; ADR and VulkanHeapRecipe contract |
| R3: D3D12 COM interop and device check | [#6](https://github.com/enderPeer/rosylnalive/pull/6) | cl.exe ABI oracle; vtable generator from the SDK headers; RTX 4080 ACCEPTED; found a tail-call use-after-return hazard (ADR R3-0001) |
| C1: cluster core (frames, transports, coordinator, chaos) | [#7](https://github.com/enderPeer/rosylnalive/pull/7) | 200 chaos seeds bit-identical; replica-loss replay; 3-process kill test |
| T1: telemetry runtime, run logs, Markdown and SVG reports | [#8](https://github.com/enderPeer/rosylnalive/pull/8) | Byte-deterministic charts; sample reports (training, routing, thinking, inference, kernels, cluster, parity, gate1) in `docs/reports/samples/` |
| S0b: governance and hygiene checks, contracts index, probes | [#9](https://github.com/enderPeer/rosylnalive/pull/9) | 124 fixtures behave as designed; hygiene job green on GitHub |
| R4a: GPU contracts (heap, tensor table, ArgBlock, backend seam), kernel catalog, shader build | [#10](https://github.com/enderPeer/rosylnalive/pull/10) | Reproducible ShaderSetHash; DXIL and SPIR-V from the pinned DXC only; known-answer vectors |
| O2: tensors, safetensors, CPU reference F32 and F64, comparators, kernel bounds | [#11](https://github.com/enderPeer/rosylnalive/pull/11), stacked on #4 | Tiny model: F64 vs torch64p about 1.5e-15, F32 vs torch32 below 1e-6; bit-equal at 1 and 32 threads |
| R4b: final SPIR-V binding and inspector | [#12](https://github.com/enderPeer/rosylnalive/pull/12), stacked on #10 | Inspector passes all kernels; 18 bad-shader fixtures each fail exactly as expected; contracts-v1.1 waits on a wording fix in R1's contract (wave 2 integration fixes) |
| Integration preview of waves 0 and 1 | [#13](https://github.com/enderPeer/rosylnalive/pull/13) (draft, not a merge unit) | No merge conflicts; local CI sequence green; cpu tier 543 tests; gpu tier green on D3D12 |

## In progress

- **Wave 2** (workflow `wf_d63d9762-c8a`), on `int/wave1`:
  - First, integration fixes: Tailcalls off for GPU and interop code, the R1 contract wording that unblocks contracts-v1.1, the contract registry, the CI steps, a real parity report, and the denormal question. The 127 wave 1 requests are also routed to later workstreams.
  - Then G2 (full-model goldens and tolerances), M1 (tokenizer), M2a (Qwen forward on the CPU), E1 (CpuDevice and conformance harness) and D1 (the D3D12 backend).
  - Then M2b (full-model CPU parity, which earns gate1-cpu) and R5 (the WARP spike).
- R2a, R2b and V1 (the Vulkan side) are waiting on approval to download vk.xml and Vulkan-Headers.

## Next

1. Merge waves 0 to 2 in order, once merging is decided.
2. **Waves 3–5:** kernels, executor, two-node pipeline, dashboard, then Gate 1.
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
