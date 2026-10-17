# LRM development plan: Phase 1 (multi-agent)

Six agents and one integrator build the Qwen3-0.6B forward pass on our own D3D12 and Vulkan stack in six waves. Oracles come first, the biggest risks are retired early, and every green checkpoint is pushed to GitHub.

## 1. Starting point

- **Repo at 42016fa:** design doc, STATUS, the query-only Vulkan binding in `gpu/vulkan`, `tools/bootstrap/toolchain.json` (DXC pin) and `tools/goldens/requirements.lock`.
- **Already present under `LRM_DATA` (`C:/Users/end/dev/lrm-data`):** verified weights, DXC 1.9.2609.5, and `venv-golden` (torch 2.14.1+cpu, transformers 5.20.1, tokenizers 0.23.3). Nothing needs downloading again.
- **Verified while planning, with the pinned DXC:**
  - An explicit `[[vk::binding(0,0)]] RWByteAddressBuffer g_buf[]` gives one SPIR-V variable at set 0 binding 0. This holds with and without `NonUniformResourceIndex`.
  - The emulated `-fvk-bind-resource-heap` form gives one aliased variable per access.
  - `-T rootsig_1_1` compiles the global root signature to 92 bytes.
- **Verified in the venv:**
  - HF Qwen3 runs RMSNorm, softmax and RoPE in fp32 even inside a `.double()` model, so a plain fp64 run is not an fp64 truth. Section 6 defines a patched oracle, `torch64p`, to fix this.
  - torch's fp32 `inv_freq` differs from any closed-form F# formula in some entries. It is therefore pinned as data.
- **Verified in .NET:** InvariantGlobalization silently disables NFC, and `System.Text.RegularExpressions` splits astral letters.

## 2. How the work runs

- **Integrator.** One long-lived session in the main checkout. It does the following:
  - creates worktrees and launches agents;
  - reviews and merges every PR;
  - runs the gpu and model tiers on the exact tree that main will become;
  - tags milestones;
  - updates STATUS through docs-only PRs.
- **Agents.** One session per workstream family, each with its own worktree and branch. At most 6 run at once, plus the integrator.
- **Slices.** An id with an a/b suffix is a slice of one family. The families are S0, R2, R4, M2, X1 and C2. Slices share the family's paths and usually one agent, and each slice is its own PR.
- **Early starts.** A workstream starts when its dependencies have merged, inside the agent cap. Waves are milestone groups, not hard barriers.
- **Execution.** The integrator runs each wave as a multi-agent workflow. Every workstream goes through implement, then two independent adversarial reviews, then fix, then a serialized merge. Agents work in worktrees under `C:/Users/end/dev/rosylnalive-wt/<id>` and never touch the main checkout. After each wave the integrator reads the results, updates STATUS and launches the next wave.
- **Step 0.** This happens before S0a, by the integrator.
  - Copy the planning probes from the session scratchpad to `LRM_DATA/spikes-staging/`, because `%TEMP%` may be cleaned. The files are `probe/*`, `bindtest/*`, `shaders/*.hlsl`, `dxc-probe/probe.hlsl`, `tools/vtbl.py`, `nfc*.fsx`, `planner-contract/nfc_inv.fsx`, `golden-smoke/smoke.py` and `redteam/{fp64check.py,inv.fsx,inv.py,inv_torch.txt}`.
  - Do not copy `dl/` (HF files), third-party docs or `extract-roslyn/`.

## 3. Workstreams

`(F)` means the slice uses its family's paths. Paths: `src/X/**` also implies `tests/X.Tests/**` where such a project exists. `Cmd/` means `apps/Lrm.Cli/Commands/`.

| ID | Name | Owns | Depends on | Size | Acceptance |
|---|---|---|---|---|---|
| S0a | Scaffold, build, test runner, minimal CI | `Lrm.slnx`, root props/targets, `build/**`, `.config/**`, `.github/**`, `tools/bootstrap/**`, `tools/test.ps1`, `tests/Lrm.Testing/**`, Cli fsproj + `Program.fs`, `docs/{STATUS,dev-plan}.md`, `docs/{ownership,project-graph}.json`, `CLAUDE.md`, `README.md` | none | L | Clean-clone Release build has 0 warnings; VkDeviceCheck is ACCEPTED from `apps/`; build-test is green |
| S1 | Lrm.Core foundation | `src/Lrm.Core/**`, `docs/contracts/{lrm-core,telemetry-facade,execution-class}.md` | S0a (stacked) | M | KATs pass; bf16 exhaustive; de-DE identical; NFC canary |
| S0b | Governance, hygiene, probes | `tools/ci/**`, `docs/contracts/{README,CHANGELOG}.md`, `docs/decisions/000*`, `spikes/probes/**` | S0a | M | Checker fixtures fail as designed; hygiene is green |
| R1 | Vulkan descriptor-heap spike | `spikes/vk-heap/**`, `docs/decisions/R1-*` | S0a | M | Bit-exact `scale`, table and bf16; 32 push dwords echo; ADR merged |
| R2a | vkgen v2, hello-compute interop, ABI oracle | `tools/vkgen/**`, `tools/abi-oracle/vulkan/**`, `src/Lrm.Gpu.Vulkan.Interop/**`, `apps/VkDeviceCheck/**`, `Cmd/DevicesVulkan.fs` | S0a | L | cl.exe oracle has 0 failures; regen is idempotent; `vkCreateDevice` OK |
| R2b | Remaining Vulkan interop | (F) | R2a | S | Debug utils, budget, device fault, coop-matrix and timestamps are bound; oracle green |
| R3 | D3D12 COM interop and device check | `tools/d3d12gen/**`, `tools/abi-oracle/d3d12/**`, `src/Lrm.Gpu.D3D12.Interop/**`, `Cmd/DevicesD3D12.fs` | S0a | M | Slots match the probe; oracle passes; RTX 4080 ACCEPTED |
| R4a | GPU core contracts, catalog, shader build | `src/Lrm.Gpu.Core/**`, `src/Lrm.Shaders/**` except `hlsl/kernels/{basic,matmul,attention}`, `src/Lrm.Shaders.Build/**`, contract docs | S0a, S1 | L | DXIL and SPIR-V compile in CI; KATs; generated files diff-clean; contracts-v1 |
| R4b | SPIR-V binding finalized, SPIR-V inspector | (F) | R4a, R1 | S | One variable at set 0 binding 0; push offsets equal the catalog; contracts-v1.1 |
| O1 | Golden tooling, oracles, tiny and tokenizer goldens | `tools/goldens/**` except `qwen/`, `goldens/{tiny,tokenizer}/**`, `goldens/README.md` | S0a | M | Byte-identical regen; oracle self-tests pass; at most 5 MiB |
| O2 | Tensors, safetensors, CpuRef F32/F64, comparators, kernel bounds | `src/Lrm.Tensor/**`, `src/Lrm.Tensor.CpuReference/**` | S1, O1 | L | Tiny model: F64 vs torch64p ≤ 1e-12 and F32 vs torch32 ≤ 1e-6; 1 and 32 threads bit-equal; batch-invariant |
| T1 | Telemetry runtime, run logs, reports | `src/Lrm.Telemetry/**`, `src/Lrm.Reports/**`, `Cmd/Report.fs` | S1 | M | Byte-deterministic MD+SVG; telemetry changes no value |
| C1 | Cluster core | `src/Lrm.Cluster/**`, `Cmd/Chaos.fs` | S1 | L | 200 seeds bit-identical; replica-loss replay; 3-process kill test |
| G2 | Full-model goldens and tolerance calibration | `tools/goldens/qwen/**`, `goldens/qwen3-0.6b/**` | O1 | M | At least 8 gate prompts (or the prefix fallback); tolerances reviewed |
| M1 | Byte-level BPE tokenizer | `src/Lrm.Tokenizer/**` | S1, O1, S0b | M | 100% of the corpus exact in CI |
| M2a | Qwen model, forward pass, CpuOps on the tiny model | `src/Lrm.Models.Qwen/**`, `Cmd/{Backends,Forward,Generate}.fs` | O1, O2 | M | Tiny logits and taps in bounds; 16-step greedy identical; prefill and decode bit-equal |
| M2b | Full-model CPU parity (gate1-cpu) | (F) | M2a, G2, T1 | M | F64 vs torch64p ≤ 1e-10; gate greedy ids identical; parity report |
| E1 | CpuDevice and conformance harness | `src/Lrm.Gpu.CpuDevice/**`, `tests/Lrm.Kernels.Conformance/**` except `Cases/{basic,matmul,attention}` | R4a, O2 | M | Every catalog kernel emulated; every contract violation raises |
| D1 | D3D12 runtime backend | `src/Lrm.Gpu.D3D12/**` | R3, R4a, S1 | L | Selftests bit-identical twice; 1.2 GB round trip; soak; hello-d3d12 |
| V1 | Vulkan runtime backend | `src/Lrm.Gpu.Vulkan/**` | R1, R2a, R4b, S1 | L | Same suite as D1; hello-vulkan |
| R5 | WARP and Agility spikes | `spikes/d3d12/**`, `docs/decisions/R5-*` | D1, E1 | S | ADR; warp CI job only if viable |
| K1 | Basic kernels | `hlsl/kernels/basic/**`, `Cases/basic/**` | R4a, O2, E1, first backend | M | Within KernelBounds; batch-invariant; every dispatch < 100 ms; both backends |
| K2 | Matmul kernel | `hlsl/kernels/matmul/**`, `Cases/matmul/**` | as K1 | M | as K1; M=1 rows equal M=37 and M=512 rows |
| K3 | Attention kernel | `hlsl/kernels/attention/**`, `Cases/attention/**` | as K1 | M | as K1; decode = prefill = chunked prefill |
| X1a | GPU executor on CpuDevice | `src/Lrm.Gpu.Exec/**`, `Cmd/GpuBackends.fs` | R4a, E1, M2a | M | Tiny model bit-identical to CpuOps F32 in CI |
| X1b | GPU executor on Vulkan and D3D12 | (F) | X1a, K1, K2, K3, D1, V1 | M | Tiny within bounds on both GPUs; full-model taps within tolerances |
| C2a | Two-stage pipeline on CPU | `src/Lrm.Serving/**`, `Cmd/{Coord,Node}.fs`, `docs/reports/cluster-dev/**` | C1, M2a | M | Chaos and failover on the tiny model bit-identical in CI |
| C2b | Two-stage pipeline on GPU | (F) | C2a, X1b | S | Same-backend tiny-model splits bit-identical |
| T2 | Dashboard over run logs | `src/Lrm.Dashboard/**`, `Cmd/Dashboard.fs` | T1 | M | Deterministic snapshot; works offline |
| I1 | Gate 1 per backend, bisect, reports | `tests/Lrm.Gate1/**`, `Cmd/Bisect.fs`, `docs/reports/{gate1/backends,inference,kernels}/**` | X1b, G2, M1, M2b, T1 | M | gate1-vulkan and gate1-d3d12 |
| I2 | Gate 1 two-node and GPU chaos | `tests/Lrm.Gate1.Cluster/**`, `docs/reports/gate1/cluster/**` | C2b, X1b, G2 | M | gate1-cluster |
| I3 | Gate 1 single-command harness | `Cmd/Gate1.fs`, `docs/reports/gate1/{README.md,summary/**,dashboard.html}` | I1, I2, T2 | S | `lrm gate1` passes twice on a fresh clone; gate1 tag |

### Workstream notes

**S0a (day 1)**
- Commit 1 is a pure `git mv`: Vulkan to `src/Lrm.Gpu.Vulkan.Interop`, the app to `apps/VkDeviceCheck`, the script to `tools/vkgen/legacy/`. Commit 2 has only the fixes needed to build.
- `Lrm.slnx` holds all 42 shells. `spikes/**` is excluded from the solution and from CI.
- `docs/project-graph.json` feeds `tools/ci/gen-project-graph.ps1`, which writes `build/ProjectGraph.g.props`. `Directory.Build.targets` imports it and injects every ProjectReference and PackageReference. Owner fsprojs hold only Compile items.
- `tools/bootstrap/bootstrap.ps1` handles dxc, vkxml and vkheaders. The tokenizer entry follows in S0b.
- Also in S0a:
  - `tools/test.ps1`;
  - `Lrm.Testing`, implementing tiers and mutexes;
  - `.gitattributes`, `.gitignore` and `.fantomasignore`;
  - the build-test CI job, including the DXC regression step.
- Lrm.Cli compile order is fixed: DevicesVulkan, DevicesD3D12, Backends, GpuBackends, Forward, Generate, Report, Dashboard, Chaos, Coord, Node, Bisect, Gate1, Program.

**S0b (parallel with wave 1)**
- Five checker scripts with fixtures: ownership, layering (reads `project-graph.json`), size, EOL and contract label.
- The hygiene job, the tokenizer bootstrap entry, and the optional SDK entry.
- `docs/decisions/0001-shader-toolchain.md`, stating the exact verified source with no `NonUniformResourceIndex`.
- `spikes/probes/`, copied from staging. Every file is read before it is committed.

**S1**
- ContentId, Hash and Canonical encoding.
- DType codes, Bf16 and F16 conversions.
- SplitMix64, Philox4x32-10 and DetRng.
- Ids, Invariant helpers and LrmPaths.
- `Process.init`, and the telemetry facade.
- The **ExecutionClass record and its id**. They live in Lrm.Core so that Models.Qwen and Cluster can name them.

**R1**
- A throwaway spike that runs both binding variants with `scale`, `args_echo`, `table_echo` and `bf16_unpack`.
- It is compared against the same HLSL on D3D12.
- The ADR records the mapping values, the features, the memory types and the reserved range.

**R2a / R2b**
- The F# generator reads vk.xml 1.4.365. It handles VkFlags64, handles, unions, basetypes, funcpointers and struct arrays.
- The ABI oracle dumps the runtime layout to C `static_assert`s, which cl.exe compiles against Vulkan-Headers.
- Slice a is the hello-compute subset; slice b is the rest.

**R3**
- The F# vtable generator counts `DECLSPEC_XFGVIRT` markers.
- Also: ComPtr, the hidden RetVal convention, the ABI oracle, and device discovery accepting SM 6.6+ with binding tier 3.

**R4a**
- HeapLayout, TensorTable and ArgBlock.
- The catalog and its generator, which writes `lrm_args.g.hlsli` and `Kernels.g.fs`.
- IGpuBackend and RecordingBackend.
- `lrm.hlsli`. Its SPIR-V `LRM_BUF` uses the statically verified explicit array, marked provisional.
- The shader build, and a test that `rootsig.bin` equals the canonical bytes. There is no DXBC groupshared parse: DXC validation enforces 32 KiB.

**R4b**
- Applies R1's ADR to the SPIR-V side.
- Adds the SPIR-V inspector. It checks one resource variable, the push-block offsets, LocalSize, no float atomics and no 16-bit types.

**O1**
- `common.py`, implementing the ReferenceOracles contract.
- The tiny model in the real config schema, with bf16-rounded weights and no `lm_head`.
- A tokenizer corpus of at least 2,000 cases, including case-fold contractions.
- The oracle self-tests.

**O2**
- CpuRef in F32 and F64. `ropeTable` takes pinned `inv_freq` bits as input.
- The safetensors reader and writer, the comparators, and `Goldens.open`.
- KernelBounds, in `src/Lrm.Tensor/KernelBounds.fs`.

**T1**
- The JSONL sink and the RunLog reader.
- Metering: wall time, GPU time, bytes moved, FLOPs, tokens and memory per arena. It is attributed per component, per layer, per stage and per node.
- The deterministic SVG charts. The chart kinds are line (loss and metrics over steps), multi-series per layer, heatmap (layer × step, expert × layer), histogram, a stacked kernel timeline, a stage Gantt chart (cluster) and a per-layer error-bar chart.
- Templates: parity, inference, kernels, cluster and gate1.
- Phase 2/3 templates, filled with synthetic data so the visuals are reviewed early:
  - training (loss per stage, gradient and activation statistics per layer);
  - routing (expert usage and router entropy per layer);
  - thinking (Roslyn calls and their outcomes, as a trace).
- **Visual-first rule:**
  - Every wave exit commits generated reports under `docs/reports/`.
  - Every PR that adds a metric also adds it to a template.
  - The integrator reviews the reports and dashboard through screenshots in the browser pane.

**C1**
- The frame codec and the transports (InProc, TCP on 127.0.0.1, Chaos).
- VirtualClock, BlobStore and WorkUnit ids.
- The coordinator, with epochs, leases, reissue, Ready gating and replica digests.
- ChaosProxy, and a fake 3-stage executor.

**G2**
- Runs torch32, torch64p and bf16 (calibration only).
- Builds a seeded pool of at least 200 candidate prompts and calibrates on a fixed subset.
- Selects the gate prompts, writes `tolerances.json`, and dumps the 128-wide `inv_freq` bits and the `T_spec` tables.

**M1**
- A Rune state machine for the exact Split regex, an added-token trie, NFC and BPE.
- Simple case folding in the contraction group.

**M2a**
- `Qwen3Config.load` accepts both `rope_theta` and `rope_parameters.rope_theta`.
- `Qwen3Weights.open` accepts `local:` revisions, which are verified through SHA256SUMS only.
- Also: IForwardOps, `Forward.runStage`, KvCache, `Greedy.run` with `StopAtEos`, CpuOps, and the `RopeConstants` table keyed by (theta, head_dim).
- `Cmd/Backends.fs` holds the ops registry; it registers cpu-f32 and cpu-f64.

**M2b**
- Model-tier acceptance against G2, which earns tag gate1-cpu.

**E1**
- CpuDevice decodes real ArgBlocks and tensor rows and enforces every contract rule.
- One emulator per catalog kernel.
- The conformance harness.

**D1**
- In-box D3D12 at SM 6.6. One shader-visible heap of 64 raw UAVs.
- **Per-page resource-state tracking**: `Copy` inserts UAV↔COPY transitions.
- `GetDeviceRemovedReason` is checked after every submit in test builds, with DRED breadcrumbs.

**V1**
- Implements the R1 recipe exactly.
- Compute queue family 2, timeline semaphore, `vkCmdPushDataEXT` with 128 bytes, and a global barrier between dependent dispatches.
- VK_EXT_device_fault diagnostics.

**R5**
- Runs selftests and the harness on in-box WARP.
- Agility runs only if AP-4 is approved. Nothing changes system settings.

**K1**
- `embed_gather_bf16`, `rmsnorm_rows`, `rope_apply` (table-driven), `silu_mul`, `add`, `copy_rows` and `argmax_rows`.

**K2**
- `matmul_bf16w`. 64×64 tiles; sequential k accumulation with `mad`; one kernel for decode, prefill and `lm_head`.

**K3**
- `attention_gqa_causal`. One group per (row, head); fixed 64-key blocks; online softmax; causal mask as a loop bound.

**X1a / X1b**
- Implements GpuForwardOps with a two-pass static plan.
- Weights are streamed in, and prefill attention is chunked.
- **Taps are snapshotted at record time** by Copy into a Taps arena.
- `Cmd/GpuBackends.fs` registers cpu-device, vulkan and d3d12 through `Backends.register`.

**C2a / C2b**
- QwenStageExecutor; the fp32 residual boundary; KV rebuild by exact replay. A fast rebuild must equal replay bit for bit.
- `lrm coord` and `lrm node`, and the multi-process harness.

**T2**
- One self-contained HTML file with no network access and a `--watch` mode.
- Light and dark themes, usable at phone width.

**I1**
- The per-backend parity suite, `lrm bisect`, long-prompt logit runs, and the bench-tier performance baseline.
- Fixes always go to the owner's fix branch.

**I2**
- Full-model splits Vulkan|Vulkan, D3D12|D3D12 and Vulkan|D3D12.
- GPU chaos with 2 replicas per stage plus a spare, maxPos 1024 and 25 seeds.

**I3**
- `lrm gate1` runs, in order:
  1. environment check;
  2. tokenizer;
  3. CpuRef;
  4. conformance;
  5. the I1 suite;
  6. the I2 suite;
  7. a determinism re-run.
- It then writes the summary report and the dashboard export.

## 4. Waves

| Wave | Workstreams | Agents | Exit criteria |
|---|---|---|---|
| 0 | S0a, S1 | 1 + integrator | AP-0 and AP-1 answered (asked with this plan). PR0a merged; build-test green on main; S1 merged; tag m0-scaffold. |
| 1 | S0b, R1, R2a, R2b, R3, R4a, R4b, O1, O2, T1, C1 | ≤ 6 + integrator | ABI oracles green. R4a merged: tag contracts-v1 (S1, R4a, O2, C1 contracts). R1's ADR merged, then R4b: tag contracts-v1.1. Tiny and tokenizer goldens reproducible. O2 tiny checks pass in CI. C1's 200 seeds and kill test pass. Tag wave-1-done. |
| 2 | G2, M1, M2a, M2b, E1, D1, V1, R5 | ≤ 6 | Goldens double-run identical; tolerances reviewed. Tokenizer 100%. Tag gate1-cpu. E1 selftests pass on both GPUs. Tags hello-d3d12 and hello-vulkan; selftests bit-identical across backends. R5's ADR recorded. Tag wave-2-done. |
| 3 | K1, K2, K3, X1a, X1b, C2a, C2b, T2 | 6 | Every kernel passes conformance on CpuDevice (CI) and both GPUs. X1b within tolerance with identical greedy ids. C2a and C2b green. T2 merged. Tag wave-3-done. |
| 4 | I1, I2 | 2 + ≤ 3 fix sessions | Tags gate1-vulkan, gate1-d3d12 and gate1-cluster. Reports committed. |
| 5 | I3 | 1 | `lrm gate1` passes twice on a fresh clone of main; tag gate1; STATUS marks Phase 1 done. |

D1 may start in wave 1 once R3 and R4a merge. X1a and C2a may start in wave 2 once M2a merges. Each counts against the cap. LRM_REQUIRED_BACKENDS is set to the first merged backend, then to `d3d12,vulkan` once V1 merges.

**Critical path:** S0a → S1 → R4a → E1 → K1–K3 (on D1) → X1b → I1 → I3. R1 and V1 run beside it and no longer block D3D12 work.

## 5. Contracts

Contract files change after contracts-v1 only through a `contract-change` PR. That PR needs a CHANGELOG entry and updated known-answer vectors. Each owner writes the full text in `docs/contracts/<name>.md`.

| Contract | Owner | Key terms |
|---|---|---|
| ProjectGraph | S0 | `docs/project-graph.json` is the only source of ProjectReferences and PackageReferences, injected through `build/ProjectGraph.g.props`. Core references nothing. Models.Qwen and Cluster never reference GPU projects. Gpu.Exec never references a concrete backend or CpuDevice. Shaders → Shaders.Build is a build-only edge. Only apps and tests wire backends. `spikes/**` is not built. Run apps as `dotnet <dll>`, x64 only. Every exe calls `Process.init()` first. |
| TestTiers | S0 | Expecto exes with root lists `cpu`, `gpu`, `model` and `bench`. Bench is local and never gates. A missing tier or backend fails the run. Mutexes are `Local\LrmGpu.d3d12`, `Local\LrmGpu.vulkan` and `Local\LrmModelCpu`, acquired in that order. The model tier takes a GPU mutex only for GPU cases. Workers never acquire one. AbandonedMutex counts as acquired. Ticket files in `LRM_DATA/locks` print the queue position and holder PID every minute. The wait limit is 120 min. An integrator flag file gives verify runs priority. GPU tests run twice and compare bytes. The fixed FsCheck seed can be overridden by `LRM_FSCHECK_SEED`. |
| DataRootAndToolchain | S0 | `LRM_DATA` defaults to `C:/Users/end/dev/lrm-data`. All pins (URL, size, hash, path) live in `toolchain.json`. Nothing under `LRM_DATA` is ever committed. On CI, bootstrap fills a cached temp root. |
| AgentPermissions | S0 | No committed allowlist (AP-0 declined). Agents run under the session's permission mode. Merge, tag and protection commands are run only by the integrator. |
| OwnershipAndContractChange | S0 | `ownership.json` maps families to globs, where `!` excludes. Branches are `ws/<id>-<slug>`, `ws/<id>-fix-<slug>` and `int/<slug>`. Hygiene fails out-of-path edits. There are no mid-plan handovers. Breaking changes bump the contract version and merge together with consumer fixes. Additive catalog changes get same-day review. |
| LrmCore | S1 | Canonical encoding is little-endian, with domain-separated SHA-256 (`tag‖0x00‖payload`); maps are sorted by the UTF-8 bytes of the key. DType codes: F32=1, BF16=2, F16=3, I32=4, U32=5, U8=6, F64=7, I64=8. `Bf16.narrowRne` rounds to nearest even; every non-NaN value round-trips; NaN becomes `0x7FC0`, or `0xFFC0` when the sign is set. Philox4x32-10 and DetRng(seed, stream). Invariant helpers with `R` float formatting. `Process.init` sets the invariant culture and fails loudly in globalization-invariant mode. |
| TelemetryFacade | S1 | Ambient `ITelemetry` (no-op by default) with Span, Counter, Gauge, Histogram, Event, GpuInterval, Meter, Check and Child. Reserved names: `gpu.*`, `model.*`, `cluster.*`, `conformance.case`, `golden.compare`, `tokenizer.case`, `mem.arena`. Telemetry never changes values, and timestamps never enter hashes. |
| ExecutionClass | S1 (type); R4a (GPU fill rule) | `{Backend: cpu-f32\|cpu-f64\|cpu-device\|vulkan\|d3d12\|d3d12-warp; DeviceId; Driver; ShaderSetHash option; Arch}`. Its id is `Canonical.hash('lrm.xclass/1', …)`. Bit-identity claims hold only within one class. GPU backends fill DeviceId, Driver and ShaderSetHash. |
| RunLog | T1 | `LRM_DATA/runs/<runId>/manifest.json` (schema `lrm.run/1`, includes the executionClass and driver) and `events.<node>.jsonl` with keys `v, t, k, n, d, x, u, node, seq, a`. |
| ReportTemplates | T1 | Templates in `src/Lrm.Reports/templates/*.md` with value, table, chart and each placeholders. Builders are pure functions over RunLogReader. Each chart is a byte-deterministic SVG file in a palette that works on light and dark themes. |
| HostTensorAndSafetensors | O2 | Memory-mapped reader that validates ranges, and a deterministic writer: sorted names, header padded with spaces to 8 bytes. |
| CpuReferenceSemantics | O2 | Sequential FMA reductions in increasing index order; parallelism only over output rows; no `Vector<T>`. Weight used as `w`, not `1+w`. NeoX RoPE. `ropeTable(invFreqBits, maxPos)`: freq = f32(f32(pos)·inv_freq), cos and sin = f32(f64 cos/sin). F64 mode uses the same fp32 table, widened. KV cache: K stored after norm and RoPE, V raw, both fp32. argmax returns the lowest index on ties; NaN raises. |
| ComparatorsAndGoldenReader | O2 | relL2 in fp64, maxAbs, maxUlp, KL, top-k agreement, top-2 margin. `Goldens.open` verifies SHA-256 before use. `ToleranceSpec.load`. |
| KernelBounds | O2 | `src/Lrm.Tensor/KernelBounds.fs`. Bit-exact: add, copy_rows, embed, bf16_unpack, scale_f32 and the argmax index. All others: relL2(gpu, CpuRef64) ≤ max(1e-6, 2·relL2(CpuRef32, CpuRef64)) per case; the measured reference error is logged. The factor changes only through contract-change, with evidence. |
| ReferenceOracles | O1 | Defined below. |
| GoldenSet | O1 (format); G2 (Qwen content) | `goldens/<set>/g<N>/manifest.json`, schema `lrm.golden/1`: env snapshot, oracle patches, tap methods, EOS setting, threads, double-run result, argmax tie probe. Large files live in `LRM_DATA/golden/`. No LFS and no Releases. Tiny model at `goldens/tiny/g1/model/`: `config.json` in the real schema, bf16 `model.safetensors` without `lm_head`, `SHA256SUMS`, and `REVISION` set to `local:tiny-g1`. |
| ActivationNames | O1 | `embed.out`; per layer `L{i}.` + input_layernorm, q/k/v_proj, q_norm, k_norm, q_rope, k_rope, v_cache, o_proj.in, o_proj.out, resid_mid, post_attention_layernorm, mlp gate/up, down.in, down.out, out; then `final_norm.out`, `logits`. Shapes `[T, features]`. |
| ToleranceSpec | G2 (numbers); O2 (types) | Pairs: cpu64-vs-torch64p, cpu32-vs-torch32, cpu32-vs-torch64p, gpu-vs-cpu32, gpu-vs-torch32, gpu-vs-torch64p, vulkan-vs-d3d12. e_ref = relL2(torch32, torch64p). bound = clamp(k·e_ref, floor, 10·floor). CPU32 floors: relL2 1e-5, max\|Δlogit\| 2e-3, KL 1e-7. GPU floors: relL2 1e-4, max\|Δlogit\| 1e-2, KL 1e-6. CPU64 vs torch64p: 1e-10 on the full model, 1e-12 on the tiny model. Greedy rules are in section 6. |
| HeapLayoutAndTensorTable | R4a | 64 slots, written once. Slot 0 is the table; pages are at most 1 GiB each. Arena kinds in order: Weights, Rope, KvCache, Activations, Scratch, Logits, Taps, then reserved Master, Grad, AdamM, AdamV, RoslynIn, RoslynOut. TensorRow is 64 B. The table has 16,384 rows; token 0 is null. Vulkan slot i sits at byte 16·i, D3D12 slot i at descriptor index i. |
| KernelAbiAndCatalog | R4a | 128-byte ArgBlock; dword0 = `lrm_groups_x`. D3D12 uses 32 root constants at b0; Vulkan uses `vkCmdPushDataEXT` at offset 0. The frozen catalog keeps the plan's kernel list, fields, thread counts and grids. No float atomics, wave ops or 16-bit types. groupshared at most 32 KiB; every dispatch < 100 ms; x ≤ 65,535. |
| GpuBackendInterface | R4a | IGpuBackend, IGpuHeap, ICommandList and IKernelSet, with factories `Backend.enumerate` and `tryCreate`. Vulkan requires `VK_EXT_descriptor_heap`; D3D12 requires SM 6.6 and binding tier 3. `Copy` is state-correct on every backend. |
| ShaderBuildAndSourceConvention | R4a (SPIR-V `LRM_BUF`: R4b) | Only the pinned DXC 1.9.2609.5, verified by `--version`. Incremental MSBuild target. DXIL `cs_6_6`; SPIR-V vulkan1.3. `rootsig.bin` is checked against the canonical bytes. One kernel per file, using `LRM_ARGS`, `LRM_BUF` and `lrm_tensor` only. ShaderSetHash. |
| BatchInvariance | R4a (rule) | Reduction order depends only on the reduction length. A row computed alone equals the same row in a batch, and prefill equals decode, within one class. Failover never relies on this. |
| VulkanHeapRecipe | R1 | Mapping set 0, binding 0, HEAP_WITH_CONSTANT_OFFSET, stride 16. Variant chosen, push data at offset 0 (128 B), reserved range at the end, features and memory types. |
| VulkanInteropApi | R2 | vk.xml names, ABI order, generated dispatch tables, and `bindings.json` as the single list of bound items. pNext chains live in one stack frame. Wildcard arms on every enum match. |
| D3D12ComIdiom | R3 | ComPtr struct, generated slots and IIDs, cached delegates, RetVal convention. No ComImport and no RCWs. |
| ConformanceHarness | E1 | Fields: KernelCase, CaseIo, and bounds from KernelBounds. Each case runs twice bit-identically, is checked against the bounds, checked for batch invariance and timed (minimum of the two runs < 100 ms), then emits a `conformance.case` event. |
| TokenizerApi | M1 | `QwenTokenizer.load` checks SHA-256 `aeb13307…`. Encode, EncodeBytes, Decode with errors=replace. No BOS. 151,669 real ids. |
| ForwardOpsAndQwenModel | M2 | `IForwardOps<'t>` and `Forward.runStage`. `Greedy.run(…, StopAtEos)`; parity runs pass false. The canonical schedule is prefill, then single-token decode steps. RopeConstants hold the pinned `inv_freq` bits. `Cmd/Backends.fs` holds the ops registry. |
| GpuExecutor | X1 | `GpuForwardOps.create`. Two-pass static plan; streamed weights; chunked attention; taps copied into the Taps arena at record time. `Cmd/GpuBackends.fs` registers the GPU backends. |
| ClusterProtocol | C1 | 80-byte frame header plus a SHA-256 trailer. Message kinds 1–10, plus Ready=11. `fwd.stage` units are hash-chained by PrevUnitId. First result wins; differing duplicates raise `cluster.divergence`. Epochs and leases. Deadlines count only after Ready: 30 s initially, then clamp(4·p95+50 ms, 250 ms, 30 s) once 8 samples exist. |
| StagePipeline | C2 | `lrm coord`, `lrm node` and `lrm chaos-proxy`. Stage 0 runs embed plus layers 0–13; stage 1 runs layers 14–27 and the head. KV rebuild uses exact replay. |
| Gate1Evidence | I1, I2, I3 | Schema `lrm.gate1/1`. Reports live under `docs/reports/gate1/`. A gate tag is pushed only when every check passes. |

## 6. Reference oracles and golden rules (O1, G2)

- **torch32 (the spec).** Unpatched transformers 5.20.1, fp32, eager attention. This is the official reference implementation.
- **torch64p (the truth).** The model in `.double()` with three recorded patches:
  - `Qwen3RMSNorm.forward` stays in the input dtype.
  - `eager_attention_forward` runs softmax in `query.dtype`.
  - `Qwen3RotaryEmbedding.forward` returns `T_spec` widened to fp64.
- **T_spec.**
  - Inputs: torch's own fp32 `inv_freq`, dumped as u32 bits.
  - freq = f32(f32(pos)·inv_freq); cos and sin = f32(f64 cos/sin).
  - Dumped by O1 (tiny model, committed) and G2 (Qwen positions 0..8191; local, hash committed).
  - CpuRef's table must equal it bit for bit. On a mismatch, an ADR is written and CpuRef loads the dumped table.
- **Guards.**
  - Python runs with `python -I`.
  - Environment: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `USE_HUB_KERNELS=0`.
  - The tool asserts that `kernels` is not importable.
  - `use_deterministic_algorithms(True)`; 8 threads, falling back to 1 if the double run differs.
- **Self-tests.** Patched-fp64 RMSNorm against a hand-written fp64 RMSNorm must agree within 1e-15. One patched attention layer against a hand-written fp64 version must agree within 1e-13.
- **Taps.** Module forward hooks and pre-hooks are used, plus exactly one function wrapper. The wrapper replaces `modeling_qwen3.apply_rotary_pos_emb` and records `q_rope` and `k_rope`; the manifest says `tap.method: wrap`. `k_rope` must equal the cache contents once.
- **Greedy.**
  - The manual 64-step loop ignores EOS.
  - `generate()` gets a fresh `GenerationConfig(do_sample=False, max_new_tokens=64, eos_token_id=None)`. It must give 64 tokens equal to the manual loop and to a no-cache recompute.
  - `min_new_tokens` is never used, because it masks the EOS logits.
- **Prompt pool.** At least 200 candidates, drawn by seed from committed corpora in `tools/goldens/qwen/corpus/`: C# snippets written for this repo, Apache LICENSE paragraphs, design-doc paragraphs, multilingual lines and ChatML wrappers. All candidates get torch32 greedy runs with margins. Calibration uses a fixed subset of 12 prompts.
- **Gate prompts.**
  - At most 900 tokens.
  - A minimum top-2 margin of at least max(0.05, 4·τ_logit) over 64 steps.
  - At least 8 prompts, including English, C# and ChatML.
  - Fallback if fewer pass: gate on the longest prefix before the first low-margin step (at least 16 steps). Teacher-forced argmax covers the rest wherever the margin is at least 2·τ_logit.
- **Long prompts.** About 1,000 and at most 4,000 tokens. They get logit checks on a single node only.

## 7. Git and push policy

- **Forge.** `origin` = github.com/enderPeer/rosylnalive. `main` is the only long-lived branch and is always green in CI. At wave exits it is also green on the local gpu and model tiers.
- **Worktrees.** Created with `git worktree add C:/Users/end/dev/rosylnalive-wt/<id> -b ws/<id>-<slug> origin/main`. The integrator owns the main checkout and a `rosylnalive-wt/verify` worktree.
- **Commits.** Subject `<ID> <area>: <summary>`, with a body saying why. Every message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never commit weights, large goldens, toolchains, vk.xml, headers, `artifacts/` or run logs.
- **Pushing.** Push after every green checkpoint (0-warning build, fantomas clean, the touched tiers passing). Also push at least every 90 minutes and before a session ends. Push only to your own branch with `git push -u origin <branch>`. After a rebase, use `--force-with-lease`.
- **No direct pushes to main.** STATUS and dev-plan updates go through `int/status-<date>` PRs, or are folded into the next merge.
- **PRs.**
  - Open a draft right after the first push: `gh pr create --draft --base main --title "[<ID>] <name>"`.
  - The body follows the template: goal, ownership statement, contracts touched, pasted `tools/test.ps1` summaries (device, driver, ShaderSetHash), run ids and reports, a STATUS paragraph, and requests to the integrator.
  - The body ends with the Claude Code attribution line from the session reminder.
- **Stacked slices.** Rebase with `git rebase --onto origin/main <old PR-a head sha> ws/<id>-b`. Record the old sha in the PR-b description.
- **Merging (integrator only, one PR at a time).**
  1. Require green build-test and hygiene on an up-to-date head, plus the contract-change label where needed.
  2. For GPU- or model-touching PRs, build the exact future tree in the verify worktree: `git checkout --detach origin/main && git merge --squash --no-commit origin/<branch>`.
  3. Run `tools/test.ps1 -Tier gpu` (and `model`) there.
  4. Clean up with `git reset --hard && git clean -fd -e artifacts/`.
  5. Merge with `gh pr merge --squash --delete-branch`. Repeat the check if main has moved.
  6. PR0a uses rebase-merge so the `git mv` keeps file history.
  7. If main goes red, merge an `int/revert-*` PR at once. The owner then fixes forward.
- **CI.**
  - `ci.yml` triggers on push to main, pull_request and workflow_dispatch.
  - Concurrency group `ci-${{ github.ref }}` with `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`.
  - build-test runs on windows-latest with a 30-minute timeout: checkout, setup-dotnet, cache, bootstrap, fantomas, codegen diffs, Release build (shaders), cl.exe oracles, then the cpu tier with JUnit output.
  - hygiene runs on ubuntu-latest: ownership, layering, sizes, EOL and the contract label.
  - Optional jobs: d3d12-warp (only after R5) and nightly model-cpu (only after AP-5).
  - CI liveness tests use 60 s timeouts and assert digests only.
  - `.fantomasignore` excludes `**/*.g.fs`, `tools/vkgen/legacy/**` and the two legacy generated files.
- **Branch protection (AP-2, non-blocking).**
  - Required checks build-test and hygiene; branches up to date; linear history; no force pushes; no deletion.
  - `enforce_admins` is false, which keeps an emergency path. Policy still forbids direct pushes.
  - Until AP-2 is answered, the integrator enforces these rules by hand.
- **Tags.** Annotated, pushed by the integrator after verification: m0-scaffold, contracts-v1, contracts-v1.1, wave-1-done, hello-d3d12, hello-vulkan, gate1-cpu, wave-2-done, wave-3-done, gate1-vulkan, gate1-d3d12, gate1-cluster, gate1.

## 8. Verification gates

1. **Every PR.** CI must be green. GPU, model and kernel PRs paste local tier summaries, which the integrator re-runs on the future tree.
2. **Contract drift.** Known-answer vectors cover Canonical encodings, Philox, bf16, ArgBlock, TensorRow, WorkUnit ids, frames and ExecutionClassId. ShaderSetHash reproduces, and generated files are diff-clean.
3. **ABI.** cl.exe `static_assert`s run against the real headers. D3D12 slots match the probe. The SPIR-V inspector and the `rootsig.bin` known answer pass.
4. **Oracle chain.** torch64p is the truth and torch32 the spec. Then CpuRef F64 (1e-12 on the tiny model, 1e-10 on the full model), then CpuRef F32 (calibrated), then CpuDevice (bit-exact to CpuRef F32), then each GPU (KernelBounds and tolerances). HF tokenizers is the oracle for M1.
5. **Determinism.** CpuRef gives the same bits at 1 and 32 threads. Every GPU test runs twice. Batch invariance holds. Prefill equals decode. Fast rebuild equals replay. The divergence tripwire is armed.
6. **Fault tolerance.** C1 in CI: simulation, 200 seeds, a 3-process kill test. C2a in CI: the tiny model with failover and 50 proxy seeds. I2 locally: the full model with 25 GPU chaos seeds.
7. **Wave exits.** On a fresh clone: bootstrap, build, then the cpu, gpu and model tiers twice. Then tag and update STATUS through a PR.
8. **Driver.** Gates use the driver recorded in the run manifest (616.92 at planning time). After a driver change, the integrator re-runs the gpu and model tiers on main, re-baselines, and records it in STATUS.
9. **Gate 1.** `lrm gate1` passes twice in a row on a fresh clone of main. The checks:
   - the tokenizer corpus is exact;
   - CpuRef F32 and F64 match the goldens;
   - the conformance matrix passes;
   - per backend: logits within tolerance, gate greedy ids identical, teacher-forced agreement;
   - same-backend splits are bit-identical;
   - the mixed split is within tolerance;
   - GPU chaos runs are bit-identical;
   - the determinism re-run passes.

## 9. Approval points

| AP | When | What | If declined |
|---|---|---|---|
| AP-0 | With this plan, before any agent starts | Commit the `.claude/settings.json` allowlist (AgentPermissions), plus the integrator's local merge, tag and protection permissions. | **Declined by the integrator (2026-10-17).** Agent permissions remain the user's own settings, so no allowlist is committed. Workflow agents run under the session's permission mode. S0a does not create `.claude/settings.json`. |
| AP-1 | With this plan | Downloads: vk.xml v1.4.365 (3,341,644 B); Vulkan-Headers v1.4.365 (size stated); Expecto 11.1.1, Expecto.FsCheck 11.1.1, FsCheck 3.4.0 and fantomas 8.0.10. On CI runners only: the DXC nupkg (53,395,273 B) and tokenizer.json (11,422,654 B). | Phase 1 cannot start. **Status:** test and formatting packages and DXC approved 2026-10-17; vk.xml, Vulkan-Headers and the CI tokenizer fetch were asked of the user the same day. |
| AP-2 | Any time; non-blocking | Branch protection through `gh api`. | Rules enforced by hand. |
| AP-3 | Optional, when V1 starts | LunarG SDK 1.4.363.0, copy-only install into `LRM_DATA`. | Work proceeds without validation layers. |
| AP-4 | Optional, R5 | Agility SDK NuGet 1.619.6 (35.5 MB). | Spike skipped. |
| AP-5 | Optional, after gate1-cpu | Nightly CI that caches the 1.5 GB weights. | Model tier stays local. |
| AP-6 | Contingency | A second golden environment (transformers 5.18.0, about 150 MB). | Rely on the other oracles. |
| AP-7 | Contingency | WARP NuGet, or any public upload of goldens. | Not done. |

**Hand-off note:** you may want to pause NVIDIA driver auto-updates for the length of Phase 1. Agents will never change that setting.

## 10. Risks

| Risk | Mitigation |
|---|---|
| The Vulkan heap mapping is unproven at runtime. | R1 spike in wave 1. R4a ships D3D12 without waiting for it; R4b and V1 follow the ADR. |
| HF's fp64 path is not fp64 (verified). | torch64p patches with self-tests; e_ref is measured against torch64p. |
| `inv_freq` bits are implementation-specific (verified). | Pinned torch values. A cpu-tier bit-equality test; `T_spec` is shared by torch64p and CpuRef. |
| Kernel bounds below the fp32 floor. | KernelBounds is relative to CpuRef32's own error. |
| Too few high-margin gate prompts. | A pool of at least 200 seeded prompts, plus the prefix fallback with teacher forcing. |
| Agents blocked on permission prompts. | AP-0 allowlist, asked up front. |
| GPU test contention and timeouts. | Per-backend mutexes, ticket queue, 120 min wait, integrator priority, minimum-of-two-runs timing. |
| No validation layers. | CpuDevice enforces the rules; conformance; run-twice; DRED and device-fault diagnostics; optional AP-3. |
| transformers 5.20.1 is very new. | Independent oracles, a class and config assertion guard, AP-6. |
| Golden nondeterminism across threads. | Double run with hash compare and a 1-thread fallback. |
| Bit identity is impossible across backends. | ExecutionClass scopes every claim; cross-class checks use tolerances. |
| TDR on long dispatches. | 100 ms budget; chunked prefill. |
| Driver auto-update mid-plan. | Manifest-recorded driver; re-baseline procedure. |
| VRAM in I2 chaos runs. | maxPos 1024, gate prompts at most 900 tokens, digests only. |
| Single coordinator. | Replay recovers worker state; the limit is stated in the report; coordinator replication is Phase 2. |
| Integrator bottleneck. | Automated checks; 6-agent cap; deep review only for contract and runtime PRs. |
| Unverified facts: cl.exe and SDK headers on runners, WARP SM 6.6, Expecto.FsCheck with FsCheck 3, fantomas on F# 10. | Header-hash-gated CI steps; R5; an FsCheck adapter fallback; drop the fantomas check rather than block work. |

## 11. Decisions log

1. **Base plan.** Plan C's oracle-first order, with ideas grafted from A and B. Early starts are explicit; at most 6 agents run.
2. **Ownership.** Owned by family. Slices share paths; there are no mid-plan handovers.
3. **References.** ProjectReferences come only from `project-graph.json`; fsprojs hold Compile items only.
4. **Shaders.** HLSL 2021 at SM 6.6, compiled only by the pinned DXC. One source per kernel; explicit `g_buf[]` on SPIR-V; no Slang in Phase 1.
5. **R4 split.** R4a (contracts-v1, D3D12-ready) and R4b (SPIR-V binding, contracts-v1.1).
6. **ExecutionClass.** Lives in Lrm.Core.
7. **Validation.** `rootsig.bin` is checked against canonical bytes. DXC validation enforces groupshared; there is no DXBC parse.
8. **Precision.** Weights stay bf16, widened exactly. Activations, residual and KV cache are fp32.
9. **Oracles.** torch32 is the spec, torch64p the truth. bf16 goldens serve Phase 2 calibration only.
10. **RoPE.** `inv_freq` is pinned data. `T_spec` is shared by torch64p and CpuRef.
11. **Kernel bounds.** Relative to the reference's own fp32 error.
12. **Tiny model.** Real config schema, bf16-rounded weights, no `lm_head`, a `local:` revision.
13. **M2 and the CLI.** M2 is split so the tiny model unblocks X1a and C2a. The CLI backend registry is split into `Backends.fs` (M2) and `GpuBackends.fs` (X1), in a fixed compile order.
14. **S0 split.** PR0a lands on day 1; PR0b, the governance part, follows in parallel.
15. **Mutexes.** Per-backend mutexes plus a model-CPU mutex, with a bench tier for timing.
16. **EOS.** Ignored in every parity run; `generate()` gets an explicit config with no EOS.
17. **Failover.** Exact replay is primary. Fast rebuild is allowed only for invariant classes and must equal replay.
18. **D3D12 and taps.** D3D12 tracks resource state per page. Taps are copied at record time.
19. **Deadlines.** Count only after a node is Ready.
20. **Status updates.** Through PRs; `enforce_admins` is false; CI cancels in-progress runs only for PRs.
21. **Gates.** Use the manifest-recorded driver.

## 12. Phase 2 outline

1. **contracts-v2.** Adds:
   - backward and loss catalog entries;
   - the Master, Grad, AdamM and AdamV arenas, each spanning several 1 GiB pages;
   - the `grad.shard` and `optim.step` unit kinds;
   - a checkpoint format (safetensors + manifest + hash chain);
   - tolerances v2: gradient relL2 at most 1e-3 against torch32.
2. **Oracles first.**
   - CpuRef VJPs in F64 and F32, checked with fp64 finite differences and the tiny-model gradient and AdamW goldens.
   - G3 full-model fp32 gradient goldens (local) with committed norms and seeded slices.
   - One torch AdamW step, `foreach=False`.
3. **Lrm.Autodiff.** An op-level tape over IForwardOps with a reserved RoslynHop node, and per-layer checkpointing. The tied-embedding gradient is a sort-based scatter-add plus the lm_head gradient.
4. **Backward kernels.**
   - matmul dX and dW with a fixed-order K reduction;
   - RMSNorm, SiLU, RoPE (transposed rotation) and residual;
   - attention backward;
   - fused, chunked cross-entropy over 151,936 classes.
   All are batch-invariant, have no float atomics, keep every dispatch under 100 ms, and pass conformance on all backends.
5. **Lrm.Optim.** AdamW with fp32 master weights; a bf16 compute copy; about 10.7 GB of state on the 16 GB GPU, measured; ZeRO-1 sharding; accumulation in shard-ID order.
6. **Fault-tolerant data parallelism.**
   - fixed shard-order reduction trees;
   - recompute of lagging shards;
   - a replicated coordinator (3 replicas, majority leases);
   - state held on at least 2 nodes;
   - asynchronous checkpoints.
7. **Gate 2 (tag gate2).** One-step gradients match; a short fine-tune lowers the held-out loss; a chaos run (lag, drops, kills, a partition, coordinator failover) gives bit-identical weights.
8. **Training observability.** Fills the Phase 1 training and loss templates and the dashboard views without API changes.
9. **Performance track after Gate 1.** Vulkan cooperative-matrix GEMM, pre-recorded decode command buffers, fused kernels, an optional bf16 KV cache. All admitted through conformance, with replay-based failover.
10. **Phase 3 outline.**
    1. A toy RoslynHop using the straight-through estimator.
    2. Sandboxed Roslyn facts.
    3. An evaluation harness with kill criteria.
    4. Ablations at two model sizes.
    5. The C#/IL environment.
    The RoslynIn and RoslynOut arenas are already reserved.
