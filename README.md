# LRM — Large Roslyn Model

A transformer whose tokens, structure and coherence come from Roslyn. It reads and writes C# and IL as Roslyn objects, and its experts are calls to Roslyn's public API made from inside its layers. Written in F#, on its own GPU stack for Direct3D 12 and Vulkan (`VK_EXT_descriptor_heap`), designed for networked GPU clusters.

- [Design document](docs/LRM-design.md)
- [Status](docs/STATUS.md)

## Layout

| Path | Contents |
| --- | --- |
| `docs/LRM-design.md` | Architecture, backward pass, training curriculum, systems, evaluation, roadmap and review |
| `docs/project-graph.json` | Every project, its kind and owner, and every project and package reference |
| `docs/ownership.json` | Which workstream family owns which paths |
| `docs/contracts/` | The contract texts, one file per contract, each written by its owner; `README.md` indexes them and `CHANGELOG.md` records every version |
| `docs/decisions/` | Architecture decision records; `0000-adr-process.md` describes the format |
| `src/` | The libraries: `Lrm.Core`, tensors and the CPU reference, tokenizer, Qwen model, GPU core, shaders, D3D12 and Vulkan interop and backends, executor, cluster, serving, telemetry, reports, dashboard |
| `src/Lrm.Gpu.Vulkan.Interop` | F# P/Invoke binding to the Vulkan C API (query only so far) |
| `apps/Lrm.Cli` | The `lrm` command line |
| `apps/VkDeviceCheck` | Device check that rejects Vulkan devices without `VK_EXT_descriptor_heap` |
| `tests/` | Expecto test exes, one per library, plus conformance and Gate 1 suites; `Lrm.Testing` is the shared runner |
| `tools/` | Bootstrap, test runner, CI scripts (including the hygiene checks), code generators, golden-output tooling |
| `spikes/` | Throwaway spikes and the planning probes (`spikes/probes`); never built and not in CI |
| `build/` | Files generated from `docs/project-graph.json` and `tools/bootstrap/toolchain.json`, and the NuGet lock files in `build/locks/` |
| `roslyn-research-2026-10-17.zip` | Earlier Roslyn research: notes plus compiler-pipeline, tokenization and IL round-trip demos |

Most projects are still shells; each workstream of the development plan fills its own.

## Roadmap

1. **Phase 1:** run Qwen3-0.6B forward on our stack. Gate: logits and greedy tokens match the reference.
2. **Phase 2:** train it (backward pass, multi-node, fault tolerant). Gate: gradients match; a chaos run gives identical weights.
3. **Phase 3:** the LRM itself.

## Build and test

Requires the .NET 10 SDK (10.0.402, pinned in `global.json`) and PowerShell 7 on Windows x64. Large files (toolchain, weights, goldens, run logs) live outside the repo under `LRM_DATA`, by default `C:/Users/end/dev/lrm-data`.

```bash
pwsh tools/bootstrap/bootstrap.ps1        # verify the pinned DXC under LRM_DATA
dotnet build Lrm.slnx -c Release          # warnings are errors
pwsh tools/test.ps1 -Tier cpu             # also: gpu, model (local only), bench (informational)
dotnet tool restore && dotnet fantomas .  # format before committing
pwsh tools/ci/hygiene.ps1                 # ownership, layering, sizes, line endings, contract label
```

Build output goes to `artifacts/`. Apps run as `dotnet <dll>`, for example `dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll --help`.

## Device check

```bash
dotnet run --project apps/VkDeviceCheck -c Release
```
