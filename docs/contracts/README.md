# Contracts

A contract is the agreed interface between workstreams: types, encodings, file formats, rules and the known-answer vectors that pin them. Each contract has one text here, written in full by its owner (dev plan, section 5). The machine-readable parts live next to the code they govern, and the text says where.

- **How a contract changes:** `ownership-and-contract-change.md`. In short, contracts are drafts until the tag `contracts-v1`. After the tag, a contract file changes only through a PR labelled `contract-change`, which adds an entry to `CHANGELOG.md` and updates the vectors. The hygiene job enforces the label and the entry.
- **The record of every version** is `CHANGELOG.md`. Any family may append its entry there; only S0 rewrites lines.
- **Freeze tags.** `contracts-v1` covers the S1, R4a, O2 and C1 contracts at the end of wave 1. `contracts-v1.1` adds R4b's SPIR-V binding (ShaderBuildAndSourceConvention) after R1's ADR.

## Index

Status as of 2026-10-18. "v1" means written in full and in review or merged; "draft" means a text exists on an open draft PR; "planned" means not written yet. The integrator updates this table as PRs land.

| Contract | Owner | Text | Status |
| --- | --- | --- | --- |
| ProjectGraph | S0 | `project-graph.md` | v1 (PR #2; hygiene enforcement in PR #9) |
| TestTiers | S0 | `test-tiers.md` | v1 (PR #2) |
| DataRootAndToolchain | S0 | `data-root-and-toolchain.md` | v1 (PR #2; tokenizer and SDK entries in PR #9) |
| AgentPermissions | S0 | `agent-permissions.md` | v1 (PR #2; enforcement in PR #9) |
| OwnershipAndContractChange | S0 | `ownership-and-contract-change.md` | v1 (PR #9) |
| LrmCore | S1 | `lrm-core.md` | v1 (PR #3) |
| TelemetryFacade | S1 | `telemetry-facade.md` | v1 (PR #3) |
| ExecutionClass | S1 (type); R4a (GPU fill rule) | `execution-class.md` | v1 (PR #3) |
| RunLog | T1 | `run-log.md` | draft (PR #8) |
| ReportTemplates | T1 | `report-templates.md` | draft (PR #8) |
| HostTensorAndSafetensors | O2 | `host-tensor.md` | planned |
| CpuReferenceSemantics | O2 | `cpu-reference.md` | planned |
| ComparatorsAndGoldenReader | O2 | `comparators.md` | planned |
| KernelBounds | O2 | `kernel-bounds.md` | planned |
| ToleranceSpec | O2 (types); G2 (numbers) | `tolerance-spec.md` | planned |
| ReferenceOracles | O1 | `reference-oracles.md` | draft (PR #4) |
| GoldenSet | O1 (format); G2 (Qwen content) | `golden-set.md` | draft (PR #4) |
| ActivationNames | O1 | `activation-names.md` | draft (PR #4) |
| HeapLayoutAndTensorTable | R4 | `heap-and-tensor-table.md` | planned |
| KernelAbiAndCatalog | R4 | `kernel-abi.md` | planned |
| GpuBackendInterface | R4 | `gpu-backend.md` | planned |
| ShaderBuildAndSourceConvention | R4 (SPIR-V `LRM_BUF`: R4b) | `shader-build.md` | planned |
| BatchInvariance | R4 | `batch-invariance.md` | planned |
| VulkanHeapRecipe | R1 | `vulkan-heap-recipe.md` | draft (PR #5) |
| VulkanInteropApi | R2 | `vulkan-interop.md` | planned |
| D3D12ComIdiom | R3 | `d3d12-com-idiom.md` | draft (PR #6) |
| ClusterProtocol | C1 | `cluster.md` | draft (PR #7) |
| ConformanceHarness | E1 | `conformance-harness.md` | planned (wave 2) |
| TokenizerApi | M1 | `tokenizer.md` | planned (wave 2) |
| ForwardOpsAndQwenModel | M2 | `qwen-model.md` | planned (wave 2) |
| GpuExecutor | X1 | `gpu-executor.md` | planned (wave 3) |
| StagePipeline | C2 | `stage-pipeline.md` | planned (wave 3) |
| Gate1Evidence | I1, I2, I3 | `gate1-evidence.md` | planned (wave 4) |

## Writing a contract text

- **Heading:** `# <Name> contract`.
- **First line:** `Owner: …. Status: vN. Consumers: ….`, plus where the code and tests are.
- **Body:** the rules, as short statements a reviewer can check one by one.
- **Vectors:** a section with known-answer vectors wherever there is an encoding, a hash, a layout or a decision table. They are computed independently of the code under test (for example with `sha256sum`), and the tests check them.
- **Verification:** a section that names the tests and what they prove.
