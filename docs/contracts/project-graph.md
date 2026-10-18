# ProjectGraph contract

Owner: S0 (S0a; the hygiene layering check and its fixtures are S0b's). Status: v1. Consumers: every workstream.

`docs/project-graph.json` is the only source of ProjectReferences and PackageReferences. Every project, its kind, its owner and its edges are listed there, and nowhere else.

## The graph

- **Schema** `lrm.project-graph/1`, with these sections:
  - `kinds`: `lib`, `app`, `test` and `tool`.
  - `packages`: the packages a project may name.
  - `groups`: named sets of projects that rules can refer to.
  - `rules`: the layering rules.
  - `projects`: one entry per project, with `name`, `path`, `kind`, `owner`, `refs`, and optional `buildRefs` and `packages`.
- **Where projects live:**
  - A `lib` lives under `src/` or `tests/`, an `app` under `apps/`, a `test` under `tests/`, and a `tool` under `src/` or `tools/`.
  - The file name is `<name>.fsproj`, and paths use forward slashes.
  - Every fsproj in the repo is registered, except those under `spikes/**`.
- **Generated files.** `tools/ci/gen-project-graph.ps1` validates the graph and writes three files. The output is byte-deterministic (UTF-8 without BOM, LF).
  - `build/ProjectKinds.g.props`: identity, kind and owner, imported by `Directory.Build.props`.
  - `build/ProjectGraph.g.props`: the reference items, imported by `Directory.Build.targets`.
  - `Lrm.slnx`: every registered project and nothing else.

## Layering rules

These are checked on the transitive closure of `refs` and `buildRefs`:

- Core references nothing.
- Models.Qwen and Cluster never reach a GPU project (`Lrm.Gpu.*`, `Lrm.Shaders`, `Lrm.Shaders.Build`).
- Gpu.Exec never reaches a concrete backend (Vulkan, D3D12, CpuDevice) or an interop project.
- Only apps and tests wire backends. A lib or tool never reaches one.
- An interop project is used only by its own backend, by apps and by tests.
- `Lrm.Shaders -> Lrm.Shaders.Build` is the only build-only edge. It is generated with `ReferenceOutputAssembly=false`.
- Nothing references an app, a test exe or a tool, except through that build-only edge.
- The graph has no cycles.
- `spikes/**` is never in the graph, the solution or CI.

## Project files

- **What an fsproj may hold.** An fsproj is `<Project Sdk="Microsoft.NET.Sdk">` holding ItemGroups of `Compile`, `EmbeddedResource`, `None` and `Content` items, and nothing else. An ItemGroup may carry only `Condition` and `Label`.
- **Per-project build logic** goes in `<Name>.lrm.targets` next to the fsproj. Examples are R4a's incremental shader target and extra properties. The project's owner owns the file, and `Directory.Build.targets` imports it. It may add targets, properties and file items, but never references.
- **Shared settings** come from `Directory.Build.props`:
  - `net10.0` and `x64`;
  - `UseAppHost=false`;
  - warnings are errors;
  - `Deterministic`;
  - `InvariantGlobalization=false`;
  - output under `artifacts/<bin|obj>/<project>/<config>/`.
- **Reproducible builds.** With `CI=true` the build sets `ContinuousIntegrationBuild`, which maps source and obj paths to `/_/`.
  - One commit then builds to identical binaries in every clone whose path has the same length.
  - F# 10.0.402 sizes the PE CodeView entry from the full PDB path, so clones at other path lengths differ only in that zero padding. IL, metadata, resources and PDBs are identical.
  - `-p:LrmReproducible=true` also turns PDBs off and gives byte-identical binaries at any path, for example to compare builds across nodes.
- **Running apps.** Apps run as `dotnet <dll>`, x64 only. Every exe calls `Lrm.Core.Process.init ()` first.

## Packages

- **Versions.** `Directory.Packages.props` pins every version through central package management. Every package was approved under AP-1: Expecto 11.1.1, Expecto.FsCheck 11.1.1 and FsCheck 3.4.0. FSharp.Core stays at the SDK default (`$(FSCorePackageVersion)`), which `Directory.Build.targets` adds to every project.
- **Lock files.** NuGet lock files in `build/locks/<project>.packages.lock.json` pin the whole closure, transitive packages included.
  - As of S0a, the only transitive package is Mono.Cecil 0.11.6, a dependency of Expecto 11.1.1.
  - A normal restore updates the lock files.
  - With `CI=true`, restore runs in locked mode and fails on a stale lock file.
- **Spikes** use neither central package management nor lock files.

## Enforcement

- **Build time.** `Directory.Build.targets`, target `LrmCheckProjectGraph`, which runs before restore and before compile, fails in these cases:
  - the project is not registered;
  - any ProjectReference, PackageReference, PackageDownload, Reference or FrameworkReference exists before the graph's items are added. Such an item came from the fsproj, `Directory.Build.props` or a `.lrm.targets` file, whatever metadata it carries. The SDK's implicit `Microsoft.NETCore.App` is the only exception;
  - a target adds a project or package reference later without the graph's `LrmGraph` marker.
- **CI.** These steps fail the build-test job:
  - `gen-project-graph.ps1 -Check` fails on a rule violation, an unregistered fsproj, an fsproj holding more than file items, a stray `.lrm.targets`, or generated files that are out of date;
  - the locked restore fails on stale lock files.
- **Hygiene.** The `hygiene` job (ubuntu-latest) runs the same validator as its layering check (`tools/ci/check-layering.ps1`), so a layering break fails both required checks. Its self-test (`tools/ci/test-hygiene.ps1`) edits a temporary copy of the graph inputs and proves that each rule fails as designed; the table below lists the cases.

### Vectors

Each edit below, applied to a copy of the v1 graph, must fail `gen-project-graph.ps1 -Check` with the message shown (`tools/ci/fixtures/layering/cases.json`). The unmodified copy must pass.

| Edit | Fails with |
| --- | --- |
| `src/Lrm.Tokenizer/Lrm.Tokenizer.fsproj` gains `<ProjectReference Include="../Lrm.Gpu.Core/Lrm.Gpu.Core.fsproj" />` | `<ProjectReference> is not allowed` |
| `src/Lrm.Tensor/Lrm.Tensor.fsproj` gains a `<PropertyGroup>` | `<PropertyGroup> is not allowed` |
| `Lrm.Models.Qwen` refs `Lrm.Gpu.Core` | `rule model-and-cluster-never-gpu: Lrm.Models.Qwen -> Lrm.Gpu.Core` |
| `Lrm.Cluster` refs `Lrm.Tensor`, and `Lrm.Tensor` refs `Lrm.Gpu.Core` | `rule model-and-cluster-never-gpu: Lrm.Cluster -> Lrm.Tensor -> Lrm.Gpu.Core` (transitive) |
| `Lrm.Core` refs `Lrm.Telemetry` | `rule core-is-leaf: Lrm.Core -> Lrm.Telemetry`, and a cycle |
| `Lrm.Gpu.Exec` refs `Lrm.Gpu.CpuDevice` | `rule exec-never-backend` |
| `Lrm.Serving` refs `Lrm.Gpu.Vulkan` | `rule only-apps-and-tests-wire-backends` |
| `Lrm.Gpu.Core` refs `Lrm.Gpu.D3D12.Interop` | `rule interop-only-under-its-backend` |
| `Lrm.Reports` refs `Lrm.Cli` | `rule no-refs-to-executables` |
| `Lrm.Gpu.Core` buildRefs `Lrm.Shaders.Build` | `rule build-only-edges: … is not an allowed build-only edge` |
| `Lrm.Tensor` refs `Lrm.Tensor.CpuReference` | `cycle: … Lrm.Tensor -> Lrm.Tensor.CpuReference -> Lrm.Tensor` |
| a new `src/Lrm.Rogue/Lrm.Rogue.fsproj` on disk | `is not registered in docs/project-graph.json` |
| a project at `spikes/vk-heap/VkHeapSpike.fsproj` in the graph | `rule no-spikes` |
| `Lrm.Tokenizer` names the package `Newtonsoft.Json` | `is not in the top-level packages list` |
| an allowed new edge, `Lrm.Tokenizer` refs `Lrm.Telemetry`, without regenerating | `build/ProjectGraph.g.props is out of date` |

## Changing the graph

1. Ask the integrator (S0) for the edge or the new project, with the reason.
2. S0 edits `docs/project-graph.json` and runs `pwsh tools/ci/gen-project-graph.ps1`.
3. S0 restores so that the lock files update, then commits the JSON, the generated files and `build/locks/` together.

A new package also needs an approval point and a `PackageVersion` entry.
