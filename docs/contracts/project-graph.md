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
  - `build/ProjectKinds.g.props`: identity, kind and owner, imported by `Directory.Build.props`. It also gives each project its group membership: `LrmGroups` lists the groups whose patterns match the project's name, and `LrmReaches` the groups that its references reach, transitively. Both are `;`-wrapped and sorted (`;backends;gpu;interop;`) and absent when empty, so a root setting tests one group with `$(LrmGroups.Contains(';gpu;'))`.
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
- **Per-project build logic** goes in `<Name>.lrm.targets` next to the fsproj. Examples are R4a's incremental shader target and extra properties. The project's owner owns the file, and `Directory.Build.targets` imports it. It may add targets, properties and file items, but never references. Precisely, a `.lrm.targets` file:
  - imports nothing: no `<Import>`, `<ImportGroup>` or `<Sdk>`, and no `Sdk` or `TreatAsLocalProperty` on `<Project>`;
  - adds no reference item, as an element or as a task `Output ItemName`. That is any item whose name contains `Reference` or `Package`, `Analyzer`, and the guard's own `_LrmHand*` and `_LrmStray*` items;
  - sets no property, as an element or as a task `Output PropertyName`, that the root build files set (`Directory.Build.props`, `Directory.Build.targets`, `Directory.Packages.props` and `build/*.props`, for example `LrmIsSpike`, `TreatWarningsAsErrors` or `LrmDxcExe`). It also sets none that steers restore, the guard or the build's identity: `Restore*`, `NuGet*`, central package management, the `Directory*` and `Custom*` import switches, `CI`, `ContinuousIntegrationBuild`, and the Lrm graph and identity properties. Its own properties and items, such as R4a's `LrmShadersOut` and `LrmShaderInput`, are fine.
  - MSBuild names ignore case, and so do these rules.
- **Compile items** in an fsproj or a `.lrm.targets` file name only files inside the project's own directory: no `..` segment, no rooted path, and no property or item expansion other than `$(MSBuildThisFileDirectory)`, `$(MSBuildProjectDirectory)` and `$(IntermediateOutputPath)`. Code from another project comes only through a reference in the graph.
- **No other MSBuild extension files.** Outside `spikes/**`, `Directory.Build.props`, `Directory.Build.targets`, `Directory.Packages.props`, `global.json` and `NuGet.Config` exist only at the root, spelled exactly so. `Directory.Build.rsp`, `Directory.Solution.props`, `Directory.Solution.targets` and `<project>.user` files never exist. A nested `Directory.Build.*` would replace the root one for its subtree, so it could add references or switch off the guard and the locked restore without touching the graph.
- **Shared settings** come from `Directory.Build.props`:
  - `net10.0` and `x64`;
  - `UseAppHost=false`;
  - warnings are errors;
  - `Deterministic`;
  - `InvariantGlobalization=false`;
  - output under `artifacts/<bin|obj>/<project>/<config>/`;
  - **no F# tail calls in GPU code** (ADR R3-0001, D3D12ComIdiom rule 3): `Tailcalls=false` for every project in the `gpu`, `interop` or `backends` group, and for every test exe and app whose references reach one of those groups. A tail call frees the caller's frame before the callee runs, so a stack address passed in tail position is read after its frame is gone. The root owns `Tailcalls`; no `.lrm.targets` sets it. `Lrm.Testing`'s `TailCalls.scan` lists an assembly's `tail.` call sites, and the cpu tier checks the D3D12 interop, its tests, `Lrm.Gpu.Core`, `Lrm.Shaders` and `Lrm.Gpu.Core.Tests` with it (Lrm.Core, outside the groups, is the control that still has tail calls).
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
- **Hygiene.** The `hygiene` job (ubuntu-latest) runs the same validator as its layering check (`tools/ci/check-layering.ps1`), so a layering break fails both required checks. The layering check also scans the MSBuild extension files against the rules under Project files above. The build-time guard cannot do that alone. `Directory.Build.targets` decides spike status from the project's path (`spikes/**`), not from the `LrmIsSpike` property, so a property cannot switch the guard or the `.lrm.targets` import off; the layering scan still refuses a `.lrm.targets` that sets `LrmIsSpike`, which steers the lock-file settings in `Directory.Build.props`. Its self-test (`tools/ci/test-hygiene.ps1`) edits a temporary copy of the graph inputs and proves that each rule fails as designed; the tables below list the cases.

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

The MSBuild extension scan has its own cases. Each one writes the files shown into the copy (here `T` is `src/Lrm.Tokenizer/Lrm.Tokenizer.lrm.targets`) and must fail with the message shown.

| Files | Fails with |
| --- | --- |
| `T` sets `LrmIsSpike=true` and `RestoreLockedMode=false` and adds a `ProjectReference` to `Lrm.Gpu.D3D12` | `sets the property LrmIsSpike`, `sets the property RestoreLockedMode`, `<ProjectReference> is a reference item` |
| `T` sets `lrmisspike` and `NUGETLOCKFILEPATH` and adds a `packagereference` | the same three messages, in the file's case |
| `T` adds a ProjectReference through `CreateItem` `Output ItemName`, and sets `RestoreLockedMode` through `CreateProperty` | `a task Output adds the reference item ProjectReference`, `a task Output sets the property RestoreLockedMode` |
| `T` removes `_LrmHandProjectReference` and adds `Analyzer` and `PackageDownload` items in a target | `is a reference item`, three times |
| `T` has `<Import>` and `<ImportGroup>` | `<Import> is not allowed` |
| `T` sets `TreatWarningsAsErrors`, `LrmDxcExe` and `InvariantGlobalization` | `sets the property …, which the root build files own`, three times |
| `T` has `<Project Sdk="…" TreatAsLocalProperty="CI">` | `<Project Sdk=...> is not allowed`, `<Project TreatAsLocalProperty=...> is not allowed` |
| the fsproj has `Compile Include="$(LrmRepoRoot)src/Lrm.Gpu.D3D12/Device.fs"`, and `T` has `Compile Include="../Lrm.Gpu.D3D12/Device.fs"` | `reaches outside the project directory`, twice |
| `src/Lrm.Tokenizer/Directory.Build.targets` with a ProjectReference | `a nested Directory.Build.targets replaces the root one` |
| `tests/Directory.Build.props`, `src/Directory.Packages.props`, `tests/Lrm.Core.Tests/DIRECTORY.BUILD.PROPS`, `src/Lrm.Tokenizer/nuget.config`, `tools/global.json` | `a nested … replaces the root one`, five times |
| `src/Lrm.Tokenizer/Directory.Build.rsp`, `src/Lrm.Tokenizer/Lrm.Tokenizer.fsproj.user`, `Directory.Solution.targets` | `MSBuild reads … by itself` or `MSBuild imports <project>.user` |

A `.lrm.targets` that sets `Tailcalls`, to `true` or to `false`, fails with `sets the property Tailcalls, which the root build files own`. Two cases must pass: R4a's shader target with a project-owned compiler switch (`OtherFlags`), and build files under `spikes/**`.

## Changing the graph

1. Ask the integrator (S0) for the edge or the new project, with the reason.
2. S0 edits `docs/project-graph.json` and runs `pwsh tools/ci/gen-project-graph.ps1`.
3. S0 restores so that the lock files update, then commits the JSON, the generated files and `build/locks/` together.

A new package also needs an approval point and a `PackageVersion` entry.
