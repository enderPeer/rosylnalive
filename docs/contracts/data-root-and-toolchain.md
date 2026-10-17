# DataRootAndToolchain contract

Owner: S0 (S0a). Status: v1. Consumers: every workstream that reads toolchains, weights, goldens, run logs or locks.

## Data root

- **What lives there.** `LRM_DATA` holds everything large or machine-local: `toolchain/`, `downloads/`, `models/`, `golden/`, `runs/`, `locks/` and `venv-golden/`. Nothing under it is ever committed.
- **Default.** It defaults to `C:/Users/end/dev/lrm-data`. The single source of that default is `dataRoot` in `tools/bootstrap/toolchain.json`, read in four places:
  - MSBuild: `build/Toolchain.g.props` resolves `$(LrmData)` from the `LrmData` property, else `LRM_DATA`, else `$(LrmDefaultData)`, the generated copy of `dataRoot`;
  - scripts: `tools/test.ps1` and `tools/bootstrap/bootstrap.ps1` read `toolchain.json`;
  - .NET code: Lrm.Core and Lrm.Testing carry the default as `[assembly: AssemblyMetadata("Lrm.DefaultDataRoot", …)]`. `Lrm.Testing.Env.dataRoot ()` returns `LRM_DATA` or that default, and S1's `LrmPaths` should read the same metadata.
- **On CI** the data root is `$RUNNER_TEMP/lrm-data`. Its `toolchain/` folder is cached and keyed on the hash of `toolchain.json`.

## Toolchain pins

`tools/bootstrap/toolchain.json` (schema `lrm.toolchain/1`) holds every pin: URL, size, hash and path under `LRM_DATA`. `tools/bootstrap/gen-toolchain-props.ps1` turns it into `build/Toolchain.g.props`, which provides these MSBuild properties:

- `$(LrmDxcExe)`, `$(LrmDxcVersion)` and `$(LrmDxcVersionOutput)`;
- `$(LrmVkXml)` and `$(LrmVkXmlVersion)`;
- `$(LrmVkHeaders)` and `$(LrmVkHeadersVersion)`.

CI fails when the generated file is out of date. Builds and tools read these properties and never repeat a path or a version.

## Bootstrap

`pwsh tools/bootstrap/bootstrap.ps1 [-AllowDownload] [-Only a,b] [-ComputePins]` handles every entry with `managedBy: bootstrap`, according to its `kind`:

| Kind | Entry | Verified on every run | Install |
| --- | --- | --- | --- |
| `dxc-nupkg` | dxc 1.9.2609.5 | SHA-256 of `dxc.exe`, `dxcompiler.dll`, `dxil.dll` and `dxv.exe`; the exact `--version` line; a valid Microsoft Authenticode signature | From the pinned nupkg (size and SHA-512 checked) in `downloads/`, extracted through a staging folder |
| `file` | vkxml 1.4.365 | Size and SHA-256 | Downloaded to `.partial`, checked, moved into place |
| `zip` | vkheaders 1.4.365 | Every file against `.lrm-install.json`, which records the archive hash and each file's SHA-256; no extra files; any `files` pins | Archive (size and SHA-256 checked) in `downloads/`; the `extract` patterns, without the top folder, are extracted through a staging folder |

- **Approval gates everything.** Only an entry whose `approval` says "approved" is verified or installed. A pending or declined entry is skipped with a message and never fetched.
- **Downloads happen only with `-AllowDownload`.** CI uses it, and so does a one-time local install after an approval.
- **Missing pins.** An approved `file` or `zip` entry also needs its `url`, `size` and `sha256` pinned. `-ComputePins -AllowDownload` downloads it once into `downloads/pins/`, prints the values to pin, and installs nothing. The integrator checks the values against the upstream release and commits them.
- **Exit code.** It is 0 when every approved entry verifies. A second run downloads nothing.
- **Current state (S0a).** DXC is approved (AP-1). vk.xml and Vulkan-Headers are pending AP-1, so bootstrap skips them. Once they are approved, the integrator flips `approval`, pins the hash with `-ComputePins`, and R2a runs bootstrap.
- **DXC regression.** CI runs `tools/bootstrap/dxc-regression.ps1` after bootstrap. It checks the pinned DXC's version line, then compiles `tools/bootstrap/dxc-regression/*.hlsl` and checks the facts the plan relies on:
  - validated DXIL `cs_6_6`;
  - SPIR-V with one descriptor variable at set 0 binding 0 and one push-constant block;
  - the 92-byte `rootsig_1_1` root signature.
