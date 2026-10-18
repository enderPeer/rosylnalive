# DataRootAndToolchain contract

Owner: S0 (S0a; the tokenizer and SDK entries, local-copy verification and the bootstrap self-test are S0b's). Status: v1. Consumers: every workstream that reads toolchains, weights, goldens, run logs or locks.

## Data root

- **What lives there.** `LRM_DATA` holds everything large or machine-local: `toolchain/`, `downloads/`, `models/`, `golden/`, `runs/`, `locks/`, `venv-golden/` and `spikes-staging/`. Nothing under it is ever committed, and the hygiene size check forbids the usual strays (`tools/ci/hygiene.json`: weights, `vk.xml`, SDK headers, packages, binaries).
- **Default.** It defaults to `C:/Users/end/dev/lrm-data`. The single source of that default is `dataRoot` in `tools/bootstrap/toolchain.json`, read in four places:
  - MSBuild: `build/Toolchain.g.props` resolves `$(LrmData)` from the `LrmData` property, else `LRM_DATA`, else `$(LrmDefaultData)`, the generated copy of `dataRoot`;
  - scripts: `tools/test.ps1` and `tools/bootstrap/bootstrap.ps1` read `toolchain.json`;
  - .NET code: Lrm.Core and Lrm.Testing carry the default as `[assembly: AssemblyMetadata("Lrm.DefaultDataRoot", …)]`. `Lrm.Testing.Env.dataRoot ()` and `Lrm.Core.LrmPaths.dataRoot ()` return `LRM_DATA` or that default.
- **Worktrees** use the same data root by absolute path. Nothing under it is per worktree.
- **On CI** the data root is `$RUNNER_TEMP/lrm-data`. Its `toolchain/` folder is cached and keyed on the hash of `toolchain.json`.

## Toolchain pins

`tools/bootstrap/toolchain.json` (schema `lrm.toolchain/1`) holds every pin: URL, size, hash and path under `LRM_DATA`. `tools/bootstrap/gen-toolchain-props.ps1` turns it into `build/Toolchain.g.props`, which provides these MSBuild properties:

- `$(LrmDxcExe)`, `$(LrmDxcVersion)` and `$(LrmDxcVersionOutput)`;
- `$(LrmVkXml)` and `$(LrmVkXmlVersion)`;
- `$(LrmVkHeaders)` and `$(LrmVkHeadersVersion)`;
- `$(LrmTokenizerJson)` and `$(LrmTokenizerSha256)`.

CI fails when the generated file is out of date. Builds and tools read these properties and never repeat a path, a version or a hash.

| Entry | Managed by | Kind | Approval | Install path under `LRM_DATA` |
| --- | --- | --- | --- | --- |
| `dxc` | bootstrap | `dxc-nupkg` | approved (AP-1); CI runners download it | `toolchain/dxc/1.9.2609.5` |
| `vkxml` | bootstrap | `file` | pending (AP-1) | `toolchain/vulkan/1.4.365/vk.xml` |
| `vkheaders` | bootstrap | `zip` | pending (AP-1) | `toolchain/vulkan-headers/1.4.365` |
| `tokenizer` | bootstrap | `file` | pending for CI runners (AP-1); the local copy is verified | `models/Qwen3-0.6B/tokenizer.json` |
| `vulkanSdk` | manual, optional | `installer` | not requested (AP-3, asked when V1 starts) | `toolchain/vulkan-sdk/1.4.363.0` |
| `goldenVenv` | manual | | | `venv-golden` |
| `model` | manual | | | `models/Qwen3-0.6B` |

## Bootstrap

`pwsh tools/bootstrap/bootstrap.ps1 [-AllowDownload] [-Only a,b] [-ComputePins]` handles every entry with `managedBy: bootstrap`, according to its `kind`:

| Kind | Entries | Verified on every run | Install |
| --- | --- | --- | --- |
| `dxc-nupkg` | dxc 1.9.2609.5 | SHA-256 of `dxc.exe`, `dxcompiler.dll`, `dxil.dll` and `dxv.exe`; the exact `--version` line; a valid Microsoft Authenticode signature | From the pinned nupkg (size and SHA-512 checked) in `downloads/`, extracted through a staging folder |
| `file` | vkxml 1.4.365, tokenizer | Size and SHA-256 | Downloaded to `.partial`, checked, moved into place |
| `zip` | vkheaders 1.4.365 | Every file against `.lrm-install.json`, which records the archive hash and each file's SHA-256; no extra files; any `files` pins | Archive (size and SHA-256 checked) in `downloads/`; the `extract` patterns, without the top folder, are extracted through a staging folder |

- **Approval gates every download.** Only an entry whose `approval` says "approved" is installed. A pending or declined entry is never fetched, not even with `-AllowDownload`.
- **Pinned local copies are verified anyway.** A pending or declined `file` entry with `size` and `sha256` pinned is still verified when a copy is already at its install path:
  - a copy that matches the pins prints `OK (local copy verified; download not approved)`;
  - a copy that does not match fails the run;
  - a missing copy is skipped with a message.

  This is how the tokenizer, which came with the verified weights, is checked on the development machine while its CI download awaits approval. Every other unapproved entry is skipped.
- **Downloads happen only with `-AllowDownload`.** CI uses it, and so does a one-time local install after an approval.
- **Missing pins.** An approved `file` or `zip` entry also needs its `url`, `size` and `sha256` pinned. `-ComputePins -AllowDownload` downloads it once into `downloads/pins/`, prints the values to pin, and installs nothing. The integrator checks the values against the upstream release and commits them.
- **Exit code.** It is 0 when every approved entry and every present pinned copy verifies. **A second run downloads nothing.**
- **Manual entries.** Bootstrap never touches `vulkanSdk`, `goldenVenv` or `model`.
  - The SDK entry records the optional AP-3 install: LunarG SDK 1.4.363.0, copied into its install folder with no registry, PATH or system change. Its validation layers are enabled per test process, never system-wide. Its URL, command line, size and hash are confirmed and pinned when AP-3 is asked.
- **Current state (S0b).**
  - DXC is approved (AP-1).
  - vk.xml, Vulkan-Headers and the tokenizer's CI download are pending AP-1.
  - Locally, the tokenizer is verified. On CI runners it is absent until the download is approved.
  - Once an entry is approved, the integrator flips its `approval` and, for vk.xml and the headers, pins the hash with `-ComputePins`.
- **Self-test.** `tools/bootstrap/test-bootstrap.ps1` runs bootstrap in child processes, against a fixture `toolchain.json` (`-ToolchainPath`), a temporary data root and a loopback HTTP server that counts requests. CI's build-test job runs it after bootstrap. It checks the following:
  - an approved file downloads once, and the second run makes no request;
  - a pending entry is never requested;
  - a corrupted pending copy fails;
  - a missing approved file fails without `-AllowDownload`;
  - a download that misses its pin fails and leaves no `.partial` behind.
- **DXC regression.** CI runs `tools/bootstrap/dxc-regression.ps1` after bootstrap. It checks the pinned DXC's version line, then compiles `tools/bootstrap/dxc-regression/*.hlsl` and checks the facts the plan relies on (ADR `docs/decisions/0001-shader-toolchain.md`):
  - validated DXIL `cs_6_6`;
  - SPIR-V with one descriptor variable at set 0 binding 0 and one push-constant block;
  - the 92-byte `rootsig_1_1` root signature.

## Vectors

The pins below are the known answers at v1. `toolchain.json` is authoritative, and a change to any of them is a change to this contract.

| Pin | Value |
| --- | --- |
| Default data root | `C:/Users/end/dev/lrm-data` |
| DXC version line | `dxcompiler.dll: 1.9(5512-01b62ad4)(1.9.2609.5) - 1.9.2609.5 (01b62ad47)` |
| DXC nupkg | 53,395,273 bytes; SHA-512 (base64) `gV/8G+1EOTpuiDaIBe53sB75lABIcZrXWqUGAvNSNg6qW8u+kP9CVM5nA0MbUkdP8N5CiUU/BEPTRXR2AL6r4w==` |
| `dxc.exe` SHA-256 | `aa7965b3763d9c710baaa5a28cedcc633449962a75d1391d161a85304a41b3e2` |
| Qwen3-0.6B revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `tokenizer.json` | 11,422,654 bytes; SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. TokenizerApi checks the same hash |
| `model.safetensors` SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
