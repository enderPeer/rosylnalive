# ShaderBuildAndSourceConvention contract

Owner: R4. R4a owns contracts-v1; R4b finalizes the SPIR-V binding (`LRM_BUF` on SPIR-V) as contracts-v1.1, following R1's ADR. Status: v1, frozen at the contracts-v1 tag; the SPIR-V binding is marked provisional until v1.1. Consumers: the kernels (K1 to K3), the backends (D1, V1, R5), E1 and X1. Code: `src/Lrm.Shaders/` (`include/lrm.hlsli`, `rootsig/rootsig.hlsl`, `hlsl/**`, `Lrm.Shaders.lrm.targets`, `Shaders.fs`), `src/Lrm.Shaders.Build/` and `src/Lrm.Gpu.Core/ShaderSet.fs`, `SourceConvention.fs`. Tests: `tests/Lrm.Gpu.Core.Tests/ShaderTests.fs`; CI step `src/Lrm.Shaders.Build/check-shaders.ps1`.

Every kernel is written once in HLSL 2021 and compiled by **one pinned compiler** to both targets. The compiled set is embedded in `Lrm.Shaders` with a manifest, and its **ShaderSetHash** enters every GPU ExecutionClass.

## The compiler

- **Only DXC 1.9.2609.5** from `LRM_DATA`, `$(LrmDxcExe)` (DataRootAndToolchain). Never the Windows Kits dxc, which has no SPIR-V backend.
- Before compiling, the build checks that the file exists, that `dxc --version` prints exactly `$(LrmDxcVersionOutput)`, and that `dxc.exe`, `dxcompiler.dll`, `dxil.dll` and `dxv.exe` match their SHA-256 pins in `tools/bootstrap/toolchain.json`. **A missing or different DXC fails the build** with a pointer to `pwsh tools/bootstrap/bootstrap.ps1`.

| Target | Arguments (recorded in the manifest and hashed) |
| --- | --- |
| DXIL | `-T cs_6_6 -E main -HV 2021 -O3 -WX -denorm preserve` |
| SPIR-V | `-T cs_6_6 -E main -HV 2021 -O3 -WX -spirv -fspv-target-env=vulkan1.3` |
| Root signature | `-T rootsig_1_1 -E LRM_ROOTSIG` |

The build adds `-I include`, `-Fo <blob>` and the source path, relative to `src/Lrm.Shaders`; the blobs do not depend on the checkout path (verified with the pinned DXC). `-WX` makes every warning an error. No 16-bit types are enabled.

- **fp32 denormals.** DXIL gets **`-denorm preserve`** (`fp32-denorm-mode=preserve`, shader model 6.2 and later), as on the CPU reference. SPIR-V gets **no `-denorm`**: it would declare `OpCapability DenormPreserve` and `OpExecutionMode main DenormPreserve 32`, which the Vulkan spec allows only on devices with `shaderDenormPreserveFloat32` (VUID-RuntimeSpirv-shaderDenormPreserveFloat32-06297), and the RTX 4080 reports `false` under driver 616.92. The SPIR-V blobs therefore declare no float-controls mode, and each backend measures its denormal mode with the probe (GpuBackendInterface); bit-exact claims involving denormals are class-dependent (KernelAbiAndCatalog). The cpu tier checks that every SPIR-V blob has no denorm capability and LocalSize as its only execution mode.

## The build (LrmCompileShaders)

`src/Lrm.Shaders/Lrm.Shaders.lrm.targets` defines the target, which runs before resources are named in every build of `Lrm.Shaders`. `Lrm.Shaders.Build` is built first through the build-only edge `Lrm.Shaders -> Lrm.Shaders.Build` (ProjectGraph). The tool's `compile` step:

1. verifies the pinned DXC (above);
2. loads the catalog and refuses stale generated files (KernelAbiAndCatalog);
3. requires every `.hlsl` under `hlsl/` to be a catalog kernel at its catalog path; catalog kernels without a source are listed as **missing**;
4. lints every kernel source and every helper header (`.hlsli` under `hlsl/`) against the convention below, and checks that each helper a kernel includes exists beside it and that no helper has the name of a file in `include/`;
5. compiles each kernel to DXIL and SPIR-V, in parallel, through a **content-keyed cache**: the key is `Canonical.hash "lrm.shader-cache/2"` over the DXC version, the target and its flags, and the path and bytes of every file in the kernel's **include closure**: the source, the helpers it includes, and every file under `include/` (subfolders included). The convention makes that list exact: a kernel includes only `lrm.hlsli` and helpers beside it, and helpers include nothing. A kernel whose key is unchanged is not recompiled;
6. checks every blob: DXIL is a DXBC container with a DXIL part, **signed** (non-zero container digest, written by the validator), with the catalog's numthreads in its PSV0 part; SPIR-V is a module with exactly one `LocalSize` equal to the catalog's threads;
7. compiles `rootsig/rootsig.hlsl` and fails unless `rootsig.bin` equals the canonical bytes;
8. writes the blobs, `rootsig.bin` and `shaders.manifest.json` to `$(IntermediateOutputPath)shaders/`, and removes blobs of kernels that no longer exist.

The target's inputs are the sources, the helpers, every file under `include/`, the root signature, `catalog.json`, `Kernels.g.fs`, `toolchain.json`, every assembly in the tool's folder (the tool, and `Lrm.Gpu.Core` with the lint, catalog and manifest code, whose implementation changes reference assemblies would otherwise hide), the DXC executable and two stamps, each rewritten only when its text changes:

- `sources.stamp` lists the input files, so **adding, deleting or renaming** a source or header reruns the compile step, which then prunes the blobs of kernels that are gone (a timestamp check alone never sees a deleted file);
- `dxc.stamp` holds the DXC path, its pinned version line, and the name and write time of every file beside `dxc.exe` (`dxc.exe`, `dxcompiler.dll`, `dxil.dll`, `dxv.exe`, the four SHA-256 pins).

Its output is the manifest. **A no-change rebuild skips the target, so dxc never starts**; pointing the build at another DXC, or replacing a DXC file in place, always reruns the compile step, which then refuses it. So an incremental build embeds the same set as a clean build of the same tree (CI checks this after editing headers and after adding and deleting a file). The outputs are embedded as resources named `lrm.shaders/<path>`: `shaders.manifest.json`, `rootsig.bin`, `dxil/<kernel>.dxil` and `spirv/<kernel>.spv`.

`Lrm.Shaders.Shaders.embedded ()` reads the set once. Loading (`ShaderSet.ofManifest`) verifies every blob's SHA-256 against the manifest, that the root signature is canonical, and that both ShaderSetHash values reproduce; any mismatch raises `GpuContractException`.

## The manifest (schema `lrm.shaders/1`)

```json
{
  "schema": "lrm.shaders/1",
  "dxcVersion": "dxcompiler.dll: 1.9(5512-01b62ad4)(1.9.2609.5) - 1.9.2609.5 (01b62ad47)",
  "catalogHash": "<64 hex>",
  "rootSignature": { "resource": "rootsig.bin", "bytes": 92, "sha256": "<64 hex>" },
  "targets": {
    "dxil":  { "shaderSetHash": "<64 hex>", "flags": [ ... ], "kernels": [ { "name", "resource", "bytes", "sha256", "threads": [x, y, z] } ] },
    "spirv": { ... }
  },
  "missing": [ "<catalog kernels without a source, in catalog order>" ]
}
```

It is deterministic: kernels sorted by name, no timestamps, no paths, UTF-8 with LF line ends. Two clean builds of one commit give the same manifest, byte for byte, and so the same ShaderSetHash (CI checks this on every run).

## ShaderSetHash

There is **one ShaderSetHash per target**, because a backend runs one target and its ExecutionClass should change only when its own blobs change. For target *t*:

`ShaderSetHash(t) = Canonical.hash "lrm.shaderset/1"` over the canonical encoding of, in order:
1. the target name, `dxil` or `spirv` (string);
2. the DXC version line (string);
3. the catalog hash (32 raw bytes);
4. the target's DXC arguments (a sequence of strings);
5. for DXIL, `Some` SHA-256 of `rootsig.bin`; for SPIR-V, `None` (an option of 32 raw bytes);
6. the blobs of *t*, sorted by kernel name (ordinal), as a sequence of (kernel name string, blob SHA-256 as 32 raw bytes).

The GPU fill rule (GpuBackendInterface) puts the hash of the backend's target into its ExecutionClass. Missing kernels are not part of it; when K1 to K3 add kernels, both hashes change, as they should.

## The root signature

`rootsig/rootsig.hlsl` defines `LRM_ROOTSIG` as `RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=32, b0)`: a directly indexed heap and the 128-byte argument block as 32 root constants. The build compiles it with `-T rootsig_1_1 -E LRM_ROOTSIG`, and the result must equal **the canonical 92 bytes** (`RootSignature.canonical`), which D1 passes to `CreateRootSignature` for every PSO. Kernels do not embed a root signature.

| Bytes | Hex | Meaning |
| --- | --- | --- |
| 0..3 | `44584243` | `DXBC` |
| 4..19 | `14462d179f897b3ba160caa35fcb04a6` | container digest |
| 20..23 | `01000000` | version 1.0 |
| 24..27 | `5c000000` | 92 bytes |
| 28..31 | `01000000` | 1 part |
| 32..35 | `24000000` | part 0 at byte 36 |
| 36..43 | `52545330 30000000` | `RTS0`, 48 bytes |
| 44..55 | `02000000 01000000 18000000` | version 1.1; 1 parameter at byte 24 |
| 56..63 | `00000000 30000000` | 0 static samplers at byte 48 |
| 64..67 | `00040000` | flags `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` (0x400) |
| 68..79 | `01000000 00000000 24000000` | parameter 0: 32-bit constants, visibility all, payload at byte 36 |
| 80..91 | `00000000 00000000 20000000` | register b0, space 0, 32 values |

These are also the bytes the planning probe compiled (`LRM_DATA/spikes-staging/bindtest/rs.bin`), and `tools/bootstrap/dxc-regression.ps1` checks the 92-byte length on every CI run.

## lrm.hlsli

The one header every kernel includes first.

| Name | What it is |
| --- | --- |
| `LRM_ARGS(kernel)` | Declares `lrm_args` of type `lrm_args_<kernel>` (generated): a `ConstantBuffer` at `b0` on DXIL, a `[[vk::push_constant]]` block on SPIR-V |
| `LRM_NUMTHREADS(kernel)` | `[numthreads(...)]` from the catalog (`LRM_THREADS_<kernel>`) |
| `LRM_BUF(slot)` | The buffer in a heap slot: `ResourceDescriptorHeap[slot]` on DXIL; on SPIR-V `g_buf[slot]`, with `[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0)` |
| `lrm_buf(slot)` | The same as a function returning `RWByteAddressBuffer` |
| `lrm_tensor(token)` | The decoded table row of a token (`lrm_tensor_t`, the 16 words in row order) |
| `LRM_GROUP_LINEAR(gid)` | `gid.y * lrm_args.lrm_groups_x + gid.x`, the logical group of a linear kernel |
| `lrm_load_f32`, `lrm_store_f32`, `lrm_load4_f32`, `lrm_store4_f32`, `lrm_load_u32`, `lrm_store_u32`, `lrm_load_i32`, `lrm_store_i32`, `lrm_load_bf16`, `lrm_load_bf16_bits`, `lrm_load2_bf16`, `lrm_bf16_widen` | Element access by index from the tensor's start; bf16 through u32 words, with no 16-bit types |
| `LRM_TABLE_SLOT`, `LRM_ROW_BYTES`, `LRM_DTYPE_*`, `LRM_FLAG_*`, `LRM_F32_QUIET_NAN_BITS` | The contract constants |

- **The SPIR-V binding is provisional.** The explicit array at set 0, binding 0 was verified statically with the pinned DXC: it gives exactly one resource variable, with and without `NonUniformResourceIndex` (verified while planning; CI's DXC regression step checks the form without it on every run). R4b replaces that one block of `lrm.hlsli` with R1's recipe (VulkanHeapRecipe) as contracts-v1.1. Kernels do not change, because they use only `LRM_BUF`, `lrm_buf` and `lrm_tensor`.
- **No `NonUniformResourceIndex`.** Every heap index comes from a table row selected by a token in the arguments, so it is dynamically uniform.

## The source convention

One kernel per file, at its catalog path, with entry point `main`. A kernel may include **helper headers**: `.hlsli` files beside it, a family's own helpers (for example `hlsl/kernels/basic/rows.hlsli`). The build lints every kernel (`SourceConvention.lint`) and every helper under `hlsl/` (`SourceConvention.lintHelper`), on the text without comments. Both follow the **code rules**:

- the only directives are `#include` (kernels only) and `#define`; no `#pragma`, no `#undef`, no other directive, and **no conditional compilation** (`#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`), so both targets compile one body;
- no macro named `LRM_*` or `lrm_*`, and no `[[vk::...]]` attributes;
- none of these identifiers: `ResourceDescriptorHeap`, `SamplerDescriptorHeap`, `NonUniformResourceIndex`, `register`, `cbuffer`, `tbuffer`, `ConstantBuffer`, `ByteAddressBuffer`, `StructuredBuffer`, `RWStructuredBuffer`, `AppendStructuredBuffer`, `ConsumeStructuredBuffer`, `Buffer`, `RWBuffer`, `SamplerState`, `IncrementCounter`, `DecrementCounter`, `half`, `min16float`, `min10float`, `min16int`, `min12int`, `min16uint`, `float16_t`, `int16_t`, `uint16_t`, `SV_DispatchThreadID`, `numthreads`, `vk` (SPIR-V-only intrinsics such as `vk::RawBufferLoad`);
- no identifier starting with `Wave`, `Quad`, `Interlocked`, `Texture`, `RWTexture` or `__` (compiler macros such as `__spirv__`).

A **kernel** also:

- includes `"lrm.hlsli"` first and once; every other include is a helper beside the kernel, named by a plain file name that ends in `.hlsli` and does not start with `lrm` (never a path), and the build checks that the file exists;
- uses `LRM_ARGS(<kernel>)` and `LRM_NUMTHREADS(<kernel>)` each exactly once, for this kernel, and defines exactly one `void main(...)`.

A **helper** also:

- has a file name that ends in `.hlsli`, does not start with `lrm` and is not the name of a file in `include/` (DXC searches the including file's folder first, so a helper may never shadow `lrm.hlsli` or `lrm_args.g.hlsli`);
- **includes nothing**: the kernel includes `lrm.hlsli` first and then each helper it uses;
- has no `LRM_ARGS`, no `LRM_NUMTHREADS` and no `main`.

So a kernel's include closure is its source, its helpers and `include/`, which is exactly what the cache key hashes. A kernel reaches arguments only through `LRM_ARGS`, memory only through `LRM_BUF`, `lrm_buf` and `lrm_tensor`, and groups only through `SV_GroupID` (with `LRM_GROUP_LINEAR`), `SV_GroupThreadID` and `SV_GroupIndex`. DXC enforces the rest: groupshared at most 32 KiB (the validator for `cs_6_6`; SPIR-V compiles the same body, and R4b's inspector also sums its Workgroup variables) and warnings as errors.

## Vectors

**S1, ShaderSetHash of a synthetic set.** DxcVersion `d`; CatalogHash SHA-256("c"); DXIL arguments `["-O3"]`; rootsig bytes `rs`; DXIL blobs `a` = SHA-256("A") and `b` = SHA-256("B"); nothing for SPIR-V. Computed independently in Python; `ShaderTests.fs` checks it.
- DXIL: `cdd90f127b63cf16e1c99616e7327d93de08e6aa9816fd5aeea2d5dbfca75241`
- SPIR-V: `ef3e37506c8bcee5603764779da39299a62edf3216681cc775d67637cc930e97`

**The contracts-v1 build** (the five selftests; the nine K1 to K3 kernels are missing), recomputed independently in Python from the manifest:
- DXIL ShaderSetHash `b12cabbf300be63253a5e5bf1dbba6aa6fd34d0ef3a6f4303e712ae6f148f4be`
- SPIR-V ShaderSetHash `a1a15c99c3fb545f908085c72e58afeeca4563dee6e546eca0e359d4c06064c9` (built without `-denorm preserve`; the earlier draft's `1314aecf...` had it)
- `rootsig.bin` SHA-256 `a54491c90354dcc8625ff97a8828a57eb4ad512ff605af44011067b236cc6972`

These two hashes change with every kernel K1 to K3 add; they are recorded here as the contracts-v1 baseline, not as constants.

## Verification

- **cpu tier** (`ShaderTests.fs`): the canonical root signature (constant, embedded bytes and decoded fields); the embedded set loads and verifies; every catalog kernel is built or missing; every DXIL blob is a signed DXBC container with the catalog's numthreads; every SPIR-V blob has the catalog's LocalSize; the DXIL flags declare `-denorm preserve`, and the SPIR-V flags and blobs declare no denormal mode; the manifest regenerates byte for byte; vector S1; that the hash covers blobs, compiler, catalog and flags; that a tampered blob or hash is refused; that the selftests are lint-clean; and that the kernel and helper lints reject each forbidden construct (among them a wave op and a path include in a helper).
- **CI** (`check-shaders.ps1`, after the Release build):
  1. generated files are current;
  2. a clean rebuild into a fresh folder gives a byte-identical manifest (ShaderSetHash reproduces);
  3. a no-change rebuild skips `LrmCompileShadersCore`;
  4. adding a helper file reruns the target, and deleting it reruns the target again, after which the manifest equals a clean compile;
  5. on a scratch copy of the shader tree: editing a header in an `include/` subfolder, and editing a kernel's helper, each gives an incremental compile equal to a clean compile; deleting a kernel source prunes its blobs; and the build refuses a helper that uses a wave op or includes a path;
  6. a missing DXC and a DXC with another version line both fail.

  The DXC regression step keeps checking the bindless facts this design relies on.
- **R4b** (contracts-v1.1): the SPIR-V inspector checks one resource variable at set 0 binding 0, push-block offsets equal to the catalog, LocalSize, no float atomics or 16-bit types, and at most 32 KiB of Workgroup-storage variables in total.
