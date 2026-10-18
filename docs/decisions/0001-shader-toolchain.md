# ADR 0001: Shader toolchain and bindless source convention

- **Status:** accepted, 2026-10-18. The SPIR-V binding form (decision 4) is provisional until R1's ADR (`docs/decisions/R1-*.md`) and R4b, which together close contracts-v1.1.
- **Owner:** S0.
- **Scope:** every GPU kernel, the shader build (R4a, R4b) and the backends (D1, V1).
- **Related:**
  - the contract ShaderBuildAndSourceConvention (`docs/contracts/shader-build.md`, R4);
  - the pins in `tools/bootstrap/toolchain.json`;
  - the CI regression `tools/bootstrap/dxc-regression.ps1`.

## Context

LRM's kernels run on Direct3D 12 and on Vulkan, through one resource heap written once at startup. Kernels receive 32-bit tensor tokens as root constants or push data, and look rows up in a tensor table in slot 0 (CLAUDE.md, fixed decisions). This calls for:
- one shader language and one compiler that emits both DXIL and SPIR-V;
- a source form whose SPIR-V matches the Vulkan set/binding to heap mapping path of `VK_EXT_descriptor_heap`, which the plan starts with;
- facts that can be re-checked on every CI run, because compiler behaviour around descriptor heaps is easy to get subtly wrong.

## Decision

1. **Language.** HLSL 2021 (`-HV 2021`), compute shaders at Shader Model 6.6 (`cs_6_6`).
2. **Compiler.** Only DXC 1.9.2609.5 from the NuGet package `Microsoft.Direct3D.DXC`, installed under `LRM_DATA` by bootstrap and pinned in `tools/bootstrap/toolchain.json`. Bootstrap verifies it on every run by:
   - the SHA-256 of its four binaries;
   - a valid Microsoft Authenticode signature;
   - the exact `--version` line.

   Never use the Windows Kits `dxc`, which has no SPIR-V backend. Phase 1 uses neither Slang nor glslang.
3. **Targets:**
   - **DXIL:** `-T cs_6_6 -E main -HV 2021 -O3 -WX`. It gives a validated, hashed DXBC container. `-O3` is DXC's default, and an explicit `-O3` gives identical bytes.
   - **SPIR-V:** the same flags plus `-spirv -fspv-target-env=vulkan1.3`.
   - **Root signature:** `-T rootsig_1_1 -E LRM_ROOTSIG` over `RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=32, b0)`. That is a directly indexed heap and the 128-byte argument block, and it compiles to 92 bytes.
4. **Binding source convention:**
   - `LRM_BUF(i)` is `ResourceDescriptorHeap[i]` on DXIL. On SPIR-V it is `g_buf[i]` over one explicit runtime array, `[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);`.
   - Arguments are one `ConstantBuffer<Args>` at `b0`, marked `[[vk::push_constant]]` on SPIR-V.
   - Tokens index 64-byte rows of the tensor table at `LRM_BUF(0)`.
5. **No `NonUniformResourceIndex`.** Every heap index in LRM is dynamically uniform. It comes from root constants or push data, or from a table row loaded at an address that is itself uniform. So no kernel wraps an index in `NonUniformResourceIndex`. A kernel that truly needs divergent resource indices needs a new ADR first.
6. **The exact verified source** is below. It is byte-identical to `tools/bootstrap/dxc-regression/bindless.hlsl` (SHA-256 `02081356c500a70a3d5215dfc2bfa10f83ea8a4b773762a16e0b92798ad7e336`, LF line endings), which CI compiles on every build-test run.

```hlsl
// DXC regression kernel, compiled by tools/bootstrap/dxc-regression.ps1 on every CI run.
// It uses the bindless form of the dev plan: D3D12 indexes ResourceDescriptorHeap directly, and SPIR-V uses one
// explicit runtime array at set 0, binding 0, without NonUniformResourceIndex. Arguments arrive as push data or root
// constants; tensor tokens index a 64-byte row table in slot 0.
struct Args
{
    uint srcToken;
    uint dstToken;
    uint n;
    float scale;
};

#ifdef __spirv__
[[vk::push_constant]] ConstantBuffer<Args> args : register(b0);
[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);
#define LRM_BUF(i) g_buf[(i)]
#else
ConstantBuffer<Args> args : register(b0);
#define LRM_BUF(i) ResourceDescriptorHeap[(i)]
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    RWByteAddressBuffer table = LRM_BUF(0);
    uint srcSlot = table.Load(args.srcToken * 64);
    uint srcOffset = table.Load(args.srcToken * 64 + 4);
    uint dstSlot = table.Load(args.dstToken * 64);
    uint dstOffset = table.Load(args.dstToken * 64 + 4);
    if (id.x >= args.n)
        return;
    RWByteAddressBuffer src = LRM_BUF(srcSlot);
    RWByteAddressBuffer dst = LRM_BUF(dstSlot);
    dst.Store(dstOffset + id.x * 4, asuint(asfloat(src.Load(srcOffset + id.x * 4)) * args.scale));
}
```

## Evidence

Verified on 2026-10-18 on the development machine (Windows 11, 32 threads) with the pinned DXC, whose `--version` printed `dxcompiler.dll: 1.9(5512-01b62ad4)(1.9.2609.5) - 1.9.2609.5 (01b62ad47)`. Compiling needs no GPU.

| Command (source above, `bindless.hlsl`) | Result |
| --- | --- |
| `dxc -T cs_6_6 -E main -HV 2021 -WX bindless.hlsl -Fo bindless.dxil` | 3,868 bytes, SHA-256 `0d097221e81cbaa8622520e6ef5988aa5e3cd89068a5e866588329ff5c9ed552`; `dxv bindless.dxil` prints `Validation succeeded`; a DXBC container with a non-zero hash; every heap handle comes from `createHandleFromHeap(…, nonUniformIndex=false)` |
| `dxc … -spirv -fspv-target-env=vulkan1.3 bindless.hlsl -Fo bindless.spv -Fc bindless.spvasm` | 1,884 bytes, SHA-256 `13bbe253b67dd8ebcda3c094e0c02fe17126c9fd6ed6b6430a0c0b979ac2edf4` |
| `dxc -T rootsig_1_1 -E LRM_ROOTSIG rootsig.hlsl -Fo rootsig.bin` | 92 bytes, SHA-256 `a54491c90354dcc8625ff97a8828a57eb4ad512ff605af44011067b236cc6972`, byte-identical to the planning probe's output |
| each command a second time, and the DXIL and SPIR-V again with an explicit `-O3` | identical bytes |

**The SPIR-V of the source above** declares exactly one resource variable, at set 0 binding 0. Its arguments are one push-constant block, and it has no `NonUniform` decoration and no `ShaderNonUniform` capability:

```
OpCapability Shader
OpCapability RuntimeDescriptorArray
OpExtension "SPV_EXT_descriptor_indexing"
OpDecorate %g_buf DescriptorSet 0
OpDecorate %g_buf Binding 0
%args = OpVariable %_ptr_PushConstant_type_ConstantBuffer_Args PushConstant
%g_buf = OpVariable %_ptr_StorageBuffer__runtimearr_type_RWByteAddressBuffer StorageBuffer
```

**The same form with `NonUniformResourceIndex`** is the planning probe `spikes/probes/bindtest/explicit.hlsl`. It also gives one variable `%g_buf` at set 0 binding 0, so the layout does not depend on it. It adds `OpCapability ShaderNonUniform` and a `NonUniform` decoration per indexed access (two here). On the descriptor-set path, that requires the non-uniform storage-buffer indexing feature. Dropping it costs nothing, because the indices are uniform.

**The emulated heap**, `ResourceDescriptorHeap[…]` compiled with `-fvk-bind-resource-heap 0 0`, is the probe `spikes/probes/bindtest/emul.hlsl`. It gives one aliased variable per access: `%ResourceDescriptorHeap`, `%ResourceDescriptorHeap_0` and `%ResourceDescriptorHeap_1`, all at set 0 binding 0. That breaks R4b's check of exactly one resource variable at set 0 binding 0, so this form is not used.

**Re-checked on every CI run.** `tools/bootstrap/dxc-regression.ps1` (build-test job) repeats the checks: the version line, validated DXIL, exactly one descriptor variable `g_buf` at set 0 binding 0, one push-constant block, and the 92-byte root signature.

## Consequences

- R4a's `lrm.hlsli` and the shader build follow decisions 1 to 5. The SPIR-V `LRM_BUF` is marked provisional there.
- R1's spike checks at run time that the Vulkan heap mapping (set 0, binding 0) serves this exact variable form. R1's ADR records the mapping, and R4b then finalizes the SPIR-V half; the R4b inspector enforces one variable at set 0 binding 0. If R1 shows that another form is needed, an R1 or R4 ADR supersedes decision 4, and this ADR's status line points to it.
- Kernel authors keep heap indices uniform (decision 5). Conformance and the SPIR-V inspector catch a `NonUniform` decoration.
- Changing the compiler means a new pin in `toolchain.json`, a new `versionOutput`, a re-run of this evidence and a new ADR that supersedes this one.

## Alternatives

- **Slang:** deferred by plan decision 4. It adds a second compiler and a second source of truth for Phase 1.
- **The Windows Kits DXC:** it has no SPIR-V backend.
- **GLSL with glslang** for Vulkan: it needs two kernel sources per kernel, which breaks "one kernel per file".
- **`NonUniformResourceIndex` everywhere:** it gives the same variable layout, but adds a capability and a device feature dependency that LRM's uniform indices never need.
- **`-fvk-bind-resource-heap`:** it gives one aliased variable per access.
