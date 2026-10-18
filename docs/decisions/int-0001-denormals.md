# int-0001: fp32 denormals on the GPU backends

- **Status:** accepted (integrator, 2026-10-18, `int/wave1-fixes`).
- **Affects:** GpuBackendInterface (the denormal probe), KernelAbiAndCatalog (class-dependent denormals), ShaderBuildAndSourceConvention (the DXC flags), KernelBounds (O2), the ConformanceHarness (E1), D1, V1, K1 to K3.
- **Evidence:** the gpu-tier test "the fp32 denormal probe on the real scale_f32 blob is bit-identical twice" in `tests/Lrm.Gpu.D3D12.Interop.Tests/GpuTests.fs`; its run log renders to [`docs/reports/waves/wave-1/d3d12-denormals/kernels.md`](../reports/waves/wave-1/d3d12-denormals/kernels.md).

## Context

Two wave-1 results disagreed about fp32 denormals on the GPU:

- **R1's spike** (pitfall 10 of [R1-heap-recipe.md](R1-heap-recipe.md)) found denormals flushed to zero on both the D3D12 and the Vulkan path of the RTX 4080, and every NaN operand turned into `0x7FFFFFFF`. The spike's blobs were built with `-T cs_6_6 -HV 2021 -O3 -WX`, with no denormal mode.
- **R4a** builds every DXIL blob with `-denorm preserve`, which declares `fp32-denorm-mode=preserve` (shader model 6.2 and later), so that bit-exact kernels do not depend on a driver's flush choice. SPIR-V gets no `-denorm`, because the RTX 4080 reports `shaderDenormPreserveFloat32 = false` under driver 616.92 and a declared `DenormPreserve` mode would need it. GpuBackendInterface therefore makes the mode a measured property of the ExecutionClass: every shader backend runs the `scale_f32` probe after `LoadKernels`, classifies the bits as `preserve`, `flush` or `mixed`, and records `gpu.fp32_denormals`. D1 is to expect `preserve`.

Nothing had run the probe on hardware: there is no D3D12 backend before D1 (wave 2).

## What was run

The probe of GpuBackendInterface, through R3's interop and nothing else of LRM's GPU stack that is not yet built:

- the real `scale_f32` DXIL blob from the embedded shader set (DXIL ShaderSetHash `b12cabbf300be63253a5e5bf1dbba6aa6fd34d0ef3a6f4303e712ae6f148f4be`, flags `-T cs_6_6 -E main -HV 2021 -O3 -WX -denorm preserve`);
- the canonical `rootsig.bin`, and LRM's heap layout: slot 0 the 1 MiB tensor table with tokens 1 (x) and 2 (y) built by `TensorRow.create`, slot 1 a 64 KiB page;
- the ArgBlock {groups_x 1, x 1, y 2, n 7, a = 0.5} as the 32 root constants;
- `Fp32Denormals.probeInputs`, the output words poisoned with signalling-NaN patterns before each run, and a check that no poison is left;
- `Fp32Denormals.classify` on the 7 result words, run twice (`Determinism.testTwice`), under the D3D12 mutex of the gpu tier.

Device: NVIDIA GeForce RTX 4080 (`pci:10de:2704`), UMD 32.0.16.1692 (driver 616.92), Windows 11 10.0.26200. Runs `20261018-025830-95d7263e` (the probe alone) and `20261018-031937-1f674011` (the full gpu tier of this branch).

| Input | Preserve | Flush | Measured |
| --- | --- | --- | --- |
| `3f800000` (1.0, control) | `3f000000` | `3f000000` | `3f000000` |
| `01000000` (control) | `00800000` | `00800000` | `00800000` |
| `00800000` | `00400000` | `00000000` | `00400000` |
| `80800000` | `80400000` | `80000000` | `80400000` |
| `00000002` | `00000001` | `00000000` | `00000001` |
| `807ffffe` | `803fffff` | `80000000` | `803fffff` |
| `00400000` | `00200000` | `00000000` | `00200000` |

**The mode is `preserve`**, the same bits in both runs. Denormal inputs and denormal results are kept, exactly as on the CPU reference.

**Why R1 saw a flush.** With the pinned DXC, `scale_f32.hlsl` compiled with `-denorm preserve` carries `attributes #0 = { "fp32-denorm-mode"="preserve" }` in its DXIL; compiled without the flag, as R1's spike compiled its sources, it carries no denormal attribute, which leaves the choice to the driver. On the same device and driver the spike's blobs flushed. The flag, not the hardware, decides the D3D12 mode.

**A second finding: DXIL float arithmetic is `fast`.** Without `-Gis` (IEEE strictness), DXC marks the multiply `fmul fast`. Fast-math flags allow the compiler and driver to assume no NaN or infinity and to reassociate or contract. For a single multiply the result above is still exact, but a NaN test written as `v != v` is folded away: compiled with LRM's flags, a scratch kernel that stores `(v != v) ? 1 : 0` loses the load and stores a constant 0. On SPIR-V the same expression becomes `OpFOrdNotEqual`, which is false for a NaN. `isnan(v)` survives on both targets (`dx.op.isSpecialFloat` IsNaN, `OpIsNan`), and with `-Gis` DXIL keeps `fcmp uno`. NaN payloads were not re-measured here; R1's observation (`0x7FFFFFFF` for any NaN operand on both backends) stands.

## Decision

1. **D3D12 keeps `-denorm preserve`.** On the RTX 4080 with driver 616.92 the D3D12 class of the contracts-v1 shader set is `preserve`. For a D3D12 class that measures `preserve`, a bit-exact fp32 arithmetic kernel (`scale_f32`, `add`) equals CpuRef F32 as it is, denormals included.
2. **Vulkan stays measured, not assumed.** SPIR-V blobs keep no float-controls mode. V1 runs the probe after `LoadKernels` and records the mode; R1's spike suggests `flush` on this device. A `flush` class compares against CpuRef with every fp32 denormal input and result set to zero of the same sign (KernelAbiAndCatalog).
3. **No bit-identity across backends for fp32 arithmetic on denormals.** A D3D12 class and a Vulkan class may differ on exactly those elements. That is already the ExecutionClass rule: cross-class checks use tolerances, and a flushed denormal is an absolute error below 2^-126 per value.
4. **NaN tests use `isnan`.** Kernels never test for NaN with `v != v` or other self-comparisons; `isnan()` (or a bit test on `asuint`) is the only form that survives both targets. This matters first for `argmax_rows`, whose contract gives `0xFFFFFFFF` for a row holding a NaN.
5. **`-Gis` is not adopted here.** Adding it to the DXIL flags would remove the fast-math flags and make DXIL arithmetic IEEE-strict like SPIR-V, but it changes every DXIL blob and the DXIL ShaderSetHash, so it belongs in a ShaderBuildAndSourceConvention contract-change with evidence from K1 to K3's conformance runs, not in this integration fix.

## Consequences

- **D1** runs the probe after `LoadKernels` exactly as this test does, expects `preserve`, and reports any other mode as a finding; this test is the reference until D1's own selftest replaces it.
- **V1** runs the probe after `LoadKernels`, records the mode in its class, and applies the mode's comparison rule in its selftests.
- **E1** compares fp32 arithmetic kernels in the class's measured mode, emulates `preserve` in CpuDevice (which matches D3D12), and treats NaN payloads from GPU arithmetic as class-dependent (R1 saw `0x7FFFFFFF`).
- **O2 (KernelBounds)** states, for the BitExact fp32 arithmetic kernels (`scale_f32`, `add`), that bit-exact means "in the class's denormal mode" and how NaN results compare; relative bounds already cover a flushed denormal.
- **K1 to K3** use `isnan()` for NaN rules (`argmax_rows`), never `v != v`, and keep bit-exact claims of fp32 arithmetic class-dependent.
- **Reports.** The kernels template's `denormals` table shows each backend's measured mode next to its class.

## Alternatives

- **Flush everywhere** (no `-denorm` on DXIL either, and CpuRef modelling flush). Rejected: it gives up an exact match that the D3D12 hardware provides, and CpuRef would then need a second mode for every arithmetic op.
- **Require `shaderDenormPreserveFloat32` on Vulkan.** Rejected by R4a already: the RTX 4080 does not report it under 616.92, so the only GPU would be rejected.
