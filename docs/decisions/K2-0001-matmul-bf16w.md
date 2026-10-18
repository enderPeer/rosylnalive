# ADR K2-0001: matmul_bf16w is one tiled kernel whose every output is a sequential fused chain over k

- **Status:** proposed (K2, 2026-10-18), with PR #25.
- **Owner:** K2.
- **Scope:** `src/Lrm.Shaders/hlsl/kernels/matmul/matmul_bf16w.hlsl` and `tests/Lrm.Kernels.Conformance/Cases/matmul/`. It binds X1 (how it calls the kernel: lm_head chunking, the decode cost), I1 (the kernels report and the bench), E1 (GPU-only cases in its model tests) and T1 (the throughput records).

## Context

The frozen catalog entry (KernelAbiAndCatalog, `catalog.json` id 12) fixes `matmul_bf16w`: x f32 [m, k], w bf16 [n, k] (PyTorch Linear layout, as Qwen stores it), y f32 [m, n]; 16 x 16 threads; grid (ceildiv(n, 64), ceildiv(m, 64), 1), not linear. One kernel serves decode (m = 1), prefill (m up to 4,000) and lm_head (n = 151,936, k = 1,024). The rules around it:

- **BatchInvariance:** every output accumulates p = 0..k-1 sequentially in increasing order with `mad`, whatever m is, so a row alone (m = 1) equals the same row at m = 37 and m = 512 bit for bit.
- **KernelBounds:** relative to CpuRef F32's own error; CpuRef F32 is `Rows.dotBf16`, one `FusedMultiplyAdd` per step from +0.
- **Kernel rules:** no atomics, wave ops or 16-bit types; groupshared at most 32 KiB; one body for DXIL and SPIR-V; every dispatch under 100 ms (the faster of two runs).
- **DXIL is fast-math** (ADR int-0001): a separate multiply and add could be contracted or reassociated, so the accumulation is written with `mad` only.
- **Phase 1** puts correctness before speed, with no cooperative matrix.

## Decision

1. **Tiling.** Group (gid.x, gid.y) owns the 64 x 64 output tile at rows 64·gid.y and columns 64·gid.x. Thread (tx, ty) owns rows i0 + ty + 16a and columns j0 + tx + 16b for a, b in 0..3, a 4 x 4 register tile. The strided columns make each store of a half-warp 16 consecutive floats.
2. **The chain.** Every output starts from +0 and takes `acc = mad(x[i, p], widen(w[j, p]), acc)` for p = 0, 1, ..., k-1, one `mad` per p in increasing order. The loop stops at k exactly, so padding is never accumulated, and k = 0 gives +0. There is no split-K and no reduction across threads or groups. Which thread computes an output depends on i mod 64 and j mod 64 only, never on m or n.
3. **k slabs of 32 in groupshared memory.** Per slab, the group stages x (64 rows x 32) and w (64 columns x 32, widened exactly to f32) as float4 runs of 4 consecutive p, `float4 xs[64][9]` and `ws[64][9]`. The ninth run is padding so the 8 threads of a 128-bit phase read distinct bank groups. That is 18,432 bytes. The next slab is loaded into registers while the current one is used, with two barriers per slab.
4. **Rows past m are neither read nor computed.** A thread computes only its valid row slots (`live`), and staging loads skip rows past m and columns past n. A decode tile (m = 1) does 1/64 of a full tile's arithmetic. Skipping changes no step of a valid output.
5. **Loads.**
   - x is read with `Load4` at 4-byte alignment, which is legal for raw buffers and lets odd k work.
   - w is read with `Load4` as 8 bf16 per thread when k is even and the slab is full, and element by element (`lrm_load_bf16`) otherwise. For odd k, w rows start in the middle of a u32 word.
   - Elements past k are staged as 0 and never read.
   - Loads only move exact values, so the load path cannot change a result.
6. **No second code path in Phase 1.** There is no gemv variant and no m-dependent path. The plan admits a gemv variant only if it is bit-identical (see Alternatives).
7. **Cases.**
   - Every Qwen3-0.6B projection at m = 1, 37, 512 and 4,000, and lm_head at m = 1 and 64. All cases of one shape share one weight and the first rows of one activation matrix, so row r is the same inputs in every m.
   - 15 edge shapes: tile edges in m and n, every k remainder against the slab and the run, odd k, k = 0, and the widest grid without folding, n = 64 x 65,535.
   - m = 512 and 4,000 and both lm_head cases are GPU-only. They are not built in the cpu tier at all (LRM_TEST_TIER), because E1's model tests run every discovered case on CpuDevice, GPU-only cases included (request to E1).
8. **The bench** (`MatmulBench.fs`, run by `bench.fsx` until a tier lists it) measures throughput at the Qwen shapes. It also checks rows across m = 1, 37 and 512 calls against each other and against CpuRef F32, and runs two arithmetic probes, recording `gpu.matmul.*`. `report.fsx` renders `templates/matmul.md` with T1's engine and charts until the kernels template charts these records.

## Evidence

All runs: NVIDIA GeForce RTX 4080 (`pci:10de:2704`), driver 616.92 (UMD 32.0.16.1692), Windows 11 10.0.26200, commit `558815b`, DXIL ShaderSetHash `2d513d8126713397f063ce95670a82760ac55c0e77ded350a7208a369c3c951e`, SPIR-V ShaderSetHash `3ed88c0a917b8c3cee01b1b7cd3f693058d97a806774e762be557141781c539a`. Reports: `tests/Lrm.Kernels.Conformance/Cases/matmul/reports/`.

- **Blobs.** DXC 1.9.2609.5 compiles both targets with `-WX`, and the SPIR-V blob passes all 16 inspector rules. The DXIL has 80 `dx.op.tertiary` FMad (opcode 46) and no `fmul` or `fadd`; the SPIR-V has 80 GLSL.std.450 `Fma`. Groupshared is 2 x 2,304 floats = 18,432 bytes, and there are no local arrays (`alloca`).
- **Conformance on D3D12** (gpu tier, run `20261018-132406-e1c048e8`):
  - all 41 matmul cases pass (37 K2 and 4 E1), each twice bit-identical, with all 91 batch splits matching;
  - on every case the measured relL2 equals CpuRef F32's own reference error, which indicates the same bits as CpuRef F32;
  - the slowest case is 26.0 ms, the faster of its two runs (down_proj at m = 4,000).
- **Conformance on CpuDevice** (cpu tier, run `20261018-131829-8cc803cc`): the 25 K2 cases that CI runs and E1's 4 pass, bit-exact to CpuRef F32. The conformance exe's cpu tier takes 8.4 s.
- **Across calls** (bench run `20261018-132542-33830713`). The outputs hold a signalling-NaN poison before the dispatches. For all six shapes, 9 rows (0, 1, 15, 16, 36, 63, 64, 100 and 511) computed alone equal the same rows of the m = 512 call and, below 37, of the m = 37 call. All are equal to CpuRef F32 bit for bit: 0 differing elements of 9 x (2,048 + 1,024 + 1,024 + 3,072 + 1,024 + 151,936).
- **Arithmetic probes** (same run, informational):

  | Probe | x | w | Result | Meaning |
  | --- | --- | --- | --- | --- |
  | `fused` | `bf810001 3f800001` | `3f80 3f81` | `30800000` (2^-30) | one fused multiply-add per step; a rounded product and a sum give `00000000` |
  | `order` | `3f800000 4b800000 cb800000` | `3f80 3f80 3f80` | `00000000` | p in increasing order; decreasing p gives `3f800000` |

  The fp32 denormal probe measures `preserve`.
- **Throughput, warm** (same run): the faster of 5 back-to-back dispatches after a warm-up.

  | Shape (n x k) | m = 1 | m = 37 | m = 512 | m = 4,000 |
  | --- | --- | --- | --- | --- |
  | q_proj (2,048 x 1,024) | 38.9 µs, 108 GB/s | 64.5 µs | 184 µs, 11.7 TFLOP/s | 1.24 ms, 13.5 TFLOP/s |
  | k_proj, v_proj (1,024 x 1,024) | 38.9 µs, 54 GB/s | 64.5 µs | 96.3 µs, 11.2 TFLOP/s | 0.64 ms, 13.1 TFLOP/s |
  | o_proj (1,024 x 2,048) | 76.8 µs, 55 GB/s | 127 µs | 188 µs, 11.4 TFLOP/s | 1.27 ms, 13.2 TFLOP/s |
  | gate_proj, up_proj (3,072 x 1,024) | 38.9 µs, 162 GB/s | 64.5 µs | 273 µs, 11.8 TFLOP/s | 1.85 ms, 13.6 TFLOP/s |
  | down_proj (1,024 x 3,072) | 115 µs, 55 GB/s | 191 µs | 280 µs, 11.5 TFLOP/s | 1.92 ms, 13.1 TFLOP/s |
  | lm_head (151,936 x 1,024) | 719 µs, 434 GB/s | | 10.6 ms, 15.0 TFLOP/s | m = 256: 5.28 ms; m = 1,024: 21.3 ms, 15.0 TFLOP/s |

  - A decode token's 196 projections take 28 x 386 µs = 10.8 ms, plus 0.72 ms for lm_head.
  - Decode is bound by latency, not bandwidth. At m = 1 the grid has n / 64 groups (16 for n = 1,024, on 76 SMs), and each walks k / 32 slabs one after another, so the time is about k / 32 slab round trips.
- **Cold GPU.** Each conformance case runs after seconds of host work, while the GPU idles in a low power state; the bench runs it warm.
  - In a scratch run of q_proj at m = 4,000, a dispatch after 0 to 1 s idle took 1.25 ms. After 3 s idle it took 10.8 ms, and so did the next six back-to-back dispatches.
  - The conformance times are therefore 8 to 15 times the warm ones (down_proj at m = 4,000: 26.0 ms against 1.92 ms).
- **The checks catch a batch-variant kernel.** A scratch mutation flipped the last mantissa bit of every output when m > 1, a 1-ulp change. It made the cross-call check fail on all six shapes, with 14 x n differences each (9 rows against m = 512 and 5 against m = 37); CpuRef still agreed with the untouched m = 1 rows. It also made all six m = 37 conformance cases fail through their batch splits only, inside the relative bound. Reverted; the ShaderSetHash returned to `2d513d81…`.

## Consequences

- **X1:**
  - Call the kernel with the actual m of each dispatch. Prefill chunks up to 4,000 rows are within the budget for every projection: at most 1.92 ms warm, and 26.0 ms (a 3.8 x margin) when the GPU starts from idle clocks.
  - Chunk lm_head to at most 256 rows per dispatch, and compute only the logits a step needs. That is 5.3 ms warm, and about 40 to 80 ms if the GPU starts from idle. The 1 GiB page limit alone caps lm_head's y at 1,766 rows.
  - Rows are batch-invariant, so chunking never changes a bit.
- **I1:** the bench's throughput sweep belongs in the bench tier and its cross-call check in the gpu tier. Until the conformance exe lists `MatmulBench.bench ()` and `MatmulBench.gpu ()`, run `dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/bench.fsx [bench|gpu]`.
- **E1:** the model tests should skip `GpuOnly` cases as the CpuDevice suite does; then the LRM_TEST_TIER gate in `MatmulCases.fs` can go.
- **T1:** the kernels template should chart `gpu.matmul.throughput` and `gpu.matmul.bandwidth` (gauges with `shape`, `m`, `n`, `k`, `time_ns`), and table `gpu.matmul.batch_invariance`, `gpu.matmul.cpuref_exact` and `gpu.matmul.probe`. `templates/matmul.md` and `report.fsx` show one way.
- **V1:** the same cases and bench run on Vulkan once it exists. The probes tell whether its `Fma` is fused. A flush class compares the denormal cases in its mode (KernelBounds).
- **What would reopen this:** a backend whose rows alone differ from batched rows (BatchInvariance), or a dispatch at the Qwen shapes over 100 ms.

## Alternatives

- **64-wide k slabs** (32 KiB of groupshared, with an XOR swizzle instead of padding), measured on the same device:
  - decode got 25 to 35% faster (k_proj 38.9 to 29 µs, down_proj 115 to 83 µs);
  - prefill fell from 13 to 15 TFLOP/s to 10.5 to 11 TFLOP/s, because twice the groupshared halves the groups per SM;
  - lm_head decode slowed from 0.72 to 0.89 ms.

  Rejected.
- **A gemv path for small m inside the kernel.** For example, one thread per (row, column) for m <= 4 that reads w straight from memory. Decode could approach the bandwidth bound: 1.19 GB of weights per token is about 1.7 ms at the RTX 4080's rated 717 GB/s (a specification figure, not measured here), against 11.5 ms now. It would be a second, m-dependent code path. The plan admits it only if it is bit-identical, and on D3D12 that would hold if both paths compile `mad` to the same fused step. Deferred to the post-Gate 1 performance track (dev plan, Phase 2 item 9), where it would go through conformance and the cross-call check.
- **Split-K for decode.** Forbidden: it changes the order of the reduction with the shape (BatchInvariance).
- **Cooperative matrix or wave operations.** Excluded in Phase 1 by the kernel rules.
