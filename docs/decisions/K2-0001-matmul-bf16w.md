# ADR K2-0001: matmul_bf16w is one tiled kernel whose every output is a sequential fused chain over k

- **Status:** proposed (K2, 2026-10-18), with PR #25.
- **Owner:** K2.
- **Scope:** `src/Lrm.Shaders/hlsl/kernels/matmul/matmul_bf16w.hlsl` and `tests/Lrm.Kernels.Conformance/Cases/matmul/`. It binds X1 (how it calls the kernel: the lm_head row cap, the decode cost), I1 (the kernels report and the bench), E1 (GPU-only cases in its model tests), T1 (the throughput records) and V1 (the Vulkan runs still owed).

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
   - w is read per thread in chunks of 8 bf16: with `Load4` when k is even and the chunk ends at or below k, and element by element (`lrm_load_bf16`) otherwise. For odd k, w rows start in the middle of a u32 word. For even k, only the chunk that crosses k is read element by element.
   - Elements past k are staged as 0 and never read.
   - Loads only move exact values, so the load path cannot change a result.
6. **No second code path in Phase 1.** There is no gemv variant and no m-dependent path. The plan admits a gemv variant only if it is bit-identical (see Alternatives).
7. **Cases.**
   - Every Qwen3-0.6B projection at m = 1, 37, 512 and 4,000, and lm_head at m = 1 and 64 (X1's row cap, below). All cases of one shape share one weight and the first rows of one activation matrix, so row r is the same inputs in every m.
   - 19 edge shapes:
     - tile edges in m and n;
     - k mod 4 = 0, 1, 2 and 3, a tail of 0 to 3 steps;
     - a full or partial last slab, for odd and for even k;
     - even k whose w chunk crosses k (k = 2, 34, 44 and 66), next to chunks below k and past it;
     - odd k, k below one slab and k = 0;
     - the widest grid without folding, n = 64 x 65,535.

     They cover the remainders of k that select a different path, not every remainder against the 32-wide slab.
   - m = 512 and 4,000 and both lm_head cases are GPU-only. They are not built in the cpu tier at all, because E1's model tests run every discovered case on CpuDevice, GPU-only cases included (request to E1). The tier is read as Runner.main reads it: the last `--tier` on the command line, else LRM_TEST_TIER.
8. **The bench** (`MatmulBench.fs`, run by `bench.fsx` until a tier lists it). It measures throughput at the Qwen shapes, warm and from idle clocks. It also checks rows across m = 1, 37 and 512 calls against each other and against CpuRef F32, and runs two arithmetic probes. It records `gpu.matmul.*`, and `report.fsx` renders `templates/matmul.md` from them with T1's engine and charts, until the kernels template charts these records.
   - **Warm:** after 1 s of load, the sweep runs back to back. Each point is the fastest of 5 dispatches after a warm-up dispatch.
   - **Cold:** after the warm sweep, lm_head at every m and the projections at m = 4,000 are timed as single dispatches, each alone after the GPU has idled 3 s. Each point takes two and keeps the slower (`cold_ns`), because the clocks do not always drop within the gap.
   - **Budget:** judged on the slower of warm and cold, for what X1 dispatches: the projections up to 4,000 rows and lm_head up to 64. lm_head is measured up to 1,024 rows to show where the cap comes from.
   - FLOPs and traffic are the FLOP registry's (`OpCost.ofOp`, `Op.MatMul`).

## Evidence

All runs: NVIDIA GeForce RTX 4080 (`pci:10de:2704`), driver 616.92 (UMD 32.0.16.1692), Windows 11 10.0.26200. DXIL ShaderSetHash `2d513d8126713397f063ce95670a82760ac55c0e77ded350a7208a369c3c951e`, SPIR-V ShaderSetHash `3ed88c0a917b8c3cee01b1b7cd3f693058d97a806774e762be557141781c539a`; the kernel has not changed since `f98c360`. The conformance runs are at commit `e1f776a` and the bench at `3f84e15`. Reports: `tests/Lrm.Kernels.Conformance/Cases/matmul/reports/`.

- **Blobs.** DXC 1.9.2609.5 compiles both targets with `-WX`, and the SPIR-V blob passes all 16 inspector rules. The DXIL has 80 `dx.op.tertiary` FMad (opcode 46) and no `fmul` or `fadd`; the SPIR-V has 80 GLSL.std.450 `Fma`. Groupshared is 2 x 2,304 floats = 18,432 bytes, and there are no local arrays (`alloca`).
- **Conformance on D3D12** (gpu tier, run `20261018-140156-4152f402`):
  - all 45 matmul cases pass (41 K2 and 4 E1), each twice bit-identical, with all 102 batch splits matching;
  - on every case the measured relL2 equals CpuRef F32's own reference error, which indicates the same bits as CpuRef F32;
  - the slowest case is 23.4 ms, the faster of its two runs (down_proj at m = 4,000). lm_head at m = 64 takes 1.53 ms.
- **Conformance on CpuDevice** (cpu tier, run `20261018-140141-3dc48e1e`): the 29 K2 cases that CI runs and E1's 4 pass, bit-exact to CpuRef F32. The conformance exe's cpu tier runs 130 tests in 7.8 s. It runs the same 130 when the tier comes from `--tier cpu` alone.
- **Across calls** (bench run `20261018-140617-1ce83fea`). The outputs hold a signalling-NaN poison before the dispatches. For all six shapes, 9 rows (0, 1, 15, 16, 36, 63, 64, 100 and 511) computed alone equal the same rows of the m = 512 call and, below 37, of the m = 37 call. All are equal to CpuRef F32 bit for bit: 0 differing elements of 9 x (2,048 + 1,024 + 1,024 + 3,072 + 1,024 + 151,936).
- **Arithmetic probes** (same run, informational):

  | Probe | x | w | Result | Meaning |
  | --- | --- | --- | --- | --- |
  | `fused` | `bf810001 3f800001` | `3f80 3f81` | `30800000` (2^-30) | one fused multiply-add per step; a rounded product and a sum give `00000000` |
  | `order` | `3f800000 4b800000 cb800000` | `3f80 3f80 3f80` | `00000000` | p in increasing order; decreasing p gives `3f800000` |

  The fp32 denormal probe measures `preserve`.
- **Throughput, warm** (same run): the fastest of 5 back-to-back dispatches, after 1 s of load.

  | Shape (n x k) | m = 1 | m = 37 | m = 512 | m = 4,000 |
  | --- | --- | --- | --- | --- |
  | q_proj (2,048 x 1,024) | 34.8 µs, 121 GB/s | 57.3 µs | 166 µs, 12.9 TFLOP/s | 1.11 ms, 15.1 TFLOP/s |
  | k_proj, v_proj (1,024 x 1,024) | 34.8 µs, 60 GB/s | 57.3 µs | 85.0 µs, 12.6 TFLOP/s | 0.57 ms, 14.7 TFLOP/s |
  | o_proj (1,024 x 2,048) | 68.6 µs, 61 GB/s | 114 µs | 169 µs, 12.7 TFLOP/s | 1.13 ms, 14.9 TFLOP/s |
  | gate_proj, up_proj (3,072 x 1,024) | 34.8 µs, 181 GB/s | 57.3 µs | 243 µs, 13.3 TFLOP/s | 1.66 ms, 15.2 TFLOP/s |
  | down_proj (1,024 x 3,072) | 102 µs, 62 GB/s | 170 µs | 250 µs, 12.9 TFLOP/s | 1.73 ms, 14.6 TFLOP/s |
  | lm_head (151,936 x 1,024) | 650 µs, 479 GB/s | | 10.7 ms, 14.9 TFLOP/s | m = 64: 1.33 ms, 15.0 TFLOP/s; m = 1,024: 21.5 ms |

  - A decode token's 196 projections take 28 x 345 µs = 9.7 ms, plus 0.65 ms for lm_head.
  - Decode is bound by latency, not bandwidth. At m = 1 the grid has n / 64 groups (16 for n = 1,024, on 76 SMs), and each walks k / 32 slabs one after another, so the time is about k / 32 slab round trips.
  - An earlier sweep without the 1 s of load (commit `558815b`) measured decode 10 to 13% slower and 13.1 to 13.6 TFLOP/s at m = 4,000, because its first points ran before the clocks were up.
- **From idle clocks** (same run): the slower of two single dispatches, each after 3 s idle, against the warm time.

  | Dispatch | Warm | Cold | Of the budget | X1 dispatches it |
  | --- | --- | --- | --- | --- |
  | q_proj, m = 4,000 | 1.11 ms | 15.7 ms | 16% | yes |
  | k_proj, v_proj, m = 4,000 | 0.57 ms | 7.5 ms | 8% | yes |
  | o_proj, m = 4,000 | 1.13 ms | 16.4 ms | 16% | yes |
  | gate_proj, up_proj, m = 4,000 | 1.66 ms | 23.8 ms | 24% | yes |
  | down_proj, m = 4,000 | 1.73 ms | 25.7 ms | 26% | yes |
  | lm_head, m = 1 | 0.65 ms | 18.0 ms | 18% | yes |
  | lm_head, m = 64 | 1.33 ms | 20.7 ms | 21% | yes (the cap) |
  | lm_head, m = 128 | 2.66 ms | 43.5 ms | 43% | no |
  | lm_head, m = 256 | 5.32 ms | 85.0 ms | 85% | no |
  | lm_head, m = 512 | 10.7 ms | 183 ms | 183% | no |
  | lm_head, m = 1,024 | 21.5 ms | 322 ms | 322% | no |

  - A dispatch from idle clocks takes about 13 to 28 times its warm time.
  - Other runs agree. Two earlier bench runs gave lm_head 46 to 50 ms at m = 128 and 92 to 94 ms at m = 256. A reviewer's scratch run gave 84 to 93 ms at m = 256 and 87 to 164 ms at m = 512 and 1,024; it measured 19 to 24 ms at m = 128.
  - The clocks do not always drop within 3 s: in one run lm_head at m = 1 took 0.73 ms after the gap, and in the next 18.0 ms. Hence two samples per point, of which the slower is kept.
  - The conformance harness runs each case after seconds of host work, so its times are of the same kind: 23.4 ms for down_proj at m = 4,000.
- **The checks catch a batch-variant kernel.** A scratch mutation flipped the last mantissa bit of every output when m > 1, a 1-ulp change.
  - It made the cross-call check fail on all six shapes, with 14 x n differences each (9 rows against m = 512 and 5 against m = 37). CpuRef still agreed with the untouched m = 1 rows.
  - It also made all six m = 37 conformance cases fail, through their batch splits only, inside the relative bound.
  - The mutation was reverted, and the ShaderSetHash returned to `2d513d81…`.
- **Vulkan has not run.** V1 does not exist yet, because vk.xml is not approved for download. Only the SPIR-V blob is checked (above). The acceptance "both backends" stays open until the Vulkan runs listed under Consequences are done.

## Consequences

- **X1:**
  - Call the kernel with the actual m of each dispatch. Prefill chunks of up to 4,000 rows are within the budget for every projection: at most 1.73 ms warm and 25.7 ms from idle clocks (a 3.9 x margin).
  - Give lm_head at most 64 rows per dispatch (`MatmulBench.LmHeadMaxRows`), and compute only the logits a step needs.
    - 64 rows take 1.33 ms warm and about 21 ms from idle clocks, the same margin as the projections' chunks.
    - They already run at 15.0 TFLOP/s, so a larger chunk would gain nothing.
    - From idle clocks, 128 rows take 43 to 50 ms and 256 rows 85 to 94 ms, too close to the budget.
    - The 1 GiB page limit alone caps lm_head's y at 1,766 rows.
  - Rows are batch-invariant, so chunking never changes a bit.
- **I1:** the bench's throughput sweep belongs in the bench tier, and its cross-call check in the gpu tier. Until the conformance exe lists `MatmulBench.bench ()` and `MatmulBench.gpu ()`, the GFLOP/s sweep, the cold dispatches and the cross-call check run only outside the test tiers, with `dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/bench.fsx [bench|gpu]`. `tools/test.ps1 -Tier bench` and `-Tier gpu` do not produce `gpu.matmul.*` records yet.
- **E1:** the model tests should skip `GpuOnly` cases, as the CpuDevice suite does. Then the tier gate in `MatmulCases.fs` can go.
- **T1:**
  - The kernels template should chart the `gpu.matmul.throughput` and `gpu.matmul.bandwidth` gauges. Their attributes are `shape`, `m`, `n`, `k`, `time_ns`, `cold_ns`, `cold_fast_ns`, `idle_ms` and `x1_dispatch`.
  - It should table the `gpu.matmul.batch_invariance`, `gpu.matmul.cpuref_exact` and `gpu.matmul.probe` checks.
  - `templates/matmul.md` and `report.fsx` show one way.
  - The bench run's kernels report counts the cold lm_head dispatches past X1's cap (m = 512 and 1,024) as over the budget. They are measured on purpose, and X1 never makes them.
- **V1 and K2 (open, before wave-3-done):**
  - Run the K2 conformance cases on Vulkan in the gpu tier.
  - Run the bench (`bench.fsx`) on Vulkan for the cross-call check, the cold dispatches and the probes. The `fused` and `order` probes tell whether Vulkan's `Fma` is one fused step in increasing p, as CpuRef F32 is.
  - A flush class compares the denormal cases in its mode (KernelBounds).
  - Until these runs are done, "both backends" is not met.
- **What would reopen this:** a backend whose rows alone differ from batched rows (BatchInvariance), or a dispatch X1 makes at the Qwen shapes over 100 ms, warm or from idle clocks.

## Alternatives

- **64-wide k slabs** (32 KiB of groupshared, with an XOR swizzle instead of padding), measured on the same device in the earlier sweep without the 1 s of load:
  - decode got 25 to 35% faster (k_proj 38.9 to 29 µs, down_proj 115 to 83 µs);
  - prefill fell from 13 to 15 TFLOP/s to 10.5 to 11 TFLOP/s, because twice the groupshared halves the groups per SM;
  - lm_head decode slowed from 0.72 to 0.89 ms.

  Rejected.
- **A larger lm_head chunk** (128 or 256 rows, as an earlier draft of this ADR said). Rejected: from idle clocks it takes 43 to 94 ms (above), and per row it is no faster warm.
- **A gemv path for small m inside the kernel.** For example, one thread per (row, column) for m <= 4 that reads w straight from memory.
  - Decode could approach the bandwidth bound: 1.19 GB of weights per token is about 1.7 ms at the RTX 4080's rated 717 GB/s (a specification figure, not measured here), against 10.3 ms now.
  - It would be a second, m-dependent code path. The plan admits it only if it is bit-identical. On D3D12 that would hold if both paths compile `mad` to the same fused step.
  - Deferred to the post-Gate 1 performance track (dev plan, Phase 2 item 9), where it would go through conformance and the cross-call check.
- **Split-K for decode.** Forbidden: it changes the order of the reduction with the shape (BatchInvariance).
- **Cooperative matrix or wave operations.** Excluded in Phase 1 by the kernel rules.
