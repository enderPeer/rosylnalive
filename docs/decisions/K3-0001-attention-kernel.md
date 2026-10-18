# ADR K3-0001: attention_gqa_causal is an online softmax over 64-key blocks in CpuRef's dot order, with staged keys

- **Status:** proposed (K3, 2026-10-18), with the K3 pull request (#27).
- **Owner:** K3.
- **Scope:** `src/Lrm.Shaders/hlsl/kernels/attention/attention_gqa_causal.hlsl`, its conformance cases in `tests/Lrm.Kernels.Conformance/Cases/attention/`, and the facts that the executor (X1: chunk sizes), the conformance harness (E1: timing) and the kernels reports (T1, I1) build on.

## Context

- **The catalog entry is frozen** (KernelAbiAndCatalog, kernel 13). It has 64x1x1 threads and the grid `(rows, q_heads, 1)`, and it is not linear, so it is never folded. Its fields are `q`, `k`, `v`, `y`, `rows`, `q_row0`, `past`, `q_heads`, `kv_heads`, `head_dim` and `scale`. Its argument rules are `kv_heads >= 1` dividing `q_heads`, and `head_dim >= 1` of any size. Query row i reads q row `q_row0 + i`, has position `past + i` and attends keys 0 to `past + i`.
- **BatchInvariance, "Attention".** Keys are visited in blocks of 64 anchored at position 0, in increasing block order, and the online-softmax state is updated once per block. The order inside a block depends only on the block size and head_dim. The causal mask is the loop bound. Nothing depends on `rows`, `q_row0` or `past` except which queries a dispatch computes.
- **KernelBounds.** Attention is relative: relL2(gpu, CpuRef64) <= max(1e-6, 2 · relL2(CpuRef32, CpuRef64)), plus the flushed-denormal allowance. CpuRef's attention (`Rows.attention`) is a plain softmax:
  - the score is `dot(q, k_j) * scale`, with the dot as `fma` in increasing d from +0;
  - then the row maximum, `exp(s - m)`, a sequential sum and a division;
  - then the sum of `fma(p_j, v_j[d], .)` in increasing j.
- **Kernel rules.** No atomics, no wave operations and no 16-bit types. groupshared is at most 32 KiB. Every dispatch is under 100 ms on the RTX 4080, as the faster of two runs in the harness. DXIL arithmetic is fast-math (ADR int-0001).
- **The guidance.** Positions up to at least 4,096 keys must run under 100 ms, chunking the prefill if needed. There is no -inf arithmetic and no NaN.

## Decision

1. **The order** (the kernel's header states it the same way). One group per (query row, query head), with thread t on key `64b + t` of block b.
   - The score is `s_j = (sum of mad(q[d], k_j[d], .) in increasing d from +0) * scale`, CpuRef's dot order.
   - The block maximum is exact. The running maximum is `m = M_0` after block 0 and `m' = max(m, M_b)` after block b, and `alpha = exp(m - m')`.
   - The weights are `e_j = exp(s_j - m')`. The block sum is the sum of `e_j` in increasing j from +0, and the block value is the sum of `mad(e_j, v_j[d], .)` in increasing j.
   - The state starts at `l = o = +0`. Per block, `l = mad(l, alpha, block sum)` and `o[d] = mad(o[d], alpha, block value[d])`. At the end, `y[d] = o[d] / l`.
   - The last block stops at key p: no key or value after p is read.
2. **No -inf and no NaN from finite inputs.** The running maximum starts at block 0's maximum, not at -inf. `l >= 1`, because the key holding the maximum adds `exp(0) = 1` and nothing rescales it afterwards.
3. **Any head_dim.** Output dimensions are handled in passes of 128: in pass c, thread t owns `128c + t` and `128c + 64 + t`. A pass recomputes the scores and weights of the pass before exactly, with the same instructions on the same inputs. Qwen3-0.6B, with head_dim 128, runs one pass.
4. **Group order.** Group `gid` computes head `gid.y` of row `rows - 1 - gid.x`, so the rows with the most keys start first. It is still one group per (row, head), and which group computes a pair changes no bit. This saves about 5% on a 4096-token prefill (exploration table below).
5. **Memory.**
   - Keys are staged through groupshared 32 dimensions at a time (64 x 33 floats, a padded stride, so the warps' reads are conflict-free). Each thread issues 16 coalesced loads before its 16 stores.
   - A last chunk narrower than 32 dimensions is staged element by element, so a small head_dim still spreads its loads over all 64 threads.
   - Values are read straight from the cache, 16 keys of loads before their 16 mads.
   - groupshared is 9,088 bytes, as the SPIR-V inspector counts it.
6. **No contract change.** The catalog entry's fields, threads, grid and docs are used as frozen. No contract text changes.
7. **Gated cases stay at about 4 ms on a warm GPU.** The harness times dispatches while the GPU is still at its idle clock (Evidence, "Cold clocks"), so larger dispatches are measured by the bench (`Cases/attention/bench/attention-bench.fsx`) and are not gated. They are: chunks of 512 and 1024 queries at 4096 keys, a 4096-token prefill in one dispatch, and the 65,535-query grid.
8. **Chunk size for X1.** At positions up to 4096, prefill attention should be chunked to at most 512 queries per dispatch: the last chunk takes 8.7 ms warm. A 1024-query chunk takes 16.5 ms warm, and a whole 4096-token prefill 36.7 ms warm. Both are under 100 ms only once the clock is up.

## Evidence

**Machine and build.**
- GPU: NVIDIA GeForce RTX 4080 (`pci:10de:2704`), driver 616.92 (UMD 32.0.16.1692), on Windows 11 10.0.26200.
- Compiler: DXC 1.9.2609.5, with the shader set's flags (DXIL `-denorm preserve`).
- ShaderSetHash: DXIL `a9ac4697732f854bb8bc53bdc5110ee4632da53b3a6417d8bbaef971a2f7e599`, SPIR-V `20ed6c4df196dc4128cd551164a2671ac43ec3c62a3ef91db85eab3e5dafb6a6`.
- Commit `7817370`. Blobs: DXIL 11,444 bytes, SPIR-V 24,568 bytes. The SPIR-V blob passes all 16 inspector rules (`check-shaders.ps1`).
- **Vulkan has not run:** there is no V1 backend yet (vk.xml is not approved).

**Conformance on D3D12.** Run `20261018-131421-a93dfcb4` (`pwsh tools/test.ps1 -Tier gpu`, D3D12 mutex), rendered in [`reports/d3d12/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/attention/reports/d3d12/kernels.md).
- All 35 attention cases pass: K3's 33 and E1's 2. Each one runs twice bit-identically with poisoned outputs, stays within KernelBounds and matches every batch-invariance split (decode at `past` and at `past + rows - 1`, and the middle third as a chunked prefill).
- In every case, the cache rows after the last readable key and the q rows the dispatch does not own hold NaN. The y rows outside the dispatch stayed poisoned.

| Case | relL2(gpu, CpuRef64) | relL2(CpuRef32, CpuRef64) | Bound | Time in the harness |
| --- | ---: | ---: | ---: | ---: |
| `qwen_decode_p63` | 4.80e-7 | 4.67e-7 | 1e-6 | 0.007 ms |
| `qwen_prefill_t130` | 3.59e-7 | 3.63e-7 | 1e-6 | 0.062 ms |
| `qwen_decode_p4095` | 7.55e-7 | 1.43e-6 | 2.86e-6 | 0.343 ms |
| `qwen_chunk_t128_past3968` | 6.74e-7 | 1.58e-6 | 3.16e-6 | 10.6 ms |
| `adv_large_scores` | 2.00e-6 | 2.00e-6 | 4.00e-6 | 0.167 ms |
| `adv_negative_scores` | 2.00e-5 | 2.00e-5 | 4.01e-5 | 0.086 ms |
| `adv_rising_scores` | 1.11e-5 | 1.11e-5 | 2.23e-5 | 0.236 ms |
| `adv_zero_queries` | 1.97e-7 | 8.21e-7 | 1.64e-6 | 1.16 ms |
| `grid_rows8192` | 1.96e-7 | 6.59e-7 | 1.32e-6 | 18.2 ms |

- **The GPU's error follows the reference's.** For short contexts, the measured error equals CpuRef32's own to within a few percent. Both start from the same fp32 scores, because the dot is CpuRef's order with `mad` and the scale is the same multiply. That rounding dominates both, so every case sits at or below half its bound.
- **Long contexts are more accurate than CpuRef32.** At 4,000 keys or more, and with q = 0 over 2,001 keys, the GPU's error is 0.24 to 0.53 of CpuRef32's. The GPU sums 64 keys per block and then about 64 blocks, while CpuRef32 adds up to 4,096 terms in one sequential sum.
- The worst ratio of measured to bound is 0.52.

**CpuDevice (cpu tier, CI).** The 32 cases not marked GPU-only (all but `grid_rows8192`) pass on CpuDevice, bit-exact to CpuRef F32, and fail their negative control. This judges the cases and the harness's view of them, not the HLSL: CpuDevice emulates the kernel with `Rows.attention`. `pwsh tools/test.ps1 -Tier cpu` passed: 906 tests, 1 ignored (the GPU-only case).

**Warm timings.** Run `20261018-131553-c3e46607` (`dotnet fsi --tailcalls- tests/Lrm.Kernels.Conformance/Cases/attention/bench/attention-bench.fsx`), rendered in [`reports/d3d12-bench/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/attention/reports/d3d12-bench/kernels.md). Each configuration was dispatched 8 times back to back after 8 untimed dispatches. The table gives the minimum, and the median is within 1%.

| Step | Configuration (16:8 heads, head_dim 128 unless noted) | Groups | Warm |
| ---: | --- | ---: | ---: |
| 0 | decode at 511 | 16 | 0.042 ms |
| 1 | decode at 1023 | 16 | 0.082 ms |
| 2 | decode at 2047 | 16 | 0.162 ms |
| 3 | decode at 4095 | 16 | 0.322 ms |
| 4 | chunk of 128 at 3968 | 2,048 | 2.61 ms |
| 5 | chunk of 512 at 3584 | 8,192 | 8.65 ms |
| 6 | chunk of 1024 at 3072 | 16,384 | 16.5 ms |
| 7 | prefill of 512 | 8,192 | 0.606 ms |
| 8 | prefill of 1024 | 16,384 | 2.25 ms |
| 9 | prefill of 2048 | 32,768 | 8.92 ms |
| 10 | prefill of 4096 | 65,536 | 36.5 ms |
| 11 | 8,192 queries, 1:1 head, head_dim 1 | 8,192 | 2.54 ms |
| 12 | 16,384 queries, head_dim 1 | 16,384 | 10.2 ms |
| 13 | 65,535 queries, head_dim 1 | 65,535 | 166 ms |

- **Decode is latency-bound.** It runs 16 groups, about 5 µs per 64-key block.
- **Prefill is bound by L2 traffic.** Each (row, head) reads its keys and values once: 140 GB for the 4096-token prefill, which is 3.8 TB/s at 36.5 ms.
- **The 65,535-query grid is over the budget even warm.** At head_dim 1, the cost per block is fixed: three barriers, a 64-element maximum and sum, and value loads that one thread issues. The last query has 1,024 blocks. The executor never issues such a dispatch, so the edge is measured and not gated. The gated wide grid has 8,192 queries.

**Cold clocks.** The harness computes a case's CPU references (up to seconds) and its uploads with the GPU idle. Then it times the dispatch, and the GPU has not raised its clock yet.
- `nvidia-smi --query-gpu=utilization.gpu,clocks.sm` sampled every 0.5 s during a harness run showed 98% utilization at 210 MHz. The clock reached 1,965 MHz only after the dispatch. Under the bench's back-to-back load it read 2,820 MHz.
- The same dispatches, harness against bench:

| Configuration | Harness (faster of two runs) | Run | Warm (bench) | Ratio |
| --- | ---: | --- | ---: | ---: |
| chunk of 128 at 3968 | 10.6 ms | `20261018-131421-a93dfcb4` | 2.61 ms | 4.1 |
| 8,192 queries, head_dim 1 | 18.2 ms | `20261018-131421-a93dfcb4` | 2.54 ms | 7.2 |
| chunk of 512 at 3584 | 95.9 ms | `20261018-131050-4b3bbe70` | 8.65 ms | 11.1 |
| 16,384 queries, head_dim 1 | 63.1 ms | `20261018-131050-4b3bbe70` | 10.2 ms | 6.2 |
| chunk of 1024 at 3072 | 69.3 ms | `20261018-130202-8dd9f3e6` | 16.5 ms | 4.2 |
| prefill of 4096 | 45.1 ms | `20261018-130202-8dd9f3e6` | 36.5 ms | 1.2 |

- The ratio depends on what ran just before. The 4096-token prefill followed the 1024-query chunk, which had raised the clock. 2,820 / 210 bounds the ratio at about 13, so a gated case must stay below about 7 ms warm. The chunk of 512 at 96 ms shows why the earlier case set, which had it, was dropped.

**Exploration.** These are scratch-bench measurements of the variants tried on the way, on the same machine and shapes. They are minimum warm times for the 4096-token prefill and decode at 4095. No run log was kept; the variants differ from the final kernel only as described.

| Variant | Prefill of 4096 | Decode at 4095 |
| --- | ---: | ---: |
| Staged 32-dimension keys, plain loops (the first staged version) | 59.8 ms | 0.681 ms |
| The same, staging steps behind run-time selects (a slower code shape on this driver) | 126 ms | 1.56 ms |
| Values batched by 8 | 53.7 ms | 0.589 ms |
| Values batched by 16 | 52.8 ms | 0.554 ms |
| Values by 16, key staging batched by 8 | 41.0 ms | 0.399 ms |
| Values by 16, key staging batched by 16 (**final**) | 36.8 to 40.4 ms | 0.324 ms |
| Final, rows in grid order instead of reversed | 38.8 ms | 0.321 ms |
| 64-dimension key chunks (17 KB groupshared, fewer groups per SM) | 66.9 ms | 0.292 ms |
| 16-dimension key chunks (8 chunks, 15 barriers per block) | 35.2 ms | 0.390 ms |
| Fully unrolled 64-load key staging and predicated 16-key value batches | 114 ms | 0.563 ms |

- Before the keys were staged, one load per element per thread, strided by the row: the harness measured the 4096-token prefill at 169 ms and the chunk of 128 at 30.6 ms. Those are cold-clock numbers, so they are only roughly comparable.
- The final kernel balances decode and prefill. The 16-dimension chunks are 4% faster on prefill and 20% slower on decode.

## Consequences

- **X1** chunks prefill attention by `q_row0` and `past` (decision 8). The kernel accepts any `q_row0`, `past` and `rows` up to 65,535, and gives the same bits for a row whatever the chunking.
- **E1 (request).** The budget is measured on an idle-clocked GPU, so a dispatch's time in the harness can be up to about 13 times its warm time, and a kernel family can flake at the budget for reasons outside it. Possible fixes:
  - dispatch each case a few times untimed (or until its GPU time stops falling) before the two timed runs;
  - or time the fastest of N runs.

  Every family's cases are affected, K2's lm_head matmul first.
- **T1 (request).** The bench tags its `gpu.dispatch` intervals with `case` and `step`. The kernels template's slowest-dispatches table shows `step` but not `case`; showing `case` would name the configurations in the bench report.
- **V1.** The SPIR-V blob is built and inspected but not yet run. On a flushing class, the `adv_denormal_values` and `adv_dominant_first` cases exercise the flushed reference and the allowance of KernelBounds.
- **What would reopen this.** A full-model attention tap of X1b outside tolerance. A device whose compiler turns this code into a slower shape, as the run-time-select variant shows a driver can. Or a contract change to several queries per group, the next performance step (below).

## Alternatives

- **Several queries per group.** The two query heads of a KV head, or several rows, could share one group's key and value loads. That would halve the L2 traffic or better, and L2 traffic bounds prefill. It changes the catalog's grid and doc, which is a contract change, and the budget is met without it.
- **Splitting the keys of a row over groups**, with a second pass to combine. It needs another kernel or atomics, and a split chosen from the batch shape is what BatchInvariance forbids.
- **A tree dot product across the 64 threads.** It needs a barrier tree per key. It also moves the dot away from CpuRef's order, so the GPU's score rounding would no longer coincide with CpuRef32's, and the error would stop tracking the reference's (1.00 of it now, against a factor of 2).
- **-inf as the initial maximum and as the mask.** The guidance forbids -inf arithmetic, and a block whose keys are all masked would give inf - inf = NaN.
- **`exp2` with log2(e) folded into the scale.** It would change the scores away from CpuRef's `dot * scale`, for no measured gain.
- **Staging values through groupshared.** It would cut the per-block cost at small head_dim (the 65,535-query edge). It adds a barrier per block and shared-memory traffic at head_dim 128, where values already stream at L2 speed.
