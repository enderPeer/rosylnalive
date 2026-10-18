# ADR K3-0001: attention_gqa_causal is an online softmax over 64-key blocks in CpuRef's dot order, with staged keys

- **Status:** proposed (K3, 2026-10-18), with the K3 pull request (#27).
- **Open: the Vulkan half of the acceptance.** K3's acceptance asks for KernelBounds and the 100 ms budget on both backends. Only D3D12 has run. The SPIR-V blob is built and inspected but has never executed, because there is no V1 backend yet. So the flush-class paths (`adv_denormal_values`, `adv_mixed_denormal_values` and `adv_dominant_first` against a flushed reference, if the Vulkan class flushes) and the precision of `GLSL.std.450 Exp` are unverified. Wave 3 is not tagged until the attention cases pass on V1.
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
2. **No -inf, and no NaN or Inf inside the value range.** The running maximum starts at block 0's maximum, not at -inf. `1 <= l <= p + 1`, because the key holding the maximum adds `exp(0) = 1` and nothing rescales it afterwards.
   - **The precondition** is finite scores and `sum_j |v_j[d]| < FLT_MAX` over the keys 0 to p. It is enough that `max |v| · (p + 1) < FLT_MAX`, about 8.3e34 at 4,096 keys.
   - **Why.** `o` stays unnormalized until the final division. Every `e_j <= 1`, so `|o[d]| <= sum_j |v_j[d]|`. Values past that sum overflow `o` to Inf, while `y = o / l` and CpuRef, which divides first, are finite.
   - **A review probe showed it** on D3D12. With q = 0 (every weight 1) and v in [0.5e36, 1.5e36) over 2,001 keys, every row from about 340 keys on was Inf: 340 · 1e36 passes FLT_MAX (3.4e38). Decode, prefill and chunked prefill still agreed bit for bit.
   - Model activations are many orders of magnitude inside the range. `adv_large_values` (1e30 over at most 103 keys) stays inside it too.
   - Keeping `o` normalized per block (`o = o · alpha · l_old / l_new + ...`) would drop the precondition. It costs a division per block and changes the order, and Phase 1 does not need it.
3. **Any head_dim.** Output dimensions are handled in passes of 128: in pass c, thread t owns `128c + t` and `128c + 64 + t`. A pass recomputes the scores and weights of the pass before exactly, with the same instructions on the same inputs. Qwen3-0.6B, with head_dim 128, runs one pass.
4. **Group order.** Group `gid` computes head `gid.y` of row `rows - 1 - gid.x`, so the rows with the most keys start first. It is still one group per (row, head), and which group computes a pair changes no bit. This saves about 5% on a 4096-token prefill (exploration table below).
5. **Memory.**
   - Keys are staged through groupshared 32 dimensions at a time (64 x 33 floats, a padded stride, so the warps' reads are conflict-free). Each thread issues 16 coalesced loads before its 16 stores.
   - A last chunk narrower than 32 dimensions is staged element by element, so a small head_dim still spreads its loads over all 64 threads.
   - Values are read straight from the cache, 16 keys of loads before their 16 mads.
   - groupshared is 9,088 bytes, as the SPIR-V inspector counts it.
6. **No contract change.** The catalog entry's fields, threads, grid and docs are used as frozen. No contract text changes.
7. **Gated cases stay at about 4 ms on a warm GPU.** The harness times dispatches while the GPU is still at its idle clock (Evidence, "Cold clocks"), so larger dispatches are measured by the bench (`Cases/attention/bench/attention-bench.fsx`) and are not gated. They are: chunks of 256, 512 and 1024 queries at 4096 keys, prefills of 900 tokens and more in one dispatch, and the 65,535-query grid.
8. **Chunk size for X1: at most 2^20 query-key pairs per dispatch.** A dispatch of `rows` queries at `past` covers `pairs = rows · past + rows · (rows + 1) / 2` query-key pairs per head, because query i attends `past + i + 1` keys.
   - **The cost.** For Qwen3-0.6B (16:8 heads, head_dim 128), once a dispatch fills the GPU, it costs about 4.4 ns per pair warm and up to 66 ns per pair cold. The 100 ms rule is measured on an idle-clocked GPU (the faster of two runs in the harness), so it holds up to about 1.5M pairs.
   - **The rule.** X1 keeps each prefill attention dispatch at or below `2^20 = 1,048,576` pairs. That is about 4.6 ms warm and at most about 70 ms cold.
   - **At positions up to 4096, that is at most 256 queries per dispatch.** The last chunk, 256 queries at 3840, is 1,015,936 pairs. It takes 4.44 ms warm and 59 to 62 ms cold.
   - **One dispatch is enough for a prompt of up to 1,447 tokens at `past` 0.** Every gate prompt (at most 900 tokens) fits: a 900-token prefill is 405,450 pairs, 1.75 ms warm and at most 21 ms cold. A 4,000-token long prompt must be chunked.
   - **512 queries and more are not safe yet.** At these positions they are safe only once the harness warms the GPU before timing (the E1 request in Consequences). 512 queries at 3584 (1.97M pairs) take 8.7 ms warm but took 103 to 123 ms cold. A whole 4096-token prefill (8.4M pairs) takes 36.8 ms warm and took 223 to 349 ms cold.
   - **Other head layouts are sized with the bench.** The cost per pair scales with them, and at a small head_dim the fixed cost per block dominates (bench steps 13 to 15).

## Evidence

**Machine and build.**
- GPU: NVIDIA GeForce RTX 4080 (`pci:10de:2704`), driver 616.92 (UMD 32.0.16.1692), on Windows 11 10.0.26200.
- Compiler: DXC 1.9.2609.5, with the shader set's flags (DXIL `-denorm preserve`).
- ShaderSetHash: DXIL `a9ac4697732f854bb8bc53bdc5110ee4632da53b3a6417d8bbaef971a2f7e599`, SPIR-V `20ed6c4df196dc4128cd551164a2671ac43ec3c62a3ef91db85eab3e5dafb6a6`.
- Commits `7817370` (the kernel) and `7d1ff23` (comments in its header, a new case and the bench's cold mode). The blobs are the same at both: DXIL 11,444 bytes, SPIR-V 24,568 bytes. The SPIR-V blob passes all 16 inspector rules (`check-shaders.ps1`).
- **Vulkan has not run:** there is no V1 backend yet (vk.xml is not approved). See "Open" above.

**Conformance on D3D12.** Run `20261018-134200-f1727848` (`pwsh tools/test.ps1 -Tier gpu`, D3D12 mutex, commit `7d1ff23`), rendered in [`reports/d3d12/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/attention/reports/d3d12/kernels.md).
- All 36 attention cases pass: K3's 34 and E1's 2. Each one runs twice bit-identically with poisoned outputs, stays within KernelBounds and matches every batch-invariance split (decode at `past` and at `past + rows - 1`, and the middle third as a chunked prefill).
- In every case, the cache rows after the last readable key and the q rows the dispatch does not own hold NaN. The y rows outside the dispatch stayed poisoned.

| Case | relL2(gpu, CpuRef64) | relL2(CpuRef32, CpuRef64) | Bound | Time in the harness |
| --- | ---: | ---: | ---: | ---: |
| `qwen_decode_p63` | 4.80e-7 | 4.67e-7 | 1e-6 | 0.008 ms |
| `qwen_prefill_t130` | 3.59e-7 | 3.63e-7 | 1e-6 | 0.062 ms |
| `qwen_decode_p4095` | 7.55e-7 | 1.43e-6 | 2.86e-6 | 0.347 ms |
| `qwen_chunk_t128_past3968` | 6.74e-7 | 1.58e-6 | 3.16e-6 | 2.65 ms |
| `adv_large_scores` | 2.00e-6 | 2.00e-6 | 4.00e-6 | 0.090 ms |
| `adv_negative_scores` | 2.00e-5 | 2.00e-5 | 4.01e-5 | 0.044 ms |
| `adv_rising_scores` | 1.11e-5 | 1.11e-5 | 2.23e-5 | 0.123 ms |
| `adv_zero_queries` | 1.97e-7 | 8.21e-7 | 1.64e-6 | 0.587 ms |
| `adv_denormal_values` | 5.14e-7 | 3.89e-6 | 13.4 | 0.041 ms |
| `adv_mixed_denormal_values` | 1.74e-7 | 1.86e-7 | 1e-6 | 0.042 ms |
| `grid_rows8192` | 1.96e-7 | 6.59e-7 | 1.32e-6 | 9.88 ms |

- **The GPU's error follows the reference's.** For the Qwen3-0.6B shapes up to 137 keys, the measured error is 0.95 to 1.03 of CpuRef32's own. Both start from the same fp32 scores, because the dot is CpuRef's order with `mad` and the scale is the same multiply. That rounding dominates both, so every case sits at or below about half its bound. The other layouts are at 0.57 to 1.24 of CpuRef32's error, and the 1.24 (`tiny_prefill_t9`, 6.6e-8) is far under the 1e-6 floor.
- **Long contexts are more accurate than CpuRef32.** At 4,000 keys or more, and with q = 0 over 2,001 keys, the GPU's error is 0.24 to 0.53 of CpuRef32's. The GPU sums 64 keys per block and then about 64 blocks, while CpuRef32 adds up to 4,096 terms in one sequential sum.
- The worst ratio of measured to bound is 0.52.
- **The denormal cases.** Every result of `adv_denormal_values` is below 2^-126, so the flushed-denormal allowance sets its bound to 13.4. The case therefore checks the runs, the splits and that no NaN appears. It does not check accuracy, which no relL2 bound can judge for denormal results. `adv_mixed_denormal_values` puts the same denormal values on the odd keys and normal ones on the even keys. Every result is normal, so its bound has no allowance, and it passes at 0.94 of CpuRef32's error.
- The harness's times vary with the clock it meets. `qwen_chunk_t128_past3968` took 2.65 ms here and 10.6 ms in the earlier run `20261018-131421-a93dfcb4` (same blobs).

**CpuDevice (cpu tier, CI).** The 33 cases not marked GPU-only (all but `grid_rows8192`) pass on CpuDevice, bit-exact to CpuRef F32, and fail their negative control. They also pass on the flushing, NaN-canonical and combined CpuDevice models against references in those modes. This judges the cases and the harness's view of them, not the HLSL: CpuDevice emulates the kernel with `Rows.attention`. `pwsh tools/test.ps1 -Tier cpu` passed: 907 tests, 1 ignored (the GPU-only case).

**Warm timings.** Run `20261018-134252-1e087772` (`dotnet fsi --tailcalls- tests/Lrm.Kernels.Conformance/Cases/attention/bench/attention-bench.fsx`), rendered in [`reports/d3d12-bench/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/attention/reports/d3d12-bench/kernels.md). Each configuration was dispatched 8 times back to back after 8 untimed dispatches. The table gives the minimum. The median is within 3%, except for the chunk of 256, where it is 11% higher (4.91 ms). "Pairs" counts the query-key pairs per head (decision 8).

| Step | Configuration (16:8 heads, head_dim 128 unless noted) | Groups | Pairs | Warm | Per pair |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | decode at 511 | 16 | 512 | 0.042 ms | |
| 1 | decode at 1023 | 16 | 1,024 | 0.082 ms | |
| 2 | decode at 2047 | 16 | 2,048 | 0.163 ms | |
| 3 | decode at 4095 | 16 | 4,096 | 0.323 ms | 79 ns |
| 4 | chunk of 128 at 3968 | 2,048 | 516,160 | 2.60 ms | 5.0 ns |
| 5 | chunk of 256 at 3840 | 4,096 | 1,015,936 | 4.44 ms | 4.4 ns |
| 6 | chunk of 512 at 3584 | 8,192 | 1,966,336 | 8.69 ms | 4.4 ns |
| 7 | chunk of 1024 at 3072 | 16,384 | 3,670,528 | 16.4 ms | 4.5 ns |
| 8 | prefill of 512 | 8,192 | 131,328 | 0.605 ms | 4.6 ns |
| 9 | prefill of 900 | 14,400 | 405,450 | 1.75 ms | 4.3 ns |
| 10 | prefill of 1024 | 16,384 | 524,800 | 2.24 ms | 4.3 ns |
| 11 | prefill of 2048 | 32,768 | 2,098,176 | 8.92 ms | 4.3 ns |
| 12 | prefill of 4096 | 65,536 | 8,390,656 | 36.8 ms | 4.4 ns |
| 13 | 8,192 queries, 1:1 head, head_dim 1 | 8,192 | 33,558,528 | 2.54 ms | |
| 14 | 16,384 queries, head_dim 1 | 16,384 | 134,225,920 | 10.2 ms | |
| 15 | 65,535 queries, head_dim 1 | 65,535 | 2,147,450,880 | 169 ms | |

- **Decode is latency-bound.** It runs 16 groups, about 5 µs per 64-key block.
- **Prefill and chunks cost 4.3 to 5.0 ns per pair** once they fill the GPU, from 2,048 groups up. That linear cost is what decision 8's rule rests on.
- **Prefill is bound by L2 traffic.** Each (row, head) reads its keys and values once: 137 GB for the 4096-token prefill, which is 3.7 TB/s at 36.8 ms.
- **The 65,535-query grid is over the budget even warm.** At head_dim 1, the cost per block is fixed: three barriers, a 64-element maximum and sum, and value loads that one thread issues. The last query has 1,024 blocks. The executor never issues such a dispatch, so the edge is measured and not gated. The gated wide grid has 8,192 queries.

**Cold clocks.** The harness computes a case's CPU references (up to seconds) and its uploads with the GPU idle. Then it times the dispatch, and the GPU has not raised its clock yet.
- `nvidia-smi --query-gpu=utilization.gpu,clocks.sm` sampled every 0.5 s during a harness run showed 98% utilization at 210 MHz. The clock reached 1,965 MHz only after the dispatch. Under the bench's back-to-back load it read 2,820 MHz.
- **The bench's cold mode** reproduces what the harness meets. Run `20261018-134306-e0f3eaaa` (`attention-bench.fsx cold`) is rendered in [`reports/d3d12-bench-cold/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/attention/reports/d3d12-bench-cold/kernels.md). It times one dispatch after 2.5 s of idle GPU, three times per configuration:

| Step | Configuration | Pairs | Cold (three runs) | Warm | Slowest / warm | Slowest per pair |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 3 | decode at 4095 | 4,096 | 2.82, 3.36, 3.36 ms | 0.323 ms | 10.4 | |
| 4 | chunk of 128 at 3968 | 516,160 | 28.5, 31.8, 34.0 ms | 2.60 ms | 13.1 | 66 ns |
| 5 | chunk of 256 at 3840 | 1,015,936 | 58.7, 58.8, 62.0 ms | 4.44 ms | 14.0 | 61 ns |
| 6 | chunk of 512 at 3584 | 1,966,336 | 103.2, 103.2, 122.6 ms | 8.69 ms | 14.1 | 62 ns |
| 7 | chunk of 1024 at 3072 | 3,670,528 | 192.6, 217.8, 218.7 ms | 16.4 ms | 13.3 | 60 ns |
| 9 | prefill of 900 | 405,450 | 2.01, 15.5, 20.8 ms | 1.75 ms | 11.9 | 51 ns |
| 10 | prefill of 1024 | 524,800 | 26.9, 29.3, 29.4 ms | 2.24 ms | 13.1 | 56 ns |
| 12 | prefill of 4096 | 8,390,656 | 223, 331, 349 ms | 36.8 ms | 9.5 | 42 ns |

- **A cold dispatch takes up to about 14 times its warm time,** close to the ratio of the clocks (2,820 / 210 MHz = 13.4). A long dispatch raises the clock while it runs, so the 4096-token prefill comes out lower, at 9.5. The fastest 900-token run (2.01 ms) met a clock still raised by the 1024-query chunk before it: 2.5 s is not always enough idle after 200 ms of load. So a dispatch must stay below about 7 ms warm to be safe at idle clocks. A gated case stays at about 4 ms (decision 7), and decision 8 gives X1 at most about 4.6 ms.
- **A review probe on `b499367` (the same blobs) measured the same.** Cold, 512 queries at 3584 took 120.0 to 122.0 ms, 256 queries at 3840 took 28 to 63 ms, and a 900-token prefill took 10 to 25 ms.
- **The harness itself, as the faster of two runs, saw the same dispatches at these times:**

| Configuration | Harness (faster of two runs) | Run | Warm (bench) | Ratio |
| --- | ---: | --- | ---: | ---: |
| chunk of 128 at 3968 | 10.6 ms | `20261018-131421-a93dfcb4` | 2.60 ms | 4.1 |
| 8,192 queries, head_dim 1 | 18.2 ms | `20261018-131421-a93dfcb4` | 2.54 ms | 7.2 |
| chunk of 512 at 3584 | 95.9 ms | `20261018-131050-4b3bbe70` | 8.69 ms | 11.0 |
| 16,384 queries, head_dim 1 | 63.1 ms | `20261018-131050-4b3bbe70` | 10.2 ms | 6.2 |
| chunk of 1024 at 3072 | 69.3 ms | `20261018-130202-8dd9f3e6` | 16.4 ms | 4.2 |
| prefill of 4096 | 45.1 ms | `20261018-130202-8dd9f3e6` | 36.8 ms | 1.2 |

- The ratio depends on what ran just before. The 4096-token prefill followed the 1024-query chunk, which had raised the clock. The chunk of 512 at 96 ms shows why the earlier case set, which had it, was dropped. The cold bench shows it would also break the budget as a chunk size for X1.

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

- **X1** chunks prefill attention by `q_row0` and `past`, at no more than 2^20 query-key pairs per dispatch (decision 8). For Qwen3-0.6B at positions up to 4096, that is 256 queries per dispatch, and a prompt of up to 900 tokens takes one dispatch. The kernel accepts any `q_row0`, `past` and `rows` up to 65,535, and gives the same bits for a row whatever the chunking, so the chunk size is free to change, for example to 512 once E1 warms the GPU before timing.
- **E1 (request).** The budget is measured on an idle-clocked GPU, so a dispatch's time in the harness can be up to about 14 times its warm time, and a kernel family can flake at the budget for reasons outside it. Possible fixes:
  - dispatch each case a few times untimed (or until its GPU time stops falling) before the two timed runs;
  - or time the fastest of N runs.

  Every family's cases are affected, K2's lm_head matmul first.
- **T1 (request).** The bench tags its `gpu.dispatch` intervals with `case` and `step`. The kernels template's slowest-dispatches table shows `step` but not `case`; showing `case` would name the configurations in the bench report.
- **V1 (open).** The SPIR-V blob is built and inspected but not yet run, so half of K3's acceptance is open (Status). When V1 lands, its gpu tier runs the 36 attention cases. On a flushing class, `adv_denormal_values`, `adv_mixed_denormal_values` and `adv_dominant_first` exercise the flushed reference and the allowance of KernelBounds. The bounds also judge `GLSL.std.450 Exp`, whose precision the Vulkan specification sets on its own terms (3 + 2·|x| ULP). The cold bench numbers are for D3D12; X1's rule needs the same check on Vulkan.
- **What would reopen this.** A full-model attention tap of X1b outside tolerance. A device whose compiler turns this code into a slower shape, as the run-time-select variant shows a driver can. Or a contract change to several queries per group, the next performance step (below).

## Alternatives

- **Several queries per group.** The two query heads of a KV head, or several rows, could share one group's key and value loads. That would halve the L2 traffic or better, and L2 traffic bounds prefill. It changes the catalog's grid and doc, which is a contract change, and the budget is met without it.
- **Splitting the keys of a row over groups**, with a second pass to combine. It needs another kernel or atomics, and a split chosen from the batch shape is what BatchInvariance forbids.
- **A tree dot product across the 64 threads.** It needs a barrier tree per key. It also moves the dot away from CpuRef's order, so the GPU's score rounding would no longer coincide with CpuRef32's, and the error would stop tracking the reference's (1.00 of it now, against a factor of 2).
- **-inf as the initial maximum and as the mask.** The guidance forbids -inf arithmetic, and a block whose keys are all masked would give inf - inf = NaN.
- **`exp2` with log2(e) folded into the scale.** It would change the scores away from CpuRef's `dot * scale`, for no measured gain.
- **Staging values through groupshared.** It would cut the per-block cost at small head_dim (the 65,535-query edge). It adds a barrier per block and shared-memory traffic at head_dim 128, where values already stream at L2 speed.
