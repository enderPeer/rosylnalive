# ADR K1-0001: The basic kernels: fixed reduction trees, integer NaN tests and `precise` where CpuRef's rounding is the rule

- **Status:** proposed (K1, 2026-10-18), with the K1 pull request.
- **Owner:** K1.
- **Scope:** the seven kernels in `src/Lrm.Shaders/hlsl/kernels/basic/` (`embed_gather_bf16`, `rmsnorm_rows`, `rope_apply`, `silu_mul`, `add`, `copy_rows`, `argmax_rows`) and their conformance cases in `tests/Lrm.Kernels.Conformance/Cases/basic/`. It binds X1 (which dispatches them) and V1 (whose SPIR-V class runs the same bodies), and it records measured facts that I1's per-backend parity relies on.

## Context

The catalog entries of the seven kernels are frozen at contracts-v1 (KernelAbiAndCatalog): fields, thread counts and grids. Inside them, the kernels must:

- meet KernelBounds: `embed_gather_bf16`, `add`, `copy_rows` and the `argmax_rows` index bit-exact (`add` and `argmax_rows` in the class's measured fp32 denormal mode); `rmsnorm_rows`, `rope_apply` and `silu_mul` within the relative bound;
- be batch-invariant (BatchInvariance): a reduction's order depends only on its length;
- use no atomics, wave operations or 16-bit types, at most 32 KiB of groupshared, one body for DXIL and SPIR-V (ShaderBuildAndSourceConvention v1.1);
- finish every dispatch in under 100 ms for the Qwen3-0.6B shapes.

Two compiler facts shape the code. DXC marks DXIL float arithmetic `fast` (no `-Gis`, ADR int-0001), which allows reassociation, FMA contraction and the assumption that no value is NaN; so `v != v` is folded away. And SPIR-V declares no denormal mode, so a Vulkan class may flush where D3D12 preserves.

## Decision

1. **No catalog change.** Every kernel keeps its frozen entry. Element-wise kernels (`silu_mul`, `add`, `copy_rows`) run one thread per element at `LRM_GROUP_LINEAR(gid) * 256 + tid.x`; `embed_gather_bf16` uses its two-dimensional grid (`gid.x` a block of 256 columns, `gid.y` the row) and never folds; `rmsnorm_rows`, `rope_apply` and `argmax_rows` run one group per row at `LRM_GROUP_LINEAR(gid)`. Every kernel returns early when its group or thread has no work, judged from its own fields, so the extra groups of a folded grid write nothing.
2. **Row reductions use a fixed tree.** `rmsnorm_rows` and `argmax_rows` (256 threads) let thread t walk columns t, t + 256, ... in increasing order, then combine the 256 partials in groupshared with h = 128, 64, ..., 1 and a barrier after each step. The order depends only on `cols` and the catalog's thread count. Groupshared use is 1 KiB (`rmsnorm_rows`) and 3 KiB (`argmax_rows`), well under 32 KiB. A group whose row is out of range returns before its first barrier, as a whole.
3. **`rmsnorm_rows`** accumulates `mad(x, x, s)` per thread from +0, divides the tree's sum by `cols`, takes `rsqrt(mean + eps)` and writes `widen(w) * (x * r)`, CpuRef's element order. The sum's order differs from CpuRef's single sequential sum, so the kernel is judged by the relative bound, as the catalog says.
4. **`argmax_rows` tests NaN with integers.** A column is NaN when `(bits & 0x7FFFFFFF) > 0x7F800000`; a NaN never reaches a float comparison, and the row's flag is OR-ed through the tree, giving `0xFFFFFFFF`. A thread keeps its first maximum (it replaces only on `>`), and the tree takes the greater value or, on equal values, the lower index, so the result is the lowest index of the maximum whatever thread held it; +0 and -0 tie. The comparisons stay float comparisons, so a flushing class compares denormals as zero, as KernelBounds' flushed reference does.
5. **`precise` where CpuRef's rounding is the rule.** `add`'s sum and `rope_apply`'s four products, difference and sum are `precise`. DXIL then has plain `fadd`, `fmul` and `fsub` (no `fast`), and SPIR-V gets `NoContraction` (the inspector accepts it): each operation rounds once, in the order of CpuRef's `Rows.add` and `Rows.ropeHead`, and no product is contracted into an FMA.
6. **`silu_mul` uses the GPU's own `exp` and division** (DXIL `Exp` of `-g * log2(e)`, a fast `fdiv`), in CpuRef's order `g / (1 + exp(-g))` then `* up`. Its measured error is a twentieth of the bound (Evidence), so no range-reduced `exp` is written.
7. **Bit moves are integer-only.** `embed_gather_bf16` shifts the bf16 bits into the high half of a u32 and writes `0x7FC00000` for an id at or above `vocab`; `copy_rows` moves u32 words. No float instruction touches the data, so NaN payloads, signed zeros and denormals pass unchanged in every denormal mode.
8. **u32 index arithmetic.** Every element index is below its tensor's count, which one page (1 GiB) bounds by 2^28 four-byte or 2^29 bf16 elements, so the products of the kernel-abi.md minimum counts (`id * cols + c`, `row * cols`, `(row0 + r) * stride + c`) cannot wrap.
9. **The conformance cases** (`Cases/basic/BasicCases.fs`, 78 cases): the Qwen3-0.6B shapes at decode (T = 1) and prefill (T = 37 and 512), and edge shapes (one row or element, odd sizes, partial groups, outputs whose capacity exceeds the work, in-place aliases, folded grids, ties, NaN, denormals, every bf16 pattern). The `rope_apply` cases read T_spec computed from the pinned `inv_freq` bits by CpuRef's exact trigonometry, checked at discovery against the ReferenceOracles hash `138c99b1…` and three entry vectors; positions the case does not use hold quiet NaNs, so a kernel that reads a wrong position fails. Six cases are `Case.gpuOnly`: the whole Qwen embedding table (151,936 × 1024 bf16, 311 MB), and folded grids of Qwen's shapes (k_norm of 8,193 tokens, q RoPE of 4,097 tokens, and `silu_mul`, `add` and `copy_rows` over 65,537 groups).
10. **The GPU-only cases are not built in the cpu tier** (`LRM_TEST_TIER=cpu`). CpuDevice never plans them (`Suites.casesFor`), but `HarnessTests`' arithmetic-model tests plan every discovered case on three CpuDevices at once; with these six cases the cpu tier of the conformance exe ran out of memory under an 8 GB GC heap limit (without a limit its working set peaked at 26.9 GB). Without them it passes under a 4 GB limit (peak 3.3 GB, 11 s). The gate goes when those tests filter GPU-only cases as `Suites.casesFor` does (request to E1).

## Evidence

Measured on the development machine: NVIDIA GeForce RTX 4080 (`pci:10de:2704`), D3D12 UMD 32.0.16.1692 (driver 616.92), Windows 11 10.0.26200, pinned DXC 1.9.2609.5. With the seven kernels the shader set has 12 of 14 catalog kernels (`matmul_bf16w` and `attention_gqa_causal` missing); DXIL ShaderSetHash `9a7fe626e4d3f931f6fbb59f7067821781ff355c799f963635ce43de4a2511bb`, SPIR-V ShaderSetHash `d18ab1abd9914933ea835256d259f6c489b4485f093e85925f371b28f06e43a6`. All 12 SPIR-V blobs pass the inspector's 16 rules (`check-shaders.ps1`); `argmax_rows` declares 3,072 Workgroup bytes and `rmsnorm_rows` 1,024.

**Conformance on D3D12** (gpu tier of `Lrm.Kernels.Conformance`, d3d12 mutex; run `20261018-125923-5c5d258f` at commit `74b14a7`, class id `95b5c71f37ff`, rendered in [`tests/Lrm.Kernels.Conformance/Cases/basic/reports/d3d12/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/basic/reports/d3d12/kernels.md)). The class measured `preserve`. Every case passed: run twice with fresh poison and bit-identical, within its bound, batch-invariant over its splits, and timed. The table covers K1's 78 cases and E1's 20 harness cases of the same kernels (the run's other 28 cases are the selftests).

| Kernel | Bound | Cases (K1 + E1) | Passed | Largest measured | Its reference error | Largest measured / bound | Splits matched | Slowest dispatch (faster run) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `embed_gather_bf16` | exact | 9 + 1 | 10 | 0 elements differ | | | 26 of 26 | 54.3 µs (`rows65535_cols3`) |
| `rmsnorm_rows` | relative, 1e-6 | 16 + 6 | 22 | 9.43e-8 (`cols3072`) | 4.63e-7 | 0.094 | 53 of 53 | 1.70 ms (`e1_rows70000_cols2`) |
| `rope_apply` | relative, 1e-6 | 13 + 2 | 15 | 3.67e-8 (`e1_tokens1_heads2_pos15`) | 3.67e-8 | 0.037 | 31 of 31 | 46.0 µs (`qwen_q_4097_tokens_in_place`) |
| `silu_mul` | relative, 1e-6 | 9 + 2 | 11 | 5.75e-8 (`e1_in_place_up`) | 4.59e-8 | 0.058 | 30 of 30 | 5.92 ms (`folded_n16777472`) |
| `add` | exact, fp32 arithmetic | 10 + 4 | 14 | 0 elements differ | | | 39 of 39 | 320 µs (`folded_n16777472`) |
| `copy_rows` | exact | 11 + 3 | 14 | 0 elements differ | | | 33 of 33 | 1.24 ms (`rows65537_cols256_folded`) |
| `argmax_rows` | exact (the index) | 10 + 2 | 12 | 0 elements differ | | | 29 of 29 | 1.43 ms (`e1_rows70000_cols2`) |

Every relative bound here is the floor, 1e-6: no case's reference error exceeds 5e-7.

- **`rope_apply` equals CpuRef F32.** In every case its measured relL2 against CpuRef F64 equals CpuRef F32's own relL2 to the last digit, which (short of an exact cancellation inside relL2) means its outputs are CpuRef F32's bits: decision 5 holds on this class.
- **`rmsnorm_rows` sits below CpuRef F32's own error** in 20 of its 22 cases: the tree sums pairwise, which rounds less than one sequential sum of 1,024 or 8,191 squares.
- **Timing.** Dispatch times are the faster of two runs with the GPU's own timestamps, taken while the harness uploads and checks between dispatches, so the GPU mostly sits at low clocks (it idles at P8: 210 MHz SM, 405 MHz memory) and its residency state varies with the session's 942 MiB heap. The same dispatch varies by an order of magnitude between runs: the folded `add` took 18.7 ms in an earlier full run (`20261018-124928-eb2f05da`) and 320 µs in this one, and the folded `silu_mul` took 270 µs in a run of only the `silu_mul` cases (`20261018-130059-7f0d5a53`) against 5.92 ms here. The numbers show the 100 ms budget holds with a wide margin; they are not the kernels' throughput, which belongs to the bench tier.

**Conformance on CpuDevice** (cpu tier, run `20261018-130019-da5f259d`, [`reports/cpu-device/kernels.md`](../../tests/Lrm.Kernels.Conformance/Cases/basic/reports/cpu-device/kernels.md)): K1's 72 CI cases pass bit-exact to CpuRef F32, and every case fails its negative control. The arithmetic-model tests also pass every K1 case against flushed and NaN-canonical references; before decision 10 they ran the six GPU-only cases too, and those passed as well.

**Mutation check on D3D12.** One build with one bug in each kernel (run `20261018-125623-babc5f61`), then the clean sources again:

| Kernel | Bug | Cases that failed |
| --- | --- | --- |
| `embed_gather_bf16` | `id <= vocab` instead of `<` | 5 of 10: every case with an id equal to `vocab` |
| `rmsnorm_rows` | the tree starts at h = 64, dropping partials 128..255 | 11 of 22: every case with more than 128 columns |
| `rope_apply` | the position ignores `pos0` | 14 of 15: all but decode at position 0 |
| `silu_mul` | `gid.x` instead of `LRM_GROUP_LINEAR(gid)` | 1 of 11: the folded GPU-only case |
| `add` | reads `b[i + 1]` | 14 of 14 |
| `copy_rows` | destination rows strided by `cols`, not `dst_stride` | 4 of 14: every case whose `dst_stride` differs from `cols` where the stride is used |
| `argmax_rows` | equal values take the higher index | 2 of 12: the tie cases |

**Disassembly** (pinned DXC, the build's flags, `-Fc`): `add` and `rope_apply` have no `fast` flag on their arithmetic; `argmax_rows` tests NaN with `and i32 %x, 2147483647` and `icmp ugt i32 ..., 2139095040`, and its float compares are `fcmp fast ogt` and `oeq` on values that are never NaN; `silu_mul` multiplies by `0xBFF7154760000000` (−log2 e) with `fmul fast` before `Exp`; `rmsnorm_rows` has `FMad` and `Rsqrt`. In SPIR-V, `rope_apply` carries six and `add` one `NoContraction` decoration.

## Consequences

- **V1:** the SPIR-V blobs of these kernels pass the inspector, but no Vulkan device has run them; V1 runs the same cases through the harness. If V1's class measures `flush`, `add` and `argmax_rows` are compared in that mode by the harness, and the relative kernels get KernelBounds' flushed allowance.
- **X1:** `embed_gather_bf16` takes at most 65,535 rows per dispatch (grid y), so a longer prefill is split. `rope_apply` reads the Rope arena's table at `pos0 + row / heads`. `argmax_rows` returns `0xFFFFFFFF` for a row with a NaN, which the host turns into CpuRef's error.
- **E1:** the request in decision 10.
- **Reopen** if a backend's class fails one of these cases, if `-Gis` is adopted for DXIL (decision 5 would then be the default and `silu_mul`'s numbers change), or if a driver update changes the measured errors.

## Alternatives

- **A sequential sum per row in one thread** for `rmsnorm_rows` would match CpuRef's order exactly, but leaves 255 of 256 threads idle; the relative bound is the rule for this kernel, and the tree is more accurate.
- **Integer ordering keys for `argmax_rows`** (comparing sign-adjusted bits) avoid float compares entirely, but rank -0 below +0 and a denormal above zero in a flushing class, so they would disagree with CpuRef's tie rule and with the flushed reference.
- **A range-reduced `exp` for `silu_mul`** (n = round(g · log2 e), then a short polynomial) would shrink its error, but under fast-math DXC may reassociate the reduction, and the measured error already leaves a factor of twenty.
- **`-Gis` for the whole DXIL set** would remove fast-math everywhere, but it changes every DXIL blob and the ShaderSetHash, a ShaderBuildAndSourceConvention contract-change (ADR int-0001, decision 5); `precise` gives the same rounding to the two kernels that need it.
