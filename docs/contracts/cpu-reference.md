# CpuReferenceSemantics contract

Owner: O2. Status: v1, frozen at the contracts-v1 tag. Consumers:
- M2: CpuOps, the cpu-f32 and cpu-f64 backends, built on these ops;
- E1: CpuDevice emulators, which must be bit-exact to CpuRef F32;
- K1, K2 and K3: the kernels, which these definitions specify;
- KernelBounds, X1, C2 and I1: the reference outputs.

Code: `src/Lrm.Tensor.CpuReference/CpuExec.fs`, `Rope.fs`, `Rows.fs` and `CpuRef.fs`. Tests: `tests/Lrm.Tensor.Tests/CpuRefTests.fs`, `RopeTests.fs` and `TinyModelTests.fs` (cpu tier).

After contracts-v1, this text and its vectors change only through a `contract-change` PR, with a CHANGELOG entry and updated vectors. The oracles CpuRef is measured against are ReferenceOracles (`reference-oracles.md`); the tap names are ActivationNames (`activation-names.md`).

**The F# CPU reference is the specification of every forward kernel** (CLAUDE.md). It has two modes:
- **CpuRef F32** computes in float32, like the GPUs. CpuDevice reproduces it bit for bit; each GPU stays within KernelBounds of it.
- **CpuRef F64** computes in float64. It is the fp64 truth on our side, matched against torch64p.

## One definition, two modes

Every op is one generic F# definition over `'T`, instantiated as `float32` (F32) or `float` (F64), so both modes run exactly the same sequence of operations. Any other `'T`, such as `Half`, raises `ArgumentException`.

- **Activations** are row-major `HostTensor<'T>` (HostTensorAndSafetensors), read as `[Rows, Cols]`.
- **The row ops** `softmax`, `logSumExp`, `argmax`, `argmaxRows`, `topK` and `kl` need rank 2 or more. Rank 0 and rank 1 raise `ArgumentException`. A vector `[n]` has `Rows = n` and `Cols = 1`, so a logits vector would otherwise silently become n rows of one element. Reshape one row of logits to `[1, n]`.
- **Weights** are BF16 `HostTensor<uint16>`, widened exactly to `'T` (`Bf16.widen`; in F64 through the exact float32 value). They are never rounded.
- **Results** are new tensors. `Rows.*` are the row kernels on spans, with no allocation except in `topK`, for CpuDevice to run over heap memory. The tensor ops in `CpuRef.*` apply them per row.

## Arithmetic rules

1. **Reductions are sequential, in increasing index order, from +0.**
2. **Products in a reduction are fused.** Each step of a dot product is one fused multiply-add: `acc = fma(x[i], y[i], acc)`, with a single rounding, through `'T.FusedMultiplyAdd`, which is `MathF.FusedMultiplyAdd` or `Math.FusedMultiplyAdd`. It is correctly rounded whether the CPU has FMA3 or not, so the bits never depend on that.
3. **Plain sums use plain adds**, such as the softmax denominator.
4. **Element-wise formulas round after every operation, in the order written.** RyuJIT never contracts `a*b + c` on its own, and CpuRef uses no `Vector<T>` and no other SIMD type.
5. **Division and square root** are IEEE, correctly rounded.
6. **`exp` and `log`** come from the .NET runtime (`MathF.Exp`, `Math.Exp`, `Math.Log`). Their bits can differ between runtimes and CPUs. This is why ExecutionClass (`execution-class.md`) records the runtime as `Driver` and the ISA flags as `Arch`.
7. **The RoPE table uses no library function at all** (see ropeTable below).
8. **Parallelism.** `CpuExec.Threads` splits work only over independent outputs: rows, (query row, head) pairs, or blocks of 64 output columns of one matmul row. Every output element is computed whole, on one thread, in the fixed order. **The bits are the same for every thread count.** With `CpuExec.Serial` (1 thread), every op is a plain loop on the calling thread.
9. **Batch invariance** (BatchInvariance contract). An output row depends only on its own input row, plus the cache rows it attends to. A row computed alone equals the same row inside any batch, and a decode step equals the same position inside a prefill, bit for bit.

## Ops

In the formulas, `widen` is exact bf16 widening and `fma(a, b, c)` is `a·b + c` with one rounding.

| Op | Definition |
| --- | --- |
| `embed table ids` | `y[r, c] = widen(table[ids[r], c])`, shape `[ids, hidden]`. An id outside `0 .. vocab − 1` raises `ArgumentOutOfRangeException` |
| `embedGather table ids` | The `embed_gather_bf16` kernel's form: the same, but an id at or above `vocab` gives a row of float32 quiet NaN, bits `7fc00000`, which F64 widens to `7ff8000000000000` |
| `rmsNorm x w eps` | Per row of `n = w.Length` elements: `ss = Σ fma(x[c], x[c], ·)`; `mean = ss / n`; `r = 1 / sqrt(mean + eps)`; `y[c] = widen(w[c]) · (x[c] · r)`. The weight is used as `w`, never `1 + w` (Qwen3RMSNorm). `x` may hold any number of rows, so `[T, heads·head_dim]` with a `head_dim` weight is the per-head q_norm and k_norm. `eps` is `rms_norm_eps` rounded once to `'T` |
| `matmulBf16W x w` | `x` is `[m, k]` and `w` is `[n, k]`, the PyTorch Linear layout. `y[i, j] = Σ_p fma(x[i, p], widen(w[j, p]), ·)` over `p = 0 .. k − 1` from +0, giving `[m, n]`. The tied `lm_head` is this op with the embedding table as `w` |
| `ropeTable invFreqBits maxPos` | T_spec (below): float32 `cos` and `sin`, each `[maxPos, head_dim/2]` |
| `ropeApply table x heads pos0` | NeoX rotate-half per head of `head_dim = table.HeadDim`. Row `t` has position `pos0 + t`, which must lie inside the table. For `i < h = head_dim/2`, with `c = cos[pos, i]` and `s = sin[pos, i]` widened: `y[i] = (x[i]·c) − (x[i+h]·s)` and `y[i+h] = (x[i+h]·c) + (x[i]·s)`, each product rounded before the add. This is `q·cos + rotate_half(q)·sin` term by term |
| `attention q k v shape past scale` | Causal grouped-query attention. `q` is `[rows, QHeads·HeadDim]`, after norm and RoPE. `k` and `v` are caches `[Tk, KvHeads·HeadDim]` with `Tk ≥ past + rows`. Query row `i` has position `past + i`. Query head `h` reads KV head `h / (QHeads / KvHeads)` and keys `j = 0 .. past + i`; this loop bound is the causal mask. Then: `s[j] = dot(q_h, k_j) · scale`, the dot rounded, then one multiply; `m = max_j s[j]`; `e[j] = exp(s[j] − m)`; `l = Σ_j e[j]`; `p[j] = e[j] / l`; `o[d] = Σ_j fma(p[j], v_j[d], ·)` in increasing `j` from +0. The output is `[rows, QHeads·HeadDim]`, heads concatenated in HF order |
| `attentionScale headDim` | `head_dim ** −0.5`, computed in float64 (`Math.Pow`, as Python's `**`) and rounded once to `'T` |
| `siluMul gate up` | `s = g / (1 + exp(−g))`, then `y = s · u` |
| `add a b` | `a + b` |
| `copyRows src r0 dst d0 rows` | Copies whole rows, bit for bit |
| `softmax x` | Per row: `m = max`; `e[i] = exp(x[i] − m)`; `l = Σ e`; `y[i] = e[i] / l`. A NaN anywhere makes the whole row NaN. As in torch, a −∞ entry gives 0, while a +∞ entry or an all −∞ row gives NaN |
| `logSumExp x` | Per row: `m + log(Σ exp(x[i] − m))`, with `m` the row maximum. When `m` is infinite, the result is `m`, as in `torch.logsumexp`: +∞ for a row holding +∞, and −∞ for a row of only −∞. The formula alone would give ∞ − ∞ = NaN. A NaN anywhere gives NaN |
| `argmax x` | Per row: the lowest index of the maximum, by strict `>`, so +0 and −0 tie. A row holding a NaN raises `ArgumentException`; an empty row raises too |
| `argmaxRows x` | The `argmax_rows` kernel's form: the same index, or `0xFFFFFFFF` for a row holding a NaN |
| `topK x k` | Per row: the indices of the k largest values, largest first, equal values in index order. A NaN raises |
| `kl p q` | Per row of logits: `KL(P ‖ Q) = Σ_i fma(exp(lp[i]), lp[i] − lq[i], ·)`, where `lp = p − logSumExp p` and `lq = q − logSumExp q`. A term whose computed `exp(lp[i])` is 0 is skipped (0 · log 0 = 0), as in `Metrics.klRow`. So a logit masked to −∞ in both rows adds nothing, where the formula would add 0 · (−∞ − −∞) = NaN. A skipped term with a finite `lp[i] − lq[i]` would have added a zero. Mass of P where Q has none gives +∞. A NaN, a +∞ logit or a row of only −∞ gives NaN |

### The KV cache

- **K is stored after k_norm and RoPE; V is stored raw**, the `v_proj` output. These are the `attn.k_rope` and `attn.v_cache` taps.
- **The cache is in the mode's dtype.** It is float32 for CpuRef F32, which is what every GPU keeps, and float64 for CpuRef F64 (ActivationNames: a tap has the dtype of its computation).
  - This refines the plan's wording "both fp32" (dev plan, section 5). The cache is fp32 in F32 mode and on every GPU, and in the mode's dtype in CpuRef.
  - An fp32 cache in F64 mode would round K and V to float32 at every layer. That error alone exceeds the 1e-12 acceptance against torch64p, whose cache is float64.
- **Layout.** Each cache is `[maxPos, KvHeads·HeadDim]`, row `p` for position `p`. Attention reads only rows `0 .. past + rows − 1`, so rows beyond them may hold anything.

### ropeTable: T_spec without libm

`ropeTable invFreqBits maxPos` (also `RopeTable.create exec`) builds the ReferenceOracles table:

1. `freq[pos, j] = f32(f32(pos) · inv_freq[j])`, one float32 multiply of the exact position (`pos < 2^24`) and the pinned bits.
   - `maxPos` lies in `1 .. 2^24`.
   - `maxPos · head_dim/2` is at most `Array.MaxLength` (2,147,483,591), because cos and sin are each one .NET array. For example, head_dim 256 at 2^24 positions is 2^31 entries.
   - Anything else raises `ArgumentException` naming `maxPos`, before any allocation.
2. `cos[pos, j] = f32(RN64(cos freq))` and `sin[pos, j] = f32(RN64(sin freq))`. RN64 is the correctly rounded float64, and f32 rounds that to nearest even.

**The float64 values are computed in exact fixed-point integer arithmetic** (`ExactTrig.cosSin`), never by the C runtime. This is recorded in `docs/decisions/O2-0001-exact-rope-table.md`.
- **Reduction.** The argument is reduced by π/2, which is computed with Machin's formula. The precision carries 64 guard bits beyond the integer part of `x / (π/2)`.
- **Series.** sin and cos of the reduced argument (`|r| ≤ π/4`) are Taylor series in Q-bit fixed point, starting with Q = 128.
- **Certified rounding.** The total error is far below 2^16 units of 2^−Q. A float64 is accepted only when both ends of that interval round to it; otherwise Q doubles and the step repeats.
- **Tiny arguments.** For `|x| < 2^−27`, `RN64(sin x) = x` and `RN64(cos x) = 1` exactly.

So the table is the same on every machine, runtime and thread count, and it equals T_spec bit for bit. **F64 mode uses the same float32 table, widened.** The exact table for Qwen3-0.6B (8192 × 64) takes 0.2 s on 32 threads.

## CpuExec

`CpuExec` is `{ Threads: int }`:
- `CpuExec.Serial` is 1 thread;
- `CpuExec.Machine` is one thread per logical processor;
- `CpuExec.WithThreads n` is n threads, with n at least 1.

An exception inside a parallel loop is rethrown as itself, not wrapped in `AggregateException`.

## Telemetry

Tensor ops open a `cpuref.<op>` span (`embed`, `rmsnorm`, `matmul`, `rope`, `attention`, `silu_mul`, `add`) with `rows`, `cols` and `threads`, and only when the ambient facade is enabled. Telemetry never changes a value (TelemetryFacade). Comparisons against goldens emit `golden.compare` through the comparators (ComparatorsAndGoldenReader).

## Vectors

**Arithmetic.** These were computed independently, with Python float arithmetic:

| Case | Input | Result |
| --- | --- | --- |
| Fused dot step | `x = (1, 1 + 2^−12)`, `y = (−1, 1 + 2^−12)`, F32 | `2^−11 + 2^−24`, bits `3a000400`. An unfused product would give `3a000000` |
| Index order | `x = (1, 1e8, −1e8)`, `y = (1, 1, 1)`, F32 | `0`. The sum from the end would be 1 |
| rmsNorm | `x = (3, 4)`, `w = bf16(1, 2)`, `eps = 0`, F32 | `r = 3e90d0c3`; `y = (3f593924, 4010d0c3)`. With `w = 0` the result is 0, since the weight is used as `w` |
| Attention | one head, `head_dim = 2`, `past = 1`, `scale = 1`; `q = (1, 0)`; `k = ((0, 5), (ln 3, 7))`; `v = ((4, 8), (0, 4))` | weights (1/4, 3/4); `o = (1, 5)` |
| attentionScale | head_dim 32 (F64); head_dim 128 (F32) | `0.1767766952966369`; bits `3db504f3` |
| embedGather | id ≥ vocab | F32 row `7fc00000`; F64 row `7ff8000000000000` |
| argmax | `(1, 3, 3)`; `(−0, +0)`; a row with NaN | 1; 0; raises (`argmaxRows`: `ffffffff`) |

**RoPE.** The table equals T_spec bit for bit:
- tiny (theta 1e6, head_dim 32, positions 0..511): digest `615b4204…b6d60801`, equal to `goldens/tiny/g1/rope.safetensors`;
- Qwen3-0.6B (head_dim 128, positions 0..8191): digest `da820d28…8f6ebfaf`;
- the ReferenceOracles entry vectors for both.

The 64 Qwen `inv_freq` bits come from the planning probe `redteam/inv_torch.txt`, and their SHA-256 is the contract's `138c99b1…`. The correctly rounded `cos 1` and `sin 1` are `3fe14a280fb5068c` and `3feaed548f090cee` (mpmath, 200 bits).

**Acceptance on the tiny model** (`goldens/tiny/g1`; measured, informative). In the per-op columns each op is fed the oracle's own input taps. In the chain columns the full CpuRef forward runs from the ids.

| Op | Taps (p0) | F64 vs torch64p, worst relL2 | F32 vs torch32, worst relL2 |
| --- | ---: | ---: | ---: |
| embed | 1 | 0 | 0 |
| rmsnorm | 9 | 2.4e-16 | 1.1e-7 |
| matmul | 15 | 4.8e-16 | 2.4e-7 |
| rope | 4 | 0 | 2.1e-8 |
| attention | 2 | 1.5e-16 | 6.1e-8 |
| silu_mul | 2 | 3.2e-17 | 1.5e-8 |
| add | 4 | 0 | 0 |
| chain, all 39 taps (p0, p1) | 39 | 1.5e-15 | 8.1e-7 (vs torch64p 8.8e-7) |
| chain, logits (p2, 70 tokens) | 1 | 1.2e-15 | 7.2e-7 |

The bounds are 1e-12 (F64) and 1e-6 (F32) per op. The chain is held to 1e-12 (F64) and to the CPU32 relL2 floor of 1e-5 (F32), with max |Δlogit| ≤ 2e-3 and KL ≤ 1e-7. In addition:
- 64 greedy steps from p0 and p1 give the golden ids in both modes;
- the forward is bit-identical at 1, 7 and 32 threads;
- prefill equals token-by-token decode, for the logits and both caches.

## Verification

The cpu tier of `Lrm.Tensor.Tests` covers:
- **The vectors** above.
- **Exactness:** exact bf16 widening, including subnormal and NaN bits; a one-hot matmul returns the widened weight column; the shape checks.
- **Infinities and ranks:**
  - `logSumExp` gives +∞, −∞ and NaN as above;
  - `kl` stays finite for logits masked to −∞ in both rows, in both modes. It equals the KL without the masked entry bit for bit, and `Metrics.klRow` to 1e-13;
  - every row op refuses a vector and a scalar;
  - `ropeTable` refuses a table beyond the array limit.
- **Batch invariance for T in {1, 7, 33, 128}, in both modes:** every row of embed, rmsNorm, matmul, RoPE, attention, siluMul and add equals the row computed alone. Each attention row equals a decode step over the same cache, and a prefill split into two chunks equals the whole.
- **Thread counts:** every op is bit-identical at 1, 7 and 32 threads, in both modes, at T = 128.
- **Properties** (fixed FsCheck seed):
  - argmax is the lowest index of the maximum;
  - topK equals a stable descending sort;
  - matmul rows do not depend on the batch, for random m, n and k;
  - `ExactTrig.cosSin` is within one float64 ulp of the C runtime, and even or odd in x.
- **The RoPE table:** T_spec bit for bit, both digests, the entry vectors, and the same table at 1 and 32 threads. An informational line reports how many Qwen entries a `Math.Cos`/`Math.Sin` table would get wrong on this machine (0 here).
- **The tiny model:** per-op and chain acceptance, greedy decoding, thread counts and prefill = decode, as in the table above. Each comparison emits a `golden.compare` record.
