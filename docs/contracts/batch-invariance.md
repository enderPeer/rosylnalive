# BatchInvariance contract

Owner: R4 (R4a) owns the rule. The proofs belong to the owners of what computes: O2 (the CPU reference), K1 to K3 (the kernels), E1 (the conformance harness), M2 (the Qwen forward pass), X1 (the GPU executor) and C1/C2 (failover). Status: v1, frozen at the contracts-v1 tag.

## The rule

**The order of every floating-point reduction depends only on the reduction's length, never on how much other work runs beside it.** Consequently, within one ExecutionClass:

1. **A row computed alone equals the same row computed inside any batch**, bit for bit. A row is one output row of a row-wise operation: one token's hidden vector, one query row of attention, one row of a matmul.
2. **Prefill equals decode.** Running T tokens as one prefill gives the same bits for every row as running them one at a time with the KV cache (the canonical schedule: prefill, then single-token decode steps), and as any chunked prefill.

Bit-identity is claimed only within one ExecutionClass (ExecutionClass contract). Across classes, for example D3D12 against Vulkan, results are compared with tolerances (ToleranceSpec, KernelBounds).

## What it means for each operation

- **Elementwise kernels** (`scale_f32`, `bf16_unpack`, `add`, `silu_mul`, `copy_rows`, `embed_gather_bf16`, `rope_apply`) compute each element from its own inputs only. They are invariant by construction. Grid folding (KernelAbiAndCatalog) changes which group computes an element, never how.
- **Row reductions** (`rmsnorm_rows`, `argmax_rows`): one thread group per row. The order in which the group combines a row's elements (per-thread partial sums, then a fixed tree in groupshared) depends only on the row length and the catalog's fixed thread count, never on the number of rows or on which group handles the row. `argmax_rows` compares exactly and takes the lowest index on ties, so its result is order independent anyway.
- **Matmul** (`matmul_bf16w`): every output element y[i, j] accumulates p = 0, 1, ..., k-1 **sequentially in increasing order** with `mad`, starting from 0, whatever m and n are. Tiling over i and j decides which group computes y[i, j], never the order of its sum. There is no split-K and no reduction across groups. Rows beyond m inside a tile are computed or skipped without touching valid rows. So the rows of an M = 1 decode equal the same rows at M = 37 and M = 512, and the logits of the last prefill row equal the logits of a decode step at the same position.
- **Attention** (`attention_gqa_causal`): the result for a query at position p depends only on q, the keys and values at positions 0..p, and the fixed algorithm:
  - keys are visited in **blocks of 64 anchored at position 0** (block b covers positions 64b..64b+63), in increasing block order;
  - the online softmax state (running maximum, running sum, accumulator) is updated once per block, in that order;
  - inside a block, the reduction order depends only on the block size and head_dim;
  - the causal mask is the loop bound: the last block stops at position p, and positions after p are never read.

  Nothing depends on `rows`, `q_row0` or `past` except which queries a dispatch computes, so decode, prefill and chunked prefill agree.
- **CPU reference** (O2, CpuReferenceSemantics): sequential FMA reductions in increasing index order, parallel only over output rows, no `Vector<T>`, so the same bits at 1 and 32 threads.
- **Forbidden because they break this rule**: atomics (the order of arrival varies), wave and quad operations (the wave size varies between devices), and any reduction whose shape depends on the batch, such as split-K chosen from m or reductions across rows.

## Failover never relies on it

Fault tolerance (ClusterProtocol, StagePipeline) uses **exact replay**: a lost worker's state is rebuilt by re-running the same schedule of the same work units, which gives the same bits in the same ExecutionClass whatever the kernels do. A **fast rebuild** (for example rebuilding a KV cache by one prefill instead of replaying the decode steps) is allowed only for classes whose invariance has been verified, and must equal the replay bit for bit; C2 checks it, and a mismatch raises `cluster.divergence`. Batch invariance is therefore a hard kernel property and a tripwire, not a premise of correctness.

## Proof obligations

| Who | What they show |
| --- | --- |
| O2 | CpuRef F32 and F64 rows alone equal rows in a batch, and 1 thread equals 32 threads |
| K1, K2, K3 | Conformance cases with the same row alone and in batches; for K2, M = 1 rows equal M = 37 and M = 512 rows; for K3, decode equals prefill equals chunked prefill, past up to 4000 |
| E1 | Every conformance case gets a batch-invariance check, and CpuDevice is bit-exact to CpuRef F32 |
| M2 | Prefill and decode give bit-equal logits on the tiny model |
| X1 | The tiny model on CpuDevice is bit-identical to CpuOps F32 over prefill plus 16 decode steps |
| C2 | A fast rebuild equals exact replay |

## Verification

The rule is checked where things compute: CpuRef tests (O2), the conformance harness on CpuDevice in CI and on each GPU locally (E1, K1 to K3), the model and executor tests (M2, X1) and the pipeline tests (C2). R4a's own part is structural: the catalog fixes thread counts and grids, the shader convention forbids atomics and wave operations, and grid folding never changes the arithmetic.
