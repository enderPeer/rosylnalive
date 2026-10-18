namespace Lrm.Gpu.Exec

open System

/// How the GPU executor records its dispatches (GpuExecutor contract, "Options").
type GpuExecOptions =
    {
        /// Bracket every dispatch with a GPU timer, reported at Wait as a `gpu.dispatch` interval with `kernel`,
        /// `layer`, `step` and `phase` (what the kernels template charts). Create the backend without
        /// DeviceOptions.TimeDispatches, or every dispatch is reported twice. Timers are recorded only while the
        /// ambient telemetry is enabled; they never change a value.
        TimeDispatches: bool
        /// The most query-key pairs per head one attention dispatch covers: a dispatch of `rows` queries whose first is
        /// at position `past` covers rows · past + rows · (rows + 1) / 2 (ADR K3-0001, decision 8: 2^20).
        AttentionPairsPerDispatch: int64
        /// The most multiply-adds (rows · n · k) one matmul dispatch covers. The default is lm_head of Qwen3-0.6B at
        /// 64 rows, 64 · 151,936 · 1,024 (ADR K2-0001, the lm_head row cap).
        MatMulMacsPerDispatch: int64
        /// The most rows a forward of the stage with the head takes with HeadRows.All: its logits [rows, vocab] are
        /// planned at min(T, MaxLogitsRows, the rows one 1 GiB page holds), and BeginForward refuses an All-head forward
        /// of more rows. HeadRows.Last (greedy prefill) and single-row forwards run at every T the plan holds. A plan that
        /// only ever reads the last row (greedy decoding) sets 1 and keeps no [T, vocab] logits.
        MaxLogitsRows: int
        /// True: the ops own the backend, and Dispose (or a failed create) disposes it, with its heap and weights.
        OwnsBackend: bool
    }

    /// Timers on, 2^20 query-key pairs per attention dispatch, the lm_head cap per matmul dispatch, All-head logits up
    /// to what one page holds, and the ops own the backend.
    static member Default =
        {
            TimeDispatches = true
            AttentionPairsPerDispatch = 1L <<< 20
            MatMulMacsPerDispatch = 64L * 151936L * 1024L
            MaxLogitsRows = Int32.MaxValue
            OwnsBackend = true
        }

[<RequireQualifiedAccess>]
module GpuExecOptions =
    /// Raises ArgumentException unless both budgets and MaxLogitsRows are at least 1.
    let validate (o: GpuExecOptions) =
        if o.AttentionPairsPerDispatch < 1L then
            invalidArg "options" (sprintf "%d query-key pairs per attention dispatch" o.AttentionPairsPerDispatch)

        if o.MatMulMacsPerDispatch < 1L then
            invalidArg "options" (sprintf "%d multiply-adds per matmul dispatch" o.MatMulMacsPerDispatch)

        if o.MaxLogitsRows < 1 then
            invalidArg "options" (sprintf "%d logits rows for HeadRows.All" o.MaxLogitsRows)

/// How the executor splits long work so every dispatch stays inside the 100 ms budget (KernelAbiAndCatalog, kernel
/// rules). Rows are batch-invariant (BatchInvariance), so a split never changes a bit.
[<RequireQualifiedAccess>]
module Chunking =
    /// The query-key pairs per head of one attention dispatch of `rows` queries whose first is at position `past`:
    /// query i attends past + i + 1 keys.
    let attentionPairs (rows: int64) (past: int64) = rows * past + rows * (rows + 1L) / 2L

    /// The attention dispatches of `rows` query rows at positions past .. past + rows - 1, as (q_row0, rows) pairs in
    /// order: each takes the most rows whose pairs stay within maxPairs, and at least one.
    let attention (maxPairs: int64) (rows: int) (past: int) : (int * int) list =
        if maxPairs < 1L then
            invalidArg "maxPairs" (sprintf "%d pairs" maxPairs)

        if rows < 0 || past < 0 then
            invalidArg "rows" (sprintf "%d rows at %d" rows past)

        let chunk (r0: int) =
            let p = int64 past + int64 r0
            let left = int64 (rows - r0)
            let fits (n: int64) = attentionPairs n p <= maxPairs
            // n^2 / 2 + n (p + 1/2) <= maxPairs, solved in float64, then corrected in integers.
            let b = 2.0 * float p + 1.0

            let estimate =
                int64 (Math.Floor((-b + Math.Sqrt(b * b + 8.0 * float maxPairs)) / 2.0))

            let mutable n = max 1L (min left estimate)

            while n > 1L && not (fits n) do
                n <- n - 1L

            while n < left && fits (n + 1L) do
                n <- n + 1L

            int n

        let rec go (r0: int) (acc: (int * int) list) =
            if r0 >= rows then
                List.rev acc
            else
                let n = chunk r0
                go (r0 + n) ((r0, n) :: acc)

        go 0 []

    /// The most rows of x [m, k] one matmul_bf16w dispatch with n outputs takes: maxMacs / (n · k), rounded down to a
    /// multiple of the kernel's 64-row tile when it is at least 64, and at least 1.
    let matmulRows (maxMacs: int64) (n: int) (k: int) : int =
        if maxMacs < 1L then
            invalidArg "maxMacs" (sprintf "%d multiply-adds" maxMacs)

        let perRow = max 1L (int64 n * int64 k)
        let r = maxMacs / perRow

        if r >= 64L then
            int (min (r / 64L * 64L) (int64 Int32.MaxValue / 64L * 64L))
        else
            int (max 1L r)

    /// The matmul dispatches of m rows: (row0, rows) pairs in order, each at most matmulRows rows.
    let matmul (maxMacs: int64) (m: int) (n: int) (k: int) : (int * int) list =
        let step = matmulRows maxMacs n k

        [ for r0 in 0..step .. m - 1 -> r0, min step (m - r0) ]
