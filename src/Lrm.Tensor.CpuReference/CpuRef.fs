namespace Lrm.Tensor.CpuReference

open System
open System.Numerics
open Lrm.Core
open Lrm.Tensor

/// The head layout of grouped-query attention: query head h reads KV head h / (QHeads / KvHeads).
type AttentionShape =
    {
        QHeads: int
        KvHeads: int
        HeadDim: int
    }

/// The F# CPU reference (CpuReferenceSemantics contract): the executable specification of every forward kernel, in
/// float32 (CpuRef F32) and float64 (CpuRef F64) through one generic definition. Activations are row-major
/// HostTensor<'T> with 'T float32 or float; weights are BF16 HostTensor<uint16>, widened exactly. Results are
/// bit-identical for every CpuExec thread count, and every output row depends only on its own inputs, never on the
/// batch it is computed in.
[<RequireQualifiedAccess>]
module CpuRef =
    let private span (name: string) (exec: CpuExec) (rows: int) (cols: int) =
        if Telemetry.enabled () then
            Telemetry.span name [ Attr.int "rows" rows; Attr.int "cols" cols; Attr.int "threads" exec.Threads ]
        else
            Telemetry.noop.Span(name, [])

    let private result<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (shape: Shape)
        (data: 'T[])
        =
        HostTensor<'T>(Elt.dtype<'T>(), shape, data)

    let private bf16 (what: string) (w: HostTensor<uint16>) =
        if w.DType <> DType.BF16 then
            invalidArg what (sprintf "%s is %s, not bf16" what (DType.name w.DType))

    let private sameShape (what: string) (a: Shape) (b: Shape) =
        if not (a.Equals b) then
            invalidArg what (sprintf "shapes differ: %O and %O" a b)

    /// The row ops (softmax, logSumExp, argmax, argmaxRows, topK, kl) read a tensor as [Rows, Cols]. A rank-1 [n] has
    /// Rows = n and Cols = 1, so a logits vector would silently become n rows of one element: refuse rank 0 and 1.
    let private rowsOnly (what: string) (s: Shape) =
        if s.Rank < 2 then
            invalidArg
                what
                (sprintf "%s is %O: a row op needs [rows, cols] (rank 2 or more); reshape a vector [n] to [1, n]" what s)

    /// The pinned RoPE table T_spec for positions 0 .. maxPos - 1 (RopeTable.create on every processor).
    let ropeTable (invFreqBits: uint32[]) (maxPos: int) =
        RopeTable.create CpuExec.Machine invFreqBits maxPos

    /// The attention score scale head_dim ** -0.5, computed in float64 (as Python's ** does) and rounded once to 'T.
    let attentionScale<'T when 'T :> IFloatingPointIeee754<'T>> (headDim: int) : 'T =
        if headDim < 1 then
            invalidArg "headDim" "head_dim must be positive"

        'T.CreateTruncating(Math.Pow(float headDim, -0.5))

    /// bf16 bits widened exactly to 'T, same shape.
    let widen<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (w: HostTensor<uint16>)
        : HostTensor<'T> =
        bf16 "w" w
        result w.Shape (w.Data |> Array.map Rows.widen<'T>)

    let private gather<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (table: HostTensor<uint16>)
        (count: int)
        (idOf: int -> int64)
        =
        bf16 "table" table

        if table.Shape.Rank <> 2 then
            invalidArg "table" (sprintf "an embedding table is [vocab, hidden], not %O" table.Shape)

        let vocab = table.Rows
        let cols = table.Cols
        let y = Array.zeroCreate<'T>(count * cols)
        // The catalog's out-of-range row: float32 quiet NaN 0x7FC00000, widened.
        let missing = Rows.ofF32<'T>(BitConverter.UInt32BitsToSingle 0x7FC00000u)
        use _ = span "cpuref.embed" exec count cols

        Par.forEach exec count (fun r ->
            let id = idOf r
            let dst = Span<'T>(y, r * cols, cols)

            if id >= 0L && id < int64 vocab then
                let src = ReadOnlySpan<uint16>(table.Data, int id * cols, cols)

                for c in 0 .. cols - 1 do
                    dst[c] <- Rows.widen<'T> src[c]
            else
                dst.Fill missing)

        result (Shape.matrix count cols) y

    /// Embedding rows: y[r, c] = widen(table[ids[r], c]), shape [ids, hidden]. Raises ArgumentOutOfRangeException for an
    /// id outside 0 .. vocab - 1.
    let embed<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (table: HostTensor<uint16>)
        (ids: int[])
        : HostTensor<'T> =
        for i in 0 .. ids.Length - 1 do
            if ids[i] < 0 || ids[i] >= table.Rows then
                raise (
                    ArgumentOutOfRangeException(
                        "ids",
                        sprintf "id %d at %d is outside the vocabulary (%d)" ids[i] i table.Rows
                    )
                )

        gather exec table ids.Length (fun r -> int64 ids[r])

    /// The embed_gather_bf16 kernel's semantics: as embed, but an id at or above vocab gives a row of quiet NaN
    /// (float32 bits 0x7FC00000, widened in F64 mode).
    let embedGather<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (table: HostTensor<uint16>)
        (ids: uint32[])
        : HostTensor<'T> =
        gather exec table ids.Length (fun r -> int64 ids[r])

    /// RMSNorm over rows of w.Length elements (Rows.rmsNorm per row), weight used as w. x holds any number of such rows,
    /// so [T, heads * head_dim] with a head_dim weight is the per-head q_norm and k_norm. Same shape as x.
    let rmsNorm<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (x: HostTensor<'T>)
        (w: HostTensor<uint16>)
        (eps: 'T)
        : HostTensor<'T> =
        bf16 "w" w
        let width = w.Length

        if width = 0 || x.Length % width <> 0 then
            invalidArg "w" (sprintf "%d elements are not rows of %d" x.Length width)

        let rows = x.Length / width
        let y = Array.zeroCreate<'T> x.Length
        use _ = span "cpuref.rmsnorm" exec rows width

        Par.forEach exec rows (fun r ->
            Rows.rmsNorm
                (ReadOnlySpan<'T>(x.Data, r * width, width))
                (ReadOnlySpan<uint16>(w.Data))
                eps
                (Span<'T>(y, r * width, width)))

        result x.Shape y

    /// Output columns per parallel matmul task.
    [<Literal>]
    let private MatmulColumnBlock = 64

    /// y[i, j] = sum_p x[i, p] * widen(w[j, p]) (Rows.dotBf16): x [m, k], w [n, k] in the PyTorch Linear layout, y
    /// [m, n]. Each output element is one sequential length-k reduction, so row i is the same for every m.
    let matmulBf16W<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (x: HostTensor<'T>)
        (w: HostTensor<uint16>)
        : HostTensor<'T> =
        bf16 "w" w

        if w.Shape.Rank <> 2 then
            invalidArg "w" (sprintf "a weight is [n, k], not %O" w.Shape)

        let m = x.Rows
        let k = x.Cols
        let n = w.Rows

        if w.Cols <> k then
            invalidArg "w" (sprintf "x is [%d, %d] but w is %O" m k w.Shape)

        let y = Array.zeroCreate<'T>(m * n)
        let blocks = (n + MatmulColumnBlock - 1) / MatmulColumnBlock
        use _ = span "cpuref.matmul" exec m n

        Par.forEach exec (m * blocks) (fun t ->
            let i = t / blocks
            let j0 = (t % blocks) * MatmulColumnBlock
            let j1 = min n (j0 + MatmulColumnBlock)
            let xi = ReadOnlySpan<'T>(x.Data, i * k, k)

            for j in j0 .. j1 - 1 do
                y[i * n + j] <- Rows.dotBf16 xi (ReadOnlySpan<uint16>(w.Data, j * k, k)))

        result (Shape.matrix m n) y

    /// NeoX RoPE (Rows.ropeHead) of x [T, heads * head_dim] with head_dim = table.HeadDim: row t has position
    /// pos0 + t, which must be inside the table. Same shape as x.
    let ropeApply<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (table: RopeTable)
        (x: HostTensor<'T>)
        (heads: int)
        (pos0: int)
        : HostTensor<'T> =
        let hd = table.HeadDim
        let rows = x.Rows

        if heads < 1 || x.Cols <> heads * hd then
            invalidArg "x" (sprintf "x is %O, not [T, %d * %d]" x.Shape heads hd)

        if pos0 < 0 || pos0 + rows > table.MaxPos then
            raise (
                ArgumentOutOfRangeException(
                    "pos0",
                    sprintf "positions %d .. %d outside the table (max_pos %d)" pos0 (pos0 + rows - 1) table.MaxPos
                )
            )

        let y = Array.zeroCreate<'T> x.Length
        let cols = x.Cols
        use _ = span "cpuref.rope" exec rows cols

        Par.forEach exec rows (fun t ->
            let pos = pos0 + t

            for h in 0 .. heads - 1 do
                let at = t * cols + h * hd

                Rows.ropeHead
                    (ReadOnlySpan<'T>(x.Data, at, hd))
                    (table.CosRow pos)
                    (table.SinRow pos)
                    (Span<'T>(y, at, hd)))

        result x.Shape y

    /// Causal grouped-query attention (Rows.attention per (row, head)). q [rows, QHeads * HeadDim] holds queries after
    /// norm and RoPE; query row i has position past + i and attends keys 0 .. past + i. kCache and vCache are
    /// [Tk, KvHeads * HeadDim] with Tk >= past + rows: K after k_norm and RoPE, V raw, in the mode's dtype. The result
    /// is [rows, QHeads * HeadDim], heads concatenated. A row computed alone (decode) equals the same row inside a
    /// prefill, bit for bit.
    let attention<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (q: HostTensor<'T>)
        (kCache: HostTensor<'T>)
        (vCache: HostTensor<'T>)
        (shape: AttentionShape)
        (past: int)
        (scale: 'T)
        : HostTensor<'T> =
        let hd = shape.HeadDim

        if
            hd < 1
            || shape.QHeads < 1
            || shape.KvHeads < 1
            || shape.QHeads % shape.KvHeads <> 0
        then
            invalidArg "shape" (sprintf "%A is not a GQA layout" shape)

        let rows = q.Rows
        let qCols = shape.QHeads * hd
        let kvCols = shape.KvHeads * hd

        if q.Cols <> qCols then
            invalidArg "q" (sprintf "q is %O, not [rows, %d]" q.Shape qCols)

        if kCache.Cols <> kvCols || vCache.Cols <> kvCols then
            invalidArg "kCache" (sprintf "caches are %O and %O, not [Tk, %d]" kCache.Shape vCache.Shape kvCols)

        if past < 0 || past + rows > kCache.Rows || past + rows > vCache.Rows then
            invalidArg "past" (sprintf "positions %d .. %d need %d cache rows" past (past + rows - 1) (past + rows))

        let group = shape.QHeads / shape.KvHeads
        let y = Array.zeroCreate<'T>(rows * qCols)
        use _ = span "cpuref.attention" exec rows qCols

        Par.forEach exec (rows * shape.QHeads) (fun t ->
            let i = t / shape.QHeads
            let h = t % shape.QHeads
            let keys = past + i + 1
            let scores = Array.zeroCreate<'T> keys

            Rows.attention
                (ReadOnlySpan<'T>(q.Data, i * qCols + h * hd, hd))
                (ReadOnlySpan<'T>(kCache.Data))
                (ReadOnlySpan<'T>(vCache.Data))
                kvCols
                ((h / group) * hd)
                keys
                scale
                (Span<'T>(scores))
                (Span<'T>(y, i * qCols + h * hd, hd)))

        result (Shape.matrix rows qCols) y

    /// silu(gate) * up element-wise (Rows.siluMul). Same shape.
    let siluMul<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (gate: HostTensor<'T>)
        (up: HostTensor<'T>)
        : HostTensor<'T> =
        sameShape "up" gate.Shape up.Shape
        let y = Array.zeroCreate<'T> gate.Length
        use _ = span "cpuref.silu_mul" exec gate.Rows gate.Cols

        Par.forChunks exec gate.Length 16384 (fun lo hi ->
            Rows.siluMul
                (ReadOnlySpan<'T>(gate.Data, lo, hi - lo))
                (ReadOnlySpan<'T>(up.Data, lo, hi - lo))
                (Span<'T>(y, lo, hi - lo)))

        result gate.Shape y

    /// a + b element-wise. Same shape.
    let add<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (a: HostTensor<'T>)
        (b: HostTensor<'T>)
        : HostTensor<'T> =
        sameShape "b" a.Shape b.Shape
        let y = Array.zeroCreate<'T> a.Length
        use _ = span "cpuref.add" exec a.Rows a.Cols

        Par.forChunks exec a.Length 16384 (fun lo hi ->
            Rows.add
                (ReadOnlySpan<'T>(a.Data, lo, hi - lo))
                (ReadOnlySpan<'T>(b.Data, lo, hi - lo))
                (Span<'T>(y, lo, hi - lo)))

        result a.Shape y

    /// Copies `rows` rows of src (from srcRow0) into dst (from dstRow0); both have the same row length. Bit-exact.
    let copyRows<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (src: HostTensor<'T>)
        (srcRow0: int)
        (dst: HostTensor<'T>)
        (dstRow0: int)
        (rows: int)
        =
        let cols = src.Cols

        if dst.Cols <> cols then
            invalidArg "dst" (sprintf "row lengths differ: %d and %d" cols dst.Cols)

        if
            rows < 0
            || srcRow0 < 0
            || dstRow0 < 0
            || srcRow0 + rows > src.Rows
            || dstRow0 + rows > dst.Rows
        then
            raise (ArgumentOutOfRangeException("rows", "rows outside a tensor"))

        Array.Copy(src.Data, srcRow0 * cols, dst.Data, dstRow0 * cols, rows * cols)

    /// Softmax of every row (Rows.softmax). Same shape.
    let softmax<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (x: HostTensor<'T>)
        : HostTensor<'T> =
        rowsOnly "x" x.Shape
        let cols = x.Cols
        let y = Array.zeroCreate<'T> x.Length

        Par.forEach exec x.Rows (fun r ->
            Rows.softmax (ReadOnlySpan<'T>(x.Data, r * cols, cols)) (Span<'T>(y, r * cols, cols)))

        result x.Shape y

    /// logSumExp of every row (Rows.logSumExp): shape [rows].
    let logSumExp<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (x: HostTensor<'T>)
        : HostTensor<'T> =
        rowsOnly "x" x.Shape
        let cols = x.Cols
        let y = Array.zeroCreate<'T> x.Rows
        Par.forEach exec x.Rows (fun r -> y[r] <- Rows.logSumExp (ReadOnlySpan<'T>(x.Data, r * cols, cols)))
        result (Shape.vector x.Rows) y

    /// The lowest index of each row's maximum; raises ArgumentException when a row holds a NaN.
    let argmax<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (x: HostTensor<'T>)
        : int[] =
        rowsOnly "x" x.Shape
        let cols = x.Cols
        Array.init x.Rows (fun r -> Rows.argmax (ReadOnlySpan<'T>(x.Data, r * cols, cols)))

    /// The argmax_rows kernel's semantics: each row's lowest maximal index, or 0xFFFFFFFF for a row holding a NaN.
    let argmaxRows<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (x: HostTensor<'T>)
        : uint32[] =
        rowsOnly "x" x.Shape
        let cols = x.Cols

        Array.init x.Rows (fun r ->
            match Rows.tryArgmax (ReadOnlySpan<'T>(x.Data, r * cols, cols)) with
            | ValueSome i -> uint32 i
            | ValueNone -> UInt32.MaxValue)

    /// The k largest indices of each row (Rows.topK).
    let topK<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (x: HostTensor<'T>)
        (k: int)
        : int[][] =
        rowsOnly "x" x.Shape
        let cols = x.Cols
        Array.init x.Rows (fun r -> Rows.topK (ReadOnlySpan<'T>(x.Data, r * cols, cols)) k)

    /// KL(P || Q) of each row of two logit tensors (Rows.kl): shape [rows].
    let kl<'T
        when 'T :> IFloatingPointIeee754<'T>
        and 'T: unmanaged
        and 'T: struct
        and 'T: (new: unit -> 'T)
        and 'T :> ValueType>
        (exec: CpuExec)
        (p: HostTensor<'T>)
        (q: HostTensor<'T>)
        : HostTensor<'T> =
        sameShape "q" p.Shape q.Shape
        rowsOnly "p" p.Shape
        let cols = p.Cols
        let y = Array.zeroCreate<'T> p.Rows

        Par.forEach exec p.Rows (fun r ->
            y[r] <- Rows.kl (ReadOnlySpan<'T>(p.Data, r * cols, cols)) (ReadOnlySpan<'T>(q.Data, r * cols, cols)))

        result (Shape.vector p.Rows) y
