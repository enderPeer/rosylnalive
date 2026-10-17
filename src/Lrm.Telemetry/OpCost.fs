namespace Lrm.Telemetry

open Lrm.Core

/// The cost of one operation by the FLOP registry's conventions (RunLog contract, "FLOP registry"). Counts are exact
/// integers, so the CPU reference, the CPU device and both GPU backends meter the same work identically.
type OpCost =
    {
        /// Floating-point operations: a multiply-add is 2, exp, rsqrt, division and comparison are 1 each.
        Flops: int64
        /// The minimum bytes read: every input element once.
        BytesRead: int64
        /// The bytes written: every output element once.
        BytesWritten: int64
    }

    member c.BytesMoved = c.BytesRead + c.BytesWritten

    static member Zero =
        {
            Flops = 0L
            BytesRead = 0L
            BytesWritten = 0L
        }

    static member (+)(a: OpCost, b: OpCost) =
        {
            Flops = a.Flops + b.Flops
            BytesRead = a.BytesRead + b.BytesRead
            BytesWritten = a.BytesWritten + b.BytesWritten
        }

/// The operations of the Phase 1 forward pass, with the shapes that set their cost. Activations are fp32 unless a
/// dtype is given.
[<RequireQualifiedAccess>]
type Op =
    /// out[m, n] = x[m, k] . w[n, k]^T: q/k/v/o projections, the MLP and lm_head.
    | MatMul of m: int64 * n: int64 * k: int64 * activation: DType * weight: DType * output: DType
    /// RMSNorm over `rows` rows of `cols` with a weight of `cols`.
    | RmsNorm of rows: int64 * cols: int64 * weight: DType
    /// NeoX RoPE on `rows` positions x `heads` heads of `headDim`, from a cos/sin table.
    | Rope of rows: int64 * heads: int64 * headDim: int64
    /// Causal GQA attention for `rows` new queries at positions past .. past+rows-1 over `past + rows` keys.
    | Attention of rows: int64 * past: int64 * heads: int64 * kvHeads: int64 * headDim: int64
    /// silu(gate) * up over `elements` elements.
    | SiluMul of elements: int64
    /// a + b over `elements` elements (the residual adds).
    | Add of elements: int64
    /// Gathers `rows` embedding rows of `cols` from a table of `weight`, widened to fp32.
    | Embed of rows: int64 * cols: int64 * weight: DType
    /// Copies `rows` rows of `cols` elements of `dtype`.
    | CopyRows of rows: int64 * cols: int64 * dtype: DType
    /// The index of the largest of `cols` values in each of `rows` rows.
    | Argmax of rows: int64 * cols: int64

/// Model dimensions for the transformer-level helpers (Qwen3 layout).
type TransformerDims =
    {
        Hidden: int64
        Heads: int64
        KvHeads: int64
        HeadDim: int64
        Intermediate: int64
        Layers: int64
        Vocab: int64
        /// The dtype of the weights (bf16 for Qwen3).
        Weight: DType
    }

[<RequireQualifiedAccess>]
module TransformerDims =
    /// Qwen3-0.6B (config.json of revision c1899de2).
    let qwen3_0_6b =
        {
            Hidden = 1024L
            Heads = 16L
            KvHeads = 8L
            HeadDim = 128L
            Intermediate = 3072L
            Layers = 28L
            Vocab = 151936L
            Weight = DType.BF16
        }

/// The FLOP registry (RunLog contract): the cost of each operation, by name and by shape.
[<RequireQualifiedAccess>]
module OpCost =
    let private sz (t: DType) = int64 (DType.size t)
    let private f32 = 4L

    /// The registry name of an operation, as used in the `op` attribute of metered records.
    let name (op: Op) =
        match op with
        | Op.MatMul _ -> "matmul"
        | Op.RmsNorm _ -> "rmsnorm"
        | Op.Rope _ -> "rope"
        | Op.Attention _ -> "attention"
        | Op.SiluMul _ -> "silu_mul"
        | Op.Add _ -> "add"
        | Op.Embed _ -> "embed"
        | Op.CopyRows _ -> "copy_rows"
        | Op.Argmax _ -> "argmax"

    /// Every registry name.
    let names =
        [
            "matmul"
            "rmsnorm"
            "rope"
            "attention"
            "silu_mul"
            "add"
            "embed"
            "copy_rows"
            "argmax"
        ]

    /// The (query, key) pairs a causal attention chunk scores: rows*past + rows*(rows+1)/2.
    let causalPairs (rows: int64) (past: int64) = rows * past + rows * (rows + 1L) / 2L

    /// The cost of one operation.
    let ofOp (op: Op) : OpCost =
        match op with
        | Op.MatMul(m, n, k, act, w, out) ->
            {
                Flops = 2L * m * n * k
                BytesRead = m * k * sz act + n * k * sz w
                BytesWritten = m * n * sz out
            }
        | Op.RmsNorm(rows, cols, w) ->
            // Per element: square-accumulate (2), scale (1), weight (1); per row: mean with eps (1), rsqrt (1).
            {
                Flops = 4L * rows * cols + 2L * rows
                BytesRead = rows * cols * f32 + cols * sz w
                BytesWritten = rows * cols * f32
            }
        | Op.Rope(rows, heads, headDim) ->
            // Per rotated pair: 4 multiplies and 2 adds, so 3 per element; the table holds cos and sin per pair.
            let elements = rows * heads * headDim

            {
                Flops = 3L * elements
                BytesRead = elements * f32 + rows * headDim * f32
                BytesWritten = elements * f32
            }
        | Op.Attention(rows, past, heads, kvHeads, headDim) ->
            // Per scored pair and head: q.k (2d), softmax (3: exp, sum, scale), p.v (2d).
            let pairs = causalPairs rows past

            {
                Flops = heads * pairs * (4L * headDim + 3L)
                BytesRead = rows * heads * headDim * f32 + 2L * kvHeads * (past + rows) * headDim * f32
                BytesWritten = rows * heads * headDim * f32
            }
        | Op.SiluMul n ->
            // silu(g) * u = g / (1 + exp(-g)) * u: negate, exp, add, divide, multiply.
            {
                Flops = 5L * n
                BytesRead = 2L * n * f32
                BytesWritten = n * f32
            }
        | Op.Add n ->
            {
                Flops = n
                BytesRead = 2L * n * f32
                BytesWritten = n * f32
            }
        | Op.Embed(rows, cols, w) ->
            {
                Flops = 0L
                BytesRead = rows * cols * sz w + rows * 4L
                BytesWritten = rows * cols * f32
            }
        | Op.CopyRows(rows, cols, t) ->
            {
                Flops = 0L
                BytesRead = rows * cols * sz t
                BytesWritten = rows * cols * sz t
            }
        | Op.Argmax(rows, cols) ->
            {
                Flops = rows * cols
                BytesRead = rows * cols * f32
                BytesWritten = rows * 4L
            }

    /// The total cost of several operations.
    let sum (ops: Op seq) =
        ops |> Seq.fold (fun acc op -> acc + ofOp op) OpCost.Zero

    /// The operations of one Qwen3 decoder layer for `rows` new tokens after `past` cached positions, named by the
    /// ActivationNames tap that each one produces.
    let layerOps (d: TransformerDims) (rows: int64) (past: int64) : (string * Op) list =
        let mm n k =
            Op.MatMul(rows, n, k, DType.F32, d.Weight, DType.F32)

        [
            "input_layernorm", Op.RmsNorm(rows, d.Hidden, d.Weight)
            "q_proj", mm (d.Heads * d.HeadDim) d.Hidden
            "k_proj", mm (d.KvHeads * d.HeadDim) d.Hidden
            "v_proj", mm (d.KvHeads * d.HeadDim) d.Hidden
            "q_norm", Op.RmsNorm(rows * d.Heads, d.HeadDim, d.Weight)
            "k_norm", Op.RmsNorm(rows * d.KvHeads, d.HeadDim, d.Weight)
            "q_rope", Op.Rope(rows, d.Heads, d.HeadDim)
            "k_rope", Op.Rope(rows, d.KvHeads, d.HeadDim)
            "attention", Op.Attention(rows, past, d.Heads, d.KvHeads, d.HeadDim)
            "o_proj", mm d.Hidden (d.Heads * d.HeadDim)
            "resid_mid", Op.Add(rows * d.Hidden)
            "post_attention_layernorm", Op.RmsNorm(rows, d.Hidden, d.Weight)
            "mlp.gate_proj", mm d.Intermediate d.Hidden
            "mlp.up_proj", mm d.Intermediate d.Hidden
            "mlp.silu_mul", Op.SiluMul(rows * d.Intermediate)
            "mlp.down_proj", mm d.Hidden d.Intermediate
            "out", Op.Add(rows * d.Hidden)
        ]

    /// The operations of a whole forward step: embedding, every layer, the final norm, then lm_head and argmax on the
    /// last `headRows` rows (1 for greedy decoding; `rows` when every position's logits are needed).
    let forwardOps (d: TransformerDims) (rows: int64) (past: int64) (headRows: int64) : (string * Op) list =
        [
            yield "embed", Op.Embed(rows, d.Hidden, d.Weight)
            for layer in 0L .. d.Layers - 1L do
                for tap, op in layerOps d rows past do
                    yield "L" + Invariant.i64 layer + "." + tap, op
            yield "final_norm", Op.RmsNorm(rows, d.Hidden, d.Weight)
            yield "lm_head", Op.MatMul(headRows, d.Vocab, d.Hidden, DType.F32, d.Weight, DType.F32)
            yield "argmax", Op.Argmax(headRows, d.Vocab)
        ]

    /// The cost of one forward step (see forwardOps).
    let forward (d: TransformerDims) (rows: int64) (past: int64) (headRows: int64) =
        forwardOps d rows past headRows |> List.map snd |> sum
