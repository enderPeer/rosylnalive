/// K3's conformance cases for attention_gqa_causal (ConformanceHarness contract, "Where cases live"; ADR K3-0001).
///
/// Every case runs through E1's harness: twice with fresh poison and bit-identical, within KernelBounds (relative, with
/// CpuRef F64), with its batch-invariance splits (the first query alone is a decode step at position `past`, the last
/// query alone a decode step at `past + rows - 1`, the middle third a chunked prefill) and under the 100 ms budget on a
/// shader backend. The matrix:
///
/// - **Qwen3-0.6B shapes** (16 query heads, 8 KV heads, head_dim 128): decode at the block edges (positions 0, 63, 64,
///   127), prefill and chunked prefill across block edges, and long context: decode at 4000 and 4095, the last
///   128-query chunk of a 4096-token prefill (keys to 4096) and, GPU-only, its last 512-query chunk. A whole
///   4096-token prefill in one dispatch is measured by the bench (ADR K3-0001), not gated here: the harness times a
///   cold GPU, and that dispatch is too close to the budget before the clocks rise.
/// - **Other layouts**: the tiny model (4:2, head_dim 32), MHA (4:4), MQA (8:1), odd head counts (6:3, 3:1), head_dim 1,
///   3, 64, 96, 130, 200 and 256 (more than one 128-dimension pass), one query of one head, and, GPU-only, a wide grid
///   (16,384 queries of one head; the grid is never folded, ADR K3-0001 has the 65,535-query edge).
/// - **Adversarial magnitudes**: scores in the thousands and near 1e24 (a softmax without the maximum subtracted
///   overflows), scores around -113 everywhere (a softmax without it gives 0 / 0), a running maximum that rises in every
///   block, one dominant key first (later blocks underflow, partly to denormals) and last (every earlier block is
///   earlier block by about exp(-94)), all scores equal (q = 0) over 2,001 keys, tied keys, and values near 1e30, 1e-30
///   and denormal.
///
/// **The causal mask and the rows are poisoned.** Every cache row after the last key a dispatch may read (positions
/// past + rows onwards, 64 of them, so the rest of the last block too) and every q row outside q_row0 .. q_row0 + rows - 1
/// holds the quiet NaN 7fc00000, so a kernel that reads one key past the loop bound, or the wrong query row, returns NaN
/// and fails. y has two spare rows after the last written one and q_row0 spare rows before it, which the harness checks
/// stay untouched.
module Lrm.Kernels.Conformance.Cases.Attention

open System
open Lrm.Core
open Lrm.Kernels.Conformance
open Lrm.Tensor.CpuReference

/// The head layout of a case.
type private Heads =
    {
        QHeads: int
        KvHeads: int
        HeadDim: int
    }

    member h.QCols = h.QHeads * h.HeadDim
    member h.KvCols = h.KvHeads * h.HeadDim

/// Qwen3-0.6B: 16 query heads, 8 KV heads, head_dim 128.
let private qwen =
    {
        QHeads = 16
        KvHeads = 8
        HeadDim = 128
    }

/// The tiny model of goldens/tiny/g1 (4 query heads, 2 KV heads, head_dim 32).
let private tiny =
    {
        QHeads = 4
        KvHeads = 2
        HeadDim = 32
    }

let private heads qh kvh hd =
    {
        QHeads = qh
        KvHeads = kvh
        HeadDim = hd
    }

/// The quiet NaN in every q row and cache row a dispatch must not read.
[<Literal>]
let private Unread = 0x7FC00000u

/// Cache rows after the last readable key that hold the quiet NaN: a whole block, so the rest of the last key block of
/// the last query is poisoned whatever its position.
[<Literal>]
let private UnreadKeys = 64

/// Spare y (and q) rows after the last query row of a dispatch.
[<Literal>]
let private SpareRows = 2

/// The values a dispatch reads: q for its `rows` query rows ([rows, q_heads * head_dim]), and k and v for its `keys`
/// cache rows ([keys, kv_heads * head_dim]).
type private Data = DetRng -> Heads -> int -> int -> float32[] * float32[] * float32[]

/// q, k and v uniform in [-qr, qr), [-kr, kr) and [-vr, vr).
let private uniform (qr: float32) (kr: float32) (vr: float32) : Data =
    fun r h rows keys ->
        let q = CaseGen.uniform r (rows * h.QCols) -qr qr
        let k = CaseGen.uniform r (keys * h.KvCols) -kr kr
        let v = CaseGen.uniform r (keys * h.KvCols) -vr vr
        q, k, v

/// Every element of the key at position `j` set to `value` (all KV heads).
let private setKey (h: Heads) (k: float32[]) (j: int) (value: float32) =
    Array.fill k (j * h.KvCols) h.KvCols value

/// Positive queries in [0.5, 1.5) and keys in [-1, 1), except that the keys at `dominant` are 8.4 in every element: their
/// score is about 95 above every other (q . k_j * scale with q > 0), so every other weight is below 2^-126 and many are
/// denormal or zero.
let private dominantAt (dominant: int -> int list) : Data =
    fun r h rows keys ->
        let q = CaseGen.uniform r (rows * h.QCols) 0.5f 1.5f
        let k = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f
        let v = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f

        for j in dominant keys do
            setKey h k j 8.4f

        q, k, v

/// Scores that rise with the position, about 0.17 per key (q = 0.5, k_j = 0.03 j + noise): the running maximum moves in
/// every block, so every block rescales the state by about exp(-11).
let private rising: Data =
    fun r h rows keys ->
        let q = Array.create (rows * h.QCols) 0.5f
        let noise = CaseGen.uniform r (keys * h.KvCols) -0.05f 0.05f
        let k = noise |> Array.mapi (fun e n -> 0.03f * float32 (e / h.KvCols) + n)
        let v = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f
        q, k, v

/// Scores of about -113 for every key (q in [0.5, 1.5), k in [-11, -9)): exp of a score underflows to 0, so a softmax
/// that does not subtract the maximum divides 0 by 0.
let private negative: Data =
    fun r h rows keys ->
        let q = CaseGen.uniform r (rows * h.QCols) 0.5f 1.5f
        let k = CaseGen.uniform r (keys * h.KvCols) -11.0f -9.0f
        let v = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f
        q, k, v

/// q = 0: every score is exactly 0, every weight exactly 1, and y is the mean of the values.
let private zeroQueries: Data =
    fun r h rows keys ->
        let q = Array.zeroCreate<float32>(rows * h.QCols)
        let k = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f
        let v = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f
        q, k, v

/// Every key equal to the first (random) one: every score of a query and head ties, so the maximum is reached in every
/// block.
let private tiedKeys: Data =
    fun r h rows keys ->
        let q = CaseGen.uniform r (rows * h.QCols) -1.0f 1.0f
        let first = CaseGen.uniform r h.KvCols -1.0f 1.0f
        let k = Array.init (keys * h.KvCols) (fun e -> first[e % h.KvCols])
        let v = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f
        q, k, v

/// Values that are fp32 denormals in [2^-136, 2^-126) with random signs (q and k in [-1, 1)).
let private denormalValues: Data =
    fun r h rows keys ->
        let q = CaseGen.uniform r (rows * h.QCols) -1.0f 1.0f
        let k = CaseGen.uniform r (keys * h.KvCols) -1.0f 1.0f

        let v =
            CaseGen.words r (keys * h.KvCols)
            |> Array.map (fun w ->
                BitConverter.UInt32BitsToSingle((w &&& 0x80000000u) ||| 0x00002000u ||| (w &&& 0x007FFFFFu)))

        q, k, v

/// One case: `rows` queries from q row `q0` at positions past .. past + rows - 1, with q and y holding q0 + rows + 2
/// rows and the caches past + rows + 64 rows; every row the dispatch must not read holds the quiet NaN.
let private case (name: string) (h: Heads) (rows: int) (q0: int) (past: int) (data: Data) (doc: string) : KernelCase =
    let keys = past + rows
    let q, k, v = data (CaseGen.rng ("attention." + name)) h rows keys
    let qCap = q0 + rows + SpareRows
    let kvCap = keys + UnreadKeys

    let place (cap: int) (cols: int) (at: int) (values: float32[]) =
        let words = Array.create (cap * cols) Unread
        let bits = values |> Array.map BitConverter.SingleToUInt32Bits
        Array.blit bits 0 words (at * cols) bits.Length
        words

    Case.create "attention_gqa_causal" name
    |> Case.f32Bits "q" (place qCap h.QCols q0 q)
    |> Case.dims "q" [ int64 qCap; int64 h.QCols ]
    |> Case.f32Bits "k" (place kvCap h.KvCols 0 k)
    |> Case.dims "k" [ int64 kvCap; int64 h.KvCols ]
    |> Case.f32Bits "v" (place kvCap h.KvCols 0 v)
    |> Case.dims "v" [ int64 kvCap; int64 h.KvCols ]
    |> Case.output "y" (int64 qCap * int64 h.QCols)
    |> Case.dims "y" [ int64 qCap; int64 h.QCols ]
    |> Case.u32 "rows" (uint32 rows)
    |> Case.u32 "q_row0" (uint32 q0)
    |> Case.u32 "past" (uint32 past)
    |> Case.u32 "q_heads" (uint32 h.QHeads)
    |> Case.u32 "kv_heads" (uint32 h.KvHeads)
    |> Case.u32 "head_dim" (uint32 h.HeadDim)
    |> Case.f32Arg "scale" (CpuRef.attentionScale<float32> h.HeadDim)
    |> Case.doc doc

let private small = uniform 1.0f 1.0f 1.0f

/// Qwen3-0.6B decode, prefill and chunked prefill, with scores of a few units (q and k in [-2, 2)).
let private qwenCases =
    let d = uniform 2.0f 2.0f 1.0f

    [
        case "qwen_decode_p0" qwen 1 0 0 d "Decode at position 0: one key, one block of one"
        case "qwen_decode_p63" qwen 1 0 63 d "Decode at position 63: one full block"
        case "qwen_decode_p64" qwen 1 0 64 d "Decode at position 64: a full block and a block of one key"
        case "qwen_decode_p127_q5" qwen 1 5 127 d "Decode at position 127 from q row 5: two full blocks"
        case "qwen_prefill_t7" qwen 7 0 0 d "Prefill of 7 tokens (the ABI vector A4's shape)"
        case
            "qwen_prefill_t130"
            qwen
            130
            0
            0
            d
            "Prefill of 130 tokens over 3 key blocks; the splits are decode at 0 and 129 and a chunk at 43 .. 85"
        case "qwen_chunk_t64_past64" qwen 64 0 64 d "The second 64-token chunk of a 128-token prefill"
        case "qwen_chunk_t37_past100_q3" qwen 37 3 100 d "An odd chunk: 37 tokens at 100 .. 136 from q row 3"
        case "qwen_decode_p4000" qwen 1 0 4000 d "Decode at position 4000: 63 key blocks"
        case "qwen_decode_p4095" qwen 1 0 4095 d "Decode at position 4095: 4096 keys, 64 full blocks"
        case
            "qwen_chunk_t128_past3968"
            qwen
            128
            0
            3968
            d
            "The last 128-token chunk of a 4096-token prefill: keys to 4096; decode at 3968 and 4095"
        case
            "qwen_chunk_t512_past3584"
            qwen
            512
            0
            3584
            d
            "The last 512-token chunk of a chunked 4096-token prefill (8,192 groups): keys to 4096"
        |> Case.gpuOnly
    ]

/// Other head layouts and head sizes, and the edges of the grid.
let private layoutCases =
    [
        case "tiny_prefill_t9" tiny 9 0 0 small "The tiny model's layout (4:2, head_dim 32), a 9-token prefill"
        case "tiny_chunk_t6_past70_q2" tiny 6 2 70 small "The tiny model's layout, a chunk at 70 .. 75 from q row 2"
        case
            "mha_4x4_hd64_t20_past50"
            (heads 4 4 64)
            20
            0
            50
            small
            "MHA (4:4), head_dim 64: threads 0 .. 63 own one dim each"
        case
            "mqa_8x1_hd96_t5_past200"
            (heads 8 1 96)
            5
            0
            200
            small
            "MQA (8:1), head_dim 96: half the threads own two dims"
        case "gqa_6x3_hd130_t3_past66" (heads 6 3 130) 3 1 66 small "6:3, head_dim 130: a second pass of two dims"
        case "gqa_3x1_hd3_t11_past129" (heads 3 1 3) 11 0 129 small "3:1, head_dim 3: three threads own a dim"
        case "hd1_t70" (heads 1 1 1) 70 0 0 small "One head of head_dim 1, 70 tokens: two blocks for the last"
        case
            "hd200_t4_past61"
            (heads 2 1 200)
            4
            0
            61
            small
            "head_dim 200: a full and a partial pass, across a block edge"
        case "hd256_t3_past140" (heads 4 2 256) 3 0 140 small "head_dim 256: two full passes"
        case "one_query_one_head" (heads 1 1 128) 1 0 0 small "The smallest dispatch: one query of one head, one key"
        case
            "grid_rows16384"
            (heads 1 1 1)
            16384
            0
            0
            small
            "A wide grid (16,384 groups in x, not folded): the last query attends 16,384 keys in 256 blocks"
        |> Case.gpuOnly
    ]

/// Magnitudes that break a naive softmax, and value ranges at the edges of fp32.
let private adversarialCases =
    [
        case "adv_large_scores" qwen 4 0 300 (uniform 40.0f 40.0f 1.0f) "Scores in the thousands: exp(s) overflows"
        case
            "adv_huge_scores"
            qwen
            3
            0
            150
            (uniform 1e12f 1e12f 1.0f)
            "Scores near 1e24: one weight is 1 and the others underflow to 0"
        case "adv_negative_scores" qwen 3 0 120 negative "Scores of about -113 everywhere: exp(s) underflows to 0"
        case "adv_rising_scores" qwen 3 0 400 rising "The maximum rises in every block: each block rescales by exp(-11)"
        case
            "adv_dominant_first"
            qwen
            3
            0
            200
            (dominantAt (fun _ -> [ 0 ]))
            "Key 0 dominates by about 95: later weights are denormal or 0"
        case
            "adv_dominant_last"
            qwen
            4
            0
            146
            (dominantAt (fun keys -> [ keys - 1 ]))
            "The last key (position 149) dominates: the last query rescales its state by about exp(-94) in its last block"
        case "adv_zero_queries" qwen 3 0 2000 zeroQueries "q = 0: all 2,001 weights are exactly 1; y is the mean of v"
        case "adv_tied_keys" qwen 2 0 300 tiedKeys "Every key equal: the scores of a query tie in every block"
        case "adv_large_values" qwen 3 0 100 (uniform 1.0f 1.0f 1e30f) "Values near 1e30"
        case "adv_small_values" qwen 3 0 100 (uniform 1.0f 1.0f 1e-30f) "Values near 1e-30"
        case "adv_denormal_values" qwen 2 0 70 denormalValues "Denormal values: y is denormal"
    ]

/// Every case of K3.
let cases: KernelCase list = qwenCases @ layoutCases @ adversarialCases
