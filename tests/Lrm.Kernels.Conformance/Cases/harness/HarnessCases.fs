/// The harness's own cases for the K1 to K3 kernels: a few small cases per kernel, so that every catalog kernel runs
/// through the harness on CpuDevice in CI (relative bounds with CpuRef F64, two-dimensional grids, strided rows,
/// in-place aliases, folded row grids, the decode = prefill splits, NaN operands and denormal inputs in every mode). K1
/// to K3 own the real case matrices under Cases/basic, Cases/matmul and Cases/attention; these stay as the harness's
/// regression set. Every name here starts with `e1_`, a prefix the contract reserves for this file ("Where cases
/// live"), so a K1 to K3 case can never take one of these ids (Session.Create refuses a repeated id for the whole
/// session).
module Lrm.Kernels.Conformance.Cases.Harness

open System
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Kernels.Conformance
open Lrm.Tensor.CpuReference

let private rng (name: string) = CaseGen.rng ("harness." + name)

let private bits (xs: float32[]) =
    xs |> Array.map BitConverter.SingleToUInt32Bits

/// fp32 denormals in the upper part of the range, [2^-128, 2^-126), with random signs: a scale of 1000 makes them normal.
let private upperDenormals (r: DetRng) (n: int) =
    CaseGen.words r n
    |> Array.map (fun w -> (w &&& 0x80000000u) ||| 0x00200000u ||| (w &&& 0x001FFFFFu))

/// fp32 values of magnitude [2^e, 2^(e+1)) with random signs and mantissas.
let private binade (r: DetRng) (n: int) (e: int) =
    CaseGen.words r n
    |> Array.map (fun w -> (w &&& 0x80000000u) ||| (uint32 (127 + e) <<< 23) ||| (w &&& 0x007FFFFFu))

let private embedCases =
    let r = rng "embed"
    let vocab, cols = 50, 300
    let table = CaseGen.uniformBf16 r (vocab * cols) -1.0f 1.0f
    // A NaN payload in the table moves bit for bit.
    table[3 * cols + 7] <- 0x7FC3us

    [
        Case.create "embed_gather_bf16" "rows5_cols300"
        |> Case.u32s "ids" [| 3u; 0u; 49u; 3u; 60u |]
        |> Case.bf16 "table" table
        |> Case.output "y" (5L * int64 cols)
        |> Case.u32 "rows" 5u
        |> Case.u32 "cols" (uint32 cols)
        |> Case.u32 "vocab" (uint32 vocab)
        |> Case.doc "Two column groups per row; a repeated id; id 60 >= vocab gives the quiet-NaN row"
    ]

let private rmsnormCases =
    let r = rng "rmsnorm"

    let norm (name: string) (rows: int) (cols: int) (doc: string) =
        Case.create "rmsnorm_rows" name
        |> Case.f32 "x" (CaseGen.uniform r (rows * cols) -3.0f 3.0f)
        |> Case.bf16 "w" (CaseGen.uniformBf16 r cols 0.5f 1.5f)
        |> Case.output "y" (int64 rows * int64 cols)
        |> Case.u32 "rows" (uint32 rows)
        |> Case.u32 "cols" (uint32 cols)
        |> Case.f32Arg "eps" 1e-6f
        |> Case.doc doc

    let cols = 64
    // Row 0 is all denormals: preserve gives normal results (x * 1000), flush gives signed zeros. Row 1 has a few
    // denormals among normal values; w holds two bf16 denormals.
    let x0 = upperDenormals r cols
    let x1 = CaseGen.uniform r cols -3.0f 3.0f |> bits
    let x2 = CaseGen.uniform r cols -3.0f 3.0f |> bits

    for c in [ 5; 17; 40 ] do
        x1[c] <- (upperDenormals r 1)[0]

    let w = CaseGen.uniformBf16 r cols 0.5f 1.5f
    w[3] <- 0x0011us
    w[60] <- 0x8040us

    // x^2 stays normal (|x| >= 2^-63), the mean is about 2.6e-38 and eps = 0x00400000 (5.9e-39) a denormal of a fifth
    // of it: flushing eps changes every result by about a tenth, and nothing becomes 0 / 0 in either mode.
    let tiny =
        CaseGen.uniform r (2 * cols) 1.2e-19f 2.0e-19f
        |> Array.mapi (fun i v -> if i % 3 = 0 then -v else v)

    [
        Case.create "rmsnorm_rows" "denormal_rows"
        |> Case.f32Bits "x" (Array.concat [ x0; x1; x2 ])
        |> Case.bf16 "w" w
        |> Case.output "y" (3L * int64 cols)
        |> Case.u32 "rows" 3u
        |> Case.u32 "cols" (uint32 cols)
        |> Case.f32Arg "eps" 1e-6f
        |> Case.doc "fp32 and bf16 denormal inputs: a flushing class's reference flushes them (row 0 becomes zeros)"
        Case.create "rmsnorm_rows" "denormal_eps"
        |> Case.f32 "x" tiny
        |> Case.bf16 "w" (CaseGen.uniformBf16 r cols 0.5f 1.5f)
        |> Case.output "y" (2L * int64 cols)
        |> Case.u32 "rows" 2u
        |> Case.u32 "cols" (uint32 cols)
        |> Case.f32ArgBits "eps" 0x00400000u
        |> Case.doc "A denormal eps that decides a tenth of every result: a flushing class's reference flushes it"
        norm "rows7_cols300" 7 300 "A reduction longer than the group"
        norm "rows1_cols1024" 1 1024 "Decode: one row of Qwen3's hidden size"
        norm "rows70000_cols2" 70000 2 "70,000 rows: a folded linear grid (35000 x 2)"
        (norm "in_place" 4 64 "y aliases x"
         |> fun c ->
             { c with
                 Io =
                     { c.Io with
                         Tensors = c.Io.Tensors |> List.filter (fun t -> t.Field <> "y")
                     }
             }
             |> Case.inPlace "y" "x")
    ]

let private ropeCases =
    let r = rng "rope"
    let headDim = 32
    let half = headDim / 2

    let invFreq =
        Array.init half (fun i ->
            BitConverter.SingleToUInt32Bits(float32 (1.0 / Math.Pow(10000.0, float (2 * i) / float headDim))))

    let table = RopeTable.create CpuExec.Serial invFreq 16

    let rope (name: string) (tokens: int) (heads: int) (pos0: int) (doc: string) =
        let rows = tokens * heads

        Case.create "rope_apply" name
        |> Case.f32 "x" (CaseGen.uniform r (rows * headDim) -2.0f 2.0f)
        |> Case.f32 "cos_table" table.Cos
        |> Case.f32 "sin_table" table.Sin
        |> Case.output "y" (int64 rows * int64 headDim)
        |> Case.u32 "rows" (uint32 rows)
        |> Case.u32 "heads" (uint32 heads)
        |> Case.u32 "head_dim" (uint32 headDim)
        |> Case.u32 "pos0" (uint32 pos0)
        |> Case.doc doc

    [
        rope "tokens5_heads4_pos3" 5 4 3 "Prefill of 5 tokens from position 3, 4 heads"
        rope "tokens1_heads2_pos15" 1 2 15 "Decode at the table's last position"
    ]

let private siluCases =
    let r = rng "silu"
    let n = 1000

    let silu (name: string) (doc: string) =
        Case.create "silu_mul" name
        |> Case.f32 "gate" (CaseGen.uniform r n -8.0f 8.0f)
        |> Case.f32 "up" (CaseGen.uniform r n -2.0f 2.0f)
        |> Case.output "y" (int64 n)
        |> Case.u32 "n" (uint32 n)
        |> Case.doc doc

    [
        silu "n1000" "Four groups, the last partial"
        (silu "in_place_up" "y aliases up"
         |> fun c ->
             { c with
                 Io =
                     { c.Io with
                         Tensors = c.Io.Tensors |> List.filter (fun t -> t.Field <> "y")
                     }
             }
             |> Case.inPlace "y" "up")
    ]

let private addCases =
    let r = rng "add"

    [
        Case.create "add" "n1000"
        |> Case.f32 "a" (CaseGen.uniform r 1000 -10.0f 10.0f)
        |> Case.f32 "b" (CaseGen.uniform r 1000 -10.0f 10.0f)
        |> Case.output "y" 1000L
        |> Case.u32 "n" 1000u
        |> Case.doc "The residual add"
        Case.create "add" "specials"
        |> Case.f32Bits
            "a"
            [|
                0x7FC00001u
                0x7F800000u
                0x00000001u
                0x80000000u
                0x3F800000u
                0x00400000u
            |]
        |> Case.f32Bits
            "b"
            [|
                0x3F800000u
                0xFF800000u
                0x00000001u
                0x00000000u
                0x7F800001u
                0x80400000u
            |]
        |> Case.output "y" 6L
        |> Case.u32 "n" 6u
        |> Case.doc "NaN payloads, inf - inf, denormal sums and signed zeros"
        Case.create "add" "both_nan"
        |> Case.f32Bits
            "a"
            (Array.init 600 (fun i ->
                [|
                    0x7FC00001u
                    0x7F800001u
                    0xFFC00002u
                    0x3F800000u
                    0x7FC00004u
                    0xFF800007u
                |][i % 6]))
        |> Case.f32Bits
            "b"
            (Array.init 600 (fun i ->
                [|
                    0x7FC00002u
                    0x7FC00003u
                    0x7F800005u
                    0x7FC00006u
                    0x3F800000u
                    0xFFC00008u
                |][i % 6]))
        |> Case.output "y" 600L
        |> Case.u32 "n" 600u
        |> Case.doc
            "Two NaN operands with different payloads: CpuDevice and CpuRef give the first NaN operand, quieted (the NaN rule)"
        Case.create "add" "in_place"
        |> Case.f32 "a" (CaseGen.uniform r 513 -1.0f 1.0f)
        |> Case.f32 "b" (CaseGen.uniform r 513 1.0f 2.0f)
        |> Case.inPlace "y" "a"
        |> Case.u32 "n" 513u
        |> Case.doc "y aliases a"
    ]

let private copyCases =
    let r = rng "copy"

    [
        Case.create "copy_rows" "f32_strided"
        |> Case.f32Bits "src" (CaseGen.words r 60)
        |> Case.output "dst" 60L
        |> Case.u32 "rows" 5u
        |> Case.u32 "cols" 7u
        |> Case.u32 "src_row0" 1u
        |> Case.u32 "dst_row0" 2u
        |> Case.u32 "src_stride" 9u
        |> Case.u32 "dst_stride" 8u
        |> Case.doc "Strided rows into strided rows; random words, NaN patterns included, move bit for bit"
        Case.create "copy_rows" "u32_packed"
        |> Case.u32s "src" (CaseGen.words r 600)
        |> Case.outputOf "dst" DType.U32 700L
        |> Case.u32 "rows" 20u
        |> Case.u32 "cols" 30u
        |> Case.u32 "src_row0" 0u
        |> Case.u32 "dst_row0" 3u
        |> Case.u32 "src_stride" 30u
        |> Case.u32 "dst_stride" 30u
        |> Case.doc "Packed rows (one run), three groups"
        Case.create "copy_rows" "i32_one_row"
        |> Case.i32s "src" (Array.init 300 (fun i -> i - 150))
        |> Case.outputOf "dst" DType.I32 300L
        |> Case.u32 "rows" 1u
        |> Case.u32 "cols" 300u
        |> Case.u32 "src_row0" 0u
        |> Case.u32 "dst_row0" 0u
        |> Case.u32 "src_stride" 0u
        |> Case.u32 "dst_stride" 0u
        |> Case.doc "One row (strides unused)"
    ]

let private argmaxCases =
    let r = rng "argmax"
    let cols = 300
    let x = CaseGen.uniform r (6 * cols) -5.0f 5.0f |> bits
    // Row 1: a tie at the maximum, the lowest index wins.
    x[1 * cols + 10] <- 0x41200000u
    x[1 * cols + 200] <- 0x41200000u
    // Row 2: a NaN gives 0xFFFFFFFF.
    x[2 * cols + 50] <- 0x7FC00000u
    // Row 3: -0 and +0 tie at the maximum.
    for c in 0 .. cols - 1 do
        x[3 * cols + c] <- 0xBF800000u

    x[3 * cols + 5] <- 0x80000000u
    x[3 * cols + 6] <- 0x00000000u
    // Row 4: every element -inf.
    for c in 0 .. cols - 1 do
        x[4 * cols + c] <- 0xFF800000u

    [
        Case.create "argmax_rows" "rows6_specials"
        |> Case.f32Bits "x" x
        |> Case.outputU32 "y" 6L
        |> Case.u32 "rows" 6u
        |> Case.u32 "cols" (uint32 cols)
        |> Case.doc "Ties take the lowest index, a NaN row gives 0xFFFFFFFF, -0 ties +0, an all -inf row gives 0"
        Case.create "argmax_rows" "rows70000_cols2"
        |> Case.f32 "x" (CaseGen.uniform r 140000 -1.0f 1.0f)
        |> Case.outputU32 "y" 70000L
        |> Case.u32 "rows" 70000u
        |> Case.u32 "cols" 2u
        |> Case.doc "A folded linear grid"
    ]

let private matmulCases =
    let r = rng "matmul"

    let mm (name: string) (m: int) (n: int) (k: int) (doc: string) =
        Case.create "matmul_bf16w" name
        |> Case.f32
            "x"
            (if m * k = 0 then
                 [| 0.0f |]
             else
                 CaseGen.uniform r (m * k) -1.0f 1.0f)
        |> Case.bf16
            "w"
            (if n * k = 0 then
                 [| 0us |]
             else
                 CaseGen.uniformBf16 r (n * k) -1.0f 1.0f)
        |> Case.output "y" (int64 m * int64 n)
        |> Case.u32 "m" (uint32 m)
        |> Case.u32 "n" (uint32 n)
        |> Case.u32 "k" (uint32 k)
        |> Case.doc doc

    let k = 8
    // Row 0 of x is fp32 denormals, row 1 is about 2^30; w's rows are about 2^30, bf16 denormals and about 1. The
    // products 2^-128 * 2^30 and 2^30 * 2^-129 are normal, so preserve and flush give different results there.
    let x = Array.append (upperDenormals r k) (binade r k 30)

    let w =
        Array.concat
            [
                CaseGen.words r k
                |> Array.map (fun v -> uint16 ((v &&& 0x8000u) ||| (uint32 (127 + 30) <<< 7) ||| (v &&& 0x7Fu)))
                CaseGen.words r k
                |> Array.map (fun v -> uint16 ((v &&& 0x8000u) ||| 0x10u ||| (v &&& 0x6Fu)))
                CaseGen.uniformBf16 r k -1.0f 1.0f
            ]

    [
        Case.create "matmul_bf16w" "denormals"
        |> Case.f32Bits "x" x
        |> Case.bf16 "w" w
        |> Case.output "y" 6L
        |> Case.u32 "m" 2u
        |> Case.u32 "n" 3u
        |> Case.u32 "k" (uint32 k)
        |> Case.doc
            "fp32 and bf16 denormal inputs whose products are normal: a flushing class's reference flushes the inputs"
        mm "m37_n70_k64" 37 70 64 "Two tiles in n; the M = 1 splits are decode rows"
        mm "m1_n130_k256" 1 130 256 "Decode: one row, three tiles"
        mm "m3_n5_k0" 3 5 0 "k = 0: every output is +0"
    ]

let private attentionCases =
    let r = rng "attention"

    let attn
        (name: string)
        (rows: int)
        (q0: int)
        (past: int)
        (qh: int)
        (kvh: int)
        (hd: int)
        (maxPos: int)
        (doc: string)
        =
        let qc = qh * hd
        let kvc = kvh * hd

        Case.create "attention_gqa_causal" name
        |> Case.f32 "q" (CaseGen.uniform r ((q0 + rows) * qc) -1.0f 1.0f)
        |> Case.f32 "k" (CaseGen.uniform r (maxPos * kvc) -1.0f 1.0f)
        |> Case.f32 "v" (CaseGen.uniform r (maxPos * kvc) -1.0f 1.0f)
        |> Case.output "y" (int64 (q0 + rows) * int64 qc)
        |> Case.u32 "rows" (uint32 rows)
        |> Case.u32 "q_row0" (uint32 q0)
        |> Case.u32 "past" (uint32 past)
        |> Case.u32 "q_heads" (uint32 qh)
        |> Case.u32 "kv_heads" (uint32 kvh)
        |> Case.u32 "head_dim" (uint32 hd)
        |> Case.f32Arg "scale" (CpuRef.attentionScale<float32> hd)
        |> Case.doc doc

    [
        attn "prefill5_past3_q0_1" 5 1 3 4 2 16 10 "GQA 4:2, a chunk of 5 queries at positions 3..7, q rows from 1"
        attn "decode_past70" 1 0 70 2 1 8 80 "Decode at position 70: two 64-key blocks"
    ]

/// The prefix the contract reserves for the names of this file's cases.
[<Literal>]
let Prefix = "e1_"

/// Every harness case, each name with the reserved prefix.
let cases: KernelCase list =
    embedCases
    @ rmsnormCases
    @ ropeCases
    @ siluCases
    @ addCases
    @ copyCases
    @ argmaxCases
    @ matmulCases
    @ attentionCases
    |> List.map (fun c -> { c with Name = Prefix + c.Name })
