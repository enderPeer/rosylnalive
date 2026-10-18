/// K1's conformance cases for the seven basic kernels (ConformanceHarness contract, "Where cases live"): embed_gather_bf16,
/// rmsnorm_rows, rope_apply, silu_mul, add, copy_rows and argmax_rows. Each kernel gets the Qwen3-0.6B shapes (hidden
/// 1024, q 2048, kv 1024, intermediate 3072, vocab 151,936, head_dim 128, 16 q heads and 8 kv heads) at decode (T = 1)
/// and prefill (T = 37, 512) sizes, and small and edge shapes: one row, odd sizes, partial groups, outputs whose
/// capacity exceeds the work, in-place aliases, folded grids above 65,535 groups, ties, NaN and denormals. The harness
/// runs each case twice with fresh poison, judges it by KernelBounds, splits it for batch invariance and times it.
///
/// Cases too large to emulate in CI (the 311 MB Qwen embedding table, and folded grids of Qwen's shapes, which need
/// more than 16.7 million elements for the element-wise kernels) are `Case.gpuOnly`: GPU backends run them, CpuDevice
/// never plans them. They are built only outside the cpu tier (see `large` at the end).
module Lrm.Kernels.Conformance.Cases.Basic

open System
open System.Buffers.Binary
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Kernels.Conformance
open Lrm.Tensor.CpuReference
open Lrm.Testing

// ---- Qwen3-0.6B (config.json of revision c1899de2) -----------------------------------------------------------------

[<Literal>]
let Hidden = 1024

[<Literal>]
let QDim = 2048

[<Literal>]
let KvDim = 1024

[<Literal>]
let Intermediate = 3072

[<Literal>]
let Vocab = 151936

[<Literal>]
let HeadDim = 128

[<Literal>]
let QHeads = 16

[<Literal>]
let KvHeads = 8

/// max_position_embeddings.
[<Literal>]
let MaxPositions = 40960

/// rms_norm_eps.
let Eps = 1e-6f

/// torch's float32 inv_freq for theta 1,000,000 and head_dim 128, as u32 bits (ReferenceOracles contract, "T_spec";
/// the planning probe redteam/inv_torch.txt). Checked against the contract's SHA-256 below.
let qwenInvFreq =
    [|
        0x3f800000u
        0x3f4e4badu
        0x3f263de0u
        0x3f05f6efu
        0x3ed7e89bu
        0x3eadfcffu
        0x3e8c3504u
        0x3e61f835u
        0x3e361887u
        0x3e12bd91u
        0x3dec7fd6u
        0x3dbe94c6u
        0x3d99940du
        0x3d778513u
        0x3d47763fu
        0x3d20bc1du
        0x3d0186e3u
        0x3cd0c1a8u
        0x3ca8398bu
        0x3c879008u
        0x3c5a7bf2u
        0x3c301052u
        0x3c0de12du
        0x3be4aa46u
        0x3bb8449cu
        0x3b947daeu
        0x3b6f520eu
        0x3b40dac5u
        0x3b1b690du
        0x3afa78f0u
        0x3ac9d75cu
        0x3aa2a6f7u
        0x3a83126fu
        0x3a533f28u
        0x3a2a3b44u
        0x3a092e02u
        0x39dd1725u
        0x39b229fbu
        0x398f9272u
        0x39676492u
        0x393a7753u
        0x39164324u
        0x38f22ce2u
        0x38c327b5u
        0x389d43a4u
        0x387d75d4u
        0x384c3fbeu
        0x382497abu
        0x3804a2b2u
        0x37d5c441u
        0x37ac431du
        0x378ad0edu
        0x375fba4fu
        0x37344a0eu
        0x371148e3u
        0x36ea2732u
        0x36bcb0c1u
        0x36980e02u
        0x36751070u
        0x36457babu
        0x361f23e4u
        0x36003decu
        0x35ceaf79u
        0x35a68e4cu
    |]

// ---- data -----------------------------------------------------------------------------------------------------------

let private rng (name: string) = CaseGen.rng ("basic." + name)

let private bits (xs: float32[]) =
    xs |> Array.map BitConverter.SingleToUInt32Bits

let private f32 (x: float32) = BitConverter.SingleToUInt32Bits x

/// The seed of a large input: the first 8 bytes of SHA-256 of its name.
let private seedOf (name: string) =
    BinaryPrimitives.ReadUInt64LittleEndian(ReadOnlySpan((Hash.sha256Utf8 ("lrm.conformance.basic/" + name)).ToBytes()))

/// Word i of a large input: the high half of SplitMix64's output function at seed + i * gamma. One mix per word and no
/// sequential state, so inputs of millions of elements cost little to build (CaseGen's DetRng is for small inputs).
let inline private wordAt (seed: uint64) (i: int) =
    uint32 (SplitMix64.Mix(seed + uint64 i * SplitMix64.Gamma) >>> 32)

/// Uniform float32 values in [lo, hi) from wordAt, each a multiple of 2^-24 of the span (as CaseGen.uniform).
let private bigUniform (name: string) (n: int) (lo: float32) (hi: float32) =
    let seed = seedOf name
    Array.init n (fun i -> lo + (hi - lo) * (float32 (wordAt seed i >>> 8) * 5.9604644775390625e-8f))

/// A bf16 table of n elements written straight into its bytes, so the 311 MB Qwen table exists once: finite values of
/// magnitude [2^-7, 2^-3), small as embedding entries are, with random signs and mantissas.
let private bigBf16Table (name: string) (n: int64) =
    let seed = seedOf name
    let bytes = Array.zeroCreate<byte>(int (2L * n))
    let h = MemoryMarshal.Cast<byte, uint16>(Span bytes)

    for i in 0 .. h.Length - 1 do
        let w = wordAt seed i
        h[i] <- uint16 ((w &&& 0x8000u) ||| ((120u + (w >>> 16) % 4u) <<< 7) ||| (w &&& 0x7Fu))

    bytes

/// Hidden states with a few outlier features: uniform in [-2, 2), and three columns 200 times larger, so one column
/// dominates each row's sum of squares as massive activations do in a residual stream.
let private hiddenStates (r: DetRng) (rows: int) (cols: int) =
    let x = CaseGen.uniform r (rows * cols) -2.0f 2.0f

    for row in 0 .. rows - 1 do
        for c in [ 7; 400; 911 ] do
            if c < cols then
                x[row * cols + c] <- 200.0f * x[row * cols + c]

    x

/// Some bf16 bit patterns that must move unchanged: denormals, NaN payloads (quiet and signalling), infinities, signed
/// zeros and the largest finite values.
let private bf16Specials =
    [|
        0x0001us
        0x807Fus
        0x0040us
        0x7FC1us
        0xFF81us
        0x7F80us
        0xFF80us
        0x8000us
        0x0000us
        0x7F7Fus
        0xFF7Fus
        0x3F80us
    |]

// ---- T_spec ---------------------------------------------------------------------------------------------------------

let private hexOfWords (ws: uint32[]) =
    let bytes = Array.zeroCreate<byte>(4 * ws.Length)

    for i in 0 .. ws.Length - 1 do
        BinaryPrimitives.WriteUInt32LittleEndian(Span(bytes, 4 * i, 4), ws[i])

    (Hash.sha256 bytes).Hex

/// T_spec at (pos, j) for the pinned bits: freq = f32(f32(pos) * inv_freq[j]), cos and sin = f32(RN64(cos/sin freq)),
/// computed by CpuRef's exact trigonometry (ReferenceOracles, "T_spec"; ADR O2-0001).
let private tspecEntry (inv: uint32[]) (pos: int) (j: int) =
    let struct (c, s) = ExactTrig.cosSin (RopeTable.angle pos inv[j])
    f32 (float32 c), f32 (float32 s)

/// The inverse frequencies, checked once against the contract's SHA-256 and T_spec entry vectors: a wrong word here
/// fails every case of this file at discovery, loudly.
let private invFreq =
    lazy
        (let hash = hexOfWords qwenInvFreq

         if hash <> "138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e" then
             failwithf "the Qwen3 inv_freq bits hash to %s, not the ReferenceOracles value" hash

         for pos, j, cos, sin in
             [
                 4095, 0, 0xbd871e6cu, 0xbf7f7136u
                 4095, 63, 0x3f7fff27u, 0x3ba683b4u
                 8191, 1, 0xbf7c693cu, 0xbe2add27u
             ] do
             let c, s = tspecEntry qwenInvFreq pos j

             if c <> cos || s <> sin then
                 failwithf "T_spec(%d, %d) is (%08x, %08x), the contract says (%08x, %08x)" pos j c s cos sin

         qwenInvFreq)

/// cos and sin tables (u32 bits) of `capacity` positions of inv.Length entries each: T_spec at positions pos0 ..
/// pos0 + count - 1, and quiet NaNs (0x7FC1pppp and 0x7FC2pppp, pppp the position) at every other position, so a
/// kernel that reads a wrong position gives NaN results, which every relative check fails.
let private tspecTables (inv: uint32[]) (pos0: int) (count: int) (capacity: int) =
    let half = inv.Length
    let cos = Array.zeroCreate<uint32>(capacity * half)
    let sin = Array.zeroCreate<uint32>(capacity * half)

    for pos in 0 .. capacity - 1 do
        for j in 0 .. half - 1 do
            let at = pos * half + j

            if pos >= pos0 && pos < pos0 + count then
                let c, s = tspecEntry inv pos j
                cos[at] <- c
                sin[at] <- s
            else
                cos[at] <- 0x7FC10000u ||| (uint32 pos &&& 0xFFFFu)
                sin[at] <- 0x7FC20000u ||| (uint32 pos &&& 0xFFFFu)

    cos, sin

// ---- embed_gather_bf16 ----------------------------------------------------------------------------------------------

let private embed
    (name: string)
    (ids: uint32[])
    (table: byte[])
    (cols: int)
    (vocab: int)
    (capacity: int64)
    (doc: string)
    =
    Case.create "embed_gather_bf16" name
    |> Case.u32s "ids" ids
    |> Case.inputBytes "table" DType.BF16 table
    |> Case.output "y" capacity
    |> Case.u32 "rows" (uint32 ids.Length)
    |> Case.u32 "cols" (uint32 cols)
    |> Case.u32 "vocab" (uint32 vocab)
    |> Case.doc doc

let private bf16Bytes (h: uint16[]) =
    MemoryMarshal.AsBytes(ReadOnlySpan h).ToArray()

/// Random ids below vocab, with the given ids first.
let private idsWith (r: DetRng) (first: uint32 list) (rows: int) (vocab: int) =
    Array.append (Array.ofList first) (Array.init (rows - first.Length) (fun _ -> r.NextBelow(uint32 vocab)))

/// The real shape: the whole Qwen table (every id's row is 1024 bf16; rows 1 and 151,935 start with the special
/// patterns) and 37 ids.
let private embedLarge () =
    let r = rng "embed.large"
    let table = bigBf16Table "embed.qwen_table" (int64 Vocab * int64 Hidden)
    let h = MemoryMarshal.Cast<byte, uint16>(Span table)

    for row in [ 1; Vocab - 1 ] do
        for j in 0 .. bf16Specials.Length - 1 do
            h[row * Hidden + 3 * j] <- bf16Specials[j]

    let ids =
        idsWith
            r
            [
                0u
                uint32 Vocab - 1u
                1u
                uint32 Vocab - 2u
                uint32 Vocab
                0xFFFFFFFFu
                0x80000000u
                75968u
                uint32 Vocab - 1u
                0u
            ]
            37
            Vocab

    [
        embed
            "qwen_vocab_prefill37"
            ids
            table
            Hidden
            Vocab
            (37L * int64 Hidden)
            "The real shape: a 151,936 x 1024 table (311 MB), ids 0 and 151,935 and three out of range (NaN rows); splits give the decode rows"
        |> Case.gpuOnly
    ]

let private embedCases =
    let r = rng "embed"

    let small (vocab: int) (cols: int) =
        let t = CaseGen.uniformBf16 r (vocab * cols) -0.1f 0.1f

        for j in 0 .. min (bf16Specials.Length - 1) (t.Length - 1) do
            t[j * 7 % t.Length] <- bf16Specials[j]

        bf16Bytes t

    // Every bf16 pattern once: 256 rows of 256 columns, row i holding patterns 256 i .. 256 i + 255.
    let allPatterns = bf16Bytes (Array.init 65536 uint16)

    let permutation =
        let p = Array.init 256 uint32

        for i in 255..-1..1 do
            let j = int (r.NextBelow(uint32 (i + 1)))
            let t = p[i]
            p[i] <- p[j]
            p[j] <- t

        p

    [
        embed
            "qwen_vocab_cols4"
            (idsWith r [ uint32 Vocab - 1u; uint32 Vocab; 0u; 0xFFFFFFFFu; 0x80000000u; 151000u ] 9 Vocab)
            (small Vocab 4)
            4
            Vocab
            40L
            "The whole vocabulary with 4 columns: ids at both ends, three out of range; capacity beyond the work"
        embed
            "qwen_hidden_decode"
            [| 4095u |]
            (small 4096 Hidden)
            Hidden
            4096
            (int64 Hidden)
            "Decode: one id, Qwen's hidden size (4 column groups)"
        embed
            "qwen_hidden_prefill37"
            (idsWith r [ 0u; 4095u; 4096u; 17u; 17u ] 37 4096)
            (small 4096 Hidden)
            Hidden
            4096
            (37L * int64 Hidden)
            "Prefill: 37 ids, Qwen's hidden size, a repeated id and one out of range"
        embed "cols1" [| 2u; 0u; 9u; 10u |] (small 10 1) 1 10 4L "One column: 255 idle threads per group; id 10 = vocab"
        embed
            "cols257"
            (idsWith r [ 6u ] 3 7)
            (small 7 257)
            257
            7
            (3L * 257L + 5L)
            "Two column groups, the second with one column; capacity beyond the work"
        embed "cols255_rows5" (idsWith r [] 5 11) (small 11 255) 255 11 (5L * 255L) "A partial column group"
        embed
            "rows65535_cols3"
            (idsWith r [ 299u; 300u; 0u ] 65535 300)
            (small 300 3)
            3
            300
            (65535L * 3L)
            "65,535 rows: the most a grid's y holds (embedding grids never fold)"
        embed
            "all_bf16_patterns"
            permutation
            allPatterns
            256
            256
            65536L
            "Every one of the 65,536 bf16 patterns moves bit for bit: NaN payloads, denormals, infinities, signed zeros"
    ]

// ---- rmsnorm_rows ---------------------------------------------------------------------------------------------------

let private rmsnorm
    (name: string)
    (x: float32[])
    (w: uint16[])
    (rows: int)
    (cols: int)
    (capacity: int64)
    (doc: string)
    =
    Case.create "rmsnorm_rows" name
    |> Case.f32 "x" x
    |> Case.bf16 "w" w
    |> Case.output "y" capacity
    |> Case.u32 "rows" (uint32 rows)
    |> Case.u32 "cols" (uint32 cols)
    |> Case.f32Arg "eps" Eps
    |> Case.doc doc

let private withoutOutput (c: KernelCase) =
    { c with
        Io =
            { c.Io with
                Tensors = c.Io.Tensors |> List.filter (fun t -> t.Field <> "y")
            }
    }

let private rmsnormCases =
    let r = rng "rmsnorm"
    let weight (n: int) = CaseGen.uniformBf16 r n 0.05f 4.0f

    let hidden (name: string) (rows: int) (doc: string) =
        rmsnorm name (hiddenStates r rows Hidden) (weight Hidden) rows Hidden (int64 rows * int64 Hidden) doc

    let heads (name: string) (tokens: int) (h: int) (doc: string) =
        let rows = tokens * h

        rmsnorm
            name
            (CaseGen.uniform r (rows * HeadDim) -6.0f 6.0f)
            (weight HeadDim)
            rows
            HeadDim
            (int64 rows * int64 HeadDim)
            doc

    let plain (name: string) (rows: int) (cols: int) (doc: string) =
        rmsnorm
            name
            (CaseGen.uniform r (rows * cols) -3.0f 3.0f)
            (weight cols)
            rows
            cols
            (int64 rows * int64 cols + 3L)
            doc

    // Rows with zeros: one row all +0, one all -0, so mean = 0 and r = 1 / sqrt(eps); the others normal.
    let zeros =
        let cols = 64
        let x = CaseGen.uniform r (4 * cols) -1.0f 1.0f

        for c in 0 .. cols - 1 do
            x[cols + c] <- 0.0f
            x[2 * cols + c] <- -0.0f

        rmsnorm "zero_rows" x (weight cols) 4 cols (4L * int64 cols) "Rows of +0 and -0: signed zeros out"

    [
        hidden "qwen_hidden_decode" 1 "input_layernorm at decode: one row of 1024, four columns per thread"
        hidden "qwen_hidden_prefill37" 37 "input_layernorm over a 37-token prefill, with massive activations"
        hidden "qwen_hidden_prefill512" 512 "A 512-token prefill chunk"
        heads "qwen_q_norm_decode" 1 QHeads "q_norm at decode: 16 heads of 128, half the threads idle"
        heads "qwen_q_norm_prefill37" 37 QHeads "q_norm over a 37-token prefill: 592 rows of 128"
        heads "qwen_k_norm_prefill37" 37 KvHeads "k_norm over a 37-token prefill: 296 rows of 128"
        (hidden "qwen_final_norm_in_place" 37 "The final norm in place: y aliases x"
         |> withoutOutput
         |> Case.inPlace "y" "x")
        plain "cols1_rows5" 5 1 "One column: y = w * x / sqrt(x^2 + eps); 255 idle threads"
        plain "cols255" 3 255 "One column short of a full group"
        plain "cols257" 3 257 "One column past a full group: thread 0 takes two"
        plain "cols1000_rows3" 3 1000 "An odd row length"
        plain "cols3072" 2 Intermediate "Qwen's intermediate size: 12 columns per thread"
        plain "cols8191_rows2" 2 8191 "A long odd row: 32 columns per thread, the last thread 31"
        zeros
        plain "rows65600_cols4" 65600 4 "65,600 rows: a folded grid (32800 x 2)"
    ]

let private rmsnormLarge () =
    let r = rng "rmsnorm.large"

    [
        rmsnorm
            "qwen_k_norm_8193_tokens_in_place"
            (bigUniform "rmsnorm.k_norm_8193" (8193 * KvHeads * HeadDim) -6.0f 6.0f)
            (CaseGen.uniformBf16 r HeadDim 0.05f 4.0f)
            (8193 * KvHeads)
            HeadDim
            0L
            "k_norm of an 8,193-token prefill in place: 65,544 rows, a folded grid of Qwen's shape"
        |> withoutOutput
        |> Case.inPlace "y" "x"
        |> Case.gpuOnly
    ]

// ---- rope_apply -----------------------------------------------------------------------------------------------------

let private ropeCase
    (name: string)
    (x: float32[])
    (cos: uint32[])
    (sin: uint32[])
    (rows: int)
    (heads: int)
    (headDim: int)
    (pos0: int)
    (doc: string)
    =
    Case.create "rope_apply" name
    |> Case.f32 "x" x
    |> Case.f32Bits "cos_table" cos
    |> Case.f32Bits "sin_table" sin
    |> Case.output "y" (int64 rows * int64 headDim)
    |> Case.u32 "rows" (uint32 rows)
    |> Case.u32 "heads" (uint32 heads)
    |> Case.u32 "head_dim" (uint32 headDim)
    |> Case.u32 "pos0" (uint32 pos0)
    |> Case.doc doc

let private ropeCases =
    let r = rng "rope"

    /// A Qwen case from T_spec: tokens * heads rows of 128 at positions pos0 .. pos0 + tokens - 1, the table two
    /// positions longer than needed (NaN rows before pos0 and after the last position).
    let qwen (name: string) (tokens: int) (heads: int) (pos0: int) (doc: string) =
        let rows = tokens * heads
        let cos, sin = tspecTables (invFreq.Force()) pos0 tokens (pos0 + tokens + 2)
        ropeCase name (CaseGen.uniform r (rows * HeadDim) -4.0f 4.0f) cos sin rows heads HeadDim pos0 doc

    /// A case with a synthetic table: inverse frequencies 10000^(-2j / head_dim) rounded to float32 (any table
    /// exercises a table-driven kernel), T_spec's formula for the entries.
    let synthetic (name: string) (tokens: int) (heads: int) (headDim: int) (pos0: int) (doc: string) =
        let half = headDim / 2

        let inv =
            Array.init half (fun j -> f32 (float32 (Math.Pow(10000.0, -2.0 * float j / float headDim))))

        let rows = tokens * heads
        let cos, sin = tspecTables inv pos0 tokens (pos0 + tokens + 1)
        ropeCase name (CaseGen.uniform r (rows * headDim) -2.0f 2.0f) cos sin rows heads headDim pos0 doc

    let inPlace (c: KernelCase) =
        c |> withoutOutput |> Case.inPlace "y" "x"

    [
        qwen "qwen_q_decode_pos0" 1 QHeads 0 "Decode at position 0: cos 1 and sin 0, y = x"
        qwen "qwen_q_decode_pos8191" 1 QHeads 8191 "Decode at position 8191, the end of G2's dumped T_spec"
        qwen
            "qwen_k_decode_pos40959"
            1
            KvHeads
            40959
            "k at the last position Qwen3 allows (max_position_embeddings - 1)"
        qwen "qwen_q_prefill37_pos0" 37 QHeads 0 "q over a 37-token prefill from position 0: 592 rows"
        qwen "qwen_k_prefill37_pos512" 37 KvHeads 512 "k over a chunked prefill of 37 tokens from position 512"
        (qwen "qwen_q_prefill512_in_place" 512 QHeads 0 "q over a 512-token prefill chunk, in place: 8,192 rows"
         |> inPlace)
        (qwen "qwen_k_prefill7_pos3000_in_place" 7 KvHeads 3000 "k in place at positions 3000 to 3006"
         |> inPlace)
        synthetic "head_dim2_heads1" 5 1 2 0 "The smallest head: one pair, 63 idle threads"
        synthetic "head_dim6_heads5_pos7" 3 5 6 7 "Odd heads and an odd half (3)"
        synthetic "head_dim130_heads3" 4 3 130 2 "A half of 65 pairs: thread 0 rotates pairs 0 and 64"
        synthetic "head_dim256_heads2" 3 2 256 11 "A half of 128 pairs: two per thread"
        synthetic "heads64_head_dim16_tokens1025" 1025 64 16 100 "65,600 rows: a folded grid (32800 x 2)"
    ]

/// The folded Qwen grid: 4,097 tokens of 16 q heads, 65,552 rows (folded to 32776 x 2), in place. Its table is random
/// unit pairs, not T_spec, because 4,097 exact T_spec positions would cost seconds to build.
let private ropeLarge () =
    let tokens = 4097
    let rows = tokens * QHeads
    let half = HeadDim / 2
    let angles = bigUniform "rope.folded_angles" (tokens * half) -3.14159f 3.14159f
    let cos = angles |> Array.map (fun a -> f32 (float32 (Math.Cos(float a))))
    let sin = angles |> Array.map (fun a -> f32 (float32 (Math.Sin(float a))))

    [
        ropeCase
            "qwen_q_4097_tokens_in_place"
            (bigUniform "rope.folded_x" (rows * HeadDim) -4.0f 4.0f)
            cos
            sin
            rows
            QHeads
            HeadDim
            0
            "q of a 4,097-token prefill in place: 65,552 rows, a folded grid of Qwen's shape (random unit table)"
        |> withoutOutput
        |> Case.inPlace "y" "x"
        |> Case.gpuOnly
    ]

// ---- silu_mul -------------------------------------------------------------------------------------------------------

let private silu (name: string) (gate: float32[]) (up: float32[]) (capacity: int64) (doc: string) =
    Case.create "silu_mul" name
    |> Case.f32 "gate" gate
    |> Case.f32 "up" up
    |> Case.output "y" capacity
    |> Case.u32 "n" (uint32 gate.Length)
    |> Case.doc doc

let private siluCases =
    let r = rng "silu"

    let mlp (name: string) (n: int) (doc: string) =
        silu name (CaseGen.uniform r n -8.0f 8.0f) (CaseGen.uniform r n -4.0f 4.0f) (int64 n) doc

    // Gates over the whole range: exp(-g) overflows to +inf for g below about -88.7 (silu gives a signed zero), and is
    // a denormal for g near 88.7; tiny gates give g / 2; signed zeros give signed zeros.
    let wide =
        let n = 4096
        let g = CaseGen.uniform r n -100.0f 100.0f

        let specials =
            [|
                -88.72f
                -88.73f
                -89.0f
                -103.0f
                88.72f
                87.0f
                1e-30f
                -1e-30f
                0.0f
                -0.0f
                1.0f
                -1.0f
            |]

        Array.blit specials 0 g 0 specials.Length

        silu
            "wide_range"
            g
            (CaseGen.uniform r n -4.0f 4.0f)
            (int64 n + 7L)
            "Gates from -100 to 100: exp overflow, tiny and zero gates; capacity beyond n"

    [
        mlp "qwen_mlp_decode" Intermediate "Decode: one row of Qwen's intermediate size (12 groups)"
        mlp "qwen_mlp_prefill37" (37 * Intermediate) "A 37-token prefill: 113,664 elements"
        mlp "qwen_mlp_prefill512" (512 * Intermediate) "A 512-token prefill chunk: 1,572,864 elements"
        (mlp "qwen_mlp_in_place_gate" (37 * Intermediate) "y aliases gate"
         |> withoutOutput
         |> Case.inPlace "y" "gate")
        wide
        mlp "n1" 1 "One element"
        mlp "n255" 255 "One group, one thread idle"
        mlp "n257" 257 "Two groups, the second with one element"
    ]

let private siluLarge () =
    let n = 65537 * 256

    [
        silu
            "folded_n16777472"
            (bigUniform "silu.folded_gate" n -8.0f 8.0f)
            (bigUniform "silu.folded_up" n -4.0f 4.0f)
            (int64 n)
            "65,537 groups: folded to 32769 x 2, so one extra group must write nothing"
        |> Case.gpuOnly
    ]

// ---- add ------------------------------------------------------------------------------------------------------------

let private addCase (name: string) (a: uint32[]) (b: uint32[]) (capacity: int64) (doc: string) =
    Case.create "add" name
    |> Case.f32Bits "a" a
    |> Case.f32Bits "b" b
    |> Case.output "y" capacity
    |> Case.u32 "n" (uint32 a.Length)
    |> Case.doc doc

let private addCases =
    let r = rng "add"

    let residual (name: string) (n: int) (doc: string) =
        addCase name (CaseGen.uniform r n -50.0f 50.0f |> bits) (CaseGen.uniform r n -2.0f 2.0f |> bits) (int64 n) doc

    // Every ordered pair of 24 special operands: zeros, denormals (whose sums are denormal, normal or zero), the
    // normal limits, infinities and NaN payloads.
    let pairs =
        let specials =
            [|
                0x00000000u
                0x80000000u
                0x00000001u
                0x80000001u
                0x007FFFFFu
                0x807FFFFFu
                0x00400000u
                0x00800000u
                0x80800000u
                0x00800001u
                0x3F800000u
                0xBF800000u
                0x33800000u
                0x4B800000u
                0x7F7FFFFFu
                0xFF7FFFFFu
                0x7F000000u
                0x7F800000u
                0xFF800000u
                0x7FC00000u
                0x7FC00001u
                0xFFC00000u
                0x7F800001u
                0xFFBFFFFFu
            |]

        let a =
            Array.init (specials.Length * specials.Length) (fun i -> specials[i / specials.Length])

        let b =
            Array.init (specials.Length * specials.Length) (fun i -> specials[i % specials.Length])

        addCase
            "special_pairs"
            a
            b
            (int64 a.Length)
            "Every ordered pair of 24 specials: signed zeros, denormal sums, overflow to inf, inf - inf, NaN payloads"

    // Cancellation: a + (-a) is +0, near-cancellation leaves few bits.
    let cancel =
        let n = 1000
        let a = CaseGen.uniform r n -1e6f 1e6f

        let b = a |> Array.mapi (fun i v -> if i % 2 = 0 then -v else -(v * (1.0f + 1e-6f)))

        addCase "cancellation" (bits a) (bits b) (int64 n) "Exact cancellation to +0 and near-cancellation"

    [
        residual "qwen_residual_decode" Hidden "The residual add at decode: 1024 elements, 4 groups"
        residual "qwen_residual_prefill37" (37 * Hidden) "The residual add over a 37-token prefill"
        (residual "qwen_residual_prefill512_in_place" (512 * Hidden) "The residual stream updated in place: y aliases a"
         |> withoutOutput
         |> Case.inPlace "y" "a")
        (residual "in_place_b" 1000 "y aliases b"
         |> withoutOutput
         |> Case.inPlace "y" "b")
        pairs
        cancel
        residual "n1" 1 "One element"
        residual "n255" 255 "One group, one thread idle"
        (addCase
            "n257_capacity"
            (CaseGen.uniform r 257 -1.0f 1.0f |> bits)
            (CaseGen.uniform r 257 -1.0f 1.0f |> bits)
            300L
            "Two groups, the second with one element; nothing past n is written")
    ]

let private addLarge () =
    let n = 65537 * 256

    [
        addCase
            "folded_n16777472"
            (bigUniform "add.folded_a" n -50.0f 50.0f |> bits)
            (bigUniform "add.folded_b" n -2.0f 2.0f |> bits)
            (int64 n)
            "65,537 groups: folded to 32769 x 2, so one extra group must write nothing"
        |> Case.gpuOnly
    ]

// ---- copy_rows ------------------------------------------------------------------------------------------------------

let private copy
    (name: string)
    (src: CaseTensor)
    (dstElements: int64)
    (rows: int)
    (cols: int)
    (srcRow0: int)
    (dstRow0: int)
    (srcStride: int)
    (dstStride: int)
    (doc: string)
    =
    Case.create "copy_rows" name
    |> Case.tensor src
    |> Case.outputOf "dst" src.DType dstElements
    |> Case.u32 "rows" (uint32 rows)
    |> Case.u32 "cols" (uint32 cols)
    |> Case.u32 "src_row0" (uint32 srcRow0)
    |> Case.u32 "dst_row0" (uint32 dstRow0)
    |> Case.u32 "src_stride" (uint32 srcStride)
    |> Case.u32 "dst_stride" (uint32 dstStride)
    |> Case.doc doc

let private srcOf (dtype: DType) (words: uint32[]) =
    {
        Field = "src"
        DType = dtype
        Data = CaseData.Input(MemoryMarshal.AsBytes(ReadOnlySpan words).ToArray())
        Dims = None
    }

let private copyCases =
    let r = rng "copy"
    let f32Words n = CaseGen.uniform r n -3.0f 3.0f |> bits
    let maxPos = 1024

    [
        copy
            "qwen_kv_cache_append_prefill37"
            (srcOf DType.F32 (f32Words (37 * KvDim)))
            (int64 maxPos * int64 KvDim)
            37
            KvDim
            0
            512
            KvDim
            KvDim
            "37 k rows appended to a 1024-position cache at position 512 (one packed run)"
        copy
            "qwen_kv_cache_append_decode"
            (srcOf DType.F32 (f32Words KvDim))
            (int64 maxPos * int64 KvDim)
            1
            KvDim
            0
            (maxPos - 1)
            KvDim
            KvDim
            "Decode: one row into the cache's last position"
        copy
            "qwen_tap_q_prefill37"
            (srcOf DType.F32 (f32Words (37 * QDim)))
            (37L * int64 QDim)
            37
            QDim
            0
            0
            QDim
            QDim
            "A tap snapshot of q over a 37-token prefill: 75,776 elements"
        copy
            "qwen_column_block"
            (srcOf DType.F32 (f32Words (10 * 3 * KvDim)))
            (10L * int64 KvDim + 9L)
            10
            KvDim
            0
            0
            (3 * KvDim)
            KvDim
            "The first 1024 columns of rows of 3072: a column block (src_stride 3072, dst_stride 1024)"
        copy
            "strided_gaps"
            (srcOf DType.U32 (CaseGen.words r (8 * 50)))
            (12L * 64L)
            8
            50
            0
            3
            50
            64
            "Destination rows with gaps of 14 words that must keep their poison"
        copy
            "broadcast_src_stride0"
            (srcOf DType.F32 (f32Words 300))
            (7L * 300L)
            7
            300
            0
            0
            0
            300
            "src_stride 0: every row copies the same source row"
        copy
            "overlapping_src_rows"
            (srcOf DType.I32 (Array.init 600 (fun i -> uint32 (i - 300))))
            (40L * 512L)
            40
            512
            3
            0
            1
            512
            "src_stride 1 < cols: source rows overlap (row r is a window from r + 3)"
        copy
            "nan_payload_words"
            (srcOf DType.F32 (CaseGen.words r (16 * 33)))
            (16L * 33L)
            16
            33
            0
            0
            33
            33
            "Random words as f32: every NaN payload and denormal moves bit for bit"
        copy
            "cols1_rows1000"
            (srcOf DType.U32 (CaseGen.words r 3000))
            2300L
            1000
            1
            500
            100
            2
            2
            "One column of 1000 rows, both strided, from source row 500 to destination row 100"
        copy
            "high_row0"
            (srcOf DType.F32 (f32Words (70000 + 5)))
            (60000L + 5L)
            1
            5
            70000
            60000
            1
            1
            "Row offsets past 65,535 with stride 1"
    ]

let private copyLarge () =
    let r = rng "copy.large"

    [
        copy
            "rows65537_cols256_folded"
            (srcOf DType.F32 (CaseGen.uniform r (65536 + 256) -3.0f 3.0f |> bits))
            (65537L * 256L)
            65537
            256
            0
            0
            1
            256
            "65,537 rows of 256: 65,537 groups folded to 32769 x 2, one extra group writes nothing; source windows overlap (stride 1), so every row differs"
        |> Case.gpuOnly
    ]

// ---- argmax_rows ----------------------------------------------------------------------------------------------------

let private argmax (name: string) (x: uint32[]) (rows: int) (cols: int) (capacity: int64) (doc: string) =
    Case.create "argmax_rows" name
    |> Case.f32Bits "x" x
    |> Case.outputU32 "y" capacity
    |> Case.u32 "rows" (uint32 rows)
    |> Case.u32 "cols" (uint32 cols)
    |> Case.doc doc

let private argmaxCases =
    let r = rng "argmax"

    let logits (name: string) (rows: int) =
        let x = bigUniform ("argmax." + name) (rows * Vocab) -12.0f 12.0f |> bits

        // One clear peak per row at a random column.
        for row in 0 .. rows - 1 do
            x[row * Vocab + int (r.NextBelow(uint32 Vocab))] <- f32 30.0f

        x

    // Ties and extremes over the vocabulary, one row each.
    let ties =
        let rows = 8
        let x = bigUniform "argmax.ties" (rows * Vocab) -12.0f 12.0f |> bits
        let set row c (v: uint32) = x[row * Vocab + c] <- v
        let fill row (v: uint32) = Array.fill x (row * Vocab) Vocab v
        // Row 0: the maximum at 256 (thread 0's second column) and 255 (thread 255's first): 255 wins.
        set 0 256 (f32 40.0f)
        set 0 255 (f32 40.0f)
        // Row 1: the maximum at the last column and at column 0: 0 wins.
        set 1 (Vocab - 1) (f32 40.0f)
        set 1 0 (f32 40.0f)
        // Row 2: three equal maxima in three threads; 999 wins.
        set 2 1000 (f32 40.0f)
        set 2 999 (f32 40.0f)
        set 2 151000 (f32 40.0f)
        // Row 3: every value negative but -0 at 77 and +0 at 3: they tie, 3 wins.
        fill 3 (f32 -1.0f)
        set 3 77 0x80000000u
        set 3 3 0x00000000u
        // Row 4: every value equal: 0 wins.
        fill 4 (f32 2.5f)
        // Row 5: +inf at 151,000 and the last column: 151,000 wins.
        set 5 151000 0x7F800000u
        set 5 (Vocab - 1) 0x7F800000u
        // Row 6: every value -inf: 0 wins.
        fill 6 0xFF800000u
        // Row 7: the largest finite value among -inf, at a thread's last column.
        fill 7 0xFF800000u
        set 7 (Vocab - 7) 0x7F7FFFFFu

        argmax
            "qwen_vocab_ties"
            x
            rows
            Vocab
            (int64 rows)
            "Ties across threads take the lowest index; -0 ties +0; infinities"

    // NaN rows interleaved with clean rows, so the NaN flag of one row cannot leak into the next.
    let nans =
        let cols = 1000
        let rows = 12
        let x = CaseGen.uniform r (rows * cols) -5.0f 5.0f |> bits

        let nanAt =
            [
                0, 0, 0x7FC00000u
                2, cols - 1, 0x7F800001u
                4, 500, 0xFFC00000u
                6, 256, 0xFFFFFFFFu
                8, 255, 0x7FFFFFFFu
                10, 13, 0x7FC00000u
            ]

        for row, c, v in nanAt do
            x[row * cols + c] <- v

        // Row 10 also holds +inf, which must not hide the NaN.
        x[10 * cols + 600] <- 0x7F800000u
        argmax "nan_rows" x rows cols (int64 rows) "Even rows hold a NaN somewhere (0xFFFFFFFF), odd rows none"

    // A positive denormal at 10 above +0 at 5, every other value negative: preserve gives 10; a flushing class ties
    // them, so 5 (KernelBounds' flushed reference).
    let denormal =
        let cols = 300
        let x = Array.create (2 * cols) (f32 -1.0f)
        x[10] <- 0x00000001u
        x[5] <- 0x00000000u
        x[cols + 290] <- 0x00400000u
        x[cols + 20] <- 0x80400000u
        argmax "denormal_max" x 2 cols 2L "A denormal maximum above zero; a flushing class ties it with zero"

    let plain (name: string) (rows: int) (cols: int) (doc: string) =
        argmax name (CaseGen.uniform r (rows * cols) -4.0f 4.0f |> bits) rows cols (int64 rows + 2L) doc

    [
        argmax "qwen_logits_decode" (logits "decode" 1) 1 Vocab 1L "Greedy decode: one row of 151,936 logits"
        argmax "qwen_logits_prefill37" (logits "prefill37" 37) 37 Vocab 37L "37 rows of logits (22 MB)"
        ties
        nans
        denormal
        plain "cols1_rows3" 3 1 "One column: index 0; 255 threads without a candidate"
        plain "cols100" 4 100 "Fewer columns than threads"
        plain "cols255" 3 255 "One column short of a group"
        plain "cols257" 3 257 "Thread 0 takes two columns"
        plain "rows65537_cols3" 65537 3 "65,537 rows: a folded grid (32769 x 2) with one extra group"
    ]

/// The GPU-only cases, which need from 1 MB to 311 MB of host memory each. CpuDevice never plans them
/// (Suites.casesFor), but HarnessTests' arithmetic-model tests plan every discovered case on three CpuDevices at once,
/// and with these that takes more than 8 GB, more than a CI runner has. So the cpu tier (LRM_TEST_TIER=cpu, which
/// tools/test.ps1 and CI set) does not build them; every other run, the gpu tier included, does. The gate goes once
/// those tests leave GPU-only cases out as Suites.casesFor does (request to E1 in the K1 pull request).
let large () =
    if Env.tryGet "LRM_TEST_TIER" = Some "cpu" then
        []
    else
        embedLarge ()
        @ rmsnormLarge ()
        @ ropeLarge ()
        @ siluLarge ()
        @ addLarge ()
        @ copyLarge ()

/// Every K1 case: the CI cases of every kernel, then the GPU-only ones (outside the cpu tier).
let cases: KernelCase list =
    embedCases
    @ rmsnormCases
    @ ropeCases
    @ siluCases
    @ addCases
    @ copyCases
    @ argmaxCases
    @ large ()
