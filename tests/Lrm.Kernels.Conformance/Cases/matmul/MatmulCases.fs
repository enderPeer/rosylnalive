/// K2's conformance cases for matmul_bf16w (ConformanceHarness contract, "Where cases live"): every Qwen3-0.6B
/// projection and lm_head at decode (m = 1) and prefill (m = 37, 512 and 4,000) sizes, plus edge shapes for the tile,
/// slab and load paths of the kernel. The harness runs each case twice with poisoned outputs, judges it against
/// KernelBounds (relative, against CpuRef F32 and F64), checks batch invariance on its splits (the first row alone,
/// the last row alone, the middle third) and times it against the 100 ms budget.
///
/// Batch invariance across calls: all cases of one Qwen shape share one weight and the first rows of one activation
/// matrix, so row r is the same row of the same inputs in the m = 1, 37, 512 and 4,000 cases. Within each case the
/// harness's splits recompute single rows (m = 1) and the middle third alone, and they must equal the whole case bit
/// for bit; across cases, every row is compared with the one CpuRef F32 reference of that row.
///
/// The m = 512 and m = 4,000 prefills and lm_head are GPU-only: CpuRef F32 and F64 of them cost tens of G
/// multiply-adds, which the cpu tier (CI, four emulation threads) does not spend. Decode and m = 37 of every projection
/// and every edge shape run in CI on CpuDevice too. The GPU-only cases are built only outside the cpu tier
/// (LRM_TEST_TIER, which tools/test.ps1 sets): the harness's own model tests (HarnessTests "models") run every
/// discovered case on CpuDevice, GPU-only cases included, so in the cpu tier they would emulate them three times.
module Lrm.Kernels.Conformance.Cases.Matmul

open System
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Kernels.Conformance
open Lrm.Testing

/// Deterministic data by element index: SplitMix64's mix of (seed, index), so large tensors are cheap to build and
/// any prefix of a tensor is the same whatever its length.
module private Data =
    let private word (seed: uint64) (i: int64) =
        SplitMix64.Mix(seed + uint64 i * SplitMix64.Gamma)

    /// float32 values in [-1, 1): a 24-bit integer over 2^23, minus 1, so every value is exact.
    let f32 (seed: uint64) (n: int64) : float32[] =
        Array.init (int n) (fun i ->
            let u = uint32 (word seed (int64 i) >>> 40)
            float32 (int u - (1 <<< 23)) / 8388608.0f)

    /// bf16 bits of values in [-1, 1): an 8-bit mantissa, exponents 2^-1 to 2^-8 and a sign, so every weight is a
    /// normal bf16 (Qwen's weights are normal bf16 values of a similar spread).
    let bf16 (seed: uint64) (n: int64) : byte[] =
        let h = Array.zeroCreate<uint16>(int n)

        for i in 0 .. h.Length - 1 do
            let z = word seed (int64 i)
            let sign = uint16 ((z >>> 63) <<< 15)
            let exponent = uint16 (126UL - (z >>> 32) % 8UL)
            let mantissa = uint16 (z &&& 0x7FUL)
            h[i] <- sign ||| (exponent <<< 7) ||| mantissa

        MemoryMarshal.AsBytes(ReadOnlySpan h).ToArray()

    let bytesOf (xs: float32[]) =
        MemoryMarshal.AsBytes(ReadOnlySpan xs).ToArray()

/// One Qwen3-0.6B weight shape: n output features of k inputs (hidden 1024, q 16 x 128 = 2048, kv 8 x 128 = 1024,
/// intermediate 3072, vocabulary 151,936).
type private Shape =
    {
        Name: string
        N: int
        K: int
        /// What the projection is in the model.
        Doc: string
    }

let private qwenShapes =
    [
        {
            Name = "q_proj"
            N = 2048
            K = 1024
            Doc = "q_proj: hidden 1024 to 16 heads x 128"
        }
        {
            Name = "kv_proj"
            N = 1024
            K = 1024
            Doc = "k_proj and v_proj: hidden 1024 to 8 kv heads x 128"
        }
        {
            Name = "o_proj"
            N = 1024
            K = 2048
            Doc = "o_proj: 16 heads x 128 back to hidden 1024"
        }
        {
            Name = "gate_up_proj"
            N = 3072
            K = 1024
            Doc = "gate_proj and up_proj: hidden 1024 to intermediate 3072"
        }
        {
            Name = "down_proj"
            N = 1024
            K = 3072
            Doc = "down_proj: intermediate 3072 to hidden 1024, the longest reduction"
        }
    ]

let private lmHead =
    {
        Name = "lm_head"
        N = 151936
        K = 1024
        Doc = "lm_head: hidden 1024 to the 151,936-entry vocabulary (2,374 column tiles)"
    }

/// The longest prefill a case uses; the shared activations of a shape have this many rows.
[<Literal>]
let private MaxRows = 4000

/// One matmul case from x's first m * k values and the whole of w.
let private mm (name: string) (m: int) (n: int) (k: int) (x: byte[]) (w: byte[]) (doc: string) =
    Case.create "matmul_bf16w" name
    |> Case.inputBytes "x" DType.F32 x
    |> Case.inputBytes "w" DType.BF16 w
    |> Case.output "y" (int64 m * int64 n)
    |> Case.u32 "m" (uint32 m)
    |> Case.u32 "n" (uint32 n)
    |> Case.u32 "k" (uint32 k)
    |> Case.doc doc

/// The cases of one Qwen shape at each m: the same weight, and x as the first m rows of one [MaxRows, k] matrix.
let private shapeCases (s: Shape) (seed: uint64) (ms: (int * bool) list) =
    let w = Data.bf16 (seed + 1UL) (int64 s.N * int64 s.K)
    let rows = ms |> List.map fst |> List.max
    let x = Data.f32 seed (int64 rows * int64 s.K) |> Data.bytesOf

    [
        for m, gpuOnly in ms do
            let phase =
                if m = 1 then
                    "decode"
                else
                    Invariant.format $"prefill of {m} rows"

            let c =
                mm
                    (Invariant.format $"qwen_{s.Name}_m{m}")
                    m
                    s.N
                    s.K
                    (if m = rows then x else Array.sub x 0 (m * s.K * 4))
                    w
                    (Invariant.format $"{s.Doc}; {phase}; rows are the first rows of the shape's shared activations")

            if gpuOnly then Case.gpuOnly c else c
    ]

/// True in the cpu tier, where the GPU-only cases are not built (see the module's doc).
let private cpuTier =
    match Env.tryGet "LRM_TEST_TIER" with
    | Some t -> String.Equals(t.Trim(), "cpu", StringComparison.OrdinalIgnoreCase)
    | None -> false

/// The m values of the projections: decode and m = 37 everywhere, the m = 512 and 4,000 prefills on GPUs only.
let private projectionRows =
    [
        1, false
        37, false
        if not cpuTier then
            512, true
            MaxRows, true
    ]

let private qwenCases =
    qwenShapes
    |> List.mapi (fun i s -> shapeCases s (0x4B324D4D00000000UL + uint64 i * 0x100UL) projectionRows)
    |> List.concat

/// lm_head (GPU only): decode, and 64 rows of a prefill's logits.
let private lmHeadCases =
    if cpuTier then
        []
    else
        shapeCases lmHead 0x4B324D4D0000FF00UL [ 1, true; 64, true ]

/// Small and edge shapes: tile edges in m and n, every remainder of k against the 32-wide slab and the 4-wide run,
/// odd k (bf16 rows that start in the middle of a u32 word, read element by element), k below one slab, k = 0, and
/// the widest grid the catalog allows without folding (matmul_bf16w is not linear, so x is at most 65,535 groups).
let private edgeCases =
    let edge (name: string) (m: int) (n: int) (k: int) (doc: string) =
        let seed = DetRng.streamOf ("k2.matmul." + name)

        let x =
            if m * k = 0 then
                [| 0.0f |]
            else
                Data.f32 (0x4B32000000000000UL + seed) (int64 m * int64 k)

        let w =
            if n * k = 0 then
                Array.zeroCreate<byte> 2
            else
                Data.bf16 (0x4B32100000000000UL + seed) (int64 n * int64 k)

        mm name m n k (Data.bytesOf x) w doc

    [
        edge "m1_n1_k1" 1 1 1 "One output of one product: a single group with one live thread"
        edge "m5_n7_k1" 5 7 1 "k = 1: one step per output"
        edge "m70_n65_k0" 70 65 0 "k = 0 over 2 x 2 tiles: every output is +0, nothing is read"
        edge "m3_n67_k33" 3 67 33 "Odd k: w rows alternate between word-aligned and half-word starts; a 1-step tail"
        edge "m9_n33_k31" 9 33 31 "k below one slab: 7 runs of 4 and a 3-step tail"
        edge "m17_n96_k35" 17 96 35 "One slab and a 3-step second slab; n not a multiple of 64"
        edge "m63_n63_k63" 63 63 63 "One row and one column short of a tile; odd k"
        edge "m64_n64_k64" 64 64 64 "Exactly one tile and two slabs"
        edge "m65_n65_k65" 65 65 65 "One row and one column into a second tile; odd k"
        edge "m127_n129_k97" 127 129 97 "Tiles 2 x 3 with a partial last row and column tile; a 1-step tail"
        edge "m129_n130_k256" 129 130 256 "Three row tiles and three column tiles, eight full slabs"
        edge "m2_n200_k1023" 2 200 1023 "Odd k near Qwen's hidden size: element-wise w loads throughout"
        edge "m3_n64_k1025" 3 64 1025 "k = 1025: 32 full slabs and a 1-step 33rd"
        edge "m33_n17_k3072" 33 17 3072 "down_proj's k with a narrow n: one partial column tile"
        edge
            "m1_n4194240_k1"
            1
            (64 * 65535)
            1
            "n = 64 x 65,535: the widest grid without folding, 65,535 column tiles in x"
    ]

/// Every matmul_bf16w case of K2.
let cases: KernelCase list = qwenCases @ lmHeadCases @ edgeCases
