/// CpuReferenceSemantics: the op definitions (known answers), batch invariance for T in {1, 7, 33, 128}, and
/// bit-identity at every thread count.
module Lrm.Tensor.Tests.CpuRefTests

open System
open System.Numerics
open Expecto
open FsCheck
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Testing
open Lrm.Tensor.Tests.Common

let private ro (xs: 'a[]) = ReadOnlySpan<'a>(xs)
let private bits32 (x: float32) = BitConverter.SingleToUInt32Bits x
let private bits64 (x: float) = BitConverter.DoubleToInt64Bits x
let private m (r: int) (c: int) (xs: float32[]) = HostTensor.f32 (Shape.matrix r c) xs
let private md (r: int) (c: int) (xs: float[]) = HostTensor.f64 (Shape.matrix r c) xs
let private w (r: int) (c: int) (xs: uint16[]) = HostTensor.bf16 (Shape.matrix r c) xs

let private tinyShape =
    {
        QHeads = 4
        KvHeads = 2
        HeadDim = 32
    }

/// Every op of the forward on T rows of random input, as a list of (name, result) for one mode.
let private allOps<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (exec: CpuExec)
    (t: int)
    : (string * IHostTensor) list =
    let dtype =
        if typeof<'T> = typeof<float32> then
            DType.F32
        else
            DType.F64

    let x (stream: string) (cols: int) =
        HostTensor.create dtype (Shape.matrix t cols) (random<'T> 5UL stream (t * cols) 1.0)

    let rope = TinyModel.model.Value.Rope
    let table = w 256 64 (randomBf16 5UL "ops/embed" (256 * 64) 0.5f)
    let wq = w 192 64 (randomBf16 5UL "ops/wq" (192 * 64) 0.2f)
    let norm = HostTensor.bf16 (Shape.vector 32) (randomBf16 5UL "ops/norm" 32 2.0f)
    let ids = Array.init t (fun i -> (i * 37 + 11) % 256)
    let q = x "ops/q" 128
    let k = x "ops/k" 64
    let v = x "ops/v" 64
    let eps = 'T.CreateTruncating 1e-6
    let scale = CpuRef.attentionScale<'T> 32

    [
        "embed", CpuRef.embed<'T> exec table ids :> IHostTensor
        "rmsnorm", CpuRef.rmsNorm exec q norm eps
        "matmul", CpuRef.matmulBf16W exec (x "ops/h" 64) wq
        "rope", CpuRef.ropeApply exec rope q 4 3
        "attention", CpuRef.attention exec q k v tinyShape 0 scale
        "silu_mul", CpuRef.siluMul exec q (x "ops/u" 128)
        "add", CpuRef.add exec q (x "ops/u" 128)
        "softmax", CpuRef.softmax exec q
        "logsumexp", CpuRef.logSumExp exec q
        "kl", CpuRef.kl exec q (x "ops/u" 128)
    ]

/// Each row of an op's T-row result equals the op run on that row alone (rows of 1), bit for bit; attention rows run
/// alone as decode steps over the same cache.
let private batchInvariant<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (t: int)
    =
    let exec = CpuExec.Machine

    let dtype =
        if typeof<'T> = typeof<float32> then
            DType.F32
        else
            DType.F64

    let x (stream: string) (cols: int) =
        HostTensor.create dtype (Shape.matrix t cols) (random<'T> 7UL stream (t * cols) 1.0)

    let rope = TinyModel.model.Value.Rope
    let table = w 256 64 (randomBf16 7UL "bi/embed" (256 * 64) 0.5f)
    let wq = w 192 64 (randomBf16 7UL "bi/wq" (192 * 64) 0.2f)
    let norm = HostTensor.bf16 (Shape.vector 32) (randomBf16 7UL "bi/norm" 32 2.0f)
    let ids = Array.init t (fun i -> (i * 37 + 11) % 256)
    let h = x "bi/h" 64
    let q = x "bi/q" 128
    let k = x "bi/k" 64
    let v = x "bi/v" 64
    let u = x "bi/u" 128
    let eps = 'T.CreateTruncating 1e-6
    let scale = CpuRef.attentionScale<'T> 32
    let label (op: string) (r: int) = Invariant.format $"{op} T={t} row {r}"

    let embed = CpuRef.embed<'T> exec table ids
    let norm2 = CpuRef.rmsNorm exec q norm eps
    let mm = CpuRef.matmulBf16W exec h wq
    let ro' = CpuRef.ropeApply exec rope q 4 5
    let att = CpuRef.attention exec q k v tinyShape 0 scale
    let silu = CpuRef.siluMul exec q u
    let sum = CpuRef.add exec q u

    for r in 0 .. t - 1 do
        expectSameBits (label "embed" r) (rowOf embed r) (CpuRef.embed<'T> exec table [| ids[r] |])
        expectSameBits (label "rmsnorm" r) (rowOf norm2 r) (CpuRef.rmsNorm exec (rowOf q r) norm eps)
        expectSameBits (label "matmul" r) (rowOf mm r) (CpuRef.matmulBf16W exec (rowOf h r) wq)
        expectSameBits (label "rope" r) (rowOf ro' r) (CpuRef.ropeApply exec rope (rowOf q r) 4 (5 + r))
        expectSameBits (label "attention" r) (rowOf att r) (CpuRef.attention exec (rowOf q r) k v tinyShape r scale)
        expectSameBits (label "silu_mul" r) (rowOf silu r) (CpuRef.siluMul exec (rowOf q r) (rowOf u r))
        expectSameBits (label "add" r) (rowOf sum r) (CpuRef.add exec (rowOf q r) (rowOf u r))

    // A chunked prefill (rows [0, a) then [a, T)) equals the whole prefill.
    let a = t / 2

    if a > 0 then
        let second = CpuRef.attention exec (rowsOf q a t) k v tinyShape a scale
        expectSameBits (Invariant.format $"attention T={t} chunked at {a}") second (rowsOf att a t)

/// The ops at 1 thread equal the ops at 7 and 32 threads, bit for bit.
let private sameAtEveryThreadCount (mode: string) (run: CpuExec -> (string * IHostTensor) list) =
    let a = run CpuExec.Serial

    for n in [ 7; 32 ] do
        for (name, x), (_, y) in List.zip a (run (CpuExec.WithThreads n)) do
            expectSameBits (Invariant.format $"{mode} {name}, 1 vs {n} threads") x y

let tests =
    testList
        "cpuref"
        [
            testList
                "definitions"
                [
                    testCase "dot fuses every step: fma(a2, b2, a1*b1), not a rounded product"
                    <| fun () ->
                        let e = MathF.ScaleB(1.0f, -12)
                        // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 needs 25 bits; fused, the 2^-24 survives the add of -1.
                        let d = Rows.dot (ro [| 1.0f; 1.0f + e |]) (ro [| -1.0f; 1.0f + e |])
                        Expect.equal (bits32 d) (bits32 (MathF.ScaleB(1.0f, -11) + MathF.ScaleB(1.0f, -24))) "fused"
                        Expect.equal (bits32 d) 0x3a000400u "contract vector"

                    testCase "reductions run in increasing index order"
                    <| fun () ->
                        // 1 + 1e8 rounds to 1e8 in float32, so the ordered sum is 0; summed from the end it would be 1.
                        Expect.equal (Rows.dot (ro [| 1.0f; 1e8f; -1e8f |]) (ro [| 1.0f; 1.0f; 1.0f |])) 0.0f "ordered"

                    testCase "rmsNorm uses the weight as w, not 1 + w"
                    <| fun () ->
                        let y = Array.zeroCreate<float32> 2

                        Rows.rmsNorm
                            (ro [| 3.0f; 4.0f |])
                            (ro [| Bf16.narrowRne 1.0f; Bf16.narrowRne 2.0f |])
                            0.0f
                            (Span y)

                        let r = 1.0f / MathF.Sqrt 12.5f
                        Expect.equal (bits32 y[0]) (bits32 (1.0f * (3.0f * r))) "w = 1"
                        Expect.equal (bits32 y[1]) (bits32 (2.0f * (4.0f * r))) "w = 2"
                        Expect.equal (bits32 y[0], bits32 y[1]) (0x3f593924u, 0x4010d0c3u) "contract vector"
                        Rows.rmsNorm (ro [| 3.0f; 4.0f |]) (ro [| 0us; 0us |]) 0.0f (Span y)
                        Expect.sequenceEqual y [| 0.0f; 0.0f |] "w = 0 gives 0"

                    testCase "softmax, logSumExp and KL known answers; NaN spreads"
                    <| fun () ->
                        let y = Array.zeroCreate<float> 2
                        Rows.softmax (ro [| 0.0; 0.0 |]) (Span y)
                        Expect.sequenceEqual y [| 0.5; 0.5 |] "uniform"
                        Expect.equal (Rows.logSumExp (ro [| 0.0; 0.0 |])) (log 2.0) "log 2"
                        Rows.softmax (ro [| 1.0; nan |]) (Span y)
                        Expect.isTrue (Array.forall Double.IsNaN y) "NaN row"
                        let kl = Rows.kl (ro [| 0.0; 0.0 |]) (ro [| 0.0; log 3.0 |])
                        Expect.floatClose { absolute = 1e-15; relative = 1e-14 } kl (0.5 * log (4.0 / 3.0)) "KL(P||Q)"

                    testCase "argmax takes the lowest index on ties and raises on NaN"
                    <| fun () ->
                        Expect.equal (Rows.argmax (ro [| 1.0f; 3.0f; 3.0f |])) 1 "tie"
                        Expect.equal (Rows.argmax (ro [| -0.0f; 0.0f |])) 0 "signed zeros tie"
                        Expect.equal (Rows.argmax (ro [| -infinityf; -infinityf |])) 0 "all -inf"

                        Expect.throwsT<ArgumentException>
                            (fun () -> Rows.argmax (ro [| 1.0f; nanf |]) |> ignore)
                            "NaN raises"

                        Expect.equal (Rows.tryArgmax (ro [| nan; 1.0 |])) ValueNone "NaN first"
                        let rows = m 2 2 [| 1.0f; nanf; 2.0f; 2.0f |]
                        Expect.sequenceEqual (CpuRef.argmaxRows rows) [| UInt32.MaxValue; 0u |] "kernel semantics"

                    testCase "topK orders by value, then by index"
                    <| fun () ->
                        Expect.sequenceEqual (Rows.topK (ro [| 2.0f; 5.0f; 2.0f; 7.0f |]) 3) [| 3; 1; 0 |] "top 3"

                        Expect.throwsT<ArgumentException>
                            (fun () -> Rows.topK (ro [| 1.0f |]) 2 |> ignore)
                            "k too large"

                    testCase "silu_mul and RoPE follow their formulas, rounding each operation"
                    <| fun () ->
                        let g = [| -3.0f; 0.0f; 0.5f; 20.0f |]
                        let u = [| 2.0f; 1.0f; -1.5f; 0.25f |]
                        let y = Array.zeroCreate<float32> 4
                        Rows.siluMul (ro g) (ro u) (Span y)

                        for i in 0..3 do
                            Expect.equal (bits32 y[i]) (bits32 ((g[i] / (1.0f + MathF.Exp(-g[i]))) * u[i])) "silu"

                        let x = [| 0.5f; -1.25f; 2.0f; 0.75f |]
                        let c = [| 0.6f; 0.8f |]
                        let s = [| 0.8f; -0.6f |]
                        let r = Array.zeroCreate<float32> 4
                        Rows.ropeHead (ro x) (ro c) (ro s) (Span r)
                        Expect.equal (bits32 r[0]) (bits32 (x[0] * c[0] - x[2] * s[0])) "y[0]"
                        Expect.equal (bits32 r[2]) (bits32 (x[2] * c[0] + x[0] * s[0])) "y[h]"
                        Expect.equal (bits32 r[3]) (bits32 (x[3] * c[1] + x[1] * s[1])) "y[h+1]"

                    testCase
                        "attention: one key returns V; GQA maps head h to KV head h / group; the mask is the loop bound"
                    <| fun () ->
                        let shape = { QHeads = 4; KvHeads = 2; HeadDim = 2 }

                        let q = md 1 8 [| 1.0; 0.0; 0.0; 1.0; 2.0; 2.0; -1.0; 0.5 |]
                        // KV head 0 has V = (10, 20); KV head 1 has V = (30, 40). Row 1 is a future key full of NaN.
                        let k = md 2 4 [| 0.5; 0.5; 1.0; -1.0; nan; nan; nan; nan |]
                        let v = md 2 4 [| 10.0; 20.0; 30.0; 40.0; nan; nan; nan; nan |]
                        let y = CpuRef.attention CpuExec.Serial q k v shape 0 1.0
                        Expect.sequenceEqual y.Data [| 10.0; 20.0; 10.0; 20.0; 30.0; 40.0; 30.0; 40.0 |] "one key"

                        // Two keys with scores 0 and ln 3: weights 1/4 and 3/4.
                        let q2 = md 1 2 [| 1.0; 0.0 |]
                        let k2 = md 2 2 [| 0.0; 5.0; log 3.0; 7.0 |]
                        let v2 = md 2 2 [| 4.0; 8.0; 0.0; 4.0 |]

                        let shape1 = { QHeads = 1; KvHeads = 1; HeadDim = 2 }

                        let y2 = CpuRef.attention CpuExec.Serial q2 k2 v2 shape1 1 1.0
                        Expect.floatClose Accuracy.veryHigh y2.Data[0] 1.0 "0.25 * 4"
                        Expect.floatClose Accuracy.veryHigh y2.Data[1] 5.0 "0.25 * 8 + 0.75 * 4"

                    testCase
                        "embedding rows widen bf16 exactly; out-of-range ids raise, or give NaN rows in the kernel form"
                    <| fun () ->
                        let table = w 2 2 [| 0x3F80us; 0x0001us; 0xFF80us; 0x7F81us |]
                        let y = CpuRef.embed<float32> CpuExec.Serial table [| 1; 0 |]

                        Expect.sequenceEqual
                            (y.Data |> Array.map bits32)
                            [| 0xFF800000u; 0x7F810000u; 0x3F800000u; 0x00010000u |]
                            "exact bits"

                        Expect.throwsT<ArgumentOutOfRangeException>
                            (fun () -> CpuRef.embed<float32> CpuExec.Serial table [| 2 |] |> ignore)
                            "range"

                        let g32 = CpuRef.embedGather<float32> CpuExec.Serial table [| 5u |]
                        Expect.sequenceEqual (g32.Data |> Array.map bits32) [| 0x7FC00000u; 0x7FC00000u |] "F32 NaN row"
                        let g64 = CpuRef.embedGather<float> CpuExec.Serial table [| 2u |]

                        Expect.sequenceEqual
                            (g64.Data |> Array.map bits64)
                            [| 0x7FF8000000000000L; 0x7FF8000000000000L |]
                            "F64 NaN row"

                    testCase "matmul with a one-hot row returns the widened weight column exactly, in both modes"
                    <| fun () ->
                        let bits = randomBf16 11UL "onehot" (5 * 3) 3.0f
                        let wt = w 5 3 bits
                        let y = CpuRef.matmulBf16W CpuExec.Serial (m 1 3 [| 0.0f; 1.0f; 0.0f |]) wt

                        Expect.sequenceEqual
                            (y.Data |> Array.map bits32)
                            (Array.init 5 (fun j -> Bf16.widenBits bits[j * 3 + 1]))
                            "F32"

                        let yd = CpuRef.matmulBf16W CpuExec.Serial (md 1 3 [| 0.0; 1.0; 0.0 |]) wt
                        Expect.sequenceEqual yd.Data (Array.init 5 (fun j -> float (Bf16.widen bits[j * 3 + 1]))) "F64"

                    testCase "the attention scale is head_dim ** -0.5 rounded once"
                    <| fun () ->
                        Expect.equal (CpuRef.attentionScale<float> 32) (Math.Pow(32.0, -0.5)) "f64"
                        Expect.equal (CpuRef.attentionScale<float32> 128) (float32 (Math.Pow(128.0, -0.5))) "f32"
                        Expect.equal (bits32 (CpuRef.attentionScale<float32> 128)) 0x3db504f3u "contract vector"

                    testCase "only float32 and float are CpuRef modes"
                    <| fun () ->
                        Expect.throwsT<ArgumentException>
                            (fun () -> CpuRef.widen<Half>(w 1 1 [| 0us |]) |> ignore)
                            "Half is not a mode"

                    testCase "shape errors are caught before any work"
                    <| fun () ->
                        let x = m 2 4 (Array.zeroCreate 8)

                        Expect.throwsT<ArgumentException>
                            (fun () -> CpuRef.matmulBf16W CpuExec.Serial x (w 3 5 (Array.zeroCreate 15)) |> ignore)
                            "k"

                        Expect.throwsT<ArgumentException>
                            (fun () -> CpuRef.rmsNorm CpuExec.Serial x (w 1 3 (Array.zeroCreate 3)) 0.0f |> ignore)
                            "width"

                        Expect.throwsT<ArgumentOutOfRangeException>
                            (fun () ->
                                CpuRef.ropeApply
                                    CpuExec.Serial
                                    TinyModel.model.Value.Rope
                                    (m 1 32 (Array.zeroCreate 32))
                                    1
                                    512
                                |> ignore)
                            "position"

                        Expect.throwsT<ArgumentException>
                            (fun () ->
                                CpuRef.attention
                                    CpuExec.Serial
                                    (m 2 128 (Array.zeroCreate 256))
                                    (m 2 64 (Array.zeroCreate 128))
                                    (m 2 64 (Array.zeroCreate 128))
                                    tinyShape
                                    1
                                    1.0f
                                |> ignore)
                            "cache rows"
                ]

            testList
                "batch invariance"
                [
                    for t in [ 1; 7; 33; 128 ] do
                        testCase (Invariant.format $"every row equals the row computed alone, T = {t}")
                        <| fun () ->
                            batchInvariant<float32> t
                            batchInvariant<float> t
                ]

            testCase "every op is bit-identical at 1, 7 and 32 threads, in both modes"
            <| fun () ->
                sameAtEveryThreadCount "F32" (fun e -> allOps<float32> e 128)
                sameAtEveryThreadCount "F64" (fun e -> allOps<float> e 128)

            Props.testProperty "argmax is the lowest index of the maximum"
            <| fun (xs: NormalFloat list) (dup: byte) ->
                let a = xs |> List.map (fun x -> float32 x.Get) |> Array.ofList

                if a.Length < 2 || Array.exists (fun v -> Single.IsInfinity v) a then
                    true
                else
                    // Plant a tie with the maximum at a random later index.
                    let mx = Array.max a
                    let i = Array.findIndex ((=) mx) a
                    let j = int dup % a.Length
                    a[max i j] <- mx
                    Rows.argmax (ReadOnlySpan a) = Array.findIndex ((=) mx) a

            Props.testProperty "topK equals a stable sort by descending value"
            <| fun (xs: NormalFloat list) (k: byte) ->
                let a = xs |> List.map (fun x -> x.Get) |> Array.ofList

                if a.Length = 0 then
                    true
                else
                    let k = 1 + int k % a.Length

                    let expected =
                        a
                        |> Array.mapi (fun i v -> -v, i)
                        |> Array.sort
                        |> Array.map snd
                        |> Array.truncate k

                    Rows.topK (ReadOnlySpan a) k = expected

            Props.testProperty "matmul rows do not depend on the batch (random m, n, k)"
            <| fun (seed: int) (mRows: byte) (nCols: byte) (kLen: byte) ->
                let mr = 1 + int mRows % 9
                let nc = 1 + int nCols % 70
                let kl = 1 + int kLen % 50

                let x =
                    HostTensor.f32 (Shape.matrix mr kl) (randomF32 (uint64 seed) "p/x" (mr * kl) 2.0f)

                let wt = w nc kl (randomBf16 (uint64 seed) "p/w" (nc * kl) 1.0f)
                let y = CpuRef.matmulBf16W (CpuExec.WithThreads 4) x wt

                Seq.forall
                    (fun r -> bytesOf (rowOf y r) = bytesOf (CpuRef.matmulBf16W CpuExec.Serial (rowOf x r) wt))
                    (seq { 0 .. mr - 1 })
        ]
