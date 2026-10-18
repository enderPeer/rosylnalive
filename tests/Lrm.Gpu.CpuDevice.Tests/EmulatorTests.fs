/// One emulator per catalog kernel, each against CpuRef F32 (the specification) through the tensor ops of
/// Lrm.Tensor.CpuReference, bit for bit; grid folding over the physical grid; and the GPU arithmetic models.
module Lrm.Gpu.CpuDevice.Tests.EmulatorTests

open System
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.CpuDevice.Tests.Common
open Lrm.Tensor
open Lrm.Tensor.CpuReference

let private exec = CpuExec.Serial

let private eqBits (actual: float32[]) (expected: float32[]) (what: string) =
    Expect.equal (f32Bits actual) (f32Bits expected) what

let private debug () =
    { DeviceOptions.For ExecutionBackend.CpuDevice with
        Debug = true
    }

let private kernelTests =
    testList
        "kernels against CpuRef F32"
        [
            testCase "scale_f32: one float32 multiply per element, NaN payloads propagated as the CPU does"
            <| fun () ->
                let r = rng "scale"

                let x =
                    Array.append (uniform r 997 -8.0f 8.0f) (ofBits [| 0x7FC00123u; 0x00000001u; 0x80800000u |])

                let f = fixture [ t "x" DType.F32 1000L; t "y" DType.F32 1000L ]
                f.UploadF32("x", x)

                f.Run(fun c -> Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "x") (f.Token "y") 1000u -0.75f))
                |> ignore

                eqBits (f.F32("y", 1000)) (x |> Array.map (fun v -> v * -0.75f)) "y = a * x"

            testCase "table_echo, args_echo and groups_echo"
            <| fun () ->
                let f = fixture [ t "src" DType.BF16 30L; t "y" DType.U32 64L ]

                f.Run(fun c -> Kernels.TableEcho.dispatch c { Src = f.Token "src"; Y = f.Token "y" })
                |> ignore

                Expect.equal (f.Words("y", 16)) ((f.Row "src").ToWords()) "the row"

                f.Run(fun c ->
                    Kernels.TableEcho.dispatch
                        c
                        {
                            Src = TensorToken.Null
                            Y = f.Token "y"
                        })
                |> ignore

                Expect.equal (f.Words("y", 16)) (Array.zeroCreate 16) "the null row"

                let a =
                    Kernels.ArgsEcho.encode
                        { Kernels.ArgsEcho.decode ArgBlock.Zero with
                            Y = f.Token "y"
                        }

                let a =
                    [ 2..31 ]
                    |> List.fold (fun (b: ArgBlock) i -> b.With(i, 0xA5000000u + uint32 i)) a

                f.Run(fun c -> c.Dispatch(Kernels.ArgsEcho.Id, a)) |> ignore
                Expect.equal (f.Words("y", 32)) (a.WithGroupsX(1u).ToWords()) "all 32 words, word 0 the physical x"

                f.Run(fun c -> Kernels.GroupsEcho.dispatch c { Y = f.Token "y"; Groups = 40u })
                |> ignore

                Expect.equal (f.Words("y", 40)) (Array.init 40 uint32) "y[g] = g"

            testCase "bf16_unpack moves every bit pattern"
            <| fun () ->
                let x = Array.init 65536 uint16
                let f = fixture [ t "x" DType.BF16 65536L; t "y" DType.F32 65536L ]
                f.UploadBf16("x", x)

                f.Run(fun c ->
                    Kernels.Bf16Unpack.dispatch
                        c
                        {
                            X = f.Token "x"
                            Y = f.Token "y"
                            N = 65536u
                        })
                |> ignore

                Expect.equal (f.Words("y", 65536)) (x |> Array.map Bf16.widenBits) "every bf16 pattern, widened"

            testCase "embed_gather_bf16 = CpuRef.embedGather, an id past vocab giving 7fc00000"
            <| fun () ->
                let r = rng "embed"
                let vocab, cols, rows = 20, 300, 6
                let table = uniformBf16 r (vocab * cols) -1.0f 1.0f
                let ids = [| 0u; 19u; 7u; 20u; 7u; 4000000000u |]

                let f =
                    fixture
                        [
                            readOnly (t "table" DType.BF16 (int64 (vocab * cols)))
                            t "ids" DType.U32 6L
                            t "y" DType.F32 (int64 (rows * cols))
                        ]

                f.UploadBf16("table", table)
                f.UploadU32("ids", ids)

                f.Run(fun c ->
                    Kernels.EmbedGatherBf16.dispatch
                        c
                        {
                            Ids = f.Token "ids"
                            Table = f.Token "table"
                            Y = f.Token "y"
                            Rows = uint32 rows
                            Cols = uint32 cols
                            Vocab = uint32 vocab
                        })
                |> ignore

                let want =
                    CpuRef.embedGather<float32> exec (HostTensor.bf16 (Shape.matrix vocab cols) table) ids

                eqBits (f.F32("y", rows * cols)) want.Data "rows"

            testCase "rmsnorm_rows = CpuRef.rmsNorm, also in place"
            <| fun () ->
                let r = rng "rmsnorm"
                let rows, cols = 9, 513
                let x = uniform r (rows * cols) -2.0f 2.0f
                let w = uniformBf16 r cols 0.0f 2.0f

                let f =
                    fixture
                        [
                            t "x" DType.F32 (int64 (rows * cols))
                            readOnly (t "w" DType.BF16 (int64 cols))
                            t "y" DType.F32 (int64 (rows * cols))
                        ]

                f.UploadF32("x", x)
                f.UploadBf16("w", w)

                let norm (y: string) =
                    f.Run(fun c ->
                        Kernels.RmsnormRows.dispatch
                            c
                            {
                                X = f.Token "x"
                                W = f.Token "w"
                                Y = f.Token y
                                Rows = uint32 rows
                                Cols = uint32 cols
                                Eps = 1e-6f
                            })
                    |> ignore

                let want =
                    CpuRef.rmsNorm<float32>
                        exec
                        (HostTensor.f32 (Shape.matrix rows cols) x)
                        (HostTensor.bf16 (Shape.vector cols) w)
                        1e-6f

                norm "y"
                eqBits (f.F32("y", rows * cols)) want.Data "out of place"
                norm "x"
                eqBits (f.F32("x", rows * cols)) want.Data "in place"

            testCase "rope_apply = CpuRef.ropeApply over the pinned table"
            <| fun () ->
                let r = rng "rope"
                let headDim, heads, tokens, pos0 = 64, 3, 5, 9

                let invFreq =
                    Array.init (headDim / 2) (fun i ->
                        BitConverter.SingleToUInt32Bits(float32 (Math.Pow(1e6, -2.0 * float i / float headDim))))

                let table = RopeTable.create exec invFreq 32
                let x = uniform r (tokens * heads * headDim) -3.0f 3.0f
                let n = int64 x.Length

                let f =
                    fixture
                        [
                            t "x" DType.F32 n
                            readOnly (t "cos" DType.F32 (int64 table.Cos.Length))
                            readOnly (t "sin" DType.F32 (int64 table.Sin.Length))
                            t "y" DType.F32 n
                        ]

                f.UploadF32("x", x)
                f.UploadF32("cos", table.Cos)
                f.UploadF32("sin", table.Sin)

                f.Run(fun c ->
                    Kernels.RopeApply.dispatch
                        c
                        {
                            X = f.Token "x"
                            CosTable = f.Token "cos"
                            SinTable = f.Token "sin"
                            Y = f.Token "y"
                            Rows = uint32 (tokens * heads)
                            Heads = uint32 heads
                            HeadDim = uint32 headDim
                            Pos0 = uint32 pos0
                        })
                |> ignore

                let want =
                    CpuRef.ropeApply<float32>
                        exec
                        table
                        (HostTensor.f32 (Shape.matrix tokens (heads * headDim)) x)
                        heads
                        pos0

                eqBits (f.F32("y", x.Length)) want.Data "rotated"

            testCase "silu_mul = CpuRef.siluMul and add = CpuRef.add"
            <| fun () ->
                let r = rng "elementwise"
                let n = 777
                let g = uniform r n -10.0f 10.0f
                let u = uniform r n -10.0f 10.0f

                let f =
                    fixture
                        [
                            t "g" DType.F32 777L
                            t "u" DType.F32 777L
                            t "y" DType.F32 777L
                            t "z" DType.F32 777L
                        ]

                f.UploadF32("g", g)
                f.UploadF32("u", u)

                f.Run(fun c ->
                    Kernels.SiluMul.dispatch
                        c
                        {
                            Gate = f.Token "g"
                            Up = f.Token "u"
                            Y = f.Token "y"
                            N = uint32 n
                        }

                    Kernels.Add.dispatch
                        c
                        {
                            A = f.Token "g"
                            B = f.Token "u"
                            Y = f.Token "z"
                            N = uint32 n
                        })
                |> ignore

                let s = Shape.vector n

                eqBits
                    (f.F32("y", n))
                    (CpuRef.siluMul<float32> exec (HostTensor.f32 s g) (HostTensor.f32 s u)).Data
                    "silu_mul"

                eqBits (f.F32("z", n)) (CpuRef.add<float32> exec (HostTensor.f32 s g) (HostTensor.f32 s u)).Data "add"

            testCase "copy_rows = CpuRef.copyRows for packed rows; strided rows land at their strides"
            <| fun () ->
                let r = rng "copy"
                let src = Array.init 200 (fun _ -> r.NextU32())
                let f = fixture [ t "src" DType.U32 200L; t "dst" DType.U32 200L ]
                f.UploadU32("src", src)
                f.UploadU32("dst", Array.zeroCreate 200)

                f.Run(fun c ->
                    Kernels.CopyRows.dispatch
                        c
                        {
                            Src = f.Token "src"
                            Dst = f.Token "dst"
                            Rows = 6u
                            Cols = 10u
                            SrcRow0 = 2u
                            DstRow0 = 5u
                            SrcStride = 10u
                            DstStride = 10u
                        })
                |> ignore

                let s = HostTensor.u32 (Shape.matrix 20 10) src
                let d = HostTensor.u32 (Shape.matrix 20 10) (Array.zeroCreate 200)
                CpuRef.copyRows s 2 d 5 6
                Expect.equal (f.Words("dst", 200)) d.Data "packed"

                f.Run(fun c ->
                    Kernels.CopyRows.dispatch
                        c
                        {
                            Src = f.Token "src"
                            Dst = f.Token "dst"
                            Rows = 3u
                            Cols = 2u
                            SrcRow0 = 1u
                            DstRow0 = 0u
                            SrcStride = 7u
                            DstStride = 3u
                        })
                |> ignore

                let got = f.Words("dst", 9)

                for row in 0..2 do
                    for col in 0..1 do
                        Expect.equal got[row * 3 + col] src[(1 + row) * 7 + col] "strided element"

            testCase "argmax_rows = CpuRef.argmaxRows: lowest index on ties, ffffffff for a NaN row"
            <| fun () ->
                let r = rng "argmax"
                let rows, cols = 5, 301
                let x = uniform r (rows * cols) -1.0f 1.0f
                x[cols + 3] <- 5.0f
                x[cols + 200] <- 5.0f
                x[2 * cols + 7] <- nanf
                let f = fixture [ t "x" DType.F32 (int64 x.Length); t "y" DType.U32 5L ]
                f.UploadF32("x", x)

                f.Run(fun c ->
                    Kernels.ArgmaxRows.dispatch
                        c
                        {
                            X = f.Token "x"
                            Y = f.Token "y"
                            Rows = uint32 rows
                            Cols = uint32 cols
                        })
                |> ignore

                let want = CpuRef.argmaxRows<float32>(HostTensor.f32 (Shape.matrix rows cols) x)
                Expect.equal (f.Words("y", rows)) want "indices"
                Expect.equal want[1] 3u "the lowest tie"
                Expect.equal want[2] UInt32.MaxValue "the NaN row"

            testCase "matmul_bf16w = CpuRef.matmulBf16W over partial tiles"
            <| fun () ->
                let r = rng "matmul"
                let m, n, k = 70, 130, 200
                let x = uniform r (m * k) -1.0f 1.0f
                let w = uniformBf16 r (n * k) -1.0f 1.0f

                let f =
                    fixture
                        [
                            t "x" DType.F32 (int64 (m * k))
                            readOnly (t "w" DType.BF16 (int64 (n * k)))
                            t "y" DType.F32 (int64 (m * n))
                        ]

                f.UploadF32("x", x)
                f.UploadBf16("w", w)

                f.Run(fun c ->
                    Kernels.MatmulBf16w.dispatch
                        c
                        {
                            X = f.Token "x"
                            W = f.Token "w"
                            Y = f.Token "y"
                            M = uint32 m
                            N = uint32 n
                            K = uint32 k
                        })
                |> ignore

                let want =
                    CpuRef.matmulBf16W<float32>
                        exec
                        (HostTensor.f32 (Shape.matrix m k) x)
                        (HostTensor.bf16 (Shape.matrix n k) w)

                eqBits (f.F32("y", m * n)) want.Data "y = x w^T"

            testCase
                "attention_gqa_causal = CpuRef.attention; a chunk with q_row0 and past equals the same rows of the prefill"
            <| fun () ->
                let r = rng "attention"
                let qh, kvh, hd, tokens = 4, 2, 16, 9
                let qc, kvc = qh * hd, kvh * hd
                let q = uniform r (tokens * qc) -1.0f 1.0f
                let k = uniform r (tokens * kvc) -1.0f 1.0f
                let v = uniform r (tokens * kvc) -1.0f 1.0f
                let scale = CpuRef.attentionScale<float32> hd

                let f =
                    fixture
                        [
                            t "q" DType.F32 (int64 q.Length)
                            t "k" DType.F32 (int64 k.Length)
                            t "v" DType.F32 (int64 v.Length)
                            t "y" DType.F32 (int64 q.Length)
                        ]

                f.UploadF32("q", q)
                f.UploadF32("k", k)
                f.UploadF32("v", v)

                let attend rows q0 past =
                    f.Run(fun c ->
                        Kernels.AttentionGqaCausal.dispatch
                            c
                            {
                                Q = f.Token "q"
                                K = f.Token "k"
                                V = f.Token "v"
                                Y = f.Token "y"
                                Rows = uint32 rows
                                QRow0 = uint32 q0
                                Past = uint32 past
                                QHeads = uint32 qh
                                KvHeads = uint32 kvh
                                HeadDim = uint32 hd
                                Scale = scale
                            })
                    |> ignore

                let shape =
                    {
                        QHeads = qh
                        KvHeads = kvh
                        HeadDim = hd
                    }

                let ks = HostTensor.f32 (Shape.matrix tokens kvc) k
                let vs = HostTensor.f32 (Shape.matrix tokens kvc) v

                let want =
                    CpuRef.attention<float32> exec (HostTensor.f32 (Shape.matrix tokens qc) q) ks vs shape 0 scale

                attend tokens 0 0
                let prefill = f.F32("y", q.Length)
                eqBits prefill want.Data "prefill"
                // Chunked: rows 0..3, then 4..8 with q_row0 = past = 4.
                f.UploadF32("y", Array.zeroCreate q.Length)
                attend 4 0 0
                attend 5 4 4
                eqBits (f.F32("y", q.Length)) prefill "chunked prefill = prefill"
                // Decode: the last row alone.
                f.UploadF32("y", Array.zeroCreate q.Length)
                attend 1 8 8
                eqBits (Array.sub (f.F32("y", q.Length)) (8 * qc) qc) (Array.sub prefill (8 * qc) qc) "decode = prefill"
        ]

let private foldingTests =
    testList
        "grid folding"
        [
            testCase "scale_f32 over 65,536 groups runs folded (32768 x 2) and computes every element once"
            <| fun () ->
                let n = 65535 * 256 + 1

                let f =
                    fixtureWith CpuDeviceOptions.Default (debug ()) [ t "x" DType.F32 (int64 n) ]

                let x = Array.init n (fun i -> float32 (i % 1000))
                f.UploadF32("x", x)

                f.Run(fun c -> Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "x") (f.Token "x") (uint32 n) 2.0f))
                |> ignore

                let y = f.F32("x", n)
                Expect.isTrue (Array.forall2 (fun a b -> a = 2.0f * b) y x) "in place, every element doubled once"
                let tr = (f.Gpu :?> CpuDeviceBackend).Trace |> List.exactlyOne
                Expect.equal (tr.Grid.X, tr.Grid.Y, tr.Grid.Z) (32768, 2, 1) "folded"
                Expect.equal tr.Words[0] 32768u "word 0 is the physical x"

            testCase "groups_echo over 65,537 groups: the extra group of the folded grid writes nothing"
            <| fun () ->
                let f = fixtureWith CpuDeviceOptions.Default (debug ()) [ t "y" DType.U32 65540L ]

                f.Run(fun c -> Kernels.GroupsEcho.dispatch c { Y = f.Token "y"; Groups = 65537u })
                |> ignore

                let y = f.Words("y", 65540)
                Expect.equal (Array.sub y 0 65537) (Array.init 65537 uint32) "y[g] = g"
                Expect.equal (Array.sub y 65537 3) (Array.create 3 0x7FBADBADu) "untouched"
                let tr = (f.Gpu :?> CpuDeviceBackend).Trace |> List.exactlyOne
                Expect.equal (tr.Grid.X, tr.Grid.Y) (32769, 2) "32769 x 2 = 65538 groups, one extra"

            testCase "row kernels fold too: argmax_rows over 70,000 rows"
            <| fun () ->
                let r = rng "fold-argmax"
                let x = uniform r 140000 -1.0f 1.0f

                let f =
                    fixtureWith CpuDeviceOptions.Default (debug ()) [ t "x" DType.F32 140000L; t "y" DType.U32 70000L ]

                f.UploadF32("x", x)

                f.Run(fun c ->
                    Kernels.ArgmaxRows.dispatch
                        c
                        {
                            X = f.Token "x"
                            Y = f.Token "y"
                            Rows = 70000u
                            Cols = 2u
                        })
                |> ignore

                Expect.equal
                    (f.Words("y", 70000))
                    (CpuRef.argmaxRows<float32>(HostTensor.f32 (Shape.matrix 70000 2) x))
                    "every row"

                let tr = (f.Gpu :?> CpuDeviceBackend).Trace |> List.exactlyOne
                Expect.equal (tr.Grid.X, tr.Grid.Y) (35000, 2) "folded"
        ]

let private modelTests =
    let withModel (cpu: CpuDeviceOptions) =
        fixtureWith cpu (DeviceOptions.For ExecutionBackend.CpuDevice)

    testList
        "arithmetic models"
        [
            testCase "the default CpuDevice classifies as preserve on the probe, a flush model as flush"
            <| fun () ->
                let probe cpu =
                    let f = withModel cpu [ t "x" DType.F32 7L; t "y" DType.F32 7L ]
                    f.UploadU32("x", Fp32Denormals.probeInputs)

                    f.Run(fun c ->
                        Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "x") (f.Token "y") 7u Fp32Denormals.ProbeScale))
                    |> ignore

                    Fp32Denormals.classify (f.Words("y", 7))

                Expect.equal (probe CpuDeviceOptions.Default) Fp32DenormalMode.Preserve "preserve"

                Expect.equal
                    (probe
                        { CpuDeviceOptions.Default with
                            Fp32Denormals = Fp32DenormalMode.Flush
                        })
                    Fp32DenormalMode.Flush
                    "flush"

            testCase "flush: rmsnorm_rows sees denormal inputs as zero and flushes its results"
            <| fun () ->
                let x = ofBits [| 0x00000005u; 0x3F800000u; 0x80000003u; 0x40000000u |]
                let w = [| 0x3F80us; 0x0001us; 0x3F80us; 0x3F80us |]

                let f =
                    withModel
                        { CpuDeviceOptions.Default with
                            Fp32Denormals = Fp32DenormalMode.Flush
                        }
                        [ t "x" DType.F32 4L; t "w" DType.BF16 4L; t "y" DType.F32 4L ]

                f.UploadF32("x", x)
                f.UploadBf16("w", w)

                f.Run(fun c ->
                    Kernels.RmsnormRows.dispatch
                        c
                        {
                            X = f.Token "x"
                            W = f.Token "w"
                            Y = f.Token "y"
                            Rows = 1u
                            Cols = 4u
                            Eps = 1e-6f
                        })
                |> ignore

                let fx = x |> f32Bits |> Array.map Fp32Denormals.flushBits |> ofBits

                let fw =
                    w |> Array.map (fun b -> if b &&& 0x7F80us = 0us then b &&& 0x8000us else b)

                let want =
                    CpuRef.rmsNorm<float32>
                        exec
                        (HostTensor.f32 (Shape.matrix 1 4) fx)
                        (HostTensor.bf16 (Shape.vector 4) fw)
                        1e-6f

                Expect.equal
                    (f.Words("y", 4))
                    (want.Data |> f32Bits |> Array.map Fp32Denormals.flushBits)
                    "flushed reference"

            testCase "canonical NaN: fp32 arithmetic returns 7fffffff; bit-moving kernels keep payloads"
            <| fun () ->
                let f =
                    withModel
                        { CpuDeviceOptions.Default with
                            CanonicalNaN = true
                        }
                        [
                            t "x" DType.F32 3L
                            t "y" DType.F32 3L
                            t "b" DType.BF16 2L
                            t "z" DType.F32 2L
                        ]

                f.UploadU32("x", [| 0x7FC00123u; 0xFFC00000u; 0x3F800000u |])
                f.UploadBf16("b", [| 0x7FC1us; 0xFF81us |])

                f.Run(fun c ->
                    Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "x") (f.Token "y") 3u 2.0f)

                    Kernels.Bf16Unpack.dispatch
                        c
                        {
                            X = f.Token "b"
                            Y = f.Token "z"
                            N = 2u
                        })
                |> ignore

                Expect.equal (f.Words("y", 3)) [| 0x7FFFFFFFu; 0x7FFFFFFFu; 0x40000000u |] "canonical"
                Expect.equal (f.Words("z", 2)) [| 0x7FC10000u; 0xFF810000u |] "payloads move"
        ]

let tests = testList "emulators" [ kernelTests; foldingTests; modelTests ]
