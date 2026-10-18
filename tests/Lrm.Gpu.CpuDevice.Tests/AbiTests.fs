/// The ABI known-answer vectors of KernelAbiAndCatalog (A1 to A4) and HeapLayoutAndTensorTable (T1, T2) decode in
/// CpuDevice: it reads the real 128-byte blocks and the real 64-byte rows from the table image in slot 0.
module Lrm.Gpu.CpuDevice.Tests.AbiTests

open System
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.CpuDevice.Tests.Common
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// The words of a vector from docs/contracts/kernel-abi.md: the listed words, then zeros to 32.
let private words (text: string) =
    let listed =
        text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
        |> Array.map (fun w -> BitConverter.ToUInt32(Convert.FromHexString w, 0))

    Array.append listed (Array.zeroCreate (32 - listed.Length))

let private a1 = words "04000000 05000000 06000000 e8030000 0000003f"

let private a2 =
    words "46090000 01000000 02000000 03000000 25000000 80510200 00040000"

let private a3 = words "00c00000 01000000 02000000 03000000 00008001"

let private a4 =
    words "07000000 01000000 02000000 03000000 04000000 07000000 00000000 00000000 10000000 08000000 80000000 f304b53d"

/// A block from a vector's bytes with word 0 cleared, as a caller passes it.
let private passed (w: uint32[]) =
    let b = Array.zeroCreate<byte> 128

    for i in 0..31 do
        BitConverter.TryWriteBytes(Span(b, 4 * i, 4), (if i = 0 then 0u else w[i]))
        |> ignore

    ArgBlock.OfBytes(ReadOnlySpan b)

let private debug () =
    { DeviceOptions.For ExecutionBackend.CpuDevice with
        Debug = true
    }

let tests =
    testList
        "abi"
        [
            testCase "A1: scale_f32 {x = t5; y = t6; n = 1000; a = 0.5} runs with the vector's words and grid (4, 1, 1)"
            <| fun () ->
                let f =
                    fixtureWith
                        CpuDeviceOptions.Default
                        (debug ())
                        [
                            for i in 1..4 do
                                yield t (sprintf "pad%d" i) DType.F32 4L
                            yield t "x" DType.F32 1000L
                            yield t "y" DType.F32 1024L
                        ]

                Expect.equal ((f.Token "x").Value, (f.Token "y").Value) (5u, 6u) "tokens 5 and 6"
                let r = rng "a1"
                let x = uniform r 1000 -4.0f 4.0f
                f.UploadF32("x", x)
                f.Run(fun c -> c.Dispatch(Kernels.ScaleF32.Id, passed a1)) |> ignore
                let tr = (f.Gpu :?> CpuDeviceBackend).Trace |> List.exactlyOne
                Expect.equal tr.Words a1 "the decoded words are vector A1, word 0 included"
                Expect.equal (tr.Grid.X, tr.Grid.Y, tr.Grid.Z) (4, 1, 1) "grid"
                let y = f.Words("y", 1024)
                Expect.equal (Array.sub y 0 1000) (x |> Array.map (fun v -> v * 0.5f) |> f32Bits) "y = 0.5 x"
                Expect.equal (Array.sub y 1000 24) (Array.create 24 0x7FBADBADu) "nothing past n"

            testCase "A2 and A3 decode to their typed records, footprints and folded grids"
            <| fun () ->
                let m = Kernels.MatmulBf16w.decode (passed a2)
                Expect.equal (m.M, m.N, m.K) (37u, 151936u, 1024u) "A2 fields"

                let fp =
                    KernelRules.footprint (Kernels.MatmulBf16w.spec ()) (passed a2) (fun _ -> None)

                Expect.equal
                    (fp |> List.map (fun f -> f.Field, f.Elements.End))
                    [ "x", 37L * 1024L; "w", 151936L * 1024L; "y", 37L * 151936L ]
                    "A2 counts"

                Expect.equal
                    (Grid.physical false (Kernels.MatmulBf16w.grid m))
                    (Ok { X = 2374; Y = 1; Z = 1 })
                    "A2 grid"

                let a = Kernels.Add.decode (passed a3)
                Expect.equal a.N (8192u * 3072u) "A3 n"

                Expect.equal
                    (Grid.physical true (Kernels.Add.grid a))
                    (Ok { X = 49152; Y = 2; Z = 1 })
                    "A3 folds 98,304 groups"

                Expect.equal (words "00c00000").[0] 49152u "A3 word 0"

            testCase "A4: attention_gqa_causal runs with the vector's words, grid (7, 16, 1), equal to CpuRef"
            <| fun () ->
                let qc, kvc = 16 * 128, 8 * 128

                let f =
                    fixtureWith
                        CpuDeviceOptions.Default
                        (debug ())
                        [
                            t "q" DType.F32 (7L * int64 qc)
                            t "k" DType.F32 (7L * int64 kvc)
                            t "v" DType.F32 (7L * int64 kvc)
                            t "y" DType.F32 (7L * int64 qc)
                        ]

                let r = rng "a4"
                let q = uniform r (7 * qc) -1.0f 1.0f
                let k = uniform r (7 * kvc) -1.0f 1.0f
                let v = uniform r (7 * kvc) -1.0f 1.0f
                f.UploadF32("q", q)
                f.UploadF32("k", k)
                f.UploadF32("v", v)
                f.Run(fun c -> c.Dispatch(Kernels.AttentionGqaCausal.Id, passed a4)) |> ignore
                let tr = (f.Gpu :?> CpuDeviceBackend).Trace |> List.exactlyOne
                Expect.equal tr.Words a4 "the decoded words are vector A4"
                Expect.equal (tr.Grid.X, tr.Grid.Y, tr.Grid.Z) (7, 16, 1) "grid"

                Expect.equal
                    (BitConverter.UInt32BitsToSingle a4[11])
                    (CpuRef.attentionScale<float32> 128)
                    "scale = f32(1 / sqrt 128)"

                let want =
                    CpuRef.attention<float32>
                        CpuExec.Serial
                        (HostTensor.f32 (Shape.matrix 7 qc) q)
                        (HostTensor.f32 (Shape.matrix 7 kvc) k)
                        (HostTensor.f32 (Shape.matrix 7 kvc) v)
                        {
                            QHeads = 16
                            KvHeads = 8
                            HeadDim = 128
                        }
                        0
                        (CpuRef.attentionScale<float32> 128)

                Expect.equal (f.Words("y", 7 * qc)) (f32Bits want.Data) "y"

            testCase "T1 and T2: table_echo returns the vector rows, decoded from the table image in slot 0"
            <| fun () ->
                // Slot 1 holds T1 (a 297 MiB page that is never touched: table_echo reads only the row), slot 4 holds
                // T2, slot 5 the output.
                let t1Page = Heap.alignUp Heap.PageGranularity (256L + 151936L * 1024L * 2L)

                let layout =
                    HeapLayout.ofPages
                        [
                            ArenaKind.Weights, t1Page
                            ArenaKind.Activations, 65536L
                            ArenaKind.Activations, 65536L
                            ArenaKind.Activations, 65536L
                            ArenaKind.Scratch, 65536L
                        ]

                let g =
                    CpuDevice.create CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice)

                let h = g.CreateHeap layout
                g.LoadKernels(Lrm.Shaders.Shaders.embedded ()) |> ignore
                let table = TensorTable layout

                let tok1 =
                    table.Add(
                        "model.embed_tokens.weight",
                        TensorRow.create
                            1
                            256L
                            DType.BF16
                            [ 151936L; 1024L ]
                            TensorFlags.ReadOnly
                            (TensorRow.debugIdOf "model.embed_tokens.weight")
                    )

                let tok2 =
                    table.Add(
                        "L0.attn.q_rope",
                        TensorRow.create
                            4
                            4096L
                            DType.F32
                            [ 7L; 16L; 128L ]
                            TensorFlags.View
                            (TensorRow.debugIdOf "L0.attn.q_rope")
                    )

                let y = table.Add("y", TensorRow.create 5 0L DType.U32 [ 32L ] TensorFlags.None 0u)
                h.WriteTable(table.Seal())

                let echo (src: TensorToken) =
                    let c = g.Begin()
                    Kernels.TableEcho.dispatch c { Src = src; Y = y }
                    g.Wait(g.Submit c) |> ignore
                    let b = Array.zeroCreate<byte> 64
                    h.Download(5, 0L, Span b)
                    hex b

                Expect.equal
                    (echo tok1)
                    ("01000000000100000200000002000000"
                     + "80510200000400000100000001000000"
                     + "00040000010000000000000000000000"
                     + "0000460903000000fcd443ac00000000")
                    "T1"

                Expect.equal
                    (echo tok2)
                    ("04000000001000000100000003000000"
                     + "07000000100000008000000001000000"
                     + "00080000800000000100000000000000"
                     + "0038000005000000d4fd861100000000")
                    "T2"

                Expect.equal (echo TensorToken.Null) (String.replicate 128 "0") "the null row"
        ]
