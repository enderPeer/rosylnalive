module Lrm.Gpu.Core.Tests.ArgBlockTests

open System
open Expecto
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common
open Lrm.Testing

let private zeros (words: int) = String.replicate words "00000000"

/// KAT A1 (docs/contracts/kernel-abi.md): scale_f32 { x = t5; y = t6; n = 1000; a = 0.5 } as dispatched:
/// word 0 = ceil(1000 / 256) = 4. Independently computed with Python struct.
let a1Hex =
    "04000000" + "05000000" + "06000000" + "e8030000" + "0000003f" + zeros 27

/// KAT A2: matmul_bf16w { x = t1; w = t2; y = t3; m = 37; n = 151936; k = 1024 }: grid (2374, 1, 1), word 0 = 2374.
let a2Hex =
    "46090000"
    + "01000000"
    + "02000000"
    + "03000000"
    + "25000000"
    + "80510200"
    + "00040000"
    + zeros 25

/// KAT A3: add { a = t1; b = t2; y = t3; n = 8192 * 3072 }: 98,304 logical groups fold to (49152, 2, 1),
/// so word 0 = 49152.
let a3Hex =
    "00c00000" + "01000000" + "02000000" + "03000000" + "00008001" + zeros 27

/// KAT A4: attention_gqa_causal { q..y = t1..t4; rows = 7; q_row0 = 0; past = 0; q_heads = 16; kv_heads = 8;
/// head_dim = 128; scale = f32(1 / sqrt 128) = 0x3db504f3 }: grid (7, 16, 1), word 0 = 7.
let a4Hex =
    "07000000"
    + "01000000"
    + "02000000"
    + "03000000"
    + "04000000"
    + "07000000"
    + "00000000"
    + "00000000"
    + "10000000"
    + "08000000"
    + "80000000"
    + "f304b53d"
    + zeros 20

/// The block as Dispatch would send it: the catalog grid, folded, written into word 0.
let dispatched (kernel: KernelId) (args: ArgBlock) =
    let spec = Kernels.all[kernel.Value]

    match Grid.physical spec.Linear (KernelSpec.gridOf spec args) with
    | Ok p -> args.WithGroupsX(uint32 p.X), p
    | Error e -> failtest e

let tests =
    testList
        "arg-block"
        [
            testCase "KAT A1: scale_f32"
            <| fun () ->
                let args =
                    Kernels.ScaleF32.encode
                        {
                            X = TensorToken 5u
                            Y = TensorToken 6u
                            N = 1000u
                            A = 0.5f
                        }

                Expect.equal args[0] 0u "encode leaves word 0 for Dispatch"
                let block, p = dispatched Kernels.ScaleF32.Id args
                Expect.equal (hex (block.ToBytes())) a1Hex "bytes"
                Expect.equal (p.X, p.Y, p.Z) (4, 1, 1) "grid"

            testCase "KAT A2: matmul_bf16w for lm_head at M = 37"
            <| fun () ->
                let args =
                    Kernels.MatmulBf16w.encode
                        {
                            X = TensorToken 1u
                            W = TensorToken 2u
                            Y = TensorToken 3u
                            M = 37u
                            N = 151936u
                            K = 1024u
                        }

                let block, p = dispatched Kernels.MatmulBf16w.Id args
                Expect.equal (hex (block.ToBytes())) a2Hex "bytes"
                Expect.equal (p.X, p.Y, p.Z) (2374, 1, 1) "grid"

            testCase "KAT A3: add folds 98,304 groups into (49152, 2, 1)"
            <| fun () ->
                let args =
                    Kernels.Add.encode
                        {
                            A = TensorToken 1u
                            B = TensorToken 2u
                            Y = TensorToken 3u
                            N = 8192u * 3072u
                        }

                let block, p = dispatched Kernels.Add.Id args
                Expect.equal (hex (block.ToBytes())) a3Hex "bytes"
                Expect.equal (p.X, p.Y, p.Z) (49152, 2, 1) "folded"

            testCase "KAT A4: attention_gqa_causal prefill of 7 rows"
            <| fun () ->
                let args =
                    Kernels.AttentionGqaCausal.encode
                        {
                            Q = TensorToken 1u
                            K = TensorToken 2u
                            V = TensorToken 3u
                            Y = TensorToken 4u
                            Rows = 7u
                            QRow0 = 0u
                            Past = 0u
                            QHeads = 16u
                            KvHeads = 8u
                            HeadDim = 128u
                            Scale = float32 (1.0 / sqrt 128.0)
                        }

                let block, p = dispatched Kernels.AttentionGqaCausal.Id args
                Expect.equal (hex (block.ToBytes())) a4Hex "bytes"
                Expect.equal (p.X, p.Y, p.Z) (7, 16, 1) "one group per (row, head)"

            testCase "f32 fields keep their exact bits"
            <| fun () ->
                let enc (a: float32) =
                    (Kernels.ScaleF32.encode
                        {
                            X = TensorToken 1u
                            Y = TensorToken 2u
                            N = 1u
                            A = a
                        })[4]

                Expect.equal (enc -0.0f) 0x80000000u "negative zero"
                Expect.equal (enc (BitConverter.UInt32BitsToSingle 0x7fc00001u)) 0x7fc00001u "NaN payload"
                Expect.equal (enc (BitConverter.UInt32BitsToSingle 1u)) 1u "smallest denormal"

            testCase "bytes round-trip and the block is 128 bytes"
            <| fun () ->
                Expect.equal ArgBlock.ByteCount 128 "128 bytes"
                Expect.equal ArgBlock.WordCount 32 "32 words"
                let w = Array.init 32 (fun i -> uint32 (i * 0x01010101))
                let b = ArgBlock.OfWords w
                Expect.equal (ArgBlock.OfBytes(ReadOnlySpan(b.ToBytes()))) b "round trip"
                Expect.equal (b.ToBytes()[4..7]) [| 1uy; 1uy; 1uy; 1uy |] "little-endian word 1"
                raisesContract "32 words" (fun () -> ArgBlock.OfWords(Array.zeroCreate 31))
                raisesContract "128 bytes" (fun () -> ArgBlock.OfBytes(ReadOnlySpan(Array.zeroCreate 127)))

            Props.testProperty "folded grids cover the logical grid with fewer than y extra groups"
            <| fun (seed: uint32) ->
                let x = 1L + int64 seed % (Grid.MaxDimension * Grid.MaxDimension)

                match Grid.physical true (Grid.create x 1L 1L) with
                | Error e -> failtest e
                | Ok p ->
                    let total = int64 p.X * int64 p.Y

                    p.X >= 1
                    && int64 p.X <= Grid.MaxDimension
                    && int64 p.Y <= Grid.MaxDimension
                    && p.Z = 1
                    && total >= x
                    && total - x < int64 p.Y
                    && (x > Grid.MaxDimension || (int64 p.X = x && p.Y = 1))

            testCase "grid edge cases"
            <| fun () ->
                let ok (linear: bool) x y z =
                    match Grid.physical linear (Grid.create x y z) with
                    | Ok p -> p.X, p.Y, p.Z
                    | Error e -> failtest e

                let err (linear: bool) x y z (fragment: string) =
                    match Grid.physical linear (Grid.create x y z) with
                    | Ok p -> failtestf "expected an error, got %A" p
                    | Error e -> Expect.stringContains e fragment "reason"

                Expect.equal (ok true 65535L 1L 1L) (65535, 1, 1) "the largest unfolded x"
                Expect.equal (ok true 65536L 1L 1L) (32768, 2, 1) "the smallest folded x"
                Expect.equal (ok true 70000L 1L 1L) (35000, 2, 1) "groups_echo with 70,000 groups"
                Expect.equal (ok false 65535L 7L 3L) (65535, 7, 3) "a 3-D grid at the limit"
                err true 0L 1L 1L "empty"
                err false 65536L 2L 1L "not linear"
                err true 1L 65536L 1L "at most 65535"
                err true (65535L * 65535L + 1L) 1L 1L "65535 * 65535"
                Expect.equal (Grid.linearGroup { X = 35000; Y = 2; Z = 1 } 4999 1) 39999L "logical group"
        ]
