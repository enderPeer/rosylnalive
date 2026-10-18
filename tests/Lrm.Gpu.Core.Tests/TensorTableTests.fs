module Lrm.Gpu.Core.Tests.TensorTableTests

open System
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common

/// KAT T1 (docs/contracts/heap-and-tensor-table.md): the Qwen3-0.6B embedding, bf16 [151936, 1024], slot 1 at byte
/// 256, ReadOnly. Independently computed with Python struct and hashlib.
let t1Hex =
    "01000000" // slot 1
    + "00010000" // byte offset 256
    + "02000000" // dtype BF16
    + "02000000" // rank 2
    + "80510200" // dims[0] 151936
    + "00040000" // dims[1] 1024
    + "01000000" // dims[2] 1 (unused)
    + "01000000" // dims[3] 1 (unused)
    + "00040000" // strides[0] 1024
    + "01000000" // strides[1] 1
    + "00000000" // strides[2] 0 (unused)
    + "00000000" // strides[3] 0 (unused)
    + "00004609" // count 155582464
    + "03000000" // flags Contiguous | ReadOnly
    + "fcd443ac" // debug id of "model.embed_tokens.weight"
    + "00000000" // reserved

/// KAT T2: a rank-3 f32 view [7, 16, 128], slot 4 at byte 4096, flags Contiguous | View, debug id of "L0.attn.q_rope".
let t2Hex =
    "04000000"
    + "00100000"
    + "01000000"
    + "03000000"
    + "07000000"
    + "10000000"
    + "80000000"
    + "01000000"
    + "00080000"
    + "80000000"
    + "01000000"
    + "00000000"
    + "00380000"
    + "05000000"
    + "d4fd8611"
    + "00000000"

let tests =
    testList
        "tensor-table"
        [
            testCase "KAT T1: the embedding row encodes to the contract bytes"
            <| fun () ->
                let row =
                    TensorRow.create
                        1
                        256L
                        DType.BF16
                        [ 151936L; 1024L ]
                        TensorFlags.ReadOnly
                        (TensorRow.debugIdOf "model.embed_tokens.weight")

                Expect.equal (hex (row.ToBytes())) t1Hex "bytes"
                Expect.equal (TensorRow.OfBytes(ReadOnlySpan(unhex t1Hex))) row "decode"
                Expect.equal row.DebugId 0xac43d4fcu "debug id"

            testCase "KAT T2: a rank-3 view encodes to the contract bytes"
            <| fun () ->
                let row =
                    TensorRow.create
                        4
                        4096L
                        DType.F32
                        [ 7L; 16L; 128L ]
                        TensorFlags.View
                        (TensorRow.debugIdOf "L0.attn.q_rope")

                Expect.equal (hex (row.ToBytes())) t2Hex "bytes"
                Expect.equal row.Dims [ 7u; 16u; 128u ] "dims"
                Expect.equal row.Flags (TensorFlags.Contiguous ||| TensorFlags.View) "flags"

            testCase "the null row is all zeros and never a valid dtype"
            <| fun () ->
                Expect.equal (TensorRow.Zero.ToBytes()) (Array.zeroCreate 64) "zero"
                Expect.isNone TensorRow.Zero.DType "dtype code 0"
                Expect.isTrue TensorToken.Null.IsNull "token 0"
                Expect.isError (TensorRow.validate (smallLayout ()) TensorRow.Zero) "invalid"

            testCase "strided rows set Contiguous exactly for packed strides"
            <| fun () ->
                let packed =
                    TensorRow.strided 1 0L DType.F32 [ 2L; 3L ] [ 3L; 1L ] TensorFlags.None 0u

                let transposed =
                    TensorRow.strided 1 0L DType.F32 [ 2L; 3L ] [ 1L; 2L ] TensorFlags.None 0u

                Expect.equal packed.Flags TensorFlags.Contiguous "packed"
                Expect.equal transposed.Flags TensorFlags.None "not packed"
                Expect.equal (TensorRow.spanElements transposed) 6L "span of the transposed view"
                Expect.isOk (TensorRow.validate (smallLayout ()) transposed) "valid"

            testCase "validate enforces every row rule"
            <| fun () ->
                let l = smallLayout ()

                let bad (words: uint32[] -> unit) (fragment: string) =
                    let w = (TensorRow.create 2 0L DType.F32 [ 4L; 8L ] TensorFlags.None 0u).ToWords()
                    words w

                    match TensorRow.validate l (TensorRow.OfWords w) with
                    | Ok() -> failtestf "expected '%s'" fragment
                    | Error e -> Expect.stringContains e fragment "reason"

                bad (fun w -> w[0] <- 0u) "outside 1..63"
                bad (fun w -> w[0] <- 9u) "holds no page"
                bad (fun w -> w[1] <- 2u) "multiple of 4"
                bad (fun w -> w[2] <- 9u) "not a DType"
                bad (fun w -> w[3] <- 0u) "rank 0"
                bad (fun w -> w[3] <- 5u) "rank 5"
                bad (fun w -> w[5] <- 0u) "dims[1] is 0"
                bad (fun w -> w[6] <- 2u) "unused dims[2]"
                bad (fun w -> w[10] <- 1u) "unused dims[2]"
                bad (fun w -> w[12] <- 31u) "product of the dims"
                bad (fun w -> w[13] <- 0u) "Contiguous flag"
                bad (fun w -> w[13] <- w[13] ||| 8u) "reserved bits"
                bad (fun w -> w[15] <- 1u) "reserved"
                bad (fun w -> w[1] <- 262144u - 64u) "past the"

                let f64 = TensorRow.create 2 4L DType.F64 [ 2L ] TensorFlags.None 0u
                Expect.isError (TensorRow.validate l f64) "an f64 row needs an 8-byte aligned offset"

                let fits = TensorRow.create 2 (262144L - 128L) DType.F32 [ 32L ] TensorFlags.None 0u
                Expect.isOk (TensorRow.validate l fits) "ends exactly at the page end"

            testCase "an odd bf16 tensor's extent covers the whole last word"
            <| fun () ->
                let r = TensorRow.create 1 0L DType.BF16 [ 3L ] TensorFlags.None 0u
                Expect.equal (TensorRow.extentBytes r) 8L "6 bytes round up to 8"

            testCase "the table gives tokens 1, 2, ... and seals once"
            <| fun () ->
                let l = smallLayout ()
                let t = TensorTable l

                let a =
                    t.Add("a", TensorRow.create 2 0L DType.F32 [ 16L ] TensorFlags.None (TensorRow.debugIdOf "a"))

                let b =
                    t.Add("b", TensorRow.create 2 256L DType.F32 [ 16L ] TensorFlags.None (TensorRow.debugIdOf "b"))

                Expect.equal (a, b) (TensorToken 1u, TensorToken 2u) "tokens"
                raisesContract "already in the table" (fun () -> t.Add("a", t.Row a))
                raisesContract "needs a name" (fun () -> t.Add("", t.Row a))

                raisesContract "slot 0" (fun () ->
                    t.Add("c", TensorRow.create 0 0L DType.F32 [ 1L ] TensorFlags.None 0u))

                let s = t.Seal()
                raisesContract "sealed" (fun () -> t.Seal())
                raisesContract "sealed" (fun () -> t.Add("d", t.Row a))
                Expect.equal s.Count 2 "count"
                Expect.equal (s.Row a) (t.Row a) "row"
                Expect.equal (s.Row(TensorToken 9u)) TensorRow.Zero "unused rows are zero"
                Expect.isFalse (s.IsAssigned(TensorToken 9u)) "unused"
                Expect.isFalse (s.IsAssigned TensorToken.Null) "null"
                Expect.equal (s.TryFind "b") (Some b) "by name"
                Expect.equal (s.Name b) "b" "name"
                raisesContract "outside 0..16383" (fun () -> s.Row(TensorToken 16384u))
                let image = s.Image.ToArray()
                Expect.equal image.Length (int Heap.TableBytes) "1 MiB"
                Expect.equal image[0..63] (Array.zeroCreate 64) "row 0 is null"
                Expect.equal image[64..127] ((s.Row a).ToBytes()) "row 1 at byte 64"
                Expect.equal image[128..191] ((s.Row b).ToBytes()) "row 2 at byte 128"
                Expect.isTrue (image[192..] |> Array.forall ((=) 0uy)) "the rest is zero"
                Expect.equal s.Id (Hash.sha256 image) "id"

            testCase "placements and views"
            <| fun () ->
                let p = HeapPlanner()
                let x = p.Allocate(ArenaKind.Activations, 7L * 2048L * 4L)
                let w = p.Allocate(ArenaKind.Weights, 1024L * 2L)
                let l = p.Build()
                let t = TensorTable l
                let xt = t.Add("x", x, DType.F32, [ 7L; 2048L ], TensorFlags.None)
                let wt = t.Add("w", w, DType.BF16, [ 1024L ], TensorFlags.ReadOnly)
                Expect.equal (t.Row xt).Slot (l.SlotOf(ArenaKind.Activations, 0)) "slot from the placement"
                Expect.equal (t.Row wt).Slot 1 "weights come first"
                let heads = t.AddView("x.heads", xt, [ 7L * 16L; 128L ], 0L)
                Expect.equal (t.Row heads).Flags (TensorFlags.Contiguous ||| TensorFlags.View) "view flags"
                let last = t.AddView("x.last", xt, [ 1L; 2048L ], 6L * 2048L)
                Expect.equal (t.Row last).ByteOffset ((t.Row xt).ByteOffset + uint32 (6 * 2048 * 4)) "row 6"
                let wv = t.AddView("w.half", wt, [ 512L ], 512L)
                Expect.isTrue (TensorFlags.has TensorFlags.ReadOnly (t.Row wv).Flags) "ReadOnly is inherited"
                raisesContract "does not lie inside" (fun () -> t.AddView("x.past", xt, [ 2L; 2048L ], 6L * 2048L))
                raisesContract "placement has" (fun () -> t.Add("big", x, DType.F32, [ 8L; 2048L ], TensorFlags.None))

            testCase "the table holds 16,383 real rows"
            <| fun () ->
                let l = smallLayout ()
                let t = TensorTable l

                for i in 1..16383 do
                    t.Add(string i, TensorRow.create 3 0L DType.F32 [ 1L ] TensorFlags.None 0u)
                    |> ignore

                raisesContract "full" (fun () -> t.Add("x", TensorRow.create 3 0L DType.F32 [ 1L ] TensorFlags.None 0u))
                Expect.equal (t.Seal().Count) 16383 "count"
        ]
