/// HostTensorAndSafetensors: Shape, TensorDesc and HostTensor.
module Lrm.Tensor.Tests.TensorTests

open System
open Expecto
open Lrm.Core
open Lrm.Tensor

let tests =
    testList
        "tensor"
        [
            testCase "shapes count elements, give row-major strides and print invariantly"
            <| fun () ->
                let s = Shape.ofList [ 2; 3; 4 ]
                Expect.equal s.Elements 24L "elements"
                Expect.sequenceEqual s.RowMajorStrides [| 12L; 4L; 1L |] "strides"
                Expect.equal (s.Rows, s.Cols) (2L, 12L) "rows and cols"
                Expect.equal (string s) "[2, 3, 4]" "text"
                Expect.equal Shape.Scalar.Elements 1L "scalar"
                Expect.equal (Shape.ofList [ 5; 0 ]).Elements 0L "empty"
                Expect.equal (Shape.ofList [ 2; 3 ]) (Shape.matrix 2 3) "structural equality"
                Expect.notEqual (Shape.ofList [ 2; 3 ]) (Shape.ofList [ 3; 2 ]) "order matters"
                Expect.throwsT<ArgumentException> (fun () -> Shape.ofList [ 2; -1 ] |> ignore) "negative"

                Expect.throwsT<ArgumentException>
                    (fun () -> Shape.ofArray [| Int64.MaxValue; 2L |] |> ignore)
                    "overflow"

                let d = s.Dims
                d[0] <- 99L
                Expect.equal s[0] 2L "Dims is a copy"

            testCase "tensor descriptions know contiguity, extents and byte sizes"
            <| fun () ->
                let c = TensorDesc.Contiguous(DType.BF16, Shape.matrix 3 5)
                Expect.isTrue c.IsContiguous "packed"
                Expect.equal (c.ByteSize, c.ExtentElements) (30L, 15L) "sizes"
                let v = TensorDesc.Strided(DType.F32, Shape.matrix 3 2, [ 5L; 1L ])
                Expect.isFalse v.IsContiguous "a column slice"
                Expect.equal v.ExtentElements 12L "1 + 2*5 + 1*1"

                Expect.isTrue
                    (TensorDesc.Strided(DType.F32, Shape.matrix 1 4, [ 99L; 1L ])).IsContiguous
                    "size-1 dims ignore their stride"

                Expect.equal (string c) "bf16[3, 5]" "text"

            testCase "host tensors check their element type and widen exactly"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () -> HostTensor.create DType.F64 (Shape.vector 1) [| 1.0f |] |> ignore)
                    "carrier"

                Expect.throwsT<ArgumentException>
                    (fun () -> HostTensor.f32 (Shape.vector 2) [| 1.0f |] |> ignore)
                    "count"

                let b = HostTensor.bf16 (Shape.vector 2) [| 0x3F80us; 0xC049us |]
                Expect.sequenceEqual ((b :> IHostTensor).ToFloat64()) [| 1.0; -3.140625 |] "bf16 widening"
                let i = HostTensor.i64 (Shape.vector 2) [| -5L; 1L <<< 53 |]
                Expect.sequenceEqual ((i :> IHostTensor).ToFloat64()) [| -5.0; 9007199254740992.0 |] "int64"

                Expect.throwsT<InvalidOperationException>
                    (fun () ->
                        (HostTensor.i64 (Shape.vector 1) [| (1L <<< 53) + 1L |] :> IHostTensor).ToFloat64()
                        |> ignore)
                    "inexact int64"

                let t = HostTensor.f32 (Shape.matrix 2 2) [| 1.0f; 2.0f; 3.0f; 4.0f |]

                let back =
                    HostTensor.ofBytes<float32> DType.F32 t.Shape (ReadOnlySpan((t :> IHostTensor).ToBytes()))

                Expect.sequenceEqual back.Data t.Data "bytes round trip"
                Expect.sequenceEqual (t.Row(1).ToArray()) [| 3.0f; 4.0f |] "row"
                Expect.equal (t.Reshape(Shape.vector 4)).Rows 4 "reshape"

                Expect.throwsT<ArgumentException>
                    (fun () -> HostTensor.ofBytes<float32> DType.F32 t.Shape (ReadOnlySpan [| 0uy |]) |> ignore)
                    "short bytes"
        ]
