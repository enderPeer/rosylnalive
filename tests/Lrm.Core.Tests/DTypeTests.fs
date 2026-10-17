module Lrm.Core.Tests.DTypeTests

open System
open Expecto
open Lrm.Core

let tests =
    testList
        "dtype"
        [
            testCase "codes are frozen"
            <| fun () ->
                let expected =
                    [
                        DType.F32, 1uy, "f32", "F32", 4
                        DType.BF16, 2uy, "bf16", "BF16", 2
                        DType.F16, 3uy, "f16", "F16", 2
                        DType.I32, 4uy, "i32", "I32", 4
                        DType.U32, 5uy, "u32", "U32", 4
                        DType.U8, 6uy, "u8", "U8", 1
                        DType.F64, 7uy, "f64", "F64", 8
                        DType.I64, 8uy, "i64", "I64", 8
                    ]

                Expect.equal (DType.all |> List.length) expected.Length "count"

                for t, code, name, st, size in expected do
                    Expect.equal (DType.code t) code (sprintf "code %A" t)
                    Expect.equal (DType.ofCode (int code)) t (sprintf "ofCode %d" code)
                    Expect.equal (DType.name t) name (sprintf "name %A" t)
                    Expect.equal (DType.tryParse name) (Some t) (sprintf "parse %s" name)
                    Expect.equal (DType.safetensorsName t) st (sprintf "safetensors %A" t)
                    Expect.equal (DType.tryOfSafetensorsName st) (Some t) (sprintf "safetensors parse %s" st)
                    Expect.equal (DType.size t) size (sprintf "size %A" t)

            testCase "code 0 and unassigned codes are invalid"
            <| fun () ->
                for c in [ 0; 9; 255; -1 ] do
                    Expect.equal (DType.tryOfCode c) None (sprintf "code %d" c)

                Expect.throwsT<ArgumentException> (fun () -> DType.ofCode 0 |> ignore) "ofCode 0"

            testCase "names are matched ordinally"
            <| fun () ->
                Expect.equal (DType.tryParse "F32") None "upper case is the safetensors spelling"
                Expect.equal (DType.tryOfSafetensorsName "bf16") None "lower case is the LRM spelling"
                Expect.equal (DType.tryOfSafetensorsName "BOOL") None "unused safetensors type"

            testCase "float classification"
            <| fun () ->
                Expect.equal
                    (DType.all |> List.filter DType.isFloat)
                    [ DType.F32; DType.BF16; DType.F16; DType.F64 ]
                    "floats"
        ]
