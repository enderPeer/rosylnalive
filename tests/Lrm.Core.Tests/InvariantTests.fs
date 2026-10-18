module Lrm.Core.Tests.InvariantTests

open System
open Expecto
open Lrm.Core
open Lrm.Testing

let tests =
    testList
        "invariant"
        [
            testCase "R formatting of float32"
            <| fun () ->
                let cases =
                    [
                        1.5f, "1.5"
                        -0.0f, "-0"
                        0.1f, "0.1"
                        1e-5f, "1E-05"
                        3.4028235e38f, "3.4028235E+38"
                        1.401298464e-45f, "1E-45"
                        Single.NaN, "NaN"
                        Single.PositiveInfinity, "Infinity"
                        Single.NegativeInfinity, "-Infinity"
                    ]

                for x, expected in cases do
                    Expect.equal (Invariant.f32 x) expected (sprintf "f32 %08x" (BitConverter.SingleToUInt32Bits x))

            testCase "R formatting of float"
            <| fun () ->
                let cases =
                    [
                        1.5, "1.5"
                        -0.0, "-0"
                        0.1, "0.1"
                        1.0 / 3.0, "0.3333333333333333"
                        1e300, "1E+300"
                        5e-324, "5E-324"
                        Double.NaN, "NaN"
                        Double.PositiveInfinity, "Infinity"
                    ]

                for x, expected in cases do
                    Expect.equal (Invariant.f64 x) expected (sprintf "f64 %016x" (BitConverter.DoubleToUInt64Bits x))

            Props.testProperty "every float32 round-trips through f32 and parseF32"
            <| fun (x: float32) ->
                let back = Invariant.parseF32 (Invariant.f32 x)

                if Single.IsNaN x then
                    Single.IsNaN back
                else
                    BitConverter.SingleToUInt32Bits back = BitConverter.SingleToUInt32Bits x

            Props.testProperty "every float round-trips through f64 and parseF64"
            <| fun (x: float) ->
                let back = Invariant.parseF64 (Invariant.f64 x)

                if Double.IsNaN x then
                    Double.IsNaN back
                else
                    BitConverter.DoubleToUInt64Bits back = BitConverter.DoubleToUInt64Bits x

            Props.testProperty "float32 bit patterns round-trip through f32"
            <| fun (bits: uint32) ->
                let x = BitConverter.UInt32BitsToSingle bits

                Single.IsNaN x
                || BitConverter.SingleToUInt32Bits(Invariant.parseF32 (Invariant.f32 x)) = bits

            testCase "every NaN formats as NaN and parses to the default NaN"
            <| fun () ->
                // Sign and payload are lost: R text has a single NaN spelling.
                for bits in [ 0x7FC00000u; 0xFFC00000u; 0x7F800001u; 0x7FC12345u; 0xFFFFFFFFu ] do
                    let x = BitConverter.UInt32BitsToSingle bits
                    Expect.equal (Invariant.f32 x) "NaN" (sprintf "f32 %08x" bits)

                Expect.equal
                    (BitConverter.SingleToUInt32Bits(Invariant.parseF32 "NaN"))
                    (BitConverter.SingleToUInt32Bits Single.NaN)
                    "parseF32"

                for bits in [ 0x7FF8000000000000UL; 0xFFF8000000000000UL; 0x7FF0000000000001UL ] do
                    let x = BitConverter.UInt64BitsToDouble bits
                    Expect.equal (Invariant.f64 x) "NaN" (sprintf "f64 %016x" bits)

                Expect.equal
                    (BitConverter.DoubleToUInt64Bits(Invariant.parseF64 "NaN"))
                    (BitConverter.DoubleToUInt64Bits Double.NaN)
                    "parseF64"

            testCase "integers and interpolation"
            <| fun () ->
                Expect.equal (Invariant.i64 -1234567L) "-1234567" "i64"
                Expect.equal (Invariant.u64 UInt64.MaxValue) "18446744073709551615" "u64"
                Expect.equal (Invariant.format $"{1.5} and {2.25f}") "1.5 and 2.25" "format"
                Expect.equal (Invariant.parseI32 "-42") -42 "parseI32"
                Expect.equal (Invariant.tryParseU64 "-1") None "u64 has no sign"
                Expect.equal (Invariant.tryParseF64 "1,5") None "comma is not a decimal separator"

            testCase "hex"
            <| fun () ->
                Expect.equal (Invariant.hex [| 0x00uy; 0xABuy; 0xffuy |]) "00abff" "lowercase"
                Expect.equal (Invariant.ofHex "00ABff") [| 0x00uy; 0xABuy; 0xffuy |] "either case"
                Expect.equal (Invariant.tryOfHex "abc") None "odd length"
                Expect.equal (Invariant.tryOfHex "0x00") None "no prefix"
                Expect.equal (Invariant.tryOfHex "zz") None "not hex"

            Props.testProperty "hex round-trips"
            <| fun (bytes: byte[]) -> Invariant.ofHex (Invariant.hex bytes) = bytes
        ]
