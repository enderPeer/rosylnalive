module Lrm.Core.Tests.RngTests

open System
open Expecto
open Lrm.Core
open Lrm.Testing

let private words (b: Philox4x32Block) = [ b.X0; b.X1; b.X2; b.X3 ]

/// Random123 kat_vectors, philox4x32 10: (counter, key, expected output).
let philox10Kats =
    [
        (0u, 0u, 0u, 0u), (0u, 0u), [ 0x6627e8d5u; 0xe169c58du; 0xbc57ac4cu; 0x9b00dbd8u ]
        (0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
        (0xffffffffu, 0xffffffffu),
        [ 0x408f276du; 0x41c83b0eu; 0xa20bc7c6u; 0x6d5451fdu ]
        (0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u),
        (0xa4093822u, 0x299f31d0u),
        [ 0xd16cfe09u; 0x94fdccebu; 0x5001e420u; 0x24126ea1u ]
    ]

/// Random123 kat_vectors, philox4x32 7 (exercises the round count).
let philox7Kats =
    [
        (0u, 0u, 0u, 0u), (0u, 0u), [ 0x5f6fb709u; 0x0d893f64u; 0x4f121f81u; 0x4f730a48u ]
        (0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
        (0xffffffffu, 0xffffffffu),
        [ 0x5207ddc2u; 0x45165e59u; 0x4d8ee751u; 0x8c52f662u ]
        (0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u),
        (0xa4093822u, 0x299f31d0u),
        [ 0x4dfccabau; 0x190a87f0u; 0xc47362bau; 0xb6b5242au ]
    ]

let tests =
    testList
        "rng"
        [
            testCase "Philox4x32-10 matches the Random123 known answers"
            <| fun () ->
                for (c0, c1, c2, c3), (k0, k1), expected in philox10Kats do
                    Expect.equal (words (Philox4x32.block10 (c0, c1, c2, c3) (k0, k1))) expected (sprintf "%08x" c0)

            testCase "Philox4x32-7 matches the Random123 known answers"
            <| fun () ->
                for (c0, c1, c2, c3), (k0, k1), expected in philox7Kats do
                    Expect.equal (words (Philox4x32.generate 7 (c0, c1, c2, c3) (k0, k1))) expected (sprintf "%08x" c0)

            testCase "SplitMix64 matches the published sequences"
            <| fun () ->
                let g = SplitMix64 1234567UL

                Expect.equal
                    [ for _ in 1..5 -> g.Next() ]
                    [
                        6457827717110365317UL
                        3203168211198807973UL
                        9817491932198370423UL
                        4593380528125082431UL
                        16408922859458223821UL
                    ]
                    "seed 1234567"

                let z = SplitMix64 0UL
                Expect.equal [ z.Next(); z.Next() ] [ 0xe220a8397b1dcdafUL; 0x6e789e6aa1b965f4UL ] "seed 0"
                Expect.equal (SplitMix64(z.State).Next()) (z.Next()) "State resumes"

            testCase "DetRng maps (seed, stream, block) onto the Philox key and counter"
            <| fun () ->
                let r = DetRng(0UL, 0UL)
                Expect.equal [ for _ in 1..4 -> r.NextU32() ] (philox10Kats[0] |> fun (_, _, e) -> e) "zero vector"

                Expect.equal
                    (words (DetRng(UInt64.MaxValue, UInt64.MaxValue).Block UInt64.MaxValue))
                    (philox10Kats[1] |> fun (_, _, e) -> e)
                    "all-ones vector"

                // key = (seed lo, seed hi); counter = (block lo, block hi, stream lo, stream hi).
                let pi = DetRng(0x299f31d0a4093822UL, 0x0370734413198a2eUL)

                Expect.equal (words (pi.Block 0x85a308d3243f6a88UL)) (philox10Kats[2] |> fun (_, _, e) -> e) "pi vector"

            testCase "derived known answers"
            <| fun () ->
                Expect.equal (DetRng(0UL, 0UL).NextU64()) 0xe169c58d6627e8d5UL "NextU64: first word is the low half"
                Expect.equal (float (DetRng(0UL, 0UL).NextF32()) * 16777216.0) 6694888.0 "NextF32 = (w0 >> 8) * 2^-24"

                Expect.equal
                    (DetRng(0UL, 0UL).NextF64() * 9007199254740992.0)
                    7931020870206717.0
                    "NextF64 = (u64 >> 11) * 2^-53"

                let r = DetRng(42UL, 7UL)

                Expect.equal
                    [ for _ in 1..8 -> r.NextU32() ]
                    [
                        0x67ee6f2cu
                        0xe55410ccu
                        0x6c7eca35u
                        0x557398d3u
                        0xe5dde940u
                        0x600f6196u
                        0x8fcdf8f1u
                        0x2c8ed839u
                    ]
                    "DetRng(42, 7)"

                Expect.equal (DetRng.streamOf "test") 0x73c5a94594114dd9UL "streamOf \"test\""

            testCase "position counts words and resumes the sequence"
            <| fun () ->
                let r = DetRng(5UL, 9UL)
                r.NextU32() |> ignore
                r.NextU64() |> ignore
                Expect.equal r.Position 3UL "1 + 2 words"
                let resumed = DetRng(5UL, 9UL, r.Position)
                Expect.equal [ for _ in 1..10 -> resumed.NextU32() ] [ for _ in 1..10 -> r.NextU32() ] "resume"

            Props.testProperty "random access equals the lane of the block"
            <| fun (seed: uint64, stream: uint64, position: uint64) ->
                let position = position >>> 2
                let b = DetRng(seed, stream).Block(position / 4UL)
                let expected = (words b)[int (position % 4UL)]
                DetRng(seed, stream, position).NextU32() = expected

            Props.testProperty "sequential draws equal random access"
            <| fun (seed: uint64, stream: uint64) ->
                let r = DetRng(seed, stream)
                let sequential = [ for _ in 0..39 -> r.NextU32() ]

                let direct =
                    [ for i in 0UL .. 39UL -> DetRng.at seed stream i |> fun g -> g.NextU32() ]

                sequential = direct

            Props.testProperty "floats are in [0, 1) on their grids"
            <| fun (seed: uint64, stream: uint64) ->
                let r = DetRng(seed, stream)

                [ 1..64 ]
                |> List.forall (fun _ ->
                    let f = r.NextF32()
                    let d = r.NextF64()
                    let f24 = float f * 16777216.0
                    let d53 = d * 9007199254740992.0

                    f >= 0.0f
                    && f < 1.0f
                    && f24 = Math.Floor f24
                    && d >= 0.0
                    && d < 1.0
                    && d53 = Math.Floor d53)

            testCase "streams and seeds are independent"
            <| fun () ->
                let draw (g: DetRng) = [ for _ in 1..8 -> g.NextU32() ]
                let a = draw (DetRng(1UL, 0UL))
                Expect.notEqual a (draw (DetRng(1UL, 1UL))) "stream"
                Expect.notEqual a (draw (DetRng(2UL, 0UL))) "seed"
                Expect.equal (draw (DetRng(1UL, 0UL).WithStream 1UL)) (draw (DetRng(1UL, 1UL))) "WithStream"
                Expect.notEqual (DetRng.streamOf "a") (DetRng.streamOf "b") "streamOf"

            testCase "NextBelow is in range and unbiased"
            <| fun () ->
                Expect.throwsT<ArgumentException> (fun () -> DetRng(0UL, 0UL).NextBelow 0u |> ignore) "bound 0"
                let r = DetRng(3UL, 4UL)
                Expect.equal [ for _ in 1..10 -> r.NextBelow 1u ] (List.replicate 10 0u) "bound 1"
                let counts = Array.zeroCreate<int> 10

                for _ in 1..100_000 do
                    let x = r.NextBelow 10u
                    counts[int x] <- counts[int x] + 1

                for i, c in Array.indexed counts do
                    Expect.isTrue (abs (c - 10_000) < 500) (sprintf "bucket %d has %d of 100000" i c)

                // Rejection: with bound 2^31 + 1, words below 2^31 - 1 are redrawn, so draws consume extra words.
                let g = DetRng(11UL, 12UL)

                for _ in 1..100 do
                    Expect.isTrue (g.NextBelow 0x80000001u < 0x80000001u) "large bound"

                Expect.isGreaterThan g.Position 100UL "rejections consumed words"

            Props.testProperty "NextInt stays in [lo, hi)"
            <| fun (seed: uint64, lo: int, span: uint16) ->
                let hi = int (min (int64 Int32.MaxValue) (int64 lo + int64 span + 1L))

                hi <= lo
                || (let r = DetRng(seed, 0UL)

                    [ 1..20 ]
                    |> List.forall (fun _ ->
                        let x = r.NextInt(lo, hi)
                        x >= lo && x < hi))
        ]
