module Lrm.Core.Tests.NumericsTests

open System
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open Expecto
open Lrm.Core

// ---- independent oracles -------------------------------------------------------------------------------------------

let private twoTo128 = Math.ScaleB(1.0, 128)

/// The magnitude of 15-bit bf16 magnitude bits, with the infinity pattern standing for 2^128, the next value of the
/// sequence. IEEE 754 rounds to infinity exactly when that next value is the nearest (ties to even).
let private bf16Magnitude (bits: uint32) =
    if bits = 0x7F80u then
        twoTo128
    else
        float (BitConverter.UInt32BitsToSingle(bits <<< 16))

/// bf16 round-to-nearest-even by exact distance: the two bf16 neighbours of |x| are compared in float64, where both
/// distances are exact. This shares nothing with Bf16.narrowRne's bias trick.
let bf16Oracle (u: uint32) : uint16 =
    let sign = uint16 ((u &&& 0x80000000u) >>> 16)
    let a = u &&& 0x7FFFFFFFu

    if a > 0x7F800000u then
        sign ||| 0x7FC0us
    elif a &&& 0xFFFFu = 0u then
        sign ||| uint16 (a >>> 16)
    else
        let lo = a >>> 16
        let hi = lo + 1u
        let x = float (BitConverter.UInt32BitsToSingle a)
        let dlo = x - bf16Magnitude lo
        let dhi = bf16Magnitude hi - x

        let pick =
            if dlo < dhi then lo
            elif dhi < dlo then hi
            elif lo &&& 1u = 0u then lo
            else hi

        sign ||| uint16 pick

/// binary16 round-to-nearest-even in float64 arithmetic: scale |x| to units of its binary16 spacing (2^-24 below
/// 2^-14), round with MidpointRounding.ToEven, then encode. Results of 65536 or more are infinity.
let f16Oracle (u: uint32) : uint16 =
    let sign = if u &&& 0x80000000u <> 0u then 0x8000us else 0us
    let a = u &&& 0x7FFFFFFFu

    if a > 0x7F800000u then
        sign ||| 0x7E00us
    elif a = 0x7F800000u then
        sign ||| 0x7C00us
    else
        let x = float (BitConverter.UInt32BitsToSingle a)

        if x = 0.0 then
            sign
        else
            let e = max (Math.ILogB x) -14
            let spacing = Math.ScaleB(1.0, e - 10)
            let v = Math.Round(x / spacing, MidpointRounding.ToEven) * spacing

            if v >= 65536.0 then
                sign ||| 0x7C00us
            elif v < 6.103515625e-05 then
                sign ||| uint16 (v * 16777216.0)
            else
                let ev = Math.ILogB v
                let frac = v / Math.ScaleB(1.0, ev) - 1.0
                sign ||| uint16 ((ev + 15) <<< 10) ||| uint16 (frac * 1024.0)

/// The value of binary16 bits from the IEEE 754 definition, in float64 (finite values only).
let private f16Value (h: uint16) =
    let s = if h &&& 0x8000us <> 0us then -1.0 else 1.0
    let e = int ((h >>> 10) &&& 0x1Fus)
    let m = float (h &&& 0x3FFus)

    if e = 0 then
        s * m * Math.ScaleB(1.0, -24)
    else
        s * (1.0 + m / 1024.0) * Math.ScaleB(1.0, e - 15)

let private halfBits (x: float32) : uint16 =
    BitConverter.HalfToUInt16Bits(Half.op_Explicit x: Half)

// ---- exhaustive sweep ----------------------------------------------------------------------------------------------

type SweepResult =
    {
        Checked: int64
        Failures: int64
        FirstFailure: uint32 option
        Elapsed: TimeSpan
    }

/// Checks every one of the 2^32 float32 bit patterns, in parallel over the high 16 bits.
let inline sweepAllFloat32 ([<InlineIfLambda>] check: uint32 -> bool) : SweepResult =
    let failures = ref 0L
    let first = ref UInt64.MaxValue
    let gate = obj ()
    let clock = Stopwatch.StartNew()

    Parallel.For(
        0,
        65536,
        fun hi ->
            let high = uint32 hi <<< 16
            let mutable bad = 0L
            let mutable firstBad = UInt64.MaxValue

            for lo in 0..65535 do
                let u = high ||| uint32 lo

                if not (check u) then
                    bad <- bad + 1L

                    if uint64 u < firstBad then
                        firstBad <- uint64 u

            if bad > 0L then
                Interlocked.Add(&failures.contents, bad) |> ignore
                lock gate (fun () -> first.Value <- min first.Value firstBad)
    )
    |> ignore

    {
        Checked = 1L <<< 32
        Failures = failures.Value
        FirstFailure =
            if first.Value = UInt64.MaxValue then
                None
            else
                Some(uint32 first.Value)
        Elapsed = clock.Elapsed
    }

let private expectClean (what: string) (r: SweepResult) (describe: uint32 -> string) =
    printfn "%s: %d patterns in %.1f s on %d threads" what r.Checked r.Elapsed.TotalSeconds Environment.ProcessorCount

    match r.FirstFailure with
    | None -> Expect.equal r.Failures 0L what
    | Some u -> failtestf "%s: %d mismatches; first at 0x%08x: %s" what r.Failures u (describe u)

// ---- tests ---------------------------------------------------------------------------------------------------------

/// (float32 bits, expected bf16 bits, what it shows): the bf16 KAT table of docs/contracts/lrm-core.md.
let bf16Kats =
    [
        0x00000000u, 0x0000us, "+0"
        0x80000000u, 0x8000us, "-0"
        0x3F800000u, 0x3F80us, "1.0"
        0x3F807FFFu, 0x3F80us, "just below the tie"
        0x3F808000u, 0x3F80us, "tie, even neighbour below"
        0x3F808001u, 0x3F81us, "just above the tie"
        0x3F818000u, 0x3F82us, "tie, even neighbour above"
        0xBF818000u, 0xBF82us, "negative tie"
        0x40490FDBu, 0x4049us, "pi"
        0x7F7F7FFFu, 0x7F7Fus, "largest value that stays finite"
        0x7F7F8000u, 0x7F80us, "tie above the largest bf16 rounds to infinity"
        0x7F7FFFFFu, 0x7F80us, "FLT_MAX rounds to infinity"
        0x7F800000u, 0x7F80us, "+infinity"
        0xFF800000u, 0xFF80us, "-infinity"
        0x00000001u, 0x0000us, "smallest subnormal rounds to zero"
        0x00008000u, 0x0000us, "subnormal tie rounds to even zero"
        0x00008001u, 0x0001us, "subnormal above the tie"
        0x00018000u, 0x0002us, "subnormal tie rounds up to even"
        0x007FFFFFu, 0x0080us, "largest subnormal rounds to the smallest normal"
        0x7F800001u, 0x7FC0us, "signalling NaN (truncation would give infinity)"
        0x7FC00000u, 0x7FC0us, "quiet NaN"
        0x7FFFFFFFu, 0x7FC0us, "NaN with every payload bit"
        0xFF800001u, 0xFFC0us, "negative signalling NaN"
        0xFFFFFFFFu, 0xFFC0us, "negative NaN"
    ]

/// (float32 bits, expected binary16 bits, what it shows): the f16 KAT table of docs/contracts/lrm-core.md.
let f16Kats =
    [
        0x3F800000u, 0x3C00us, "1.0"
        0xC0000000u, 0xC000us, "-2.0"
        0x3DCCCCCDu, 0x2E66us, "0.1f"
        0x477FE000u, 0x7BFFus, "65504, the largest half"
        0x477FEFFFu, 0x7BFFus, "just below 65520"
        0x477FF000u, 0x7C00us, "65520 ties to infinity"
        0x7F800000u, 0x7C00us, "+infinity"
        0x38800000u, 0x0400us, "2^-14, the smallest normal"
        0x387FC000u, 0x03FFus, "1023 * 2^-24, the largest subnormal"
        0x387FE000u, 0x0400us, "1023.5 * 2^-24 ties to the smallest normal"
        0x33800000u, 0x0001us, "2^-24, the smallest subnormal"
        0x33000000u, 0x0000us, "2^-25 ties to even zero"
        0x33000001u, 0x0001us, "just above 2^-25"
        0x33C00000u, 0x0002us, "1.5 * 2^-24 ties to 2"
        0x34200000u, 0x0002us, "2.5 * 2^-24 ties to 2"
        0x80000000u, 0x8000us, "-0"
        0x7F800001u, 0x7E00us, "signalling NaN"
        0x7FC00000u, 0x7E00us, "quiet NaN"
        0xFF800001u, 0xFE00us, "negative NaN"
    ]

let tests =
    testList
        "numerics"
        [
            testList
                "bf16"
                [
                    testCase "known answers"
                    <| fun () ->
                        for bits, expected, what in bf16Kats do
                            let got = Bf16.narrowRne (BitConverter.UInt32BitsToSingle bits)
                            Expect.equal got expected (sprintf "0x%08x (%s)" bits what)
                            Expect.equal (bf16Oracle bits) expected (sprintf "oracle 0x%08x (%s)" bits what)

                    testCase "widen is exact for all 65,536 patterns"
                    <| fun () ->
                        for b in 0..65535 do
                            let bits = uint16 b
                            let w = BitConverter.SingleToUInt32Bits(Bf16.widen bits)
                            Expect.equal w (uint32 bits <<< 16) (sprintf "widen 0x%04x" bits)

                    testCase "every non-NaN bf16 round-trips; NaN narrows to 0x7FC0 or 0xFFC0 by sign"
                    <| fun () ->
                        let mutable nans = 0

                        for b in 0..65535 do
                            let bits = uint16 b
                            let back = Bf16.narrowRne (Bf16.widen bits)

                            if Bf16.isNaN bits then
                                nans <- nans + 1
                                let expected = if bits &&& 0x8000us <> 0us then 0xFFC0us else 0x7FC0us
                                Expect.equal back expected (sprintf "NaN 0x%04x" bits)
                            else
                                Expect.equal back bits (sprintf "round trip 0x%04x" bits)

                        Expect.equal nans 254 "bf16 has 2 * 127 NaN patterns"

                    testCase "narrowRne over all 2^32 float32 patterns equals the exact-distance oracle"
                    <| fun () ->
                        let r =
                            sweepAllFloat32 (fun u -> Bf16.narrowRne (BitConverter.UInt32BitsToSingle u) = bf16Oracle u)

                        expectClean "bf16 narrowRne exhaustive" r (fun u ->
                            sprintf
                                "narrowRne 0x%04x, oracle 0x%04x"
                                (Bf16.narrowRne (BitConverter.UInt32BitsToSingle u))
                                (bf16Oracle u))

                    testCase "arrays"
                    <| fun () ->
                        let xs = [| 1.0f; -2.5f; Single.NaN; 3.0e38f |]
                        let bits = Bf16.narrowArray xs
                        Expect.equal bits (xs |> Array.map Bf16.narrowRne) "narrowArray"
                        Expect.equal (Bf16.widenArray bits |> Array.map Bf16.narrowRne) bits "widenArray"
                ]

            testList
                "f16"
                [
                    testCase "known answers"
                    <| fun () ->
                        for bits, expected, what in f16Kats do
                            let got = F16.narrowRne (BitConverter.UInt32BitsToSingle bits)
                            Expect.equal got expected (sprintf "0x%08x (%s)" bits what)
                            Expect.equal (f16Oracle bits) expected (sprintf "oracle 0x%08x (%s)" bits what)

                    testCase "widen is exact for all 65,536 patterns"
                    <| fun () ->
                        for b in 0..65535 do
                            let h = uint16 b
                            let w = F16.widen h

                            if F16.isNaN h then
                                let expected =
                                    (uint32 (h &&& 0x8000us) <<< 16)
                                    ||| 0x7F800000u
                                    ||| (uint32 (h &&& 0x3FFus) <<< 13)

                                Expect.equal (BitConverter.SingleToUInt32Bits w) expected (sprintf "NaN 0x%04x" h)
                            elif (h &&& 0x7FFFus) = 0x7C00us then
                                Expect.isTrue (Single.IsInfinity w) (sprintf "infinity 0x%04x" h)
                            else
                                Expect.equal (float w) (f16Value h) (sprintf "value 0x%04x" h)

                                Expect.equal
                                    (BitConverter.SingleToUInt32Bits w)
                                    (BitConverter.SingleToUInt32Bits(float32 (BitConverter.UInt16BitsToHalf h)))
                                    (sprintf "System.Half 0x%04x" h)

                    testCase "every non-NaN half round-trips; NaN narrows to 0x7E00 or 0xFE00 by sign"
                    <| fun () ->
                        for b in 0..65535 do
                            let h = uint16 b
                            let back = F16.narrowRne (F16.widen h)

                            if F16.isNaN h then
                                let expected = if h &&& 0x8000us <> 0us then 0xFE00us else 0x7E00us
                                Expect.equal back expected (sprintf "NaN 0x%04x" h)
                            else
                                Expect.equal back h (sprintf "round trip 0x%04x" h)

                    testCase "narrowRne over all 2^32 float32 patterns equals the float64 oracle and System.Half"
                    <| fun () ->
                        let r =
                            sweepAllFloat32 (fun u ->
                                let x = BitConverter.UInt32BitsToSingle u
                                let got = F16.narrowRne x
                                got = f16Oracle u && (Single.IsNaN x || got = halfBits x))

                        expectClean "f16 narrowRne exhaustive" r (fun u ->
                            let x = BitConverter.UInt32BitsToSingle u

                            sprintf
                                "narrowRne 0x%04x, oracle 0x%04x, System.Half 0x%04x"
                                (F16.narrowRne x)
                                (f16Oracle u)
                                (halfBits x))
                ]
        ]
