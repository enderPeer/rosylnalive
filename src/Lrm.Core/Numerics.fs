namespace Lrm.Core

open System

/// bfloat16 conversions (LrmCore contract). A bf16 value is carried as its raw uint16 bits: 1 sign bit, 8 exponent
/// bits and 7 mantissa bits, the top half of a float32.
[<RequireQualifiedAccess>]
module Bf16 =
    /// The positive quiet NaN every NaN narrows to when its sign bit is clear.
    [<Literal>]
    let QuietNaN = 0x7FC0us

    /// The negative quiet NaN every NaN narrows to when its sign bit is set.
    [<Literal>]
    let NegativeQuietNaN = 0xFFC0us

    /// True when the bits encode a NaN.
    let isNaN (bits: uint16) = (bits &&& 0x7FFFus) > 0x7F80us

    /// Exact widening: the bf16 bits become the top half of the float32. NaN payloads are kept.
    let inline widen (bits: uint16) : float32 =
        BitConverter.UInt32BitsToSingle(uint32 bits <<< 16)

    /// Round to nearest, ties to even. Values beyond the largest finite bf16 (after rounding) become infinities;
    /// subnormals round like any other value (no flushing). Every non-NaN bf16 round-trips through widen. A NaN
    /// narrows to 0x7FC0, or 0xFFC0 when its sign bit is set, whatever its payload.
    let narrowRne (x: float32) : uint16 =
        let u = BitConverter.SingleToUInt32Bits x

        if (u &&& 0x7FFFFFFFu) > 0x7F800000u then
            if u &&& 0x80000000u <> 0u then
                NegativeQuietNaN
            else
                QuietNaN
        else
            // Adding 0x7FFF plus the lowest kept bit rounds the discarded half to nearest even. The carry can ripple
            // into the exponent (the next binade, or infinity) but never past the sign bit for a non-NaN input.
            uint16 ((u + 0x7FFFu + ((u >>> 16) &&& 1u)) >>> 16)

    /// Widens every element of `source` into `destination`, which must be at least as long.
    let widenInto (source: ReadOnlySpan<uint16>, destination: Span<float32>) =
        if destination.Length < source.Length then
            invalidArg "destination" "destination is shorter than source"

        for i in 0 .. source.Length - 1 do
            destination[i] <- widen source[i]

    /// Narrows every element of `source` into `destination`, which must be at least as long.
    let narrowInto (source: ReadOnlySpan<float32>, destination: Span<uint16>) =
        if destination.Length < source.Length then
            invalidArg "destination" "destination is shorter than source"

        for i in 0 .. source.Length - 1 do
            destination[i] <- narrowRne source[i]

    /// Widens an array.
    let widenArray (source: uint16[]) : float32[] =
        let result = Array.zeroCreate source.Length
        widenInto (ReadOnlySpan source, Span result)
        result

    /// Narrows an array.
    let narrowArray (source: float32[]) : uint16[] =
        let result = Array.zeroCreate source.Length
        narrowInto (ReadOnlySpan source, Span result)
        result

/// IEEE 754 binary16 (half) conversions, carried as raw uint16 bits: 1 sign bit, 5 exponent bits, 10 mantissa bits.
[<RequireQualifiedAccess>]
module F16 =
    /// The positive quiet NaN every NaN narrows to when its sign bit is clear.
    [<Literal>]
    let QuietNaN = 0x7E00us

    /// The negative quiet NaN every NaN narrows to when its sign bit is set.
    [<Literal>]
    let NegativeQuietNaN = 0xFE00us

    /// True when the bits encode a NaN.
    let isNaN (bits: uint16) = (bits &&& 0x7FFFus) > 0x7C00us

    /// Exact widening to float32. Subnormals become normal float32 values; NaN payloads are kept (shifted into the
    /// top of the float32 mantissa).
    let widen (bits: uint16) : float32 =
        let sign = uint32 (bits &&& 0x8000us) <<< 16
        let exponent = int ((bits >>> 10) &&& 0x1Fus)
        let mantissa = uint32 (bits &&& 0x3FFus)

        if exponent = 0x1F then
            BitConverter.UInt32BitsToSingle(sign ||| 0x7F800000u ||| (mantissa <<< 13))
        elif exponent <> 0 then
            BitConverter.UInt32BitsToSingle(sign ||| (uint32 (exponent + 112) <<< 23) ||| (mantissa <<< 13))
        elif mantissa = 0u then
            BitConverter.UInt32BitsToSingle sign
        else
            // A subnormal is mantissa * 2^-24, exact in float32: an integer below 2^10 times a power of two.
            let magnitude = float32 mantissa * 5.9604644775390625e-8f
            if sign <> 0u then -magnitude else magnitude

    /// Round to nearest, ties to even, with subnormal results (no flushing). Values at or above 65520 become
    /// infinities. A NaN narrows to 0x7E00, or 0xFE00 when its sign bit is set, whatever its payload.
    let narrowRne (x: float32) : uint16 =
        let u = BitConverter.SingleToUInt32Bits x
        let sign = uint16 ((u >>> 16) &&& 0x8000u)
        let a = u &&& 0x7FFFFFFFu

        if a > 0x7F800000u then
            sign ||| QuietNaN
        elif a >= 0x477FF000u then
            // 65520 = 65504 + half an ulp; the tie rounds to the even neighbour, which is infinity.
            sign ||| 0x7C00us
        elif a >= 0x38800000u then
            // Normal half: rebias the exponent (127 -> 15) and round the 13 discarded mantissa bits to nearest even.
            let m = a - 0x38000000u
            sign ||| uint16 ((m + 0xFFFu + ((m >>> 13) &&& 1u)) >>> 13)
        else
            let exponent = int (a >>> 23)

            if exponent < 102 then
                // Below 2^-25, half the smallest subnormal: rounds to zero.
                sign
            else
                // Subnormal half: the result counts units of 2^-24. The float32 significand (with its implicit bit)
                // is significand * 2^(exponent - 150), so the unit count is significand >> (126 - exponent).
                let significand = (a &&& 0x7FFFFFu) ||| 0x800000u
                let shift = 126 - exponent
                let q = significand >>> shift
                let rest = significand &&& ((1u <<< shift) - 1u)
                let half = 1u <<< (shift - 1)

                let q =
                    if rest > half || (rest = half && (q &&& 1u) = 1u) then
                        q + 1u
                    else
                        q

                // q = 0x400 is the smallest normal half; the bit pattern carries into the exponent field correctly.
                sign ||| uint16 q
