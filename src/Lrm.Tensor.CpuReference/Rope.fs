namespace Lrm.Tensor.CpuReference

open System
open System.Collections.Concurrent
open System.Numerics
open Lrm.Core

/// Correctly rounded float64 sin and cos of float32 arguments, in exact fixed-point integer arithmetic, so the result
/// depends on no libm (CpuReferenceSemantics contract, docs/decisions/O2-0001-exact-rope-table.md).
///
/// For |x| >= 2^-27 the argument is reduced by pi/2 with 64 guard bits beyond the integer part of x / (pi/2), and sin
/// and cos of the reduced argument r (|r| <= pi/4 + tiny) are summed as Taylor series in Q-bit fixed point (Q = 128
/// first). The total error is far below 2^-112 (2^16 units of 2^-Q); the float64 rounding is accepted only when both
/// ends of that error interval round to the same float64, otherwise Q doubles. Below 2^-27, sin x rounds to x and
/// cos x to 1 exactly.
[<RequireQualifiedAccess>]
module ExactTrig =
    /// floor(pi * 2^bits), cached per precision. Machin: pi = 16 atan(1/5) - 4 atan(1/239), with 32 guard bits.
    let private piCache = ConcurrentDictionary<int, BigInteger>()

    let private piFixed (bits: int) =
        piCache.GetOrAdd(
            bits,
            fun bits ->
                let guard = 32
                let one = BigInteger.One <<< (bits + guard)

                let atanInv (n: int) =
                    let n2 = BigInteger(n * n)
                    let mutable power = one / BigInteger n
                    let mutable sum = power
                    let mutable k = 1

                    while not power.IsZero do
                        power <- power / n2
                        let t = power / BigInteger(2 * k + 1)
                        sum <- if k % 2 = 1 then sum - t else sum + t
                        k <- k + 1

                    sum

                // Each series term is truncated once (error < 1 unit at 2^-(bits + guard)); a few hundred units of
                // error vanish in the 32 guard bits, so the result is floor(pi * 2^bits) or one below it.
                ((BigInteger 16 * atanInv 5) - (BigInteger 4 * atanInv 239)) >>> guard
        )

    /// RN64(v / 2^q) when every value in [v - e, v + e] rounds to the same float64, else ValueNone.
    let private roundToDouble (v: BigInteger) (q: int) (e: BigInteger) : float voption =
        let rn (v: BigInteger) =
            if v.IsZero then
                0.0
            else
                let negative = v.Sign < 0
                let a = BigInteger.Abs v
                let n = int (a.GetBitLength())
                let s = n - 53

                let magnitude =
                    if s <= 0 then
                        Math.ScaleB(float a, -q)
                    else
                        let kept = a >>> s
                        let rest = a - (kept <<< s)
                        let half = BigInteger.One <<< (s - 1)

                        let kept =
                            if rest > half || (rest = half && not kept.IsEven) then
                                kept + BigInteger.One
                            else
                                kept

                        // kept <= 2^53 converts exactly; the scale is a power of two inside the normal range here.
                        Math.ScaleB(float kept, s - q)

                if negative then -magnitude else magnitude

        let lo = v - e
        let hi = v + e

        if lo.Sign <> hi.Sign then
            ValueNone
        else
            let a = rn lo
            let b = rn hi

            if BitConverter.DoubleToInt64Bits a = BitConverter.DoubleToInt64Bits b then
                ValueSome a
            else
                ValueNone

    let private errorUnits = BigInteger.One <<< 16

    /// 2^-27.
    let private tiny = MathF.ScaleB(1.0f, -27)

    /// (cos, sin) at precision q, or ValueNone when the float64 rounding is not certain at that precision.
    let private attempt (magnitude: float32) (q: int) : struct (float * float) voption =
        let bits = BitConverter.SingleToUInt32Bits magnitude
        let biased = int ((bits >>> 23) &&& 0xFFu)
        let fraction = bits &&& 0x7FFFFFu
        // magnitude = m * 2^e exactly, m < 2^24.
        let m, e =
            if biased = 0 then
                fraction, -149
            else
                fraction ||| 0x800000u, biased - 150

        let guard = max 0 (e + 24) + 64
        let scale = q + guard
        let x = BigInteger m <<< (e + scale)
        let halfPi = piFixed (scale - 1)
        let k = (x + (halfPi >>> 1)) / halfPi
        let r = (x - k * halfPi) >>> guard
        let r2 = (r * r) >>> q

        let mutable term = r
        let mutable s = r
        let mutable n = 1

        while not term.IsZero do
            term <- -((term * r2) >>> q) / BigInteger((2 * n) * (2 * n + 1))
            s <- s + term
            n <- n + 1

        let mutable termC = BigInteger.One <<< q
        let mutable c = termC
        n <- 1

        while not termC.IsZero do
            termC <- -((termC * r2) >>> q) / BigInteger((2 * n - 1) * (2 * n))
            c <- c + termC
            n <- n + 1

        let struct (cv, sv) =
            match int (k % BigInteger 4) with
            | 0 -> struct (c, s)
            | 1 -> struct (-s, c)
            | 2 -> struct (-c, -s)
            | _ -> struct (s, -c)

        match roundToDouble cv q errorUnits, roundToDouble sv q errorUnits with
        | ValueSome a, ValueSome b -> ValueSome(struct (a, b))
        | _ -> ValueNone

    /// The correctly rounded float64 (cos x, sin x) of a float32 x. NaN for a non-finite x.
    let cosSin (x: float32) : struct (float * float) =
        if not (Single.IsFinite x) then
            struct (nan, nan)
        elif abs x < tiny then
            // |x| < 2^-27: cos x = 1 - x^2/2 and sin x = x - x^3/6 lie within half a float64 ulp of 1 and x.
            struct (1.0, float x)
        else
            let rec go q =
                if q > 4096 then
                    invalidOp (sprintf "no certain rounding for cos/sin(%s)" (Invariant.f32 x))

                match attempt (abs x) q with
                | ValueSome(struct (c, s)) -> struct (c, (if x < 0.0f then -s else s))
                | ValueNone -> go (q * 2)

            go 128

/// The RoPE table T_spec (ReferenceOracles and CpuReferenceSemantics contracts): for pos < MaxPos and j < HeadDim/2,
/// freq = f32(f32(pos) * inv_freq[j]), Cos = f32(RN64(cos freq)) and Sin = f32(RN64(sin freq)), row-major
/// [MaxPos, HeadDim/2]. F64 mode widens the same float32 values.
[<Sealed>]
type RopeTable internal (invFreqBits: uint32[], maxPos: int, cos: float32[], sin: float32[]) =
    let half = invFreqBits.Length

    /// The pinned inverse frequencies, as float32 bits (a copy).
    member _.InvFreqBits = Array.copy invFreqBits

    /// Head size: twice the number of frequencies.
    member _.HeadDim = 2 * half

    /// HeadDim / 2.
    member _.Half = half

    /// Positions 0 .. MaxPos - 1.
    member _.MaxPos = maxPos

    /// The cos values of one position (Half entries).
    member _.CosRow(pos: int) =
        if pos < 0 || pos >= maxPos then
            raise (ArgumentOutOfRangeException("pos", sprintf "position %d outside the table (max_pos %d)" pos maxPos))

        ReadOnlySpan<float32>(cos, pos * half, half)

    /// The sin values of one position (Half entries).
    member _.SinRow(pos: int) =
        if pos < 0 || pos >= maxPos then
            raise (ArgumentOutOfRangeException("pos", sprintf "position %d outside the table (max_pos %d)" pos maxPos))

        ReadOnlySpan<float32>(sin, pos * half, half)

    /// A copy of every cos value, row-major [MaxPos, Half].
    member _.Cos = Array.copy cos

    /// A copy of every sin value, row-major [MaxPos, Half].
    member _.Sin = Array.copy sin

    /// SHA-256 of every cos value then every sin value, float32 little-endian, row-major: the ReferenceOracles digest.
    member _.Digest() : ContentId =
        use h = new Hash.Incremental()
        h.Append(Runtime.InteropServices.MemoryMarshal.AsBytes(ReadOnlySpan cos))
        h.Append(Runtime.InteropServices.MemoryMarshal.AsBytes(ReadOnlySpan sin))
        h.Finish()

[<RequireQualifiedAccess>]
module RopeTable =
    /// The angle of (pos, j): one float32 multiply of the exact float32 position and the pinned inverse frequency.
    let angle (pos: int) (invFreqBits: uint32) =
        float32 pos * BitConverter.UInt32BitsToSingle invFreqBits

    /// Builds T_spec from the pinned inverse-frequency bits for positions 0 .. maxPos - 1. Rows are computed in
    /// parallel; every entry is exact, so the table is the same at every thread count and on every machine. Raises
    /// ArgumentException for maxPos outside 1 .. 2^24, or when maxPos * invFreqBits.Length exceeds Array.MaxLength.
    let create (exec: CpuExec) (invFreqBits: uint32[]) (maxPos: int) =
        if isNull invFreqBits || invFreqBits.Length = 0 then
            invalidArg "invFreqBits" "no inverse frequencies"

        if maxPos < 1 || maxPos > (1 <<< 24) then
            invalidArg "maxPos" (sprintf "max_pos %d outside 1 .. 2^24 (positions must be exact in float32)" maxPos)

        let half = invFreqBits.Length

        // Each of cos and sin is one array of maxPos * half float32 values (head_dim 256 at 2^24 positions is 2^31).
        if int64 maxPos * int64 half > int64 Array.MaxLength then
            invalidArg
                "maxPos"
                (sprintf
                    "max_pos %d with %d frequencies is %d entries per table, beyond the .NET array limit of %d"
                    maxPos
                    half
                    (int64 maxPos * int64 half)
                    Array.MaxLength)

        for b in invFreqBits do
            if not (Single.IsFinite(BitConverter.UInt32BitsToSingle b)) then
                invalidArg "invFreqBits" (sprintf "inv_freq %08x is not finite" b)

        let cos = Array.zeroCreate<float32>(maxPos * half)
        let sin = Array.zeroCreate<float32>(maxPos * half)

        Par.forEach exec maxPos (fun pos ->
            for j in 0 .. half - 1 do
                let struct (c, s) = ExactTrig.cosSin (angle pos invFreqBits[j])
                cos[pos * half + j] <- float32 c
                sin[pos * half + j] <- float32 s)

        RopeTable(Array.copy invFreqBits, maxPos, cos, sin)
