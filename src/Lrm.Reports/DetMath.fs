/// Transcendental functions built only from IEEE add, subtract, multiply and divide (ReportTemplates contract,
/// "Determinism"). Math.Log, Math.Exp and Math.Pow call the platform's C runtime, whose last bits can differ between
/// machines and runtime versions; chart layout and synthetic data use these instead, so a report renders to the same
/// bytes everywhere. They are accurate to a few ulps, which is plenty for charts, and never used for model numerics.
module Lrm.Reports.DetMath

open System

[<Literal>]
let Ln2 = 0.6931471805599453

[<Literal>]
let Ln10 = 2.302585092994046

[<Literal>]
let private Sqrt2 = 1.4142135623730951

/// Natural logarithm. NaN for negative input and NaN, -infinity for 0.
let ln (x: float) =
    if Double.IsNaN x || x < 0.0 then
        Double.NaN
    elif x = 0.0 then
        Double.NegativeInfinity
    elif Double.IsPositiveInfinity x then
        x
    else
        // Bring subnormals into the normal range first.
        let x, bias =
            if x < 2.2250738585072014e-308 then
                x * 18014398509481984.0, -54
            else
                x, 0

        let bits = BitConverter.DoubleToInt64Bits x
        let e = int ((bits >>> 52) &&& 0x7FFL) - 1023
        // The mantissa in [1, 2), then centred on 1 so the series converges fast.
        let m =
            BitConverter.Int64BitsToDouble((bits &&& 0x000FFFFFFFFFFFFFL) ||| 0x3FF0000000000000L)

        let m, e = if m > Sqrt2 then m * 0.5, e + 1 else m, e
        // ln m = 2 atanh(s) = 2 (s + s^3/3 + s^5/5 + ...), with |s| <= 0.172.
        let s = (m - 1.0) / (m + 1.0)
        let s2 = s * s
        let mutable term = s
        let mutable sum = s

        for k in 1..13 do
            term <- term * s2
            sum <- sum + term / float (2 * k + 1)

        2.0 * sum + float (e + bias) * Ln2

/// Base-10 logarithm.
let log10 (x: float) = ln x / Ln10

/// e^x. Infinity above 709.78, 0 below -745.2.
let exp (x: float) =
    if Double.IsNaN x then
        x
    elif x > 709.782712893384 then
        Double.PositiveInfinity
    elif x < -745.2 then
        0.0
    else
        // x = k ln2 + r with |r| <= ln2 / 2; e^r by its Taylor series; then scale by 2^k (exact). ln2 is split in two
        // (Cody and Waite, as in fdlibm): k * Ln2Hi is exact, so r keeps its precision even for large k.
        let k = Math.Round(x / Ln2)
        let r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10
        let mutable term = 1.0
        let mutable sum = 1.0

        for n in 1..22 do
            term <- term * r / float n
            sum <- sum + term

        Math.ScaleB(sum, int k)

/// 10^k for an integer k: exact up to 10^22, by repeated multiplication.
let pow10 (k: int) =
    let mutable p = 1.0

    for _ in 1 .. abs k do
        p <- p * 10.0

    if k >= 0 then p else 1.0 / p

/// 10^x for a real x.
let exp10 (x: float) = exp (x * Ln10)

/// The largest integer k with 10^k <= x, for x > 0, found by comparison only.
let floorLog10 (x: float) =
    if not (x > 0.0) || Double.IsInfinity x then
        invalidArg "x" "floorLog10 needs a positive finite value"

    let mutable k = 0

    while pow10 k > x do
        k <- k - 1

    while pow10 (k + 1) <= x do
        k <- k + 1

    k
