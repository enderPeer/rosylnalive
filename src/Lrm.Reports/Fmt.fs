/// Number formatting for reports (ReportTemplates contract). Every function uses the invariant culture, prints a
/// value that rounds to zero without a sign, and spells non-finite values "NaN", "inf" and "-inf".
module Lrm.Reports.Fmt

open System
open Lrm.Core

let private nonFinite (x: float) =
    if Double.IsNaN x then "NaN"
    elif x > 0.0 then "inf"
    else "-inf"

/// Drops the sign of a text that shows only zeros ("-0.00" -> "0.00").
let private unsignZero (s: string) =
    if
        s.StartsWith '-'
        && s |> Seq.forall (fun c -> c = '-' || c = '0' || c = '.' || c = ',')
    then
        s.Substring 1
    else
        s

/// Fixed-point with `digits` decimals: dp 2 3.14159 = "3.14".
let dp (digits: int) (x: float) =
    if Double.IsFinite x then
        unsignZero (x.ToString("F" + Invariant.i32 digits, Invariant.culture))
    else
        nonFinite x

/// Scientific with `digits` decimals: sci 2 0.000321 = "3.21e-04".
let sci (digits: int) (x: float) =
    if not (Double.IsFinite x) then
        nonFinite x
    elif x = 0.0 then
        "0"
    else
        let pattern = (if digits > 0 then "0." + String('0', digits) else "0") + "e+00"
        x.ToString(pattern, Invariant.culture)

/// A readable number: integers below 10^6 with separators, otherwise about four significant digits, and scientific
/// below 10^-3 or from 10^6.
let num (x: float) =
    if not (Double.IsFinite x) then
        nonFinite x
    elif x = 0.0 then
        "0"
    else
        let a = abs x

        if a < 1e-3 || a >= 1e6 then
            sci 2 x
        elif a >= 1000.0 then
            unsignZero (x.ToString("N0", Invariant.culture))
        elif a >= 100.0 then
            dp 1 x
        elif a >= 10.0 then
            dp 2 x
        elif a >= 1.0 then
            dp 3 x
        else
            dp 4 x

/// An integer with thousands separators: 1,234,567.
let int (n: int64) = n.ToString("N0", Invariant.culture)

/// SI suffixes for counts such as FLOPs and tokens: 1.25G, 3.40M, 512.
let si (x: float) =
    if not (Double.IsFinite x) then
        nonFinite x
    else
        let a = abs x

        if a >= 1e15 then dp 2 (x / 1e15) + "P"
        elif a >= 1e12 then dp 2 (x / 1e12) + "T"
        elif a >= 1e9 then dp 2 (x / 1e9) + "G"
        elif a >= 1e6 then dp 2 (x / 1e6) + "M"
        elif a >= 1e3 then dp 2 (x / 1e3) + "K"
        elif a = Math.Round a then dp 0 x
        else dp 2 x

/// SI suffixes followed by a unit, with a space: siWith "FLOP/s" 1.56e12 = "1.56 TFLOP/s".
let siWith (unit: string) (x: float) =
    let s = si x

    if s.Length > 0 && Char.IsLetter s[s.Length - 1] then
        s.Substring(0, s.Length - 1) + " " + s.Substring(s.Length - 1) + unit
    else
        s + " " + unit

/// Binary byte units: 1.50 GiB, 12.00 MiB, 512 B.
let bytes (x: float) =
    if not (Double.IsFinite x) then
        nonFinite x
    else
        let a = abs x

        if a >= 1099511627776.0 then
            dp 2 (x / 1099511627776.0) + " TiB"
        elif a >= 1073741824.0 then
            dp 2 (x / 1073741824.0) + " GiB"
        elif a >= 1048576.0 then
            dp 2 (x / 1048576.0) + " MiB"
        elif a >= 1024.0 then
            dp 2 (x / 1024.0) + " KiB"
        else
            dp 0 x + " B"

/// A duration given in ns: 1.25 s, 3.40 ms, 12.00 µs, 800 ns.
let duration (ns: float) =
    if not (Double.IsFinite ns) then
        nonFinite ns
    else
        let a = abs ns

        if a >= 1e9 then dp 2 (ns / 1e9) + " s"
        elif a >= 1e6 then dp 2 (ns / 1e6) + " ms"
        elif a >= 1e3 then dp 2 (ns / 1e3) + " µs"
        else dp 0 ns + " ns"

/// A ratio as a percentage with one decimal: 0.1234 -> "12.3%".
let pct (ratio: float) =
    if Double.IsFinite ratio then
        dp 1 (ratio * 100.0) + "%"
    else
        nonFinite ratio
