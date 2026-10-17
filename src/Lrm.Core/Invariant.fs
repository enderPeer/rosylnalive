/// Culture-independent formatting and parsing (LrmCore contract).
///
/// Library code formats and parses numbers only through these helpers, or with an explicit
/// CultureInfo.InvariantCulture, so no output ever depends on the machine's culture. Floats use the "R" format: the
/// shortest text that parses back to the same bits. Process.init also sets the invariant culture, but libraries must
/// not rely on that, because tests and hosts can change the current culture.
module Lrm.Core.Invariant

open System
open System.Globalization

/// The culture every LRM component formats and parses with.
let culture = CultureInfo.InvariantCulture

/// Shortest round-trip text of a float32 ("R"): "1.5", "1E-05", "-0", "NaN", "Infinity", "-Infinity".
let f32 (x: float32) = x.ToString("R", culture)

/// Shortest round-trip text of a float ("R"): "0.1", "1E+300", "-0", "NaN", "Infinity", "-Infinity".
let f64 (x: float) = x.ToString("R", culture)

/// Decimal text of an int32.
let i32 (x: int) = x.ToString(culture)

/// Decimal text of an int64.
let i64 (x: int64) = x.ToString(culture)

/// Decimal text of a uint32.
let u32 (x: uint32) = x.ToString(culture)

/// Decimal text of a uint64.
let u64 (x: uint64) = x.ToString(culture)

/// Formats an interpolated string with the invariant culture: Invariant.format $"{x} rows in {t} s".
/// F# interpolation without a format specifier otherwise uses the current culture.
let format (text: FormattableString) = FormattableString.Invariant text

let private floatStyle = NumberStyles.Float

/// Parses a float32 written by f32 (or any invariant float text). None when the text is not a number.
let tryParseF32 (text: string) : float32 option =
    match Single.TryParse(text, floatStyle, culture) with
    | true, v -> Some v
    | _ -> None

/// Parses a float written by f64 (or any invariant float text). None when the text is not a number.
let tryParseF64 (text: string) : float option =
    match Double.TryParse(text, floatStyle, culture) with
    | true, v -> Some v
    | _ -> None

/// Parses a decimal int32 with an optional leading minus sign.
let tryParseI32 (text: string) : int option =
    match Int32.TryParse(text, NumberStyles.AllowLeadingSign, culture) with
    | true, v -> Some v
    | _ -> None

/// Parses a decimal int64 with an optional leading minus sign.
let tryParseI64 (text: string) : int64 option =
    match Int64.TryParse(text, NumberStyles.AllowLeadingSign, culture) with
    | true, v -> Some v
    | _ -> None

/// Parses a decimal uint64 (digits only).
let tryParseU64 (text: string) : uint64 option =
    match UInt64.TryParse(text, NumberStyles.None, culture) with
    | true, v -> Some v
    | _ -> None

let private orFail (what: string) (text: string) =
    function
    | Some v -> v
    | None -> raise (FormatException(sprintf "not an invariant %s: '%s'" what text))

/// Parses a float32; raises FormatException when the text is not a number.
let parseF32 (text: string) =
    tryParseF32 text |> orFail "float32" text

/// Parses a float; raises FormatException when the text is not a number.
let parseF64 (text: string) = tryParseF64 text |> orFail "float" text

/// Parses an int32; raises FormatException when the text is not one.
let parseI32 (text: string) = tryParseI32 text |> orFail "int32" text

/// Parses an int64; raises FormatException when the text is not one.
let parseI64 (text: string) = tryParseI64 text |> orFail "int64" text

/// Parses a uint64; raises FormatException when the text is not one.
let parseU64 (text: string) =
    tryParseU64 text |> orFail "uint64" text

/// Lowercase hex of a byte array, two digits per byte.
let hex (bytes: byte[]) = Convert.ToHexStringLower bytes

/// Parses hex (either case, even length, no separators or prefix). None on any other text.
let tryOfHex (text: string) : byte[] option =
    if isNull text || text.Length % 2 <> 0 || not (text |> Seq.forall Uri.IsHexDigit) then
        None
    else
        Some(Convert.FromHexString text)

/// Parses hex; raises FormatException on malformed text.
let ofHex (text: string) =
    tryOfHex text |> orFail "hex string" text

/// Ordinal (culture-free) string equality.
let equals (a: string) (b: string) =
    String.Equals(a, b, StringComparison.Ordinal)

/// Lowercases with the invariant culture's rules (never the Turkish dotless i).
let lower (text: string) = text.ToLowerInvariant()
