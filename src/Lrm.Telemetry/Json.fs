namespace Lrm.Telemetry

open System
open System.Text
open Lrm.Core

/// A JSON value with ordered object keys, for deterministic writing (RunLog contract).
[<RequireQualifiedAccess>]
type JsonValue =
    | Null
    | Bool of bool
    | Int of int64
    | Float of float
    | Str of string
    | Arr of JsonValue list
    | Obj of (string * JsonValue) list

/// Per-thread scratch space, so numbers are formatted without allocating.
[<Sealed; AbstractClass>]
type internal Scratch =
    [<ThreadStatic; DefaultValue>]
    static val mutable private chars: char[]

    static member Chars =
        match Scratch.chars with
        | null ->
            let c = Array.zeroCreate 64
            Scratch.chars <- c
            c
        | c -> c

/// Deterministic JSON text (RunLog contract, "JSON text"). The same value always gives the same bytes, under any
/// culture: strings are escaped by one fixed rule, floats use the invariant "R" format, and object keys keep the order
/// they are given in.
[<RequireQualifiedAccess>]
module Json =
    let private hexDigits = "0123456789abcdef"

    /// Appends `s` as a quoted JSON string. Escapes: quote, backslash, \b \f \n \r \t, other characters below U+0020
    /// as \u00xx (lowercase hex), and U+2028 and U+2029 as \u2028 and \u2029. A lone surrogate becomes U+FFFD. Every
    /// other character is written as itself (UTF-8 in the file).
    let appendString (sb: StringBuilder) (s: string) =
        sb.Append '"' |> ignore
        let s = if isNull s then "" else s
        // Fast path: most names and values need no escape at all, so their characters go in as one block.
        let mutable plain = 0

        while plain < s.Length
              && (let c = s[plain]
                  c >= ' ' && c <> '"' && c <> '\\' && c < '\u2028' && not (Char.IsSurrogate c)) do
            plain <- plain + 1

        sb.Append(s, 0, plain) |> ignore
        let mutable i = plain

        while i < s.Length do
            let c = s[i]

            match c with
            | '"' -> sb.Append "\\\"" |> ignore
            | '\\' -> sb.Append "\\\\" |> ignore
            | '\b' -> sb.Append "\\b" |> ignore
            | '\u000C' -> sb.Append "\\f" |> ignore
            | '\n' -> sb.Append "\\n" |> ignore
            | '\r' -> sb.Append "\\r" |> ignore
            | '\t' -> sb.Append "\\t" |> ignore
            | '\u2028' -> sb.Append "\\u2028" |> ignore
            | '\u2029' -> sb.Append "\\u2029" |> ignore
            | c when c < ' ' ->
                sb.Append("\\u00").Append(hexDigits[int c >>> 4]).Append(hexDigits[int c &&& 15])
                |> ignore
            | c when Char.IsHighSurrogate c && i + 1 < s.Length && Char.IsLowSurrogate s[i + 1] ->
                sb.Append(c).Append(s[i + 1]) |> ignore
                i <- i + 1
            | c when Char.IsSurrogate c -> sb.Append '\uFFFD' |> ignore
            | c -> sb.Append c |> ignore

            i <- i + 1

        sb.Append '"' |> ignore

    /// `s` as a quoted JSON string (see appendString).
    let quote (s: string) =
        let sb = StringBuilder(s.Length + 2)
        appendString sb s
        sb.ToString()

    /// The text of a finite float in the invariant "R" format ("0.5", "1E-05", "-0"), or None for NaN and infinities.
    let finiteText (x: float) =
        if Double.IsFinite x then Some(Invariant.f64 x) else None

    /// The JSON string that stands for a non-finite float: "NaN", "Infinity" or "-Infinity".
    let nonFiniteName (x: float) =
        if Double.IsNaN x then "NaN"
        elif x > 0.0 then "Infinity"
        else "-Infinity"

    /// Parses the name of a non-finite float ("NaN", "Infinity", "-Infinity").
    let tryNonFinite (s: string) =
        match s with
        | "NaN" -> Some Double.NaN
        | "Infinity" -> Some Double.PositiveInfinity
        | "-Infinity" -> Some Double.NegativeInfinity
        | _ -> None

    /// Appends an integer in invariant decimal form (the same text as Invariant.i64), without allocating.
    let appendInt64 (sb: StringBuilder) (v: int64) =
        let buf = Scratch.Chars
        let mutable n = 0

        v.TryFormat(Span<char>(buf), &n, ReadOnlySpan<char>.Empty, Invariant.culture)
        |> ignore

        sb.Append(buf, 0, n) |> ignore

    /// Appends the "R" text of a finite float (the same text as Invariant.f64), without allocating; true when that
    /// text has a '.' or an exponent.
    let private appendR (sb: StringBuilder) (x: float) =
        let buf = Scratch.Chars
        let mutable n = 0
        x.TryFormat(Span<char>(buf), &n, "R".AsSpan(), Invariant.culture) |> ignore
        sb.Append(buf, 0, n) |> ignore
        Span<char>(buf, 0, n).IndexOfAny('.', 'E') >= 0

    /// Appends a float that is always a float: a finite value as "R" text, or the quoted name of a non-finite one.
    let appendNumber (sb: StringBuilder) (x: float) =
        if Double.IsFinite x then
            appendR sb x |> ignore
        else
            appendString sb (nonFiniteName x)

    /// Appends a float that must stay distinguishable from an integer: "R" text, plus ".0" when that text has neither
    /// a '.' nor an exponent; a non-finite value becomes {"$f":"NaN"} (or "Infinity", "-Infinity").
    let appendTypedFloat (sb: StringBuilder) (x: float) =
        if Double.IsFinite x then
            if not (appendR sb x) then
                sb.Append ".0" |> ignore
        else
            sb.Append "{\"$f\":" |> ignore
            appendString sb (nonFiniteName x)
            sb.Append '}' |> ignore

    let rec private write (sb: StringBuilder) (indent: int option) (depth: int) (v: JsonValue) =
        let newline (d: int) =
            match indent with
            | Some n -> sb.Append('\n').Append(' ', n * d) |> ignore
            | None -> ()

        let sep = if indent.IsSome then ": " else ":"

        match v with
        | JsonValue.Null -> sb.Append "null" |> ignore
        | JsonValue.Bool b -> sb.Append(if b then "true" else "false") |> ignore
        | JsonValue.Int i -> appendInt64 sb i
        | JsonValue.Float f -> appendNumber sb f
        | JsonValue.Str s -> appendString sb s
        | JsonValue.Arr [] -> sb.Append "[]" |> ignore
        | JsonValue.Arr items ->
            sb.Append '[' |> ignore

            items
            |> List.iteri (fun i item ->
                if i > 0 then
                    sb.Append ',' |> ignore

                newline (depth + 1)
                write sb indent (depth + 1) item)

            newline depth
            sb.Append ']' |> ignore
        | JsonValue.Obj [] -> sb.Append "{}" |> ignore
        | JsonValue.Obj fields ->
            sb.Append '{' |> ignore

            fields
            |> List.iteri (fun i (k, item) ->
                if i > 0 then
                    sb.Append ',' |> ignore

                newline (depth + 1)
                appendString sb k
                sb.Append sep |> ignore
                write sb indent (depth + 1) item)

            newline depth
            sb.Append '}' |> ignore

    /// Compact JSON text, with no white space.
    let compact (v: JsonValue) =
        let sb = StringBuilder()
        write sb None 0 v
        sb.ToString()

    /// JSON text indented by two spaces, with LF line ends and a final LF.
    let pretty (v: JsonValue) =
        let sb = StringBuilder()
        write sb (Some 2) 0 v
        sb.Append('\n').ToString()

    /// Unix nanoseconds as ISO 8601 UTC with milliseconds: "2026-10-18T09:30:00.000Z".
    let isoTime (unixNs: int64) =
        let ms =
            if unixNs >= 0L then
                unixNs / 1_000_000L
            else
                -((-unixNs + 999_999L) / 1_000_000L)

        DateTimeOffset
            .FromUnixTimeMilliseconds(ms)
            .UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", Invariant.culture)
