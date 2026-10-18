namespace Lrm.Reports

open System
open System.Text
open Lrm.Core

/// The colour of a mark, by the job it does (ReportTemplates contract, "Palette"). Categorical slots identify
/// entities in a fixed order; a slot past the eighth folds into Neutral ("other"). Status colours mark state and are
/// never used as series colours.
[<RequireQualifiedAccess>]
type Color =
    /// Categorical slot 0 to 7: blue, orange, aqua, yellow, magenta, green, violet, red.
    | Cat of int
    /// Grey, for "other" and for context marks.
    | Neutral
    | Good
    | Warning
    | Serious
    | Critical

[<RequireQualifiedAccess>]
module Color =
    /// The CSS class suffix of a colour: "0".."7", "n", "good", "warn", "serious", "crit".
    let key (c: Color) =
        match c with
        | Color.Cat i when i >= 0 && i < 8 -> Invariant.i32 i
        | Color.Cat _
        | Color.Neutral -> "n"
        | Color.Good -> "good"
        | Color.Warning -> "warn"
        | Color.Serious -> "serious"
        | Color.Critical -> "crit"

    /// The fill class, for example "f0".
    let fill (c: Color) = "f" + key c

    /// The stroke class, for example "s0".
    let stroke (c: Color) = "s" + key c

    /// Slot `i` in fixed order; past the eighth slot, Neutral.
    let slot (i: int) =
        if i >= 0 && i < 8 then Color.Cat i else Color.Neutral

/// The chart theme (ReportTemplates contract, "Palette"): the validated reference palette, light by default and its
/// own dark steps under prefers-color-scheme: dark. Every chart carries its own surface, so it reads on a light or a
/// dark page either way. The light categorical steps pass the colour-vision checks (worst adjacent CVD delta E 9.1,
/// normal-vision 19.6); three of them sit below 3:1 on the light surface, so every chart ships with a legend and the
/// report shows the numbers in a table beside it.
[<RequireQualifiedAccess>]
module Theme =
    /// Categorical slots, light surface.
    let categoricalLight =
        [|
            "#2a78d6"
            "#eb6834"
            "#1baf7a"
            "#eda100"
            "#e87ba4"
            "#008300"
            "#6250d6"
            "#e34948"
        |]

    /// Categorical slots, dark surface.
    let categoricalDark =
        [|
            "#3987e5"
            "#d95926"
            "#199e70"
            "#c98500"
            "#d55181"
            "#008300"
            "#9085e9"
            "#e66767"
        |]

    /// The sequential blue ramp, low to high (steps 100 to 700).
    let sequential =
        [|
            "#cde2fb"
            "#b7d3f6"
            "#9ec5f4"
            "#86b6ef"
            "#6da7ec"
            "#5598e7"
            "#3987e5"
            "#2a78d6"
            "#256abf"
            "#1c5cab"
            "#184f95"
            "#104281"
            "#0d366b"
        |]

    /// Number of sequential steps.
    let steps = sequential.Length

    type private Tokens =
        {
            Surface: string
            Ink: string
            Ink2: string
            Muted: string
            Grid: string
            Axis: string
            Border: string
            Band: string
            Neutral: string
            Missing: string
            Cat: string[]
            Seq: string[]
        }

    let private light =
        {
            Surface = "#fcfcfb"
            Ink = "#0b0b0b"
            Ink2 = "#52514e"
            // Axis ticks and labels: 4.98:1 on the light surface (dark: #898781, 4.85:1).
            Muted = "#6f6e69"
            Grid = "#e1e0d9"
            Axis = "#c3c2b7"
            Border = "#e6e5e0"
            Band = "#f0efec"
            Neutral = "#a3a29c"
            Missing = "#f0efec"
            Cat = categoricalLight
            Seq = sequential
        }

    let private dark =
        {
            Surface = "#1a1a19"
            Ink = "#f0efec"
            Ink2 = "#c3c2b7"
            Muted = "#898781"
            Grid = "#2c2c2a"
            Axis = "#383835"
            Border = "#2c2c2a"
            Band = "#242423"
            Neutral = "#6f6e69"
            Missing = "#242423"
            Cat = categoricalDark
            // Near zero recedes towards the surface: the dark ramp runs from the darkest step to the lightest.
            Seq = Array.rev sequential
        }

    let private status =
        [
            "good", "#0ca30c"
            "warn", "#fab219"
            "serious", "#ec835a"
            "crit", "#d03b3b"
        ]

    let private rules (t: Tokens) =
        let sb = StringBuilder()

        let rule (sel: string) (body: string) =
            sb.Append(sel).Append('{').Append(body).Append('}') |> ignore

        rule ".bg" ("fill:" + t.Surface)
        rule ".frame" ("fill:none;stroke:" + t.Border)
        rule ".t1" ("fill:" + t.Ink)
        rule ".t2" ("fill:" + t.Ink2)
        rule ".t3" ("fill:" + t.Muted)
        rule ".grid" ("stroke:" + t.Grid)
        rule ".axis" ("stroke:" + t.Axis)
        rule ".ref" ("stroke:" + t.Ink2)
        rule ".band" ("fill:" + t.Band)
        rule ".ring" ("stroke:" + t.Surface)
        rule ".gap" ("stroke:" + t.Surface)
        rule ".qx" ("fill:" + t.Missing)
        // An exact zero on a log colour scale: an outlined empty cell.
        rule ".qz" ("fill:" + t.Surface + ";stroke:" + t.Ink2)
        // The outline of a timeline outcome to notice (warning, serious, critical).
        rule ".emph" ("stroke:" + t.Ink2)
        rule ".fn" ("fill:" + t.Neutral)
        rule ".sn" ("stroke:" + t.Neutral)

        t.Cat
        |> Array.iteri (fun i c ->
            rule (".f" + Invariant.i32 i) ("fill:" + c)
            rule (".s" + Invariant.i32 i) ("stroke:" + c))

        t.Seq |> Array.iteri (fun i c -> rule (".q" + Invariant.i32 i) ("fill:" + c))
        sb.ToString()

    /// The <style> text every chart embeds.
    let style =
        let sb = StringBuilder()

        sb.Append("text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}")
        |> ignore

        sb.Append(".line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}")
        |> ignore

        sb.Append(rules light) |> ignore

        for k, c in status do
            sb.Append(".f" + k + "{fill:" + c + "}.s" + k + "{stroke:" + c + "}") |> ignore

        sb.Append("@media (prefers-color-scheme:dark){").Append(rules dark).Append('}').ToString()

/// A small deterministic SVG writer: one element per line, attributes in the order given, coordinates rounded to
/// 0.01 in invariant form.
[<Sealed>]
type SvgWriter() =
    let sb = StringBuilder(16384)

    /// A coordinate or length: at most two decimals, no trailing zeros, never "-0", within ±1e6.
    static member Num(x: float) =
        if not (Double.IsFinite x) then
            "0"
        else
            let s = (max -1e6 (min 1e6 x)).ToString("0.##", Invariant.culture)
            if s = "-0" then "0" else s

    /// XML-escapes text (and attribute values): & < > " are escaped, control characters are dropped, and characters
    /// XML cannot hold (U+FFFE, U+FFFF and unpaired surrogates) become U+FFFD, so any log string makes valid SVG.
    static member Escape(s: string) =
        let s = if isNull s then "" else s
        let sb = StringBuilder(s.Length)
        let mutable i = 0

        while i < s.Length do
            let c = s[i]

            match c with
            | '&' -> sb.Append "&amp;" |> ignore
            | '<' -> sb.Append "&lt;" |> ignore
            | '>' -> sb.Append "&gt;" |> ignore
            | '"' -> sb.Append "&quot;" |> ignore
            | c when c < ' ' -> ()
            | '\uFFFE'
            | '\uFFFF' -> sb.Append '\uFFFD' |> ignore
            | c when Char.IsHighSurrogate c && i + 1 < s.Length && Char.IsLowSurrogate s[i + 1] ->
                sb.Append(c).Append(s[i + 1]) |> ignore
                i <- i + 1
            | c when Char.IsSurrogate c -> sb.Append '\uFFFD' |> ignore
            | c -> sb.Append c |> ignore

            i <- i + 1

        sb.ToString()

    member private _.Attrs(attrs: (string * string) list) =
        for k, v in attrs do
            sb.Append(' ').Append(k).Append("=\"").Append(SvgWriter.Escape v).Append('"')
            |> ignore

    /// <tag attrs>
    member x.Open(tag: string, attrs: (string * string) list) =
        sb.Append('<').Append(tag) |> ignore
        x.Attrs attrs
        sb.Append(">\n") |> ignore

    /// </tag>
    member _.Close(tag: string) =
        sb.Append("</").Append(tag).Append(">\n") |> ignore

    /// <tag attrs/>
    member x.Leaf(tag: string, attrs: (string * string) list) =
        sb.Append('<').Append(tag) |> ignore
        x.Attrs attrs
        sb.Append("/>\n") |> ignore

    /// <tag attrs><title>tip</title></tag>: a mark with a hover tooltip.
    member x.LeafWithTitle(tag: string, attrs: (string * string) list, tip: string) =
        if String.IsNullOrEmpty tip then
            x.Leaf(tag, attrs)
        else
            sb.Append('<').Append(tag) |> ignore
            x.Attrs attrs

            sb.Append("><title>").Append(SvgWriter.Escape tip).Append("</title></").Append(tag).Append(">\n")
            |> ignore

    /// <tag attrs>text</tag>
    member x.Text(tag: string, attrs: (string * string) list, text: string) =
        sb.Append('<').Append(tag) |> ignore
        x.Attrs attrs

        sb.Append('>').Append(SvgWriter.Escape text).Append("</").Append(tag).Append(">\n")
        |> ignore

    /// Raw markup, already escaped.
    member _.Raw(text: string) = sb.Append(text) |> ignore

    override _.ToString() = sb.ToString()

/// Text measurement for layout. There is no font engine here, so widths are estimates from per-character classes;
/// they are deterministic, which is what matters for byte-identical output.
[<RequireQualifiedAccess>]
module TextMetrics =
    let private charWidth (c: char) =
        match c with
        | 'i'
        | 'l'
        | 'j'
        | '.'
        | ','
        | ':'
        | ';'
        | '|'
        | '!'
        | '\''
        | 'I'
        | 'f'
        | 't'
        | 'r' -> 0.32
        | ' ' -> 0.3
        | 'm'
        | 'w'
        | 'M'
        | 'W' -> 0.86
        | c when c >= 'A' && c <= 'Z' -> 0.66
        | c when c >= '0' && c <= '9' -> 0.56
        | _ -> 0.56

    /// Estimated width in px of `text` at font size `size`.
    let width (size: float) (text: string) =
        let mutable w = 0.0

        for c in text do
            w <- w + charWidth c

        w * size

    /// `text` cut to fit `maxWidth` px at `size`, with an ellipsis when cut. A cut never splits a surrogate pair.
    let fit (size: float) (maxWidth: float) (text: string) =
        if width size text <= maxWidth then
            text
        else
            let mutable s = text

            while s.Length > 1 && width size (s + "…") > maxWidth do
                let cut =
                    if
                        s.Length >= 2
                        && Char.IsLowSurrogate s[s.Length - 1]
                        && Char.IsHighSurrogate s[s.Length - 2]
                    then
                        2
                    else
                        1

                s <- s.Substring(0, s.Length - cut)

            if s.Length > 0 && Char.IsHighSurrogate s[s.Length - 1] then
                s <- s.Substring(0, s.Length - 1)

            s + "…"
