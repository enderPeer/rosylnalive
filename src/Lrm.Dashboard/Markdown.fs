namespace Lrm.Dashboard

open System
open System.Text

/// How a Markdown document is turned into HTML: what an image becomes, how headings are numbered and shifted.
type MarkdownOptions =
    {
        /// Added to every heading level (a report's `#` becomes `<h2>` with 1), capped at `<h6>`.
        HeadingShift: int
        /// The prefix of heading ids: `<prefix>-<slug>`, numbered from -2 when a slug repeats.
        IdPrefix: string
        /// The HTML of an image that stands alone on its line (alt text, source), or None for the alt text.
        Figure: string -> string -> string option
    }

/// One heading of a rendered document: level after the shift, id and text.
type MarkdownHeading =
    { Level: int; Id: string; Text: string }

/// A small, deterministic Markdown renderer for what the report templates produce (ReportTemplates contract):
/// ATX headings, paragraphs, block quotes, bullet and numbered lists, GitHub tables with alignment, fenced code,
/// rules, images on their own line, and inline code, bold, italic, links and backslash escapes. Raw HTML is never
/// passed through: every character of the source is escaped. The same text always gives the same bytes.
[<RequireQualifiedAccess>]
module Markdown =
    type private Block =
        | Heading of level: int * text: string
        | Para of lines: string list
        | Quote of Block list
        | Items of ordered: bool * items: string list
        | Table of header: string list * aligns: string list * rows: string list list
        | Code of lines: string list
        | Image of alt: string * src: string
        | Rule

    let private isBlank (s: string) = s.Trim().Length = 0

    let private headingOf (line: string) =
        let mutable level = 0

        while level < line.Length && line[level] = '#' do
            level <- level + 1

        if level >= 1 && level <= 6 && (line.Length = level || line[level] = ' ') then
            Some(level, line.Substring(level).Trim())
        else
            None

    let private isFence (line: string) =
        line.TrimStart().StartsWith("```", StringComparison.Ordinal)

    let private isQuote (line: string) =
        line.TrimStart().StartsWith(">", StringComparison.Ordinal)

    /// A bullet ("- ", "* ", "+ ") or numbered ("1. ") item: (ordered, the item text).
    let private itemOf (line: string) =
        let t = line.TrimStart()

        if t.Length >= 2 && (t[0] = '-' || t[0] = '*' || t[0] = '+') && t[1] = ' ' then
            Some(false, t.Substring(2).Trim())
        else
            let mutable d = 0

            while d < t.Length && Char.IsAsciiDigit t[d] do
                d <- d + 1

            if d > 0 && d <= 9 && d + 1 < t.Length && t[d] = '.' && t[d + 1] = ' ' then
                Some(true, t.Substring(d + 2).Trim())
            else
                None

    let private isRule (line: string) =
        let t = line.Trim()

        t.Length >= 3
        && (t |> Seq.forall (fun c -> c = '-' || c = ' ')
            || t |> Seq.forall (fun c -> c = '*' || c = ' ')
            || t |> Seq.forall (fun c -> c = '_' || c = ' '))
        && (t |> Seq.filter (fun c -> c <> ' ') |> Seq.length) >= 3

    /// Splits a table row on the pipes that are not escaped; a leading and a trailing pipe are dropped.
    let private cells (line: string) =
        let t = line.Trim()
        let parts = ResizeArray<string>()
        let sb = StringBuilder()
        let mutable i = 0

        while i < t.Length do
            let c = t[i]

            if c = '\\' && i + 1 < t.Length && t[i + 1] = '|' then
                sb.Append("\\|") |> ignore
                i <- i + 2
            elif c = '|' then
                parts.Add(sb.ToString())
                sb.Clear() |> ignore
                i <- i + 1
            else
                sb.Append c |> ignore
                i <- i + 1

        parts.Add(sb.ToString())
        let parts = List.ofSeq parts

        let parts = if t.StartsWith('|') then List.tail parts else parts

        let parts =
            if
                t.EndsWith('|')
                && not (t.EndsWith("\\|", StringComparison.Ordinal))
                && not parts.IsEmpty
            then
                List.take (parts.Length - 1) parts
            else
                parts

        parts |> List.map _.Trim()

    let private alignOf (cell: string) =
        let c = cell.Trim()

        if
            c.Length = 0
            || not (c |> Seq.forall (fun ch -> ch = '-' || ch = ':'))
            || not (c.Contains '-')
        then
            None
        else
            let left = c.StartsWith ':'
            let right = c.EndsWith ':'

            Some(
                if left && right then "center"
                elif right then "right"
                else ""
            )

    let private separatorOf (line: string) =
        if not (line.TrimStart().StartsWith('|')) then
            None
        else
            let aligns = cells line |> List.map alignOf

            if not aligns.IsEmpty && aligns |> List.forall Option.isSome then
                Some(aligns |> List.map Option.get)
            else
                None

    /// An image that stands alone on its line: ![alt](src).
    let private imageLine (line: string) =
        let t = line.Trim()

        if t.StartsWith("![", StringComparison.Ordinal) && t.EndsWith(')') then
            let close = t.IndexOf("](", StringComparison.Ordinal)

            if close > 1 && t.IndexOf(')', close) = t.Length - 1 then
                Some(t.Substring(2, close - 2), t.Substring(close + 2, t.Length - close - 3))
            else
                None
        else
            None

    let private startsBlock (lines: string[]) (i: int) =
        let line = lines[i]

        (headingOf line).IsSome
        || isFence line
        || isQuote line
        || (itemOf line).IsSome
        || (line.TrimStart().StartsWith('|')
            && i + 1 < lines.Length
            && (separatorOf lines[i + 1]).IsSome)

    let rec private parse (lines: string[]) : Block list =
        let blocks = ResizeArray<Block>()
        let mutable i = 0

        while i < lines.Length do
            let line = lines[i]

            if isBlank line then
                i <- i + 1
            elif isFence line then
                let body = ResizeArray<string>()
                i <- i + 1

                while i < lines.Length && not (isFence lines[i]) do
                    body.Add lines[i]
                    i <- i + 1

                i <- i + 1
                blocks.Add(Code(List.ofSeq body))
            else
                match headingOf line with
                | Some(level, text) ->
                    blocks.Add(Heading(level, text))
                    i <- i + 1
                | None when isQuote line ->
                    let inner = ResizeArray<string>()

                    while i < lines.Length && isQuote lines[i] do
                        let t = lines[i].TrimStart().Substring(1)
                        inner.Add(if t.StartsWith(' ') then t.Substring 1 else t)
                        i <- i + 1

                    blocks.Add(Quote(parse (Array.ofSeq inner)))
                | None when
                    line.TrimStart().StartsWith('|')
                    && i + 1 < lines.Length
                    && (separatorOf lines[i + 1]).IsSome
                    ->
                    let header = cells line
                    let aligns = (separatorOf lines[i + 1]).Value
                    i <- i + 2
                    let rows = ResizeArray<string list>()

                    while i < lines.Length && lines[i].TrimStart().StartsWith('|') do
                        rows.Add(cells lines[i])
                        i <- i + 1

                    blocks.Add(Table(header, aligns, List.ofSeq rows))
                | None when (itemOf line).IsSome ->
                    let ordered, _ = (itemOf line).Value
                    let items = ResizeArray<string>()

                    while i < lines.Length
                          && not (isBlank lines[i])
                          && (match itemOf lines[i] with
                              | Some(o, _) -> o = ordered
                              | None -> items.Count > 0 && lines[i].StartsWith("  ", StringComparison.Ordinal)) do
                        match itemOf lines[i] with
                        | Some(_, text) when not (lines[i].StartsWith("  ", StringComparison.Ordinal)) -> items.Add text
                        | _ -> items[items.Count - 1] <- items[items.Count - 1] + "\n" + lines[i].Trim()

                        i <- i + 1

                    blocks.Add(Items(ordered, List.ofSeq items))
                | None when isRule line ->
                    blocks.Add Rule
                    i <- i + 1
                | None when (imageLine line).IsSome ->
                    let alt, src = (imageLine line).Value
                    blocks.Add(Image(alt, src))
                    i <- i + 1
                | None ->
                    let para = ResizeArray<string>()
                    para.Add(line.Trim())
                    i <- i + 1

                    while i < lines.Length && not (isBlank lines[i]) && not (startsBlock lines i) do
                        para.Add(lines[i].Trim())
                        i <- i + 1

                    blocks.Add(Para(List.ofSeq para))

        List.ofSeq blocks

    let private isPunct (c: char) =
        c < '\u0080' && (Char.IsPunctuation c || Char.IsSymbol c)

    let private isWordChar (c: char) = Char.IsLetterOrDigit c

    /// The index of the closing delimiter run of `count` `ch` characters after `from`, or -1. A closing run follows
    /// a non-space character and, for '_', is not followed by a letter or digit.
    let private closingRun (s: string) (from: int) (ch: char) (count: int) =
        let mutable found = -1
        let mutable j = from

        while found < 0 && j <= s.Length - count do
            if s[j] = '\\' then
                j <- j + 2
            elif s[j] = '`' then
                // Emphasis never closes inside a code span.
                let close = s.IndexOf('`', j + 1)
                j <- if close < 0 then j + 1 else close + 1
            else
                let mutable run = 0

                while j + run < s.Length && s[j + run] = ch do
                    run <- run + 1

                if run >= count && j > from && not (Char.IsWhiteSpace s[j - 1]) then
                    let after = j + count

                    if ch <> '_' || after >= s.Length || not (isWordChar s[after]) then
                        found <- j
                    else
                        j <- j + run
                elif run > 0 then
                    j <- j + run
                else
                    j <- j + 1

        found

    /// Renders inline Markdown to HTML.
    let rec inlineHtml (figure: string -> string -> string option) (s: string) : string =
        let sb = StringBuilder()
        let mutable i = 0

        let escape (c: char) = Html.appendEscaped sb false (string c)

        while i < s.Length do
            let c = s[i]

            if c = '\\' && i + 1 < s.Length && isPunct s[i + 1] then
                escape s[i + 1]
                i <- i + 2
            elif c = '\\' && i + 1 < s.Length && s[i + 1] = '\n' then
                sb.Append "<br>\n" |> ignore
                i <- i + 2
            elif c = '`' then
                let mutable run = 0

                while i + run < s.Length && s[i + run] = '`' do
                    run <- run + 1

                let fence = String('`', run)
                let close = s.IndexOf(fence, i + run, StringComparison.Ordinal)

                if close < 0 then
                    sb.Append fence |> ignore
                    i <- i + run
                else
                    let code = s.Substring(i + run, close - i - run).Replace('\n', ' ')

                    let code =
                        if code.Length >= 2 && code.StartsWith(' ') && code.EndsWith(' ') then
                            code.Substring(1, code.Length - 2)
                        else
                            code

                    sb.Append("<code>").Append(Html.text code).Append("</code>") |> ignore
                    i <- close + run
            elif (c = '!' && i + 1 < s.Length && s[i + 1] = '[') || c = '[' then
                let image = c = '!'
                let start = if image then i + 2 else i + 1
                let close = s.IndexOf("](", start, StringComparison.Ordinal)
                let stop = if close < 0 then -1 else s.IndexOf(')', close + 2)

                if close < 0 || stop < 0 || s.IndexOf('\n', start, stop - start) >= 0 then
                    escape c
                    i <- i + 1
                else
                    let label = s.Substring(start, close - start)
                    let target = s.Substring(close + 2, stop - close - 2)

                    if image then
                        match figure label target with
                        | Some html -> sb.Append html |> ignore
                        | None -> sb.Append(Html.text label) |> ignore
                    elif Html.isLocalHref target then
                        sb
                            .Append("<a href=\"")
                            .Append(Html.attr target)
                            .Append("\">")
                            .Append(inlineHtml figure label)
                            .Append("</a>")
                        |> ignore
                    else
                        // A link to the network (a scheme, a network path such as //host or \\host, or anything a
                        // browser could turn into one) stays text: the dashboard never points outside the machine.
                        sb
                            .Append(inlineHtml figure label)
                            .Append(" (<code>")
                            .Append(Html.text target)
                            .Append("</code>)")
                        |> ignore

                    i <- stop + 1
            elif c = '*' || c = '_' then
                let mutable run = 0

                while i + run < s.Length && s[i + run] = c do
                    run <- run + 1

                let opens =
                    i + run < s.Length
                    && not (Char.IsWhiteSpace s[i + run])
                    && (c <> '_' || i = 0 || not (isWordChar s[i - 1]))

                let count = if run >= 2 then 2 else 1
                let close = if opens then closingRun s (i + count) c count else -1

                if close < 0 then
                    sb.Append(c, run) |> ignore
                    i <- i + run
                else
                    let inner = s.Substring(i + count, close - i - count)
                    let tag = if count = 2 then "strong" else "em"

                    sb
                        .Append('<')
                        .Append(tag)
                        .Append('>')
                        .Append(inlineHtml figure inner)
                        .Append("</")
                        .Append(tag)
                        .Append('>')
                    |> ignore

                    i <- close + count
            else
                escape c
                i <- i + 1

        sb.ToString()

    /// Renders a document. Returns the HTML and its headings in order.
    let render (o: MarkdownOptions) (md: string) : string * MarkdownHeading list =
        let lines = (if isNull md then "" else md).Replace("\r\n", "\n").Split('\n')
        let blocks = parse lines
        let sb = StringBuilder()
        let headings = ResizeArray<MarkdownHeading>()
        let used = Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal)
        let inl = inlineHtml o.Figure

        let idOf (text: string) =
            let slug = Html.slug text

            let n =
                (match used.TryGetValue slug with
                 | true, k -> k
                 | _ -> 0)
                + 1

            used[slug] <- n

            let s =
                if n = 1 then
                    slug
                else
                    slug + "-" + Lrm.Core.Invariant.i32 n

            if o.IdPrefix = "" then s else o.IdPrefix + "-" + s

        let rec emit (blocks: Block list) =
            for b in blocks do
                match b with
                | Heading(level, text) ->
                    let level = min 6 (max 1 (level + o.HeadingShift))
                    let id = idOf text
                    let html = inl text

                    headings.Add { Level = level; Id = id; Text = text }

                    sb.Append(Lrm.Core.Invariant.format $"<h{level} id=\"").Append(Html.attr id).Append("\">")
                    |> ignore

                    sb.Append(html).Append(Lrm.Core.Invariant.format $"</h{level}>\n") |> ignore
                | Para lines ->
                    sb.Append("<p>").Append(inl (String.concat "\n" lines)).Append("</p>\n")
                    |> ignore
                | Quote inner ->
                    sb.Append("<blockquote>\n") |> ignore
                    emit inner
                    sb.Append("</blockquote>\n") |> ignore
                | Items(ordered, items) ->
                    let tag = if ordered then "ol" else "ul"
                    sb.Append('<').Append(tag).Append(">\n") |> ignore

                    for item in items do
                        sb.Append("<li>").Append(inl item).Append("</li>\n") |> ignore

                    sb.Append("</").Append(tag).Append(">\n") |> ignore
                | Table(header, aligns, rows) ->
                    let alignAttr k =
                        match List.tryItem k aligns with
                        | Some "right" -> " class=\"d-r\""
                        | Some "center" -> " class=\"d-c\""
                        | _ -> ""

                    sb.Append("<div class=\"d-scroll\"><table class=\"d-table\">\n<thead><tr>")
                    |> ignore

                    header
                    |> List.iteri (fun k h ->
                        sb.Append("<th").Append(alignAttr k).Append('>').Append(inl h).Append("</th>")
                        |> ignore)

                    sb.Append("</tr></thead>\n<tbody>\n") |> ignore

                    for row in rows do
                        sb.Append("<tr>") |> ignore
                        let row = row @ List.replicate (max 0 (header.Length - row.Length)) ""

                        row
                        |> List.truncate (max header.Length 1)
                        |> List.iteri (fun k cell ->
                            sb.Append("<td").Append(alignAttr k).Append('>').Append(inl cell).Append("</td>")
                            |> ignore)

                        sb.Append("</tr>\n") |> ignore

                    sb.Append("</tbody></table></div>\n") |> ignore
                | Code lines ->
                    sb.Append("<pre><code>").Append(Html.text (String.concat "\n" lines)).Append("</code></pre>\n")
                    |> ignore
                | Image(alt, src) ->
                    match o.Figure alt src with
                    | Some html -> sb.Append(html).Append('\n') |> ignore
                    | None ->
                        sb.Append("<p class=\"d-muted\">").Append(Html.text alt).Append("</p>\n")
                        |> ignore
                | Rule -> sb.Append("<hr>\n") |> ignore

        emit blocks
        sb.ToString(), List.ofSeq headings
