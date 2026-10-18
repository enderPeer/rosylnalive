namespace Lrm.Reports

open System
open System.Text
open Lrm.Core

/// Column alignment in a Markdown table.
[<RequireQualifiedAccess>]
type Align =
    | Left
    | Right
    | Center

/// A table for a {{table name}} placeholder: column headers with alignment, then rows of cell text.
type Table =
    {
        Columns: (string * Align) list
        Rows: string list list
    }

/// The data a template is filled from (ReportTemplates contract). A builder returns one Scope; each {{#each}} item is
/// a nested Scope whose names shadow the outer ones.
type Scope =
    {
        Values: Map<string, string>
        Tables: Map<string, Table>
        Charts: Map<string, Chart>
        Lists: Map<string, Scope list>
    }

[<RequireQualifiedAccess>]
module Scope =
    let empty =
        {
            Values = Map.empty
            Tables = Map.empty
            Charts = Map.empty
            Lists = Map.empty
        }

    /// Adds a {{value name}}.
    let value (name: string) (text: string) (s: Scope) =
        { s with
            Values = s.Values.Add(name, text)
        }

    /// Adds a {{table name}}.
    let table (name: string) (t: Table) (s: Scope) =
        { s with
            Tables = s.Tables.Add(name, t)
        }

    /// Adds a {{chart name}}.
    let chart (name: string) (c: Chart) (s: Scope) =
        { s with
            Charts = s.Charts.Add(name, c)
        }

    /// Adds an {{#each name}} list.
    let list (name: string) (items: Scope list) (s: Scope) =
        { s with
            Lists = s.Lists.Add(name, items)
        }

/// A template that does not fit its data: an unknown directive, a bad name, an unclosed block, or a name the scope
/// does not define. The message names the template and the line.
exception TemplateError of string

/// One parsed piece of a template.
[<RequireQualifiedAccess>]
type TemplateNode =
    | Text of string
    | Value of name: string * line: int
    | Table of name: string * line: int
    | Chart of name: string * line: int
    | Each of name: string * body: TemplateNode list * line: int

/// The rendered Markdown and the charts it links to, by file name under charts/.
type RenderedReport =
    {
        Markdown: string
        Charts: (string * Chart) list
    }

/// The template engine (ReportTemplates contract, "Templates").
///
/// Placeholders: {{value NAME}}, {{table NAME}}, {{chart NAME}}, {{#each NAME}} ... {{/each}}, and {{! comment }}.
/// NAME is [a-z][a-z0-9_.]*. A line that holds only an {{#each}}, {{/each}} or comment tag disappears with its line
/// break, so blocks do not leave blank lines. Every name must resolve; a missing one is an error, never an empty
/// string.
[<RequireQualifiedAccess>]
module Template =
    let private isName (s: string) =
        s.Length > 0
        && s[0] >= 'a'
        && s[0] <= 'z'
        && s
           |> Seq.forall (fun c -> (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c = '_' || c = '.')

    type private Tag =
        | TValue of string
        | TTable of string
        | TChart of string
        | TOpen of string
        | TClose
        | TComment

    type private Raw =
        | RText of string
        | RTag of Tag * line: int

    let private lineOf (text: string) (index: int) =
        let mutable line = 1

        for i in 0 .. index - 1 do
            if text[i] = '\n' then
                line <- line + 1

        line

    let private tokenize (name: string) (text: string) : Raw list =
        let tokens = ResizeArray<Raw>()
        let mutable i = 0

        let err (at: int) (msg: string) =
            raise (TemplateError(sprintf "%s line %d: %s" name (lineOf text at) msg))

        while i < text.Length do
            let start = text.IndexOf("{{", i, StringComparison.Ordinal)

            if start < 0 then
                tokens.Add(RText(text.Substring i))
                i <- text.Length
            else
                if start > i then
                    tokens.Add(RText(text.Substring(i, start - i)))

                if start + 2 < text.Length && text[start + 2] = '!' then
                    let close = text.IndexOf("}}", start + 3, StringComparison.Ordinal)

                    if close < 0 then
                        err start "unclosed comment"

                    tokens.Add(RTag(TComment, lineOf text start))
                    i <- close + 2
                else
                    let close = text.IndexOf("}}", start + 2, StringComparison.Ordinal)

                    if close < 0 then
                        err start "unclosed {{"

                    let inner = text.Substring(start + 2, close - start - 2).Trim()
                    let parts = inner.Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)

                    let named (kind: string) (f: string -> Tag) =
                        match parts with
                        | [| _; n |] when isName n -> f n
                        | [| _; n |] -> err start (sprintf "'%s' is not a valid name in {{%s}}" n inner)
                        | _ -> err start (sprintf "{{%s}} needs exactly one name" kind)

                    let tag =
                        match parts with
                        | [| "/each" |] -> TClose
                        | _ when parts.Length > 0 ->
                            match parts[0] with
                            | "value" -> named "value" TValue
                            | "table" -> named "table" TTable
                            | "chart" -> named "chart" TChart
                            | "#each" -> named "#each" TOpen
                            | other -> err start (sprintf "unknown directive '%s'" other)
                        | _ -> err start "empty {{}}"

                    tokens.Add(RTag(tag, lineOf text start))
                    i <- close + 2

        List.ofSeq tokens

    let private isStandalone =
        function
        | TOpen _
        | TClose
        | TComment -> true
        | _ -> false

    /// Removes the line of every standalone block or comment tag: the white space before it on its line, and the
    /// white space and line break after it. Decisions are made on the original tokens, then applied together.
    let private trimStandalone (tokens: Raw list) =
        let arr = Array.ofList tokens
        let last = arr.Length - 1

        let blank (s: string) =
            s |> Seq.forall (fun c -> c = ' ' || c = '\t')

        let standalone k =
            match arr[k] with
            | RTag(tag, _) when isStandalone tag ->
                let before =
                    if k = 0 then
                        true
                    else
                        match arr[k - 1] with
                        | RText t ->
                            let nl = t.LastIndexOf '\n'
                            blank (t.Substring(nl + 1)) && (nl >= 0 || k - 1 = 0)
                        | RTag _ -> false

                let after =
                    if k = last then
                        true
                    else
                        match arr[k + 1] with
                        | RText t ->
                            let nl = t.IndexOf '\n'
                            blank (if nl < 0 then t else t.Substring(0, nl)) && (nl >= 0 || k + 1 = last)
                        | RTag _ -> false

                before && after
            | _ -> false

        let marks = Array.init arr.Length standalone

        arr
        |> Array.mapi (fun k token ->
            match token with
            | RText t ->
                let cutHead = k > 0 && marks[k - 1]
                let cutTail = k < last && marks[k + 1]

                let headEnd =
                    if cutHead then
                        (let nl = t.IndexOf '\n' in if nl < 0 then t.Length else nl + 1)
                    else
                        0

                let tailStart = if cutTail then t.LastIndexOf '\n' + 1 else t.Length
                let tailStart = max headEnd tailStart
                RText(t.Substring(headEnd, tailStart - headEnd))
            | tag -> tag)
        |> List.ofArray

    let rec private build (name: string) (tokens: Raw list) (closing: (string * int) option) =
        let nodes = ResizeArray<TemplateNode>()
        let mutable rest = tokens
        let mutable closed = false

        while not closed && not rest.IsEmpty do
            match rest with
            | RText t :: tail ->
                if t.Length > 0 then
                    nodes.Add(TemplateNode.Text t)

                rest <- tail
            | RTag(tag, line) :: tail ->
                rest <- tail

                match tag with
                | TValue n -> nodes.Add(TemplateNode.Value(n, line))
                | TTable n -> nodes.Add(TemplateNode.Table(n, line))
                | TChart n -> nodes.Add(TemplateNode.Chart(n, line))
                | TComment -> ()
                | TOpen n ->
                    let body, after = build name rest (Some(n, line))
                    nodes.Add(TemplateNode.Each(n, body, line))
                    rest <- after
                | TClose ->
                    match closing with
                    | Some _ -> closed <- true
                    | None -> raise (TemplateError(sprintf "%s line %d: {{/each}} without {{#each}}" name line))
            | [] -> ()

        match closing with
        | Some(n, line) when not closed ->
            raise (TemplateError(sprintf "%s line %d: {{#each %s}} is never closed" name line n))
        | _ -> ()

        List.ofSeq nodes, rest

    /// Parses template text (CRLF is read as LF). Raises TemplateError.
    let parse (templateName: string) (text: string) : TemplateNode list =
        let text = text.Replace("\r\n", "\n")

        let nodes, _ =
            build templateName (tokenize templateName text |> trimStandalone) None

        nodes

    /// A Markdown table cell: pipes escaped, line breaks as spaces.
    let cell (text: string) =
        text.Replace("|", "\\|").Replace("\r\n", " ").Replace('\n', ' ').Replace('\r', ' ')

    /// A GitHub-flavoured Markdown table, or "_No rows._" when it has none.
    let renderTable (t: Table) =
        if t.Rows.IsEmpty then
            "_No rows._"
        else
            let sb = StringBuilder()
            sb.Append('|') |> ignore

            for h, _ in t.Columns do
                sb.Append(' ').Append(cell h).Append(" |") |> ignore

            sb.Append("\n|") |> ignore

            for _, a in t.Columns do
                sb.Append(
                    match a with
                    | Align.Left -> " --- |"
                    | Align.Right -> " ---: |"
                    | Align.Center -> " :---: |"
                )
                |> ignore

            for row in t.Rows do
                sb.Append("\n|") |> ignore
                let cells = row @ List.replicate (max 0 (t.Columns.Length - row.Length)) ""

                for c in List.truncate t.Columns.Length cells do
                    sb.Append(' ').Append(cell c).Append(" |") |> ignore

            sb.ToString()

    /// Fills parsed nodes from `scope`. Chart files are named <prefix>-<chart>.svg, where the prefix is the template
    /// name plus "-<list>-<index>" for every enclosing {{#each}}, and linked as charts/<file>.
    let render (templateName: string) (nodes: TemplateNode list) (scope: Scope) : RenderedReport =
        let sb = StringBuilder()
        let charts = ResizeArray<string * Chart>()

        let err (line: int) (msg: string) =
            raise (TemplateError(sprintf "%s line %d: %s" templateName line msg))

        let rec find (chain: Scope list) (pick: Scope -> 'a option) =
            match chain with
            | [] -> None
            | s :: outer ->
                match pick s with
                | Some v -> Some v
                | None -> find outer pick

        let rec go (chain: Scope list) (prefix: string) (nodes: TemplateNode list) =
            for node in nodes do
                match node with
                | TemplateNode.Text t -> sb.Append t |> ignore
                | TemplateNode.Value(n, line) ->
                    match find chain (fun s -> s.Values.TryFind n) with
                    | Some v -> sb.Append v |> ignore
                    | None -> err line (sprintf "no value '%s'" n)
                | TemplateNode.Table(n, line) ->
                    match find chain (fun s -> s.Tables.TryFind n) with
                    | Some t -> sb.Append(renderTable t) |> ignore
                    | None -> err line (sprintf "no table '%s'" n)
                | TemplateNode.Chart(n, line) ->
                    match find chain (fun s -> s.Charts.TryFind n) with
                    | Some c ->
                        let file = prefix + "-" + n.Replace('.', '_') + ".svg"
                        charts.Add(file, c)
                        let alt = (ChartSvg.title c).Replace("[", "").Replace("]", "")

                        sb.Append("![").Append(alt).Append("](charts/").Append(file).Append(')')
                        |> ignore
                    | None -> err line (sprintf "no chart '%s'" n)
                | TemplateNode.Each(n, body, line) ->
                    match find chain (fun s -> s.Lists.TryFind n) with
                    | Some items ->
                        items
                        |> List.iteri (fun i item ->
                            go (item :: chain) (prefix + "-" + n.Replace('.', '_') + "-" + Invariant.i32 i) body)
                    | None -> err line (sprintf "no list '%s'" n)

        go [ scope ] templateName nodes

        {
            Markdown = sb.ToString()
            Charts = List.ofSeq charts
        }
