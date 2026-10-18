/// A small C preprocessor for the MIDL-generated Windows SDK headers: it evaluates conditionals and keeps only the
/// lines a 64-bit MSVC C (not C++) compile sees. Macros are not expanded; numeric object-like #defines are recorded
/// so array sizes and enum values can refer to them.
module Lrm.Tools.D3D12Gen.Preprocessor

open System
open System.Collections.Generic
open System.Globalization

/// One source line that survives preprocessing.
type Line =
    {
        File: string
        Number: int
        Text: string
    }

type Output =
    {
        Lines: Line list
        /// Numeric object-like macros seen on active lines, for example D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT = 8.
        Numeric: IReadOnlyDictionary<string, int64>
    }

/// The configuration the headers are read under: x64 MSVC compiling C, without CINTERFACE or COBJMACROS
/// (the C vtables are visible either way), targeting Windows 11 (NTDDI_WIN11_GE).
let private initialDefines =
    [
        "_WIN32", 1L
        "_WIN64", 1L
        "_M_X64", 100L
        "_M_AMD64", 100L
        "_MSC_VER", 1944L
        "WINVER", 0x0A00L
        "_WIN32_WINNT", 0x0A00L
        "NTDDI_VERSION", 0x0A000010L
        "NTDDI_WIN10_NI", 0x0A00000CL
    ]

type private Tok =
    | Num of int64
    | Id of string
    | Op of string

let private parseNumber (s: string) =
    let s = s.TrimEnd('u', 'U', 'l', 'L')

    if s.StartsWith("0x", StringComparison.OrdinalIgnoreCase) then
        Int64.Parse(s.Substring 2, NumberStyles.HexNumber, CultureInfo.InvariantCulture)
    elif s.Length > 1 && s.StartsWith "0" then
        Convert.ToInt64(s, 8)
    else
        Int64.Parse(s, CultureInfo.InvariantCulture)

let private tokenize (text: string) : Tok list =
    let ops =
        [
            "||"
            "&&"
            "=="
            "!="
            "<="
            ">="
            "<<"
            ">>"
            "!"
            "("
            ")"
            "<"
            ">"
            "|"
            "&"
            "+"
            "-"
            "~"
            ","
        ]

    let out = ResizeArray<Tok>()
    let mutable i = 0

    while i < text.Length do
        let c = text[i]

        if Char.IsWhiteSpace c then
            i <- i + 1
        elif Char.IsLetter c || c = '_' then
            let start = i

            while i < text.Length && (Char.IsLetterOrDigit text[i] || text[i] = '_') do
                i <- i + 1

            out.Add(Id(text.Substring(start, i - start)))
        elif Char.IsDigit c then
            let start = i

            while i < text.Length && (Char.IsLetterOrDigit text[i]) do
                i <- i + 1

            out.Add(Num(parseNumber (text.Substring(start, i - start))))
        else
            match
                ops
                |> List.tryFind (fun o -> String.CompareOrdinal(text, i, o, 0, o.Length) = 0)
            with
            | Some o ->
                out.Add(Op o)
                i <- i + o.Length
            | None -> failwithf "preprocessor: unexpected character '%c' in condition '%s'" c text

    List.ofSeq out

/// Evaluates a #if expression. Unknown identifiers are 0, as in C. The only function-like macro allowed is
/// WINAPI_FAMILY_PARTITION, which is true: the desktop family includes every partition these headers test.
let evaluate (defines: IDictionary<string, int64>) (text: string) : bool =
    let mutable toks = tokenize text

    let peek () =
        match toks with
        | t :: _ -> Some t
        | [] -> None

    let next () =
        match toks with
        | t :: rest ->
            toks <- rest
            t
        | [] -> failwithf "preprocessor: unexpected end of condition '%s'" text

    let expect o =
        match next () with
        | Op x when x = o -> ()
        | t -> failwithf "preprocessor: expected '%s' in '%s', got %A" o text t

    let rec primary () : int64 =
        match next () with
        | Num n -> n
        | Op "(" ->
            let v = orExpr ()
            expect ")"
            v
        | Op "!" -> if primary () = 0L then 1L else 0L
        | Op "-" -> -(primary ())
        | Op "~" -> ~~~(primary ())
        | Id "defined" ->
            let name =
                match next () with
                | Id n -> n
                | Op "(" ->
                    let n =
                        match next () with
                        | Id n -> n
                        | t -> failwithf "preprocessor: bad defined() in '%s': %A" text t

                    expect ")"
                    n
                | t -> failwithf "preprocessor: bad defined in '%s': %A" text t

            if defines.ContainsKey name then 1L else 0L
        | Id name when peek () = Some(Op "(") ->
            // Function-like macro call: skip the balanced argument list.
            expect "("
            let mutable depth = 1

            while depth > 0 do
                match next () with
                | Op "(" -> depth <- depth + 1
                | Op ")" -> depth <- depth - 1
                | _ -> ()

            if name = "WINAPI_FAMILY_PARTITION" then
                1L
            else
                failwithf "preprocessor: unknown function-like macro %s in '%s'" name text
        | Id name ->
            match defines.TryGetValue name with
            | true, v -> v
            | _ -> 0L
        | t -> failwithf "preprocessor: unexpected %A in '%s'" t text

    and binary (ops: (string * (int64 -> int64 -> int64)) list) (sub: unit -> int64) () =
        let mutable acc = sub ()
        let mutable go = true

        while go do
            match peek () with
            | Some(Op o) when ops |> List.exists (fun (x, _) -> x = o) ->
                next () |> ignore
                let f = ops |> List.find (fun (x, _) -> x = o) |> snd
                acc <- f acc (sub ())
            | _ -> go <- false

        acc

    and b (x: bool) = if x then 1L else 0L

    and additive () =
        binary [ "+", (+); "-", (-) ] primary ()

    and shift () =
        binary [ "<<", (fun a c -> a <<< int c); ">>", (fun a c -> a >>> int c) ] additive ()

    and relational () =
        binary
            [
                "<", (fun a c -> b (a < c))
                ">", (fun a c -> b (a > c))
                "<=", (fun a c -> b (a <= c))
                ">=", (fun a c -> b (a >= c))
            ]
            shift
            ()

    and equality () =
        binary [ "==", (fun a c -> b (a = c)); "!=", (fun a c -> b (a <> c)) ] relational ()

    and bitAnd () = binary [ "&", (&&&) ] equality ()
    and bitOr () = binary [ "|", (|||) ] bitAnd ()

    and andExpr () =
        binary [ "&&", (fun a c -> b (a <> 0L && c <> 0L)) ] bitOr ()

    and orExpr () =
        binary [ "||", (fun a c -> b (a <> 0L || c <> 0L)) ] andExpr ()

    let v = orExpr ()

    if not toks.IsEmpty then
        failwithf "preprocessor: trailing tokens in condition '%s'" text

    v <> 0L

/// Strips a trailing // or /* */ comment from a directive's argument text.
let private stripComment (s: string) =
    let s =
        match s.IndexOf "//" with
        | -1 -> s
        | i -> s.Substring(0, i)

    let rec strip (s: string) =
        match s.IndexOf "/*" with
        | -1 -> s
        | i ->
            match s.IndexOf("*/", i + 2) with
            | -1 -> s.Substring(0, i)
            | j -> strip (s.Substring(0, i) + " " + s.Substring(j + 2))

    (strip s).Trim()

type private Frame =
    {
        /// The enclosing region is active.
        Parent: bool
        /// Some branch of this #if has been taken.
        Taken: bool
        /// The current branch is active.
        Active: bool
    }

/// Runs the preprocessor over the files in order; defines made by earlier files are visible to later ones.
/// `cplusplus` reads the headers as C++ (__cplusplus defined), which exposes the MIDL_INTERFACE declarations.
let run (cplusplus: bool) (files: (string * string) list) : Output =
    let defines = Dictionary<string, int64>(StringComparer.Ordinal)

    for (k, v) in initialDefines do
        defines[k] <- v

    if cplusplus then
        defines["__cplusplus"] <- 201703L

    let numeric = Dictionary<string, int64>(StringComparer.Ordinal)
    let lines = ResizeArray<Line>()

    for (file, text) in files do
        let raw = text.Replace("\r\n", "\n").Split('\n')
        let stack = Stack<Frame>()
        let active () = stack.Count = 0 || stack.Peek().Active
        let mutable i = 0

        while i < raw.Length do
            let number = i + 1
            let mutable line = raw[i]
            i <- i + 1
            let trimmed = line.TrimStart()

            if trimmed.StartsWith "#" then
                // A directive, possibly continued with backslashes.
                while line.EndsWith "\\" && i < raw.Length do
                    line <- line.Substring(0, line.Length - 1) + " " + raw[i]
                    i <- i + 1

                let body = line.TrimStart().Substring(1).TrimStart()

                let keyword =
                    body |> Seq.takeWhile (fun c -> Char.IsLetter c) |> Seq.toArray |> String

                let arg = stripComment (body.Substring keyword.Length)

                match keyword with
                | "if" ->
                    let parent = active ()
                    let v = parent && evaluate defines arg

                    stack.Push
                        {
                            Parent = parent
                            Taken = v
                            Active = v
                        }
                | "ifdef"
                | "ifndef" ->
                    let parent = active ()
                    let name = arg.Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries).[0]
                    let has = defines.ContainsKey name
                    let v = parent && (if keyword = "ifdef" then has else not has)

                    stack.Push
                        {
                            Parent = parent
                            Taken = v
                            Active = v
                        }
                | "elif" ->
                    let f = stack.Pop()
                    let v = f.Parent && not f.Taken && evaluate defines arg

                    stack.Push
                        { f with
                            Taken = f.Taken || v
                            Active = v
                        }
                | "else" ->
                    let f = stack.Pop()

                    stack.Push
                        { f with
                            Taken = true
                            Active = f.Parent && not f.Taken
                        }
                | "endif" -> stack.Pop() |> ignore
                | "define" when active () ->
                    let name =
                        arg
                        |> Seq.takeWhile (fun c -> Char.IsLetterOrDigit c || c = '_')
                        |> Seq.toArray
                        |> String

                    let rest = arg.Substring name.Length

                    if rest.StartsWith "(" then
                        // Function-like macro: record that it exists, nothing else.
                        defines[name] <- 1L
                    else
                        let value = rest.Trim().Trim('(', ')', ' ', '\t')

                        let parsed =
                            try
                                Some(parseNumber value)
                            with _ ->
                                None

                        match parsed with
                        | Some v when value <> "" ->
                            defines[name] <- v
                            numeric[name] <- v
                        | _ -> defines[name] <- 1L
                | "undef" when active () -> defines.Remove(arg.Trim()) |> ignore
                | _ -> ()
            elif active () then
                lines.Add
                    {
                        File = file
                        Number = number
                        Text = line
                    }

        if stack.Count <> 0 then
            failwithf "preprocessor: %s ends inside %d open conditional(s)" file stack.Count

    {
        Lines = List.ofSeq lines
        Numeric = numeric
    }
