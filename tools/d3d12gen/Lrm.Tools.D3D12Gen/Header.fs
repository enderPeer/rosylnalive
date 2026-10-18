/// Parses the preprocessed SDK headers into enums, structs, typedefs, C vtables and IIDs.
///
/// Vtable slots come from counting DECLSPEC_XFGVIRT(base, method) markers: MIDL emits exactly one marker before every
/// function pointer of a C vtable, inherited methods included, so the marker's position is the slot.
module Lrm.Tools.D3D12Gen.Header

open System
open System.Collections.Generic
open System.Globalization
open Lrm.Tools.D3D12Gen.Lexer

/// A C type as written. const and volatile are dropped: they do not change the ABI.
type CType =
    | Named of string
    | Pointer of CType
    | Array of CType * int

type Field = { Name: string; Type: CType }

type Member =
    | Field of Field
    /// An anonymous union; its members are accessed as members of the enclosing struct.
    | AnonUnion of Member list
    /// An anonymous struct inside a union.
    | AnonStruct of Member list
    /// A member the generator cannot bind (a named nested aggregate or a bit field). Binding a struct that has
    /// one is an error; parsing it is not.
    | Unsupported of string

type Aggregate =
    {
        Name: string
        IsUnion: bool
        Members: Member list
        File: string
        Line: int
    }

type EnumMember =
    {
        Name: string
        Value: Result<int64, string>
    }

type EnumDef =
    {
        Name: string
        Members: EnumMember list
        File: string
        Line: int
    }

type Param =
    {
        Name: string
        Type: CType
        /// SAL annotations without arguments, for example _In_opt_ or _COM_Outptr_.
        Annotations: string list
        /// The token texts of each annotation's argument list, in the order of Annotations (empty without one):
        /// _In_reads_(NumBarriers) gives ["NumBarriers"], _In_reads_(N*sizeof(UINT)) gives ["N"; "*"; "sizeof"; ...].
        AnnotationArgs: string list list
    }

type Method =
    {
        Slot: int
        /// The interface that declares the method (the first DECLSPEC_XFGVIRT argument).
        Declarer: string
        Name: string
        Return: CType
        /// The parameters after This.
        Params: Param list
    }

type Vtable =
    {
        Interface: string
        Methods: Method list
        File: string
        Line: int
    }

type Header =
    {
        Enums: IReadOnlyDictionary<string, EnumDef>
        Aggregates: IReadOnlyDictionary<string, Aggregate>
        Typedefs: IReadOnlyDictionary<string, CType>
        Vtables: IReadOnlyDictionary<string, Vtable>
        Iids: IReadOnlyDictionary<string, Guid>
        /// Enums declared with DEFINE_ENUM_FLAG_OPERATORS.
        FlagEnums: IReadOnlySet<string>
    }

let rec describe (t: CType) =
    match t with
    | Named n -> n
    | Pointer t -> describe t + "*"
    | Array(t, n) -> sprintf "%s[%d]" (describe t) n

let private isSal (s: string) =
    s.Length > 2 && s.StartsWith "_" && s.EndsWith "_" && not (s.StartsWith "__")

let private qualifiers =
    set [ "const"; "volatile"; "struct"; "enum"; "union"; "interface" ]

type private Reader(tokens: Token[]) =
    let mutable pos = 0
    member _.Pos = pos
    member _.AtEnd = pos >= tokens.Length

    member _.Peek =
        if pos < tokens.Length then
            tokens[pos]
        else
            tokens[tokens.Length - 1]

    member _.PeekAt k =
        if pos + k < tokens.Length then
            Some tokens[pos + k]
        else
            None

    member _.Next() =
        let t = tokens[pos]
        pos <- pos + 1
        t

    member r.Is(text: string) = not r.AtEnd && r.Peek.Text = text

    member r.Expect(text: string) =
        let t = r.Next()

        if t.Text <> text then
            failwithf "%s:%d: expected '%s', got '%s'" t.File t.Line text t.Text

    member r.Ident() =
        let t = r.Next()

        if t.Kind <> Ident then
            failwithf "%s:%d: expected an identifier, got '%s'" t.File t.Line t.Text

        t.Text

    /// Skips a balanced group starting at the current '(' / '{' / '['.
    member r.SkipGroup() =
        let o = r.Next().Text

        let c =
            match o with
            | "(" -> ")"
            | "{" -> "}"
            | "[" -> "]"
            | x -> failwithf "SkipGroup at '%s'" x

        let mutable depth = 1

        while depth > 0 do
            let t = r.Next().Text

            if t = o then
                depth <- depth + 1
            elif t = c then
                depth <- depth - 1

    member r.Where = sprintf "%s:%d" r.Peek.File r.Peek.Line

/// Evaluates a constant expression (enum values, array sizes) over the tokens up to a terminator at depth 0.
let private evalTokens (lookup: string -> int64 option) (toks: Token list) : Result<int64, string> =
    let mutable rest = toks

    let peek () =
        match rest with
        | t :: _ -> Some t.Text
        | [] -> None

    let next () =
        match rest with
        | t :: r ->
            rest <- r
            t
        | [] -> failwith "unexpected end of expression"

    let num (s: string) =
        let s = s.TrimEnd('u', 'U', 'l', 'L')

        if s.StartsWith("0x", StringComparison.OrdinalIgnoreCase) then
            Int64.Parse(s.Substring 2, NumberStyles.HexNumber, CultureInfo.InvariantCulture)
        else
            Int64.Parse(s, CultureInfo.InvariantCulture)

    let rec primary () =
        let t = next ()

        match t.Kind, t.Text with
        | Number, s -> num s
        | Ident, name ->
            match lookup name with
            | Some v -> v
            | None -> failwithf "unknown identifier %s" name
        | Punct, "(" ->
            let v = orExpr ()

            if (next ()).Text <> ")" then
                failwith "expected ')'"

            v
        | Punct, "-" -> -(primary ())
        | Punct, "~" -> ~~~(primary ())
        | Punct, "+" -> primary ()
        | _, s -> failwithf "unexpected '%s'" s

    and level (ops: (string * (int64 -> int64 -> int64)) list) (sub: unit -> int64) () =
        let mutable acc = sub ()
        let mutable go = true

        while go do
            match peek () with
            | Some o when ops |> List.exists (fun (x, _) -> x = o) ->
                next () |> ignore
                acc <- (ops |> List.find (fun (x, _) -> x = o) |> snd) acc (sub ())
            | _ -> go <- false

        acc

    and mul () = level [ "*", (*); "/", (/) ] primary ()
    and add () = level [ "+", (+); "-", (-) ] mul ()

    and shift () =
        level [ "<<", (fun a b -> a <<< int b); ">>", (fun a b -> a >>> int b) ] add ()

    and band () = level [ "&", (&&&) ] shift ()
    and bxor () = level [ "^", (^^^) ] band ()
    and orExpr () = level [ "|", (|||) ] bxor ()

    try
        let v = orExpr ()

        if not rest.IsEmpty then
            Error(sprintf "trailing tokens in '%s'" (toks |> List.map _.Text |> String.concat " "))
        else
            Ok v
    with e ->
        Error(sprintf "%s in '%s'" e.Message (toks |> List.map _.Text |> String.concat " "))

/// Collects tokens up to (not including) one of the terminators at bracket depth 0.
let private takeUntil (r: Reader) (terminators: string list) =
    let acc = ResizeArray<Token>()
    let mutable depth = 0

    while not (depth = 0 && terminators |> List.contains r.Peek.Text) do
        let t = r.Next()

        match t.Text with
        | "("
        | "["
        | "{" -> depth <- depth + 1
        | ")"
        | "]"
        | "}" -> depth <- depth - 1
        | _ -> ()

        acc.Add t

    List.ofSeq acc

/// Reads SAL annotations and returns each name with the token texts of its argument list (empty without one).
let private annotations (r: Reader) =
    let acc = ResizeArray<string * string list>()

    while not r.AtEnd && r.Peek.Kind = Ident && isSal r.Peek.Text do
        let name = r.Next().Text

        if r.Is "(" then
            r.Next() |> ignore
            let args = takeUntil r [ ")" ] |> List.map _.Text
            r.Expect ")"
            acc.Add(name, args)
        else
            acc.Add(name, [])

    List.ofSeq acc

/// Reads the type specifier (qualifiers and one type name) and any pointer stars after it.
let private specifier (r: Reader) =
    let mutable name = None

    while name.IsNone do
        let t = r.Ident()

        if qualifiers.Contains t then
            ()
        elif t = "unsigned" || t = "signed" then
            // MIDL output uses typedef names; plain 'unsigned int' style spellings map to the typedefs.
            let baseName =
                if
                    r.Peek.Kind = Ident
                    && List.contains r.Peek.Text [ "int"; "char"; "short"; "long"; "__int64" ]
                then
                    r.Next().Text
                else
                    "int"

            name <- Some(t + " " + baseName)
        else
            name <- Some t

    let mutable ty = Named name.Value

    let rec stars () =
        if r.Is "*" then
            r.Next() |> ignore
            ty <- Pointer ty

            while r.Is "const" || r.Is "volatile" do
                r.Next() |> ignore

            stars ()
        elif r.Is "const" || r.Is "volatile" then
            r.Next() |> ignore
            stars ()

    stars ()
    ty

let private arrayDims (r: Reader) (lookup: string -> int64 option) (ty: CType) =
    let mutable t = ty
    let dims = ResizeArray<int>()

    while r.Is "[" do
        let where = r.Where
        r.Next() |> ignore
        let toks = takeUntil r [ "]" ]
        r.Expect "]"

        match evalTokens lookup toks with
        | Ok n -> dims.Add(int n)
        | Error e -> failwithf "%s: array size: %s" where e

    // int a[2][3] is an array of 2 arrays of 3.
    for d in Seq.rev dims do
        t <- Array(t, d)

    t

let rec private members (r: Reader) (lookup: string -> int64 option) : Member list =
    let acc = ResizeArray<Member>()

    while not (r.Is "}") do
        annotations r |> ignore

        if
            (r.Is "union" || r.Is "struct")
            && (r.PeekAt 1 |> Option.exists (fun t -> t.Text = "{"))
        then
            let isUnion = r.Next().Text = "union"
            r.Expect "{"
            let inner = members r lookup
            r.Expect "}"

            if r.Is ";" then
                r.Next() |> ignore
                acc.Add(if isUnion then AnonUnion inner else AnonStruct inner)
            else
                let where = r.Where
                let name = r.Ident()
                r.Expect ";"
                acc.Add(Unsupported(sprintf "%s: named nested aggregate %s" where name))
        else
            let ty = specifier r
            let name = r.Ident()

            if r.Is ":" then
                let where = r.Where
                takeUntil r [ ";" ] |> ignore
                r.Expect ";"
                acc.Add(Unsupported(sprintf "%s: bit field %s" where name))
            else
                let ty = arrayDims r lookup ty
                r.Expect ";"
                acc.Add(Field { Name = name; Type = ty })

    List.ofSeq acc

let private parseParams (r: Reader) (lookup: string -> int64 option) : Param list =
    r.Expect "("
    let acc = ResizeArray<Param>()

    if r.Is "void" && (r.PeekAt 1 |> Option.exists (fun t -> t.Text = ")")) then
        r.Next() |> ignore
    else
        let mutable go = not (r.Is ")")

        while go do
            let ann = annotations r
            let ty = specifier r
            let name = r.Ident()
            let ty = arrayDims r lookup ty

            acc.Add
                {
                    Name = name
                    Type = ty
                    Annotations = ann |> List.map fst
                    AnnotationArgs = ann |> List.map snd
                }

            if r.Is "," then r.Next() |> ignore else go <- false

    r.Expect ")"
    List.ofSeq acc

let private parseVtable (r: Reader) (lookup: string -> int64 option) (iface: string) file line =
    r.Expect "BEGIN_INTERFACE"
    let methods = ResizeArray<Method>()

    while not (r.Is "END_INTERFACE") do
        let where = r.Where
        r.Expect "DECLSPEC_XFGVIRT"
        r.Expect "("
        let declarer = r.Ident()
        r.Expect ","
        let markerName = r.Ident()
        r.Expect ")"
        let ret = specifier r
        r.Expect "("
        r.Expect "STDMETHODCALLTYPE"
        r.Expect "*"
        let name = r.Ident()
        r.Expect ")"

        if name <> markerName then
            failwithf "%s: DECLSPEC_XFGVIRT names %s but the function pointer is %s" where markerName name

        let ps = parseParams r lookup
        r.Expect ";"

        match ps with
        | this :: rest when this.Name = "This" ->
            methods.Add
                {
                    Slot = methods.Count
                    Declarer = declarer
                    Name = name
                    Return = ret
                    Params = rest
                }
        | _ -> failwithf "%s: %s::%s has no This parameter" where iface name

    r.Expect "END_INTERFACE"

    {
        Interface = iface
        Methods = List.ofSeq methods
        File = file
        Line = line
    }

let private parseGuid (r: Reader) =
    // DEFINE_GUID(IID_X, l, w1, w2, b1, ..., b8);
    r.Expect "("
    let name = r.Ident()
    let parts = ResizeArray<uint64>()

    while r.Is "," do
        r.Next() |> ignore
        let t = r.Next().Text.TrimEnd('u', 'U', 'l', 'L')

        let v =
            if t.StartsWith("0x", StringComparison.OrdinalIgnoreCase) then
                UInt64.Parse(t.Substring 2, NumberStyles.HexNumber, CultureInfo.InvariantCulture)
            else
                UInt64.Parse(t, CultureInfo.InvariantCulture)

        parts.Add v

    r.Expect ")"

    if parts.Count <> 11 then
        failwithf "%s: DEFINE_GUID(%s) has %d parts" r.Where name parts.Count

    let p = parts.ToArray()

    name,
    Guid(
        uint32 p[0],
        uint16 p[1],
        uint16 p[2],
        byte p[3],
        byte p[4],
        byte p[5],
        byte p[6],
        byte p[7],
        byte p[8],
        byte p[9],
        byte p[10]
    )

/// Parses the token stream of all headers (in include order).
let parse (tokens: Token[]) (numeric: IReadOnlyDictionary<string, int64>) : Header =
    let r = Reader tokens
    let enums = Dictionary<string, EnumDef>(StringComparer.Ordinal)
    let aggregates = Dictionary<string, Aggregate>(StringComparer.Ordinal)
    let typedefs = Dictionary<string, CType>(StringComparer.Ordinal)
    let vtables = Dictionary<string, Vtable>(StringComparer.Ordinal)
    let iids = Dictionary<string, Guid>(StringComparer.Ordinal)
    let flags = HashSet<string>(StringComparer.Ordinal)
    let enumValues = Dictionary<string, int64>(StringComparer.Ordinal)

    let lookup name =
        match enumValues.TryGetValue name with
        | true, v -> Some v
        | _ ->
            match numeric.TryGetValue name with
            | true, v -> Some v
            | _ -> None

    let addUnique (d: Dictionary<string, 'T>) (name: string) (v: 'T) where =
        if d.ContainsKey name then
            failwithf "%s: %s is defined twice" where name

        d[name] <- v

    while not r.AtEnd do
        let t = r.Peek

        match t.Text with
        | "typedef" ->
            r.Next() |> ignore

            if r.Is "enum" then
                r.Next() |> ignore
                let file, line = r.Peek.File, r.Peek.Line

                if r.Peek.Kind = Ident then
                    r.Next() |> ignore

                r.Expect "{"
                let ms = ResizeArray<EnumMember>()
                let mutable previous = -1L

                while not (r.Is "}") do
                    let name = r.Ident()

                    let value =
                        if r.Is "=" then
                            r.Next() |> ignore
                            evalTokens lookup (takeUntil r [ ","; "}" ])
                        else
                            Ok(previous + 1L)

                    match value with
                    | Ok v ->
                        previous <- v
                        enumValues[name] <- v
                    | Error _ -> ()

                    ms.Add { Name = name; Value = value }

                    if r.Is "," then
                        r.Next() |> ignore

                r.Expect "}"
                let name = r.Ident()
                r.Expect ";"

                addUnique
                    enums
                    name
                    {
                        Name = name
                        Members = List.ofSeq ms
                        File = file
                        Line = line
                    }
                    (sprintf "%s:%d" file line)
            elif
                (r.Is "struct" || r.Is "union")
                && (r.PeekAt 1 |> Option.exists (fun t -> t.Text = "{")
                    || r.PeekAt 2 |> Option.exists (fun t -> t.Text = "{"))
            then
                let isUnion = r.Next().Text = "union"
                let file, line = r.Peek.File, r.Peek.Line
                let tag = if r.Peek.Kind = Ident then Some(r.Next().Text) else None
                r.Expect "{"

                if r.Is "BEGIN_INTERFACE" then
                    let tag =
                        match tag with
                        | Some t when t.EndsWith "Vtbl" -> t
                        | _ -> failwithf "%s:%d: vtable without a XVtbl tag" file line

                    let iface = tag.Substring(0, tag.Length - 4)
                    let vt = parseVtable r lookup iface file line
                    r.Expect "}"
                    r.Ident() |> ignore
                    r.Expect ";"
                    addUnique vtables iface vt (sprintf "%s:%d" file line)
                else
                    let ms = members r lookup
                    r.Expect "}"
                    let name = r.Ident()

                    let agg =
                        {
                            Name = name
                            IsUnion = isUnion
                            Members = ms
                            File = file
                            Line = line
                        }

                    // Extra declarators such as "*PNAME" are ignored.
                    takeUntil r [ ";" ] |> ignore
                    r.Expect ";"
                    addUnique aggregates name agg (sprintf "%s:%d" file line)
            else
                // typedef <type> <name>; or something we do not model (function pointer typedefs).
                let toks = takeUntil r [ ";" ]
                r.Expect ";"

                match toks |> List.filter (fun t -> not (qualifiers.Contains t.Text)) with
                | [ a; b ] when a.Kind = Ident && b.Kind = Ident -> typedefs[b.Text] <- Named a.Text
                | [ a; star; b ] when a.Kind = Ident && star.Text = "*" && b.Kind = Ident ->
                    typedefs[b.Text] <- Pointer(Named a.Text)
                | _ -> ()
        | "DEFINE_GUID" ->
            r.Next() |> ignore
            let name, g = parseGuid r
            iids[name] <- g
        | "DEFINE_ENUM_FLAG_OPERATORS" ->
            r.Next() |> ignore
            r.Expect "("
            flags.Add(r.Ident()) |> ignore
            r.Expect ")"
        | "{" -> r.SkipGroup()
        | _ -> r.Next() |> ignore

    {
        Enums = enums
        Aggregates = aggregates
        Typedefs = typedefs
        Vtables = vtables
        Iids = iids
        FlagEnums = flags
    }
