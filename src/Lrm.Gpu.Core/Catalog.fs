namespace Lrm.Gpu.Core

open System
open System.Text
open Lrm.Core

/// Which workstream family writes a kernel's source, and so where it lives under src/Lrm.Shaders.
[<RequireQualifiedAccess>]
type KernelFamily =
    /// Backend self-tests (R4).
    | Selftest
    /// Embedding, row norms, RoPE, SiLU-mul, add, copy and argmax (K1).
    | Basic
    /// The bf16-weight matmul (K2).
    | Matmul
    /// Causal GQA attention (K3).
    | Attention

[<RequireQualifiedAccess>]
module KernelFamily =
    let all =
        [
            KernelFamily.Selftest
            KernelFamily.Basic
            KernelFamily.Matmul
            KernelFamily.Attention
        ]

    let name (f: KernelFamily) =
        match f with
        | KernelFamily.Selftest -> "selftest"
        | KernelFamily.Basic -> "basic"
        | KernelFamily.Matmul -> "matmul"
        | KernelFamily.Attention -> "attention"

    let tryParse (text: string) =
        all
        |> List.tryFind (fun f -> String.Equals(name f, text, StringComparison.Ordinal))

    /// The source folder, relative to src/Lrm.Shaders.
    let directory (f: KernelFamily) =
        match f with
        | KernelFamily.Selftest -> "hlsl/selftest"
        | KernelFamily.Basic -> "hlsl/kernels/basic"
        | KernelFamily.Matmul -> "hlsl/kernels/matmul"
        | KernelFamily.Attention -> "hlsl/kernels/attention"

/// How a kernel uses a token argument.
[<RequireQualifiedAccess>]
type Access =
    /// Reads the tensor's bytes.
    | Read
    /// Writes the tensor's bytes and never reads them.
    | Write
    /// Reads and writes the tensor's bytes.
    | ReadWrite
    /// Reads only the tensor's table row, never its bytes. A meta token may be the null token.
    | Meta

[<RequireQualifiedAccess>]
module Access =
    let all = [ Access.Read; Access.Write; Access.ReadWrite; Access.Meta ]

    let name (a: Access) =
        match a with
        | Access.Read -> "read"
        | Access.Write -> "write"
        | Access.ReadWrite -> "readwrite"
        | Access.Meta -> "meta"

    /// The frozen code used in the catalog hash.
    let code (a: Access) : byte =
        match a with
        | Access.Read -> 1uy
        | Access.Write -> 2uy
        | Access.ReadWrite -> 3uy
        | Access.Meta -> 4uy

    let tryParse (text: string) =
        all
        |> List.tryFind (fun a -> String.Equals(name a, text, StringComparison.Ordinal))

    /// True when the kernel writes the tensor's bytes.
    let writes (a: Access) =
        a = Access.Write || a = Access.ReadWrite

    /// True when the kernel reads the tensor's bytes.
    let reads (a: Access) = a = Access.Read || a = Access.ReadWrite

/// The type of one argument word.
[<RequireQualifiedAccess>]
type FieldKind =
    /// A tensor token, with how the kernel uses it, the dtypes it accepts (empty for meta) and the input fields this
    /// output may exactly alias (same slot, offset and extent).
    | Token of access: Access * dtypes: DType list * alias: string list
    | U32
    | I32
    | F32

[<RequireQualifiedAccess>]
module FieldKind =
    /// "token", "u32", "i32" or "f32".
    let typeName (k: FieldKind) =
        match k with
        | FieldKind.Token _ -> "token"
        | FieldKind.U32 -> "u32"
        | FieldKind.I32 -> "i32"
        | FieldKind.F32 -> "f32"

    /// The frozen code used in the catalog hash.
    let code (k: FieldKind) : byte =
        match k with
        | FieldKind.Token _ -> 1uy
        | FieldKind.U32 -> 2uy
        | FieldKind.I32 -> 3uy
        | FieldKind.F32 -> 4uy

/// One argument field of a kernel. Field i (from 0) is argument word i + 1.
type FieldSpec =
    {
        Name: string
        Kind: FieldKind
        Doc: string
    }

[<RequireQualifiedAccess>]
module FieldSpec =
    let token (name: string) (access: Access) (dtypes: DType list) (alias: string list) (doc: string) =
        {
            Name = name
            Kind = FieldKind.Token(access, dtypes, alias)
            Doc = doc
        }

    let u32 (name: string) (doc: string) =
        {
            Name = name
            Kind = FieldKind.U32
            Doc = doc
        }

    let i32 (name: string) (doc: string) =
        {
            Name = name
            Kind = FieldKind.I32
            Doc = doc
        }

    let f32 (name: string) (doc: string) =
        {
            Name = name
            Kind = FieldKind.F32
            Doc = doc
        }

/// A grid formula over a kernel's u32 fields: integer literals, field names, +, * and ceildiv(a, b).
[<RequireQualifiedAccess>]
type GridExpr =
    | Lit of int64
    | Ref of string
    | Add of GridExpr * GridExpr
    | Mul of GridExpr * GridExpr
    | CeilDiv of GridExpr * GridExpr

[<RequireQualifiedAccess>]
module GridExpr =
    type private Tok =
        | TNum of int64
        | TId of string
        | TSym of char

    let private tokenize (text: string) : Result<Tok list, string> =
        let out = ResizeArray<Tok>()
        let mutable i = 0
        let mutable err = None

        while err.IsNone && i < text.Length do
            let c = text[i]

            if c = ' ' then
                i <- i + 1
            elif Char.IsAsciiDigit c then
                let start = i

                while i < text.Length && Char.IsAsciiDigit text[i] do
                    i <- i + 1

                match
                    Int64.TryParse(text.AsSpan(start, i - start), Globalization.NumberStyles.None, Invariant.culture)
                with
                | true, v -> out.Add(TNum v)
                | _ -> err <- Some(sprintf "number too large at %d" start)
            elif (c >= 'a' && c <= 'z') || c = '_' then
                let start = i

                while i < text.Length
                      && ((text[i] >= 'a' && text[i] <= 'z') || Char.IsAsciiDigit text[i] || text[i] = '_') do
                    i <- i + 1

                out.Add(TId(text.Substring(start, i - start)))
            elif c = '(' || c = ')' || c = ',' || c = '+' || c = '*' then
                out.Add(TSym c)
                i <- i + 1
            else
                err <- Some(sprintf "unexpected character '%c' at %d" c i)

        match err with
        | Some e -> Error e
        | None -> Ok(List.ofSeq out)

    /// Parses a formula. Grammar: expr = term {'+' term}; term = atom {'*' atom};
    /// atom = integer | field | 'ceildiv' '(' expr ',' expr ')' | '(' expr ')'.
    let parse (text: string) : Result<GridExpr, string> =
        let rec expr toks =
            match term toks with
            | Error e -> Error e
            | Ok(lhs, rest) ->
                let rec loop lhs rest =
                    match rest with
                    | TSym '+' :: more ->
                        match term more with
                        | Ok(rhs, rest') -> loop (GridExpr.Add(lhs, rhs)) rest'
                        | Error e -> Error e
                    | _ -> Ok(lhs, rest)

                loop lhs rest

        and term toks =
            match atom toks with
            | Error e -> Error e
            | Ok(lhs, rest) ->
                let rec loop lhs rest =
                    match rest with
                    | TSym '*' :: more ->
                        match atom more with
                        | Ok(rhs, rest') -> loop (GridExpr.Mul(lhs, rhs)) rest'
                        | Error e -> Error e
                    | _ -> Ok(lhs, rest)

                loop lhs rest

        and atom toks =
            match toks with
            | TNum n :: rest -> Ok(GridExpr.Lit n, rest)
            | TId "ceildiv" :: TSym '(' :: rest ->
                match expr rest with
                | Ok(a, TSym ',' :: rest') ->
                    match expr rest' with
                    | Ok(b, TSym ')' :: rest'') -> Ok(GridExpr.CeilDiv(a, b), rest'')
                    | Ok _ -> Error "expected ')' after ceildiv arguments"
                    | Error e -> Error e
                | Ok _ -> Error "expected ',' in ceildiv"
                | Error e -> Error e
            | TId name :: rest -> Ok(GridExpr.Ref name, rest)
            | TSym '(' :: rest ->
                match expr rest with
                | Ok(e, TSym ')' :: rest') -> Ok(e, rest')
                | Ok _ -> Error "expected ')'"
                | Error e -> Error e
            | [] -> Error "unexpected end of formula"
            | t :: _ -> Error(sprintf "unexpected %A" t)

        if isNull text then
            Error "null formula"
        else
            match tokenize text with
            | Error e -> Error(sprintf "grid formula '%s': %s" text e)
            | Ok toks ->
                match expr toks with
                | Ok(e, []) -> Ok e
                | Ok(_, rest) -> Error(sprintf "grid formula '%s': trailing %A" text rest)
                | Error e -> Error(sprintf "grid formula '%s': %s" text e)

    /// The canonical text of a formula, which parse reads back to the same tree.
    let rec print (e: GridExpr) =
        match e with
        | GridExpr.Lit n -> Invariant.i64 n
        | GridExpr.Ref r -> r
        | GridExpr.Add(a, b) -> print a + " + " + printTerm b
        | GridExpr.Mul(a, b) -> printFactor a + " * " + printAtom b
        | GridExpr.CeilDiv(a, b) -> "ceildiv(" + print a + ", " + print b + ")"

    and private printTerm (e: GridExpr) =
        match e with
        | GridExpr.Add _ -> "(" + print e + ")"
        | _ -> print e

    and private printFactor (e: GridExpr) =
        match e with
        | GridExpr.Add _ -> "(" + print e + ")"
        | _ -> print e

    and private printAtom (e: GridExpr) =
        match e with
        | GridExpr.Add _
        | GridExpr.Mul _ -> "(" + print e + ")"
        | _ -> print e

    /// The field names a formula uses, in first-use order.
    let refs (e: GridExpr) =
        let rec go e =
            match e with
            | GridExpr.Lit _ -> []
            | GridExpr.Ref r -> [ r ]
            | GridExpr.Add(a, b)
            | GridExpr.Mul(a, b)
            | GridExpr.CeilDiv(a, b) -> go a @ go b

        go e |> List.distinct

    /// Every literal in a formula.
    let literals (e: GridExpr) =
        let rec go e =
            match e with
            | GridExpr.Lit n -> [ n ]
            | GridExpr.Ref _ -> []
            | GridExpr.Add(a, b)
            | GridExpr.Mul(a, b)
            | GridExpr.CeilDiv(a, b) -> go a @ go b

        go e

    /// Evaluates a formula in int64 with overflow checks; `lookup` gives a field's value.
    let eval (lookup: string -> int64) (e: GridExpr) : int64 =
        let rec go e =
            match e with
            | GridExpr.Lit n -> n
            | GridExpr.Ref r -> lookup r
            | GridExpr.Add(a, b) -> Checked.(+) (go a) (go b)
            | GridExpr.Mul(a, b) -> Checked.(*) (go a) (go b)
            | GridExpr.CeilDiv(a, b) -> Grid.ceilDiv (go a) (go b)

        try
            go e
        with :? OverflowException ->
            Fail.contract "grid formula '%s' overflows" (print e)

/// One catalog entry (KernelAbiAndCatalog contract).
type KernelSpec =
    {
        /// snake_case; also the source file name and the F# module name in PascalCase.
        Name: string
        Family: KernelFamily
        /// Path of the HLSL source, relative to src/Lrm.Shaders: <family directory>/<name>.hlsl.
        Source: string
        /// [numthreads(x, y, z)].
        Threads: int * int * int
        /// Grid formulas for x, y and z.
        Grid: GridExpr * GridExpr * GridExpr
        /// Argument fields in word order (field i is word i + 1).
        Fields: FieldSpec list
        Doc: string
    }

    /// True when the grid's y and z formulas are the literal 1, so a grid x above 65,535 may be folded.
    member k.Linear =
        let _, y, z = k.Grid
        y = GridExpr.Lit 1L && z = GridExpr.Lit 1L

    /// The index of field `name` in Fields.
    member k.TryFieldIndex(name: string) =
        k.Fields |> List.tryFindIndex (fun f -> f.Name = name)

    /// The argument word of field `name` (index + 1).
    member k.Word(name: string) =
        match k.TryFieldIndex name with
        | Some i -> i + 1
        | None -> Fail.contract "kernel %s has no field '%s'" k.Name name

/// A kernel's index in the catalog (Kernels.all).
[<Struct>]
type KernelId =
    | KernelId of int

    member x.Value =
        let (KernelId v) = x
        v

    override x.ToString() = sprintf "kernel#%d" x.Value

/// The kernel catalog: every kernel, its fields, thread counts and grids. Source of truth:
/// src/Lrm.Gpu.Core/Catalog/catalog.json; Kernels.g.fs and lrm_args.g.hlsli are generated from it.
type Catalog =
    {
        Version: int
        Kernels: KernelSpec list
    }

[<RequireQualifiedAccess>]
module KernelSpec =
    let private isIdent (s: string) =
        not (String.IsNullOrEmpty s)
        && s.Length <= 48
        && s[0] >= 'a'
        && s[0] <= 'z'
        && s
           |> Seq.forall (fun c -> (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c = '_')
        && not (s.EndsWith "_")
        && not (s.Contains "__")

    /// Words that HLSL, the preprocessor or common intrinsics use; a field must not be named after one.
    let reservedWords =
        set
            [
                "abs"
                "all"
                "any"
                "asfloat"
                "asint"
                "asuint"
                "auto"
                "bool"
                "break"
                "buffer"
                "case"
                "catch"
                "cbuffer"
                "ceil"
                "centroid"
                "char"
                "clamp"
                "class"
                "column_major"
                "const"
                "continue"
                "cos"
                "cross"
                "default"
                "delete"
                "discard"
                "do"
                "dot"
                "double"
                "dword"
                "else"
                "enum"
                "exp"
                "exp2"
                "explicit"
                "export"
                "extern"
                "false"
                "float"
                "floor"
                "fma"
                "for"
                "frac"
                "goto"
                "groupshared"
                "half"
                "if"
                "in"
                "inline"
                "inout"
                "int"
                "interface"
                "length"
                "lerp"
                "linear"
                "log"
                "log2"
                "long"
                "mad"
                "matrix"
                "max"
                "min"
                "mutable"
                "namespace"
                "new"
                "nointerpolation"
                "noperspective"
                "normalize"
                "operator"
                "out"
                "packoffset"
                "pow"
                "precise"
                "private"
                "protected"
                "public"
                "rcp"
                "register"
                "return"
                "round"
                "row_major"
                "rsqrt"
                "sample"
                "sampler"
                "saturate"
                "shared"
                "short"
                "sign"
                "signed"
                "sin"
                "sizeof"
                "snorm"
                "sqrt"
                "static"
                "step"
                "string"
                "struct"
                "switch"
                "tan"
                "tbuffer"
                "template"
                "texture"
                "this"
                "throw"
                "true"
                "trunc"
                "try"
                "typedef"
                "typename"
                "uint"
                "uniform"
                "union"
                "unorm"
                "unsigned"
                "using"
                "vector"
                "virtual"
                "void"
                "volatile"
                "while"
            ]

    /// PascalCase of a snake_case name: "src_row0" -> "SrcRow0".
    let pascal (name: string) =
        name.Split('_')
        |> Array.map (fun s ->
            if s.Length = 0 then
                s
            else
                string (Char.ToUpperInvariant s[0]) + s.Substring 1)
        |> String.concat ""

    /// Every rule a single entry must follow (KernelAbiAndCatalog contract); empty when it is valid.
    let problems (k: KernelSpec) : string list =
        let p = ResizeArray<string>()

        let add fmt =
            Printf.kprintf (fun s -> p.Add(sprintf "%s: %s" k.Name s)) fmt

        if not (isIdent k.Name) then
            add "the name must be snake_case: a-z, 0-9 and single '_', starting with a letter, at most 48 characters"

        if k.Name.StartsWith "lrm" then
            add "names starting with 'lrm' are reserved"

        let expected = KernelFamily.directory k.Family + "/" + k.Name + ".hlsl"

        if k.Source <> expected then
            add "the source must be %s" expected

        let tx, ty, tz = k.Threads

        if
            tx < 1
            || ty < 1
            || tz < 1
            || tx > 1024
            || ty > 1024
            || tz > 64
            || tx * ty * tz > 1024
        then
            add "threads %dx%dx%d break the limits (x, y <= 1024; z <= 64; x*y*z <= 1024)" tx ty tz

        if k.Fields.IsEmpty || k.Fields.Length > ArgBlock.WordCount - 1 then
            add "a kernel has 1 to 31 fields, not %d" k.Fields.Length

        if String.IsNullOrWhiteSpace k.Doc then
            add "the kernel needs a doc"

        let names = k.Fields |> List.map _.Name

        for f in k.Fields do
            if not (isIdent f.Name) then
                add "field '%s' must be snake_case" f.Name

            if f.Name.StartsWith "lrm" then
                add "field '%s': names starting with 'lrm' are reserved" f.Name

            if reservedWords.Contains f.Name then
                add "field '%s' is a reserved HLSL word" f.Name

            if String.IsNullOrWhiteSpace f.Doc then
                add "field '%s' needs a doc" f.Name

            match f.Kind with
            | FieldKind.Token(access, dtypes, alias) ->
                if access = Access.Meta then
                    if not dtypes.IsEmpty then
                        add "meta field '%s' accepts any dtype, so it lists none" f.Name

                    if not alias.IsEmpty then
                        add "meta field '%s' cannot alias" f.Name
                else
                    if dtypes.IsEmpty then
                        add "token field '%s' must list its dtypes" f.Name

                    if (List.distinct dtypes).Length <> dtypes.Length then
                        add "token field '%s' lists a dtype twice" f.Name

                if not alias.IsEmpty && not (Access.writes access) then
                    add "only written fields may alias ('%s' is %s)" f.Name (Access.name access)

                for a in alias do
                    match k.Fields |> List.tryFind (fun g -> g.Name = a) with
                    | Some { Kind = FieldKind.Token(ga, _, _) } when Access.reads ga && a <> f.Name -> ()
                    | _ -> add "field '%s' may alias only read token fields of this kernel, not '%s'" f.Name a
            | _ -> ()

        if (List.distinct names).Length <> names.Length then
            add "field names must be unique"

        if (names |> List.map pascal |> List.distinct).Length <> names.Length then
            add "field names must stay unique in PascalCase"

        if
            not (
                k.Fields
                |> List.exists (fun f ->
                    match f.Kind with
                    | FieldKind.Token(a, _, _) -> Access.writes a
                    | _ -> false)
            )
        then
            add "a kernel writes at least one token"

        let gx, gy, gz = k.Grid

        for axis, g in [ "x", gx; "y", gy; "z", gz ] do
            for r in GridExpr.refs g do
                match k.Fields |> List.tryFind (fun f -> f.Name = r) with
                | Some { Kind = FieldKind.U32 } -> ()
                | Some _ -> add "grid %s uses '%s', which is not a u32 field" axis r
                | None -> add "grid %s uses unknown field '%s'" axis r

            if GridExpr.literals g |> List.exists (fun n -> n < 1L) then
                add "grid %s literals must be at least 1" axis

        List.ofSeq p

    /// The value of u32 field `name` in an argument block.
    let fieldValue (k: KernelSpec) (args: ArgBlock) (name: string) : int64 = int64 args[k.Word name]

    /// The logical grid for an argument block, from the catalog formulas.
    let gridOf (k: KernelSpec) (args: ArgBlock) : Grid =
        let gx, gy, gz = k.Grid
        let v = fieldValue k args

        {
            X = GridExpr.eval v gx
            Y = GridExpr.eval v gy
            Z = GridExpr.eval v gz
        }

[<RequireQualifiedAccess>]
module Catalog =
    /// The domain tag of the catalog hash.
    [<Literal>]
    let Tag = "lrm.catalog/1"

    /// Every rule of the catalog: each entry is valid and kernel names are unique.
    let problems (c: Catalog) : string list =
        [
            if c.Version < 1 then
                sprintf "catalog version %d must be at least 1" c.Version
            if c.Kernels.IsEmpty then
                "the catalog has no kernels"
            let names = c.Kernels |> List.map _.Name

            if (List.distinct names).Length <> names.Length then
                "kernel names must be unique"

            if (names |> List.map KernelSpec.pascal |> List.distinct).Length <> names.Length then
                "kernel names must stay unique in PascalCase"

            yield! c.Kernels |> List.collect KernelSpec.problems
        ]

    /// The catalog when it is valid, otherwise every problem.
    let validate (c: Catalog) : Result<Catalog, string> =
        match problems c with
        | [] -> Ok c
        | ps -> Error(String.concat "\n" ps)

    /// The canonical encoding of everything that defines the ABI, docs excluded: the version, then per kernel its
    /// name, family, source, threads, grid formulas (canonical text) and fields (name, kind code, and for tokens the
    /// access code, dtype codes and aliases).
    let encode (w: CanonicalWriter) (c: Catalog) =
        w.U32(uint32 c.Version)

        w.Seq(
            c.Kernels,
            fun k ->
                w.String k.Name
                w.String(KernelFamily.name k.Family)
                w.String k.Source
                let tx, ty, tz = k.Threads
                w.U32(uint32 tx)
                w.U32(uint32 ty)
                w.U32(uint32 tz)
                let gx, gy, gz = k.Grid
                w.String(GridExpr.print gx)
                w.String(GridExpr.print gy)
                w.String(GridExpr.print gz)

                w.Seq(
                    k.Fields,
                    fun f ->
                        w.String f.Name
                        w.U8(FieldKind.code f.Kind)

                        match f.Kind with
                        | FieldKind.Token(access, dtypes, alias) ->
                            w.U8(Access.code access)
                            w.Seq(dtypes, fun d -> w.U8(DType.code d))
                            w.Seq(alias, fun a -> w.String a)
                        | _ -> ()
                )
        )

    /// Canonical.hash "lrm.catalog/1" of the encoding. Doc text does not change it.
    let hash (c: Catalog) : ContentId =
        Canonical.hash Tag (fun w -> encode w c)

    /// The entry named `name`.
    let tryFind (name: string) (c: Catalog) =
        c.Kernels |> List.tryFind (fun k -> k.Name = name)
