/// Decides what is bound and how: the closure of the roots in bindings.json, the C layout of every struct (x64), the
/// F# type of every member and parameter, the values of every bound enum, and the shape of every generated wrapper.
/// Everything is sorted by name so that the output is a pure function of bindings.json and vk.xml.
module Lrm.Tools.VkGen.Bind

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Lrm.Tools.VkGen.Registry

/// tools/vkgen/bindings.json.
type Bindings =
    {
        CoreVersion: int * int
        Extensions: string list
        Structs: string list
        Enums: string list
        FuncPointers: string list
        GlobalCommands: string list
        InstanceCommands: string list
        DeviceCommands: string list
        OutputDirectory: string
        Namespace: string
        ToolchainEntry: string
    }

let loadBindings (path: string) : Bindings =
    use doc = JsonDocument.Parse(File.ReadAllText path)
    let root = doc.RootElement

    let strings (e: JsonElement) =
        [ for x in e.EnumerateArray() -> x.GetString() ]

    let v = root.GetProperty("coreVersion").GetString().Split('.')
    let commands = root.GetProperty "commands"

    {
        CoreVersion = Int32.Parse v[0], Int32.Parse v[1]
        Extensions = strings (root.GetProperty "extensions")
        Structs = strings (root.GetProperty "structs")
        Enums = strings (root.GetProperty "enums")
        FuncPointers = strings (root.GetProperty "funcpointers")
        GlobalCommands = strings (commands.GetProperty "global")
        InstanceCommands = strings (commands.GetProperty "instance")
        DeviceCommands = strings (commands.GetProperty "device")
        OutputDirectory = root.GetProperty("output").GetProperty("directory").GetString()
        Namespace = root.GetProperty("output").GetProperty("namespace").GetString()
        ToolchainEntry = root.GetProperty("registry").GetProperty("toolchainEntry").GetString()
    }

// ---- C scalars -----------------------------------------------------------------------------------------------------

/// (F# type, size, ABI kind) of a C scalar: 'i' integer, 'f' floating point.
let scalars: IReadOnlyDictionary<string, string * int * char> =
    readOnlyDict
        [
            "char", ("byte", 1, 'i')
            "int8_t", ("sbyte", 1, 'i')
            "uint8_t", ("byte", 1, 'i')
            "int16_t", ("int16", 2, 'i')
            "uint16_t", ("uint16", 2, 'i')
            "int32_t", ("int32", 4, 'i')
            "int", ("int32", 4, 'i')
            "uint32_t", ("uint32", 4, 'i')
            "int64_t", ("int64", 8, 'i')
            "uint64_t", ("uint64", 8, 'i')
            "float", ("float32", 4, 'f')
            "double", ("float", 8, 'f')
            "size_t", ("unativeint", 8, 'i')
        ]

/// F# keywords and reserved identifiers that vk.xml uses as member or parameter names.
let private keywords =
    set
        [
            "abstract"
            "and"
            "as"
            "assert"
            "base"
            "begin"
            "class"
            "default"
            "delegate"
            "do"
            "done"
            "downcast"
            "downto"
            "elif"
            "else"
            "end"
            "exception"
            "extern"
            "false"
            "finally"
            "fixed"
            "for"
            "fun"
            "function"
            "global"
            "if"
            "in"
            "inherit"
            "inline"
            "interface"
            "internal"
            "lazy"
            "let"
            "match"
            "member"
            "module"
            "mutable"
            "namespace"
            "new"
            "not"
            "null"
            "of"
            "open"
            "or"
            "override"
            "private"
            "public"
            "rec"
            "return"
            "select"
            "sig"
            "static"
            "struct"
            "then"
            "to"
            "true"
            "try"
            "type"
            "upcast"
            "use"
            "val"
            "void"
            "when"
            "while"
            "with"
            "yield"
            "const"
            "break"
            "checked"
            "component"
            "constraint"
            "continue"
            "event"
            "external"
            "include"
            "mixin"
            "parallel"
            "process"
            "protected"
            "pure"
            "sealed"
            "tailcall"
            "trait"
            "virtual"
            "object"
            "atomic"
            "constructor"
            "eager"
            "functor"
            "measure"
            "method"
            "volatile"
            "params"
            "base"
        ]

/// An F# identifier for a C name: keywords are escaped with double backticks.
let ident (name: string) =
    if keywords.Contains name then "``" + name + "``" else name

// ---- the bound model -----------------------------------------------------------------------------------------------

type FieldKind =
    /// A scalar, basetype, enum or bitmask: 'i' or 'f'.
    | KScalar of kind: char
    | KPointer
    | KHandle
    | KStruct
    | KArray of elementKind: char * elementSize: int * count: int

type BField =
    {
        Decl: CDecl
        FsName: string
        FsType: string
        Kind: FieldKind
        Offset: int
        Size: int
    }

type BStruct =
    {
        Def: StructDef
        Fields: BField list
        Size: int
        Align: int
        /// The VkStructureType member named by the sType member's `values`.
        SType: (string * int64) option
    }

type BArray =
    {
        Name: string
        ElementC: string
        ElementFs: string
        ElementKind: char
        ElementSize: int
        Count: int
        Size: int
        Align: int
    }

type BEnum =
    {
        Name: string
        IsBitmask: bool
        Bits64: bool
        Members: (string * int64) list
    }

type Provider =
    | Core of int * int
    | Ext of name: string * instance: bool

type Level =
    | Global
    | Instance
    | Device

/// How a wrapper passes one C parameter.
type WParam =
    /// Passed through by value.
    | ByValue of name: string * fsType: string
    /// const T*, one element: inref<T>, pinned for the call.
    | InRef of name: string * fsType: string
    /// T*, one element (out or in/out): byref<T>, pinned for the call.
    | OutRef of name: string * fsType: string
    /// const T* sized by a by-value count: ReadOnlySpan<T>; the count is the span's length.
    | SpanIn of name: string * elementFs: string * count: string
    /// T* sized by a by-value count: Span<T>.
    | SpanOut of name: string * elementFs: string * count: string
    /// Any other pointer: nativeint, as in C.
    | Raw of name: string
    /// A count every one of whose arrays is a span: dropped from the wrapper.
    | Count of name: string * fsType: string * arrays: string list
    /// pAllocator: dropped from the wrapper, NULL is passed.
    | Allocator of name: string

type BCommand =
    {
        Def: Command
        Level: Level
        Providers: Provider list
        /// F# types of the delegate's parameters, in C order.
        DelegateParams: (string * string) list
        DelegateReturn: string
        Wrapper: WParam list
        /// The VkResult members that are success codes (empty when the command does not return VkResult).
        SuccessCodes: string list
    }

type BFuncPointer =
    {
        Name: string
        Return: string
        Params: (string * string) list
        Text: string
    }

type Model =
    {
        Registry: Registry
        Bindings: Bindings
        Basetypes: (string * string) list
        Handles: (string * bool) list
        Enums: BEnum list
        /// Bitmask typedefs and the F# type each abbreviates.
        Bitmasks: (string * string) list
        /// Type aliases (KHR/EXT names) and the F# type each abbreviates.
        Aliases: (string * string) list
        Arrays: BArray list
        /// In dependency order: a struct follows every struct and array it holds by value.
        Structs: BStruct list
        FuncPointers: BFuncPointer list
        Commands: BCommand list
        Constants: (ApiConstant * string) list
        /// Extension constants of the bound extensions: (name, value text, F# literal).
        ExtensionConstants: (string * string) list
        BoundExtensions: Extension list
    }

// ---- dependency expressions ----------------------------------------------------------------------------------------

/// Evaluates a vk.xml `depends` expression: names joined by ',' (or) and '+' (and), with parentheses.
let evalDepends (satisfied: string -> bool) (expr: string) : bool =
    let tokens =
        [
            let mutable i = 0

            while i < expr.Length do
                let c = expr[i]

                if c = '(' || c = ')' || c = ',' || c = '+' then
                    yield string c
                    i <- i + 1
                elif Char.IsWhiteSpace c then
                    i <- i + 1
                else
                    let start = i

                    while i < expr.Length
                          && not ("(),+".Contains expr[i])
                          && not (Char.IsWhiteSpace expr[i]) do
                        i <- i + 1

                    yield expr.Substring(start, i - start)
        ]

    let rec orExpr (ts: string list) =
        let v, rest = andExpr ts

        match rest with
        | "," :: rest ->
            let w, rest = orExpr rest
            (v || w), rest
        | _ -> v, rest

    and andExpr ts =
        let v, rest = atom ts

        match rest with
        | "+" :: rest ->
            let w, rest = andExpr rest
            (v && w), rest
        | _ -> v, rest

    and atom ts =
        match ts with
        | "(" :: rest ->
            let v, rest = orExpr rest

            match rest with
            | ")" :: rest -> v, rest
            | _ -> failwithf "unbalanced depends expression '%s'" expr
        | name :: rest ->
            let name =
                match name.IndexOf "::" with
                | -1 -> name
                | i -> name.Substring(0, i)

            satisfied name, rest
        | [] -> failwithf "empty depends expression '%s'" expr

    let v, rest = orExpr tokens

    if not rest.IsEmpty then
        failwithf "trailing tokens in depends expression '%s'" expr

    v

// ---- building the model --------------------------------------------------------------------------------------------

let private fail fmt = Printf.kprintf failwith fmt

let build (reg: Registry) (b: Bindings) : Model =
    let boundFeatures =
        reg.Features |> List.filter (fun f -> f.Version <= b.CoreVersion)

    let boundExtensions =
        b.Extensions
        |> List.map (fun n ->
            match reg.Extensions.TryGetValue n with
            | true, x -> x
            | _ -> fail "bindings.json: extension %s is not a vulkan extension in vk.xml" n)

    let boundExtNames = set b.Extensions

    let satisfied (name: string) =
        if name.StartsWith "VK_" && name.Contains "_VERSION_" then
            reg.Features
            |> List.exists (fun f -> f.Name = name && f.Version <= b.CoreVersion)
        else
            boundExtNames.Contains name

    let activeRequires (rs: Require list) =
        rs
        |> List.filter (fun r ->
            match r.Depends with
            | None -> true
            | Some d -> evalDepends satisfied d)

    let boundRequires =
        [
            for f in boundFeatures do
                for r in activeRequires f.Requires do
                    yield Core f.Version, r
            for x in boundExtensions do
                for r in activeRequires x.Requires do
                    yield Ext(x.Name, (x.Kind = "instance")), r
        ]

    // Every enum value vk.xml defines, anywhere, by name: aliases resolve through it.
    let globalValues = Dictionary<string, EnumValue>(StringComparer.Ordinal)

    for e in reg.Enums.Values do
        for m in e.Members do
            globalValues[m.Name] <- m.Value

    let allRequires =
        [
            for f in reg.Features do
                yield! f.Requires
            for x in reg.Extensions.Values do
                yield! x.Requires
        ]

    for r in allRequires do
        for e in r.Enums do
            match e.Extends, e.Value with
            | Some _, Some v when not (globalValues.ContainsKey e.Name) -> globalValues[e.Name] <- v
            | _ -> ()

    let rec resolveValue (seen: string list) (name: string) : int64 =
        match globalValues.TryGetValue name with
        | true, Num v -> v
        | true, AliasOf a when not (List.contains a seen) -> resolveValue (name :: seen) a
        | true, AliasOf a -> fail "enum alias cycle at %s -> %s" name a
        | _ -> fail "enum value %s is not defined in vk.xml" name

    let typeDef (name: string) =
        match reg.Types.TryGetValue name with
        | true, t -> t
        | _ -> fail "type %s is not in vk.xml" name

    // ---- closure ----
    let basetypes = SortedDictionary<string, string>(StringComparer.Ordinal)
    let handles = SortedDictionary<string, bool>(StringComparer.Ordinal)
    let enumNames = SortedSet<string>(StringComparer.Ordinal)

    let bitmasks =
        SortedDictionary<string, (string option * bool)>(StringComparer.Ordinal)

    let aliases = SortedDictionary<string, string>(StringComparer.Ordinal)
    let structs = SortedDictionary<string, StructDef>(StringComparer.Ordinal)
    let funcPointers = SortedSet<string>(StringComparer.Ordinal)
    let constantNames = SortedSet<string>(StringComparer.Ordinal)

    let rec visit (name: string) =
        if not (scalars.ContainsKey name) && name <> "void" then
            match typeDef name with
            | TypeAlias(n, target) ->
                if not (aliases.ContainsKey n) then
                    aliases[n] <- target
                    visit target
            | Basetype(n, u) ->
                if not (scalars.ContainsKey u) then
                    fail "basetype %s has a non-scalar underlying type %s" n u

                basetypes[n] <- u
            | Bitmask(n, bits, b64) ->
                if not (bitmasks.ContainsKey n) then
                    bitmasks[n] <- (bits, b64)
                    basetypes[(if b64 then "VkFlags64" else "VkFlags")] <- (if b64 then "uint64_t" else "uint32_t")

                    match bits with
                    | Some fb when reg.Types.ContainsKey fb -> visit fb
                    | _ -> ()
            | Handle(n, d) -> handles[n] <- d
            | EnumType n -> enumNames.Add n |> ignore
            | Struct s ->
                if not (structs.ContainsKey s.Name) then
                    structs[s.Name] <- s

                    for m in s.Members do
                        if m.BitField.IsSome then
                            fail "%s.%s is a bit field, which the generator does not bind" s.Name m.Name

                        for d in m.Dims do
                            match d with
                            | Constant c -> constantNames.Add c |> ignore
                            | Literal _ -> ()

                        if m.Pointers = 0 then
                            match reg.Types.TryGetValue m.TypeName with
                            | true, FuncPointer _ -> () // a PFN member is a pointer
                            | _ -> visit m.TypeName
            | FuncPointer(n, ret, ps) ->
                funcPointers.Add n |> ignore

                for p in ret :: ps do
                    if p.Pointers = 0 then
                        visit p.TypeName
            | Other(n, c) -> fail "type %s (%s) cannot be bound" n c

    for s in b.Structs do
        match typeDef s with
        | Struct _
        | TypeAlias _ -> visit s
        | _ -> fail "bindings.json: %s is not a struct" s

    for e in b.Enums do
        match typeDef e with
        | EnumType _
        | TypeAlias _ -> visit e
        | _ -> fail "bindings.json: %s is not an enum" e

    for f in b.FuncPointers do
        match typeDef f with
        | FuncPointer _ -> visit f
        | _ -> fail "bindings.json: %s is not a funcpointer" f

    let listed =
        [
            for n in b.GlobalCommands -> n, Global
            for n in b.InstanceCommands -> n, Instance
            for n in b.DeviceCommands -> n, Device
        ]

    let duplicates =
        listed |> List.countBy fst |> List.filter (fun (_, c) -> c > 1) |> List.map fst

    if not duplicates.IsEmpty then
        fail "bindings.json lists commands twice: %s" (String.concat ", " duplicates)

    let commandDef (name: string) =
        match reg.Commands.TryGetValue name with
        | true, c -> c
        | _ ->
            match reg.CommandAliases.TryGetValue name with
            | true, target -> fail "bindings.json: %s is an alias of %s; bind the target" name target
            | _ -> fail "bindings.json: command %s is not in vk.xml" name

    for (n, _) in listed do
        let c = commandDef n

        for p in c.Return :: c.Params do
            match reg.Types.TryGetValue p.TypeName with
            | true, FuncPointer _ -> ()
            | _ when p.TypeName = "VkAllocationCallbacks" -> ()
            | _ when scalars.ContainsKey p.TypeName || p.TypeName = "void" -> ()
            | _ -> visit p.TypeName

    // ---- enums: values from their <enums> blocks and from the bound requires ----
    let enumAdditions = Dictionary<string, ResizeArray<string>>(StringComparer.Ordinal)

    for (_, r) in boundRequires do
        for e in r.Enums do
            match e.Extends with
            | Some target ->
                match enumAdditions.TryGetValue target with
                | true, l -> l.Add e.Name
                | _ -> enumAdditions[target] <- ResizeArray [ e.Name ]
            | None -> ()

    let boundEnums =
        [
            for n in enumNames ->
                let block =
                    match reg.Enums.TryGetValue n with
                    | true, e -> Some e
                    | _ -> None

                let fromBlock =
                    match block with
                    | Some e -> e.Members |> List.map _.Name
                    | None -> []

                let added =
                    match enumAdditions.TryGetValue n with
                    | true, l ->
                        l
                        |> Seq.distinct
                        |> Seq.filter (fun x -> not (List.contains x fromBlock))
                        |> Seq.map (fun x -> x, resolveValue [] x)
                        |> Seq.sortBy (fun (x, v) -> v, x)
                        |> List.ofSeq
                    | _ -> []

                let members = (fromBlock |> List.map (fun x -> x, resolveValue [] x)) @ added
                let isBitmask = block |> Option.exists _.IsBitmask
                let bits64 = block |> Option.exists _.Bits64

                {
                    Name = n
                    IsBitmask = isBitmask
                    Bits64 = bits64
                    Members = members
                }
        ]

    let enumByName = boundEnums |> List.map (fun e -> e.Name, e) |> dict

    // Bitmask typedefs abbreviate their FlagBits enum when it has members, else VkFlags or VkFlags64.
    let bitmaskTargets =
        [
            for KeyValue(n, (bits, b64)) in bitmasks ->
                let fallback = if b64 then "VkFlags64" else "VkFlags"

                match bits with
                | Some fb ->
                    let target =
                        match aliases.TryGetValue fb with
                        | true, t -> t
                        | _ -> fb

                    match enumByName.TryGetValue target with
                    | true, e when not e.Members.IsEmpty -> n, target
                    | _ -> n, fallback
                | None -> n, fallback
        ]

    let emptyEnums =
        boundEnums |> List.filter (fun e -> e.Members.IsEmpty) |> List.map _.Name |> set

    let enumsWithMembers = boundEnums |> List.filter (fun e -> not e.Members.IsEmpty)

    // ---- layouts ----
    let layoutCache = Dictionary<string, int * int * int list>(StringComparer.Ordinal)

    let rec resolveAlias (n: string) =
        match aliases.TryGetValue n with
        | true, t -> resolveAlias t
        | _ -> n

    let rec typeLayout (name: string) : int * int =
        match scalars.TryGetValue name with
        | true, (_, size, _) -> size, size
        | _ ->
            match typeDef name with
            | TypeAlias(_, t) -> typeLayout t
            | Basetype(_, u) -> typeLayout u
            | Bitmask(_, _, b64) -> (if b64 then 8, 8 else 4, 4)
            | Handle _ -> 8, 8
            | EnumType n ->
                match enumByName.TryGetValue n with
                | true, e when e.Bits64 -> 8, 8
                | _ -> 4, 4
            | FuncPointer _ -> 8, 8
            | Struct s ->
                let size, align, _ = structLayout s
                size, align
            | Other(n, _) -> fail "no layout for %s" n

    and declLayout (d: CDecl) : int * int =
        if d.Pointers > 0 then
            8, 8
        else
            let size, align = typeLayout d.TypeName

            let count =
                d.Dims
                |> List.fold
                    (fun acc dim ->
                        match dim with
                        | Literal n -> acc * n
                        | Constant c ->
                            match reg.Constants.TryGetValue c with
                            | true, k -> acc * int (parseNumber k.Value)
                            | _ -> fail "array size %s is not an API constant" c)
                    1

            size * count, align

    /// The C layout of a struct or union on x64: (size, alignment, member offsets).
    and structLayout (s: StructDef) : int * int * int list =
        match layoutCache.TryGetValue s.Name with
        | true, l -> l
        | _ ->
            let parts = s.Members |> List.map declLayout

            let l =
                if s.IsUnion then
                    let align = parts |> List.map snd |> List.max
                    let size = parts |> List.map fst |> List.max
                    (size + align - 1) / align * align, align, parts |> List.map (fun _ -> 0)
                else
                    let mutable off = 0
                    let mutable align = 1

                    let offs =
                        [
                            for (sz, al) in parts do
                                off <- (off + al - 1) / al * al
                                yield off
                                off <- off + sz
                                align <- max align al
                        ]

                    (off + align - 1) / align * align, align, offs

            layoutCache[s.Name] <- l
            l

    // ---- F# types ----
    let arrays = SortedDictionary<string, BArray>(StringComparer.Ordinal)

    let rec scalarKind (name: string) : char =
        match scalars.TryGetValue name with
        | true, (_, _, k) -> k
        | _ ->
            match typeDef name with
            | TypeAlias(_, t) -> scalarKind t
            | Basetype(_, u) -> scalarKind u
            | Bitmask _
            | EnumType _ -> 'i'
            | _ -> fail "%s is not a scalar" name

    /// The F# type that names a C type by value.
    let rec fsTypeOf (name: string) : string =
        match scalars.TryGetValue name with
        | true, (fs, _, _) -> fs
        | _ ->
            match typeDef name with
            | TypeAlias(_, t) -> fsTypeOf t
            | Basetype(n, _) -> n
            | Bitmask(n, _, _) -> n
            | Handle(n, _) -> n
            | EnumType n when emptyEnums.Contains n ->
                // An enum with no bound members is its underlying integer.
                match enumByName.TryGetValue n with
                | true, e when e.Bits64 -> "VkFlags64"
                | true, e when e.IsBitmask -> "VkFlags"
                | _ -> "int32"
            | EnumType n -> n
            | Struct s -> s.Name
            | FuncPointer _ -> "nativeint"
            | Other(n, _) -> fail "no F# type for %s" n

    let rec elementName (name: string) : string =
        match scalars.TryGetValue name with
        | true, (fs, _, _) -> fs
        | _ ->
            match typeDef name with
            | TypeAlias(_, t) -> elementName t
            | Basetype(_, u) -> elementName u
            | _ -> fsTypeOf name

    let rec kindOf (name: string) : FieldKind =
        match scalars.TryGetValue name with
        | true, (_, _, k) -> KScalar k
        | _ ->
            match typeDef name with
            | TypeAlias(_, t) -> kindOf t
            | Basetype(_, u) -> kindOf u
            | Bitmask _
            | EnumType _ -> KScalar 'i'
            | Handle _ -> KHandle
            | FuncPointer _ -> KPointer
            | Struct _ -> KStruct
            | Other(n, _) -> fail "no kind for %s" n

    let fieldOf (s: StructDef) (d: CDecl) (offset: int) : BField =
        let size, _ = declLayout d

        let fsType, kind =
            if d.Pointers > 0 then
                "nativeint", KPointer
            else
                match d.Dims with
                | [] ->
                    match reg.Types.TryGetValue d.TypeName with
                    | true, FuncPointer _ -> "nativeint", KPointer
                    | _ -> fsTypeOf d.TypeName, kindOf d.TypeName
                | [ _ ] ->
                    let esize, ealign = typeLayout d.TypeName
                    let count = size / esize
                    let elem = elementName d.TypeName
                    let arrayName = sprintf "Array%d_%s" count elem

                    let ekind =
                        match kindOf d.TypeName with
                        | KScalar k -> k
                        | KHandle -> 'p'
                        | KStruct -> 's'
                        | _ -> fail "%s.%s: unsupported array element" s.Name d.Name

                    if not (arrays.ContainsKey arrayName) then
                        arrays[arrayName] <-
                            {
                                Name = arrayName
                                ElementC = d.TypeName
                                ElementFs = elem
                                ElementKind = ekind
                                ElementSize = esize
                                Count = count
                                Size = size
                                Align = ealign
                            }

                    arrayName, KArray(ekind, esize, count)
                | _ -> fail "%s.%s is a multi-dimensional array, which the generator does not bind" s.Name d.Name

        {
            Decl = d
            FsName = ident d.Name
            FsType = fsType
            Kind = kind
            Offset = offset
            Size = size
        }

    let structureTypes =
        match enumByName.TryGetValue "VkStructureType" with
        | true, e -> e.Members |> dict
        | _ -> dict []

    let boundStructs =
        [
            for s in structs.Values ->
                let size, align, offs = structLayout s
                let fields = List.map2 (fieldOf s) s.Members offs

                let sType =
                    s.Members
                    |> List.tryFind (fun m -> m.Name = "sType" && m.TypeName = "VkStructureType")
                    |> Option.bind _.Values
                    |> Option.map (fun v ->
                        match structureTypes.TryGetValue v with
                        | true, x -> v, x
                        | _ ->
                            fail
                                "%s: its sType %s is not bound; add the core version or extension that defines it"
                                s.Name
                                v)

                {
                    Def = s
                    Fields = fields
                    Size = size
                    Align = align
                    SType = sType
                }
        ]

    // Dependency order: a struct after every struct it holds by value (directly or as array elements).
    let structByName = boundStructs |> List.map (fun s -> s.Def.Name, s) |> dict

    let deps (s: BStruct) =
        s.Def.Members
        |> List.choose (fun m ->
            if m.Pointers > 0 then
                None
            else
                let t = resolveAlias m.TypeName
                if structByName.ContainsKey t then Some t else None)
        |> List.distinct

    let ordered =
        let placed = HashSet<string>(StringComparer.Ordinal)
        let result = ResizeArray<BStruct>()
        let pending = SortedSet<string>(structByName.Keys, StringComparer.Ordinal)

        while pending.Count > 0 do
            let ready =
                pending
                |> Seq.tryFind (fun n -> deps structByName[n] |> List.forall placed.Contains)

            match ready with
            | Some n ->
                pending.Remove n |> ignore
                placed.Add n |> ignore
                result.Add structByName[n]
            | None -> fail "struct dependency cycle among %s" (String.concat ", " pending)

        List.ofSeq result

    // ---- commands ----
    let providersOf (name: string) =
        boundRequires
        |> List.filter (fun (_, r) -> List.contains name r.Commands)
        |> List.map fst
        |> List.distinct
        |> List.sortBy (function
            | Core(a, c) -> (0, a, c, "")
            | Ext(n, _) -> (1, 0, 0, n))

    let dispatchable (t: string) =
        match reg.Types.TryGetValue t with
        | true, Handle(n, true) -> Some n
        | _ -> None

    let levelOf (c: Command) =
        if c.Name = "vkGetDeviceProcAddr" then
            Instance
        elif c.Name = "vkGetInstanceProcAddr" then
            Global
        else
            match
                c.Params
                |> List.tryHead
                |> Option.bind (fun p -> if p.Pointers = 0 then dispatchable p.TypeName else None)
            with
            | Some "VkInstance"
            | Some "VkPhysicalDevice" -> Instance
            | Some "VkDevice"
            | Some "VkQueue"
            | Some "VkCommandBuffer" -> Device
            | _ -> Global

    let paramFs (p: CDecl) =
        if p.Pointers > 0 then
            "nativeint"
        elif not p.Dims.IsEmpty then
            fail "parameter %s is an array, which the generator does not bind" p.Name
        else
            fsTypeOf p.TypeName

    let returnFs (p: CDecl) =
        if p.Pointers > 0 then "nativeint"
        elif p.TypeName = "void" then "unit"
        else fsTypeOf p.TypeName

    let isBoundValueType (t: string) =
        scalars.ContainsKey t
        || (match reg.Types.TryGetValue t with
            | true, (Struct _ | Handle _ | Basetype _ | Bitmask _ | EnumType _ | TypeAlias _) -> true
            | _ -> false)

    let wrapperOf (c: Command) : WParam list =
        let byName = c.Params |> List.map (fun p -> p.Name, p) |> dict

        let countOf (p: CDecl) =
            match p.Len with
            | Some l when byName.ContainsKey l && byName[l].Pointers = 0 -> Some l
            | _ -> None

        let elementFs (p: CDecl) =
            if p.TypeName = "void" then "byte" else fsTypeOf p.TypeName

        let spanable (p: CDecl) =
            p.Pointers = 1
            && countOf p |> Option.isSome
            && (p.TypeName = "void" || (p.TypeName <> "char" && isBoundValueType p.TypeName))

        // A count is dropped only when every array it sizes becomes a span.
        let counts =
            c.Params
            |> List.choose (fun p -> countOf p |> Option.map (fun n -> n, p))
            |> List.groupBy fst
            |> List.filter (fun (_, ps) -> ps |> List.forall (snd >> spanable))
            |> List.map (fun (n, ps) -> n, ps |> List.map (fun (_, p) -> p.Name))
            |> dict

        [
            for p in c.Params ->
                let fs = paramFs p

                if counts.ContainsKey p.Name then
                    Count(ident p.Name, fs, counts[p.Name] |> List.map ident)
                elif p.TypeName = "VkAllocationCallbacks" && p.Pointers = 1 then
                    Allocator(ident p.Name)
                elif p.Pointers = 0 then
                    ByValue(ident p.Name, fs)
                elif spanable p && counts.ContainsKey((countOf p).Value) then
                    let count = ident (countOf p).Value

                    if p.Const then
                        SpanIn(ident p.Name, elementFs p, count)
                    else
                        SpanOut(ident p.Name, elementFs p, count)
                elif p.Pointers = 1 && p.Len.IsNone && p.TypeName <> "void" && p.TypeName <> "char" then
                    let pointerOptional = (List.tryHead p.Optional = Some "true")

                    if pointerOptional || not (isBoundValueType p.TypeName) then
                        Raw(ident p.Name)
                    elif p.Const then
                        InRef(ident p.Name, fsTypeOf p.TypeName)
                    else
                        OutRef(ident p.Name, fsTypeOf p.TypeName)
                elif p.Pointers = 2 && p.TypeName = "void" && not p.Const && p.Len.IsNone then
                    OutRef(ident p.Name, "nativeint")
                else
                    Raw(ident p.Name)
        ]

    let resultByValue =
        match enumByName.TryGetValue "VkResult" with
        | true, e -> e.Members
        | _ -> []

    let resultMembers = resultByValue |> List.map fst |> set

    let commands =
        [
            for (n, listedLevel) in listed ->
                let c = commandDef n
                let level = levelOf c

                if level <> listedLevel then
                    fail "bindings.json lists %s as %A, but vk.xml makes it %A" n listedLevel level

                let providers = providersOf n

                if providers.IsEmpty then
                    fail
                        "%s is not provided by core %d.%d or a bound extension"
                        n
                        (fst b.CoreVersion)
                        (snd b.CoreVersion)

                // A success code named by an unbound alias (VK_PIPELINE_COMPILE_REQUIRED_EXT) becomes the bound
                // member with the same value (VK_PIPELINE_COMPILE_REQUIRED).
                let success =
                    if c.Return.TypeName = "VkResult" then
                        c.SuccessCodes
                        |> List.map (fun s ->
                            if resultMembers.Contains s then
                                s
                            else
                                let v = resolveValue [] s

                                match resultByValue |> List.tryFind (fun (_, x) -> x = v) with
                                | Some(m, _) -> m
                                | None -> fail "%s: success code %s is not a bound VkResult value" n s)
                        |> List.distinct
                    else
                        []

                {
                    Def = c
                    Level = level
                    Providers = providers
                    DelegateParams = c.Params |> List.map (fun p -> ident p.Name, paramFs p)
                    DelegateReturn = returnFs c.Return
                    Wrapper = wrapperOf c
                    SuccessCodes = success
                }
        ]

    let fps =
        [
            for n in funcPointers ->
                match typeDef n with
                | FuncPointer(_, ret, ps) ->
                    {
                        Name = n
                        Return = returnFs ret
                        Params = ps |> List.map (fun p -> ident p.Name, paramFs p)
                        Text = sprintf "%s (*%s)(%s)" ret.TypeName n (ps |> List.map _.Text |> String.concat ", ")
                    }
                | _ -> fail "%s is not a funcpointer" n
        ]

    // ---- constants ----
    for (_, r) in boundRequires do
        for e in r.Enums do
            if
                e.Extends.IsNone
                && e.Value.IsNone
                && e.Text.IsNone
                && reg.Constants.ContainsKey e.Name
            then
                constantNames.Add e.Name |> ignore

    let constants =
        [
            for n in constantNames ->
                let k = reg.Constants[n]

                let literal =
                    match k.CType with
                    | "uint32_t" -> sprintf "%du" (uint32 (parseNumber k.Value))
                    | "uint64_t" -> sprintf "%dUL" (uint64 (parseNumber k.Value))
                    | "float" -> k.Value.TrimEnd('F', 'f') + "f"
                    | t -> fail "API constant %s has type %s" n t

                k, literal
        ]

    let extensionConstants =
        [
            for x in boundExtensions do
                for r in activeRequires x.Requires do
                    for e in r.Enums do
                        if e.Extends.IsNone then
                            match e.Text, e.Value with
                            | Some t, _ -> yield e.Name, sprintf "\"%s\"" t
                            | None, Some(Num v) -> yield e.Name, sprintf "%du" (uint32 v)
                            | _ -> ()
        ]
        |> List.distinctBy fst

    {
        Registry = reg
        Bindings = b
        Basetypes = basetypes |> Seq.map (fun (KeyValue(k, v)) -> k, v) |> List.ofSeq
        Handles = handles |> Seq.map (fun (KeyValue(k, v)) -> k, v) |> List.ofSeq
        Enums = enumsWithMembers
        Bitmasks = bitmaskTargets
        Aliases = aliases.Keys |> Seq.map (fun k -> k, fsTypeOf k) |> List.ofSeq
        Arrays = arrays.Values |> List.ofSeq
        Structs = ordered
        FuncPointers = fps
        Commands = commands
        Constants = constants
        ExtensionConstants = extensionConstants
        BoundExtensions = boundExtensions
    }
