/// Resolves the bound items of bindings.json against the parsed headers: type mapping, C layout, the closure of
/// structs and enums, interface inheritance, and the per-method calling shape the wrappers use.
module Lrm.Tools.D3D12Gen.Bind

open System
open System.Collections.Generic
open Lrm.Tools.D3D12Gen.Header
open Lrm.Tools.D3D12Gen.CppPass

/// A C scalar and its F# spelling. Sizes and alignments are the x64 MSVC ones.
type Prim =
    {
        FSharp: string
        Size: int
        Align: int
    }

let private prim f s = { FSharp = f; Size = s; Align = s }

/// C scalar typedefs and keywords used by the bound headers.
let primitives: IReadOnlyDictionary<string, Prim> =
    let i32 = prim "int" 4
    let u32 = prim "uint32" 4
    let u64 = prim "uint64" 8
    let i64 = prim "int64" 8
    let u16 = prim "uint16" 2
    let i16 = prim "int16" 2
    let u8 = prim "byte" 1
    let i8 = prim "sbyte" 1
    let usize = prim "unativeint" 8
    let isize = prim "nativeint" 8

    readOnlyDict
        [
            for n in [ "BOOL"; "INT"; "int"; "signed int"; "LONG"; "long"; "INT32"; "HRESULT" ] -> n, i32
            for n in [ "UINT"; "UINT32"; "ULONG"; "DWORD"; "unsigned int"; "unsigned long" ] -> n, u32
            for n in [ "UINT64"; "ULONGLONG"; "ULONG64"; "DWORD64"; "unsigned __int64" ] -> n, u64
            for n in [ "INT64"; "LONGLONG"; "LONG64"; "__int64" ] -> n, i64
            for n in [ "SIZE_T"; "ULONG_PTR"; "UINT_PTR"; "DWORD_PTR" ] -> n, usize
            for n in [ "SSIZE_T"; "LONG_PTR"; "INT_PTR" ] -> n, isize
            for n in [ "UINT16"; "USHORT"; "WORD"; "unsigned short"; "WCHAR"; "wchar_t" ] -> n, u16
            for n in [ "INT16"; "SHORT"; "short" ] -> n, i16
            for n in [ "UINT8"; "BYTE"; "UCHAR"; "unsigned char"; "BOOLEAN" ] -> n, u8
            for n in [ "INT8"; "CHAR"; "char" ] -> n, i8
            yield "FLOAT", prim "float32" 4
            yield "float", prim "float32" 4
            yield "DOUBLE", prim "float" 8
            yield "double", prim "float" 8
            // A union { struct { DWORD LowPart; LONG HighPart; }; LONGLONG QuadPart; }: 8 bytes, 8-aligned.
            yield "LARGE_INTEGER", i64
            yield
                "GUID",
                {
                    FSharp = "Guid"
                    Size = 16
                    Align = 4
                }
            yield
                "IID",
                {
                    FSharp = "Guid"
                    Size = 16
                    Align = 4
                }
        ]

/// Pointer-sized handle and string typedefs: always nativeint.
let private handles =
    set
        [
            "HANDLE"
            "HWND"
            "HMONITOR"
            "HMODULE"
            "HDC"
            "HINSTANCE"
            "LPVOID"
            "PVOID"
            "LPCVOID"
            "LPCWSTR"
            "LPWSTR"
            "LPCSTR"
            "LPSTR"
            "REFIID"
            "REFGUID"
            "REFCLSID"
        ]

/// Structs from headers the generator does not read (winnt.h, windef.h), modeled by hand. The ABI oracle checks
/// them against the real definitions like every other struct.
let builtinAggregates =
    let f n t = Field { Name = n; Type = Named t }

    [
        {
            Name = "LUID"
            IsUnion = false
            Members = [ f "LowPart" "DWORD"; f "HighPart" "LONG" ]
            File = "winnt.h"
            Line = 0
        }
        {
            Name = "RECT"
            IsUnion = false
            Members = [ f "left" "LONG"; f "top" "LONG"; f "right" "LONG"; f "bottom" "LONG" ]
            File = "windef.h"
            Line = 0
        }
    ]

/// The IID of IUnknown (unknwnbase.h), the root of every bound interface.
let iidIUnknown = Guid("00000000-0000-0000-c000-000000000046")

type Resolved =
    | RPrim of cname: string * Prim
    | RHandle of cname: string
    | REnum of string
    | RStruct of string
    | RInterface of string
    | RPointer of Resolved
    | RArray of Resolved * int
    | RVoid
    | RUnknown of string

type Bindings =
    {
        SdkVersion: string
        SdkRoot: string
        Headers: (string * string) list
        Interfaces: string list
        Structs: string list
        Enums: string list
        OpaqueEnums: string list
        /// Interface::Method -> the wrapper name, for methods whose C name an F# intrinsic member would hide.
        Renames: IReadOnlyDictionary<string, string>
        /// Interface::Method entries whose HRESULT is a value, not a failure (no raising wrapper).
        HresultValues: Set<string>
    }

let loadBindings (path: string) : Bindings =
    use doc = Text.Json.JsonDocument.Parse(IO.File.ReadAllText path)
    let r = doc.RootElement

    let strings (name: string) =
        match r.TryGetProperty name with
        | true, v -> [ for x in v.EnumerateArray() -> x.GetString() ]
        | _ -> []

    let sdk = r.GetProperty "sdk"

    {
        SdkVersion = sdk.GetProperty("version").GetString()
        SdkRoot = sdk.GetProperty("root").GetString()
        Headers =
            [
                for p in sdk.GetProperty("headers").EnumerateObject() -> p.Name, p.Value.GetString()
            ]
        Interfaces = strings "interfaces"
        Structs = strings "structs"
        Enums = strings "enums"
        OpaqueEnums = strings "opaqueEnums"
        Renames =
            match r.TryGetProperty "renames" with
            | true, v -> readOnlyDict [ for p in v.EnumerateObject() -> p.Name, p.Value.GetString() ]
            | _ -> readOnlyDict []
        HresultValues = set (strings "hresultValues")
    }

/// One flattened field of a bound struct, with its byte offset.
type LaidField =
    {
        Name: string
        Type: Resolved
        Offset: int
    }

type Layout =
    {
        Size: int
        Align: int
        Fields: LaidField list
        /// True when the struct has union members, so the F# type needs explicit offsets.
        Explicit: bool
    }

/// How one parameter is passed by the typed wrapper.
type ParamShape =
    /// Passed through as the delegate's type.
    | Plain
    /// ComPtr<'P> where 'P :> the interface.
    | InterfaceIn of string
    /// A non-null string, pinned for the call.
    | WideString
    /// The REFIID of an (riid, ppv) pair, filled from IidOf<'T>.
    | Riid
    /// The void** of an (riid, ppv) pair; the wrapper returns ComPtr<'T>.
    | OutGeneric
    /// An Interface** out parameter; the wrapper returns ComPtr<Interface>.
    | OutInterface of string
    /// The hidden RetVal pointer of a struct-returning method; the wrapper returns the struct.
    | RetVal of string
    /// An _In_ pointer to a bound struct: the wrapper takes inref<T> and pins it for the call.
    | InStruct of string
    /// The last parameter, an _Out_ pointer to one value (a scalar, enum or bound struct) of an HRESULT method:
    /// the wrapper returns the value.
    | OutValue of string

type BoundMethod =
    {
        Interface: string
        Method: Method
        /// The wrapper's name (the C name unless renamed).
        WrapperName: string
        Shapes: ParamShape list
        /// Delegate parameter types after This, in F# spelling.
        DelegateParams: string list
        DelegateReturn: string
        ReturnsHresult: bool
        HresultIsValue: bool
    }

type BoundInterface =
    {
        Name: string
        Base: string option
        Iid: Guid
        File: string
        /// All slots, inherited ones included.
        Slots: Method list
        /// Methods this interface declares.
        Own: BoundMethod list
    }

type Model =
    {
        Bindings: Bindings
        Header: Header
        Interfaces: BoundInterface list
        /// Structs in dependency order (field types first).
        Structs: (Aggregate * Layout) list
        /// Inline array types: (name, element F# type, element C name, count).
        Arrays: (string * string * string * int) list
        Enums: (EnumDef * string * bool) list
        Opaque: Set<string>
        /// Emission order of struct-like items: Choice1Of2 = array type name, Choice2Of2 = struct name.
        Order: Choice<string, string> list
        /// The F# spelling of a resolved type.
        FsType: Resolved -> string
        Resolve: CType -> Resolved
    }

let private fail fmt =
    Printf.kprintf (fun s -> raise (InvalidOperationException s)) fmt

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
            "atomic"
            "break"
            "checked"
            "component"
            "constraint"
            "constructor"
            "continue"
            "eager"
            "event"
            "external"
            "functor"
            "include"
            "method"
            "mixin"
            "object"
            "parallel"
            "process"
            "protected"
            "pure"
            "sealed"
            "tailcall"
            "trait"
            "virtual"
            "volatile"
            "params"
            "asr"
            "land"
            "lor"
            "lsl"
            "lsr"
            "lxor"
            "mod"
            "sig"
            "base"
            "this"
        ]

/// An identifier as F# source: keywords are wrapped in double backticks.
let ident (s: string) =
    if keywords.Contains s then "``" + s + "``" else s

/// Member names that an F# intrinsic member of ComPtr or System.Object would hide.
let private hidden =
    set
        [
            "GetType"
            "GetHashCode"
            "ToString"
            "Equals"
            "Dispose"
            "Ptr"
            "IsNull"
            "AddRef"
        ]

let build (b: Bindings) (h: Header) (cpp: IReadOnlyDictionary<string, CppInterface>) : Model =
    let aggregates = Dictionary<string, Aggregate>(StringComparer.Ordinal)

    for kv in h.Aggregates do
        aggregates[kv.Key] <- kv.Value

    for a in builtinAggregates do
        if aggregates.ContainsKey a.Name then
            fail "%s is both builtin and in the headers" a.Name

        aggregates[a.Name] <- a

    let opaque = set b.OpaqueEnums

    let isInterface (n: string) =
        n = "IUnknown" || h.Vtables.ContainsKey n || cpp.ContainsKey n

    let rec resolveNamed (n: string) (seen: Set<string>) =
        if seen.Contains n then
            fail "typedef cycle at %s" n

        if n = "void" then
            RVoid
        elif primitives.ContainsKey n then
            RPrim(n, primitives[n])
        elif handles.Contains n then
            RHandle n
        elif h.Enums.ContainsKey n then
            REnum n
        elif aggregates.ContainsKey n then
            RStruct n
        elif isInterface n then
            RInterface n
        else
            match h.Typedefs.TryGetValue n with
            | true, (Named t) when t = n -> RUnknown n
            | true, t -> resolveIn t (seen.Add n)
            | _ -> RUnknown n

    and resolveIn (t: CType) seen =
        match t with
        | Named n -> resolveNamed n seen
        | Pointer t -> RPointer(resolveIn t seen)
        | CType.Array(t, n) -> RArray(resolveIn t seen, n)

    let resolve t = resolveIn t Set.empty

    // ---- layout ------------------------------------------------------------------------------------------
    let layouts = Dictionary<string, Layout>(StringComparer.Ordinal)

    let alignUp (x: int) (a: int) = (x + a - 1) / a * a

    let rec sizeAlign (r: Resolved) (where: string) : int * int =
        match r with
        | RPrim(_, p) -> p.Size, p.Align
        | RHandle _
        | RPointer _ -> 8, 8
        | REnum _ -> 4, 4
        | RArray(e, n) ->
            let s, a = sizeAlign e where
            s * n, a
        | RStruct s ->
            let l = layoutOf s
            l.Size, l.Align
        | RInterface i -> fail "%s: interface %s used by value" where i
        | RVoid -> fail "%s: void used by value" where
        | RUnknown u -> fail "%s: unknown type %s" where u

    and measure (isUnion: bool) (ms: Member list) (where: string) : int * int =
        let mutable cur = 0
        let mutable maxA = 1

        for m in ms do
            let s, a =
                match m with
                | Field f -> sizeAlign (resolve f.Type) (where + "." + f.Name)
                | AnonUnion inner -> measure true inner where
                | AnonStruct inner -> measure false inner where
                | Unsupported why -> fail "%s: %s" where why

            maxA <- max maxA a

            if isUnion then
                cur <- max cur s
            else
                cur <- alignUp cur a + s

        alignUp cur maxA, maxA

    and flatten (isUnion: bool) (baseOffset: int) (ms: Member list) (where: string) : LaidField list =
        let mutable cur = 0

        [
            for m in ms do
                let s, a =
                    match m with
                    | Field f -> sizeAlign (resolve f.Type) (where + "." + f.Name)
                    | AnonUnion inner -> measure true inner where
                    | AnonStruct inner -> measure false inner where
                    | Unsupported why -> fail "%s: %s" where why

                let off = if isUnion then 0 else alignUp cur a

                match m with
                | Field f ->
                    yield
                        {
                            Name = f.Name
                            Type = resolve f.Type
                            Offset = baseOffset + off
                        }
                | AnonUnion inner -> yield! flatten true (baseOffset + off) inner where
                | AnonStruct inner -> yield! flatten false (baseOffset + off) inner where
                | Unsupported why -> fail "%s: %s" where why

                if not isUnion then
                    cur <- off + s
        ]

    and layoutOf (name: string) : Layout =
        match layouts.TryGetValue name with
        | true, l -> l
        | _ ->
            let a = aggregates[name]
            let size, align = measure a.IsUnion a.Members name
            let fields = flatten a.IsUnion 0 a.Members name

            let hasUnion =
                a.IsUnion
                || a.Members
                   |> List.exists (function
                       | AnonUnion _ -> true
                       | _ -> false)

            let l =
                {
                    Size = size
                    Align = align
                    Fields = fields
                    Explicit = hasUnion
                }

            layouts[name] <- l
            l

    // ---- F# spelling ---------------------------------------------------------------------------------------
    let arrays = Dictionary<string, string * string * int>(StringComparer.Ordinal)

    let rec cName (r: Resolved) =
        match r with
        | RPrim(n, _) -> n.Replace(' ', '_')
        | RHandle n -> n
        | REnum n
        | RStruct n
        | RInterface n -> n
        | RPointer _ -> "PTR"
        | RArray(e, n) -> sprintf "%s_%d" (cName e) n
        | RVoid -> "void"
        | RUnknown n -> n

    let rec fsType (r: Resolved) (where: string) : string =
        match r with
        | RPrim(_, p) -> p.FSharp
        | RHandle _
        | RPointer _ -> "nativeint"
        | REnum e -> if opaque.Contains e then "int" else e
        | RStruct s -> s
        | RArray(e, n) ->
            let name = sprintf "Array%d_%s" n (cName e)
            let elem = fsType e where

            if not (arrays.ContainsKey name) then
                arrays[name] <- (elem, cName e, n)

            name
        | RInterface i -> fail "%s: interface %s by value" where i
        | RVoid -> "unit"
        | RUnknown u -> fail "%s: unknown type %s" where u

    // ---- interfaces -----------------------------------------------------------------------------------------
    let bound = b.Interfaces |> List.distinct
    let boundSet = set ("IUnknown" :: bound)

    let parentOf (i: string) =
        if i = "IUnknown" then
            None
        else
            match cpp.TryGetValue i with
            | true, c -> Some c.Base
            | _ -> fail "%s: no MIDL_INTERFACE declaration found" i

    for i in bound do
        if not (h.Vtables.ContainsKey i) then
            fail "%s: no C vtable (typedef struct %sVtbl) found" i i

        let rec walk (x: string) =
            match parentOf x with
            | Some p when not (boundSet.Contains p) -> fail "%s derives from %s, which is not bound" i p
            | Some p -> walk p
            | None -> ()

        walk i

    let iidOf (i: string) =
        if i = "IUnknown" then
            iidIUnknown
        else
            let fromDefine =
                match h.Iids.TryGetValue("IID_" + i) with
                | true, g -> Some g
                | _ -> None

            let fromCpp = cpp[i].Iid

            match fromDefine with
            | Some g when g <> fromCpp -> fail "%s: DEFINE_GUID %O differs from MIDL_INTERFACE %O" i g fromCpp
            | _ -> fromCpp

    // IUnknown's slots are the first three of every vtable.
    let unknownSlots =
        let vt = h.Vtables[bound.Head]
        vt.Methods |> List.filter (fun m -> m.Declarer = "IUnknown")

    if unknownSlots |> List.map _.Name <> [ "QueryInterface"; "AddRef"; "Release" ] then
        fail "unexpected IUnknown slots %A" (unknownSlots |> List.map _.Name)

    let slotsOf (i: string) =
        if i = "IUnknown" then
            unknownSlots
        else
            h.Vtables[i].Methods

    // Each vtable must be its parent's vtable followed by the interface's own methods.
    for i in bound do
        let mine = slotsOf i
        let p = (parentOf i).Value
        let parent = slotsOf p

        let prefix = mine |> List.truncate parent.Length

        if
            (prefix |> List.map (fun m -> m.Declarer, m.Name))
            <> (parent |> List.map (fun m -> m.Declarer, m.Name))
        then
            fail "%s: vtable does not start with %s's vtable" i p

        for m in mine |> List.skip parent.Length do
            if m.Declarer <> i then
                fail "%s: slot %d (%s) is declared by %s, expected %s" i m.Slot m.Name m.Declarer i

    let ancestors (i: string) =
        let rec go x acc =
            match parentOf x with
            | Some p -> go p (p :: acc)
            | None -> acc

        go i [] |> List.rev

    // ---- struct and enum closure ----------------------------------------------------------------------
    let structOrder = ResizeArray<Choice<string, string>>()
    let structSeen = HashSet<string>(StringComparer.Ordinal)
    let enumSeen = HashSet<string>(StringComparer.Ordinal)

    let rec visitType (r: Resolved) (where: string) =
        match r with
        | REnum e -> enumSeen.Add e |> ignore
        | RStruct s -> visitStruct s
        | RPointer inner ->
            match inner with
            | REnum _
            | RStruct _ -> visitType inner where
            | _ -> ()
        | RArray(e, _) ->
            visitType e where
            let name = fsType r where
            let c = Choice1Of2 name

            if not (structOrder.Contains c) then
                structOrder.Add c
        | RUnknown u -> fail "%s: unknown type %s" where u
        | _ -> ()

    and visitStruct (s: string) =
        if structSeen.Add s then
            if not (aggregates.ContainsKey s) then
                fail "struct %s not found" s

            let l = layoutOf s

            for f in l.Fields do
                visitType f.Type (s + "." + f.Name)

            structOrder.Add(Choice2Of2 s)

    for s in b.Structs do
        visitStruct s

    for e in b.Enums do
        if not (h.Enums.ContainsKey e) then
            fail "enum %s not found" e

        enumSeen.Add e |> ignore

    // By-value parameters, RetVal structs and enum returns of every bound method.
    for i in bound do
        for m in slotsOf i |> List.filter (fun m -> m.Declarer = i) do
            for p in m.Params do
                match resolve p.Type with
                | REnum _
                | RStruct _ as r -> visitType r (i + "::" + m.Name + " " + p.Name)
                | RPointer(RStruct st) when p.Name = "RetVal" -> visitStruct st
                | _ -> ()

            match resolve m.Return with
            | REnum _ as r -> visitType r m.Name
            | _ -> ()


    // ---- method shapes ---------------------------------------------------------------------------------
    let isBoundInterface (r: Resolved) =
        match r with
        | RInterface n when boundSet.Contains n -> Some n
        | _ -> None

    let methodShape (iface: string) (m: Method) : BoundMethod =
        let key = iface + "::" + m.Name
        let where = key
        let ps = m.Params |> List.map (fun p -> p, resolve p.Type)
        let ret = resolve m.Return

        let isOutAnn (p: Param) =
            p.Annotations
            |> List.exists (fun a -> a.StartsWith "_COM_Outptr_" || a.StartsWith "_Outptr_")

        let n = ps.Length

        let outCount =
            ps
            |> List.filter (fun (p, _) -> List.contains "_Out_" p.Annotations)
            |> List.length

        let shapes =
            ps
            |> List.mapi (fun k (p, r) ->
                let isLast = k = n - 1

                match r with
                | RHandle "REFIID" when
                    k + 1 < n
                    && (match snd ps[k + 1] with
                        | RPointer(RPointer RVoid) -> isOutAnn (fst ps[k + 1]) && k + 1 = n - 1
                        | _ -> false)
                    ->
                    Riid
                | RPointer(RPointer RVoid) when k > 0 && (snd ps[k - 1] = RHandle "REFIID") && isOutAnn p && isLast ->
                    OutGeneric
                | RPointer(RPointer inner) when isLast && isOutAnn p && (isBoundInterface inner).IsSome ->
                    OutInterface (isBoundInterface inner).Value
                | RPointer inner when isLast && p.Name = "RetVal" ->
                    match inner with
                    | RStruct s -> RetVal s
                    | other -> fail "%s: RetVal of %A" where other
                | RPointer inner when (isBoundInterface inner).IsSome -> InterfaceIn (isBoundInterface inner).Value
                | RHandle "LPCWSTR" -> WideString
                | RPointer(RStruct st) when List.contains "_In_" p.Annotations && structSeen.Contains st -> InStruct st
                | RPointer inner when
                    isLast
                    && outCount = 1
                    && List.contains "_Out_" p.Annotations
                    && (match inner with
                        | RPrim _ -> true
                        | REnum e -> enumSeen.Contains e && not (opaque.Contains e)
                        | RStruct st -> structSeen.Contains st
                        | _ -> false)
                    ->
                    OutValue(fsType inner where)
                | _ -> Plain)

        let delegateParams =
            ps
            |> List.map (fun (p, r) ->
                match r with
                | RArray _ -> "nativeint"
                | r -> fsType r (where + " " + p.Name))

        let retType =
            match ret with
            | RVoid -> "unit"
            | RPointer _ -> "nativeint"
            | RStruct s -> fail "%s returns %s by value without a RetVal pointer" where s
            | r -> fsType r where

        match shapes |> List.tryLast with
        | Some(RetVal s) when ret <> RPointer(RStruct s) ->
            fail "%s: RetVal methods must return the RetVal pointer" where
        | _ -> ()

        let wrapperName =
            match b.Renames.TryGetValue key with
            | true, n -> n
            | _ ->
                if hidden.Contains m.Name then
                    fail "%s: the wrapper name %s is hidden by an intrinsic member; add a rename" where m.Name

                m.Name

        {
            Interface = iface
            Method = m
            WrapperName = wrapperName
            Shapes = shapes
            DelegateParams = delegateParams
            DelegateReturn = retType
            ReturnsHresult = (ret = RPrim("HRESULT", primitives["HRESULT"]))
            HresultIsValue = b.HresultValues.Contains key
        }

    for key in b.HresultValues do
        let i, m =
            match key.Split "::" with
            | [| i; m |] -> i, m
            | _ -> fail "hresultValues: bad entry %s" key

        if
            not (
                boundSet.Contains i
                && (slotsOf i |> List.exists (fun x -> x.Name = m && x.Declarer = i))
            )
        then
            fail "hresultValues: %s is not a bound method" key

    let interfaces =
        let order = ResizeArray<string>()

        let rec visit (i: string) =
            if not (order.Contains i) then
                parentOf i |> Option.iter visit
                order.Add i

        visit "IUnknown"

        for i in bound do
            visit i

        [
            for i in order ->
                let own =
                    if i = "IUnknown" then
                        []
                    else
                        slotsOf i |> List.filter (fun m -> m.Declarer = i) |> List.map (methodShape i)

                // A method name may appear only once along an inheritance chain, so overloads stay unambiguous.
                for m in own do
                    for a in ancestors i do
                        if slotsOf a |> List.exists (fun x -> x.Declarer = a && x.Name = m.Method.Name) then
                            fail "%s::%s hides %s::%s" i m.Method.Name a m.Method.Name

                {
                    Name = i
                    Base = parentOf i
                    Iid = iidOf i
                    File =
                        if i = "IUnknown" then
                            "unknwnbase.h"
                        else
                            IO.Path.GetFileName h.Vtables[i].File
                    Slots = slotsOf i
                    Own = own
                }
        ]

    for e in b.OpaqueEnums do
        enumSeen.Remove e |> ignore

    let enums =
        enumSeen
        |> Seq.sort
        |> Seq.map (fun e ->
            let def = h.Enums[e]

            let names = def.Members |> List.map _.Name

            // The longest common prefix that ends at '_' and leaves every member a valid F# identifier.
            let common =
                let first = names.Head
                let mutable len = first.Length

                for n in names do
                    let mutable k = 0

                    while k < len && k < n.Length && n[k] = first[k] do
                        k <- k + 1

                    len <- k

                let cut = first.Substring(0, len)
                let last = cut.LastIndexOf '_'
                if last < 0 then "" else cut.Substring(0, last + 1)

            // A few members that would start with a digit (DXGI_FORMAT_420_OPAQUE) get a leading underscore
            // instead (_420_OPAQUE); when more than a tenth would, the prefix gives up one segment instead.
            let rec usable (prefix: string) =
                let rests = names |> List.map (fun n -> n.Substring prefix.Length)

                let digits =
                    rests |> List.filter (fun r -> r.Length > 0 && Char.IsDigit r[0]) |> List.length

                let ok =
                    digits * 10 <= names.Length
                    && rests |> List.forall (fun r -> r.Length > 0 && r <> "_")

                if ok || prefix = "" then
                    prefix
                else
                    let trimmed = prefix.Substring(0, prefix.Length - 1)
                    let k = trimmed.LastIndexOf '_'
                    usable (if k < 0 then "" else trimmed.Substring(0, k + 1))

            def, usable common, h.FlagEnums.Contains e)
        |> List.ofSeq

    {
        Bindings = b
        Header = h
        Interfaces = interfaces
        Structs =
            [
                for s in structOrder do
                    match s with
                    | Choice2Of2 n -> aggregates[n], layoutOf n
                    | _ -> ()
            ]
        Arrays = [ for kv in arrays -> let (e, c, n) = kv.Value in kv.Key, e, c, n ]
        Enums = enums
        Opaque = opaque
        Order = List.ofSeq structOrder
        FsType = fun r -> fsType r "emit"
        Resolve = resolve
    }
