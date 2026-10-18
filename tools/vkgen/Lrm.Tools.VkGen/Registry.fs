/// Reads the Khronos registry (vk.xml) into a plain model for one API ("vulkan"): types, enums, API constants,
/// commands, features and extensions. Nothing here decides what is bound; Bind.fs does. Elements whose `api`
/// attribute does not name "vulkan" (Vulkan SC variants) are dropped while reading, so every name maps to one
/// definition.
module Lrm.Tools.VkGen.Registry

open System
open System.Collections.Generic
open System.Globalization
open System.Text
open System.Text.RegularExpressions
open System.Xml.Linq

/// One array dimension of a member: a literal or an API constant such as VK_UUID_SIZE.
type ArrayDim =
    | Literal of int
    | Constant of string

/// A C declaration of a struct member, a command parameter or a return value: `const T* const* name[N]`.
type CDecl =
    {
        Name: string
        TypeName: string
        /// `const` appears before the type: the pointee of a pointer is const.
        Const: bool
        /// The number of `*`.
        Pointers: int
        Dims: ArrayDim list
        BitField: int option
        Len: string option
        /// The `optional` attribute, split at commas ("true", "false,true").
        Optional: string list
        /// The `values` attribute (the sType value of an sType member).
        Values: string option
        /// The declaration as C text, for doc comments.
        Text: string
    }

type StructDef =
    {
        Name: string
        IsUnion: bool
        Members: CDecl list
        Extends: string list
        ReturnedOnly: bool
        AllowDuplicate: bool
    }

type TypeDef =
    | Basetype of name: string * underlying: string
    /// A VkFlags or VkFlags64 typedef and the FlagBits enum that names its bits, if any.
    | Bitmask of name: string * flagBits: string option * bits64: bool
    | Handle of name: string * dispatchable: bool
    | EnumType of name: string
    | Struct of StructDef
    | FuncPointer of name: string * ret: CDecl * pars: CDecl list
    | TypeAlias of name: string * target: string
    /// Defines, includes, platform types and opaque structs: never bound.
    | Other of name: string * category: string

type EnumValue =
    | Num of int64
    | AliasOf of string

type EnumMember = { Name: string; Value: EnumValue }

/// An `<enums>` block: the members a C enum or FlagBits type is declared with.
type EnumsBlock =
    {
        Name: string
        IsBitmask: bool
        Bits64: bool
        Members: EnumMember list
    }

type ApiConstant =
    {
        Name: string
        /// uint32_t, uint64_t or float.
        CType: string
        /// The value as written: 256, (~0U), (~0ULL), 1000.0F.
        Value: string
    }

/// An `<enum>` inside a `<require>`.
type RequireEnum =
    {
        Name: string
        Extends: string option
        /// None for a reference to an existing name (an API constant or an enum value).
        Value: EnumValue option
        /// The string value of an extension-name constant.
        Text: string option
    }

type Require =
    {
        Depends: string option
        Types: string list
        Commands: string list
        Enums: RequireEnum list
    }

type Feature =
    {
        Name: string
        Version: int * int
        Requires: Require list
    }

type Extension =
    {
        Name: string
        Number: int
        /// "instance" or "device".
        Kind: string
        Requires: Require list
    }

type Command =
    {
        Name: string
        Return: CDecl
        Params: CDecl list
        SuccessCodes: string list
        ErrorCodes: string list
    }

type Registry =
    {
        Types: IReadOnlyDictionary<string, TypeDef>
        Enums: IReadOnlyDictionary<string, EnumsBlock>
        Constants: IReadOnlyDictionary<string, ApiConstant>
        Commands: IReadOnlyDictionary<string, Command>
        /// Alias name to target name.
        CommandAliases: IReadOnlyDictionary<string, string>
        Features: Feature list
        Extensions: IReadOnlyDictionary<string, Extension>
        HeaderVersion: int
    }

let private attr (name: string) (e: XElement) =
    match e.Attribute(XName.Get name) with
    | null -> None
    | a -> Some a.Value

/// True when an element is for the "vulkan" API: no `api` attribute, or one that lists it.
let private forVulkan (e: XElement) =
    match attr "api" e with
    | None -> true
    | Some apis -> apis.Split(',') |> Array.exists (fun a -> a.Trim() = "vulkan")

let private children (name: string) (e: XElement) =
    e.Elements(XName.Get name) |> Seq.filter forVulkan |> List.ofSeq

let private splitList (s: string option) =
    match s with
    | None -> []
    | Some s ->
        s.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
        |> List.ofArray

let private dimRx = Regex(@"\[([A-Za-z0-9_]+)\]", RegexOptions.CultureInvariant)

/// Parses a <member>, <param> or <proto>: text and <type>, <name> and <enum> children in order.
let parseDecl (e: XElement) : CDecl =
    let pre = StringBuilder()
    let mid = StringBuilder()
    let post = StringBuilder()
    let text = StringBuilder()
    let mutable typeName = None
    let mutable name = None

    for n in e.Nodes() do
        match n with
        | :? XText as t ->
            text.Append(t.Value) |> ignore

            match typeName, name with
            | None, _ -> pre.Append(t.Value) |> ignore
            | Some _, None -> mid.Append(t.Value) |> ignore
            | Some _, Some _ -> post.Append(t.Value) |> ignore
        | :? XElement as c ->
            match c.Name.LocalName with
            | "type" ->
                typeName <- Some c.Value
                text.Append(c.Value) |> ignore
            | "name" ->
                name <- Some c.Value
                text.Append(c.Value) |> ignore
            | "enum" ->
                post.Append(c.Value) |> ignore
                text.Append(c.Value) |> ignore
            | _ -> () // <comment>
        | _ -> ()

    let typeName =
        match typeName with
        | Some t -> t
        | None -> failwithf "declaration without a <type>: %s" (e.ToString())

    let name =
        match name with
        | Some n -> n
        | None -> failwithf "declaration without a <name>: %s" (e.ToString())

    let postText = post.ToString().Trim()

    let bitField =
        if postText.StartsWith ":" then
            Some(Int32.Parse(postText.Substring(1).Trim(), CultureInfo.InvariantCulture))
        else
            None

    let dims =
        [
            for m in dimRx.Matches postText do
                let v = m.Groups[1].Value

                match Int32.TryParse(v, NumberStyles.None, CultureInfo.InvariantCulture) with
                | true, n -> Literal n
                | _ -> Constant v
        ]

    {
        Name = name
        TypeName = typeName
        Const = pre.ToString().Contains "const"
        Pointers = mid.ToString() |> Seq.filter (fun c -> c = '*') |> Seq.length
        Dims = dims
        BitField = bitField
        Len = attr "len" e
        Optional = splitList (attr "optional" e)
        Values = attr "values" e
        Text = Regex.Replace(text.ToString().Trim(), @"\s+", " ")
    }

/// Parses an enum value: decimal or hex, optionally negative, or (~0U), (~1U), (~0ULL).
let parseNumber (s: string) : int64 =
    let s = s.Trim()

    let complement =
        Regex.Match(s, @"^\(~(\d+)(U|ULL)\)$", RegexOptions.CultureInvariant)

    if complement.Success then
        let n = Int64.Parse(complement.Groups[1].Value, CultureInfo.InvariantCulture)

        if complement.Groups[2].Value = "U" then
            int64 (~~~(uint32 n))
        else
            int64 (~~~(uint64 n))
    elif s.StartsWith "0x" || s.StartsWith "0X" then
        int64 (UInt64.Parse(s.Substring 2, NumberStyles.HexNumber, CultureInfo.InvariantCulture))
    elif s.StartsWith "-0x" then
        -(Int64.Parse(s.Substring 3, NumberStyles.HexNumber, CultureInfo.InvariantCulture))
    else
        Int64.Parse(s, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture)

/// The value of an extension enum: 1000000000 + (extension - 1) * 1000 + offset, negated with dir="-".
let extensionValue (extNumber: int) (offset: int) (negative: bool) =
    let v = 1000000000L + int64 (extNumber - 1) * 1000L + int64 offset
    if negative then -v else v

let private parseRequireEnum (extNumber: int option) (e: XElement) : RequireEnum =
    let name = (attr "name" e).Value
    let extends = attr "extends" e

    let value, text =
        match attr "alias" e, attr "bitpos" e, attr "offset" e, attr "value" e with
        | Some a, _, _, _ -> Some(AliasOf a), None
        | _, Some b, _, _ -> Some(Num(1L <<< Int32.Parse(b, CultureInfo.InvariantCulture))), None
        | _, _, Some o, _ ->
            let ext =
                match attr "extnumber" e with
                | Some n -> Int32.Parse(n, CultureInfo.InvariantCulture)
                | None ->
                    match extNumber with
                    | Some n -> n
                    | None -> failwithf "enum %s has an offset but no extension number" name

            Some(Num(extensionValue ext (Int32.Parse(o, CultureInfo.InvariantCulture)) (attr "dir" e = Some "-"))), None
        | _, _, _, Some v when v.StartsWith "\"" -> None, Some(v.Trim('"'))
        | _, _, _, Some v when extends.IsSome -> Some(Num(parseNumber v)), None
        | _, _, _, Some v ->
            // An extension constant such as VK_EXT_DESCRIPTOR_HEAP_SPEC_VERSION.
            Some(Num(parseNumber v)), None
        | _ -> None, None

    {
        Name = name
        Extends = extends
        Value = value
        Text = text
    }

let private parseRequires (extNumber: int option) (e: XElement) : Require list =
    [
        for r in children "require" e do
            {
                Depends = attr "depends" r
                Types = children "type" r |> List.map (fun t -> (attr "name" t).Value)
                Commands = children "command" r |> List.map (fun c -> (attr "name" c).Value)
                Enums = children "enum" r |> List.map (parseRequireEnum extNumber)
            }
    ]

let private parseType (e: XElement) : TypeDef option =
    let category = defaultArg (attr "category" e) ""

    let nameOf () =
        match attr "name" e with
        | Some n -> n
        | None ->
            match e.Element(XName.Get "name") with
            | null -> ""
            | n -> n.Value

    match attr "alias" e with
    | Some target when category <> "" -> Some(TypeAlias(nameOf (), target))
    | _ ->
        match category with
        | "basetype" ->
            match e.Element(XName.Get "type") with
            | null -> Some(Other(nameOf (), category))
            | _ when e.Value.Contains "*" -> Some(Other(nameOf (), category)) // typedef void* VkRemoteAddressNV
            | t -> Some(Basetype(nameOf (), t.Value))
        | "bitmask" ->
            let underlying = e.Element(XName.Get "type").Value

            let bits =
                match attr "requires" e with
                | Some r -> Some r
                | None -> attr "bitvalues" e

            Some(Bitmask(nameOf (), bits, (underlying = "VkFlags64")))
        | "handle" ->
            let macro = e.Element(XName.Get "type").Value
            Some(Handle(nameOf (), (macro = "VK_DEFINE_HANDLE")))
        | "enum" -> Some(EnumType(nameOf ()))
        | "struct"
        | "union" ->
            Some(
                Struct
                    {
                        Name = nameOf ()
                        IsUnion = (category = "union")
                        Members = children "member" e |> List.map parseDecl
                        Extends = splitList (attr "structextends" e)
                        ReturnedOnly = (attr "returnedonly" e = Some "true")
                        AllowDuplicate = (attr "allowduplicate" e = Some "true")
                    }
            )
        | "funcpointer" ->
            let proto = parseDecl (e.Element(XName.Get "proto"))
            Some(FuncPointer(proto.Name, proto, children "param" e |> List.map parseDecl))
        | "" ->
            // <type name="..." requires="..."/>: a platform type (uint32_t, HANDLE and so on).
            match attr "name" e with
            | Some n -> Some(Other(n, "external"))
            | None -> None
        | c -> Some(Other(nameOf (), c))

let private typeName =
    function
    | Basetype(n, _)
    | Bitmask(n, _, _)
    | Handle(n, _)
    | EnumType n
    | FuncPointer(n, _, _)
    | TypeAlias(n, _)
    | Other(n, _) -> n
    | Struct s -> s.Name

/// Reads vk.xml.
let load (path: string) : Registry =
    let doc = XDocument.Load(path, LoadOptions.PreserveWhitespace)
    let root = doc.Root
    let types = Dictionary<string, TypeDef>(StringComparer.Ordinal)
    let mutable headerVersion = 0

    for t in children "type" (root.Element(XName.Get "types")) do
        match parseType t with
        | Some(Other("VK_HEADER_VERSION", "define")) ->
            let m = Regex.Match(t.Value, @"VK_HEADER_VERSION\s+(\d+)")

            if m.Success then
                headerVersion <- Int32.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)
        | Some def ->
            let n = typeName def

            if n <> "" && not (types.ContainsKey n) then
                types[n] <- def
        | None -> ()

    let enums = Dictionary<string, EnumsBlock>(StringComparer.Ordinal)
    let constants = Dictionary<string, ApiConstant>(StringComparer.Ordinal)

    for es in children "enums" root do
        let name = (attr "name" es).Value

        match attr "type" es with
        | Some "constants" ->
            for c in children "enum" es do
                match attr "alias" c with
                | Some _ -> ()
                | None ->
                    let n = (attr "name" c).Value

                    constants[n] <-
                        {
                            Name = n
                            CType = (attr "type" c).Value
                            Value = (attr "value" c).Value
                        }
        | Some kind when kind = "enum" || kind = "bitmask" ->
            let members =
                [
                    for c in children "enum" es do
                        let n = (attr "name" c).Value

                        match attr "alias" c, attr "bitpos" c, attr "value" c with
                        | Some a, _, _ -> { Name = n; Value = AliasOf a }
                        | _, Some b, _ ->
                            {
                                Name = n
                                Value = Num(1L <<< Int32.Parse(b, CultureInfo.InvariantCulture))
                            }
                        | _, _, Some v -> { Name = n; Value = Num(parseNumber v) }
                        | _ -> failwithf "enum %s.%s has no value" name n
                ]

            enums[name] <-
                {
                    Name = name
                    IsBitmask = (kind = "bitmask")
                    Bits64 = (attr "bitwidth" es = Some "64")
                    Members = members
                }
        | _ -> ()

    let commands = Dictionary<string, Command>(StringComparer.Ordinal)
    let aliases = Dictionary<string, string>(StringComparer.Ordinal)

    for c in children "command" (root.Element(XName.Get "commands")) do
        match attr "alias" c with
        | Some target -> aliases[(attr "name" c).Value] <- target
        | None ->
            let proto = parseDecl (c.Element(XName.Get "proto"))

            commands[proto.Name] <-
                {
                    Name = proto.Name
                    Return = proto
                    Params = children "param" c |> List.map parseDecl
                    SuccessCodes = splitList (attr "successcodes" c)
                    ErrorCodes = splitList (attr "errorcodes" c)
                }

    let features =
        [
            for f in children "feature" root do
                let v = (attr "number" f).Value.Split('.')

                {
                    Name = (attr "name" f).Value
                    Version =
                        Int32.Parse(v[0], CultureInfo.InvariantCulture), Int32.Parse(v[1], CultureInfo.InvariantCulture)
                    Requires = parseRequires None f
                }
        ]

    let extensions = Dictionary<string, Extension>(StringComparer.Ordinal)

    for x in (root.Element(XName.Get "extensions")).Elements(XName.Get "extension") do
        let supported = splitList (attr "supported" x)

        if supported |> List.contains "vulkan" then
            let number = Int32.Parse((attr "number" x).Value, CultureInfo.InvariantCulture)
            let name = (attr "name" x).Value

            extensions[name] <-
                {
                    Name = name
                    Number = number
                    Kind = defaultArg (attr "type" x) ""
                    Requires = parseRequires (Some number) x
                }

    if headerVersion = 0 then
        failwith "vk.xml has no VK_HEADER_VERSION define"

    {
        Types = types
        Enums = enums
        Constants = constants
        Commands = commands
        CommandAliases = aliases
        Features = features
        Extensions = extensions
        HeaderVersion = headerVersion
    }
