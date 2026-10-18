/// The ABI oracle (VulkanInteropApi contract): dumps the runtime layout of everything the interop binds as C and C++
/// static assertions, then compiles them with cl.exe against Vulkan-Headers 1.4.365 and runs the result.
///
/// - Struct and union sizes, alignments, field offsets, field sizes and field kinds come from the loaded types
///   (RuntimeHelpers.SizeOf, Marshal.SizeOf, Marshal.OffsetOf, a measured alignment, the field's type), not from the
///   generator, so a wrong mapping fails here even when alignment keeps every offset and the size right.
/// - Enum values (32-bit in C, 64-bit flag bits in C++), handle sizes, API and extension constants.
/// - Signatures: each command and funcpointer delegate's Invoke against vulkan_core.h's PFN_* typedef (kind and size
///   of the return value and of every parameter).
/// - pNext structure types: every struct's sType and every structextends relation of the runtime metadata, against
///   vulkan.hpp (vk::T::structureType, vk::T::NativeType and vk::StructExtends), which Khronos generates from the
///   same registry independently of vkgen.
///
/// The dumped files are committed under tools/abi-oracle/vulkan; the cpu tier checks they are current.
module Lrm.Gpu.Vulkan.Interop.Tests.AbiOracle

open System
open System.Diagnostics
open System.IO
open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Text
open System.Text.Json
open Lrm.Gpu.Vulkan.Interop

let interop = typeof<VkInstance>.Assembly

/// File names of the generated oracle sources, in tools/abi-oracle/vulkan.
let cFile = "vk_abi.g.c"
let cppFile = "vk_sig.g.cpp"

let private attr<'A when 'A :> Attribute> (t: Type) : 'A option =
    match t.GetCustomAttribute(typeof<'A>, false) with
    | null -> None
    | a -> Some(a :?> 'A)

/// A struct whose only job is to measure the runtime alignment of 'T: the offset of `value` after one byte.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type AlignProbe<'T when 'T: struct> =
    val mutable pad: byte
    val mutable value: 'T

type private Marker =
    static member Align<'T when 'T: struct>() : int =
        let mutable p = Unchecked.defaultof<AlignProbe<'T>>
        int (Unsafe.ByteOffset(&Unsafe.As<AlignProbe<'T>, byte>(&p), &Unsafe.As<'T, byte>(&p.value)))

/// The alignment the runtime gives a value type.
let alignOf (t: Type) : int =
    typeof<Marker>
        .GetMethod("Align", BindingFlags.Static ||| BindingFlags.Public ||| BindingFlags.NonPublic)
        .MakeGenericMethod([| t |])
        .Invoke(null, [||])
    :?> int

/// The ABI code of a .NET type in a field or signature: kind and size (see lrm_vk_sig.hpp).
let rec code (t: Type) : char * int =
    if t = typeof<Void> then
        'v', 0
    elif t = typeof<nativeint> || t.IsPointer then
        'p', 8
    elif t = typeof<unativeint> then
        'i', 8
    elif t.IsEnum then
        code (Enum.GetUnderlyingType t)
    elif t = typeof<float32> then
        'f', 4
    elif t = typeof<float> then
        'f', 8
    elif t.IsPrimitive then
        'i', Marshal.SizeOf t
    elif (attr<VkHandleAttribute> t).IsSome then
        'h', 8
    elif (attr<VkArrayAttribute> t).IsSome then
        'a', RuntimeHelpers.SizeOf t.TypeHandle
    elif t.IsValueType then
        's', RuntimeHelpers.SizeOf t.TypeHandle
    else
        failwithf "%s cannot appear in a blittable signature" t.FullName

type FieldInfo =
    {
        Name: string
        Offset: int
        Size: int
        Kind: char
        /// For an inline array: (element kind, element size, count).
        Array: (char * int * int) option
    }

type StructInfo =
    {
        CName: string
        Type: Type
        IsUnion: bool
        ManagedSize: int
        MarshalSize: int
        Align: int
        Blittable: bool
        Fields: FieldInfo list
        /// The value of the StructureType static property, for structs with an sType.
        SType: int option
    }

let private fieldsOf (t: Type) =
    t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
    |> Array.map (fun f ->
        let kind, _ = code f.FieldType

        let array =
            match attr<VkArrayAttribute> f.FieldType with
            | Some a ->
                let e = f.FieldType.GetField("e0").FieldType
                let ek, es = code e
                Some(ek, es, a.Count)
            | None -> None

        {
            Name = f.Name
            Offset = int (Marshal.OffsetOf(t, f.Name))
            Size = RuntimeHelpers.SizeOf f.FieldType.TypeHandle
            Kind = kind
            Array = array
        })
    |> List.ofArray

/// Every bound struct and union, sorted by C name.
let structs () : StructInfo list =
    interop.GetTypes()
    |> Array.choose (fun t ->
        let name, isUnion =
            match attr<VkStructAttribute> t, attr<VkUnionAttribute> t with
            | Some a, _ -> Some a.Name, false
            | _, Some u -> Some u.Name, true
            | _ -> None, false

        name
        |> Option.map (fun n ->
            let blittable =
                try
                    let h = GCHandle.Alloc(Activator.CreateInstance t, GCHandleType.Pinned)
                    h.Free()
                    true
                with _ ->
                    false

            let sType =
                match t.GetProperty("StructureType", BindingFlags.Static ||| BindingFlags.Public) with
                | null -> None
                | p -> Some(Convert.ToInt32(p.GetValue null))

            {
                CName = n
                Type = t
                IsUnion = isUnion
                ManagedSize = RuntimeHelpers.SizeOf t.TypeHandle
                MarshalSize = Marshal.SizeOf t
                Align = alignOf t
                Blittable = blittable
                Fields = fieldsOf t
                SType = sType
            }))
    |> Array.sortBy _.CName
    |> List.ofArray

type EnumInfo =
    {
        CName: string
        Type: Type
        Size: int
        IsBitmask: bool
        /// (C name, value as the unsigned bits of the underlying type, or the signed value of a 32-bit enum).
        Members: (string * int64) list
    }

let enums () : EnumInfo list =
    interop.GetTypes()
    |> Array.choose (fun t ->
        attr<VkEnumAttribute> t
        |> Option.map (fun a ->
            let u = Enum.GetUnderlyingType t

            {
                CName = a.Name
                Type = t
                Size = Marshal.SizeOf u
                IsBitmask = t.IsDefined(typeof<FlagsAttribute>, false)
                Members =
                    t.GetFields(BindingFlags.Public ||| BindingFlags.Static)
                    |> Array.filter _.IsLiteral
                    |> Array.map (fun f ->
                        let raw = f.GetRawConstantValue()

                        let v =
                            match raw with
                            | :? uint64 as x -> int64 x
                            | :? uint32 as x -> int64 x
                            | x -> Convert.ToInt64 x

                        f.Name, v)
                    |> List.ofArray
            }))
    |> Array.sortBy _.CName
    |> List.ofArray

let handles () : (string * int) list =
    interop.GetTypes()
    |> Array.choose (fun t ->
        attr<VkHandleAttribute> t
        |> Option.map (fun a -> a.Name, RuntimeHelpers.SizeOf t.TypeHandle))
    |> Array.sortBy fst
    |> List.ofArray

/// (C typedef name, return code, parameter codes) for every command and funcpointer delegate.
let delegates () =
    interop.GetTypes()
    |> Array.choose (fun t ->
        let name =
            match attr<VkCommandAttribute> t, attr<VkFuncPointerAttribute> t with
            | Some c, _ -> Some("PFN_" + c.Name)
            | _, Some f -> Some f.Name
            | _ -> None

        name
        |> Option.map (fun n ->
            let invoke = t.GetMethod "Invoke"
            n, code invoke.ReturnType, invoke.GetParameters() |> Array.map (fun p -> code p.ParameterType)))
    |> Array.sortBy (fun (n, _, _) -> n)
    |> List.ofArray

/// (name, value) of every VkConst literal.
let constants () : (string * obj) list =
    match interop.GetType "Lrm.Gpu.Vulkan.Interop.VkConst" with
    | null -> failwith "module VkConst not found"
    | t ->
        t.GetFields(BindingFlags.Public ||| BindingFlags.Static)
        |> Array.filter _.IsLiteral
        |> Array.map (fun f -> f.Name, f.GetRawConstantValue())
        |> Array.sortBy fst
        |> List.ofArray

let private header =
    "/* <auto-generated> ABI oracle for Lrm.Gpu.Vulkan.Interop: the runtime layout of every bound item, dumped by\n"
    + "   Lrm.Gpu.Vulkan.Interop.Tests (abi-oracle) and checked by cl.exe against Vulkan-Headers.\n"
    + "   Do not edit; regenerate with: pwsh tools/abi-oracle/vulkan/run.ps1 -Update */\n"

let private cppHeader =
    "// <auto-generated> ABI oracle for Lrm.Gpu.Vulkan.Interop: signatures, field kinds, 64-bit flags, strings, sTypes\n"
    + "// and pNext extensions, dumped by Lrm.Gpu.Vulkan.Interop.Tests (abi-oracle) and checked by cl.exe against\n"
    + "// Vulkan-Headers (vulkan_core.h and vulkan.hpp). Do not edit; regenerate with: pwsh tools/abi-oracle/vulkan/run.ps1 -Update\n"

/// The C source: handles, layouts, 32-bit enum values and integer constants as C11 static assertions.
let cSource () : string * int =
    let sb = StringBuilder()
    let line (s: string) = sb.Append(s).Append('\n') |> ignore
    let mutable checks = 0

    let check (s: string) =
        checks <- checks + 1
        line s

    sb.Append header |> ignore
    line "#include \"lrm_vk_abi.h\""
    line ""
    check (sprintf "LRM_HEADER_VERSION(%d);" VkConst.VK_HEADER_VERSION)
    line ""
    let hs = handles ()
    line (sprintf "/* Handles: %d. */" hs.Length)

    for (n, s) in hs do
        check (sprintf "LRM_HANDLE(%s, %d);" n s)

    line ""
    let ss = structs ()
    line (sprintf "/* Struct and union layouts: %d. */" ss.Length)

    for s in ss do
        check (sprintf "LRM_SIZE(%s, %d);" s.CName s.ManagedSize)
        check (sprintf "LRM_ALIGN(%s, %d);" s.CName s.Align)

        for f in s.Fields do
            check (sprintf "LRM_OFFSET(%s, %s, %d);" s.CName f.Name f.Offset)
            check (sprintf "LRM_FIELD_SIZE(%s, %s, %d);" s.CName f.Name f.Size)

    line ""
    let es = enums ()

    line (
        sprintf
            "/* Enums: %d (64-bit flag bits are checked in C++: vulkan_core.h makes them static const). */"
            es.Length
    )

    for e in es do
        check (sprintf "LRM_ENUM_SIZE(%s, %d);" e.CName e.Size)

        if e.Size = 4 then
            for (m, v) in e.Members do
                if e.IsBitmask then
                    check (sprintf "LRM_FLAG(%s, %d);" m (uint32 v))
                else
                    check (sprintf "LRM_ENUM(%s, %d);" m v)

    line ""
    line "/* Integer API and extension constants. */"

    for (n, v) in constants () do
        match v with
        | :? uint32 as x -> check (sprintf "LRM_CONST(%s, %d);" n x)
        | :? uint64 as x -> check (sprintf "LRM_CONST(%s, %d);" n x)
        | :? int as x when n <> "VK_HEADER_VERSION" -> check (sprintf "LRM_CONST(%s, %d);" n x)
        | _ -> ()

    line ""
    line (sprintf "const int lrm_static_checks_c = %d;" checks)
    sb.ToString(), checks

/// The C++ source: delegate signatures, field kinds, 64-bit flags, strings, sTypes, pNext extensions and aliases.
let cppSource () : string * int =
    let sb = StringBuilder()
    let line (s: string) = sb.Append(s).Append('\n') |> ignore
    let mutable checks = 0

    let check (s: string) =
        checks <- checks + 1
        line s

    sb.Append cppHeader |> ignore
    line "#include \"lrm_vk_sig.hpp\""
    line ""
    let ss = structs ()
    line (sprintf "// Struct fields: the kind and size of every field of %d structs and unions." ss.Length)

    for s in ss do
        for f in s.Fields do
            match f.Array with
            | Some(ek, es, n) -> check (sprintf "LRM_FIELD_ARRAY(%s, %s, '%c', %d, %d);" s.CName f.Name ek es n)
            | None -> check (sprintf "LRM_FIELD_KIND(%s, %s, '%c', %d);" s.CName f.Name f.Kind f.Size)

    line ""
    line "// sType of every struct that has one (vulkan.hpp)."

    for s in ss do
        match s.SType with
        | Some v -> check (sprintf "LRM_STYPE(%s, %s, %d);" s.CName (s.CName.Substring 2) v)
        | None -> ()

    line ""

    line
        "// pNext extensions the runtime metadata (VkMeta.Structs) claims, for bound parents (vulkan.hpp StructExtends)."

    let bound = ss |> List.map _.CName |> set

    for m in VkMeta.Structs |> List.sortBy _.Name do
        for p in m.Extends |> List.sort do
            if bound.Contains p then
                check (sprintf "LRM_EXTENDS(%s, %s);" (m.Name.Substring 2) (p.Substring 2))

    line ""
    line "// Type aliases."

    for (alias, t) in VkMeta.Aliases do
        match attr<VkStructAttribute> t, attr<VkUnionAttribute> t, attr<VkEnumAttribute> t with
        | Some a, _, _ -> check (sprintf "LRM_ALIAS(%s, %s);" alias a.Name)
        | _, Some u, _ -> check (sprintf "LRM_ALIAS(%s, %s);" alias u.Name)
        | _, _, Some e -> check (sprintf "LRM_ALIAS(%s, %s);" alias e.Name)
        | _ -> ()

    line ""
    line "// 64-bit flag bits."

    for e in enums () do
        if e.Size = 8 then
            for (m, v) in e.Members do
                check (sprintf "LRM_FLAG64(%s, %d);" m (uint64 v))

    line ""
    line "// Extension names and other non-integer constants."

    for (n, v) in constants () do
        match v with
        | :? string as s -> check (sprintf "LRM_STRING(%s, \"%s\");" n s)
        | :? float32 as f ->
            let text = f.ToString("R", Globalization.CultureInfo.InvariantCulture)

            let literal =
                if text.Contains "." || text.Contains "E" then
                    text + "f"
                else
                    text + ".0f"

            check (sprintf "static_assert(%s == %s, \"%s\");" n literal n)
        | _ -> ()

    line ""
    let ds = delegates ()
    line (sprintf "// Command and funcpointer delegates: %d, against the PFN_* typedefs." ds.Length)

    for (f, (rk, rs), ps) in ds do
        check (sprintf "LRM_SIG_SHAPE(%s, '%c', %d, %d);" f rk rs ps.Length)

        ps
        |> Array.iteri (fun i (k, s) -> check (sprintf "LRM_SIG_ARG(%s, %d, '%c', %d);" f i k s))

    line ""
    line (sprintf "extern \"C\" int lrm_static_checks_cpp(void) { return %d; }" checks)
    sb.ToString(), checks

/// tools/abi-oracle/vulkan in the repo.
let oracleDir (repo: string) =
    Path.Combine(repo, "tools", "abi-oracle", "vulkan")

/// Writes both sources into `dir`; returns the paths.
let dump (dir: string) =
    Directory.CreateDirectory dir |> ignore
    let cPath = Path.Combine(dir, cFile)
    let cppPath = Path.Combine(dir, cppFile)
    File.WriteAllText(cPath, fst (cSource ()))
    File.WriteAllText(cppPath, fst (cppSource ()))
    [ cPath; cppPath ]

/// The committed sources that differ from the runtime dump (empty when current).
let stale (repo: string) =
    let dir = oracleDir repo

    [
        for (name, text) in [ cFile, fst (cSource ()); cppFile, fst (cppSource ()) ] do
            let path = Path.Combine(dir, name)

            if not (File.Exists path) || File.ReadAllText path <> text then
                name
    ]

/// The pinned Vulkan-Headers include directory (the vkheaders entry of tools/bootstrap/toolchain.json), after
/// checking vulkan_core.h against the install record bootstrap wrote. Error when the headers are missing or differ.
let headers (repo: string) : Result<string, string> =
    let toolchain = Path.Combine(repo, "tools", "bootstrap", "toolchain.json")
    use doc = JsonDocument.Parse(File.ReadAllText toolchain)
    let e = doc.RootElement.GetProperty "vkheaders"
    let version = e.GetProperty("version").GetString()
    let root = Lrm.Core.LrmPaths.under (e.GetProperty("install").GetString())
    let record = Path.Combine(root, ".lrm-install.json")
    let core = Path.Combine(root, "include", "vulkan", "vulkan_core.h")

    if not (File.Exists record) || not (File.Exists core) then
        Error(
            sprintf "Vulkan-Headers %s are not installed at %s (run: pwsh tools/bootstrap/bootstrap.ps1)" version root
        )
    else
        use r = JsonDocument.Parse(File.ReadAllText record)

        let expected =
            r.RootElement.GetProperty("files").GetProperty("include/vulkan/vulkan_core.h").GetString()

        let actual =
            Convert.ToHexString(Security.Cryptography.SHA256.HashData(File.ReadAllBytes core)).ToLowerInvariant()

        if actual <> expected then
            Error(sprintf "%s has SHA-256 %s, but bootstrap installed %s" core actual expected)
        else
            Ok(Path.Combine(root, "include"))

type Toolchain = { Vswhere: string; Vcvars: string }

/// VS 2022 Build Tools (or any VS with the x64 C++ tools) through vswhere.
let findToolchain () : Toolchain option =
    let pf86 = Environment.GetFolderPath Environment.SpecialFolder.ProgramFilesX86

    let vswhere =
        Path.Combine(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")

    if not (File.Exists vswhere) then
        None
    else
        let psi =
            ProcessStartInfo(
                vswhere,
                "-latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath",
                RedirectStandardOutput = true,
                UseShellExecute = false
            )

        use p = Process.Start psi
        let out = p.StandardOutput.ReadToEnd().Trim()
        p.WaitForExit()
        let vcvars = Path.Combine(out, "VC", "Auxiliary", "Build", "vcvars64.bat")

        if out = "" || not (File.Exists vcvars) then
            None
        else
            Some { Vswhere = vswhere; Vcvars = vcvars }

type OracleRun =
    {
        ExitCode: int
        Output: string
        /// The oracle's own summary line, when it ran.
        Summary: string option
        /// Failed assertions reported by the compiler.
        Failures: string list
        StaticChecks: int
    }

/// Compiles the oracle sources in `srcDir` with cl.exe against `include` and runs the result. Objects and the exe go
/// to `outDir`.
let compileAndRunIn (tc: Toolchain) (include': string) (srcDir: string) (outDir: string) : OracleRun =
    let outDir = Path.GetFullPath outDir
    Directory.CreateDirectory outDir |> ignore
    let src = Path.GetFullPath srcDir
    let q (p: string) = "\"" + p + "\""
    let exe = "vk_abi_oracle.exe"

    // cl runs in outDir, so objects land there without a /Fo path (a quoted "dir\" would escape its quote).
    let common =
        sprintf "/nologo /W4 /WX /external:anglebrackets /external:W0 /c /I%s /I%s" (q src) (q include')

    let lines =
        [
            "@echo off"
            // vcvars64.bat looks for vswhere on PATH.
            sprintf "set \"PATH=%s;%%PATH%%\"" (Path.GetDirectoryName tc.Vswhere)
            sprintf "call %s >nul || exit /b 90" (q tc.Vcvars)
            sprintf "cd /d %s || exit /b 90" (q outDir)
            // Both translation units compile before either failure stops the script, so one run reports every
            // failed assertion.
            sprintf
                "cl %s /std:c11 /TC %s %s"
                common
                (q (Path.Combine(src, "oracle_main.c")))
                (q (Path.Combine(src, cFile)))
            "set LRM_C=%ERRORLEVEL%"
            sprintf "cl %s /std:c++17 /EHsc /TP %s" common (q (Path.Combine(src, cppFile)))
            "set LRM_CPP=%ERRORLEVEL%"
            "if not %LRM_C%==0 exit /b 91"
            "if not %LRM_CPP%==0 exit /b 92"
            sprintf "link /nologo /OUT:%s oracle_main.obj vk_abi.g.obj vk_sig.g.obj || exit /b 93" exe
            ".\\" + exe
        ]

    let script = Path.Combine(outDir, "oracle.cmd")
    File.WriteAllText(script, String.concat "\r\n" lines + "\r\n")
    let psi = ProcessStartInfo("cmd.exe", "/d /c " + q script, UseShellExecute = false)
    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    use p = Process.Start psi
    let stdout = p.StandardOutput.ReadToEndAsync()
    let stderr = p.StandardError.ReadToEndAsync()
    p.WaitForExit()
    let output = stdout.Result + stderr.Result

    let outLines =
        output.Replace("\r\n", "\n").Split('\n')
        |> Array.map _.TrimEnd()
        |> List.ofArray

    let failures =
        outLines
        |> List.filter (fun l ->
            l.Contains "static_assert failed"
            || l.Contains "static assertion failed"
            || l.Contains "error C2338"
            || (l.Contains " error " && l.Contains ": error"))

    {
        ExitCode = p.ExitCode
        Output = output
        Summary = outLines |> List.tryFind (fun l -> l.StartsWith "vulkan ABI oracle:")
        Failures = failures
        StaticChecks = snd (cSource ()) + snd (cppSource ())
    }

/// Copies the committed oracle sources to `dir`, applying (file, old, new) replacements to the generated files; each
/// replacement must apply exactly once. The oracle's self-test compiles such mutants and expects them to fail.
let mutant (repo: string) (dir: string) (edits: (string * string * string) list) =
    Directory.CreateDirectory dir |> ignore

    for f in Directory.GetFiles(oracleDir repo) do
        let name = Path.GetFileName f

        if
            name.EndsWith ".c"
            || name.EndsWith ".cpp"
            || name.EndsWith ".h"
            || name.EndsWith ".hpp"
        then
            let mutable text = File.ReadAllText f

            for (file, o, n) in edits do
                if file = name then
                    let at = text.IndexOf(o, StringComparison.Ordinal)

                    if at < 0 || text.IndexOf(o, at + 1, StringComparison.Ordinal) >= 0 then
                        failwithf "mutant: '%s' must occur exactly once in %s" o name

                    text <- text.Replace(o, n)

            File.WriteAllText(Path.Combine(dir, name), text)
