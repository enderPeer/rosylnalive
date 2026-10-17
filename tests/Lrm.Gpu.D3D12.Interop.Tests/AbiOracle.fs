/// The ABI oracle (D3D12ComIdiom contract): dumps the runtime layout of everything the interop binds as C and C++
/// static assertions, then compiles them with cl.exe against the Windows SDK headers and runs the result.
///
/// - Struct sizes and field offsets come from the loaded types (Marshal.SizeOf, RuntimeHelpers.SizeOf,
///   Marshal.OffsetOf), not from the generator, so a wrong mapping fails here.
/// - Vtable slots: offsetof(IVtbl, Method) == slot * sizeof(void*) for every Slots constant, sizeof(IVtbl) == Count.
/// - Signatures: each unmanaged delegate's Invoke and each P/Invoke against the C declaration (kind and size of the
///   return value and of every parameter).
/// - Enum values, HRESULT values, and the IIDs (compared at run time).
///
/// The dumped files are committed under tools/abi-oracle/d3d12; the cpu tier checks they are current.
module Lrm.Gpu.D3D12.Interop.Tests.AbiOracle

open System
open System.Diagnostics
open System.IO
open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Text
open Lrm.Gpu.D3D12.Interop

let interop = typeof<IUnknown>.Assembly

/// File names of the generated oracle sources, in tools/abi-oracle/d3d12.
let cFile = "d3d12_abi.g.c"
let cppFile = "d3d12_sig.g.cpp"

type StructInfo =
    {
        CName: string
        Type: Type
        ManagedSize: int
        MarshalSize: int
        Blittable: bool
        /// Field name and Marshal.OffsetOf, in declaration order.
        Fields: (string * int) list
    }

let private attr<'A when 'A :> Attribute> (t: Type) : 'A option =
    match t.GetCustomAttribute(typeof<'A>, false) with
    | null -> None
    | a -> Some(a :?> 'A)

let structs () : StructInfo list =
    interop.GetTypes()
    |> Array.choose (fun t ->
        match attr<CStructAttribute> t with
        | None -> None
        | Some a ->
            let blittable =
                try
                    let h = GCHandle.Alloc(Activator.CreateInstance t, GCHandleType.Pinned)
                    h.Free()
                    true
                with _ ->
                    false

            Some
                {
                    CName = a.Name
                    Type = t
                    ManagedSize = RuntimeHelpers.SizeOf t.TypeHandle
                    MarshalSize = Marshal.SizeOf t
                    Blittable = blittable
                    Fields =
                        t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
                        |> Array.map (fun f -> f.Name, int (Marshal.OffsetOf(t, f.Name)))
                        |> List.ofArray
                })
    |> Array.sortBy _.CName
    |> List.ofArray

/// The C name of an F# enum case: the prefix plus the case name, minus the underscore added to a digit-leading name.
let cEnumName (prefix: string) (case: string) =
    if case.Length > 1 && case[0] = '_' && Char.IsDigit case[1] then
        prefix + case.Substring 1
    else
        prefix + case

let enums () : (string * Type * (string * int64) list) list =
    interop.GetTypes()
    |> Array.choose (fun t ->
        match attr<CEnumAttribute> t with
        | None -> None
        | Some a ->
            let members =
                t.GetFields(BindingFlags.Public ||| BindingFlags.Static)
                |> Array.filter _.IsLiteral
                |> Array.map (fun f -> cEnumName a.Prefix f.Name, Convert.ToInt64(f.GetRawConstantValue()))
                |> List.ofArray

            Some(a.Name, t, members))
    |> Array.sortBy (fun (n, _, _) -> n)
    |> List.ofArray

let private nested (moduleName: string) =
    match interop.GetType("Lrm.Gpu.D3D12.Interop." + moduleName) with
    | null -> failwithf "module %s not found" moduleName
    | t -> t.GetNestedTypes() |> Array.sortBy _.Name |> List.ofArray

/// (interface, [(method, slot)], count) from the generated Slots module.
let slots () =
    nested "Slots"
    |> List.map (fun t ->
        let lits =
            t.GetFields(BindingFlags.Public ||| BindingFlags.Static)
            |> Array.filter _.IsLiteral
            |> Array.map (fun f -> f.Name, f.GetRawConstantValue() :?> int)
            |> List.ofArray

        let count = lits |> List.find (fun (n, _) -> n = "Count") |> snd
        t.Name, lits |> List.filter (fun (n, _) -> n <> "Count") |> List.sortBy snd, count)

/// The ABI code of a .NET type in a signature: kind char and size.
let rec code (t: Type) : char * int =
    if t = typeof<Void> then
        'v', 0
    elif t = typeof<nativeint> || t = typeof<unativeint> || t.IsPointer then
        'w', 8
    elif t.IsEnum then
        code (Enum.GetUnderlyingType t)
    elif t = typeof<float32> then
        'f', 4
    elif t = typeof<float> then
        'f', 8
    elif t.IsPrimitive then
        'i', Marshal.SizeOf t
    elif t.IsValueType then
        's', Marshal.SizeOf t
    else
        failwithf "%s cannot appear in a blittable signature" t.FullName

/// (interface, method, return code, parameter codes) for every generated delegate.
let delegates () =
    nested "Fn"
    |> List.collect (fun m ->
        m.GetNestedTypes()
        |> Array.filter (fun t -> typeof<Delegate>.IsAssignableFrom t)
        |> Array.sortBy _.Name
        |> Array.map (fun d ->
            let invoke = d.GetMethod "Invoke"

            m.Name,
            d.Name,
            code invoke.ReturnType,
            invoke.GetParameters() |> Array.map (fun p -> code p.ParameterType))
        |> List.ofArray)

/// (export, return code, parameter codes) for every P/Invoke in the Native module.
let exports () =
    match interop.GetType "Lrm.Gpu.D3D12.Interop.Native" with
    | null -> failwith "module Native not found"
    | t ->
        t.GetMethods(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Static)
        |> Array.filter (fun m -> m.Attributes.HasFlag MethodAttributes.PinvokeImpl)
        |> Array.sortBy _.Name
        |> Array.map (fun m ->
            m.Name, code m.ReturnType, m.GetParameters() |> Array.map (fun p -> code p.ParameterType))
        |> List.ofArray

/// (interface, IID) from the generated Iids module.
let iids () =
    match interop.GetType "Lrm.Gpu.D3D12.Interop.Iids" with
    | null -> failwith "module Iids not found"
    | t ->
        t.GetProperties(BindingFlags.Public ||| BindingFlags.Static)
        |> Array.filter (fun p -> p.PropertyType = typeof<Guid>)
        |> Array.map (fun p -> p.Name, p.GetValue null :?> Guid)
        |> Array.sortBy fst
        |> List.ofArray

let private header =
    "/* <auto-generated> ABI oracle for Lrm.Gpu.D3D12.Interop: the runtime layout of every bound item, dumped by\n"
    + "   Lrm.Gpu.D3D12.Interop.Tests (abi-oracle) and checked by cl.exe against the Windows SDK headers.\n"
    + "   Do not edit; regenerate with: pwsh tools/abi-oracle/d3d12/run.ps1 -Update */\n"

let private cppHeader =
    "// <auto-generated> ABI oracle for Lrm.Gpu.D3D12.Interop: the call signature of every unmanaged delegate and
"
    + "// P/Invoke, dumped by Lrm.Gpu.D3D12.Interop.Tests (abi-oracle) and checked by cl.exe against the SDK headers.
"
    + "// Do not edit; regenerate with: pwsh tools/abi-oracle/d3d12/run.ps1 -Update
"

let private guidInit (g: Guid) =
    let b = g.ToByteArray()

    sprintf
        "{ 0x%08xu, 0x%04x, 0x%04x, { 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x } }"
        (BitConverter.ToUInt32(b, 0))
        (BitConverter.ToUInt16(b, 4))
        (BitConverter.ToUInt16(b, 6))
        b[8]
        b[9]
        b[10]
        b[11]
        b[12]
        b[13]
        b[14]
        b[15]

/// The C source: layouts, enums, slots and HRESULTs as static assertions, plus the IID table.
let cSource () : string * int =
    let sb = StringBuilder()
    let line (s: string) = sb.Append(s).Append('\n') |> ignore
    let mutable checks = 0

    let check (s: string) =
        checks <- checks + 1
        line s

    sb.Append header |> ignore
    line "#define LRM_INSTANTIATE_IIDS"
    line "#include \"lrm_abi.h\""
    line ""
    let ss = structs ()
    line (sprintf "/* Struct layouts: %d structs. */" ss.Length)

    for s in ss do
        check (sprintf "LRM_SIZE(%s, %d);" s.CName s.ManagedSize)

        for (f, off) in s.Fields do
            check (sprintf "LRM_OFFSET(%s, %s, %d);" s.CName f off)

    line ""
    let es = enums ()
    line (sprintf "/* Enum values: %d enums. */" es.Length)

    for (_, _, members) in es do
        for (m, v) in members do
            check (sprintf "LRM_ENUM(%s, %dLL);" m v)

    line ""
    let sl = slots ()
    line (sprintf "/* Vtable slots: %d interfaces. */" sl.Length)

    for (i, ms, count) in sl do
        for (m, n) in ms do
            check (sprintf "LRM_SLOT(%s, %s, %d);" i m n)

        check (sprintf "LRM_SLOTS(%s, %d);" i count)

    line ""
    line "/* HRESULTs. */"

    for (n, v) in HResult.known do
        check (sprintf "LRM_HRESULT(%s, 0x%08X);" n v)

    line ""
    let ids = iids ()
    line "/* IIDs, compared at run time by oracle_main.c. */"
    line "const lrm_iid_check lrm_iids[] = {"

    for (n, g) in ids do
        line (sprintf "    { \"%s\", &IID_%s, %s }," n n (guidInit g))

    line "};"
    line (sprintf "const int lrm_iid_count = %d;" ids.Length)
    line (sprintf "const int lrm_static_checks_c = %d;" (checks + 1))
    check (sprintf "LRM_STATIC_ASSERT(sizeof(lrm_iids) / sizeof(lrm_iids[0]) == %d, \"IID table\");" ids.Length)
    sb.ToString(), checks

/// The C++ source: every delegate and P/Invoke signature against the SDK declaration.
let cppSource () : string * int =
    let sb = StringBuilder()
    let line (s: string) = sb.Append(s).Append('\n') |> ignore
    let mutable checks = 0

    let check (s: string) =
        checks <- checks + 1
        line s

    sb.Append(cppHeader) |> ignore
    line "#include \"lrm_abi_sig.hpp\""
    line ""

    let shape (f: string) (name: string) (rk: char, rs: int) (ps: (char * int)[]) =
        check (sprintf "LRM_SIG_SHAPE(%s, \"%s\", '%c', %d, %d);" f name rk rs ps.Length)

        ps
        |> Array.iteri (fun i (k, s) -> check (sprintf "LRM_SIG_ARG(%s, \"%s\", %d, '%c', %d);" f name i k s))

    let ds = delegates ()
    line (sprintf "// Vtable methods: %d unmanaged delegates." ds.Length)

    for (i, m, ret, ps) in ds do
        shape (sprintf "LRM_VFN(%s, %s)" i m) (i + "::" + m) ret ps

    line ""
    let xs = exports ()
    line (sprintf "// DLL exports: %d P/Invokes." xs.Length)

    for (n, ret, ps) in xs do
        shape (sprintf "LRM_XFN(%s)" n) n ret ps

    line ""
    line (sprintf "extern \"C\" int lrm_static_checks_cpp(void) { return %d; }" checks)
    sb.ToString(), checks

/// Writes both sources into `dir`; returns the paths.
let dump (dir: string) =
    Directory.CreateDirectory dir |> ignore
    let c, _ = cSource ()
    let cpp, _ = cppSource ()
    let cPath = Path.Combine(dir, cFile)
    let cppPath = Path.Combine(dir, cppFile)
    File.WriteAllText(cPath, c)
    File.WriteAllText(cppPath, cpp)
    [ cPath; cppPath ]

/// tools/abi-oracle/d3d12 in the repo.
let oracleDir (repo: string) =
    Path.Combine(repo, "tools", "abi-oracle", "d3d12")

/// The committed sources that differ from the runtime dump (empty when current).
let stale (repo: string) =
    let dir = oracleDir repo

    [
        for (name, text) in [ cFile, fst (cSource ()); cppFile, fst (cppSource ()) ] do
            let path = Path.Combine(dir, name)

            if not (File.Exists path) || File.ReadAllText path <> text then
                name
    ]

type Toolchain =
    {
        Vswhere: string
        Vcvars: string
        /// The pinned SDK version passed to vcvars64.bat, when that SDK is installed.
        SdkVersion: string option
    }

/// The pinned Windows SDK (tools/d3d12gen/bindings.json).
let pinnedSdk = "10.0.26100.0"

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
            let sdkInclude = Path.Combine(pf86, "Windows Kits", "10", "Include", pinnedSdk)

            Some
                {
                    Vswhere = vswhere
                    Vcvars = vcvars
                    SdkVersion = if Directory.Exists sdkInclude then Some pinnedSdk else None
                }

type OracleRun =
    {
        ExitCode: int
        Output: string
        /// The oracle's own summary line, when it ran.
        Summary: string option
        /// Failed assertions reported by the compiler, or failed IID checks.
        Failures: string list
        StaticChecks: int
    }

/// Compiles the oracle sources in `srcDir` with cl.exe and runs the result. Objects and the exe go to `outDir`.
let compileAndRunIn (tc: Toolchain) (srcDir: string) (outDir: string) : OracleRun =
    let outDir = Path.GetFullPath outDir
    Directory.CreateDirectory outDir |> ignore
    let src = Path.GetFullPath srcDir
    let q (p: string) = "\"" + p + "\""
    let exe = "d3d12_abi_oracle.exe"
    // cl runs in outDir, so objects land there without a /Fo path (a quoted "dir\" would escape its quote).
    let common = "/nologo /W4 /WX /external:anglebrackets /external:W0 /c /I" + q src

    let lines =
        [
            "@echo off"
            // vcvars64.bat looks for vswhere on PATH.
            sprintf "set \"PATH=%s;%%PATH%%\"" (Path.GetDirectoryName tc.Vswhere)
            sprintf "call %s %s >nul || exit /b 90" (q tc.Vcvars) (defaultArg tc.SdkVersion "")
            sprintf "cd /d %s || exit /b 90" (q outDir)
            // Both translation units compile before either failure stops the script, so one run reports every
            // failed assertion.
            sprintf
                "cl %s /std:c11 /TC %s %s"
                common
                (q (Path.Combine(src, "oracle_main.c")))
                (q (Path.Combine(src, cFile)))
            "set LRM_C=%ERRORLEVEL%"
            sprintf "cl %s /std:c++17 /EHsc /DCINTERFACE /TP %s" common (q (Path.Combine(src, cppFile)))
            "set LRM_CPP=%ERRORLEVEL%"
            "if not %LRM_C%==0 exit /b 91"
            "if not %LRM_CPP%==0 exit /b 92"
            sprintf "link /nologo /OUT:%s oracle_main.obj d3d12_abi.g.obj d3d12_sig.g.obj uuid.lib || exit /b 93" exe
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
            || l.StartsWith "FAIL "
            || (l.Contains " error " && l.Contains ": error"))

    let staticChecks = snd (cSource ()) + snd (cppSource ())

    {
        ExitCode = p.ExitCode
        Output = output
        Summary = outLines |> List.tryFind (fun l -> l.StartsWith "d3d12 ABI oracle:")
        Failures = failures
        StaticChecks = staticChecks
    }

/// Compiles the committed oracle sources (tools/abi-oracle/d3d12) and runs the result.
let compileAndRun (tc: Toolchain) (repo: string) (outDir: string) : OracleRun =
    compileAndRunIn tc (oracleDir repo) outDir

/// Copies the committed oracle sources to `dir`, applying (old, new) replacements to the generated files; each
/// replacement must apply exactly once. The oracle's self-test compiles such mutants and expects them to fail.
let mutant (repo: string) (dir: string) (edits: (string * string * string) list) =
    Directory.CreateDirectory dir |> ignore

    for f in Directory.GetFiles(oracleDir repo) do
        let name = Path.GetFileName f

        if name <> "run.ps1" && name <> "README.md" then
            let mutable text = File.ReadAllText f

            for (file, o, n) in edits do
                if file = name then
                    let at = text.IndexOf(o, StringComparison.Ordinal)

                    if at < 0 || text.IndexOf(o, at + 1, StringComparison.Ordinal) >= 0 then
                        failwithf "mutant: '%s' must occur exactly once in %s" o name

                    text <- text.Replace(o, n)

            File.WriteAllText(Path.Combine(dir, name), text)
