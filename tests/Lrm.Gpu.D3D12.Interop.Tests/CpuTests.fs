/// cpu tier: generated slots, IIDs and layouts against the probe, the ABI oracle, the COM idiom on fake objects,
/// description builders and the device-check rules. Nothing here needs a GPU.
module Lrm.Gpu.D3D12.Interop.Tests.CpuTests

#nowarn "9"
#nowarn "51"

open System
open System.IO
open System.Reflection
open System.Reflection.Metadata
open System.Reflection.PortableExecutable
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Expecto
open Lrm.Gpu.D3D12.Interop
open Lrm.Gpu.D3D12.Interop.Tests.FakeCom
open Lrm.Testing
open Microsoft.FSharp.NativeInterop

let private bytesOf<'T when 'T: unmanaged> (x: 'T) =
    let mutable v = x
    let p = NativePtr.toNativeInt &&v
    let out = Array.zeroCreate<byte> sizeof<'T>
    Marshal.Copy(p, out, 0, out.Length)
    out

let private hex (b: byte[]) =
    Convert.ToHexString(b).ToLowerInvariant()

/// Runs `f` and returns the 'E it raises; fails the test when it raises nothing.
let private raises<'E when 'E :> exn> (f: unit -> unit) : 'E =
    try
        f ()
        failtestf "expected %s" typeof<'E>.Name
    with :? 'E as e ->
        e

let private probe =
    testList
        "probe"
        [
            testCase "vtable slots equal the probe's"
            <| fun () ->
                for (i, m, generated, expected) in ProbeVectors.slots do
                    Expect.equal generated expected (sprintf "%s::%s" i m)

            testCase "IIDs equal the probe's"
            <| fun () ->
                for (i, generated, expected) in ProbeVectors.iids do
                    Expect.equal generated expected i

            testCase "struct sizes equal the probe's buffers"
            <| fun () ->
                for (name, t, expected) in ProbeVectors.sizes do
                    Expect.equal (Marshal.SizeOf t) expected name

            testCase "field offsets equal the probe's writes"
            <| fun () ->
                for (t, f, expected) in ProbeVectors.offsets do
                    Expect.equal (int (Marshal.OffsetOf(t, f))) expected (t.Name + "." + f)
        ]

let private forbiddenTypes =
    set
        [
            "System.Runtime.InteropServices.ComImportAttribute"
            "System.Runtime.InteropServices.ComWrappers"
            "System.Runtime.InteropServices.InterfaceTypeAttribute"
            "System.Runtime.InteropServices.CoClassAttribute"
            "System.Runtime.InteropServices.ClassInterfaceAttribute"
            "System.Runtime.InteropServices.Marshalling.GeneratedComInterfaceAttribute"
            "System.Runtime.InteropServices.Marshalling.GeneratedComClassAttribute"
            "System.Runtime.InteropServices.Marshalling.StrategyBasedComWrappers"
        ]

let private forbiddenMarshal =
    set
        [
            "GetObjectForIUnknown"
            "GetTypedObjectForIUnknown"
            "GetUniqueObjectForIUnknown"
            "GetComInterfaceForObject"
            "GetIUnknownForObject"
            "GetIDispatchForObject"
            "ReleaseComObject"
            "FinalReleaseComObject"
            "CreateWrapperOfType"
            "GetComObjectData"
            "SetComObjectData"
            "QueryInterface"
            "AddRef"
            "Release"
        ]

/// Type references and Marshal member references of the interop assembly, read from its metadata.
let private references () =
    use fs = File.OpenRead typeof<IUnknown>.Assembly.Location
    use pe = new PEReader(fs)
    let md = pe.GetMetadataReader()

    let typeName (h: TypeReferenceHandle) =
        let t = md.GetTypeReference h
        md.GetString t.Namespace + "." + md.GetString t.Name

    let types = md.TypeReferences |> Seq.map typeName |> Set.ofSeq

    let marshal =
        md.MemberReferences
        |> Seq.choose (fun h ->
            let m = md.GetMemberReference h

            if m.Parent.Kind = HandleKind.TypeReference then
                let parent = typeName (TypeReferenceHandle.op_Explicit m.Parent)

                if parent = "System.Runtime.InteropServices.Marshal" then
                    Some(md.GetString m.Name)
                else
                    None
            else
                None)
        |> Set.ofSeq

    let imports =
        md.TypeDefinitions
        |> Seq.map md.GetTypeDefinition
        |> Seq.filter (fun t -> t.Attributes.HasFlag TypeAttributes.Import)
        |> Seq.map (fun t -> md.GetString t.Name)
        |> List.ofSeq

    types, marshal, imports

/// The extension methods of the generated <I>Methods types.
let private wrappers () =
    typeof<IUnknown>.Assembly.GetTypes()
    |> Array.filter (fun t -> t.Name.EndsWith "Methods" && t.IsAbstract && t.IsSealed)
    |> Array.collect (fun t -> t.GetMethods(BindingFlags.Public ||| BindingFlags.Static ||| BindingFlags.DeclaredOnly))
    |> Array.filter (fun m -> m.IsDefined(typeof<ExtensionAttribute>, false))

/// The directory of the dotnet host running this exe (tools/test.ps1 runs `dotnet <exe>.dll`).
let private dotnetRoot () =
    let host = Environment.ProcessPath

    if not (isNull host) && Path.GetFileNameWithoutExtension host = "dotnet" then
        Path.GetDirectoryName host
    else
        match Env.tryGet "DOTNET_ROOT" with
        | Some r -> r
        | None -> failtestf "cannot find the dotnet root (host %s, no DOTNET_ROOT)" host

/// Compiles TailCallConsumer.fs (an embedded resource) with the pinned SDK's F# compiler, optimized and with tail
/// calls on (the Release defaults), against the interop next to this exe. Returns the assembly's path.
let private compileConsumer () : string =
    let repo = Env.repoRoot ()

    let sdk =
        use doc =
            Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(repo, "global.json")))

        doc.RootElement.GetProperty("sdk").GetProperty("version").GetString()

    let root = dotnetRoot ()
    let fsc = Path.Combine(root, "sdk", sdk, "FSharp", "fsc.dll")

    if not (File.Exists fsc) then
        failtestf "the F# compiler of SDK %s is not at %s" sdk fsc

    let dir = Path.Combine(repo, "artifacts", "d3d12-interop-tests", "tailcalls")
    Directory.CreateDirectory dir |> ignore
    let src = Path.Combine(dir, "TailCallConsumer.fs")
    let dll = Path.Combine(dir, "TailCallConsumer.dll")

    do
        use r =
            Assembly.GetExecutingAssembly().GetManifestResourceStream "TailCallConsumer.fs"

        use w = File.Create src
        r.CopyTo w

    let isManaged (path: string) =
        try
            use fs = File.OpenRead path
            use pe = new PEReader(fs)
            pe.HasMetadata
        with _ ->
            false

    let framework =
        Directory.GetFiles(RuntimeEnvironment.GetRuntimeDirectory(), "*.dll")
        |> Array.filter isManaged
        |> Array.sort

    let bin = AppContext.BaseDirectory

    let args =
        [
            "--nologo"
            "--noframework"
            "--targetprofile:netcore"
            "--target:library"
            "--optimize+"
            "--tailcalls+"
            "--deterministic+"
            "--nowarn:9,51"
            "-o:" + dll
            "-r:" + Path.Combine(bin, "FSharp.Core.dll")
            "-r:" + typeof<IUnknown>.Assembly.Location
            "-r:" + Path.Combine(bin, "Lrm.Core.dll")
            yield! framework |> Array.map (fun f -> "-r:" + f)
            src
        ]

    // One argument per line: fsc does not split response-file lines at spaces (C:\Program Files\dotnet on CI).
    let rsp = Path.Combine(dir, "fsc.rsp")
    File.WriteAllLines(rsp, args)

    let host =
        Path.Combine(
            root,
            (if OperatingSystem.IsWindows() then
                 "dotnet.exe"
             else
                 "dotnet")
        )

    let psi = Diagnostics.ProcessStartInfo host

    psi.ArgumentList.Add fsc
    psi.ArgumentList.Add("@" + rsp)
    psi.UseShellExecute <- false
    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    use p = Diagnostics.Process.Start psi
    let out = p.StandardOutput.ReadToEndAsync()
    let err = p.StandardError.ReadToEndAsync()
    p.WaitForExit()

    if p.ExitCode <> 0 then
        failtestf "fsc exited %d:\n%s%s" p.ExitCode out.Result err.Result

    dll

let private idiom =
    testList
        "idiom"
        [
            testCase "no tail calls in the interop or its tests (Tailcalls=false)"
            <| fun () ->
                // In a Debug build every project has Tailcalls=false, so an empty scan would pass trivially.
                if TailCalls.builtWithoutOptimization (Assembly.GetExecutingAssembly()) then
                    skiptest TailCalls.UnoptimizedBuild

                for asm in [ typeof<IUnknown>.Assembly; Assembly.GetExecutingAssembly() ] do
                    Expect.isEmpty (TailCalls.scan asm.Location) (asm.GetName().Name + " has no tail. prefixes")

            testCase "every generated wrapper takes this by reference (inref<ComPtr<'I>>)"
            <| fun () ->
                let ws = wrappers ()
                Expect.isGreaterThan ws.Length 300 "wrappers are generated"

                for m in ws do
                    let p = m.GetParameters()[0]
                    let name = m.DeclaringType.Name + "." + m.Name
                    Expect.isTrue p.ParameterType.IsByRef (name + ": this is a byref")
                    Expect.isTrue p.IsIn (name + ": this is an inref")
                    let e = p.ParameterType.GetElementType()

                    Expect.isTrue
                        (e.IsGenericType && e.GetGenericTypeDefinition() = typedefof<ComPtr<IUnknown>>)
                        (name + ": this is a ComPtr")

            testCase "a consumer built with tail calls gets no tail. prefix at wrapper calls"
            <| fun () ->
                let dll = compileConsumer ()
                let sites = TailCalls.scan dll
                let atWrappers = sites |> List.filter (fun s -> s.Callee.Contains "Methods.")

                Lrm.Core.Telemetry.check
                    "gpu.interop.tailcalls"
                    atWrappers.IsEmpty
                    [
                        Lrm.Core.Attr.str Lrm.Core.TelemetryKeys.Backend "d3d12"
                        Lrm.Core.Attr.int "tail_calls" sites.Length
                        Lrm.Core.Attr.int "tail_calls_at_wrappers" atWrappers.Length
                    ]

                Expect.isEmpty atWrappers "no call to a generated wrapper is a tail call"

                // The control: the same call made straight through a cached delegate is a tail call, so tail calls
                // were on, and the low-level Vtbl.get path stays the caller's responsibility.
                Expect.equal
                    (sites |> List.map (fun s -> s.Caller, s.Callee))
                    [
                        "Lrm.Gpu.D3D12.Interop.TailCallConsumer.controlDelegate", "ResourceBarrier.Invoke"
                    ]
                    "only the control has a tail call"

            testCase "no ComImport, RCWs or ComWrappers in the interop assembly"
            <| fun () ->
                let types, marshal, imports = references ()
                Expect.isEmpty imports "no type is a ComImport (TypeAttributes.Import)"
                Expect.isEmpty (Set.intersect types forbiddenTypes) "no COM interop types are referenced"
                Expect.isEmpty (Set.intersect marshal forbiddenMarshal) "no Marshal COM helpers are called"

                let asm = typeof<IUnknown>.Assembly
                Expect.isFalse (asm.GetTypes() |> Array.exists (fun t -> t.IsCOMObject || t.IsImport)) "no COM types"

            testCase "runtime marshalling is disabled for the interop assembly"
            <| fun () ->
                let a =
                    typeof<IUnknown>.Assembly.GetCustomAttribute<DisableRuntimeMarshallingAttribute>()

                Expect.isNotNull (box a) "DisableRuntimeMarshalling"

            testCase "every bound struct is blittable and its managed and native sizes agree"
            <| fun () ->
                let ss = AbiOracle.structs ()
                Expect.isGreaterThan ss.Length 40 "structs are bound"

                for s in ss do
                    Expect.isTrue s.Blittable (s.CName + " is blittable")
                    Expect.equal s.ManagedSize s.MarshalSize (s.CName + " managed and native sizes")

            testCase "every delegate signature is blittable and every delegate has its slot"
            <| fun () ->
                let slots =
                    AbiOracle.slots ()
                    |> List.map (fun (i, ms, _) -> i, ms |> List.map fst |> Set.ofList)
                    |> Map.ofList

                let ds = AbiOracle.delegates ()
                Expect.isGreaterThan ds.Length 200 "delegates are bound"

                for (i, m, _, ps) in ds do
                    Expect.isTrue (slots[i].Contains m) (sprintf "%s::%s has a slot" i m)
                    Expect.equal (fst ps[0]) 'w' (sprintf "%s::%s takes This first" i m)

                    let d =
                        typeof<IUnknown>.Assembly.GetType(sprintf "Lrm.Gpu.D3D12.Interop.Fn+%s+%s" i m)

                    Expect.isNotNull
                        (d.GetCustomAttribute<UnmanagedFunctionPointerAttribute>())
                        (sprintf "%s::%s is an unmanaged delegate" i m)

            testCase "every interface marker has an IID"
            <| fun () ->
                for (i, ms, count) in AbiOracle.slots () do
                    let t = typeof<IUnknown>.Assembly.GetType("Lrm.Gpu.D3D12.Interop." + i)
                    Expect.isNotNull t (i + " marker type")
                    Expect.isTrue t.IsInterface (i + " is a marker interface")
                    Expect.isEmpty (t.GetMethods()) (i + " has no members")
                    Expect.equal ms.Length count (i + " slot count")

                    let iid =
                        typeof<IidOf<IUnknown>>
                            .GetGenericTypeDefinition()
                            .MakeGenericType(t)
                            .GetProperty("Value")
                            .GetValue
                            null
                        :?> Guid

                    Expect.notEqual iid Guid.Empty (i + " IID")

            testCase "interfaces derive as in the SDK"
            <| fun () ->
                Expect.isTrue (typeof<ID3D12Pageable>.IsAssignableFrom typeof<ID3D12Resource>) "Resource : Pageable"

                Expect.isTrue
                    (typeof<ID3D12DeviceChild>.IsAssignableFrom typeof<ID3D12Pageable>)
                    "Pageable : DeviceChild"

                Expect.isTrue (typeof<ID3D12Device>.IsAssignableFrom typeof<ID3D12Device4>) "Device4 : Device"
                Expect.isTrue (typeof<ID3D12CommandList>.IsAssignableFrom typeof<ID3D12GraphicsCommandList>) "List"
                Expect.isTrue (typeof<IDXGIAdapter1>.IsAssignableFrom typeof<IDXGIAdapter4>) "Adapter4 : Adapter1"
                Expect.isFalse (typeof<ID3D12Device>.IsAssignableFrom typeof<ID3D12Resource>) "Resource is no Device"
        ]

/// Runs the generator in --check mode from its build output next to this exe.
let private generatorCheck () =
    let config =
        Path.GetFileName(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location))

    let repo = Env.repoRoot ()

    let tool =
        Path.Combine(repo, "artifacts", "bin", "Lrm.Tools.D3D12Gen", config, "Lrm.Tools.D3D12Gen.dll")

    if not (File.Exists tool) then
        skiptestf "%s is not built" tool

    let psi =
        Diagnostics.ProcessStartInfo(
            "dotnet",
            sprintf "\"%s\" --check --repo \"%s\"" tool repo,
            UseShellExecute = false
        )

    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    use p = Diagnostics.Process.Start psi
    let out = p.StandardOutput.ReadToEndAsync()
    let err = p.StandardError.ReadToEndAsync()
    p.WaitForExit()

    let attrs =
        [
            Lrm.Core.Attr.str Lrm.Core.TelemetryKeys.Backend "d3d12"
            Lrm.Core.Attr.str "check" "codegen_diff"
        ]

    match p.ExitCode with
    | 0 -> Lrm.Core.Telemetry.check "gpu.codegen.d3d12" true attrs
    | 4 ->
        // Not a pass: a machine without the pinned SDK servicing build cannot regenerate byte for byte. Say so in the
        // run log and on the console, then skip (the summary counts it as ignored).
        let reason = err.Result.Trim()
        Lrm.Core.Telemetry.event "gpu.codegen.d3d12.skipped" (Lrm.Core.Attr.str "reason" reason :: attrs)
        eprintfn "d3d12gen codegen diff SKIPPED (not passed): %s" reason
        skiptestf "the pinned Windows SDK headers are not available: %s" reason
    | code ->
        Lrm.Core.Telemetry.check "gpu.codegen.d3d12" false attrs
        failtestf "d3d12gen --check exited %d: %s%s" code out.Result err.Result

let private abi =
    testList
        "abi"
        [
            testCase "d3d12gen regenerates the committed files byte for byte" generatorCheck

            testCase "the committed oracle sources equal the runtime dump"
            <| fun () ->
                let stale = AbiOracle.stale (Env.repoRoot ())

                Expect.isEmpty
                    stale
                    "tools/abi-oracle/d3d12 is stale; regenerate with: pwsh tools/abi-oracle/d3d12/run.ps1 -Update"

            testCase "the dump checks every slot, struct and IID"
            <| fun () ->
                let c, n = AbiOracle.cSource ()
                let _, m = AbiOracle.cppSource ()
                Expect.stringContains c "LRM_SLOT(ID3D12Device, CreateCommittedResource, 27);" "probe slot"
                Expect.stringContains c "LRM_SIZE(D3D12_RESOURCE_DESC, 56);" "probe size"
                Expect.stringContains c "LRM_SLOTS(ID3D12GraphicsCommandList, 60);" "slot count"
                Expect.isGreaterThan n 1500 "C checks"
                Expect.isGreaterThan m 1000 "C++ checks"

            testCase "cl.exe compiles the oracle against the SDK headers with 0 failures"
            <| fun () ->
                match AbiOracle.findToolchain () with
                | None -> skiptest "no Visual Studio C++ toolchain (vswhere) on this machine"
                | Some tc ->
                    let repo = Env.repoRoot ()
                    let out = Path.Combine(repo, "artifacts", "abi-oracle", "d3d12", "cpu-tier")
                    let r = AbiOracle.compileAndRun tc repo out

                    Lrm.Core.Telemetry.check
                        "gpu.abi.oracle"
                        (r.ExitCode = 0)
                        [
                            Lrm.Core.Attr.str Lrm.Core.TelemetryKeys.Backend "d3d12"
                            Lrm.Core.Attr.int "static_checks" r.StaticChecks
                            Lrm.Core.Attr.int "failures" r.Failures.Length
                        ]

                    if r.ExitCode <> 0 || not r.Failures.IsEmpty then
                        failtestf "ABI oracle failed (exit %d):\n%s" r.ExitCode r.Output

                    Expect.isSome r.Summary "the oracle printed its summary"
                    Expect.stringContains r.Summary.Value "; 0 failures" "0 failures"

            testCase "the oracle rejects a wrong size, offset, slot, enum, HRESULT, signature or IID"
            <| fun () ->
                match AbiOracle.findToolchain () with
                | None -> skiptest "no Visual Studio C++ toolchain (vswhere) on this machine"
                | Some tc ->
                    let repo = Env.repoRoot ()
                    let root = Path.Combine(repo, "artifacts", "abi-oracle", "d3d12", "mutants")
                    let c = AbiOracle.cFile
                    let cpp = AbiOracle.cppFile

                    let compileTime =
                        [
                            c, "LRM_SIZE(D3D12_RESOURCE_DESC, 56);", "LRM_SIZE(D3D12_RESOURCE_DESC, 48);"
                            c,
                            "LRM_OFFSET(D3D12_RESOURCE_DESC, Width, 16);",
                            "LRM_OFFSET(D3D12_RESOURCE_DESC, Width, 8);"
                            c,
                            "LRM_SLOT(ID3D12Device, CreateCommittedResource, 27);",
                            "LRM_SLOT(ID3D12Device, CreateCommittedResource, 28);"
                            // D3D12_MESSAGE.ID is an int whose offset (8) and the struct size are kept by alignment
                            // even when the field is 2 bytes wide: only the field size catches it.
                            c, "LRM_FIELD_SIZE(D3D12_MESSAGE, ID, 4);", "LRM_FIELD_SIZE(D3D12_MESSAGE, ID, 2);"
                            c, "LRM_ENUM(D3D12_HEAP_TYPE_UPLOAD, 2LL);", "LRM_ENUM(D3D12_HEAP_TYPE_UPLOAD, 3LL);"
                            c,
                            "LRM_HRESULT(DXGI_ERROR_NOT_FOUND, 0x887A0002);",
                            "LRM_HRESULT(DXGI_ERROR_NOT_FOUND, 0x887A0003);"
                            cpp,
                            "LRM_SIG_ARG(LRM_VFN(ID3D12GraphicsCommandList, Dispatch), \"ID3D12GraphicsCommandList::Dispatch\", 1, 'i', 4);",
                            "LRM_SIG_ARG(LRM_VFN(ID3D12GraphicsCommandList, Dispatch), \"ID3D12GraphicsCommandList::Dispatch\", 1, 'i', 8);"
                            // A UINT64 bound as a double of the same size and offset: only the field kind catches it.
                            cpp,
                            "LRM_FIELD_KIND(D3D12_RESOURCE_DESC, Width, 'i', 8);",
                            "LRM_FIELD_KIND(D3D12_RESOURCE_DESC, Width, 'f', 8);"
                        ]

                    AbiOracle.mutant repo (Path.Combine(root, "src1")) compileTime

                    let r =
                        AbiOracle.compileAndRunIn tc (Path.Combine(root, "src1")) (Path.Combine(root, "out1"))

                    Expect.notEqual r.ExitCode 0 "the compile-time mutants fail"

                    for needle in
                        [
                            "sizeof(D3D12_RESOURCE_DESC) == 48"
                            "offsetof(D3D12_RESOURCE_DESC, Width) == 8"
                            "ID3D12Device::CreateCommittedResource is slot 28"
                            "D3D12_HEAP_TYPE_UPLOAD == 3LL"
                            "DXGI_ERROR_NOT_FOUND == 0x887A0003"
                            "ID3D12GraphicsCommandList::Dispatch: parameter 1"
                            "sizeof(D3D12_MESSAGE.ID) == 2"
                            "D3D12_RESOURCE_DESC.Width is 'f'8"
                        ] do
                        Expect.stringContains r.Output needle ("reported: " + needle)

                    let iid = "{ \"ID3D12Device\", &IID_ID3D12Device, { 0x189819f1u"

                    AbiOracle.mutant
                        repo
                        (Path.Combine(root, "src2"))
                        [ c, iid, "{ \"ID3D12Device\", &IID_ID3D12Device, { 0x189819f2u" ]

                    let r =
                        AbiOracle.compileAndRunIn tc (Path.Combine(root, "src2")) (Path.Combine(root, "out2"))

                    Expect.equal r.ExitCode 1 "the IID mutant fails at run time"
                    Expect.stringContains r.Output "FAIL IID_ID3D12Device" "reported"
        ]

/// A fake device whose CreateFence fails and whose removed reason is DXGI_ERROR_DEVICE_REMOVED.
let private fakeDevice () =
    let dev =
        new FakeObject(Slots.ID3D12Device.Count, [ Iids.ID3D12Object; Iids.ID3D12Device ])

    dev.Set(
        Slots.ID3D12Device.GetDeviceRemovedReason,
        Fn.ID3D12Device.GetDeviceRemovedReason(fun _ -> HResult.DXGI_ERROR_DEVICE_REMOVED)
    )

    dev.Set(
        Slots.ID3D12Device.CreateFence,
        Fn.ID3D12Device.CreateFence(fun _ _ _ _ ppv ->
            Marshal.WriteIntPtr(ppv, 0n)
            HResult.E_OUTOFMEMORY)
    )

    dev

let private com =
    testList
        "com"
        [
            testCase "ComPtr releases a reference once and tracks live references"
            <| fun () ->
                use o = new FakeObject(3, [])
                let before = Com.liveReferences ()
                let mutable p = ComPtr<IUnknown>.Attach o.Ptr
                Expect.equal (Com.liveReferences ()) (before + 1L) "attached"
                ComPtr.release &p
                ComPtr.release &p
                Expect.isTrue p.IsNull "release clears the variable"
                Expect.equal o.Refs 0 "released exactly once"
                Expect.equal (Com.liveReferences ()) before "no live references left"

            testCase "ComPtr has no public constructor; Borrow wraps a pointer without counting it"
            <| fun () ->
                let ctors =
                    typeof<ComPtr<IUnknown>>.GetConstructors(BindingFlags.Public ||| BindingFlags.Instance)

                Expect.isEmpty ctors "no public constructor: Attach (owned) or Borrow (not owned)"
                use o = new FakeObject(3, [])
                let before = Com.liveReferences ()
                let b = ComPtr<IUnknown>.Borrow o.Ptr
                Expect.equal b.Ptr o.Ptr "the pointer"
                Expect.equal (Com.liveReferences ()) before "a borrowed pointer is not counted"
                Expect.equal o.Refs 1 "and gets no reference"
                use owned = b.AddRef()
                Expect.equal (Com.liveReferences ()) (before + 1L) "AddRef gives an owned, counted reference"
                Expect.equal o.Refs 2 "AddRef reached the object"

            testCase "AddRef and QueryInterface hand out owned references"
            <| fun () ->
                use o = new FakeObject(Slots.ID3D12Object.Count, [ Iids.ID3D12Object ])
                let before = Com.liveReferences ()
                use p = ComPtr<IUnknown>.Attach o.Ptr
                use q = p.AddRef()
                use r = p.QueryInterface<ID3D12Object>()
                Expect.equal o.Refs 3 "three references"
                Expect.equal r.Ptr o.Ptr "same object"
                Expect.isTrue (p.TryQueryInterface<ID3D12Device>()).IsNone "not a device"
                Expect.equal (Com.liveReferences ()) (before + 3L) "three live"

            testCase "an unsupported QueryInterface raises D3D12Exception with E_NOINTERFACE"
            <| fun () ->
                use o = new FakeObject(3, [])
                use p = ComPtr<IUnknown>.Attach o.Ptr

                let e = raises<D3D12Exception>(fun () -> p.QueryInterface<ID3D12Fence>() |> ignore)

                Expect.equal e.Code HResult.E_NOINTERFACE "code"
                Expect.equal e.HResult HResult.E_NOINTERFACE "Exception.HResult"
                Expect.isNone e.RemovedReason "not a D3D12 object"
                Expect.equal o.Refs 1 "no reference leaked"

            testCase "a failed generated call raises with the device-removed reason"
            <| fun () ->
                use o = fakeDevice ()
                use dev = ComPtr<ID3D12Device>.Attach o.Ptr

                let e =
                    raises<D3D12Exception>(fun () ->
                        dev.CreateFence<_, ID3D12Fence>(0UL, D3D12_FENCE_FLAGS.NONE) |> ignore)

                Expect.equal e.Call "ID3D12Device::CreateFence" "call"
                Expect.equal e.Code HResult.E_OUTOFMEMORY "code"
                Expect.equal e.RemovedReason (Some HResult.DXGI_ERROR_DEVICE_REMOVED) "removed reason"
                Expect.isTrue e.DeviceLost "device lost"

                Expect.equal
                    e.Message
                    "ID3D12Device::CreateFence failed with 0x8007000E (E_OUTOFMEMORY); device removed reason 0x887A0005 (DXGI_ERROR_DEVICE_REMOVED)"
                    "message"

                Expect.equal o.Refs 1 "the removed-reason query released its reference"
                Expect.equal (dev.GetDeviceRemovedReason()) HResult.DXGI_ERROR_DEVICE_REMOVED "HRESULT as a value"

            testCase "struct returns use the hidden RetVal pointer"
            <| fun () ->
                use o = new FakeObject(Slots.ID3D12DescriptorHeap.Count, [])

                o.Set(
                    Slots.ID3D12DescriptorHeap.GetCPUDescriptorHandleForHeapStart,
                    Fn.ID3D12DescriptorHeap.GetCPUDescriptorHandleForHeapStart(fun _ retVal ->
                        Marshal.WriteInt64(retVal, 0x1234_5678_9abcL)
                        retVal)
                )

                use heap = ComPtr<ID3D12DescriptorHeap>.Attach o.Ptr
                let h = heap.GetCPUDescriptorHandleForHeapStart()
                Expect.equal h.ptr 0x1234_5678_9abcun "the handle written through RetVal"

            testCase "inref descriptions are passed by address and riid comes from the result type"
            <| fun () ->
                use o = new FakeObject(Slots.ID3D12Device.Count, [ Iids.ID3D12Device ])

                use queue =
                    new FakeObject(Slots.ID3D12CommandQueue.Count, [ Iids.ID3D12CommandQueue ])

                let mutable seen = None

                o.Set(
                    Slots.ID3D12Device.CreateCommandQueue,
                    Fn.ID3D12Device.CreateCommandQueue(fun _ pDesc riid ppv ->
                        let desc = NativePtr.read (NativePtr.ofNativeInt<D3D12_COMMAND_QUEUE_DESC> pDesc)
                        seen <- Some(desc.Type, Marshal.PtrToStructure<Guid> riid)
                        Marshal.WriteIntPtr(ppv, queue.Ptr)
                        HResult.S_OK)
                )

                use dev = ComPtr<ID3D12Device>.Attach o.Ptr
                let desc = Desc.commandQueue D3D12_COMMAND_LIST_TYPE.COMPUTE
                use q: ComPtr<ID3D12CommandQueue> = dev.CreateCommandQueue(&desc)
                Expect.equal q.Ptr queue.Ptr "the out pointer"
                Expect.equal seen (Some(D3D12_COMMAND_LIST_TYPE.COMPUTE, Iids.ID3D12CommandQueue)) "desc and riid"

            testCase "delegates are cached per function pointer"
            <| fun () ->
                use a = new FakeObject(3, [])
                use b = new FakeObject(3, [])
                let d1 = Vtbl.get<Fn.IUnknown.AddRef> a.Ptr Slots.IUnknown.AddRef
                let d2 = Vtbl.get<Fn.IUnknown.AddRef> a.Ptr Slots.IUnknown.AddRef
                let d3 = Vtbl.get<Fn.IUnknown.AddRef> b.Ptr Slots.IUnknown.AddRef
                let d4 = Vtbl.get<Fn.IUnknown.AddRef> a.Ptr Slots.IUnknown.AddRef
                Expect.isTrue (obj.ReferenceEquals(d1, d2)) "same delegate for the same pointer"
                Expect.isFalse (obj.ReferenceEquals(d1, d3)) "another vtable, another delegate"
                Expect.isTrue (obj.ReferenceEquals(d1, d4)) "still cached after a miss"

            testCase "calls on a null pointer raise instead of crashing"
            <| fun () ->
                let p = ComPtr<ID3D12Fence>.Null
                Expect.equal (p.Release()) 0u "releasing null is a no-op"
                Expect.throwsT<NullReferenceException> (fun () -> p.GetCompletedValue() |> ignore) "null call"
        ]

/// Reads `n` values of 'T at `p` (none for a null pointer).
let private readArray<'T when 'T: unmanaged> (p: nativeint) (n: int) : 'T list =
    if p = 0n then
        []
    else
        [ for i in 0 .. n - 1 -> NativePtr.get (NativePtr.ofNativeInt<'T> p) i ]

let private spans =
    testList
        "spans"
        [
            testCase "ResourceBarrier: a span passes its length and a pinned pointer, one element a count of 1"
            <| fun () ->
                use o = new FakeObject(Slots.ID3D12GraphicsCommandList.Count, [])
                let seen = ResizeArray<uint32 * bool * nativeint list>()

                o.Set(
                    Slots.ID3D12GraphicsCommandList.ResourceBarrier,
                    Fn.ID3D12GraphicsCommandList.ResourceBarrier(fun _ n p ->
                        let bs = readArray<D3D12_RESOURCE_BARRIER> p (int n)
                        seen.Add(n, (p = 0n), bs |> List.map _.UAV.pResource))
                )

                use l = ComPtr<ID3D12GraphicsCommandList>.Attach o.Ptr
                let mutable a = Desc.uavBarrier (ComPtr<ID3D12Resource>.Borrow 0x10n)
                let b = Desc.uavBarrier (ComPtr<ID3D12Resource>.Borrow 0x20n)
                l.ResourceBarrier(ReadOnlySpan [| a; b |])
                l.ResourceBarrier &b
                l.ResourceBarrier ReadOnlySpan<D3D12_RESOURCE_BARRIER>.Empty
                l.ResourceBarrierRaw(1u, NativePtr.toNativeInt &&a)

                Expect.equal
                    (List.ofSeq seen)
                    [
                        2u, false, [ 0x10n; 0x20n ]
                        1u, false, [ 0x20n ]
                        0u, true, []
                        1u, false, [ 0x10n ]
                    ]
                    "count, null pointer, barriers"

            testCase "ExecuteCommandLists and SetDescriptorHeaps take ComPtr spans or one ComPtr"
            <| fun () ->
                use q = new FakeObject(Slots.ID3D12CommandQueue.Count, [])
                use g = new FakeObject(Slots.ID3D12GraphicsCommandList.Count, [])
                let lists = ResizeArray<nativeint list>()
                let heaps = ResizeArray<nativeint list>()

                q.Set(
                    Slots.ID3D12CommandQueue.ExecuteCommandLists,
                    Fn.ID3D12CommandQueue.ExecuteCommandLists(fun _ n pp -> lists.Add(readArray<nativeint> pp (int n)))
                )

                g.Set(
                    Slots.ID3D12GraphicsCommandList.SetDescriptorHeaps,
                    Fn.ID3D12GraphicsCommandList.SetDescriptorHeaps(fun _ n pp ->
                        heaps.Add(readArray<nativeint> pp (int n)))
                )

                use queue = ComPtr<ID3D12CommandQueue>.Attach q.Ptr
                use list = ComPtr<ID3D12GraphicsCommandList>.Attach g.Ptr
                let other = ComPtr<ID3D12GraphicsCommandList>.Borrow 0x200n
                queue.ExecuteCommandLists list
                queue.ExecuteCommandLists(ReadOnlySpan [| list; other |])
                list.SetDescriptorHeaps(ComPtr<ID3D12DescriptorHeap>.Borrow 0x300n)

                Expect.equal (List.ofSeq lists) [ [ g.Ptr ]; [ g.Ptr; 0x200n ] ] "command lists"
                Expect.equal (List.ofSeq heaps) [ [ 0x300n ] ] "descriptor heaps"

            testCase "SetComputeRoot32BitConstants takes a uint32 span; Values[4] must hold 4"
            <| fun () ->
                use o = new FakeObject(Slots.ID3D12GraphicsCommandList.Count, [])
                let constants = ResizeArray<uint32 * uint32 list * uint32>()
                let clears = ResizeArray<uint32 list>()

                o.Set(
                    Slots.ID3D12GraphicsCommandList.SetComputeRoot32BitConstants,
                    Fn.ID3D12GraphicsCommandList.SetComputeRoot32BitConstants(fun _ index n p offset ->
                        constants.Add(index, readArray<uint32> p (int n), offset))
                )

                o.Set(
                    Slots.ID3D12GraphicsCommandList.ClearUnorderedAccessViewUint,
                    Fn.ID3D12GraphicsCommandList.ClearUnorderedAccessViewUint(fun _ _ _ _ values _ _ ->
                        clears.Add(readArray<uint32> values 4))
                )

                use l = ComPtr<ID3D12GraphicsCommandList>.Attach o.Ptr
                l.SetComputeRoot32BitConstants(3u, ReadOnlySpan [| 7u; 8u; 9u |], 5u)
                Expect.equal (List.ofSeq constants) [ 3u, [ 7u; 8u; 9u ], 5u ] "index, values, offset"

                let gpu = D3D12_GPU_DESCRIPTOR_HANDLE(ptr = 1UL)
                let cpu = D3D12_CPU_DESCRIPTOR_HANDLE(ptr = 2un)
                let r = ComPtr<ID3D12Resource>.Borrow 0x30n
                l.ClearUnorderedAccessViewUint(gpu, cpu, r, ReadOnlySpan [| 1u; 2u; 3u; 4u |], 0u, 0n)

                Expect.throwsT<ArgumentException>
                    (fun () -> l.ClearUnorderedAccessViewUint(gpu, cpu, r, ReadOnlySpan [| 1u; 2u; 3u |], 0u, 0n))
                    "three values are refused"

                Expect.equal (List.ofSeq clears) [ [ 1u; 2u; 3u; 4u ] ] "only the valid call reached COM"

            testCase "RetVal is returned when other parameters follow it"
            <| fun () ->
                use o = new FakeObject(Slots.ID3D12Device.Count, [])
                let seen = ResizeArray<uint32 * uint64 list>()

                o.Set(
                    Slots.ID3D12Device.GetResourceAllocationInfo,
                    Fn.ID3D12Device.GetResourceAllocationInfo(fun _ retVal mask n descs ->
                        let ds = readArray<D3D12_RESOURCE_DESC> descs (int n)
                        seen.Add(mask, ds |> List.map _.Width)

                        let info =
                            D3D12_RESOURCE_ALLOCATION_INFO(
                                SizeInBytes = (ds |> List.sumBy _.Width),
                                Alignment = 65536UL
                            )

                        NativePtr.write (NativePtr.ofNativeInt retVal) info
                        retVal)
                )

                o.Set(
                    Slots.ID3D12Device.GetCustomHeapProperties,
                    Fn.ID3D12Device.GetCustomHeapProperties(fun _ retVal nodeMask kind ->
                        let props =
                            D3D12_HEAP_PROPERTIES(
                                Type = D3D12_HEAP_TYPE.CUSTOM,
                                CPUPageProperty =
                                    (if kind = D3D12_HEAP_TYPE.UPLOAD then
                                         D3D12_CPU_PAGE_PROPERTY.WRITE_COMBINE
                                     else
                                         D3D12_CPU_PAGE_PROPERTY.NOT_AVAILABLE),
                                MemoryPoolPreference = D3D12_MEMORY_POOL.L0,
                                CreationNodeMask = nodeMask,
                                VisibleNodeMask = nodeMask
                            )

                        NativePtr.write (NativePtr.ofNativeInt retVal) props
                        retVal)
                )

                use dev = ComPtr<ID3D12Device>.Attach o.Ptr
                let d = Desc.buffer 4096UL D3D12_RESOURCE_FLAGS.NONE
                let e = Desc.buffer 1000UL D3D12_RESOURCE_FLAGS.NONE
                let one = dev.GetResourceAllocationInfo(1u, &d)
                let two = dev.GetResourceAllocationInfo(0u, ReadOnlySpan [| d; e |])
                Expect.equal (one.SizeInBytes, one.Alignment) (4096UL, 65536UL) "one description"
                Expect.equal two.SizeInBytes 5096UL "two descriptions"
                Expect.equal (List.ofSeq seen) [ 1u, [ 4096UL ]; 0u, [ 4096UL; 1000UL ] ] "mask and widths"

                let p = dev.GetCustomHeapProperties(0u, D3D12_HEAP_TYPE.UPLOAD)

                Expect.equal
                    (p.Type, p.CPUPageProperty, p.MemoryPoolPreference)
                    (D3D12_HEAP_TYPE.CUSTOM, D3D12_CPU_PAGE_PROPERTY.WRITE_COMBINE, D3D12_MEMORY_POOL.L0)
                    "heap properties"

            testCase "Map returns the pointer; CreateRootSignature takes the blob as a byte span"
            <| fun () ->
                use res = new FakeObject(Slots.ID3D12Resource.Count, [])
                use o = new FakeObject(Slots.ID3D12Device.Count, [])

                use rs =
                    new FakeObject(Slots.ID3D12RootSignature.Count, [ Iids.ID3D12RootSignature ])

                let mutable blob = []
                let mutable iid = Guid.Empty

                res.Set(
                    Slots.ID3D12Resource.Map,
                    Fn.ID3D12Resource.Map(fun _ _ _ ppData ->
                        Marshal.WriteIntPtr(ppData, 0x4000n)
                        HResult.S_OK)
                )

                o.Set(
                    Slots.ID3D12Device.CreateRootSignature,
                    Fn.ID3D12Device.CreateRootSignature(fun _ _ p length riid ppv ->
                        blob <- readArray<byte> p (int length)
                        iid <- Marshal.PtrToStructure<Guid> riid
                        Marshal.WriteIntPtr(ppv, rs.Ptr)
                        HResult.S_OK)
                )

                use r = ComPtr<ID3D12Resource>.Attach res.Ptr
                Expect.equal (r.Map(0u, 0n)) 0x4000n "the mapped pointer"
                use dev = ComPtr<ID3D12Device>.Attach o.Ptr

                use rootSig: ComPtr<ID3D12RootSignature> =
                    dev.CreateRootSignature(0u, ReadOnlySpan [| 1uy; 2uy; 3uy |])

                Expect.equal rootSig.Ptr rs.Ptr "the root signature"
                Expect.equal blob [ 1uy; 2uy; 3uy ] "the blob bytes and their length"
                Expect.equal iid Iids.ID3D12RootSignature "riid from the result type"
        ]

let private descs =
    testList
        "desc"
        [
            testCase "a UAV buffer description has the probe's bytes"
            <| fun () ->
                let d = Desc.buffer 4096UL D3D12_RESOURCE_FLAGS.ALLOW_UNORDERED_ACCESS

                Expect.equal
                    (hex (bytesOf d))
                    ("01000000"
                     + "00000000"
                     + "0000000000000000"
                     + "0010000000000000"
                     + "01000000"
                     + "0100"
                     + "0100"
                     + "00000000"
                     + "01000000"
                     + "00000000"
                     + "01000000"
                     + "04000000"
                     + "00000000")
                    "D3D12_RESOURCE_DESC"

            testCase "a raw buffer UAV description has the probe's bytes"
            <| fun () ->
                let d = Desc.rawBufferUav 0UL 1024u

                Expect.equal
                    (hex (bytesOf d))
                    ("27000000"
                     + "01000000"
                     + "0000000000000000"
                     + "00040000"
                     + "00000000"
                     + "0000000000000000"
                     + "01000000"
                     + "00000000")
                    "D3D12_UNORDERED_ACCESS_VIEW_DESC"

            testCase "a transition barrier has the probe's bytes"
            <| fun () ->
                let r = ComPtr<ID3D12Resource>.Borrow 0x1122334455667788n

                let b =
                    Desc.transition r D3D12_RESOURCE_STATES.COMMON D3D12_RESOURCE_STATES.UNORDERED_ACCESS

                Expect.equal
                    (hex (bytesOf b))
                    ("00000000"
                     + "00000000"
                     + "8877665544332211"
                     + "ffffffff"
                     + "00000000"
                     + "08000000"
                     + "00000000")
                    "D3D12_RESOURCE_BARRIER"

            testCase "heap properties, queue and descriptor heap descriptions"
            <| fun () ->
                Expect.equal
                    (hex (bytesOf (Desc.heapProperties D3D12_HEAP_TYPE.UPLOAD)))
                    "0200000000000000000000000000000000000000"
                    "D3D12_HEAP_PROPERTIES"

                Expect.equal
                    (hex (bytesOf (Desc.commandQueue D3D12_COMMAND_LIST_TYPE.COMPUTE)))
                    "02000000000000000000000000000000"
                    "D3D12_COMMAND_QUEUE_DESC"

                Expect.equal
                    (hex (bytesOf (Desc.descriptorHeap D3D12_DESCRIPTOR_HEAP_TYPE.CBV_SRV_UAV 64u true)))
                    "00000000400000000100000000000000"
                    "D3D12_DESCRIPTOR_HEAP_DESC"
        ]

let private adapter =
    {
        Index = 0
        Description = "NVIDIA GeForce RTX 4080"
        VendorId = 0x10DEu
        DeviceId = 0x2704u
        SubSysId = 0u
        Revision = 0u
        DedicatedVideoMemory = 16UL <<< 30
        DedicatedSystemMemory = 0UL
        SharedSystemMemory = 0UL
        Luid = 0L
        Software = false
        UmdVersion = Some "32.0.16.1692"
        DriverVersion = "616.92"
    }

let private caps =
    {
        MaxFeatureLevel = D3D_FEATURE_LEVEL.LEVEL_12_2
        HighestShaderModel = D3D_SHADER_MODEL.SHADER_MODEL_6_8
        ResourceBindingTier = D3D12_RESOURCE_BINDING_TIER.TIER_3
        ResourceHeapTier = D3D12_RESOURCE_HEAP_TIER.TIER_2
        HighestRootSignature = D3D_ROOT_SIGNATURE_VERSION.VERSION_1_2
        DoublePrecisionShaderOps = true
        WaveOps = true
        WaveLaneCountMin = 32u
        WaveLaneCountMax = 32u
        Int64ShaderOps = true
        Native16BitShaderOps = true
        AtomicInt64OnTypedResource = true
        AtomicInt64OnGroupShared = true
        AtomicInt64OnDescriptorHeapResource = true
        EnhancedBarriers = true
        Uma = false
        CacheCoherentUma = false
        MaxGpuVirtualAddressBitsPerResource = 40u
        MaxGpuVirtualAddressBitsPerProcess = 40u
        NodeCount = 1u
        RuntimeSdkVersion = Some 616u
    }

let private deviceCheck =
    testList
        "devicecheck"
        [
            testCase "an RTX 4080-like adapter is accepted"
            <| fun () -> Expect.equal (DeviceCheck.verdict adapter (Ok caps)) Accepted "accepted"

            testCase "SM 6.6 with tier 3 is the boundary"
            <| fun () ->
                let c =
                    { caps with
                        HighestShaderModel = D3D_SHADER_MODEL.SHADER_MODEL_6_6
                    }

                Expect.equal (DeviceCheck.verdict adapter (Ok c)) Accepted "SM 6.6 is enough"

            testCase "SM 6.5, tier 2, software adapters and missing devices are rejected"
            <| fun () ->
                let sm65 =
                    { caps with
                        HighestShaderModel = D3D_SHADER_MODEL.SHADER_MODEL_6_5
                    }

                let tier2 =
                    { caps with
                        ResourceBindingTier = D3D12_RESOURCE_BINDING_TIER.TIER_2
                    }

                Expect.equal
                    (DeviceCheck.verdict adapter (Ok sm65))
                    (Rejected [ "shader model 6.5 is below the required 6.6" ])
                    "SM 6.5"

                Expect.equal
                    (DeviceCheck.verdict adapter (Ok tier2))
                    (Rejected [ "resource binding tier 2 is below the required 3" ])
                    "tier 2"

                match DeviceCheck.verdict { adapter with Software = true } (Ok caps) with
                | Rejected [ r ] -> Expect.stringContains r "software adapter" "software"
                | v -> failtestf "software adapter: %A" v

                match DeviceCheck.verdict adapter (Error "D3D12CreateDevice failed") with
                | Rejected [ r ] -> Expect.stringContains r "no D3D12 device" "no device"
                | v -> failtestf "no device: %A" v

            testCase "driver versions"
            <| fun () ->
                Expect.equal (DeviceCheck.umdParts 0x0020_0000_0010_069CL) [| 32us; 0us; 16us; 1692us |] "UMD parts"
                Expect.equal (DeviceCheck.formatDriverVersion 0x10DEu [| 32us; 0us; 16us; 1692us |]) "616.92" "616.92"
                Expect.equal (DeviceCheck.formatDriverVersion 0x10DEu [| 31us; 0us; 15us; 5222us |]) "552.22" "552.22"
                Expect.equal (DeviceCheck.formatDriverVersion 0x10DEu [| 31us; 0us; 15us; 920us |]) "509.20" "509.20"

                Expect.equal
                    (DeviceCheck.formatDriverVersion 0x8086u [| 31us; 0us; 101us; 5186us |])
                    "31.0.101.5186"
                    "other vendors keep the UMD version"

            testCase "shader model and feature level names"
            <| fun () ->
                Expect.equal (DeviceCheck.formatShaderModel D3D_SHADER_MODEL.SHADER_MODEL_6_6) "6.6" "SM"
                Expect.equal (DeviceCheck.formatFeatureLevel D3D_FEATURE_LEVEL.LEVEL_12_2) "12_2" "FL"
        ]

/// Sequenced: the COM tests compare the process-wide live-reference count.
let tests =
    testSequenced (testList "d3d12.interop" [ probe; idiom; abi; com; spans; descs; deviceCheck ])
