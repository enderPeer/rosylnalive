module Lrm.Gpu.Core.Tests.ShaderTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common
open Lrm.Shaders
open Lrm.Testing

/// The canonical rootsig.bin, as the planning probe compiled it (LRM_DATA/spikes-staging/bindtest/rs.bin) and as
/// docs/contracts/shader-build.md lists it.
let rootSigKat =
    "44584243 14462d179f897b3ba160caa35fcb04a6 01000000 5c000000 01000000 24000000"
    + " 52545330 30000000 02000000 01000000 18000000 00000000 30000000 00040000"
    + " 01000000 00000000 24000000 00000000 00000000 20000000"

/// KAT S1 (docs/contracts/shader-build.md): ShaderSetHash of a synthetic set, computed independently in Python.
/// DxcVersion "d", CatalogHash SHA-256("c"), DXIL flags ["-O3"], rootsig bytes "rs", DXIL blobs a = SHA-256("A") and
/// b = SHA-256("B"); no SPIR-V flags or blobs.
let s1Dxil = "cdd90f127b63cf16e1c99616e7327d93de08e6aa9816fd5aeea2d5dbfca75241"
let s1Spirv = "ef3e37506c8bcee5603764779da39299a62edf3216681cc775d67637cc930e97"

let selftests =
    [ "scale_f32"; "table_echo"; "args_echo"; "groups_echo"; "bf16_unpack" ]

let private source (k: KernelSpec) =
    File.ReadAllText(Path.Combine(Env.repoRoot (), "src", "Lrm.Shaders", k.Source))

let private kernel (name: string) =
    Kernels.all |> Array.find (fun k -> k.Name = name)

/// A minimal clean kernel for lint fixtures.
let private cleanScale =
    "#include \"lrm.hlsli\"\nLRM_ARGS(scale_f32)\nLRM_NUMTHREADS(scale_f32)\nvoid main(uint3 gid : SV_GroupID) { }\n"

let tests =
    testList
        "shaders"
        [
            testCase "the root signature is canonical"
            <| fun () ->
                Expect.equal (hex (RootSignature.canonical ())) (rootSigKat.Replace(" ", "")) "the constant"
                Expect.equal (RootSignature.canonical ()).Length 92 "92 bytes"
                let embedded = Shaders.readResource ShaderSet.RootSignatureResource
                Expect.equal (hex embedded) (rootSigKat.Replace(" ", "")) "the embedded rootsig.bin"
                Expect.isTrue (RootSignature.isCanonical embedded) "isCanonical"

                match Dxbc.parts embedded with
                | Ok [ p ] ->
                    Expect.equal (p.FourCC, p.Offset, p.Size) ("RTS0", 44, 48) "one RTS0 part"

                    let u32 o =
                        BitConverter.ToUInt32(embedded, p.Offset + o)

                    Expect.equal (u32 0) 2u "D3D_ROOT_SIGNATURE_VERSION_1_1"
                    Expect.equal (u32 4) 1u "one parameter"
                    Expect.equal (u32 20) 0x400u "CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED"
                    Expect.equal (u32 24) 1u "32-bit constants"
                    Expect.equal (u32 44) 32u "32 root constants (128 bytes)"
                | other -> failtestf "unexpected parts %A" other

            testCase "the embedded set loads and verifies"
            <| fun () ->
                let set = Shaders.embedded ()
                Expect.equal set.CatalogHash (ContentId.parse Kernels.CatalogHashHex) "built for this catalog"

                Expect.equal
                    set.DxcVersion
                    "dxcompiler.dll: 1.9(5512-01b62ad4)(1.9.2609.5) - 1.9.2609.5 (01b62ad47)"
                    "pinned DXC"

                let present = set.Blobs |> List.map _.Kernel |> List.distinct
                let names = Kernels.all |> Array.map _.Name |> Set.ofArray

                Expect.equal
                    (Set.union (Set.ofList present) (Set.ofList set.Missing))
                    names
                    "every catalog kernel is built or listed as missing"

                Expect.isEmpty (Set.intersect (Set.ofList present) (Set.ofList set.Missing)) "not both"

                for s in selftests do
                    Expect.contains present s "every selftest is built"

                for k in present do
                    for t in ShaderTarget.all do
                        Expect.isSome (ShaderSet.tryBlob k t set) (sprintf "%s has a %s blob" k (ShaderTarget.name t))

            testCase "DXIL blobs are signed DXBC containers with the catalog's numthreads"
            <| fun () ->
                for b in ShaderSet.blobsFor ShaderTarget.Dxil (Shaders.embedded ()) do
                    match Dxbc.parts b.Bytes with
                    | Error e -> failtestf "%s: %s" b.Kernel e
                    | Ok ps ->
                        Expect.isTrue (ps |> List.exists (fun p -> p.FourCC = "DXIL")) (b.Kernel + " has a DXIL part")
                        Expect.isTrue (Dxbc.isSigned b.Bytes) (b.Kernel + " is signed")

                        Expect.equal
                            (Dxbc.numThreads b.Bytes)
                            (Some (kernel b.Kernel).Threads)
                            (b.Kernel + " numthreads")

            testCase "SPIR-V blobs are modules with the catalog's LocalSize"
            <| fun () ->
                for b in ShaderSet.blobsFor ShaderTarget.Spirv (Shaders.embedded ()) do
                    Expect.isTrue (Spirv.isModule b.Bytes) (b.Kernel + " is SPIR-V")
                    Expect.equal (Spirv.localSize b.Bytes) (Some (kernel b.Kernel).Threads) (b.Kernel + " LocalSize")

            testCase "the manifest regenerates byte for byte, and ShaderSetHash reproduces"
            <| fun () ->
                let set = Shaders.embedded ()
                Expect.equal (ShaderSet.manifestJson set) (Shaders.manifestText ()) "manifest bytes"

                for t in ShaderTarget.all do
                    Expect.equal (Shaders.shaderSetHash t) (ShaderSet.setHash t set) (ShaderTarget.name t)

                Expect.notEqual
                    (ShaderSet.setHash ShaderTarget.Dxil set)
                    (ShaderSet.setHash ShaderTarget.Spirv set)
                    "one hash per target"

            testCase "KAT S1: ShaderSetHash of a synthetic set"
            <| fun () ->
                let blob (name: string) (content: string) : ShaderBlob =
                    {
                        Kernel = name
                        Target = ShaderTarget.Dxil
                        Bytes = [||]
                        Sha256 = Hash.sha256Utf8 content
                        Threads = (1, 1, 1)
                    }

                let set =
                    {
                        DxcVersion = "d"
                        CatalogHash = Hash.sha256Utf8 "c"
                        Flags = [ ShaderTarget.Dxil, [ "-O3" ] ]
                        RootSignature = Text.Encoding.UTF8.GetBytes "rs"
                        // Out of order on purpose: the hash sorts by kernel name.
                        Blobs = [ blob "b" "B"; blob "a" "A" ]
                        Missing = []
                    }

                Expect.equal (ShaderSet.setHash ShaderTarget.Dxil set).Hex s1Dxil "dxil"
                Expect.equal (ShaderSet.setHash ShaderTarget.Spirv set).Hex s1Spirv "spirv"

            testCase "ShaderSetHash covers every input"
            <| fun () ->
                let set = Shaders.embedded ()
                let h = ShaderSet.setHash ShaderTarget.Dxil set
                let first = set.Blobs.Head

                let changedBlob =
                    { set with
                        Blobs =
                            { first with
                                Sha256 = Hash.sha256Utf8 "other"
                            }
                            :: set.Blobs.Tail
                    }

                Expect.notEqual (ShaderSet.setHash ShaderTarget.Dxil changedBlob) h "a blob"
                Expect.notEqual (ShaderSet.setHash ShaderTarget.Dxil { set with DxcVersion = "x" }) h "the compiler"

                Expect.notEqual
                    (ShaderSet.setHash
                        ShaderTarget.Dxil
                        { set with
                            CatalogHash = Hash.sha256Utf8 "c"
                        })
                    h
                    "the catalog"

                Expect.notEqual
                    (ShaderSet.setHash
                        ShaderTarget.Dxil
                        { set with
                            Flags = [ ShaderTarget.Dxil, [ "-O0" ] ]
                        })
                    h
                    "the flags"

            testCase "a tampered manifest or blob is refused"
            <| fun () ->
                let text = Shaders.manifestText ()

                let tamper (read: string -> byte[]) =
                    ShaderSet.ofManifest text read |> ignore

                raisesContract "does not match its SHA-256" (fun () ->
                    tamper (fun name ->
                        let b = Shaders.readResource name

                        if name = "dxil/scale_f32.dxil" then
                            b[b.Length - 1] <- b[b.Length - 1] ^^^ 1uy

                        b))

                let dxilHash = (Shaders.shaderSetHash ShaderTarget.Dxil).Hex
                let otherHash = text.Replace(dxilHash, (Hash.sha256Utf8 "other").Hex)
                Expect.notEqual otherHash text "the hash is in the manifest"

                raisesContract "not valid JSON" (fun () -> ShaderSet.ofManifest "{" Shaders.readResource |> ignore)

                raisesContract "shader manifest" (fun () ->
                    ShaderSet.ofManifest (text.Replace("\"targets\"", "\"targetz\"")) Shaders.readResource
                    |> ignore)

                raisesContract "schema" (fun () ->
                    ShaderSet.ofManifest (text.Replace("lrm.shaders/1", "lrm.shaders/2")) Shaders.readResource
                    |> ignore)

                raisesContract "does not reproduce" (fun () ->
                    ShaderSet.ofManifest otherHash Shaders.readResource |> ignore)

            testCase "the selftest sources follow the source convention"
            <| fun () ->
                for name in selftests do
                    let k = kernel name
                    Expect.equal (SourceConvention.lint k (source k)) [] name

            testCase "the source convention rejects what the contract forbids"
            <| fun () ->
                let k = kernel "scale_f32"
                Expect.equal (SourceConvention.lint k cleanScale) [] "the clean fixture"

                let rejects (fragment: string) (text: string) =
                    let ps = SourceConvention.lint k text

                    Expect.isTrue (ps |> List.exists (fun p -> p.Contains fragment)) (sprintf "'%s' in %A" fragment ps)

                let add (code: string) = cleanScale + code + "\n"
                rejects "NonUniformResourceIndex" (add "uint a = NonUniformResourceIndex(1);")
                rejects "ResourceDescriptorHeap" (add "RWByteAddressBuffer b = ResourceDescriptorHeap[0];")
                rejects "'half'" (add "half h;")
                rejects "16-bit" (add "uint16_t u;")
                rejects "WaveActiveSum" (add "float s = WaveActiveSum(1.0);")
                rejects "QuadReadAcrossX" (add "float s = QuadReadAcrossX(1.0);")
                rejects "InterlockedAdd" (add "void f() { uint o; InterlockedAdd(o, 1, o); }")
                rejects "SV_DispatchThreadID" (add "void g(uint3 i : SV_DispatchThreadID) { }")
                rejects "register" (add "RWByteAddressBuffer b : register(u1);")
                rejects "ConstantBuffer" (add "ConstantBuffer<lrm_args_add> c;")
                rejects "StructuredBuffer" (add "StructuredBuffer<float> s;")
                rejects "Texture2D" (add "Texture2D t;")
                rejects "#pragma" (add "#pragma dxc diagnostic push")
                rejects "[[vk::" (add "[[vk::binding(1, 0)]] RWByteAddressBuffer b;")
                rejects "lrm.hlsli" (add "#define LRM_BUF(x) x")
                rejects "numthreads" (add "[numthreads(1, 1, 1)] void h() { }")
                rejects "first include" ("#include \"other.hlsli\"\n" + cleanScale)
                rejects "beside the kernel" (add "#include \"../basic/x.hlsli\"")
                rejects "LRM_ARGS(scale_f32) exactly once" (cleanScale.Replace("LRM_ARGS(scale_f32)", "LRM_ARGS(add)"))
                rejects "LRM_NUMTHREADS(scale_f32)" (cleanScale.Replace("LRM_NUMTHREADS(scale_f32)", ""))
                rejects "exactly one entry point" (add "void main(uint3 t : SV_GroupThreadID) { }")

                Expect.equal
                    (SourceConvention.lint
                        k
                        (cleanScale + "// NonUniformResourceIndex half WaveActiveSum\n/* register */\n"))
                    []
                    "comments are ignored"

            testCase "the shader manifest lists the missing kernels in catalog order"
            <| fun () ->
                let set = Shaders.embedded ()
                let order = Kernels.all |> Array.map _.Name |> List.ofArray
                let positions = set.Missing |> List.map (fun m -> List.findIndex ((=) m) order)
                Expect.equal positions (List.sort positions) "catalog order"
        ]
