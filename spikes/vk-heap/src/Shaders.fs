/// Compiles the spike kernels with the pinned DXC (DataRootAndToolchain: LRM_DATA/toolchain/dxc/1.9.2609.5) into
/// DXIL and both SPIR-V binding variants, and inspects the SPIR-V: resource variables and their set/binding, the
/// push-constant block layout, LocalSize and capabilities.
module VkHeapSpike.Shaders

open System
open System.Diagnostics
open System.IO
open Lrm.Core

let PinnedDxcVersion = "1.9.2609.5"

let kernels = [ "scale_f32"; "args_echo"; "table_echo"; "bf16_unpack" ]

/// How the SPIR-V side reaches the heap.
type Variant =
    /// [[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[]: one runtime array at set 0 binding 0.
    | Explicit
    /// ResourceDescriptorHeap[i] with -fvk-bind-resource-heap 0 0: one aliased variable per use, all at set 0 binding 0.
    | Emulated

let variantName =
    function
    | Explicit -> "explicit"
    | Emulated -> "emulated"

let dxcPath () =
    Path.Combine(LrmPaths.toolchain (), "dxc", PinnedDxcVersion, "build", "native", "bin", "x64", "dxc.exe")

let private run (exe: string) (args: string list) =
    let psi = ProcessStartInfo(exe)

    for a in args do
        psi.ArgumentList.Add a

    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    psi.UseShellExecute <- false
    use p = Process.Start psi
    let out = p.StandardOutput.ReadToEndAsync()
    let err = p.StandardError.ReadToEnd()
    p.WaitForExit()
    p.ExitCode, out.Result, err

/// `dxc --version`; fails unless it is the pinned build.
let dxcVersion () =
    let exe = dxcPath ()

    if not (File.Exists exe) then
        failwithf "pinned DXC not found at %s (run tools/bootstrap/bootstrap.ps1)" exe

    let code, out, err = run exe [ "--version" ]
    let text = (out + err).Trim()

    if code <> 0 || not (text.Contains PinnedDxcVersion) then
        failwithf "dxc --version: expected %s, got '%s' (exit %d)" PinnedDxcVersion text code

    text

let hlslDir () =
    Path.Combine(AppContext.BaseDirectory, "hlsl")

let common = [ "-HV"; "2021"; "-O3"; "-WX" ]

let compile (outDir: string) (target: string) (args: string list) (source: string) (outName: string) =
    Directory.CreateDirectory outDir |> ignore
    let outPath = Path.Combine(outDir, outName)
    let asmPath = outPath + ".txt"

    let code, out, err =
        run
            (dxcPath ())
            ([ "-T"; target ] @ args @ [ "-Fo"; outPath; "-Fc"; asmPath; Path.Combine(hlslDir (), source) ])

    if code <> 0 then
        failwithf "dxc %s %s failed (exit %d):\n%s%s" target source code out err

    File.ReadAllBytes outPath

type ShaderSet =
    {
        Dxil: Map<string, byte[]>
        Spirv: Map<Variant * string, byte[]>
        RootSignature: byte[]
        /// Canonical.hash "lrm.r1-spike.shaders/1" over every blob, by name.
        Hash: ContentId
        DxcVersion: string
    }

let dxilArgs = [ "-E"; "main" ] @ common

let spirvArgs variant =
    [ "-E"; "main" ]
    @ common
    @ [ "-spirv"; "-fspv-target-env=vulkan1.3" ]
    @ (match variant with
       | Explicit -> []
       | Emulated -> [ "-D"; "LRM_EMULATED_HEAP"; "-fvk-bind-resource-heap"; "0"; "0" ])

let compileAll (outDir: string) : ShaderSet =
    let version = dxcVersion ()

    let dxil =
        kernels
        |> List.map (fun k -> k, compile outDir "cs_6_6" dxilArgs (k + ".hlsl") (k + ".dxil"))
        |> Map.ofList

    let spirv =
        [
            for v in [ Explicit; Emulated ] do
                for k in kernels do
                    (v, k), compile outDir "cs_6_6" (spirvArgs v) (k + ".hlsl") (sprintf "%s.%s.spv" k (variantName v))
        ]
        |> Map.ofList

    let rs =
        compile outDir "rootsig_1_1" [ "-E"; "LRM_ROOTSIG" ] "rootsig.hlsl" "rootsig.bin"

    let named =
        [
            for KeyValue(k, b) in dxil do
                yield k + ".dxil", b
            for KeyValue((v, k), b) in spirv do
                yield sprintf "%s.%s.spv" k (variantName v), b
            yield "rootsig.bin", rs
        ]
        |> List.sortBy fst

    {
        Dxil = dxil
        Spirv = spirv
        RootSignature = rs
        Hash = Canonical.hash "lrm.r1-spike.shaders/1" (fun w -> w.Map(named, (fun (b: byte[]) -> w.Bytes b)))
        DxcVersion = version
    }

// ---- SPIR-V inspection (binary) ----

type ResourceVar =
    {
        Id: uint32
        Name: string
        StorageClass: uint32
        Set: uint32 option
        Binding: uint32 option
        NonWritable: bool
    }

type SpirvFacts =
    {
        Resources: ResourceVar list
        /// Byte offsets of the push-constant block's members.
        PushMemberOffsets: uint32 list
        /// Array stride of the block's first member (uint4 d[8]).
        PushArrayStride: uint32 option
        /// Bytes covered by the block (offset + array length * stride of its last member).
        PushBlockBytes: uint32 option
        LocalSize: uint32 * uint32 * uint32
        Capabilities: uint32 list
        Extensions: string list
    }

let private decodeString (w: uint32[]) (start: int) (count: int) =
    let bytes =
        [|
            for i in start .. start + count - 1 do
                yield! BitConverter.GetBytes w[i]
        |]

    let n = Array.IndexOf(bytes, 0uy)
    Text.Encoding.UTF8.GetString(bytes, 0, (if n < 0 then bytes.Length else n))

/// Parses a SPIR-V module. Storage classes: 12 StorageBuffer, 9 PushConstant, 2 Uniform, 0 UniformConstant.
let inspect (spv: byte[]) : SpirvFacts =
    let w = Native.words spv

    if w.Length < 5 || w[0] <> 0x07230203u then
        failwith "not a SPIR-V module"

    let names = Collections.Generic.Dictionary<uint32, string>()
    let decos = Collections.Generic.Dictionary<uint32, ResizeArray<uint32 * uint32 list>>()
    let memberOffsets = Collections.Generic.Dictionary<uint32, ResizeArray<uint32 * uint32>>()
    let pointers = Collections.Generic.Dictionary<uint32, uint32 * uint32>()
    let structs = Collections.Generic.Dictionary<uint32, uint32 list>()
    let arrays = Collections.Generic.Dictionary<uint32, uint32 * uint32>()
    let constants = Collections.Generic.Dictionary<uint32, uint32>()
    let vars = ResizeArray<uint32 * uint32 * uint32>()
    let caps = ResizeArray<uint32>()
    let exts = ResizeArray<string>()
    let mutable localSize = (0u, 0u, 0u)
    let mutable i = 5

    while i < w.Length do
        let count = int (w[i] >>> 16)
        let op = w[i] &&& 0xFFFFu
        let a k = w[i + k]

        match op with
        | 17u -> caps.Add(a 1)
        | 10u -> exts.Add(decodeString w (i + 1) (count - 1))
        | 5u -> names[a 1] <- decodeString w (i + 2) (count - 2)
        | 16u when a 2 = 17u -> localSize <- (a 3, a 4, a 5)
        | 71u ->
            let l =
                match decos.TryGetValue(a 1) with
                | true, l -> l
                | _ ->
                    let l = ResizeArray()
                    decos[a 1] <- l
                    l

            l.Add(a 2, [ for k in 3 .. count - 1 -> a k ])
        | 72u when a 3 = 35u ->
            let l =
                match memberOffsets.TryGetValue(a 1) with
                | true, l -> l
                | _ ->
                    let l = ResizeArray()
                    memberOffsets[a 1] <- l
                    l

            l.Add(a 2, a 4)
        | 32u -> pointers[a 1] <- (a 2, a 3)
        | 30u -> structs[a 1] <- [ for k in 2 .. count - 1 -> a k ]
        | 28u -> arrays[a 1] <- (a 2, a 3)
        | 43u when count >= 4 -> constants[a 2] <- a 3
        | 59u -> vars.Add(a 1, a 2, a 3)
        | _ -> ()

        i <- i + max count 1

    let deco (id: uint32) (d: uint32) =
        match decos.TryGetValue id with
        | true, l -> l |> Seq.tryFind (fun (k, _) -> k = d) |> Option.map snd
        | _ -> None

    let resources =
        [
            for (_, id, sc) in vars do
                if sc = 12u || sc = 2u || sc = 0u then
                    {
                        Id = id
                        Name = (match names.TryGetValue id with
                                | true, n -> n
                                | _ -> "")
                        StorageClass = sc
                        Set = deco id 34u |> Option.bind List.tryHead
                        Binding = deco id 33u |> Option.bind List.tryHead
                        NonWritable = (deco id 24u).IsSome
                    }
        ]

    let pushStruct =
        vars
        |> Seq.tryFind (fun (_, _, sc) -> sc = 9u)
        |> Option.bind (fun (ptrType, _, _) ->
            match pointers.TryGetValue ptrType with
            | true, (_, pointee) -> Some pointee
            | _ -> None)

    let offsets =
        match pushStruct with
        | Some s ->
            match memberOffsets.TryGetValue s with
            | true, l -> l |> Seq.sortBy fst |> Seq.map snd |> List.ofSeq
            | _ -> []
        | None -> []

    let firstMemberArray =
        pushStruct
        |> Option.bind (fun s ->
            match structs.TryGetValue s with
            | true, m :: _ -> Some m
            | _ -> None)

    let stride =
        firstMemberArray |> Option.bind (fun t -> deco t 6u) |> Option.bind List.tryHead

    let bytes =
        match firstMemberArray, stride, offsets with
        | Some t, Some s, [ off ] ->
            match arrays.TryGetValue t with
            | true, (_, lenId) ->
                match constants.TryGetValue lenId with
                | true, len -> Some(off + len * s)
                | _ -> None
            | _ -> None
        | _ -> None

    {
        Resources = resources
        PushMemberOffsets = offsets
        PushArrayStride = stride
        PushBlockBytes = bytes
        LocalSize = localSize
        Capabilities = List.ofSeq caps
        Extensions = List.ofSeq exts
    }
