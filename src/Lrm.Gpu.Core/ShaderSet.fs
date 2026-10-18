namespace Lrm.Gpu.Core

open System
open System.Buffers.Binary
open System.IO
open System.Text
open System.Text.Json
open Lrm.Core

/// A compiled shader form (ShaderBuildAndSourceConvention contract).
[<RequireQualifiedAccess>]
type ShaderTarget =
    /// DXIL, cs_6_6, for D3D12 and WARP.
    | Dxil
    /// SPIR-V, vulkan1.3, for Vulkan.
    | Spirv

[<RequireQualifiedAccess>]
module ShaderTarget =
    let all = [ ShaderTarget.Dxil; ShaderTarget.Spirv ]

    let name (t: ShaderTarget) =
        match t with
        | ShaderTarget.Dxil -> "dxil"
        | ShaderTarget.Spirv -> "spirv"

    let tryParse (text: string) =
        all
        |> List.tryFind (fun t -> String.Equals(name t, text, StringComparison.Ordinal))

    /// The blob file extension.
    let extension (t: ShaderTarget) =
        match t with
        | ShaderTarget.Dxil -> ".dxil"
        | ShaderTarget.Spirv -> ".spv"

    /// The target a backend runs: dxil for d3d12 and d3d12-warp, spirv for vulkan, none for CPU backends.
    let forBackend (backend: ExecutionBackend) =
        match backend with
        | ExecutionBackend.D3D12
        | ExecutionBackend.D3D12Warp -> Some ShaderTarget.Dxil
        | ExecutionBackend.Vulkan -> Some ShaderTarget.Spirv
        | ExecutionBackend.CpuF32
        | ExecutionBackend.CpuF64
        | ExecutionBackend.CpuDevice -> None

/// One compiled kernel.
type ShaderBlob =
    {
        Kernel: string
        Target: ShaderTarget
        Bytes: byte[]
        Sha256: ContentId
        /// numthreads, read back from the blob (DXIL PSV0, SPIR-V LocalSize) by the build.
        Threads: int * int * int
    }

/// The compiled shader set embedded in Lrm.Shaders: every kernel whose source exists, in both targets, plus the
/// canonical root signature. Built only by the pinned DXC.
type ShaderSet =
    {
        /// `dxc --version` of the compiler that built the set.
        DxcVersion: string
        /// Catalog.hash of the catalog the kernels were built against.
        CatalogHash: ContentId
        /// The DXC arguments per target (without input, output and include paths).
        Flags: (ShaderTarget * string list) list
        /// rootsig.bin (-T rootsig_1_1), equal to RootSignature.canonical.
        RootSignature: byte[]
        /// Blobs sorted by target, then kernel name (ordinal).
        Blobs: ShaderBlob list
        /// Catalog kernels whose source does not exist yet, in catalog order.
        Missing: string list
    }

/// The global root signature of the D3D12 backend (ShaderBuildAndSourceConvention contract).
[<RequireQualifiedAccess>]
module RootSignature =
    /// The root signature string, compiled with -T rootsig_1_1 -E LRM_ROOTSIG.
    [<Literal>]
    let Source =
        "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=32, b0)"

    /// The canonical rootsig.bin: a 92-byte DXBC container with one RTS0 part (version 1.1, flags 0x400
    /// CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED, one 32-bit-constants parameter: b0, space 0, 32 values, visibility all).
    [<Literal>]
    let CanonicalHex =
        "44584243" // "DXBC"
        + "14462d179f897b3ba160caa35fcb04a6" // container digest
        + "01000000" // version 1.0
        + "5c000000" // 92 bytes
        + "01000000" // 1 part
        + "24000000" // part 0 at byte 36
        + "52545330" // "RTS0"
        + "30000000" // 48 bytes
        + "02000000" // D3D_ROOT_SIGNATURE_VERSION_1_1
        + "01000000" // 1 parameter
        + "18000000" // parameters at byte 24
        + "00000000" // 0 static samplers
        + "30000000" // samplers at byte 48
        + "00040000" // flags: CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
        + "01000000" // parameter 0: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
        + "00000000" // shader visibility: all
        + "24000000" // payload at byte 36
        + "00000000" // shader register b0
        + "00000000" // register space 0
        + "20000000" // 32 values

    /// The canonical bytes (a fresh copy).
    let canonical () = Convert.FromHexString CanonicalHex

    /// True when `bytes` equal the canonical root signature.
    let isCanonical (bytes: byte[]) =
        not (isNull bytes)
        && ReadOnlySpan(bytes).SequenceEqual(ReadOnlySpan(canonical ()))

/// One part of a DXBC container.
type DxbcPart =
    {
        FourCC: string
        Offset: int
        Size: int
    }

/// Minimal reader of DXBC containers (DXIL blobs and rootsig.bin).
[<RequireQualifiedAccess>]
module Dxbc =
    let private u32 (b: byte[]) (o: int) =
        int (BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(b, o, 4)))

    /// The parts of a container: "DXBC", a 16-byte digest, version 1.0, the total size equal to the length, and
    /// every part inside the container.
    let parts (b: byte[]) : Result<DxbcPart list, string> =
        if isNull b || b.Length < 32 then
            Error "too short for a DXBC container"
        elif Encoding.ASCII.GetString(b, 0, 4) <> "DXBC" then
            Error "no DXBC magic"
        elif
            BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan(b, 20, 2)) <> 1us
            || BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan(b, 22, 2)) <> 0us
        then
            Error "container version is not 1.0"
        elif u32 b 24 <> b.Length then
            Error(sprintf "container size %d differs from the length %d" (u32 b 24) b.Length)
        else
            let n = u32 b 28

            if n < 0 || 32 + 4 * n > b.Length then
                Error "bad part count"
            else
                let ps =
                    [
                        for i in 0 .. n - 1 do
                            let o = u32 b (32 + 4 * i)

                            if o >= 0 && o + 8 <= b.Length then
                                let size = u32 b (o + 4)

                                if size >= 0 && o + 8 + size <= b.Length then
                                    yield
                                        Some
                                            {
                                                FourCC = Encoding.ASCII.GetString(b, o, 4)
                                                Offset = o + 8
                                                Size = size
                                            }
                                else
                                    yield None
                            else
                                yield None
                    ]

                if ps |> List.exists Option.isNone then
                    Error "a part lies outside the container"
                else
                    Ok(ps |> List.choose id)

    /// True when the container digest (bytes 4..19) is not all zero, which DXC's validator writes when it signs.
    let isSigned (b: byte[]) =
        not (isNull b) && b.Length >= 20 && b[4..19] |> Array.exists ((<>) 0uy)

    /// numthreads from the PSV0 part (PSVRuntimeInfo2 or later: NumThreadsX/Y/Z at bytes 36..47 of the runtime info).
    let numThreads (b: byte[]) : (int * int * int) option =
        match parts b with
        | Ok ps ->
            ps
            |> List.tryFind (fun p -> p.FourCC = "PSV0")
            |> Option.bind (fun p ->
                let infoSize = u32 b p.Offset

                if p.Size >= 4 + 48 && infoSize >= 48 then
                    let o = p.Offset + 4
                    Some(u32 b (o + 36), u32 b (o + 40), u32 b (o + 44))
                else
                    None)
        | Error _ -> None

/// Minimal reader of SPIR-V modules.
[<RequireQualifiedAccess>]
module Spirv =
    /// The SPIR-V magic number.
    [<Literal>]
    let Magic = 0x07230203u

    /// True for a whole number of words starting with the magic number.
    let isModule (b: byte[]) =
        not (isNull b)
        && b.Length >= 20
        && b.Length % 4 = 0
        && BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(b, 0, 4)) = Magic

    /// The LocalSize execution mode (OpExecutionMode, opcode 16, mode 17), if there is exactly one.
    let localSize (b: byte[]) : (int * int * int) option =
        if not (isModule b) then
            None
        else
            let word i =
                BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(b, 4 * i, 4))

            let n = b.Length / 4
            let found = ResizeArray<int * int * int>()
            let mutable i = 5
            let mutable ok = true

            while ok && i < n do
                let w = word i
                let count = int (w >>> 16)
                let opcode = w &&& 0xFFFFu

                if count < 1 || i + count > n then
                    ok <- false
                else
                    if opcode = 16u && count >= 6 && word (i + 2) = 17u then
                        found.Add(int (word (i + 3)), int (word (i + 4)), int (word (i + 5)))

                    i <- i + count

            if ok && found.Count = 1 then Some found[0] else None

[<RequireQualifiedAccess>]
module ShaderSet =
    /// The domain tag of ShaderSetHash.
    [<Literal>]
    let Tag = "lrm.shaderset/1"

    /// The manifest schema.
    [<Literal>]
    let Schema = "lrm.shaders/1"

    /// The embedded resource name of a blob: "<target>/<kernel><ext>", for example "dxil/scale_f32.dxil".
    let resourceName (target: ShaderTarget) (kernel: string) =
        ShaderTarget.name target + "/" + kernel + ShaderTarget.extension target

    /// The embedded resource name of the manifest.
    [<Literal>]
    let ManifestResource = "shaders.manifest.json"

    /// The embedded resource name of the root signature.
    [<Literal>]
    let RootSignatureResource = "rootsig.bin"

    /// The DXC arguments for `target` in this set.
    let flagsFor (target: ShaderTarget) (s: ShaderSet) =
        s.Flags
        |> List.tryFind (fun (t, _) -> t = target)
        |> Option.map snd
        |> Option.defaultValue []

    /// The blobs of one target, sorted by kernel name (ordinal).
    let blobsFor (target: ShaderTarget) (s: ShaderSet) =
        s.Blobs
        |> List.filter (fun b -> b.Target = target)
        |> List.sortWith (fun a b -> String.CompareOrdinal(a.Kernel, b.Kernel))

    /// The blob of `kernel` for `target`.
    let tryBlob (kernel: string) (target: ShaderTarget) (s: ShaderSet) =
        s.Blobs |> List.tryFind (fun b -> b.Kernel = kernel && b.Target = target)

    /// ShaderSetHash of one target: Canonical.hash "lrm.shaderset/1" over the target name, the DXC version line, the
    /// catalog hash, the target's DXC arguments, the root signature's SHA-256 (DXIL only, else None) and the
    /// (kernel name, blob SHA-256) pairs sorted by kernel name. The ExecutionClass of a backend carries the hash of
    /// the target it runs.
    let setHash (target: ShaderTarget) (s: ShaderSet) : ContentId =
        Canonical.hash Tag (fun w ->
            w.String(ShaderTarget.name target)
            w.String s.DxcVersion
            w.Id s.CatalogHash
            w.Seq(flagsFor target s, fun f -> w.String f)

            w.Option(
                (if target = ShaderTarget.Dxil then
                     Some(Hash.sha256 s.RootSignature)
                 else
                     None),
                w.Id
            )

            w.Seq(
                blobsFor target s,
                fun b ->
                    w.String b.Kernel
                    w.Id b.Sha256
            ))

    /// The manifest text: deterministic JSON (LF line ends, no timestamps, no paths), schema lrm.shaders/1.
    let manifestJson (s: ShaderSet) : string =
        use ms = new MemoryStream()

        do
            use w =
                new Utf8JsonWriter(
                    ms,
                    JsonWriterOptions(
                        Indented = true,
                        NewLine = "\n",
                        Encoder = Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                    )
                )

            w.WriteStartObject()
            w.WriteString("schema", Schema)
            w.WriteString("dxcVersion", s.DxcVersion)
            w.WriteString("catalogHash", s.CatalogHash.Hex)
            w.WriteStartObject("rootSignature")
            w.WriteString("resource", RootSignatureResource)
            w.WriteNumber("bytes", s.RootSignature.Length)
            w.WriteString("sha256", (Hash.sha256 s.RootSignature).Hex)
            w.WriteEndObject()
            w.WriteStartObject("targets")

            for t in ShaderTarget.all do
                w.WriteStartObject(ShaderTarget.name t)
                w.WriteString("shaderSetHash", (setHash t s).Hex)
                w.WriteStartArray("flags")

                for f in flagsFor t s do
                    w.WriteStringValue f

                w.WriteEndArray()
                w.WriteStartArray("kernels")

                for b in blobsFor t s do
                    w.WriteStartObject()
                    w.WriteString("name", b.Kernel)
                    w.WriteString("resource", resourceName t b.Kernel)
                    w.WriteNumber("bytes", b.Bytes.Length)
                    w.WriteString("sha256", b.Sha256.Hex)
                    let tx, ty, tz = b.Threads
                    w.WriteStartArray("threads")
                    w.WriteNumberValue tx
                    w.WriteNumberValue ty
                    w.WriteNumberValue tz
                    w.WriteEndArray()
                    w.WriteEndObject()

                w.WriteEndArray()
                w.WriteEndObject()

            w.WriteEndObject()
            w.WriteStartArray("missing")

            for m in s.Missing do
                w.WriteStringValue m

            w.WriteEndArray()
            w.WriteEndObject()

        Encoding.UTF8.GetString(ms.ToArray()) + "\n"

    /// Reads a manifest and its resources. Raises GpuContractException when the schema is wrong, a blob's SHA-256
    /// differs from the manifest, the root signature is not canonical, or a recomputed ShaderSetHash differs.
    let ofManifest (json: string) (readResource: string -> byte[]) : ShaderSet =
        use doc = JsonDocument.Parse json
        let r = doc.RootElement

        let str (e: JsonElement) (name: string) =
            match e.TryGetProperty name with
            | true, v when v.ValueKind = JsonValueKind.String -> v.GetString()
            | _ -> Fail.contract "shader manifest: '%s' is missing or not a string" name

        if str r "schema" <> Schema then
            Fail.contract "shader manifest: schema is not %s" Schema

        let dxc = str r "dxcVersion"
        let catalogHash = ContentId.parse (str r "catalogHash")
        let rsElem = r.GetProperty "rootSignature"
        let rs = readResource (str rsElem "resource")

        if (Hash.sha256 rs).Hex <> str rsElem "sha256" then
            Fail.contract "shader manifest: rootsig.bin does not match its SHA-256"

        if not (RootSignature.isCanonical rs) then
            Fail.contract "shader manifest: rootsig.bin is not the canonical root signature"

        let targets = r.GetProperty "targets"

        let perTarget =
            ShaderTarget.all
            |> List.map (fun t ->
                let e = targets.GetProperty(ShaderTarget.name t)
                let flags = [ for f in e.GetProperty("flags").EnumerateArray() -> f.GetString() ]

                let blobs =
                    [
                        for k in e.GetProperty("kernels").EnumerateArray() do
                            let name = str k "name"
                            let bytes = readResource (str k "resource")
                            let sha = Hash.sha256 bytes

                            if sha.Hex <> str k "sha256" then
                                Fail.contract "shader manifest: %s does not match its SHA-256" (resourceName t name)

                            let th = [ for x in k.GetProperty("threads").EnumerateArray() -> x.GetInt32() ]

                            match th with
                            | [ x; y; z ] ->
                                yield
                                    {
                                        Kernel = name
                                        Target = t
                                        Bytes = bytes
                                        Sha256 = sha
                                        Threads = (x, y, z)
                                    }
                            | _ -> Fail.contract "shader manifest: %s threads must have 3 numbers" name
                    ]

                t, flags, blobs, str e "shaderSetHash")

        let set =
            {
                DxcVersion = dxc
                CatalogHash = catalogHash
                Flags = perTarget |> List.map (fun (t, f, _, _) -> t, f)
                RootSignature = rs
                Blobs = perTarget |> List.collect (fun (_, _, b, _) -> b)
                Missing = [ for m in r.GetProperty("missing").EnumerateArray() -> m.GetString() ]
            }

        for t, _, _, recorded in perTarget do
            if (setHash t set).Hex <> recorded then
                Fail.contract "shader manifest: the %s ShaderSetHash does not reproduce" (ShaderTarget.name t)

        set
