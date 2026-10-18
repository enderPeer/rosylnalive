namespace Lrm.Tensor

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Lrm.Core

/// A golden manifest that breaks the GoldenSet rules (ComparatorsAndGoldenReader contract).
type GoldenFormatException(message: string) =
    inherit FormatException("golden: " + message)

/// Where a golden file lives (GoldenSet contract).
[<RequireQualifiedAccess>]
type GoldenLocation =
    /// Committed under goldens/<set>/<version>/.
    | Repo
    /// Under LRM_DATA/golden/<set>/<version>/, never committed.
    | Data

/// One file of a golden set, as its manifest lists it.
type GoldenFile =
    {
        /// The path as written in the manifest: repo-relative, or relative to LRM_DATA/golden for data files.
        Path: string
        /// The name inside the set folder, for example "fwd_f32.p0.safetensors" or "model/config.json".
        Name: string
        Location: GoldenLocation
        Bytes: int64
        Sha256: ContentId
        Role: string
        /// The resolved location on this machine.
        FullPath: string
    }

/// An opened golden set (ComparatorsAndGoldenReader contract). The manifest is validated on open; every file is
/// checked against its manifest size and SHA-256 when it is read, before any byte is used.
[<Sealed>]
type GoldenSet
    internal (dir: string, set: string, version: string, kind: string, manifest: JsonElement, files: GoldenFile[]) =
    let byName =
        let d = Dictionary<string, GoldenFile>(StringComparer.Ordinal)

        for f in files do
            d[f.Name] <- f

        d

    let verified (f: GoldenFile) (passed: bool) =
        Telemetry.check
            "goldens.verify"
            passed
            [
                Attr.str "set" (set + "/" + version)
                Attr.str "file" f.Name
                Attr.int64 "bytes" f.Bytes
                Attr.str "sha256" f.Sha256.Hex
            ]

    /// The set folder.
    member _.Dir = dir

    /// The set name, for example "tiny".
    member _.Set = set

    /// The version folder, for example "g1".
    member _.Version = version

    /// The manifest's kind: "tiny-model", "tokenizer-corpus" or "full-model".
    member _.Kind = kind

    /// The whole manifest. Readers ignore fields they do not know.
    member _.Manifest = manifest

    /// Every file the manifest lists, sorted by path.
    member _.Files = Array.copy files

    member _.TryFile(name: string) =
        match byName.TryGetValue name with
        | true, f -> Some f
        | _ -> None

    /// The manifest entry for a file name inside the set folder; raises KeyNotFoundException.
    member this.File(name: string) =
        match this.TryFile name with
        | Some f -> f
        | None -> raise (KeyNotFoundException(sprintf "golden %s/%s lists no file '%s'" set version name))

    /// Opens a safetensors file after checking its size and SHA-256 over the mapped bytes (IntegrityException on a
    /// mismatch). Emits a goldens.verify Check.
    member this.OpenSafetensors(name: string) : SafetensorsFile =
        let f = this.File name

        try
            let file = SafetensorsFile.OpenVerified(f.FullPath, f.Bytes, f.Sha256)
            verified f true
            file
        with :? IntegrityException ->
            verified f false
            reraise ()

    /// Reads a whole file after checking its size and SHA-256 (IntegrityException on a mismatch). The returned bytes are
    /// the verified bytes. Emits a goldens.verify Check.
    member this.ReadAllBytes(name: string) : byte[] =
        let f = this.File name

        let bytes =
            use s = new FileStream(f.FullPath, FileMode.Open, FileAccess.Read, FileShare.Read)

            if s.Length <> f.Bytes then
                verified f false

                raise (
                    IntegrityException(
                        f.FullPath,
                        Invariant.format $"{f.Bytes} bytes",
                        Invariant.format $"{s.Length} bytes"
                    )
                )

            let b = Array.zeroCreate<byte>(int s.Length)
            s.ReadExactly(b, 0, b.Length)
            b

        let actual = Hash.sha256 bytes

        if actual <> f.Sha256 then
            verified f false
            raise (IntegrityException(f.FullPath, "sha256 " + f.Sha256.Hex, "sha256 " + actual.Hex))

        verified f true
        bytes

    /// Checks every listed file's size and SHA-256; raises IntegrityException at the first mismatch.
    member this.VerifyAll() =
        for f in files do
            this.ReadAllBytes f.Name |> ignore

/// Opening golden sets (ComparatorsAndGoldenReader contract).
[<AbstractClass; Sealed>]
type Goldens =
    static member private Fail(msg: string) : 'a = raise (GoldenFormatException msg)

    static member private Str(o: JsonElement, key: string, where: string) =
        match o.TryGetProperty key with
        | true, v when v.ValueKind = JsonValueKind.String -> v.GetString()
        | _ -> Goldens.Fail(sprintf "%s: '%s' is missing or not a string" where key)

    /// Opens goldens/<set>/<version> (the folder holding manifest.json). Validates the manifest: schema lrm.golden/1,
    /// set and version equal to the folder names, and a well-formed, sorted, duplicate-free file list whose paths stay
    /// inside the set. Repo files resolve under the folder; data files under `dataRoot`/golden (LRM_DATA by default).
    static member Open(dir: string, ?dataRoot: string) : GoldenSet =
        let dir = Path.GetFullPath(Path.TrimEndingDirectorySeparator dir)
        let manifestPath = Path.Combine(dir, "manifest.json")

        let doc =
            try
                JsonDocument.Parse(File.ReadAllBytes manifestPath)
            with :? JsonException as e ->
                Goldens.Fail(sprintf "%s is not valid JSON: %s" manifestPath e.Message)

        use doc = doc
        let root = doc.RootElement.Clone()

        if root.ValueKind <> JsonValueKind.Object then
            Goldens.Fail(sprintf "%s is not a JSON object" manifestPath)

        let schema = Goldens.Str(root, "schema", manifestPath)

        if schema <> "lrm.golden/1" then
            Goldens.Fail(sprintf "%s: unknown schema '%s' (expected lrm.golden/1)" manifestPath schema)

        let set = Goldens.Str(root, "set", manifestPath)
        let version = Goldens.Str(root, "version", manifestPath)
        let kind = Goldens.Str(root, "kind", manifestPath)
        let folderVersion = Path.GetFileName dir
        let folderSet = Path.GetFileName(Path.GetDirectoryName dir)

        if set <> folderSet || version <> folderVersion then
            Goldens.Fail(sprintf "%s names %s/%s but sits in %s/%s" manifestPath set version folderSet folderVersion)

        if
            version.Length < 2
            || version[0] <> 'g'
            || not (version.Substring 1 |> Seq.forall Char.IsAsciiDigit)
            || version[1] = '0'
        then
            Goldens.Fail(sprintf "%s: version '%s' is not g<N>" manifestPath version)

        let goldenData =
            match dataRoot with
            | Some r -> Path.Combine(Path.GetFullPath r, "golden")
            | None -> LrmPaths.golden ()

        let filesElement =
            match root.TryGetProperty "files" with
            | true, v when v.ValueKind = JsonValueKind.Array -> v
            | _ -> Goldens.Fail(sprintf "%s: 'files' is missing or not an array" manifestPath)

        let repoPrefix = sprintf "goldens/%s/%s/" set version
        let dataPrefix = sprintf "%s/%s/" set version

        let files =
            [|
                for e in filesElement.EnumerateArray() do
                    if e.ValueKind <> JsonValueKind.Object then
                        Goldens.Fail(sprintf "%s: a files entry is not an object" manifestPath)

                    let path = Goldens.Str(e, "path", manifestPath + " files")
                    let location = Goldens.Str(e, "location", path)
                    let role = Goldens.Str(e, "role", path)
                    let shaText = Goldens.Str(e, "sha256", path)

                    let bytes =
                        match e.TryGetProperty "bytes" with
                        | true, v when v.ValueKind = JsonValueKind.Number ->
                            match v.TryGetInt64() with
                            | true, b when b >= 0L -> b
                            | _ -> Goldens.Fail(sprintf "%s: bytes is not a non-negative integer" path)
                        | _ -> Goldens.Fail(sprintf "%s: bytes is missing" path)

                    let sha =
                        if shaText.Length = 64 && shaText = shaText.ToLowerInvariant() then
                            match ContentId.tryParse shaText with
                            | Some id -> id
                            | None -> Goldens.Fail(sprintf "%s: sha256 '%s' is not 64 hex digits" path shaText)
                        else
                            Goldens.Fail(sprintf "%s: sha256 '%s' is not 64 lowercase hex digits" path shaText)

                    let location, prefix, baseDir =
                        match location with
                        | "repo" -> GoldenLocation.Repo, repoPrefix, dir
                        | "data" -> GoldenLocation.Data, dataPrefix, Path.Combine(goldenData, set, version)
                        | other -> Goldens.Fail(sprintf "%s: unknown location '%s'" path other)

                    if not (path.StartsWith(prefix, StringComparison.Ordinal)) then
                        Goldens.Fail(sprintf "%s: a %A file must be under %s" path location prefix)

                    let name = path.Substring prefix.Length
                    let segments = name.Split('/')

                    if
                        name.Length = 0
                        || name = "manifest.json"
                        || segments
                           |> Array.exists (fun s -> s = "" || s = "." || s = ".." || s.Contains '\\' || s.Contains ':')
                    then
                        Goldens.Fail(sprintf "%s: not a file name inside the set" path)

                    {
                        Path = path
                        Name = name
                        Location = location
                        Bytes = bytes
                        Sha256 = sha
                        Role = role
                        FullPath = Path.Combine(Array.append [| baseDir |] segments)
                    }
            |]

        let paths = files |> Array.map _.Path

        for i in 1 .. paths.Length - 1 do
            let c = Safetensors.codePointCompare paths[i - 1] paths[i]

            if c = 0 then
                Goldens.Fail(sprintf "%s: duplicate path %s" manifestPath paths[i])
            elif c > 0 then
                Goldens.Fail(
                    sprintf "%s: files are not sorted by path (%s before %s)" manifestPath paths[i - 1] paths[i]
                )

        if files |> Array.distinctBy _.Name |> Array.length <> files.Length then
            Goldens.Fail(sprintf "%s: a repo file and a data file share a name" manifestPath)

        GoldenSet(dir, set, version, kind, root, files)

    /// goldens/tiny/g1 under a repo root.
    static member TinyDir(repoRoot: string) =
        Path.Combine(repoRoot, "goldens", "tiny", "g1")

/// ActivationNames helpers (the names are O1's contract; these functions follow it).
[<RequireQualifiedAccess>]
module ActivationNames =
    /// The 18 per-layer taps, in canonical order.
    let layerTaps =
        [|
            "input_layernorm.out"
            "attn.q_proj.out"
            "attn.k_proj.out"
            "attn.v_proj.out"
            "attn.q_norm.out"
            "attn.k_norm.out"
            "attn.q_rope"
            "attn.k_rope"
            "attn.v_cache"
            "attn.o_proj.in"
            "attn.o_proj.out"
            "resid_mid"
            "post_attention_layernorm.out"
            "mlp.gate.out"
            "mlp.up.out"
            "mlp.down.in"
            "mlp.down.out"
            "out"
        |]

    /// Every tap of a model with `layers` layers, in canonical order: 18 * layers + 3 names.
    let all (layers: int) =
        [|
            yield "embed.out"
            for i in 0 .. layers - 1 do
                for t in layerTaps do
                    yield Invariant.format $"L{i}.{t}"
            yield "final_norm.out"
            yield "logits"
        |]

    /// The layer of a tap ("L3.attn.q_rope" is 3), or -1 for the model-level taps.
    let layer (name: string) =
        if name.Length > 1 && name[0] = 'L' && Char.IsAsciiDigit name[1] then
            let dot = name.IndexOf '.'

            if dot < 0 then
                invalidArg "name" (sprintf "'%s' is not a tap name" name)

            Invariant.parseI32 (name.Substring(1, dot - 1))
        else
            -1

    /// The tap kind: the part after "L{i}." ("attn.q_rope"), or the whole model-level name ("logits").
    let kind (name: string) =
        if layer name >= 0 then
            name.Substring(name.IndexOf '.' + 1)
        else
            name
