/// The SPIR-V inspector over a built shader set and over the negative fixtures (ShaderBuildAndSourceConvention v1.1).
/// check-shaders.ps1 runs both in CI after the Release build, whose compile step already inspects every blob.
module Lrm.Shaders.Build.SpirvChecks

open System
open System.IO
open System.Text
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Shaders.Build.Compiler

/// Inspects every SPIR-V blob of the shader set in `dir` (a LrmCompileShaders output folder) against the catalog of
/// `repo`. The set must have been built for that catalog.
let inspectDir (repo: string) (dir: string) : SpirvReport list =
    let manifest = Path.Combine(dir, ShaderSet.ManifestResource)

    if not (File.Exists manifest) then
        fail "%s is missing; build Lrm.Shaders first" manifest

    let set =
        try
            ShaderSet.ofManifest (File.ReadAllText(manifest, utf8)) (fun name ->
                File.ReadAllBytes(Path.Combine(dir, name)))
        with
        | :? GpuContractException as e -> fail "%s" e.Message
        | :? IOException as e -> fail "%s" e.Message

    let catalog = loadCatalog repo

    if set.CatalogHash <> Catalog.hash catalog then
        fail "the set in %s was built for catalog %s, not %s" dir set.CatalogHash.Hex (Catalog.hash catalog).Hex

    SpirvInspector.inspectSet catalog set

/// The folder of the negative fixtures, relative to the repo root.
[<Literal>]
let FixtureDir = "src/Lrm.Shaders.Build/fixtures/spirv"

/// One fixture: an HLSL file whose header names the catalog kernel it is inspected as, the rules its SPIR-V must
/// break (exactly; "none" for a positive control) and extra DXC arguments.
type Fixture =
    {
        File: string
        Kernel: string
        Expect: string list
        Flags: string list
    }

/// The outcome of one fixture.
type FixtureResult =
    {
        Fixture: Fixture
        /// The DXC output when the fixture did not compile.
        CompileError: string option
        Report: SpirvReport option
    }

    /// The broken rules, by name, in rule order.
    member r.Found =
        r.Report
        |> Option.map (SpirvInspector.rules >> List.map SpirvRule.name)
        |> Option.defaultValue []

    member r.Passed =
        r.CompileError.IsNone && List.sort r.Found = List.sort r.Fixture.Expect

/// Reads a fixture's header: `// kernel: <name>`, `// expect: <rule>, ...` (or `none`) and an optional
/// `// flags: <args>`, among the leading comment lines.
let parseFixture (file: string) (text: string) : Fixture =
    let header =
        text.Replace("\r\n", "\n").Split('\n')
        |> Seq.takeWhile (fun l -> l.StartsWith("//", StringComparison.Ordinal))
        |> Seq.map (fun l -> l.Substring(2).Trim())
        |> List.ofSeq

    let field (key: string) =
        header
        |> List.tryPick (fun l ->
            if l.StartsWith(key + ":", StringComparison.Ordinal) then
                Some(l.Substring(key.Length + 1).Trim())
            else
                None)

    let words (s: string) =
        s.Split([| ' '; ',' |], StringSplitOptions.RemoveEmptyEntries) |> List.ofArray

    let kernel =
        match field "kernel" with
        | Some k when k <> "" -> k
        | _ -> fail "%s: the header has no '// kernel:' line" file

    let expect =
        match field "expect" with
        | Some "none" -> []
        | Some e when e <> "" ->
            let rs = words e

            for r in rs do
                if (SpirvRule.tryParse r).IsNone then
                    fail
                        "%s: '%s' is not an inspector rule (%s)"
                        file
                        r
                        (SpirvRule.all |> List.map SpirvRule.name |> String.concat ", ")

            rs
        | _ -> fail "%s: the header has no '// expect:' line" file

    {
        File = file
        Kernel = kernel
        Expect = expect
        Flags = field "flags" |> Option.map words |> Option.defaultValue []
    }

/// Compiles every fixture with the pinned DXC and the SPIR-V flags (plus its own), then inspects it as its kernel.
let runFixtures (repo: string) (dxc: string) (expectedVersion: string) (toolchain: string option) =
    verifyDxc dxc expectedVersion toolchain |> ignore
    let catalog = loadCatalog repo
    let dir = Path.Combine(repo, FixtureDir)
    let includeDir = Path.Combine(repo, "src", "Lrm.Shaders", "include")

    let files =
        Directory.GetFiles(dir, "*.hlsl")
        |> Array.sortWith (fun a b -> String.CompareOrdinal(a, b))

    if files.Length = 0 then
        fail "no fixtures in %s" dir

    let scratch =
        Path.Combine(Path.GetTempPath(), "lrm-spirv-fixtures-" + Guid.NewGuid().ToString("n"))

    Directory.CreateDirectory scratch |> ignore

    try
        [
            for path in files do
                let name = Path.GetFileName path
                let fx = parseFixture name (File.ReadAllText(path, utf8))

                let kernel =
                    match Catalog.tryFind fx.Kernel catalog with
                    | Some k -> k
                    | None -> fail "%s: %s is not a catalog kernel" name fx.Kernel

                let out = Path.Combine(scratch, Path.ChangeExtension(name, ".spv"))

                let r =
                    run dxc dir (flagsOf ShaderTarget.Spirv @ fx.Flags @ [ "-I"; includeDir; "-Fo"; out; name ])

                if r.Code <> 0 then
                    {
                        Fixture = fx
                        CompileError = Some r.Output
                        Report = None
                    }
                else
                    {
                        Fixture = fx
                        CompileError = None
                        Report = Some(SpirvInspector.inspect kernel (File.ReadAllBytes out))
                    }
        ]
    finally
        Directory.Delete(scratch, true)

/// The fixture results as text lines, one per fixture plus its findings.
let fixtureLines (results: FixtureResult list) =
    [
        for r in results do
            let show (rules: string list) =
                if rules.IsEmpty then "none" else String.concat ", " rules

            yield
                sprintf
                    "%s %s (as %s): expect [%s], found [%s]"
                    (if r.Passed then "ok  " else "FAIL")
                    r.Fixture.File
                    r.Fixture.Kernel
                    (show r.Fixture.Expect)
                    (show r.Found)

            match r.CompileError with
            | Some e -> yield "       did not compile: " + e.Replace("\n", "\n       ")
            | None -> ()

            for f in r.Report |> Option.map _.Findings |> Option.defaultValue [] do
                yield "       " + SpirvInspector.describe f
    ]

/// Writes `text` with LF line ends and no BOM.
let writeText (path: string) (text: string) =
    File.WriteAllText(path, text.Replace("\r\n", "\n"), UTF8Encoding(false))
