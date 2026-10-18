/// Lrm.Shaders.Build: the catalog generator, the shader compiler and the SPIR-V inspector (ShaderBuildAndSourceConvention
/// contract).
///
///     Lrm.Shaders.Build gen-catalog [--repo <root>] [--check]
///     Lrm.Shaders.Build compile --repo <root> --out <dir> --dxc <dxc.exe> --dxc-version <line> [--toolchain <json>]
///     Lrm.Shaders.Build inspect-spirv --dir <shaders dir> [--repo <root>] [--markdown <file>]
///     Lrm.Shaders.Build spirv-fixtures --dxc <dxc.exe> --dxc-version <line> [--repo <root>] [--toolchain <json>]
///
/// gen-catalog writes Kernels.g.fs and lrm_args.g.hlsli from catalog.json (--check only reports stale files).
/// compile is what the LrmCompileShaders target of Lrm.Shaders runs: it verifies the pinned DXC, lints every kernel
/// source and helper header, compiles each kernel to DXIL (cs_6_6) and SPIR-V (vulkan1.3) with a cache keyed on its
/// include closure, checks every blob (the SPIR-V inspector reads every SPIR-V blob) and the root signature, and
/// writes the blobs and shaders.manifest.json (with ShaderSetHash) for embedding.
/// inspect-spirv runs the SPIR-V inspector over a built set and prints (or writes) its Markdown table.
/// spirv-fixtures compiles the negative fixtures in src/Lrm.Shaders.Build/fixtures/spirv and requires each to break
/// exactly the rules its header names.
module Lrm.Shaders.Build.Program

open System
open System.IO
open Lrm.Gpu.Core
open Lrm.Shaders.Build.Compiler
open Lrm.Shaders.Build.SpirvChecks

let usage =
    String.concat
        "\n  "
        [
            "usage:"
            "Lrm.Shaders.Build gen-catalog [--repo <root>] [--check]"
            "Lrm.Shaders.Build compile --repo <root> --out <dir> --dxc <dxc.exe> --dxc-version <line> [--toolchain <json>]"
            "Lrm.Shaders.Build inspect-spirv --dir <shaders dir> [--repo <root>] [--markdown <file>]"
            "Lrm.Shaders.Build spirv-fixtures --dxc <dxc.exe> --dxc-version <line> [--repo <root>] [--toolchain <json>]"
        ]

let rec parseArgs (acc: Map<string, string>) (args: string list) =
    match args with
    | "--check" :: rest -> parseArgs (acc.Add("check", "true")) rest
    | name :: value :: rest when name.StartsWith "--" -> parseArgs (acc.Add(name.Substring 2, value)) rest
    | [] -> acc
    | a :: _ -> fail "unexpected argument '%s'" a

let findRepo (given: string option) =
    match given with
    | Some r -> Path.GetFullPath r
    | None ->
        let rec up (d: DirectoryInfo) =
            if isNull d then
                fail "no Lrm.slnx above %s; pass --repo" Environment.CurrentDirectory
            elif File.Exists(Path.Combine(d.FullName, "Lrm.slnx")) then
                d.FullName
            else
                up d.Parent

        up (DirectoryInfo Environment.CurrentDirectory)

let genCatalog (repo: string) (check: bool) =
    let rerun =
        "run: dotnet artifacts/bin/Lrm.Shaders.Build/release/Lrm.Shaders.Build.dll gen-catalog"

    if check then
        match CatalogGen.stale repo with
        | Error e -> fail "%s" e
        | Ok [] -> printfn "gen-catalog: generated files are current"
        | Ok files -> fail "generated files are stale: %s; %s" (String.concat ", " files) rerun
    else
        match CatalogGen.write repo with
        | Error e -> fail "%s" e
        | Ok [] -> printfn "gen-catalog: generated files are current"
        | Ok files -> printfn "gen-catalog: wrote %s" (String.concat ", " files)

/// inspect-spirv: the inspector over a built set; 0 when every blob passes.
let inspectSpirv (repo: string) (dir: string) (markdown: string option) =
    let reports = inspectDir repo dir
    let table = SpirvInspector.markdown reports
    printf "%s" table

    match markdown with
    | Some file -> writeText (Path.GetFullPath file) table
    | None -> ()

    let failed = reports |> List.filter (fun r -> not r.Passed)

    for r in failed do
        for f in r.Findings do
            eprintfn "%s: %s" r.Kernel (SpirvInspector.describe f)

    printfn
        "inspect-spirv: %d of %d SPIR-V blob(s) pass all %d rules"
        (reports.Length - failed.Length)
        reports.Length
        SpirvRule.all.Length

    if failed.IsEmpty && not reports.IsEmpty then 0 else 1

/// spirv-fixtures: 0 when every fixture breaks exactly the rules it expects.
let spirvFixtures (repo: string) (dxc: string) (dxcVersion: string) (toolchain: string option) =
    let results = runFixtures repo dxc dxcVersion toolchain

    for line in fixtureLines results do
        printfn "%s" line

    let failed = results |> List.filter (fun r -> not r.Passed)

    printfn
        "spirv-fixtures: %d of %d fixture(s) break exactly their expected rules"
        (results.Length - failed.Length)
        results.Length

    if failed.IsEmpty then 0 else 1

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()

    try
        let need (command: string) (a: Map<string, string>) (k: string) =
            match a.TryFind k with
            | Some v -> v
            | None -> fail "%s needs --%s" command k

        match List.ofArray argv with
        | "gen-catalog" :: rest ->
            let a = parseArgs Map.empty rest
            genCatalog (findRepo (a.TryFind "repo")) (a.ContainsKey "check")
            0
        | "compile" :: rest ->
            let a = parseArgs Map.empty rest
            let need = need "compile" a

            compile
                (findRepo (a.TryFind "repo"))
                (Path.GetFullPath(need "out"))
                (need "dxc")
                (need "dxc-version")
                (a.TryFind "toolchain")
            |> ignore

            0
        | "inspect-spirv" :: rest ->
            let a = parseArgs Map.empty rest

            inspectSpirv
                (findRepo (a.TryFind "repo"))
                (Path.GetFullPath(need "inspect-spirv" a "dir"))
                (a.TryFind "markdown")
        | "spirv-fixtures" :: rest ->
            let a = parseArgs Map.empty rest
            let need = need "spirv-fixtures" a
            spirvFixtures (findRepo (a.TryFind "repo")) (need "dxc") (need "dxc-version") (a.TryFind "toolchain")
        | _ ->
            eprintfn "%s" usage
            2
    with BuildError e ->
        // The canonical MSBuild error format, so the build log shows each line as an error.
        for line in e.Split('\n') do
            eprintfn "Lrm.Shaders.Build : error LRM0001: %s" line

        1
