/// Lrm.Shaders.Build: the catalog generator and the shader compiler (ShaderBuildAndSourceConvention contract).
///
///     Lrm.Shaders.Build gen-catalog [--repo <root>] [--check]
///     Lrm.Shaders.Build compile --repo <root> --out <dir> --dxc <dxc.exe> --dxc-version <line> [--toolchain <json>]
///
/// gen-catalog writes Kernels.g.fs and lrm_args.g.hlsli from catalog.json (--check only reports stale files).
/// compile is what the LrmCompileShaders target of Lrm.Shaders runs: it verifies the pinned DXC, lints every kernel
/// source and helper header, compiles each kernel to DXIL (cs_6_6) and SPIR-V (vulkan1.3) with a cache keyed on its
/// include closure, checks every blob and the root signature, and writes the blobs and shaders.manifest.json (with
/// ShaderSetHash) for embedding.
module Lrm.Shaders.Build.Program

open System
open System.IO
open Lrm.Gpu.Core
open Lrm.Shaders.Build.Compiler

let usage =
    "usage:\n  Lrm.Shaders.Build gen-catalog [--repo <root>] [--check]\n  Lrm.Shaders.Build compile --repo <root> --out <dir> --dxc <dxc.exe> --dxc-version <line> [--toolchain <json>]"

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

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()

    try
        match List.ofArray argv with
        | "gen-catalog" :: rest ->
            let a = parseArgs Map.empty rest
            genCatalog (findRepo (a.TryFind "repo")) (a.ContainsKey "check")
            0
        | "compile" :: rest ->
            let a = parseArgs Map.empty rest

            let need (k: string) =
                match a.TryFind k with
                | Some v -> v
                | None -> fail "compile needs --%s" k

            compile
                (findRepo (a.TryFind "repo"))
                (Path.GetFullPath(need "out"))
                (need "dxc")
                (need "dxc-version")
                (a.TryFind "toolchain")
            |> ignore

            0
        | _ ->
            eprintfn "%s" usage
            2
    with BuildError e ->
        // The canonical MSBuild error format, so the build log shows each line as an error.
        for line in e.Split('\n') do
            eprintfn "Lrm.Shaders.Build : error LRM0001: %s" line

        1
