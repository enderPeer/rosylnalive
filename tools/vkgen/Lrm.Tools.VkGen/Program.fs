/// vkgen: generates the Vulkan interop (VulkanInteropApi contract) from the pinned Khronos registry (vk.xml) and
/// tools/vkgen/bindings.json.
///
///     dotnet run --project tools/vkgen/Lrm.Tools.VkGen -c Release -- [--check] [--vkxml <path>] [--repo <root>]
///
/// The vk.xml pin (version, size, SHA-256 and install path under LRM_DATA) is the vkxml entry of
/// tools/bootstrap/toolchain.json; `pwsh tools/bootstrap/bootstrap.ps1` installs it.
///
/// Exit codes: 0 done (or up to date with --check); 1 --check found stale files; 2 usage or generation error;
/// 4 vk.xml is missing or differs from the pin.
module Lrm.Tools.VkGen.Program

open System
open System.IO
open System.Security.Cryptography
open System.Text.Json
open Lrm.Tools.VkGen

type private Options =
    {
        Check: bool
        VkXml: string option
        Repo: string option
    }

let rec private parse (o: Options) =
    function
    | "--check" :: rest -> parse { o with Check = true } rest
    | "--vkxml" :: p :: rest -> parse { o with VkXml = Some p } rest
    | "--repo" :: d :: rest -> parse { o with Repo = Some d } rest
    | [] -> Ok o
    | a :: _ -> Error(sprintf "unknown argument '%s'" a)

let private findRepo () =
    let rec up (d: DirectoryInfo) =
        if isNull d then
            None
        elif File.Exists(Path.Combine(d.FullName, "Lrm.slnx")) then
            Some d.FullName
        else
            up d.Parent

    up (DirectoryInfo(Directory.GetCurrentDirectory()))
    |> Option.orElse (up (DirectoryInfo(AppContext.BaseDirectory)))

/// The pin of one toolchain entry: (version, size, sha256, install path relative to LRM_DATA).
let private toolchainPin (repo: string) (entry: string) =
    let path = Path.Combine(repo, "tools", "bootstrap", "toolchain.json")
    use doc = JsonDocument.Parse(File.ReadAllText path)
    let e = doc.RootElement.GetProperty entry

    let sha =
        match e.GetProperty("sha256").ValueKind with
        | JsonValueKind.String -> e.GetProperty("sha256").GetString()
        | _ -> failwithf "toolchain.json: %s has no sha256 pin" entry

    e.GetProperty("version").GetString(), e.GetProperty("size").GetInt64(), sha, e.GetProperty("install").GetString()

/// The generated files, in the output directory.
let outputNames = [ "Vk.g.fs"; "VkCommands.g.fs"; "VkMeta.g.fs" ]

let private run (o: Options) : int =
    match o.Repo |> Option.orElse (findRepo ()) with
    | None ->
        eprintfn "vkgen: cannot find the repo root (Lrm.slnx); pass --repo"
        2
    | Some repo ->
        let bindings =
            Bind.loadBindings (Path.Combine(repo, "tools", "vkgen", "bindings.json"))

        let version, size, sha, install = toolchainPin repo bindings.ToolchainEntry

        let vkxml =
            o.VkXml |> Option.defaultWith (fun () -> Lrm.Core.LrmPaths.under install)

        let problem =
            if not (File.Exists vkxml) then
                Some(sprintf "%s is missing (run: pwsh tools/bootstrap/bootstrap.ps1)" vkxml)
            else
                let bytes = File.ReadAllBytes vkxml
                let actual = Convert.ToHexString(SHA256.HashData bytes).ToLowerInvariant()

                if int64 bytes.Length <> size || actual <> sha then
                    Some(
                        sprintf
                            "%s has %d bytes and SHA-256 %s; the pin (tools/bootstrap/toolchain.json, vk.xml %s) is %d bytes, %s"
                            vkxml
                            bytes.Length
                            actual
                            version
                            size
                            sha
                    )
                else
                    None

        match problem with
        | Some p ->
            eprintfn "vkgen: %s; nothing generated" p
            4
        | None ->
            let registry = Registry.load vkxml
            let model = Bind.build registry bindings
            let generated = [ Emit.types model; Emit.commands model; Emit.meta model ]

            printfn
                "vkgen: vk.xml %s (VK_HEADER_VERSION %d): %d structs and unions, %d inline arrays, %d enums, %d handles, %d commands (%d global, %d instance, %d device), %d extensions"
                version
                registry.HeaderVersion
                model.Structs.Length
                model.Arrays.Length
                model.Enums.Length
                model.Handles.Length
                model.Commands.Length
                bindings.GlobalCommands.Length
                bindings.InstanceCommands.Length
                bindings.DeviceCommands.Length
                model.BoundExtensions.Length

            let stale =
                [
                    for (name, text) in List.zip outputNames generated do
                        let rel = bindings.OutputDirectory + "/" + name
                        let path = Path.Combine(repo, rel)
                        let current = if File.Exists path then File.ReadAllText path else ""

                        if current <> text then
                            if o.Check then
                                eprintfn "vkgen: %s is stale; regenerate it" rel
                            else
                                File.WriteAllText(path, text)
                                printfn "vkgen: wrote %s" rel

                            yield rel
                ]

            if o.Check && not stale.IsEmpty then
                1
            else
                if stale.IsEmpty then
                    printfn "vkgen: generated files are up to date"

                0

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()

    match
        parse
            {
                Check = false
                VkXml = None
                Repo = None
            }
            (List.ofArray argv)
    with
    | Error e ->
        eprintfn "vkgen: %s" e
        2
    | Ok o ->
        try
            run o
        with e ->
            eprintfn "vkgen: %s" e.Message
            2
