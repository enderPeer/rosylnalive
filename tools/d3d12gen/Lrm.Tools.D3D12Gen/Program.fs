/// d3d12gen: generates the D3D12 and DXGI interop (D3D12ComIdiom contract) from the pinned Windows SDK headers.
///
///     dotnet run --project tools/d3d12gen/Lrm.Tools.D3D12Gen -c Release -- [--check] [--sdk <include dir>] [--repo <root>]
///
/// Exit codes: 0 done (or up to date with --check); 1 --check found stale files; 2 usage or generation error;
/// 4 the SDK headers are missing or differ from the hashes pinned in tools/d3d12gen/bindings.json.
module Lrm.Tools.D3D12Gen.Program

open System
open System.IO
open System.Security.Cryptography
open Lrm.Tools.D3D12Gen

type private Options =
    {
        Check: bool
        Sdk: string option
        Repo: string option
    }

let rec private parse (o: Options) =
    function
    | "--check" :: rest -> parse { o with Check = true } rest
    | "--sdk" :: d :: rest -> parse { o with Sdk = Some d } rest
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

let private sha256 (bytes: byte[]) =
    Convert.ToHexString(SHA256.HashData bytes).ToLowerInvariant()

/// The Windows Kits 10 root: the KitsRoot10 value the SDK installer writes, else Program Files (x86)\Windows Kits\10.
/// The ABI oracle (tests/Lrm.Gpu.D3D12.Interop.Tests/AbiOracle.fs) resolves it the same way.
let kitsRoot () : string =
    let fromRegistry =
        if OperatingSystem.IsWindows() then
            try
                use key =
                    Microsoft.Win32.Registry.LocalMachine.OpenSubKey @"SOFTWARE\Microsoft\Windows Kits\Installed Roots"

                match key with
                | null -> None
                | k ->
                    match k.GetValue "KitsRoot10" with
                    | :? string as s when s <> "" -> Some s
                    | _ -> None
            with _ ->
                None
        else
            None

    match fromRegistry with
    | Some r -> r
    | None -> Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.ProgramFilesX86, "Windows Kits", "10")

/// The servicing builds of the Windows SDK installed here ("10.0.26100.7705"), from the installer's uninstall
/// entries; empty when none is found. Only for the message when the pinned headers differ.
let private installedSdkBuilds () : string list =
    let prefix = "Windows Software Development Kit - Windows "

    if not (OperatingSystem.IsWindows()) then
        []
    else
        try
            [
                for view in
                    [
                        Microsoft.Win32.RegistryView.Registry64
                        Microsoft.Win32.RegistryView.Registry32
                    ] do
                    use hklm =
                        Microsoft.Win32.RegistryKey.OpenBaseKey(Microsoft.Win32.RegistryHive.LocalMachine, view)

                    use un = hklm.OpenSubKey @"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"

                    if not (isNull un) then
                        for name in un.GetSubKeyNames() do
                            use k = un.OpenSubKey name

                            match k.GetValue "DisplayName" with
                            | :? string as n when n.StartsWith prefix -> yield n.Substring(prefix.Length).Trim()
                            | _ -> ()
            ]
            |> List.distinct
            |> List.sort
        with _ ->
            []

/// The generated files, relative to the repo root.
let outputs =
    [
        "src/Lrm.Gpu.D3D12.Interop/D3D12.g.fs"
        "src/Lrm.Gpu.D3D12.Interop/D3D12Methods.g.fs"
    ]

let private run (o: Options) : int =
    match o.Repo |> Option.orElse (findRepo ()) with
    | None ->
        eprintfn "d3d12gen: cannot find the repo root (Lrm.slnx); pass --repo"
        2
    | Some repo ->
        let bindingsPath = Path.Combine(repo, "tools/d3d12gen/bindings.json")
        let b = Bind.loadBindings bindingsPath

        let sdk =
            o.Sdk
            |> Option.orElse (Option.ofObj (Environment.GetEnvironmentVariable "LRM_WINSDK_INCLUDE"))
            |> Option.defaultWith (fun () -> Path.Combine(kitsRoot (), "Include", b.SdkVersion))

        let problems =
            [
                for (rel, expected) in b.Headers do
                    let path = Path.Combine(sdk, rel)

                    if not (File.Exists path) then
                        sprintf "%s is missing" path
                    else
                        let actual = sha256 (File.ReadAllBytes path)

                        if actual <> expected then
                            sprintf "%s has SHA-256 %s, pinned %s" path actual expected
            ]

        if not problems.IsEmpty then
            for p in problems do
                eprintfn "d3d12gen: %s" p

            let here =
                match installedSdkBuilds () with
                | [] -> "no Windows SDK installer entry found here"
                | builds -> sprintf "installed here: %s" (String.concat ", " builds)

            eprintfn
                "d3d12gen: the SDK %s headers in %s differ from the pin, taken from Windows SDK %s (%s); nothing generated"
                b.SdkVersion
                sdk
                b.SdkServicing
                here

            4
        else
            let files =
                b.Headers
                |> List.map (fun (rel, _) -> rel, File.ReadAllText(Path.Combine(sdk, rel)))

            let c = Preprocessor.run false files
            let header = Header.parse (Lexer.tokenize c.Lines) c.Numeric
            let cpp = Preprocessor.run true files
            let cppIfaces = CppPass.scan (Lexer.tokenize cpp.Lines)
            let model = Bind.build b header cppIfaces
            let generated = [ Emit.types model; Emit.methods model ]

            let slots = model.Interfaces |> List.sumBy _.Slots.Length
            let declared = model.Interfaces |> List.sumBy _.Own.Length

            printfn
                "d3d12gen: SDK %s: %d interfaces (%d slots, %d declared methods), %d structs, %d inline arrays, %d enums"
                b.SdkVersion
                model.Interfaces.Length
                slots
                declared
                model.Structs.Length
                model.Arrays.Length
                model.Enums.Length

            let stale =
                [
                    for (rel, text) in List.zip outputs generated do
                        let path = Path.Combine(repo, rel)
                        let current = if File.Exists path then File.ReadAllText path else ""

                        if current <> text then
                            if o.Check then
                                eprintfn "d3d12gen: %s is stale; regenerate it" rel
                            else
                                File.WriteAllText(path, text)
                                printfn "d3d12gen: wrote %s" rel

                            yield rel
                ]

            if o.Check && not stale.IsEmpty then
                1
            else
                if stale.IsEmpty then
                    printfn "d3d12gen: generated files are up to date"

                0

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()

    match
        parse
            {
                Check = false
                Sdk = None
                Repo = None
            }
            (List.ofArray argv)
    with
    | Error e ->
        eprintfn "d3d12gen: %s" e
        2
    | Ok o ->
        try
            run o
        with e ->
            eprintfn "d3d12gen: %s" e.Message
            2
