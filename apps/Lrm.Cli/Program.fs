/// `lrm`: the single command-line entry point. Owner: S0.
/// Every Commands/*.fs module exposes `commands`; this file only concatenates them in compile order and dispatches.
module Lrm.Cli.Program

open Lrm.Cli.Commands

type private Command =
    {|
        Name: string
        Summary: string
        Run: string[] -> int
    |}

let private all: Command list =
    List.concat
        [
            DevicesVulkan.commands
            DevicesD3D12.commands
            Backends.commands
            GpuBackends.commands
            Forward.commands
            Generate.commands
            Report.commands
            Dashboard.commands
            Chaos.commands
            Coord.commands
            Node.commands
            Bisect.commands
            Gate1.commands
        ]

let private usage () =
    printfn "usage: dotnet Lrm.Cli.dll <command> [args]"
    printfn ""
    printfn "commands:"
    let width = all |> List.fold (fun w c -> max w c.Name.Length) 0

    for c in all do
        printfn "  %s  %s" (c.Name.PadRight width) c.Summary

[<EntryPoint>]
let main argv =
    Lrm.Core.Process.init ()

    let duplicates =
        all |> List.countBy _.Name |> List.filter (fun (_, n) -> n > 1) |> List.map fst

    if not duplicates.IsEmpty then
        eprintfn "lrm: duplicate command names: %s" (String.concat ", " duplicates)
        70
    else
        match List.ofArray argv with
        | []
        | [ "help" ]
        | [ "--help" ]
        | [ "-h" ] ->
            usage ()
            0
        | name :: rest ->
            match all |> List.tryFind (fun c -> c.Name = name) with
            | Some c -> c.Run(Array.ofList rest)
            | None ->
                eprintfn "lrm: unknown command '%s'" name
                usage ()
                2
