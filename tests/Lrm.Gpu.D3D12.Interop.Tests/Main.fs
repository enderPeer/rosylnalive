module Lrm.Gpu.D3D12.Interop.Tests.Main

open System.IO
open Lrm.Testing

/// Root test lists per tier; tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = [ CpuTests.tests ]
        Gpu = [ Backend.D3D12, [ GpuTests.tests ] ]
    }

/// `abi-oracle [--update] [--no-compile]`: the ABI oracle outside the test runner (tools/abi-oracle/d3d12/run.ps1).
/// Dumps the runtime layout, compares it with the committed sources (or rewrites them with --update), then compiles
/// and runs the oracle with cl.exe. Exit codes: 0 passed, 1 failed or stale, 2 no C++ toolchain.
let private abiOracle (args: string list) =
    Lrm.Core.Process.init ()
    let repo = Env.repoRoot ()
    let dir = AbiOracle.oracleDir repo
    let update = List.contains "--update" args

    let stale =
        if update then
            AbiOracle.dump dir |> List.iter (printfn "abi-oracle: wrote %s")
            []
        else
            AbiOracle.stale repo

    if not stale.IsEmpty then
        for s in stale do
            eprintfn "abi-oracle: %s differs from the runtime layout; regenerate with run.ps1 -Update" s

        1
    elif List.contains "--no-compile" args then
        printfn "abi-oracle: sources are current"
        0
    else
        match AbiOracle.findToolchain () with
        | None ->
            eprintfn "abi-oracle: no Visual Studio C++ toolchain found (vswhere)"
            2
        | Some tc ->
            printfn "abi-oracle: %s, Windows SDK %s" tc.Vcvars (defaultArg tc.SdkVersion "(vcvars default)")

            let r =
                AbiOracle.compileAndRun tc repo (Path.Combine(repo, "artifacts", "abi-oracle", "d3d12"))

            if r.ExitCode <> 0 then
                eprintfn "%s" r.Output
                eprintfn "abi-oracle: FAILED (exit %d, %d failure lines)" r.ExitCode r.Failures.Length
                1
            else
                printfn "%s" (defaultArg r.Summary "abi-oracle: passed")
                0

[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "abi-oracle" :: rest -> abiOracle rest
    | _ -> Runner.main argv tiers
