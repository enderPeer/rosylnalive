module Lrm.Gpu.Vulkan.Interop.Tests.Main

open System.IO
open Lrm.Gpu.Vulkan.Interop.Tests
open Lrm.Testing

/// Root test lists per tier; tools/test.ps1 picks one via LRM_TEST_TIER.
let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = [ CpuTests.tests ]
        Gpu = [ Backend.Vulkan, [ GpuTests.tests ] ]
        Bench = [ BenchTests.tests ]
    }

/// `abi-oracle [--update] [--no-compile]`: dumps the runtime layout as the oracle sources, fails when the committed
/// ones differ (or rewrites them with --update), then compiles and runs them with cl.exe (tools/abi-oracle/vulkan).
/// Exit codes: 0 passed; 1 failed or stale; 2 no C++ toolchain or no Vulkan-Headers.
let private abiOracle (args: string list) =
    Lrm.Core.Process.init ()
    let repo = Env.repoRoot ()
    let dir = AbiOracle.oracleDir repo
    let update = List.contains "--update" args
    let noCompile = List.contains "--no-compile" args

    let current =
        if update then
            for p in AbiOracle.dump dir do
                printfn "abi-oracle: wrote %s" (Path.GetRelativePath(repo, p))

            true
        else
            match AbiOracle.stale repo with
            | [] -> true
            | files ->
                for f in files do
                    eprintfn
                        "abi-oracle: tools/abi-oracle/vulkan/%s is stale; run pwsh tools/abi-oracle/vulkan/run.ps1 -Update"
                        f

                false

    if not current then
        1
    elif noCompile then
        0
    else
        match AbiOracle.findToolchain (), AbiOracle.headers repo with
        | None, _ ->
            eprintfn "abi-oracle: no Visual Studio C++ toolchain (vswhere)"
            2
        | _, Error e ->
            eprintfn "abi-oracle: %s" e
            2
        | Some tc, Ok include' ->
            let out = Path.Combine(repo, "artifacts", "abi-oracle", "vulkan", "run")
            let r = AbiOracle.compileAndRunIn tc include' dir out

            // The wrapper checks match parameters by name: the raw delegates' names must be vulkan_core.h's.
            let names =
                AbiOracle.parameterNameProblems (File.ReadAllText(Path.Combine(include', "vulkan", "vulkan_core.h")))

            for p in names do
                eprintfn "abi-oracle: %s" p

            if r.ExitCode <> 0 || not r.Failures.IsEmpty then
                eprintfn "%s" r.Output

                for f in r.Failures do
                    eprintfn "abi-oracle: %s" f

                1
            elif not names.IsEmpty then
                1
            else
                printfn "%s" (defaultArg r.Summary "abi-oracle: passed")

                printfn
                    "abi-oracle: the parameter names of %d delegates match vulkan_core.h's PFN_* typedefs"
                    (AbiOracle.delegateMethods ()).Length

                0

[<EntryPoint>]
let main argv =
    match List.ofArray argv with
    | "abi-oracle" :: rest -> abiOracle rest
    | _ ->
        let code = Runner.main argv tiers
        // The gpu tier's run log, if it ran.
        Common.finishRun code
        code
