// Runs K2's matmul_bf16w bench (MatmulBench.fs) on every GPU backend with a provider, through Lrm.Testing's runner,
// which takes the tier's mutexes (bench: all of them; gpu: the backend's), until the conformance exe's tiers list it.
//
//   dotnet build Lrm.slnx -c Release
//   dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/bench.fsx [bench|gpu]
//
// bench (default): the throughput sweep and the cross-call checks; gpu: only the cross-call checks. Each prints its
// run log; `dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <run id>` renders the kernels report.
#I "../../../../artifacts/bin/Lrm.Kernels.Conformance/release"
#r "Lrm.Core.dll"
#r "Lrm.Testing.dll"
#r "Expecto.dll"
#r "Lrm.Kernels.Conformance.dll"

open Lrm.Testing
open Lrm.Kernels.Conformance.Cases

Lrm.Core.Process.init ()

let tier =
    match fsi.CommandLineArgs |> Array.tail with
    | [| "gpu" |] -> "gpu"
    | [||]
    | [| "bench" |] -> "bench"
    | other -> failwithf "usage: bench.fsx [bench|gpu], not %A" other

let tiers =
    { TestTiers.empty with
        Gpu = (if tier = "gpu" then MatmulBench.gpu () else [])
        Bench = (if tier = "bench" then MatmulBench.bench () else [])
    }

exit (Runner.main [| "--tier"; tier |] tiers)
