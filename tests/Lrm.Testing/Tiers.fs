namespace Lrm.Testing

open Expecto

/// Test tiers (TestTiers contract). CI runs only Cpu; Bench is local and never gates.
[<RequireQualifiedAccess>]
type Tier =
    | Cpu
    | Gpu
    | Model
    | Bench

[<RequireQualifiedAccess>]
module Tier =
    let all = [ Tier.Cpu; Tier.Gpu; Tier.Model; Tier.Bench ]

    let name =
        function
        | Tier.Cpu -> "cpu"
        | Tier.Gpu -> "gpu"
        | Tier.Model -> "model"
        | Tier.Bench -> "bench"

    let tryParse (s: string) =
        all |> List.tryFind (fun t -> name t = s.Trim().ToLowerInvariant())

/// GPU backends, declared in mutex acquisition order.
[<RequireQualifiedAccess>]
type Backend =
    | D3D12
    | Vulkan

[<RequireQualifiedAccess>]
module Backend =
    let all = [ Backend.D3D12; Backend.Vulkan ]

    let name =
        function
        | Backend.D3D12 -> "d3d12"
        | Backend.Vulkan -> "vulkan"

    let tryParse (s: string) =
        all |> List.tryFind (fun b -> name b = s.Trim().ToLowerInvariant())

    /// Parses a comma-separated list such as LRM_REQUIRED_BACKENDS ("d3d12,vulkan").
    let parseList (s: string) : Result<Backend list, string> =
        let parts =
            s.Split([| ','; ';'; ' ' |], System.StringSplitOptions.RemoveEmptyEntries)
            |> List.ofArray

        match parts |> List.filter (tryParse >> Option.isNone) with
        | [] -> Ok(parts |> List.choose tryParse |> List.distinct |> List.sort)
        | bad -> Error(sprintf "unknown backend(s) %s; expected %s" (String.concat ", " bad) "d3d12, vulkan")

/// The root test lists of one test exe. Every exe declares all four tiers, even when a list is empty.
type TestTiers =
    {
        /// Pure CPU tests. CI runs this tier with --fail-on-focused-tests.
        Cpu: Test list
        /// GPU tests grouped by backend. The runner holds that backend's mutex while the group runs.
        Gpu: (Backend * Test list) list
        /// Full-model tests. CPU groups (None) hold Local\LrmModelCpu; GPU groups also hold their backend mutex.
        Model: (Backend option * Test list) list
        /// Local timing measurements; informational only. Runs while holding every test mutex.
        Bench: Test list
    }

[<RequireQualifiedAccess>]
module TestTiers =
    /// No tests in any tier; the starting point for a shell.
    let empty =
        {
            Cpu = []
            Gpu = []
            Model = []
            Bench = []
        }
