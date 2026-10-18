namespace Lrm.Kernels.Conformance

open System
open System.Reflection
open Lrm.Core
open Lrm.Gpu.Core

/// Finds what the harness runs without naming it at compile time (ConformanceHarness contract, "Backends and
/// cases"): the GPU backends' providers in the backend assemblies this exe references, and the case lists of every
/// module under Lrm.Kernels.Conformance.Cases, so D1, V1 and K1 to K3 add a provider or a case file and nothing else.
[<RequireQualifiedAccess>]
module Discovery =
    /// The backend assembly of each GPU backend.
    let assemblies =
        [
            ExecutionBackend.D3D12, "Lrm.Gpu.D3D12"
            ExecutionBackend.Vulkan, "Lrm.Gpu.Vulkan"
        ]

    let private publicStatic = BindingFlags.Public ||| BindingFlags.Static

    /// The providers an assembly exposes: public static properties and fields of type IGpuBackendProvider (an F#
    /// module value is one), then public concrete types implementing it with a public parameterless constructor.
    let providersIn (asm: Assembly) : IGpuBackendProvider list =
        let types =
            try
                asm.GetExportedTypes()
            with :? ReflectionTypeLoadException as e ->
                e.Types |> Array.filter (fun t -> not (isNull t))

        // A member whose initialization throws (a missing native library, say) is skipped, never fatal: discovery runs
        // in every tier, the cpu tier on machines without a GPU included.
        let attempt (f: unit -> obj) =
            try
                [| f () |]
            with _ ->
                [||]

        let values =
            types
            |> Array.sortBy _.FullName
            |> Array.collect (fun t ->
                let props =
                    t.GetProperties publicStatic
                    |> Array.filter (fun p ->
                        typeof<IGpuBackendProvider>.IsAssignableFrom p.PropertyType
                        && p.GetIndexParameters().Length = 0)
                    |> Array.sortBy _.Name
                    |> Array.collect (fun p -> attempt (fun () -> p.GetValue null))

                let fields =
                    t.GetFields publicStatic
                    |> Array.filter (fun f -> typeof<IGpuBackendProvider>.IsAssignableFrom f.FieldType)
                    |> Array.sortBy _.Name
                    |> Array.collect (fun f -> attempt (fun () -> f.GetValue null))

                Array.append props fields)

        let constructed =
            types
            |> Array.filter (fun t ->
                t.IsClass
                && not t.IsAbstract
                && typeof<IGpuBackendProvider>.IsAssignableFrom t
                && not (isNull (t.GetConstructor Type.EmptyTypes)))
            |> Array.sortBy _.FullName
            |> Array.collect (fun t -> attempt (fun () -> Activator.CreateInstance t))

        Array.append values constructed
        |> Array.choose (function
            | :? IGpuBackendProvider as p -> Some p
            | _ -> None)
        |> List.ofArray

    /// The provider of a GPU backend, when its assembly has one yet (D1 and V1 add them in wave 2).
    let provider (backend: ExecutionBackend) : IGpuBackendProvider option =
        match assemblies |> List.tryFind (fun (b, _) -> b = backend) with
        | None -> None
        | Some(_, name) ->
            try
                providersIn (Assembly.Load(AssemblyName name))
                |> List.tryFind (fun p -> p.Backend = backend)
            with e ->
                eprintfn "conformance: no %s provider (%s: %s)" name (e.GetType().Name) e.Message
                None

    /// Every case list: the public `cases: KernelCase list` value of each module whose full name starts with
    /// Lrm.Kernels.Conformance.Cases., in ordinal order of the module names.
    let cases (asm: Assembly) : KernelCase list =
        asm.GetTypes()
        |> Array.filter (fun t ->
            not (isNull t.FullName)
            && t.FullName.StartsWith("Lrm.Kernels.Conformance.Cases.", StringComparison.Ordinal))
        |> Array.sortWith (fun a b -> String.CompareOrdinal(a.FullName, b.FullName))
        |> Array.collect (fun t ->
            match t.GetProperty("cases", publicStatic) with
            | null -> [||]
            | p when p.PropertyType = typeof<KernelCase list> -> (p.GetValue null :?> KernelCase list) |> Array.ofList
            | _ -> [||])
        |> List.ofArray
