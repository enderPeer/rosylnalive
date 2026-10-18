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

    /// What discovery found in a backend assembly: its providers, and every error raised on the way (a getter, a module
    /// initializer or a constructor that threw, a type that failed to load, the assembly itself), so that a broken
    /// provider fails a test instead of quietly leaving its backend without conformance tests.
    type Found =
        {
            Providers: IGpuBackendProvider list
            Errors: string list
        }

    /// The innermost exception's type and message (reflection wraps an initializer's exception twice).
    let private describe (e: exn) =
        let rec inner (e: exn) =
            match e with
            | :? TargetInvocationException
            | :? TypeInitializationException when not (isNull e.InnerException) -> inner e.InnerException
            | _ -> e

        let i = inner e
        sprintf "%s: %s" (i.GetType().Name) i.Message

    /// The providers an assembly exposes, with the errors met: public static properties and fields of type
    /// IGpuBackendProvider (an F# module value is one), then public concrete types implementing it with a public
    /// parameterless constructor. An error never stops the search: discovery runs in every tier, the cpu tier on
    /// machines without a GPU included, and the gpu tier turns the errors into a failing test.
    let discoverIn (asm: Assembly) : Found =
        let errors = ResizeArray<string>()

        let types =
            try
                asm.GetExportedTypes()
            with :? ReflectionTypeLoadException as e ->
                for l in e.LoaderExceptions do
                    if not (isNull l) then
                        errors.Add(sprintf "%s: a type failed to load (%s)" (asm.GetName().Name) (describe l))

                e.Types |> Array.filter (fun t -> not (isNull t))

        let attempt (what: string) (f: unit -> obj) =
            try
                [| f () |]
            with e ->
                errors.Add(sprintf "%s threw %s" what (describe e))
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
                    |> Array.collect (fun p -> attempt (t.FullName + "." + p.Name) (fun () -> p.GetValue null))

                let fields =
                    t.GetFields publicStatic
                    |> Array.filter (fun f -> typeof<IGpuBackendProvider>.IsAssignableFrom f.FieldType)
                    |> Array.sortBy _.Name
                    |> Array.collect (fun f -> attempt (t.FullName + "." + f.Name) (fun () -> f.GetValue null))

                Array.append props fields)

        let constructed =
            types
            |> Array.filter (fun t ->
                t.IsClass
                && not t.IsAbstract
                && typeof<IGpuBackendProvider>.IsAssignableFrom t
                && not (isNull (t.GetConstructor Type.EmptyTypes)))
            |> Array.sortBy _.FullName
            |> Array.collect (fun t -> attempt ("new " + t.FullName + "()") (fun () -> Activator.CreateInstance t))

        {
            Providers =
                Array.append values constructed
                |> Array.choose (function
                    | :? IGpuBackendProvider as p -> Some p
                    | _ -> None)
                |> List.ofArray
            Errors = List.ofSeq errors
        }

    /// The providers an assembly exposes (discoverIn without the errors).
    let providersIn (asm: Assembly) : IGpuBackendProvider list = (discoverIn asm).Providers

    /// What discovery finds for a GPU backend: the provider of its assembly when it has one yet (D1 and V1 add them in
    /// wave 2), and the errors met on the way, the assembly's load included.
    let discover (backend: ExecutionBackend) : IGpuBackendProvider option * string list =
        match assemblies |> List.tryFind (fun (b, _) -> b = backend) with
        | None -> None, []
        | Some(_, name) ->
            try
                let found = discoverIn (Assembly.Load(AssemblyName name))
                found.Providers |> List.tryFind (fun p -> p.Backend = backend), found.Errors
            with e ->
                None, [ sprintf "loading %s threw %s" name (describe e) ]

    /// The provider of a GPU backend, when its assembly has one yet; the errors are printed.
    let provider (backend: ExecutionBackend) : IGpuBackendProvider option =
        let p, errors = discover backend

        for e in errors do
            eprintfn "conformance: %s provider discovery: %s" (ExecutionBackend.name backend) e

        p

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
