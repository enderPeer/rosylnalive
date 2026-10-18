namespace Lrm.Gpu.D3D12

open System
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12.Interop

/// `hello-d3d12`: creates the d3d12 backend on the first accepted adapter (or `--index N`), loads the shader set,
/// runs the fp32 denormal probe and the five selftests once, and prints the device, the ExecutionClass and its id.
/// The CLI and the test exe both call `run`; it never takes a test mutex itself.
[<RequireQualifiedAccess>]
module Hello =
    /// The usage line.
    [<Literal>]
    let Usage = "usage: hello-d3d12 [--index <adapter>]"

    /// The parsed arguments: Ok None for the first accepted adapter, Ok (Some i) for adapter i, Error for bad usage.
    let parseArgs (args: string[]) : Result<int option, string> =
        match args with
        | [||] -> Ok None
        | [| "--index"; i |] ->
            match Invariant.tryParseI32 i with
            | Some v when v >= 0 -> Ok(Some v)
            | _ -> Error(sprintf "bad adapter index '%s'" i)
        | _ -> Error Usage

    let private gib (bytes: int64) = float bytes / 1073741824.0

    /// Runs the hello with `write` for each output line. Exit codes: 0 every selftest case is bit-exact; 1 D3D12 is
    /// unavailable, the device failed or the backend raised; 2 usage; 3 no accepted adapter (or the requested one is
    /// rejected); 4 a selftest case failed or the probe could not run.
    let runWith (write: string -> unit) (shaders: ShaderSet) (args: string[]) : int =
        match parseArgs args with
        | Error e ->
            write e
            2
        | Ok index ->
            try
                let candidates = Adapters.enumerate ()

                for c in candidates do
                    write (
                        sprintf
                            "adapter %d: %s, %s"
                            c.Info.Index
                            c.Info.Name
                            (if c.Accepted then
                                 "ACCEPTED"
                             else
                                 "REJECTED: " + String.concat "; " c.Rejections)
                    )

                let chosen =
                    match index with
                    | Some i -> candidates |> List.tryFind (fun c -> c.Info.Index = i && c.Accepted)
                    | None -> candidates |> List.tryFind _.Accepted

                match chosen with
                | None ->
                    write "hello-d3d12: no accepted D3D12 adapter"
                    3
                | Some c ->
                    let options =
                        { DeviceOptions.For ExecutionBackend.D3D12 with
                            TimeDispatches = true
                        }

                    use backend = D3D12Backend.Create(c.Info, options)
                    let b = backend :> IGpuBackend
                    let suite = SelfTests.Suite.Create b
                    let set = b.LoadKernels shaders
                    let klass = b.ExecutionClass
                    let info = c.Info

                    write (
                        sprintf
                            "device           %s (adapter %d, %s, %.1f GiB)"
                            info.Name
                            info.Index
                            klass.DeviceId
                            (gib info.MemoryBytes)
                    )

                    write (
                        sprintf
                            "driver           %s (%s)"
                            klass.Driver
                            (info.Details
                             |> List.tryFind (fun (k, _) -> k = "driver_version")
                             |> Option.map snd
                             |> Option.defaultValue "?")
                    )

                    write ("execution class  " + ExecutionClass.describe klass)
                    write ("class id         " + string (ExecutionClass.id klass))

                    let missing = shaders.Missing.Length

                    write (
                        sprintf
                            "shader set       dxil %s, %d kernels loaded, %d missing"
                            set.ShaderSetHash.Hex
                            ([ 0 .. Kernels.all.Length - 1 ]
                             |> List.filter (KernelId >> set.Contains)
                             |> List.length)
                            missing
                    )

                    write (
                        sprintf
                            "dred             %s"
                            (if backend.DredEnabled then
                                 "breadcrumbs and page faults forced on"
                             else
                                 "unavailable")
                    )

                    match backend.Fp32Denormals, backend.ProbeResults with
                    | Some mode, Some words ->
                        write (
                            sprintf
                                "fp32 denormals   %s (probe %s)"
                                (Fp32Denormals.name mode)
                                (Fp32Denormals.wordsText words)
                        )

                        let r = suite.Run(1, mode)

                        for x in r.Cases do
                            write (
                                sprintf
                                    "selftest         %-12s %-14s %s (%d elements%s)"
                                    x.Kernel
                                    x.Case
                                    (if x.Passed then "ok" else "FAILED")
                                    x.Elements
                                    (if x.Detail = "" then "" else "; " + x.Detail)
                            )

                        let gpuNs = r.Completion.Timers |> List.sumBy _.DurationNs
                        let passed = r.Cases |> List.filter _.Passed |> List.length

                        write (
                            sprintf
                                "gpu time         %.1f us for %d dispatches"
                                (float gpuNs / 1000.0)
                                r.Completion.Dispatches
                        )

                        if r.Passed then
                            write (sprintf "PASS: %d of %d selftest cases are bit-exact" passed r.Cases.Length)
                            0
                        else
                            write (sprintf "FAIL: %d of %d selftest cases are bit-exact" passed r.Cases.Length)
                            4
                    | _ ->
                        write "fp32 denormals   not measured: the shader set has no scale_f32"
                        4
            with
            | :? DllNotFoundException as e ->
                write ("hello-d3d12: " + e.Message)
                1
            | :? D3D12Exception as e ->
                write ("hello-d3d12: " + e.Message)
                1
            | :? D3D12DeviceLostException as e ->
                write ("hello-d3d12: " + e.Message)
                1
            | e ->
                // A contract violation or a failed probe here is a bug in the backend or the shader set.
                write (sprintf "hello-d3d12: %s: %s" (e.GetType().Name) e.Message)
                1

    /// Runs the hello on the console (the CLI's `lrm hello-d3d12`).
    let run (shaders: ShaderSet) (args: string[]) : int = runWith (printfn "%s") shaders args
