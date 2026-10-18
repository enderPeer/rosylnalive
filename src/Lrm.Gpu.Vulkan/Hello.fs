namespace Lrm.Gpu.Vulkan

open System
open Lrm.Gpu.Vulkan.Interop
open Lrm.Core
open Lrm.Gpu.Core

/// `hello-vulkan`: creates the vulkan backend on the first accepted device (or `--index N`), loads the shader set,
/// runs the fp32 denormal probe and the five selftests once, and prints the device, the ExecutionClass and its id. The
/// test exe calls `run` (and the CLI can); it never takes a test mutex itself.
[<RequireQualifiedAccess>]
module Hello =
    /// The usage line.
    [<Literal>]
    let Usage = "usage: hello-vulkan [--index <device>]"

    /// The parsed arguments: Ok None for the first accepted device, Ok (Some i) for device i, Error for bad usage.
    let parseArgs (args: string[]) : Result<int option, string> =
        match args with
        | [||] -> Ok None
        | [| "--index"; i |] ->
            match Invariant.tryParseI32 i with
            | Some v when v >= 0 -> Ok(Some v)
            | _ -> Error(sprintf "bad device index '%s'" i)
        | _ -> Error Usage

    let private gib (bytes: int64) = float bytes / 1073741824.0

    let private detail (info: DeviceInfo) (key: string) =
        info.Details
        |> List.tryFind (fun (k, _) -> k = key)
        |> Option.map snd
        |> Option.defaultValue "?"

    /// Runs the hello with `write` for each output line. Exit codes: 0 every selftest case is bit-exact; 1 Vulkan is
    /// unavailable, the device failed or the backend raised; 2 usage; 3 no accepted device (or the requested one is
    /// rejected); 4 a selftest case failed or the probe could not run.
    let runWith (write: string -> unit) (shaders: ShaderSet) (args: string[]) : int =
        match parseArgs args with
        | Error e ->
            write e
            2
        | Ok index ->
            try
                let candidates = PhysicalDevices.enumerate ()

                if candidates.IsEmpty then
                    write "hello-vulkan: no Vulkan loader or no physical device"

                for c in candidates do
                    write (
                        sprintf
                            "device %d: %s, %s"
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
                    write "hello-vulkan: no accepted Vulkan device"
                    3
                | Some c ->
                    let options =
                        { DeviceOptions.For ExecutionBackend.Vulkan with
                            TimeDispatches = true
                        }

                    use backend = VulkanBackend.Create(c.Info, options)
                    let b = backend :> IGpuBackend
                    let suite = SelfTests.Suite.Create b
                    let set = b.LoadKernels shaders
                    let klass = b.ExecutionClass
                    let info = c.Info

                    write (
                        sprintf
                            "device           %s (device %d, %s, %.1f GiB, Vulkan %s)"
                            info.Name
                            info.Index
                            klass.DeviceId
                            (gib info.MemoryBytes)
                            (detail info "api")
                    )

                    write (sprintf "driver           %s (%s)" klass.Driver (detail info "driver_name"))
                    write ("execution class  " + ExecutionClass.describe klass)
                    write ("class id         " + string (ExecutionClass.id klass))

                    write (
                        sprintf
                            "shader set       spirv %s, %d kernels loaded, %d missing"
                            set.ShaderSetHash.Hex
                            ([ 0 .. Kernels.all.Length - 1 ]
                             |> List.filter (KernelId >> set.Contains)
                             |> List.length)
                            shaders.Missing.Length
                    )

                    let g = backend.Geometry

                    write (
                        sprintf
                            "queue            family %d (compute, no graphics), timeline semaphore, %s ns per tick"
                            backend.QueueFamily
                            (Invariant.f64 backend.TimestampPeriod)
                    )

                    write (
                        sprintf
                            "resource heap    %s, slot stride %d, reserved range %d + %d, heapRange %d bytes"
                            (match backend.HeapPlacement with
                             | HeapPlacement.InPlace -> "in place (DEVICE_LOCAL | HOST_VISIBLE)"
                             | HeapPlacement.Copied -> "copied (DEVICE_LOCAL)")
                            g.SlotStride
                            g.ReservedRangeOffset
                            g.ReservedRangeBytes
                            g.HeapBytes
                    )

                    write (
                        "memory types     "
                        + (backend.MemoryTypeChoices
                           |> List.map (fun (k, i, f) -> sprintf "%s %d (%s)" (BufferKind.name k) i f)
                           |> String.concat ", ")
                    )

                    write (
                        sprintf
                            "diagnostics      device fault %s, debug names %s, validation %s"
                            (if backend.DeviceFaultEnabled then "on" else "unavailable")
                            (if backend.DebugNames then "on" else "unavailable")
                            (if backend.ValidationEnabled then "on" else "not installed")
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
                write ("hello-vulkan: " + e.Message)
                1
            | :? VulkanDeviceLostException as e ->
                write ("hello-vulkan: " + e.Message)
                1
            | VulkanException(command, result) ->
                write (sprintf "hello-vulkan: %s failed: %A (%d)" command result (int result))
                1
            | e ->
                // A contract violation or a failed probe here is a bug in the backend or the shader set.
                write (sprintf "hello-vulkan: %s: %s" (e.GetType().Name) e.Message)
                1

    /// Runs the hello on the console (`hello-vulkan`).
    let run (shaders: ShaderSet) (args: string[]) : int = runWith (printfn "%s") shaders args
