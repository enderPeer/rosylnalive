namespace Lrm.Core

open System
open System.Runtime.InteropServices
open System.Runtime.Intrinsics.X86

/// The backend that executes a computation (ExecutionClass contract).
[<RequireQualifiedAccess>]
type ExecutionBackend =
    /// The F# CPU reference in float32.
    | CpuF32
    /// The F# CPU reference in float64.
    | CpuF64
    /// The CPU emulation of the GPU kernel catalog.
    | CpuDevice
    | Vulkan
    | D3D12
    /// Direct3D 12 on the WARP software rasterizer.
    | D3D12Warp

[<RequireQualifiedAccess>]
module ExecutionBackend =
    /// Every backend, in declaration order.
    let all =
        [
            ExecutionBackend.CpuF32
            ExecutionBackend.CpuF64
            ExecutionBackend.CpuDevice
            ExecutionBackend.Vulkan
            ExecutionBackend.D3D12
            ExecutionBackend.D3D12Warp
        ]

    /// The frozen name used in encodings, manifests and reports.
    let name (backend: ExecutionBackend) =
        match backend with
        | ExecutionBackend.CpuF32 -> "cpu-f32"
        | ExecutionBackend.CpuF64 -> "cpu-f64"
        | ExecutionBackend.CpuDevice -> "cpu-device"
        | ExecutionBackend.Vulkan -> "vulkan"
        | ExecutionBackend.D3D12 -> "d3d12"
        | ExecutionBackend.D3D12Warp -> "d3d12-warp"

    /// Parses a name (ordinal match).
    let tryParse (text: string) =
        all |> List.tryFind (fun b -> Invariant.equals (name b) text)

    /// True for the backends that run compiled shaders (vulkan, d3d12, d3d12-warp).
    let runsShaders (backend: ExecutionBackend) =
        match backend with
        | ExecutionBackend.Vulkan
        | ExecutionBackend.D3D12
        | ExecutionBackend.D3D12Warp -> true
        | ExecutionBackend.CpuF32
        | ExecutionBackend.CpuF64
        | ExecutionBackend.CpuDevice -> false

/// Everything that can change the bits a computation produces (ExecutionClass contract). Bit-identity claims hold only
/// between runs of one class; across classes, results are compared with tolerances. GPU backends fill DeviceId,
/// Driver and ShaderSetHash by R4a's rule; CPU backends use ExecutionClass.forCpu.
type ExecutionClass =
    {
        Backend: ExecutionBackend
        /// The device model, stable across machines with the same hardware ("cpu" for CPU backends).
        DeviceId: string
        /// The driver or runtime version that executes the work.
        Driver: string
        /// The hash of the compiled shader set; required for backends that run shaders.
        ShaderSetHash: ContentId option
        /// The architecture and instruction-set features that can change results.
        Arch: string
    }

/// The id of an ExecutionClass: Canonical.hash "lrm.xclass/1" over its fields in declared order.
[<Struct>]
type ExecutionClassId =
    | ExecutionClassId of ContentId

    /// The underlying ContentId.
    member x.Value =
        let (ExecutionClassId id) = x
        id

    override x.ToString() = x.Value.Hex

[<RequireQualifiedAccess>]
module ExecutionClass =
    /// The domain tag of ExecutionClassId.
    [<Literal>]
    let Tag = "lrm.xclass/1"

    let private checkText (field: string) (value: string) =
        if isNull value then
            Some(sprintf "%s is null" field)
        elif value.Length = 0 || value.Length > 256 then
            Some(sprintf "%s must be 1 to 256 characters" field)
        elif value |> Seq.exists Char.IsControl then
            Some(sprintf "%s contains a control character" field)
        elif value.Trim().Length <> value.Length then
            Some(sprintf "%s has leading or trailing white space" field)
        else
            None

    /// The class when it is well formed: DeviceId, Driver and Arch are 1 to 256 characters with no control characters
    /// and no surrounding white space, and backends that run shaders carry a ShaderSetHash.
    let validate (c: ExecutionClass) : Result<ExecutionClass, string> =
        let problems =
            [
                checkText "DeviceId" c.DeviceId
                checkText "Driver" c.Driver
                checkText "Arch" c.Arch
                if ExecutionBackend.runsShaders c.Backend && c.ShaderSetHash.IsNone then
                    Some(sprintf "%s requires a ShaderSetHash" (ExecutionBackend.name c.Backend))
                else
                    None
            ]
            |> List.choose id

        match problems with
        | [] -> Ok c
        | ps -> Error(String.concat "; " ps)

    /// The canonical payload: backend name (string), DeviceId, Driver, ShaderSetHash (option of 32 raw bytes), Arch.
    let encode (w: CanonicalWriter) (c: ExecutionClass) =
        w.String(ExecutionBackend.name c.Backend)
        w.String c.DeviceId
        w.String c.Driver
        w.Option(c.ShaderSetHash, w.Id)
        w.String c.Arch

    /// The class id. Raises ArgumentException for a class that validate rejects.
    let id (c: ExecutionClass) : ExecutionClassId =
        match validate c with
        | Error e -> invalidArg "c" (sprintf "invalid ExecutionClass: %s" e)
        | Ok c -> ExecutionClassId(Canonical.hash Tag (fun w -> encode w c))

    /// True when two classes have the same id, so bit-identity claims between them are meaningful.
    let same (a: ExecutionClass) (b: ExecutionClass) = id a = id b

    /// One line for logs and reports.
    let describe (c: ExecutionClass) =
        let shaders =
            match c.ShaderSetHash with
            | Some h -> " shaders " + h.Short
            | None -> ""

        sprintf "%s %s, %s, %s%s" (ExecutionBackend.name c.Backend) c.DeviceId c.Driver c.Arch shaders

    /// The host's architecture string for CPU classes: OS and process architecture, plus the x86 features the C
    /// runtime's math functions dispatch on, for example "win-x64+avx2+fma".
    let hostArch () =
        let os =
            if OperatingSystem.IsWindows() then "win"
            elif OperatingSystem.IsLinux() then "linux"
            elif OperatingSystem.IsMacOS() then "osx"
            else "other"

        let arch = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant()

        let features =
            [
                if Avx2.IsSupported then
                    "+avx2"
                if Fma.IsSupported then
                    "+fma"
                if Avx512F.IsSupported then
                    "+avx512f"
            ]

        os + "-" + arch + String.concat "" features

    /// The class of a CPU backend on this host: DeviceId "cpu", Driver the .NET runtime description (for example
    /// ".NET 10.0.0"), no ShaderSetHash, Arch from hostArch.
    let forCpu (backend: ExecutionBackend) : ExecutionClass =
        if ExecutionBackend.runsShaders backend then
            invalidArg "backend" (sprintf "%s is not a CPU backend" (ExecutionBackend.name backend))

        {
            Backend = backend
            DeviceId = "cpu"
            Driver = RuntimeInformation.FrameworkDescription.Trim()
            ShaderSetHash = None
            Arch = hostArch ()
        }
