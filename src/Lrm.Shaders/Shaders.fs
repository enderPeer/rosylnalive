namespace Lrm.Shaders

open System.IO
open System.Text
open Lrm.Gpu.Core

/// The shader set that the LrmCompileShaders target compiled with the pinned DXC and embedded in this assembly
/// (ShaderBuildAndSourceConvention contract). Backends take it through IGpuBackend.LoadKernels.
[<RequireQualifiedAccess>]
module Shaders =
    type private Marker = class end

    /// The prefix of every embedded shader resource.
    [<Literal>]
    let ResourcePrefix = "lrm.shaders/"

    let private assembly = typeof<Marker>.Assembly

    /// The embedded resource names below the prefix, sorted.
    let resourceNames () =
        assembly.GetManifestResourceNames()
        |> Array.filter (fun n -> n.StartsWith(ResourcePrefix, System.StringComparison.Ordinal))
        |> Array.map (fun n -> n.Substring ResourcePrefix.Length)
        |> Array.sort

    /// The bytes of one embedded resource, such as "dxil/scale_f32.dxil" or "rootsig.bin".
    let readResource (name: string) : byte[] =
        use s = assembly.GetManifestResourceStream(ResourcePrefix + name)

        if isNull s then
            raise (GpuContractException(sprintf "Lrm.Shaders has no embedded resource %s%s" ResourcePrefix name))

        use ms = new MemoryStream()
        s.CopyTo ms
        ms.ToArray()

    let private set =
        lazy (ShaderSet.ofManifest (Encoding.UTF8.GetString(readResource ShaderSet.ManifestResource)) readResource)

    /// The embedded shader set, read once. Loading verifies every blob's SHA-256, the canonical root signature and
    /// both ShaderSetHash values (ShaderSet.ofManifest).
    let embedded () = set.Force()

    /// The ShaderSetHash of one target of the embedded set.
    let shaderSetHash (target: ShaderTarget) = ShaderSet.setHash target (embedded ())

    /// The embedded manifest text.
    let manifestText () =
        Encoding.UTF8.GetString(readResource ShaderSet.ManifestResource)
