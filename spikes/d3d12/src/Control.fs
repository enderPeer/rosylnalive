/// The denormal control of ADR int-0001 on WARP: scale_f32 compiled now with the pinned DXC and the shader set's DXIL
/// flags without "-denorm preserve", put in place of the set's scale_f32 blob, so D1's own probe (which runs inside
/// LoadKernels) measures what WARP does when the blob declares no denormal mode. On the RTX 4080 the flag decides the
/// mode (preserve with it, flush without).
module Lrm.Spikes.D3D12.Control

open System
open System.Diagnostics
open System.IO
open System.Text.Json
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Testing

/// The DXIL flags without "-denorm <mode>".
let rec stripDenorm (flags: string list) =
    match flags with
    | "-denorm" :: _ :: rest -> stripDenorm rest
    | x :: rest -> x :: stripDenorm rest
    | [] -> []

/// The pinned DXC (tools/bootstrap/toolchain.json under LRM_DATA).
let dxcExe () =
    use toolchain =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Env.repoRoot (), "tools", "bootstrap", "toolchain.json")))

    Path.Combine(Env.dataRoot (), toolchain.RootElement.GetProperty("dxc").GetProperty("exe").GetString())

/// scale_f32.hlsl compiled with `flags`: Ok (flags text, blob) or Error with the reason (DXC missing or failing).
let compileScale (flags: string list) : Result<string * byte[], string> =
    let exe = dxcExe ()

    if not (File.Exists exe) then
        Error("the pinned DXC is missing: " + exe)
    else
        let out =
            Path.Combine(Path.GetTempPath(), sprintf "lrm-r5-denorm-control-%d.dxil" Environment.ProcessId)

        let psi =
            ProcessStartInfo(
                exe,
                WorkingDirectory = Path.Combine(Env.repoRoot (), "src", "Lrm.Shaders"),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            )

        for a in flags @ [ "-I"; "include"; "-Fo"; out; "hlsl/selftest/scale_f32.hlsl" ] do
            psi.ArgumentList.Add a

        use p = Process.Start psi
        let stdout = p.StandardOutput.ReadToEndAsync()
        let stderr = p.StandardError.ReadToEnd()
        p.WaitForExit()
        stdout.Wait()

        if p.ExitCode <> 0 then
            Error(sprintf "dxc failed with %d: %s" p.ExitCode stderr)
        else
            let bytes = File.ReadAllBytes out
            File.Delete out
            Ok(String.concat " " flags, bytes)

/// `set` with its DXIL scale_f32 blob replaced by `blob`, and its DXIL flags by `flags` (so the DXIL ShaderSetHash, and
/// with it the class, differs from the set's). Every other blob, the catalog hash and the root signature are kept.
let withScale (set: ShaderSet) (flags: string list) (blob: byte[]) : ShaderSet =
    { set with
        Flags =
            set.Flags
            |> List.map (fun (t, f) -> if t = ShaderTarget.Dxil then t, flags else t, f)
        Blobs =
            set.Blobs
            |> List.map (fun b ->
                if b.Kernel = "scale_f32" && b.Target = ShaderTarget.Dxil then
                    { b with
                        Bytes = blob
                        Sha256 = Hash.sha256 blob
                    }
                else
                    b)
    }

/// The control shader set: the embedded set with scale_f32 built without "-denorm preserve".
let controlSet =
    lazy
        (let set = Lrm.Shaders.Shaders.embedded ()
         let flags = stripDenorm (ShaderSet.flagsFor ShaderTarget.Dxil set)

         compileScale flags
         |> Result.map (fun (text, blob) -> text, withScale set flags blob))
