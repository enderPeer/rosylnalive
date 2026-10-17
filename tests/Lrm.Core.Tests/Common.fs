/// Helpers shared by the Lrm.Core tests.
module Lrm.Core.Tests.Common

open System
open System.Diagnostics
open System.Globalization
open System.IO
open System.Reflection
open System.Text.Json.Nodes

/// Lowercase hex.
let hex (bytes: byte[]) = Convert.ToHexStringLower bytes

/// Bytes from hex that may contain spaces for readability.
let unhex (text: string) =
    Convert.FromHexString(text.Replace(" ", ""))

/// Runs `f` with the current culture and UI culture set to `name`, then restores both.
let withCulture (name: string) (f: unit -> 'a) : 'a =
    let culture = CultureInfo.CurrentCulture
    let ui = CultureInfo.CurrentUICulture

    try
        CultureInfo.CurrentCulture <- CultureInfo.GetCultureInfo name
        CultureInfo.CurrentUICulture <- CultureInfo.GetCultureInfo name
        f ()
    finally
        CultureInfo.CurrentCulture <- culture
        CultureInfo.CurrentUICulture <- ui

/// How a child run of this test exe gets its runtime configuration.
type RuntimeConfig =
    /// The exe's own runtimeconfig.json (System.Globalization.Invariant=false).
    | Own
    /// A copy with System.Globalization.Invariant set to this value.
    | InvariantSwitch of bool
    /// A copy without the System.Globalization.Invariant property, so the environment variable decides.
    | NoInvariantSwitch

/// The result of a child run.
type ChildResult =
    {
        ExitCode: int
        StdOut: string
        StdErr: string
    }

let private selfDll = Assembly.GetEntryAssembly().Location

let private writeConfig (dir: string) (config: RuntimeConfig) =
    let own = Path.ChangeExtension(selfDll, ".runtimeconfig.json")

    match config with
    | Own -> None
    | InvariantSwitch _
    | NoInvariantSwitch ->
        let root = JsonNode.Parse(File.ReadAllText own)
        let options = root["runtimeOptions"]
        let props = options["configProperties"].AsObject()
        props.Remove "System.Globalization.Invariant" |> ignore

        match config with
        | InvariantSwitch value -> props["System.Globalization.Invariant"] <- JsonValue.Create value
        | _ -> ()

        let path = Path.Combine(dir, "child.runtimeconfig.json")
        File.WriteAllText(path, root.ToJsonString())
        Some path

/// Runs this test exe again as a worker (LRM_TEST_WORKER=1) with `args`, extra environment variables and a runtime
/// configuration, and returns its exit code and output. DOTNET_SYSTEM_GLOBALIZATION_INVARIANT is removed from the
/// child's environment unless `env` sets it.
let runSelf (args: string list) (env: (string * string) list) (config: RuntimeConfig) : ChildResult =
    let dir =
        Path.Combine(Path.GetTempPath(), "lrm-core-tests", Guid.NewGuid().ToString("N"))

    Directory.CreateDirectory dir |> ignore

    try
        let host =
            match Environment.ProcessPath with
            | null -> "dotnet"
            | p when Path.GetFileNameWithoutExtension(p).Equals("dotnet", StringComparison.OrdinalIgnoreCase) -> p
            | _ -> "dotnet"

        let psi = ProcessStartInfo(host)
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false

        match writeConfig dir config with
        | Some path ->
            psi.ArgumentList.Add "exec"
            psi.ArgumentList.Add "--runtimeconfig"
            psi.ArgumentList.Add path
        | None -> ()

        psi.ArgumentList.Add selfDll

        for a in args do
            psi.ArgumentList.Add a

        psi.Environment.Remove "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT" |> ignore

        for k, v in Lrm.Testing.Env.workerEnvironment @ env do
            psi.Environment[k] <- v

        use p = Process.Start psi
        let out = p.StandardOutput.ReadToEndAsync()
        let err = p.StandardError.ReadToEndAsync()

        if not (p.WaitForExit 60_000) then
            p.Kill true
            failwithf "child run %A did not finish within 60 s" args

        p.WaitForExit()

        {
            ExitCode = p.ExitCode
            StdOut = out.Result
            StdErr = err.Result
        }
    finally
        try
            Directory.Delete(dir, true)
        with _ ->
            ()
