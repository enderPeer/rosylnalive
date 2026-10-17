/// Locations under the data root, LRM_DATA (DataRootAndToolchain and LrmCore contracts).
///
/// The default comes from the dataRoot in tools/bootstrap/toolchain.json, embedded at build time as the
/// Lrm.DefaultDataRoot assembly metadata of Lrm.Core (Directory.Build.props), the same source Lrm.Testing reads.
/// Nothing under the data root is ever committed. Paths never enter a content hash.
module Lrm.Core.LrmPaths

open System
open System.IO
open System.Reflection

type private Marker = class end

/// The data root built into Lrm.Core (toolchain.json's dataRoot).
let defaultDataRoot =
    typeof<Marker>.Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
    |> Seq.tryFind (fun a -> a.Key = "Lrm.DefaultDataRoot" && not (String.IsNullOrWhiteSpace a.Value))
    |> Option.map _.Value
    |> Option.defaultWith (fun () ->
        failwith "Lrm.Core was built without the Lrm.DefaultDataRoot metadata (see Directory.Build.props)")

/// The data root for a given LRM_DATA value: the value when it is set and not blank, else the default; always a full
/// path.
let resolveDataRoot (lrmData: string) =
    if String.IsNullOrWhiteSpace lrmData then
        Path.GetFullPath defaultDataRoot
    else
        Path.GetFullPath(lrmData.Trim())

/// LRM_DATA when it is set and not blank, else the default; always a full path.
let dataRoot () =
    resolveDataRoot (Environment.GetEnvironmentVariable "LRM_DATA")

/// A path under the data root `root`. `relative` must be relative and must not climb out with "..".
let underRoot (root: string) (relative: string) =
    if String.IsNullOrWhiteSpace relative || Path.IsPathRooted relative then
        invalidArg "relative" (sprintf "'%s' is not a relative path" relative)

    let root = Path.GetFullPath root
    let full = Path.GetFullPath(Path.Combine(root, relative))

    let rootWithSlash =
        Path.TrimEndingDirectorySeparator root + string Path.DirectorySeparatorChar

    if not (full.StartsWith(rootWithSlash, StringComparison.OrdinalIgnoreCase)) then
        invalidArg "relative" (sprintf "'%s' leaves the data root" relative)

    full

/// A path under the data root. `relative` must be relative and must not climb out with "..".
let under (relative: string) = underRoot (dataRoot ()) relative

/// Model weights: LRM_DATA/models.
let models () = under "models"

/// One model's folder, for example model "Qwen3-0.6B".
let model (name: string) = under (Path.Combine("models", name))

/// Pinned toolchains: LRM_DATA/toolchain.
let toolchain () = under "toolchain"

/// Downloaded archives: LRM_DATA/downloads.
let downloads () = under "downloads"

/// Large golden files: LRM_DATA/golden.
let golden () = under "golden"

/// Run logs: LRM_DATA/runs.
let runs () = under "runs"

/// One run's folder: LRM_DATA/runs/<runId>.
let run (id: RunId) = under (Path.Combine("runs", id.Value))

/// Test mutex tickets: LRM_DATA/locks.
let locks () = under "locks"

/// The golden-tool Python environment: LRM_DATA/venv-golden.
let venvGolden () = under "venv-golden"
