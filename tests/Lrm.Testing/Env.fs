/// Environment shared by every test exe: data root, repo root and worker mode (TestTiers, DataRootAndToolchain).
module Lrm.Testing.Env

open System
open System.IO

/// Default LRM_DATA (DataRootAndToolchain contract).
[<Literal>]
let DefaultDataRoot = "C:/Users/end/dev/lrm-data"

let private var (name: string) =
    match Environment.GetEnvironmentVariable name with
    | null -> None
    | s when String.IsNullOrWhiteSpace s -> None
    | s -> Some(s.Trim())

/// An environment variable, or None when unset or blank.
let tryGet (name: string) = var name

/// LRM_DATA, else the default data root.
let dataRoot () =
    defaultArg (var "LRM_DATA") DefaultDataRoot

/// Absolute path with forward slashes and no trailing slash, for comparisons.
let normalizePath (p: string) =
    Path.GetFullPath(p).Replace('\\', '/').TrimEnd('/')

/// True when both paths name the same directory (case-insensitive, as on Windows).
let samePath (a: string) (b: string) =
    String.Equals(normalizePath a, normalizePath b, StringComparison.OrdinalIgnoreCase)

/// LRM_REPO_ROOT (set by tools/test.ps1), else the nearest ancestor of the exe that holds Lrm.slnx.
let repoRoot () =
    match var "LRM_REPO_ROOT" with
    | Some r -> normalizePath r
    | None ->
        let rec up (d: DirectoryInfo) =
            if isNull d then
                None
            elif File.Exists(Path.Combine(d.FullName, "Lrm.slnx")) then
                Some(normalizePath d.FullName)
            else
                up d.Parent

        defaultArg (up (DirectoryInfo AppContext.BaseDirectory)) (normalizePath Environment.CurrentDirectory)

/// Worker processes (children started by a test) never acquire a test mutex.
let isWorker () = var "LRM_TEST_WORKER" = Some "1"

/// Environment entries a test sets on a child process so that it runs as a worker.
let workerEnvironment = [ "LRM_TEST_WORKER", "1" ]
