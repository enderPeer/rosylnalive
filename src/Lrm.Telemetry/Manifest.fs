namespace Lrm.Telemetry

open System
open System.IO
open System.Text.Json
open Lrm.Core

/// How a run ended, as recorded in its manifest.
[<RequireQualifiedAccess>]
type RunStatus =
    /// Started and not finished yet (or the process died before it could finish).
    | Running
    /// Finished and everything it checked passed.
    | Ok
    /// Finished, and something it checked failed.
    | Failed
    /// Disposed without a Finish call, for example after an exception.
    | Aborted

[<RequireQualifiedAccess>]
module RunStatus =
    let all = [ RunStatus.Running; RunStatus.Ok; RunStatus.Failed; RunStatus.Aborted ]

    let name =
        function
        | RunStatus.Running -> "running"
        | RunStatus.Ok -> "ok"
        | RunStatus.Failed -> "failed"
        | RunStatus.Aborted -> "aborted"

    let tryParse (s: string) =
        all |> List.tryFind (fun x -> Invariant.equals (name x) s)

/// LRM_DATA/runs/<runId>/manifest.json, schema lrm.run/1 (RunLog contract). It is written by the node that creates the
/// run, when the run starts and again when it finishes. Nodes that join a run declare themselves in their own event
/// files instead (the run.node header record).
type RunManifest =
    {
        /// Always "lrm.run/1".
        Schema: string
        RunId: string
        /// What the run is for: a report template name (parity, inference, kernels, cluster, gate1, training, routing,
        /// thinking) or any other safe name. `lrm report` picks the template from it.
        Kind: string
        Title: string
        Status: RunStatus
        /// Start time, ISO 8601 UTC. A label only: it never enters a hash.
        CreatedUtc: string
        /// End time, once finished.
        EndedUtc: string option
        /// The node that created the run and wrote this manifest.
        Node: string
        /// True when the run was made by a synthetic generator, not by real work. Reports say so in a banner.
        Synthetic: bool
        /// The commit the run was built from, when the host knows it.
        GitSha: string option
        /// The execution class of the creating node.
        ExecutionClass: ExecutionClass option
        /// The driver that executed the work (the class's Driver), repeated for gates that pin the driver.
        Driver: string option
        /// Free-form run parameters (model, prompt set, seed, ...), sorted by key.
        Params: (string * string) list
    }

[<RequireQualifiedAccess>]
module RunManifest =
    /// The schema name.
    [<Literal>]
    let Schema = "lrm.run/1"

    /// The file name inside the run folder.
    [<Literal>]
    let FileName = "manifest.json"

    /// The JSON form of an execution class, including its id.
    let classJson (c: ExecutionClass) =
        JsonValue.Obj
            [
                "backend", JsonValue.Str(ExecutionBackend.name c.Backend)
                "deviceId", JsonValue.Str c.DeviceId
                "driver", JsonValue.Str c.Driver
                "shaderSetHash",
                (match c.ShaderSetHash with
                 | Some h -> JsonValue.Str h.Hex
                 | None -> JsonValue.Null)
                "arch", JsonValue.Str c.Arch
                "id", JsonValue.Str (ExecutionClass.id c).Value.Hex
            ]

    let private optStr =
        function
        | Some s -> JsonValue.Str s
        | None -> JsonValue.Null

    /// The manifest as a JSON value, keys in contract order.
    let toJson (m: RunManifest) =
        JsonValue.Obj
            [
                "schema", JsonValue.Str m.Schema
                "runId", JsonValue.Str m.RunId
                "kind", JsonValue.Str m.Kind
                "title", JsonValue.Str m.Title
                "status", JsonValue.Str(RunStatus.name m.Status)
                "createdUtc", JsonValue.Str m.CreatedUtc
                "endedUtc", optStr m.EndedUtc
                "node", JsonValue.Str m.Node
                "synthetic", JsonValue.Bool m.Synthetic
                "gitSha", optStr m.GitSha
                "executionClass",
                (match m.ExecutionClass with
                 | Some c -> classJson c
                 | None -> JsonValue.Null)
                "driver", optStr m.Driver
                "params", JsonValue.Obj(m.Params |> List.sortBy fst |> List.map (fun (k, v) -> k, JsonValue.Str v))
            ]

    /// The exact file text: indented JSON with LF line ends and a final LF.
    let render (m: RunManifest) = Json.pretty (toJson m)

    exception private ManifestError of string

    let private fail fmt =
        Printf.ksprintf (fun s -> raise (ManifestError s)) fmt

    let private prop (o: JsonElement) (k: string) =
        match o.TryGetProperty k with
        | true, v -> v
        | _ -> fail "missing '%s'" k

    let private str (o: JsonElement) (k: string) =
        let v = prop o k

        if v.ValueKind <> JsonValueKind.String then
            fail "'%s' is not a string" k

        v.GetString()

    let private optString (o: JsonElement) (k: string) =
        let v = prop o k

        match v.ValueKind with
        | JsonValueKind.Null -> None
        | JsonValueKind.String -> Some(v.GetString())
        | _ -> fail "'%s' is not a string or null" k

    /// Reads an execution class written by classJson, and checks that its recorded id is the id of its fields.
    let classOfJson (o: JsonElement) : ExecutionClass =
        let backend =
            match ExecutionBackend.tryParse (str o "backend") with
            | Some b -> b
            | None -> fail "unknown backend '%s'" (str o "backend")

        let c =
            {
                Backend = backend
                DeviceId = str o "deviceId"
                Driver = str o "driver"
                ShaderSetHash =
                    optString o "shaderSetHash"
                    |> Option.map (fun h ->
                        match ContentId.tryParse h with
                        | Some id -> id
                        | None -> fail "bad shaderSetHash '%s'" h)
                Arch = str o "arch"
            }

        match ExecutionClass.validate c with
        | Error e -> fail "invalid executionClass: %s" e
        | Result.Ok c ->
            let id = (ExecutionClass.id c).Value.Hex

            match optString o "id" with
            | Some recorded when not (Invariant.equals recorded id) ->
                fail "executionClass id %s does not match its fields (%s)" recorded id
            | _ -> c

    /// Parses manifest text. Raises nothing: errors come back as Error.
    let parse (text: string) : Result<RunManifest, string> =
        try
            use doc = JsonDocument.Parse(text)
            let root = doc.RootElement

            if root.ValueKind <> JsonValueKind.Object then
                fail "the manifest is not a JSON object"

            let schema = str root "schema"

            if not (Invariant.equals schema Schema) then
                fail "unsupported schema '%s' (expected %s)" schema Schema

            let status =
                match RunStatus.tryParse (str root "status") with
                | Some s -> s
                | None -> fail "unknown status '%s'" (str root "status")

            let synthetic =
                match (prop root "synthetic").ValueKind with
                | JsonValueKind.True -> true
                | JsonValueKind.False -> false
                | _ -> fail "'synthetic' is not a boolean"

            let xclass =
                let v = prop root "executionClass"

                match v.ValueKind with
                | JsonValueKind.Null -> None
                | JsonValueKind.Object -> Some(classOfJson v)
                | _ -> fail "'executionClass' is not an object or null"

            let pars =
                let v = prop root "params"

                if v.ValueKind <> JsonValueKind.Object then
                    fail "'params' is not an object"

                v.EnumerateObject()
                |> Seq.map (fun p ->
                    if p.Value.ValueKind <> JsonValueKind.String then
                        fail "param '%s' is not a string" p.Name

                    p.Name, p.Value.GetString())
                |> List.ofSeq
                |> List.sortBy fst

            Result.Ok
                {
                    Schema = schema
                    RunId = str root "runId"
                    Kind = str root "kind"
                    Title = str root "title"
                    Status = status
                    CreatedUtc = str root "createdUtc"
                    EndedUtc = optString root "endedUtc"
                    Node = str root "node"
                    Synthetic = synthetic
                    GitSha = optString root "gitSha"
                    ExecutionClass = xclass
                    Driver = optString root "driver"
                    Params = pars
                }
        with
        | ManifestError msg -> Error msg
        | :? JsonException as e -> Error("malformed JSON: " + e.Message)

    /// Reads <runDir>/manifest.json.
    let read (runDir: string) : Result<RunManifest, string> =
        let path = Path.Combine(runDir, FileName)

        if not (File.Exists path) then
            Error(sprintf "no %s in %s" FileName runDir)
        else
            parse (File.ReadAllText path)
            |> Result.mapError (fun e -> sprintf "%s: %s" path e)

    /// Writes <runDir>/manifest.json atomically (a temporary file, then a rename).
    let write (runDir: string) (m: RunManifest) =
        let path = Path.Combine(runDir, FileName)
        let tmp = path + ".tmp"
        File.WriteAllText(tmp, render m, Text.UTF8Encoding(false))
        File.Move(tmp, path, true)

    /// A param value.
    let tryParam (key: string) (m: RunManifest) =
        m.Params
        |> List.tryFind (fun (k, _) -> Invariant.equals k key)
        |> Option.map snd
