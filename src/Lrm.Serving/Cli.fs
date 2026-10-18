namespace Lrm.Serving

open System
open System.IO
open System.Reflection
open System.Text
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen

/// Parsed `--key value` options (keys may repeat; order is kept) and `--switch` flags, for `lrm coord`, `lrm node`
/// and the harness's worker processes.
type CliArgs =
    {
        Values: (string * string) list
        Switches: Set<string>
    }

    /// The last value of a key.
    member a.Get(key: string) =
        a.Values |> List.tryFindBack (fun (k, _) -> k = key) |> Option.map snd

    /// Every value of a key, in order.
    member a.All(key: string) =
        a.Values |> List.filter (fun (k, _) -> k = key) |> List.map snd

    member a.Has(key: string) = a.Switches.Contains key

[<RequireQualifiedAccess>]
module CliArgs =
    /// Parses args against the known value options and switches (names without the leading --).
    let parse (values: string list) (switches: string list) (args: string list) : Result<CliArgs, string> =
        let rec go (acc: CliArgs) (rest: string list) =
            match rest with
            | [] ->
                Ok
                    { acc with
                        Values = List.rev acc.Values
                    }
            | (flag: string) :: tail when flag.StartsWith("--", StringComparison.Ordinal) && flag.Length > 2 ->
                let key = flag.Substring 2

                if List.contains key switches then
                    go
                        { acc with
                            Switches = acc.Switches.Add key
                        }
                        tail
                elif List.contains key values then
                    match tail with
                    | v :: tail' ->
                        go
                            { acc with
                                Values = (key, v) :: acc.Values
                            }
                            tail'
                    | [] -> Error(sprintf "option --%s needs a value" key)
                else
                    Error(sprintf "unknown option '%s'" flag)
            | extra :: _ -> Error(sprintf "unexpected argument '%s'" extra)

        go { Values = []; Switches = Set.empty } args

    /// An integer option with a default.
    let int (a: CliArgs) (key: string) (fallback: int) : Result<int, string> =
        match a.Get key with
        | None -> Ok fallback
        | Some t ->
            match Invariant.tryParseI32 t with
            | Some v -> Ok v
            | None -> Error(sprintf "--%s '%s' is not an integer" key t)

    /// An unsigned 64-bit option with a default.
    let u64 (a: CliArgs) (key: string) (fallback: uint64) : Result<uint64, string> =
        match a.Get key with
        | None -> Ok fallback
        | Some t ->
            match Invariant.tryParseU64 t with
            | Some v -> Ok v
            | None -> Error(sprintf "--%s '%s' is not an unsigned integer" key t)

    /// host:port.
    let hostPort (text: string) : Result<string * int, string> =
        match text.LastIndexOf ':' with
        | i when i > 0 ->
            match Invariant.tryParseI32 (text.Substring(i + 1)) with
            | Some p when p > 0 && p < 65536 -> Ok(text.Substring(0, i), p)
            | _ -> Error(sprintf "bad port in '%s'" text)
        | _ -> Error(sprintf "expected host:port, got '%s'" text)

/// A Result computation expression for argument checks.
type ResultBuilder() =
    member _.Bind(r: Result<'a, 'e>, f: 'a -> Result<'b, 'e>) = Result.bind f r
    member _.Return(x: 'a) : Result<'a, 'e> = Ok x
    member _.ReturnFrom(r: Result<'a, 'e>) = r

[<AutoOpen>]
module ResultBuilderInstance =
    /// result { ... } over Result<_, string>.
    let result = ResultBuilder()

/// Model folders and other paths of the serving commands.
[<RequireQualifiedAccess>]
module ServingPaths =
    /// The repo root above `start` (the folder that holds Lrm.slnx), if any.
    let rec private up (d: DirectoryInfo) =
        if isNull d then
            None
        elif File.Exists(Path.Combine(d.FullName, "Lrm.slnx")) then
            Some d.FullName
        else
            up d.Parent

    /// The repo root around the current folder, else around the running assembly.
    let repoRoot () =
        match up (DirectoryInfo Environment.CurrentDirectory) with
        | Some r -> Some r
        | None -> up (DirectoryInfo AppContext.BaseDirectory)

    /// The tiny golden folder of the repo, goldens/tiny/g1.
    let tinyGoldens () =
        repoRoot () |> Option.map (fun r -> Path.Combine(r, "goldens", "tiny", "g1"))

    /// A model folder: `tiny` is goldens/tiny/g1/model of the repo, `qwen3-0.6b` is LRM_DATA/models/Qwen3-0.6B, anything
    /// else a path.
    let modelDir (arg: string) : Result<string, string> =
        match arg with
        | "qwen3-0.6b" -> Ok(LrmPaths.model "Qwen3-0.6B")
        | "tiny" ->
            match tinyGoldens () with
            | Some g -> Ok(Path.Combine(g, "model"))
            | None -> Error "--model tiny needs the repo around the current folder; pass the model folder instead"
        | p -> Ok(Path.GetFullPath p)

    /// A path for run parameters and messages: repo-relative with forward slashes inside the repo around the current
    /// folder (so a committed report names no clone), else the full path.
    let display (path: string) =
        let full = Path.GetFullPath path

        match up (DirectoryInfo Environment.CurrentDirectory) with
        | Some r ->
            let rel = Path.GetRelativePath(r, full)

            if rel.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted rel then
                full
            else
                rel.Replace(Path.DirectorySeparatorChar, '/')
        | None -> full

    /// The commit this build came from (after '+' in the entry assembly's informational version), if known.
    let gitSha () =
        let a = Assembly.GetEntryAssembly()

        let v =
            if isNull a then
                null
            else
                a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()

        if isNull v then
            None
        else
            let s = v.InformationalVersion
            let i = s.IndexOf '+'

            if i >= 0 && i < s.Length - 1 then
                Some(s.Substring(i + 1))
            else
                None

/// One lane's prompt as the command line gives it.
[<RequireQualifiedAccess>]
type PromptSpec =
    /// --ids 1,2,3
    | Ids of int[]
    /// --ids-from FILE[:TENSOR] (an integer tensor, input_ids by default)
    | IdsFrom of string
    /// --prompt TEXT: UTF-8 bytes, for a byte-level model (a vocabulary of 256 and no tokenizer.json)
    | Text of string
    /// --prompt-set tiny: the tiny golden prompts p0, p1 and p2 (three lanes)
    | Set of string

/// What a serving job is made of on the command line (StagePipeline, "Commands"); `lrm coord`, `lrm coord local` and
/// the harness build the same ServingJob from it.
type JobSpec =
    {
        /// A model folder or alias (ServingPaths.modelDir).
        Model: string
        /// The serving backend (cpu-f32 by default).
        Backend: string
        /// The split; None for the midpoint.
        Boundaries: int list option
        Prompts: PromptSpec list
        Steps: int
        StopAtEos: bool
    }

[<RequireQualifiedAccess>]
module JobSpec =
    /// The value options JobSpec.parse reads.
    let values =
        [
            "model"
            "backend"
            "split"
            "ids"
            "ids-from"
            "prompt"
            "prompt-set"
            "steps"
        ]

    /// The switches JobSpec.parse reads.
    let switches = [ "stop-at-eos" ]

    /// The tiny model's standard serving job: p0, p1 and p2 for 16 steps each on cpu-f32, split at the midpoint.
    let tiny =
        {
            Model = "tiny"
            Backend = "cpu-f32"
            Boundaries = None
            Prompts = [ PromptSpec.Set "tiny" ]
            Steps = 16
            StopAtEos = false
        }

    let private promptOf (key: string) (value: string) : Result<PromptSpec, string> =
        match key with
        | "ids" ->
            let parsed =
                value.Split(',', StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun s -> Invariant.tryParseI32 (s.Trim()))

            if parsed.Length = 0 || parsed |> Array.exists Option.isNone then
                Error(sprintf "--ids '%s' is not a comma-separated list of integers" value)
            else
                Ok(PromptSpec.Ids(parsed |> Array.map Option.get))
        | "ids-from" -> Ok(PromptSpec.IdsFrom value)
        | "prompt" -> Ok(PromptSpec.Text value)
        | _ -> Ok(PromptSpec.Set value)

    /// The job spec of parsed options: --model (tiny by default), --backend, --split, the prompts in the order given
    /// (--ids, --ids-from, --prompt, --prompt-set, each repeatable; the tiny set by default), --steps (16) and
    /// --stop-at-eos.
    let parse (a: CliArgs) : Result<JobSpec, string> =
        result {
            let! steps = CliArgs.int a "steps" tiny.Steps

            let! boundaries =
                match a.Get "split" with
                | None -> Ok None
                | Some t -> StageLayout.parseBoundaries t |> Result.map Some

            let promptArgs =
                a.Values
                |> List.filter (fun (k, _) -> k = "ids" || k = "ids-from" || k = "prompt" || k = "prompt-set")

            let! prompts =
                promptArgs
                |> List.fold
                    (fun acc (k, v) ->
                        match acc, promptOf k v with
                        | Ok xs, Ok p -> Ok(p :: xs)
                        | Error e, _
                        | _, Error e -> Error e)
                    (Ok [])
                |> Result.map List.rev

            return
                {
                    Model = defaultArg (a.Get "model") tiny.Model
                    Backend = defaultArg (a.Get "backend") tiny.Backend
                    Boundaries = boundaries
                    Prompts = (if prompts.IsEmpty then tiny.Prompts else prompts)
                    Steps = steps
                    StopAtEos = a.Has "stop-at-eos"
                }
        }

    /// The arguments that give this spec to another process (absolute model path, explicit split).
    let toArgs (modelDir: string) (boundaries: int list) (spec: JobSpec) : string list =
        [
            yield! [ "--model"; modelDir; "--backend"; spec.Backend ]
            yield!
                [
                    "--split"
                    (if boundaries.IsEmpty then
                         "none"
                     else
                         boundaries |> List.map Invariant.i32 |> String.concat ",")
                ]
            for p in spec.Prompts do
                match p with
                | PromptSpec.Ids ids -> yield! [ "--ids"; ids |> Array.map Invariant.i32 |> String.concat "," ]
                | PromptSpec.IdsFrom f -> yield! [ "--ids-from"; Path.GetFullPath f ]
                | PromptSpec.Text t -> yield! [ "--prompt"; t ]
                | PromptSpec.Set s -> yield! [ "--prompt-set"; s ]
            yield! [ "--steps"; Invariant.i32 spec.Steps ]
            if spec.StopAtEos then
                yield "--stop-at-eos"
        ]

    /// The ids of an integer tensor of a safetensors file, FILE[:TENSOR].
    let idsFrom (spec: string) : Result<int[], string> =
        let path, tensor =
            match spec.LastIndexOf ':' with
            | i when i > 1 -> spec.Substring(0, i), spec.Substring(i + 1)
            | _ -> spec, "input_ids"

        try
            use f = SafetensorsFile.Open(Path.GetFullPath path)

            match f.TryEntry tensor with
            | None -> Error(sprintf "%s has no tensor '%s'" path tensor)
            | Some e ->
                match e.Desc.DType with
                | DType.I64 -> Ok((f.ReadI64 tensor).Data |> Array.map int)
                | DType.I32 -> Ok (f.ReadI32 tensor).Data
                | d -> Error(sprintf "%s:%s is %s, not integer ids" path tensor (DType.name d))
        with
        | :? IOException as e -> Error e.Message
        | :? FormatException as e -> Error e.Message

    /// The prompts of the tiny golden set: p0 (13 ids), p1 (1 id) and p2 (70 ids).
    let tinyPrompts () : Result<int[] list, string> =
        match ServingPaths.tinyGoldens () with
        | None -> Error "--prompt-set tiny needs the repo around the current folder"
        | Some g ->
            [ "p0"; "p1"; "p2" ]
            |> List.fold
                (fun acc p ->
                    match acc, idsFrom (Path.Combine(g, Invariant.format $"fwd_f32.{p}.safetensors")) with
                    | Ok xs, Ok ids -> Ok(ids :: xs)
                    | Error e, _
                    | _, Error e -> Error e)
                (Ok [])
            |> Result.map List.rev

    /// The prompts of a spec, one int array per lane.
    let prompts (weights: Qwen3Weights) (spec: JobSpec) : Result<int[] list, string> =
        let byteLevel =
            weights.Config.Vocab = 256
            && not (File.Exists(Path.Combine(weights.Dir, "tokenizer.json")))

        spec.Prompts
        |> List.fold
            (fun acc p ->
                match acc with
                | Error e -> Error e
                | Ok xs ->
                    let lanes =
                        match p with
                        | PromptSpec.Ids ids -> Ok [ ids ]
                        | PromptSpec.IdsFrom f -> idsFrom f |> Result.map List.singleton
                        | PromptSpec.Text t when byteLevel -> Ok [ Encoding.UTF8.GetBytes t |> Array.map int ]
                        | PromptSpec.Text _ ->
                            Error "--prompt needs a byte-level model; pass ids with --ids or --ids-from"
                        | PromptSpec.Set "tiny" -> tinyPrompts ()
                        | PromptSpec.Set other -> Error(sprintf "unknown --prompt-set '%s' (known: tiny)" other)

                    lanes |> Result.map (fun ls -> xs @ ls))
            (Ok [])

    /// The class of the spec's backend on this host (CPU backends only; a GPU class depends on its device).
    let executionClass (spec: JobSpec) : Result<ExecutionClass, string> =
        match QwenStageExecutor.classOf spec.Backend with
        | Some c -> Ok c
        | None ->
            Error(
                sprintf
                    "backend '%s' has no host-independent class; this build serves %s"
                    spec.Backend
                    (QwenStageExecutor.backends () |> List.map _.Name |> String.concat ", ")
            )

    /// The ServingJob of a spec over opened weights.
    let build (weights: Qwen3Weights) (spec: JobSpec) : Result<ServingJob, string> =
        result {
            let! cls = executionClass spec
            let! prompts = prompts weights spec

            let boundaries =
                defaultArg spec.Boundaries (StageLayout.midpoint weights.Config.Layers)

            let! _ = StageLayout.validate weights.Config.Layers boundaries

            let job: ServingJob =
                {
                    Model = ModelId.ofWeights weights
                    Layers = weights.Config.Layers
                    Hidden = weights.Config.Hidden
                    Vocab = weights.Config.Vocab
                    Boundaries = boundaries
                    Class = cls
                    Prompts = prompts
                    Steps = spec.Steps
                    StopAtEos = spec.StopAtEos
                    Eos = weights.Config.EosTokenIds
                }

            return! ServingJob.validate job
        }
