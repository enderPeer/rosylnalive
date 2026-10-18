/// The forward-ops registry (ForwardOpsAndQwenModel contract): every backend `lrm forward` and `lrm generate` can run
/// the Qwen forward on, by name. This file registers cpu-f32 and cpu-f64; GpuBackends.fs (X1) adds cpu-device, vulkan
/// and d3d12 through `register`. It also holds the command-line plumbing both commands share. Owner: M2a.
///
///     lrm backends                         lists the registered backends
module Lrm.Cli.Commands.Backends

open System
open System.Collections.Generic
open System.IO
open System.Reflection
open System.Text
open Lrm.Core
open Lrm.Tensor
open Lrm.Telemetry
open Lrm.Models.Qwen

let private gate = obj ()
let private registry = List<OpsFactory>(CpuBackends.all)

/// Adds a backend. Raises ArgumentException for a name already registered.
let register (factory: OpsFactory) =
    lock gate (fun () ->
        if registry |> Seq.exists (fun f -> f.Name = factory.Name) then
            invalidArg "factory" (sprintf "backend '%s' is registered already" factory.Name)

        registry.Add factory)

/// Every registered backend, in registration order.
let all () =
    lock gate (fun () -> List.ofSeq registry)

/// A backend by name.
let tryFind (name: string) =
    all () |> List.tryFind (fun f -> f.Name = name)

/// Command-line plumbing shared by `lrm forward` and `lrm generate`.
module Cli =
    /// A Result computation expression, for argument checks.
    type ResultBuilder() =
        member _.Bind(r: Result<'a, 'e>, f: 'a -> Result<'b, 'e>) = Result.bind f r
        member _.Return(x: 'a) : Result<'a, 'e> = Ok x
        member _.ReturnFrom(r: Result<'a, 'e>) = r

    let result = ResultBuilder()

    /// Parsed `--key value` options and `--switch` flags.
    type Parsed =
        {
            Values: Map<string, string>
            Switches: Set<string>
        }

        member p.Get(key: string) = p.Values.TryFind key
        member p.Has(key: string) = p.Switches.Contains key

    /// Parses argv against the known value options and switches (names without the leading --).
    let parse (values: string list) (switches: string list) (argv: string[]) : Result<Parsed, string> =
        let rec go (acc: Parsed) (args: string list) =
            match args with
            | [] -> Ok acc
            | (flag: string) :: rest when flag.StartsWith("--", StringComparison.Ordinal) ->
                let key = flag.Substring 2

                if List.contains key switches then
                    go
                        { acc with
                            Switches = acc.Switches.Add key
                        }
                        rest
                elif List.contains key values then
                    match rest with
                    | v :: rest' ->
                        go
                            { acc with
                                Values = acc.Values.Add(key, v)
                            }
                            rest'
                    | [] -> Error(sprintf "option --%s needs a value" key)
                else
                    Error(sprintf "unknown option '%s'" flag)
            | extra :: _ -> Error(sprintf "unexpected argument '%s'" extra)

        go
            {
                Values = Map.empty
                Switches = Set.empty
            }
            (List.ofArray argv)

    /// The repo root above the current folder (the one that holds Lrm.slnx), if any.
    let private repoRoot () =
        let rec up (d: DirectoryInfo) =
            if isNull d then
                None
            elif File.Exists(Path.Combine(d.FullName, "Lrm.slnx")) then
                Some d.FullName
            else
                up d.Parent

        up (DirectoryInfo Environment.CurrentDirectory)

    /// A path for run parameters and messages: repo-relative with forward slashes inside the repo around the
    /// current folder (so a committed report names no clone), else the full path.
    let displayPath (path: string) =
        let full = Path.GetFullPath path

        match repoRoot () with
        | Some r ->
            let rel = Path.GetRelativePath(r, full)

            if rel.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted rel then
                full
            else
                rel.Replace(Path.DirectorySeparatorChar, '/')
        | None -> full

    /// A model folder: `qwen3-0.6b` (the default) is LRM_DATA/models/Qwen3-0.6B, `tiny` is goldens/tiny/g1/model of the
    /// repo around the current folder, anything else a path.
    let modelDir (arg: string option) =
        match arg with
        | None
        | Some "qwen3-0.6b" -> Ok(LrmPaths.model "Qwen3-0.6B")
        | Some "tiny" ->
            match repoRoot () with
            | Some r -> Ok(Path.Combine(r, "goldens", "tiny", "g1", "model"))
            | None -> Error "--model tiny needs the repo around the current folder; pass the model folder instead"
        | Some p -> Ok(Path.GetFullPath p)

    /// The backends named by --backend (comma-separated; cpu-f32 by default).
    let backends (arg: string option) =
        let names =
            (defaultArg arg "cpu-f32").Split(',')
            |> Array.map (fun s -> s.Trim())
            |> Array.filter (fun s -> s.Length > 0)
            |> List.ofArray

        let found = names |> List.map (fun n -> n, tryFind n)

        match found |> List.tryFind (fun (_, f) -> f.IsNone) with
        | Some(n, _) ->
            Error(sprintf "no backend '%s' (registered: %s)" n (all () |> List.map _.Name |> String.concat ", "))
        | None when names.IsEmpty -> Error "--backend names no backend"
        | None -> Ok(found |> List.map (snd >> Option.get))

    /// An integer option, with a default.
    let intOption (p: Parsed) (key: string) (fallback: int) =
        match p.Get key with
        | None -> Ok fallback
        | Some t ->
            match Invariant.tryParseI32 t with
            | Some v -> Ok v
            | None -> Error(sprintf "--%s '%s' is not an integer" key t)

    /// True for a model folder whose ids are bytes: a 256-token vocabulary and no tokenizer.json (the tiny model).
    let byteLevel (w: Qwen3Weights) =
        w.Config.Vocab = 256 && not (File.Exists(Path.Combine(w.Dir, "tokenizer.json")))

    /// An integer tensor of a safetensors file, given as FILE[:TENSOR] (TENSOR defaults to `fallback`).
    let idsFrom (spec: string) (fallback: string) : Result<int[], string> =
        let path, tensor =
            match spec.LastIndexOf ':' with
            | i when i > 1 -> spec.Substring(0, i), spec.Substring(i + 1)
            | _ -> spec, fallback

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

    /// The prompt ids from --ids (comma-separated), --ids-from FILE[:TENSOR] (an integer tensor, `input_ids` by
    /// default) or --prompt TEXT (UTF-8 bytes for a byte-level model). Returns the ids and how they were given.
    let promptIds (p: Parsed) (w: Qwen3Weights) : Result<int[] * string, string> =
        match [ "ids"; "ids-from"; "prompt" ] |> List.filter (fun k -> (p.Get k).IsSome) with
        | [] -> Error "give the prompt with --ids, --ids-from or --prompt"
        | _ :: _ :: _ -> Error "give the prompt only once (--ids, --ids-from or --prompt)"
        | [ "ids" ] ->
            let text = (p.Get "ids").Value

            let parsed =
                text.Split(',')
                |> Array.map (fun s -> s.Trim())
                |> Array.filter (fun s -> s.Length > 0)
                |> Array.map Invariant.tryParseI32

            if parsed.Length = 0 || parsed |> Array.exists Option.isNone then
                Error(sprintf "--ids '%s' is not a comma-separated list of integers" text)
            else
                Ok(parsed |> Array.map Option.get, "--ids")
        | [ "ids-from" ] ->
            let spec = (p.Get "ids-from").Value
            idsFrom spec "input_ids" |> Result.map (fun ids -> ids, spec)
        | _ ->
            if byteLevel w then
                Ok(Encoding.UTF8.GetBytes (p.Get "prompt").Value |> Array.map int, "--prompt (UTF-8 bytes)")
            else
                Error(
                    "--prompt needs the Qwen tokenizer (M1, TokenizerApi), which is not in this build yet; "
                    + "pass the ids with --ids or --ids-from"
                )

    /// The text of generated ids for a byte-level model (invalid UTF-8 becomes U+FFFD), else None.
    let text (w: Qwen3Weights) (ids: int[]) =
        if byteLevel w then
            Some(Encoding.UTF8.GetString(ids |> Array.map byte))
        else
            None

    /// The commit this build came from (after '+' in the informational version), if known.
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

    /// The ToleranceSpec pair of a backend measured against an oracle (torch32 or torch64p), if there is one.
    let pairFor (backend: string) (oracle: string) : TolerancePair option =
        match backend, oracle with
        | "cpu-f64", "torch64p" -> Some TolerancePair.Cpu64VsTorch64p
        | "cpu-f32", "torch32" -> Some TolerancePair.Cpu32VsTorch32
        | "cpu-f32", "torch64p" -> Some TolerancePair.Cpu32VsTorch64p
        | ("cpu-device" | "vulkan" | "d3d12" | "d3d12-warp"), "torch32" -> Some TolerancePair.GpuVsTorch32
        | ("cpu-device" | "vulkan" | "d3d12" | "d3d12-warp"), "torch64p" -> Some TolerancePair.GpuVsTorch64p
        | _ -> None

    /// Text with control characters escaped, for printing generated bytes.
    let printable (s: string) =
        let sb = StringBuilder()

        for c in s do
            if c < ' ' || c = '\u007f' then
                sb.Append(@"\x").Append((int c).ToString("x2", Globalization.CultureInfo.InvariantCulture))
                |> ignore
            else
                sb.Append c |> ignore

        sb.ToString()

    /// One line naming a class.
    let describe (c: ExecutionClass) =
        ExecutionClass.describe c + "; class " + (ExecutionClass.id c).Value.Short

    let private intAttr (key: string) (r: TelemetryRecord) =
        r.Attrs
        |> List.tryPick (fun a ->
            match a.Value with
            | AttrValue.Int i when a.Key = key -> Some i
            | _ -> None)

    /// Keeps the model.forward spans of one node, for the tokens/s a command prints.
    type private Collector() =
        let spans = List<TelemetryRecord>()

        /// (calls, tokens, host ns) of the model.forward spans of a phase.
        member _.Rate(phase: string) =
            lock spans (fun () ->
                let xs =
                    spans
                    |> Seq.filter (fun r ->
                        r.Attrs
                        |> List.exists (fun a -> a.Key = "phase" && a.Value = AttrValue.Str phase))
                    |> List.ofSeq

                xs.Length,
                xs |> List.sumBy (fun r -> defaultArg (intAttr "tokens" r) 1L),
                xs |> List.sumBy _.DurationNs)

        interface ITelemetrySink with
            member _.Write r =
                if r.Kind = TelemetryKind.Span && r.Name = Forward.ForwardSpan then
                    lock spans (fun () -> spans.Add r)

    type private TeeSink(a: ITelemetrySink, b: ITelemetrySink) =
        interface ITelemetrySink with
            member _.Write r =
                a.Write r
                b.Write r

    /// The telemetry of one command: with --run-log, one run log (RunLog contract) per backend, so each run has one
    /// ExecutionClass and the inference report's headline rates are that class's alone; and a collector per backend
    /// for the rates the command prints. `titleFor` and `parametersFor` give a backend's run title and parameters.
    /// Dispose finishes, as aborted, any run the command did not finish. Telemetry never changes a value.
    type RunHost
        (kind: string, titleFor: string -> string, runLog: bool, parametersFor: string -> (string * string) list) =
        let collectors = Dictionary<int, Collector>()
        let runs = Dictionary<int, RunContext>()

        /// The run folders started so far, in backend order.
        member _.Directories =
            runs |> Seq.sortBy _.Key |> Seq.map _.Value.Directory |> List.ofSeq

        /// Makes backend `i`'s facade ambient until disposed; with --run-log it starts that backend's run.
        member _.Install(i: int, backend: string, xclass: ExecutionClass) : IDisposable =
            let collector = Collector()
            collectors[i] <- collector

            if not runLog then
                Telemetry.install (Telemetry.ofSink collector)
            else
                let run =
                    RunContext.Create(
                        { RunOptions.create kind (titleFor backend) with
                            ExecutionClass = Some xclass
                            GitSha = gitSha ()
                            Params = parametersFor backend
                        }
                    )

                runs[i] <- run
                Telemetry.install (Telemetry.ofSink (TeeSink(run.Sink, collector)))

        /// Prints backend i's prefill and decode rates, from its model.forward spans (as the inference report does).
        member _.PrintRates(i: int) =
            let line (phase: string) =
                let calls, tokens, ns = collectors[i].Rate phase

                if calls = 0 || ns <= 0L then
                    None
                else
                    let rate = float tokens / (float ns / 1e9)

                    Some(
                        Invariant.format
                            $"{phase} {tokens} token(s) in {calls} call(s), {float ns / 1e6:F3} ms, {rate:F1} tok/s"
                    )

            match [ line "prefill"; line "decode" ] |> List.choose id with
            | [] -> ()
            | xs -> printfn "  speed: %s" (String.concat "; " xs)

        /// Finishes backend i's run, if it has one, and prints where it is.
        member _.Finish(i: int, status: RunStatus) =
            match runs.TryGetValue i with
            | true, r ->
                r.Finish status
                printfn "  run log: %s (render it with: lrm report %s)" r.Directory (Path.GetFileName r.Directory)
            | _ -> ()

        interface IDisposable with
            member _.Dispose() =
                for r in runs.Values do
                    (r :> IDisposable).Dispose()

let private list (_: string[]) =
    for f in all () do
        printfn "%-12s %s" f.Name f.Summary

    0

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "backends"
            Summary = "List the backends the Qwen forward runs on (lrm forward, lrm generate)."
            Run = list
        |}
    ]
