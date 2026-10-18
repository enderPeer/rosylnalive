namespace Lrm.Models.Qwen

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Text
open System.Text.Json
open Lrm.Core
open Lrm.Tensor

/// The two oracles of a full-model golden set (ReferenceOracles): torch32, the spec, and torch64p, the truth.
[<RequireQualifiedAccess>]
type GoldenOracle =
    | Torch32
    | Torch64p

[<RequireQualifiedAccess>]
module GoldenOracle =
    /// "torch32" or "torch64p": the oracle's name in greedy.safetensors (`<prompt>.ids.<oracle>`).
    let name (o: GoldenOracle) =
        match o with
        | GoldenOracle.Torch32 -> "torch32"
        | GoldenOracle.Torch64p -> "torch64p"

    /// "f32" or "f64p": the tag of the oracle's files (`fwd_<tag>.<prompt>`, `tf_<tag>.<prompt>`, `long_<tag>.<prompt>`).
    let tag (o: GoldenOracle) =
        match o with
        | GoldenOracle.Torch32 -> "f32"
        | GoldenOracle.Torch64p -> "f64p"

/// One comparison of a candidate with an oracle: the ToleranceSpec pair whose bounds apply, and the oracle whose files
/// it reads.
type ParityTarget =
    {
        Pair: TolerancePair
        Oracle: GoldenOracle
    }

[<RequireQualifiedAccess>]
module ParityTarget =
    let create (pair: TolerancePair) (oracle: GoldenOracle) = { Pair = pair; Oracle = oracle }

    /// The targets of a class (ToleranceSpec pairs): cpu-f64 against torch64p (cpu64-vs-torch64p); cpu-f32 against
    /// torch32 and torch64p (cpu32-vs-torch32, cpu32-vs-torch64p); every other class against both
    /// (gpu-vs-torch32, gpu-vs-torch64p).
    let forBackend (backend: ExecutionBackend) =
        match backend with
        | ExecutionBackend.CpuF64 -> [ create TolerancePair.Cpu64VsTorch64p GoldenOracle.Torch64p ]
        | ExecutionBackend.CpuF32 ->
            [
                create TolerancePair.Cpu32VsTorch32 GoldenOracle.Torch32
                create TolerancePair.Cpu32VsTorch64p GoldenOracle.Torch64p
            ]
        | _ ->
            [
                create TolerancePair.GpuVsTorch32 GoldenOracle.Torch32
                create TolerancePair.GpuVsTorch64p GoldenOracle.Torch64p
            ]

/// The role of a full-model golden prompt (GoldenSet, the manifest's `prompts`).
[<RequireQualifiedAccess>]
type GoldenPromptRole =
    /// A gate prompt: prefill logits (or every tap), teacher-forced argmax and greedy ids.
    | Gate
    /// A long prompt: final_norm.out on every row, logits at the stored rows and teacher-forced argmax.
    | Long

/// One prompt of a full-model golden set.
type GoldenPrompt =
    {
        Name: string
        Role: GoldenPromptRole
        /// For a gate prompt: its fwd_* files hold every ActivationNames tap (`dumps` = `taps`), not only the logits.
        Taps: bool
        Category: string
        /// The prompt ids (no BOS).
        Ids: int[]
        /// The greedy steps a gate prompt gates on: every step, or the prefix before the first low-margin step when the
        /// prefix fallback chose it (at least 16). 0 for a long prompt.
        GatedSteps: int
        /// True when the gate uses the prefix fallback (teacher forcing covers the steps after the prefix).
        Fallback: bool
    }

/// A full-model golden set opened for parity (GoldenSet, ToleranceSpec, ReferenceOracles): the verified set, its
/// tolerances, its prompts and G2's greedy rules. Raises GoldenFormatException for a set the parity cannot use.
[<Sealed>]
type QwenGoldens
    private
    (set: GoldenSet, spec: ToleranceSpec, prompts: GoldenPrompt list, steps: int, tfMinMargin: float, tau: float) =
    /// The set, whose files are verified when they are read.
    member _.Set = set

    /// The tolerances (tolerances.json, ToleranceSpec), read from the verified bytes.
    member _.Spec = spec

    /// Every prompt, in manifest order.
    member _.Prompts = prompts

    /// The gate prompts, in manifest order.
    member _.Gate = prompts |> List.filter (fun p -> p.Role = GoldenPromptRole.Gate)

    /// The long prompts, in manifest order.
    member _.Long = prompts |> List.filter (fun p -> p.Role = GoldenPromptRole.Long)

    /// The greedy steps of every gate prompt's references (`greedy.steps`, 64).
    member _.Steps = steps

    /// Teacher forcing checks the rows whose reference margin is at least this (`greedy.teacher_forced_min_margin`,
    /// 2 * tau_logit).
    member _.TeacherForcedMinMargin = tfMinMargin

    /// tau_logit (`greedy.tau_logit`).
    member _.TauLogit = tau

    /// The prompt of that name; raises KeyNotFoundException.
    member _.Prompt(name: string) =
        match prompts |> List.tryFind (fun p -> p.Name = name) with
        | Some p -> p
        | None -> raise (KeyNotFoundException(sprintf "%s/%s has no prompt '%s'" set.Set set.Version name))

    /// The golden greedy ids of a gate prompt for an oracle (`<prompt>.ids.<oracle>` in greedy.safetensors).
    member _.GreedyIds(prompt: string, oracle: GoldenOracle) =
        use g = set.OpenSafetensors "greedy.safetensors"

        (g.ReadI64(Invariant.format $"{prompt}.ids.{GoldenOracle.name oracle}")).Data
        |> Array.map int

    /// The golden greedy margins of a gate prompt for an oracle (`<prompt>.margin.<oracle>`, float64).
    member _.GreedyMargins(prompt: string, oracle: GoldenOracle) =
        use g = set.OpenSafetensors "greedy.safetensors"
        (g.ReadF64(Invariant.format $"{prompt}.margin.{GoldenOracle.name oracle}")).Data

    /// goldens/qwen3-0.6b/g1 under a repo root.
    static member DefaultDir(repoRoot: string) =
        Path.Combine(repoRoot, "goldens", "qwen3-0.6b", "g1")

    /// Opens the set (Goldens.Open: the manifest is validated, data files resolve under dataRoot/golden) and checks
    /// what the parity relies on: kind full-model; tolerances.json for this set with G2's greedy rules; every prompt's
    /// name, role, dumps and ids; the gate prompts and their gated steps equal in the manifest and the rules.
    static member Open(dir: string, ?dataRoot: string) : QwenGoldens =
        let set =
            match dataRoot with
            | Some root -> Goldens.Open(dir, root)
            | None -> Goldens.Open dir

        let fail (msg: string) : 'a =
            raise (GoldenFormatException(sprintf "%s/%s: %s" set.Set set.Version msg))

        if set.Kind <> "full-model" then
            fail (sprintf "kind '%s'; the full-model parity needs a full-model set" set.Kind)

        let spec =
            ToleranceSpec.parse (UTF8Encoding(false, true).GetString(set.ReadAllBytes "tolerances.json"))

        let setName = set.Set + "/" + set.Version

        if spec.Set <> setName then
            fail (sprintf "tolerances.json is for '%s'" spec.Set)

        let rules =
            match spec.Greedy with
            | Some g when g.ValueKind = JsonValueKind.Object -> g
            | _ -> fail "tolerances.json has no greedy rules"

        let prop (o: JsonElement) (key: string) (where: string) =
            match o.TryGetProperty key with
            | true, v -> v
            | _ -> fail (sprintf "%s has no '%s'" where key)

        let num (o: JsonElement) (key: string) (where: string) =
            let v = prop o key where

            match v.ValueKind with
            | JsonValueKind.Number -> v.GetDouble()
            | _ -> fail (sprintf "%s.%s is not a number" where key)

        let int' (o: JsonElement) (key: string) (where: string) =
            let v = prop o key where

            match v.TryGetInt32() with
            | true, n when v.ValueKind = JsonValueKind.Number -> n
            | _ -> fail (sprintf "%s.%s is not an integer" where key)

        let str (o: JsonElement) (key: string) (where: string) =
            let v = prop o key where

            if v.ValueKind = JsonValueKind.String then
                v.GetString()
            else
                fail (sprintf "%s.%s is not a string" where key)

        let bool' (o: JsonElement) (key: string) (where: string) =
            let v = prop o key where

            match v.ValueKind with
            | JsonValueKind.True -> true
            | JsonValueKind.False -> false
            | _ -> fail (sprintf "%s.%s is not a boolean" where key)

        let steps = int' rules "steps" "greedy"
        let tfMin = num rules "teacher_forced_min_margin" "greedy"
        let tau = num rules "tau_logit" "greedy"
        let minPrefix = int' rules "prefix_fallback_min_steps" "greedy"

        if
            steps < 1
            || not (Double.IsFinite tfMin)
            || tfMin < 0.0
            || not (Double.IsFinite tau)
            || tau <= 0.0
        then
            fail "greedy: steps, teacher_forced_min_margin or tau_logit out of range"

        let gateRules = prop rules "prompts" "greedy"

        let gateNames =
            match (prop rules "gate_prompts" "greedy").ValueKind with
            | JsonValueKind.Array ->
                (prop rules "gate_prompts" "greedy").EnumerateArray()
                |> Seq.map (fun e ->
                    if e.ValueKind = JsonValueKind.String then
                        e.GetString()
                    else
                        fail "greedy.gate_prompts holds a non-string")
                |> List.ofSeq
            | _ -> fail "greedy.gate_prompts is not an array"

        let manifestPrompts =
            match set.Manifest.TryGetProperty "prompts" with
            | true, p when p.ValueKind = JsonValueKind.Array -> p.EnumerateArray() |> List.ofSeq
            | _ -> fail "the manifest has no prompts array"

        let prompts =
            [
                for p in manifestPrompts do
                    let name = str p "name" "prompts[]"
                    let where = "prompt " + name
                    let role = str p "role" where
                    let dumps = str p "dumps" where

                    let ids =
                        match (prop p "ids" where).ValueKind with
                        | JsonValueKind.Array ->
                            (prop p "ids" where).EnumerateArray()
                            |> Seq.map (fun e ->
                                match e.TryGetInt32() with
                                | true, n when e.ValueKind = JsonValueKind.Number && n >= 0 -> n
                                | _ -> fail (sprintf "%s: an id that is not a non-negative integer" where))
                            |> Array.ofSeq
                        | _ -> fail (sprintf "%s: ids is not an array" where)

                    if ids.Length = 0 then
                        fail (sprintf "%s has no ids" where)

                    match role with
                    | "gate" ->
                        if dumps <> "taps" && dumps <> "logits" then
                            fail (sprintf "%s: gate dumps '%s' (taps or logits)" where dumps)

                        let gated = int' p "gated_steps" where
                        let fallback = bool' p "fallback" where

                        let r =
                            match gateRules.TryGetProperty name with
                            | true, r -> r
                            | _ -> fail (sprintf "greedy.prompts has no %s" name)

                        if int' r "gated_steps" ("greedy.prompts." + name) <> gated then
                            fail (sprintf "%s: gated_steps differ between the manifest and greedy.prompts" where)

                        if bool' r "fallback" ("greedy.prompts." + name) <> fallback then
                            fail (sprintf "%s: fallback differs between the manifest and greedy.prompts" where)

                        if
                            gated > steps
                            || gated < 1
                            || (fallback && gated < minPrefix)
                            || (not fallback && gated <> steps)
                        then
                            fail (sprintf "%s: %d gated steps (fallback %b, %d steps)" where gated fallback steps)

                        yield
                            {
                                Name = name
                                Role = GoldenPromptRole.Gate
                                Taps = (dumps = "taps")
                                Category = str p "category" where
                                Ids = ids
                                GatedSteps = gated
                                Fallback = fallback
                            }
                    | "long" ->
                        if dumps <> "long" then
                            fail (sprintf "%s: long dumps '%s'" where dumps)

                        yield
                            {
                                Name = name
                                Role = GoldenPromptRole.Long
                                Taps = false
                                Category = str p "category" where
                                Ids = ids
                                GatedSteps = 0
                                Fallback = false
                            }
                    | r -> fail (sprintf "%s: unknown role '%s'" where r)
            ]

        if prompts |> List.distinctBy _.Name |> List.length <> prompts.Length then
            fail "a prompt name appears twice"

        let gate =
            prompts
            |> List.filter (fun p -> p.Role = GoldenPromptRole.Gate)
            |> List.map _.Name

        if List.sort gate <> List.sort gateNames then
            fail "the manifest's gate prompts differ from greedy.gate_prompts"

        QwenGoldens(set, spec, prompts, steps, tfMin, tau)

/// The options of a parity run of one prompt (QwenParity).
type ParityOptions =
    {
        /// The registry name of the candidate: the `backend` attribute of every check.
        Backend: string
        /// The comparisons to make (ParityTarget.forBackend), at least one.
        Targets: ParityTarget list
        /// The most rows one forward runs. Longer sequences are prefilled in chunks into one cache; within one class
        /// the bits equal a single prefill (BatchInvariance; the full-model self-test checks it).
        Chunk: int
        /// Keep the prompt rows the comparisons used (PromptParity.PromptRows), for further checks.
        KeepPromptRows: bool
    }

[<RequireQualifiedAccess>]
module ParityOptions =
    /// The default chunk: 512 rows (a cpu-f64 forward of 512 rows holds about 4 GB of intermediates).
    [<Literal>]
    let DefaultChunk = 512

    /// The options of a class: its registry name, its targets and the default chunk.
    let forBackend (backend: ExecutionBackend) =
        {
            Backend = ExecutionBackend.name backend
            Targets = ParityTarget.forBackend backend
            Chunk = DefaultChunk
            KeepPromptRows = false
        }

/// The parity of one prompt on one backend.
type PromptParity =
    {
        Prompt: GoldenPrompt
        Backend: string
        /// Every golden.compare check, in the order they were made.
        Checks: CheckResult list
        /// The greedy run of a gate prompt (the references' steps, with margins); None for a long prompt.
        Greedy: GreedyResult option
        /// The rows of the teacher-forced prefill (a gate prompt: prompt + steps - 1) or of the long prompt.
        Rows: int
        /// The prompt rows the comparisons used, in forward order: every tap [T, features] for a taps prompt, else
        /// `logits` [T, vocab]; for a long prompt `final_norm.out` [T, hidden] and `logits` at the stored rows. Empty
        /// unless ParityOptions.KeepPromptRows.
        PromptRows: (string * IHostTensor) list
        /// Host seconds of the prefill (with its readback), of greedy decoding, and of the whole prompt.
        PrefillSeconds: float
        GreedySeconds: float
        TotalSeconds: float
    }

    member r.Passed = r.Checks |> List.forall _.Passed

    member r.Failures = r.Checks |> List.filter (fun c -> not c.Passed)

/// Rows of one [rows, cols] tensor in F32 or F64, filled block by block from readbacks.
type private RowSink(dtype: DType, rows: int) =
    let mutable cols = -1
    let mutable data: Array = null

    member _.Write(row: int, block: IHostTensor) =
        let br = int block.Desc.Shape.Rows
        let bc = int block.Desc.Shape.Cols

        if block.Desc.DType <> dtype then
            invalidArg "block" (sprintf "a %s block for a %s tensor" (DType.name block.Desc.DType) (DType.name dtype))

        if isNull data then
            cols <- bc

            data <-
                match dtype with
                | DType.F32 -> Array.zeroCreate<float32>(rows * cols) :> Array
                | DType.F64 -> Array.zeroCreate<float>(rows * cols) :> Array
                | d -> invalidArg "dtype" (sprintf "rows of %s" (DType.name d))
        elif bc <> cols then
            invalidArg "block" (sprintf "%d columns; the tensor has %d" bc cols)

        if row < 0 || row + br > rows then
            invalidArg "row" (sprintf "rows %d .. %d of %d" row (row + br - 1) rows)

        let size = DType.size dtype
        Buffer.BlockCopy(block.ToBytes(), 0, data, row * cols * size, br * cols * size)

    member _.Tensor: IHostTensor =
        if isNull data then
            invalidOp "no rows were written"

        match dtype with
        | DType.F32 -> HostTensor.f32 (Shape.matrix rows cols) (data :?> float32[]) :> IHostTensor
        | _ -> HostTensor.f64 (Shape.matrix rows cols) (data :?> float[]) :> IHostTensor

/// The full-model parity of a backend against a full-model golden set (ForwardOpsAndQwenModel, "Full-model parity"):
/// the gate1 checks of one prompt as golden.compare records, written once over IForwardOps, so CpuOps (gate1-cpu) and
/// a GPU executor (X1b, I1) run the same comparisons.
[<RequireQualifiedAccess>]
module QwenParity =
    /// `model.parity` span: one prompt on one backend.
    [<Literal>]
    let ParitySpan = "model.parity"

    /// The `component` attribute of the model.parity span.
    [<Literal>]
    let Component = "parity"

    /// The metric of the teacher-forced check: rows whose argmax differs from the reference's, among the rows whose
    /// reference margin is at least the threshold; it passes at 0.
    [<Literal>]
    let TeacherForcedMetric = "teacher_forced_mismatch"

    /// The `tap` of the teacher-forced check.
    [<Literal>]
    let TeacherForcedTap = "teacher_forced.argmax"

    /// The `tap` of the greedy_match check (GoldenCheck.greedy).
    [<Literal>]
    let GreedyTap = "greedy.ids"

    /// The golden.compare attributes of a comparison: the keys of ComparatorsAndGoldenReader, ReportTemplates
    /// (parity, gate1) and the forward's telemetry (pair, backend, prompt, tap, layer, kind = chain).
    let attrs (target: ParityTarget) (backend: string) (prompt: string) (tap: string) =
        [
            Attr.str "pair" (TolerancePair.name target.Pair)
            Attr.str TelemetryKeys.Backend backend
            Attr.str "prompt" prompt
            Attr.str "tap" tap
            Attr.int TelemetryKeys.Layer (ActivationNames.layer tap)
            Attr.str "kind" "chain"
        ]

    /// Prefills `ids` from position 0 into a fresh cache in forwards of at most `chunk` rows, each with every row's
    /// logits and the taps of `filter`, and calls `visit row0 logits taps` after each forward, while its result and
    /// taps are valid. The cache is released at the end.
    let prefill
        (ops: IForwardOps<'t>)
        (model: Qwen3Model)
        (ids: int[])
        (chunk: int)
        (filter: TapFilter)
        (visit: int -> 't -> (string * 't) list -> unit)
        =
        if chunk < 1 then
            invalidArg "chunk" (sprintf "chunks of %d rows" chunk)

        use cache = KvCache.create ops model (Stage.whole model.Config) ids.Length
        let mutable row0 = 0

        while row0 < ids.Length do
            let n = min chunk (ids.Length - row0)
            let store = TapStore<'t>(filter)

            let options =
                { ForwardOptions.all with
                    Taps = (if filter = TapFilter.Off then None else Some store)
                }

            let logits = Forward.run ops model cache row0 ids[row0 .. row0 + n - 1] options
            visit row0 logits store.Taps
            row0 <- row0 + n

    /// The teacher-forced check: our argmax against the reference's at every row whose reference margin is at least
    /// `minMargin`. Measured: the mismatching rows; bound 0; elements: the rows checked. `detail` names the first
    /// mismatching rows.
    let teacherForced
        (attrs: Attr list)
        (mine: uint32[])
        (reference: int64[])
        (margin: float[])
        (minMargin: float)
        : CheckResult =
        if mine.Length <> reference.Length || margin.Length <> reference.Length then
            invalidArg
                "mine"
                (sprintf "%d rows, %d reference rows, %d margins" mine.Length reference.Length margin.Length)

        let rows =
            [| 0 .. reference.Length - 1 |]
            |> Array.filter (fun r -> margin[r] >= minMargin)

        let wrong = rows |> Array.filter (fun r -> int64 mine[r] <> reference[r])

        let detail =
            if wrong.Length = 0 then
                []
            else
                let shown =
                    wrong
                    |> Array.truncate 8
                    |> Array.map (fun r ->
                        Invariant.format $"row {r}: {mine[r]} for {reference[r]} (margin {Invariant.f64 margin[r]})")

                [ Attr.str GoldenCheck.DetailKey (String.Join("; ", shown)) ]

        Compare.check
            TeacherForcedMetric
            BoundOp.AtMost
            (float wrong.Length)
            0.0
            (int64 rows.Length)
            (attrs
             @ [ Attr.float "min_margin" minMargin; Attr.int "rows" reference.Length ]
             @ detail)

    let private boundOf (g: QwenGoldens) (target: ParityTarget) (name: string) (metric: ToleranceMetric) =
        g.Spec.TryBound(target.Pair, ActivationNames.kind name, metric)
        |> Option.map _.Bound

    let private checkIds (what: string) (golden: int64[]) (ids: int[]) =
        if
            golden.Length <> ids.Length
            || Array.exists2 (fun (a: int64) b -> a <> int64 b) golden ids
        then
            raise (GoldenFormatException(sprintf "%s: its input_ids are not the ids it should hold" what))

    let private requireTargets (o: ParityOptions) =
        if o.Targets.IsEmpty then
            invalidArg "options" "no parity targets"

    /// One gate prompt on one backend, every check a golden.compare record through the ambient facade:
    /// 1. one teacher-forced prefill over the tf_* sequence (prompt + the reference's first steps - 1 ids), in chunks of
    ///    ParityOptions.Chunk rows; its prompt rows against fwd_* (relL2 of every tap for a taps prompt, else of the
    ///    logits; max |Δlogit| and KL of the logits, where the pair has a bound), and its argmax against each tf_*
    ///    file's at every row whose reference margin is at least the threshold;
    /// 2. Greedy.run on the canonical schedule for the references' steps (EOS ignored, margins), and greedy_match of
    ///    the gated steps against each oracle's ids; a goldens.greedy.min_margin gauge with oracle = the backend.
    /// The ops' plan must cover the prompt's rows, the chunk's rows and (prompt + steps - 1) positions, with the taps
    /// of a taps prompt.
    let gatePrompt (ops: IForwardOps<'t>) (model: Qwen3Model) (g: QwenGoldens) (p: GoldenPrompt) (o: ParityOptions) =
        if p.Role <> GoldenPromptRole.Gate then
            invalidArg "p" (sprintf "%s is not a gate prompt" p.Name)

        requireTargets o
        let total = Stopwatch.StartNew()
        let t = p.Ids.Length

        use _ =
            Telemetry.span
                ParitySpan
                [
                    Attr.str TelemetryKeys.Component Component
                    Attr.str "prompt" p.Name
                    Attr.str TelemetryKeys.Backend o.Backend
                    Attr.str "role" "gate"
                    Attr.int "tokens" t
                ]

        // The teacher-forced references, one per target: every tf_* file holds the same sequence.
        let tf =
            [
                for target in o.Targets ->
                    let name =
                        Invariant.format $"tf_{GoldenOracle.tag target.Oracle}.{p.Name}.safetensors"

                    use f = g.Set.OpenSafetensors name
                    target, (f.ReadI64 "input_ids").Data, (f.ReadI64 "argmax").Data, (f.ReadF64 "margin").Data
            ]

        let _, sequence64, _, _ = tf.Head
        let sequence = sequence64 |> Array.map int

        for target, ids, argmax, margin in tf do
            if ids <> sequence64 then
                raise (
                    GoldenFormatException(
                        sprintf
                            "tf_%s.%s holds another sequence than the other oracle"
                            (GoldenOracle.tag target.Oracle)
                            p.Name
                    )
                )

            if argmax.Length <> sequence.Length || margin.Length <> sequence.Length then
                raise (GoldenFormatException(sprintf "tf_*.%s: argmax or margin rows differ from input_ids" p.Name))

        if sequence.Length <> t + g.Steps - 1 then
            raise (
                GoldenFormatException(
                    sprintf
                        "tf_*.%s: %d rows; the prompt and %d steps make %d"
                        p.Name
                        sequence.Length
                        g.Steps
                        (t + g.Steps - 1)
                )
            )

        checkIds (Invariant.format $"tf_*.{p.Name} (its first {t} rows)") sequence64[0 .. t - 1] p.Ids

        // 1. The teacher-forced prefill: argmax on every row, the prompt rows (logits, or every tap) read back.
        let clock = Stopwatch.StartNew()
        let filter = if p.Taps then TapFilter.All else TapFilter.Off
        let sinks = Dictionary<string, RowSink>(StringComparer.Ordinal)
        let order = List<string>()
        let argmax = Array.zeroCreate<uint32> sequence.Length

        prefill ops model sequence o.Chunk filter (fun row0 logits taps ->
            let n = ops.Rows logits
            Array.blit (ops.Argmax logits) 0 argmax row0 n
            let keep = min n (t - row0)

            if keep > 0 then
                let put (name: string) (x: 't) =
                    match sinks.TryGetValue name with
                    | true, s -> s.Write(row0, ops.ReadRows(x, 0, keep))
                    | _ ->
                        let s = RowSink(ops.DType, t)
                        s.Write(row0, ops.ReadRows(x, 0, keep))
                        sinks[name] <- s
                        order.Add name

                if p.Taps then
                    for name, x in taps do
                        put name x
                else
                    put "logits" logits)

        let rows = [ for name in order -> name, sinks[name].Tensor ]
        sinks.Clear()
        let prefillSeconds = clock.Elapsed.TotalSeconds
        let checks = List<CheckResult>()

        for target in o.Targets do
            let name =
                Invariant.format $"fwd_{GoldenOracle.tag target.Oracle}.{p.Name}.safetensors"

            use f = g.Set.OpenSafetensors name
            checkIds name (f.ReadI64 "input_ids").Data p.Ids

            for r in GoldenCheck.tensors f rows (boundOf g target) (fun tap -> attrs target o.Backend p.Name tap) true do
                checks.AddRange r.Checks

        for target, _, reference, margin in tf do
            checks.Add(
                teacherForced
                    (attrs target o.Backend p.Name TeacherForcedTap)
                    argmax
                    reference
                    margin
                    g.TeacherForcedMinMargin
            )

        // 2. Greedy decoding on the canonical schedule.
        clock.Restart()

        let greedy =
            Greedy.run
                ops
                model
                p.Ids
                { GreedyOptions.parity g.Steps with
                    Margins = true
                }

        let greedySeconds = clock.Elapsed.TotalSeconds

        for target in o.Targets do
            let golden = g.GreedyIds(p.Name, target.Oracle)

            checks.Add(
                GoldenCheck.greedy (attrs target o.Backend p.Name GreedyTap) greedy.Ids[0 .. p.GatedSteps - 1] golden
            )

        Telemetry.gauge
            "goldens.greedy.min_margin"
            (Array.min greedy.Margins)
            TelemetryUnits.NoUnit
            [
                Attr.str "prompt" p.Name
                Attr.str "oracle" o.Backend
                Attr.int "steps" greedy.Margins.Length
            ]

        {
            Prompt = p
            Backend = o.Backend
            Checks = List.ofSeq checks
            Greedy = Some greedy
            Rows = sequence.Length
            PromptRows = (if o.KeepPromptRows then rows else [])
            PrefillSeconds = prefillSeconds
            GreedySeconds = greedySeconds
            TotalSeconds = total.Elapsed.TotalSeconds
        }

    /// One long prompt on one backend: a prefill of the prompt in chunks of ParityOptions.Chunk rows with
    /// final_norm.out; `final_norm.out` (every row) and `logits` (at the file's `logits.rows`) against long_* by relL2,
    /// and max |Δlogit| and KL where the pair has a bound; the argmax of every row against the file's where its
    /// reference margin is at least the threshold. The ops' plan must cover the chunk's rows, the prompt's positions
    /// and the final_norm.out tap.
    let longPrompt (ops: IForwardOps<'t>) (model: Qwen3Model) (g: QwenGoldens) (p: GoldenPrompt) (o: ParityOptions) =
        if p.Role <> GoldenPromptRole.Long then
            invalidArg "p" (sprintf "%s is not a long prompt" p.Name)

        requireTargets o
        let total = Stopwatch.StartNew()
        let t = p.Ids.Length

        use _ =
            Telemetry.span
                ParitySpan
                [
                    Attr.str TelemetryKeys.Component Component
                    Attr.str "prompt" p.Name
                    Attr.str TelemetryKeys.Backend o.Backend
                    Attr.str "role" "long"
                    Attr.int "tokens" t
                ]

        let fileOf (target: ParityTarget) =
            Invariant.format $"long_{GoldenOracle.tag target.Oracle}.{p.Name}.safetensors"

        let refs =
            [
                for target in o.Targets ->
                    use f = g.Set.OpenSafetensors(fileOf target)
                    checkIds (fileOf target) (f.ReadI64 "input_ids").Data p.Ids
                    target, (f.ReadI64 "logits.rows").Data, (f.ReadI64 "argmax").Data, (f.ReadF64 "margin").Data
            ]

        let _, stored64, _, _ = refs.Head

        for target, stored, argmax, margin in refs do
            if stored <> stored64 then
                raise (GoldenFormatException(sprintf "%s: other logits.rows than the other oracle" (fileOf target)))

            if argmax.Length <> t || margin.Length <> t then
                raise (
                    GoldenFormatException(sprintf "%s: argmax or margin rows differ from the prompt" (fileOf target))
                )

        let stored = stored64 |> Array.map int

        if
            stored.Length = 0
            || stored |> Array.exists (fun r -> r < 0 || r >= t)
            || stored |> Array.pairwise |> Array.exists (fun (a, b) -> b <= a)
        then
            raise (
                GoldenFormatException(sprintf "long_*.%s: logits.rows must increase strictly inside the prompt" p.Name)
            )

        let clock = Stopwatch.StartNew()
        let finalNorm = RowSink(ops.DType, t)
        let logitRows = RowSink(ops.DType, stored.Length)
        let argmax = Array.zeroCreate<uint32> t

        prefill ops model p.Ids o.Chunk (TapFilter.Names [ "final_norm.out" ]) (fun row0 logits taps ->
            let n = ops.Rows logits
            Array.blit (ops.Argmax logits) 0 argmax row0 n

            let f =
                taps
                |> List.tryFind (fun (name, _) -> name = "final_norm.out")
                |> Option.map snd
                |> Option.defaultWith (fun () -> raise (ForwardException "the prefill recorded no final_norm.out"))

            finalNorm.Write(row0, ops.ReadRows(f, 0, n))

            stored
            |> Array.iteri (fun k r ->
                if r >= row0 && r < row0 + n then
                    logitRows.Write(k, ops.ReadRows(logits, r - row0, 1))))

        let outputs = [ "final_norm.out", finalNorm.Tensor; "logits", logitRows.Tensor ]

        let prefillSeconds = clock.Elapsed.TotalSeconds
        let checks = List<CheckResult>()

        for target, _, reference, margin in refs do
            use f = g.Set.OpenSafetensors(fileOf target)

            for r in
                GoldenCheck.tensors f outputs (boundOf g target) (fun tap -> attrs target o.Backend p.Name tap) false do
                if r.Checks.IsEmpty then
                    raise (GoldenFormatException(sprintf "%s holds no %s" (fileOf target) r.Name))

                checks.AddRange r.Checks

            checks.Add(
                teacherForced
                    (attrs target o.Backend p.Name TeacherForcedTap)
                    argmax
                    reference
                    margin
                    g.TeacherForcedMinMargin
            )

        {
            Prompt = p
            Backend = o.Backend
            Checks = List.ofSeq checks
            Greedy = None
            Rows = t
            PromptRows = (if o.KeepPromptRows then outputs else [])
            PrefillSeconds = prefillSeconds
            GreedySeconds = 0.0
            TotalSeconds = total.Elapsed.TotalSeconds
        }

    /// The prompt on its role's path: gatePrompt or longPrompt.
    let run (ops: IForwardOps<'t>) (model: Qwen3Model) (g: QwenGoldens) (p: GoldenPrompt) (o: ParityOptions) =
        match p.Role with
        | GoldenPromptRole.Gate -> gatePrompt ops model g p o
        | GoldenPromptRole.Long -> longPrompt ops model g p o

    /// The ops plan a prompt needs on the whole model: its positions (the teacher-forced sequence or greedy run of a
    /// gate prompt, the prompt of a long one), the rows of its largest forward (the chunk, or the greedy prefill of the
    /// prompt) and every tap.
    let plan (threads: int) (config: Qwen3Config) (g: QwenGoldens) (p: GoldenPrompt) (o: ParityOptions) =
        let positions =
            match p.Role with
            | GoldenPromptRole.Gate -> p.Ids.Length + g.Steps - 1
            | GoldenPromptRole.Long -> p.Ids.Length

        let rows =
            match p.Role with
            | GoldenPromptRole.Gate -> max (min o.Chunk positions) p.Ids.Length
            | GoldenPromptRole.Long -> min o.Chunk positions

        { OpsOptions.defaults config with
            Threads = threads
            MaxContext = positions
            MaxRows = rows
            Taps = TapFilter.All
            Caches = 1
        }
