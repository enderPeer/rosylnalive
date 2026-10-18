namespace Lrm.Tensor

open System
open System.IO
open System.Text.Json
open Lrm.Core

/// A tolerances file that breaks the ToleranceSpec rules.
type ToleranceFormatException(message: string) =
    inherit FormatException("tolerances: " + message)

/// The comparison pairs of the ToleranceSpec contract: candidate, then reference.
[<RequireQualifiedAccess>]
type TolerancePair =
    | Cpu64VsTorch64p
    | Cpu32VsTorch32
    | Cpu32VsTorch64p
    | GpuVsCpu32
    | GpuVsTorch32
    | GpuVsTorch64p
    | VulkanVsD3D12

[<RequireQualifiedAccess>]
module TolerancePair =
    let all =
        [
            TolerancePair.Cpu64VsTorch64p
            TolerancePair.Cpu32VsTorch32
            TolerancePair.Cpu32VsTorch64p
            TolerancePair.GpuVsCpu32
            TolerancePair.GpuVsTorch32
            TolerancePair.GpuVsTorch64p
            TolerancePair.VulkanVsD3D12
        ]

    /// The frozen name, for example "cpu32-vs-torch32"; also the `pair` attribute of golden.compare.
    let name =
        function
        | TolerancePair.Cpu64VsTorch64p -> "cpu64-vs-torch64p"
        | TolerancePair.Cpu32VsTorch32 -> "cpu32-vs-torch32"
        | TolerancePair.Cpu32VsTorch64p -> "cpu32-vs-torch64p"
        | TolerancePair.GpuVsCpu32 -> "gpu-vs-cpu32"
        | TolerancePair.GpuVsTorch32 -> "gpu-vs-torch32"
        | TolerancePair.GpuVsTorch64p -> "gpu-vs-torch64p"
        | TolerancePair.VulkanVsD3D12 -> "vulkan-vs-d3d12"

    let tryParse (text: string) =
        all |> List.tryFind (fun p -> name p = text)

/// The metrics a tolerance bounds.
[<RequireQualifiedAccess>]
type ToleranceMetric =
    /// relL2 per tap (Metrics.relL2).
    | RelL2
    /// max |delta| (Metrics.maxAbs); for logits, max |delta logit|.
    | MaxAbs
    /// The largest per-row KL(P_reference || P_candidate) of logits.
    | Kl

[<RequireQualifiedAccess>]
module ToleranceMetric =
    let all = [ ToleranceMetric.RelL2; ToleranceMetric.MaxAbs; ToleranceMetric.Kl ]

    /// "relL2", "maxAbs" or "kl": the JSON key and the comparator's metric name.
    let name =
        function
        | ToleranceMetric.RelL2 -> "relL2"
        | ToleranceMetric.MaxAbs -> "maxAbs"
        | ToleranceMetric.Kl -> "kl"

    let tryParse (text: string) =
        all |> List.tryFind (fun m -> name m = text)

/// The calibration rule of a pair: bound = clamp(K * e_ref, floor, 10 * floor) per metric.
type ToleranceRule =
    {
        K: float
        Floors: Map<ToleranceMetric, float>
    }

/// One bound, with the e_ref it was calibrated from (absent for fixed bounds).
type ToleranceBound = { Bound: float; ERef: float option }

/// The bounds of one pair: per tap kind, with an optional default for kinds not listed.
type PairTolerances =
    {
        Pair: TolerancePair
        Rule: ToleranceRule option
        Default: Map<ToleranceMetric, ToleranceBound>
        Taps: Map<string, Map<ToleranceMetric, ToleranceBound>>
    }

/// A parsed tolerances.json (ToleranceSpec contract; G2 owns the numbers, O2 the types).
type ToleranceSpec =
    {
        Set: string
        Pairs: Map<TolerancePair, PairTolerances>
        /// G2's greedy rules, kept as JSON.
        Greedy: JsonElement option
    }

    /// The bound for a pair, tap kind and metric: the tap kind's own entry, else the pair's default.
    member s.TryBound(pair: TolerancePair, tapKind: string, metric: ToleranceMetric) =
        s.Pairs
        |> Map.tryFind pair
        |> Option.bind (fun p ->
            match p.Taps |> Map.tryFind tapKind |> Option.bind (Map.tryFind metric) with
            | Some b -> Some b
            | None -> p.Default |> Map.tryFind metric)

    /// The bound value; raises KeyNotFoundException when the spec has none.
    member s.Bound(pair: TolerancePair, tapKind: string, metric: ToleranceMetric) =
        match s.TryBound(pair, tapKind, metric) with
        | Some b -> b.Bound
        | None ->
            raise (
                Collections.Generic.KeyNotFoundException(
                    sprintf "no %s bound for %s %s" (ToleranceMetric.name metric) (TolerancePair.name pair) tapKind
                )
            )

[<RequireQualifiedAccess>]
module ToleranceSpec =
    /// The schema tag of tolerances.json.
    [<Literal>]
    let Schema = "lrm.tolerances/1"

    /// clamp(k * eRef, floor, 10 * floor), in float64 (the ToleranceSpec rule).
    let calibrate (k: float) (floor: float) (eRef: float) =
        min (max (k * eRef) floor) (10.0 * floor)

    let private fail (msg: string) : 'a = raise (ToleranceFormatException msg)

    let private positive (where: string) (e: JsonElement) =
        match e.ValueKind with
        | JsonValueKind.Number ->
            let v = e.GetDouble()

            if Double.IsFinite v && v > 0.0 then
                v
            else
                fail (sprintf "%s: %s is not a positive finite number" where (e.GetRawText()))
        | _ -> fail (sprintf "%s: %s is not a number" where (e.GetRawText()))

    let private nonNegative (where: string) (e: JsonElement) =
        match e.ValueKind with
        | JsonValueKind.Number ->
            let v = e.GetDouble()

            if Double.IsFinite v && v >= 0.0 then
                v
            else
                fail (sprintf "%s: %s is not a non-negative finite number" where (e.GetRawText()))
        | _ -> fail (sprintf "%s: %s is not a number" where (e.GetRawText()))

    /// Raises for a duplicate key in any object of the tree, arrays included (the contract's "a duplicate key
    /// anywhere"): JsonDocument keeps every duplicate, and TryGetProperty would silently return the last one.
    let rec private noDuplicates (path: string) (e: JsonElement) =
        match e.ValueKind with
        | JsonValueKind.Object ->
            let seen = Collections.Generic.HashSet<string>(StringComparer.Ordinal)

            for p in e.EnumerateObject() do
                if not (seen.Add p.Name) then
                    fail (
                        sprintf "duplicate key '%s' %s" p.Name (if path = "" then "at the top level" else "in " + path)
                    )

                noDuplicates (if path = "" then p.Name else path + "." + p.Name) p.Value
        | JsonValueKind.Array ->
            let mutable i = 0

            for x in e.EnumerateArray() do
                noDuplicates (Invariant.format $"{path}[{i}]") x
                i <- i + 1
        | _ -> ()

    let private objectOf (where: string) (e: JsonElement) =
        if e.ValueKind <> JsonValueKind.Object then
            fail (sprintf "%s is not an object" where)

        let props = e.EnumerateObject() |> List.ofSeq

        match props |> List.countBy _.Name |> List.tryFind (fun (_, n) -> n > 1) with
        | Some(name, _) -> fail (sprintf "%s: duplicate key '%s'" where name)
        | None -> props

    let private metricOf (where: string) (key: string) =
        match ToleranceMetric.tryParse key with
        | Some m -> m
        | None -> fail (sprintf "%s: unknown metric '%s' (expected relL2, maxAbs or kl)" where key)

    let private bounds (where: string) (rule: ToleranceRule option) (e: JsonElement) =
        objectOf where e
        |> List.map (fun p ->
            let metric = metricOf where p.Name
            let at = where + "." + p.Name
            let mutable bound = None
            let mutable eRef = None

            for f in objectOf at p.Value do
                match f.Name with
                | "bound" -> bound <- Some(positive (at + ".bound") f.Value)
                | "e_ref" -> eRef <- Some(nonNegative (at + ".e_ref") f.Value)
                | other -> fail (sprintf "%s: unknown field '%s'" at other)

            let bound =
                match bound with
                | Some b -> b
                | None -> fail (sprintf "%s: 'bound' is missing" at)

            match rule, eRef with
            | Some r, Some e ->
                let floor =
                    match r.Floors |> Map.tryFind metric with
                    | Some f -> f
                    | None -> fail (sprintf "%s: the rule has no %s floor" at p.Name)

                let expected = calibrate r.K floor e

                if expected <> bound then
                    fail (
                        sprintf
                            "%s: bound %s is not clamp(%s * %s, %s, 10 * %s) = %s"
                            at
                            (Invariant.f64 bound)
                            (Invariant.f64 r.K)
                            (Invariant.f64 e)
                            (Invariant.f64 floor)
                            (Invariant.f64 floor)
                            (Invariant.f64 expected)
                    )
            | Some _, None -> fail (sprintf "%s: a calibrated pair needs e_ref for every bound" at)
            | None, _ -> ()

            metric, { Bound = bound; ERef = eRef })
        |> Map.ofList

    let private parseCore (json: string) : ToleranceSpec =
        let doc =
            try
                JsonDocument.Parse json
            with :? JsonException as e ->
                fail ("not valid JSON: " + e.Message)

        use doc = doc
        let root = doc.RootElement.Clone()

        if root.ValueKind <> JsonValueKind.Object then
            fail "the file is not a JSON object"

        noDuplicates "" root

        let str (key: string) =
            match root.TryGetProperty key with
            | true, v when v.ValueKind = JsonValueKind.String -> v.GetString()
            | _ -> fail (sprintf "'%s' is missing or not a string" key)

        let schema = str "schema"

        if schema <> Schema then
            fail (sprintf "unknown schema '%s' (expected %s)" schema Schema)

        let set = str "set"

        let pairsElement =
            match root.TryGetProperty "pairs" with
            | true, v -> v
            | _ -> fail "'pairs' is missing"

        let pairs =
            objectOf "pairs" pairsElement
            |> List.map (fun p ->
                let pair =
                    match TolerancePair.tryParse p.Name with
                    | Some x -> x
                    | None -> fail (sprintf "unknown pair '%s'" p.Name)

                let at = "pairs." + p.Name
                let fields = objectOf at p.Value

                let field (name: string) =
                    fields |> List.tryFind (fun f -> f.Name = name) |> Option.map _.Value

                for f in fields do
                    if f.Name <> "rule" && f.Name <> "default" && f.Name <> "taps" then
                        fail (sprintf "%s: unknown field '%s'" at f.Name)

                let rule =
                    field "rule"
                    |> Option.map (fun r ->
                        let rf = objectOf (at + ".rule") r

                        for f in rf do
                            if f.Name <> "k" && f.Name <> "floors" then
                                fail (sprintf "%s.rule: unknown field '%s'" at f.Name)

                        let k =
                            match rf |> List.tryFind (fun f -> f.Name = "k") with
                            | Some f -> positive (at + ".rule.k") f.Value
                            | None -> fail (sprintf "%s.rule: 'k' is missing" at)

                        let floors =
                            match rf |> List.tryFind (fun f -> f.Name = "floors") with
                            | Some f ->
                                objectOf (at + ".rule.floors") f.Value
                                |> List.map (fun m ->
                                    metricOf (at + ".rule.floors") m.Name,
                                    positive (at + ".rule.floors." + m.Name) m.Value)
                                |> Map.ofList
                            | None -> fail (sprintf "%s.rule: 'floors' is missing" at)

                        { K = k; Floors = floors })

                let defaults =
                    field "default"
                    |> Option.map (bounds (at + ".default") rule)
                    |> Option.defaultValue Map.empty

                let taps =
                    field "taps"
                    |> Option.map (fun t ->
                        objectOf (at + ".taps") t
                        |> List.map (fun k -> k.Name, bounds (at + ".taps." + k.Name) rule k.Value)
                        |> Map.ofList)
                    |> Option.defaultValue Map.empty

                pair,
                {
                    Pair = pair
                    Rule = rule
                    Default = defaults
                    Taps = taps
                })

        if pairs |> List.map fst |> List.distinct |> List.length <> pairs.Length then
            fail "a pair is listed twice"

        {
            Set = set
            Pairs = Map.ofList pairs
            Greedy =
                match root.TryGetProperty "greedy" with
                | true, g -> Some g
                | _ -> None
        }

    /// Parses and validates tolerances.json text: the schema, known pairs and metrics, positive finite bounds, no
    /// duplicate key anywhere, and for every pair with a rule, bound = clamp(k * e_ref, floor, 10 * floor) exactly for
    /// every entry. Every violation raises ToleranceFormatException.
    let parse (json: string) : ToleranceSpec =
        try
            parseCore json
        with
        // System.Text.Json decodes keys and strings lazily: a lone surrogate escape raises InvalidOperationException
        // when it is read, and a lone surrogate in the text itself raises ArgumentException when it is transcoded.
        | :? InvalidOperationException as e -> fail ("the file is not valid Unicode: " + e.Message)
        | :? ArgumentException as e -> fail ("the file is not valid Unicode: " + e.Message)

    /// Loads goldens/<set>/<version>/tolerances.json (or any path).
    let load (path: string) = parse (File.ReadAllText path)
