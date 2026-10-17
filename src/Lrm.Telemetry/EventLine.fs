namespace Lrm.Telemetry

open System
open System.Text
open System.Text.Json
open Lrm.Core

/// One telemetry record as read back from a run log (RunLog contract).
type RunEvent =
    {
        /// The node that wrote it (the events.<node>.jsonl file).
        Node: string
        /// Its position in that node's file: 0, 1, 2, ... with no gaps unless records were lost.
        Seq: int64
        Kind: TelemetryKind
        Name: string
        /// Host wall-clock time in Unix ns: the start of a span, otherwise when the record was made.
        TimeNs: int64
        /// Span: host duration in ns. GpuInterval: GPU duration in ns. Otherwise 0.
        DurationNs: int64
        Value: float
        Unit: string
        /// Span: its id. Otherwise 0.
        SpanId: int64
        /// The enclosing span's id, or 0.
        ParentId: int64
        /// GpuInterval: start in the GPU timestamp domain, in ns. Otherwise 0.
        GpuStartNs: int64
        /// The attributes, each key once, in first-occurrence order (scope attributes first).
        Attrs: Attr list
    }

    /// The value of attribute `key`.
    member e.TryAttr(key: string) =
        e.Attrs
        |> List.tryFind (fun a -> Invariant.equals a.Key key)
        |> Option.map _.Value

    /// A string attribute.
    member e.TryStr(key: string) =
        match e.TryAttr key with
        | Some(AttrValue.Str s) -> Some s
        | _ -> None

    /// An integer attribute.
    member e.TryInt(key: string) =
        match e.TryAttr key with
        | Some(AttrValue.Int i) -> Some i
        | _ -> None

    /// A numeric attribute: a float, or an integer widened to float.
    member e.TryFloat(key: string) =
        match e.TryAttr key with
        | Some(AttrValue.Float f) -> Some f
        | Some(AttrValue.Int i) -> Some(float i)
        | _ -> None

    /// A boolean attribute.
    member e.TryBool(key: string) =
        match e.TryAttr key with
        | Some(AttrValue.Bool b) -> Some b
        | _ -> None

    /// Any attribute as display text: strings as they are, numbers in invariant form, booleans as true or false.
    member e.TryText(key: string) =
        match e.TryAttr key with
        | Some(AttrValue.Str s) -> Some s
        | Some(AttrValue.Int i) -> Some(Invariant.i64 i)
        | Some(AttrValue.Float f) -> Some(Invariant.f64 f)
        | Some(AttrValue.Bool b) -> Some(if b then "true" else "false")
        | None -> None

    /// The end of a span (TimeNs + DurationNs).
    member e.EndNs = e.TimeNs + e.DurationNs

/// The events.<node>.jsonl line format, schema version 1 (RunLog contract). One record is one line of compact JSON
/// with the keys v, t, k, n, d, x, u, node, seq and a, always in that order and always present.
[<RequireQualifiedAccess>]
module EventLine =
    /// The line schema version written in "v".
    [<Literal>]
    let Version = 1

    /// Structural attribute: the span id (span records only).
    [<Literal>]
    let SpanKey = "$span"

    /// Structural attribute: the enclosing span's id (when not 0).
    [<Literal>]
    let ParentKey = "$parent"

    /// Structural attribute: the GPU start in ns (gpu records only).
    [<Literal>]
    let GpuStartKey = "$gpu_start"

    /// Each key once, at its first position, with the value of its last occurrence (the innermost scope wins).
    let dedupe (attrs: Attr list) : Attr list =
        let rec hasRepeat (xs: Attr list) =
            match xs with
            | [] -> false
            | a :: rest ->
                List.exists (fun (b: Attr) -> Invariant.equals a.Key b.Key) rest
                || hasRepeat rest

        if not (hasRepeat attrs) then
            attrs
        else
            attrs
            |> List.map _.Key
            |> List.distinct
            |> List.map (fun k -> attrs |> List.findBack (fun a -> Invariant.equals a.Key k))

    let private appendAttrValue (sb: StringBuilder) (v: AttrValue) =
        match v with
        | AttrValue.Str s -> Json.appendString sb s
        | AttrValue.Int i -> Json.appendInt64 sb i
        | AttrValue.Float f -> Json.appendTypedFloat sb f
        | AttrValue.Bool b -> sb.Append(if b then "true" else "false") |> ignore

    /// Appends the line for `record` (without the final LF).
    let append (sb: StringBuilder) (node: string) (seq: int64) (r: TelemetryRecord) =
        sb.Append("{\"v\":") |> ignore
        Json.appendInt64 sb (int64 Version)
        sb.Append(",\"t\":") |> ignore
        Json.appendInt64 sb r.TimeNs
        sb.Append(",\"k\":") |> ignore
        Json.appendString sb (TelemetryKind.name r.Kind)
        sb.Append(",\"n\":") |> ignore
        Json.appendString sb r.Name
        sb.Append(",\"d\":") |> ignore
        Json.appendInt64 sb r.DurationNs
        sb.Append(",\"x\":") |> ignore
        Json.appendNumber sb r.Value
        sb.Append(",\"u\":") |> ignore
        Json.appendString sb r.Unit
        sb.Append(",\"node\":") |> ignore
        Json.appendString sb node
        sb.Append(",\"seq\":") |> ignore
        Json.appendInt64 sb seq
        sb.Append(",\"a\":{") |> ignore
        let mutable first = true

        let key (k: string) =
            if not first then
                sb.Append ',' |> ignore

            first <- false
            Json.appendString sb k
            sb.Append ':' |> ignore

        if r.Kind = TelemetryKind.Span then
            key SpanKey
            Json.appendInt64 sb r.SpanId

        if r.ParentId <> 0L then
            key ParentKey
            Json.appendInt64 sb r.ParentId

        if r.Kind = TelemetryKind.GpuInterval then
            key GpuStartKey
            Json.appendInt64 sb r.GpuStartNs

        for a in dedupe r.Attrs do
            key a.Key
            appendAttrValue sb a.Value

        sb.Append "}}" |> ignore

    /// The line for `record`, without the final LF.
    let encode (node: string) (seq: int64) (r: TelemetryRecord) =
        let sb = StringBuilder(256)
        append sb node seq r
        sb.ToString()

    let private kinds =
        TelemetryKind.all |> List.map (fun k -> TelemetryKind.name k, k) |> Map.ofList

    let private isIntegerToken (raw: string) =
        raw.IndexOf('.') < 0 && raw.IndexOf('e') < 0 && raw.IndexOf('E') < 0

    exception private LineError of string

    let private fail fmt =
        Printf.ksprintf (fun s -> raise (LineError s)) fmt

    let private int64Of (what: string) (e: JsonElement) =
        if e.ValueKind <> JsonValueKind.Number then
            fail "%s is not a number" what

        let raw = e.GetRawText()

        if not (isIntegerToken raw) then
            fail "%s is not an integer: %s" what raw

        match Invariant.tryParseI64 raw with
        | Some v -> v
        | None -> fail "%s is out of range: %s" what raw

    let private stringOf (what: string) (e: JsonElement) =
        if e.ValueKind <> JsonValueKind.String then
            fail "%s is not a string" what

        e.GetString()

    let private floatOf (what: string) (e: JsonElement) =
        match e.ValueKind with
        | JsonValueKind.Number -> Invariant.parseF64 (e.GetRawText())
        | JsonValueKind.String ->
            match Json.tryNonFinite (e.GetString()) with
            | Some v -> v
            | None -> fail "%s is not a number" what
        | _ -> fail "%s is not a number" what

    let private attrValueOf (key: string) (e: JsonElement) =
        match e.ValueKind with
        | JsonValueKind.String -> AttrValue.Str(e.GetString())
        | JsonValueKind.True -> AttrValue.Bool true
        | JsonValueKind.False -> AttrValue.Bool false
        | JsonValueKind.Number ->
            let raw = e.GetRawText()

            if isIntegerToken raw then
                match Invariant.tryParseI64 raw with
                | Some v -> AttrValue.Int v
                | None -> fail "attribute '%s' is an integer out of range: %s" key raw
            else
                AttrValue.Float(Invariant.parseF64 raw)
        | JsonValueKind.Object ->
            let props = e.EnumerateObject() |> List.ofSeq

            match props with
            | [ p ] when p.Name = "$f" && p.Value.ValueKind = JsonValueKind.String ->
                match Json.tryNonFinite (p.Value.GetString()) with
                | Some v -> AttrValue.Float v
                | None -> fail "attribute '%s' has an unknown $f value" key
            | _ -> fail "attribute '%s' is an object other than {\"$f\":...}" key
        | _ -> fail "attribute '%s' has an unsupported JSON type" key

    let private expectedKeys = [ "v"; "t"; "k"; "n"; "d"; "x"; "u"; "node"; "seq"; "a" ]

    /// Parses one line. Every key must be present exactly once, "v" must be 1, and the structural attributes must match
    /// the kind: $span on span records only, $gpu_start on gpu records only.
    let decode (line: string) : Result<RunEvent, string> =
        try
            use doc = JsonDocument.Parse(line)
            let root = doc.RootElement

            if root.ValueKind <> JsonValueKind.Object then
                fail "the line is not a JSON object"

            let props =
                root.EnumerateObject() |> Seq.map (fun p -> p.Name, p.Value) |> List.ofSeq

            let names = props |> List.map fst

            if names.Length <> (List.distinct names).Length then
                fail "duplicate key"

            for k in names do
                if not (List.contains k expectedKeys) then
                    fail "unknown key '%s'" k

            let get (k: string) =
                match props |> List.tryFind (fun (n, _) -> n = k) with
                | Some(_, v) -> v
                | None -> fail "missing key '%s'" k

            let v = int64Of "v" (get "v")

            if v <> int64 Version then
                fail "unsupported line version %d" v

            let kindName = stringOf "k" (get "k")

            let kind =
                match kinds.TryFind kindName with
                | Some k -> k
                | None -> fail "unknown kind '%s'" kindName

            let a = get "a"

            if a.ValueKind <> JsonValueKind.Object then
                fail "a is not an object"

            let aprops = a.EnumerateObject() |> Seq.map (fun p -> p.Name, p.Value) |> List.ofSeq
            let akeys = aprops |> List.map fst

            if akeys.Length <> (List.distinct akeys).Length then
                fail "duplicate attribute key"

            let mutable spanId = None
            let mutable parent = 0L
            let mutable gpuStart = None
            let attrs = ResizeArray<Attr>()

            for k, e in aprops do
                match k with
                | SpanKey -> spanId <- Some(int64Of SpanKey e)
                | ParentKey -> parent <- int64Of ParentKey e
                | GpuStartKey -> gpuStart <- Some(int64Of GpuStartKey e)
                | k when k.StartsWith('$') -> fail "unknown structural attribute '%s'" k
                | k -> attrs.Add { Key = k; Value = attrValueOf k e }

            match kind, spanId with
            | TelemetryKind.Span, None -> fail "span record without $span"
            | k, Some _ when k <> TelemetryKind.Span -> fail "$span on a %s record" kindName
            | _ -> ()

            match kind, gpuStart with
            | TelemetryKind.GpuInterval, None -> fail "gpu record without $gpu_start"
            | k, Some _ when k <> TelemetryKind.GpuInterval -> fail "$gpu_start on a %s record" kindName
            | _ -> ()

            Ok
                {
                    Node = stringOf "node" (get "node")
                    Seq = int64Of "seq" (get "seq")
                    Kind = kind
                    Name = stringOf "n" (get "n")
                    TimeNs = int64Of "t" (get "t")
                    DurationNs = int64Of "d" (get "d")
                    Value = floatOf "x" (get "x")
                    Unit = stringOf "u" (get "u")
                    SpanId = defaultArg spanId 0L
                    ParentId = parent
                    GpuStartNs = defaultArg gpuStart 0L
                    Attrs = List.ofSeq attrs
                }
        with
        | LineError msg -> Error msg
        | :? JsonException as e -> Error("malformed JSON: " + e.Message)
        | :? FormatException as e -> Error("bad number: " + e.Message)
