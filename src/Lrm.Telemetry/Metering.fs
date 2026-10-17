namespace Lrm.Telemetry

open System
open Lrm.Core

/// Where metered use is attributed (RunLog contract, "Metering"): the node that wrote the record, and the component,
/// stage and layer attributes it carries (usually from Telemetry.scope).
type MeterKey =
    {
        Node: string
        /// The `component` attribute, or "".
        Component: string
        /// The `stage` attribute as text (an integer pipeline stage or a named stage), or "".
        Stage: string
        /// The `layer` attribute.
        Layer: int64 option
    }

/// Resource use summed over the records of one MeterKey.
type MeterTotals =
    {
        /// Host wall time: the self time of spans (duration minus the durations of their direct child spans, at least 0).
        WallNs: int64
        /// GPU time: the durations of gpu records.
        GpuNs: int64
        /// Meter records with unit B.
        Bytes: float
        /// Meter records with unit flop.
        Flops: float
        /// Meter records with unit token.
        Tokens: float
        /// Meter records with unit ns (explicitly metered time, for example a queue wait).
        MeterNs: float
        /// How many records contributed.
        Records: int
    }

    static member Zero =
        {
            WallNs = 0L
            GpuNs = 0L
            Bytes = 0.0
            Flops = 0.0
            Tokens = 0.0
            MeterNs = 0.0
            Records = 0
        }

    static member (+)(a: MeterTotals, b: MeterTotals) =
        {
            WallNs = a.WallNs + b.WallNs
            GpuNs = a.GpuNs + b.GpuNs
            Bytes = a.Bytes + b.Bytes
            Flops = a.Flops + b.Flops
            Tokens = a.Tokens + b.Tokens
            MeterNs = a.MeterNs + b.MeterNs
            Records = a.Records + b.Records
        }

/// The memory use of one heap arena on one node, from mem.arena gauges.
type ArenaUse =
    {
        Node: string
        Arena: string
        PeakBytes: float
        LastBytes: float
        Samples: int
    }

/// Metering (RunLog contract): helpers that emit metered records in the shapes reports read, and the attribution of
/// a run's records to component, stage, layer and node. The helpers only report; they never change a value.
[<RequireQualifiedAccess>]
module Metering =
    /// The attribute that names the registry operation of a metered record.
    [<Literal>]
    let OpKey = "op"

    /// Meters `amount` FLOPs as `name` (for example "model.flops").
    let flops (name: string) (amount: float) (attrs: Attr list) =
        Telemetry.meter name amount TelemetryUnits.Flops attrs

    /// Meters bytes moved as `name` (for example "gpu.bytes_moved").
    let bytes (name: string) (amount: float) (attrs: Attr list) =
        Telemetry.meter name amount TelemetryUnits.Bytes attrs

    /// Meters tokens processed as `name` (for example "model.tokens").
    let tokens (name: string) (amount: float) (attrs: Attr list) =
        Telemetry.meter name amount TelemetryUnits.Tokens attrs

    /// Meters a duration in ns as `name` (time that is not a span, such as a queue wait).
    let time (name: string) (ns: int64) (attrs: Attr list) =
        Telemetry.meter name (float ns) TelemetryUnits.Nanoseconds attrs

    /// Reports the bytes in use in a heap arena: a mem.arena gauge with the arena attribute.
    let arena (arenaName: string) (bytesInUse: int64) (attrs: Attr list) =
        Telemetry.gauge
            TelemetryNames.MemArena
            (float bytesInUse)
            TelemetryUnits.Bytes
            (Attr.str TelemetryKeys.Arena arenaName :: attrs)

    /// Meters one registry operation as two records, `<prefix>.flops` and `<prefix>.bytes_moved`, each with the op
    /// attribute (for example prefix "model" or "gpu"). Returns nothing: metering never feeds a computation.
    let op (prefix: string) (operation: Op) (attrs: Attr list) =
        let t = Telemetry.current ()

        if t.Enabled then
            let cost = OpCost.ofOp operation
            let attrs = Attr.str OpKey (OpCost.name operation) :: attrs
            t.Meter(prefix + ".flops", float cost.Flops, TelemetryUnits.Flops, attrs)
            t.Meter(prefix + ".bytes_moved", float cost.BytesMoved, TelemetryUnits.Bytes, attrs)

    /// The attribution key of a record.
    let keyOf (e: RunEvent) =
        {
            Node = e.Node
            Component = defaultArg (e.TryStr TelemetryKeys.Component) ""
            Stage = defaultArg (e.TryText TelemetryKeys.Stage) ""
            Layer = e.TryInt TelemetryKeys.Layer
        }

    /// The self time of every span: its duration minus its direct children's durations, at least 0, keyed by
    /// (node, span id).
    let selfTimes (events: RunEvent seq) : Map<string * int64, int64> =
        let spans =
            events |> Seq.filter (fun e -> e.Kind = TelemetryKind.Span) |> Array.ofSeq

        let childSum =
            spans
            |> Array.filter (fun e -> e.ParentId <> 0L)
            |> Array.groupBy (fun e -> e.Node, e.ParentId)
            |> Array.map (fun (k, xs) -> k, xs |> Array.sumBy _.DurationNs)
            |> Map.ofArray

        spans
        |> Array.map (fun e ->
            let children = childSum.TryFind(e.Node, e.SpanId) |> Option.defaultValue 0L
            (e.Node, e.SpanId), max 0L (e.DurationNs - children))
        |> Map.ofArray

    let private totalsOf (self: Map<string * int64, int64>) (e: RunEvent) =
        let z = { MeterTotals.Zero with Records = 1 }

        match e.Kind with
        | TelemetryKind.Span ->
            { z with
                WallNs = self.TryFind(e.Node, e.SpanId) |> Option.defaultValue 0L
            }
        | TelemetryKind.GpuInterval -> { z with GpuNs = e.DurationNs }
        | TelemetryKind.Meter ->
            match e.Unit with
            | TelemetryUnits.Bytes -> { z with Bytes = e.Value }
            | TelemetryUnits.Flops -> { z with Flops = e.Value }
            | TelemetryUnits.Tokens -> { z with Tokens = e.Value }
            | TelemetryUnits.Nanoseconds -> { z with MeterNs = e.Value }
            | _ -> z
        | _ -> MeterTotals.Zero

    let private contributes (e: RunEvent) =
        match e.Kind with
        | TelemetryKind.Span
        | TelemetryKind.GpuInterval
        | TelemetryKind.Meter -> true
        | _ -> false

    /// Resource use per attribution key, sorted by node, component, stage and layer (ordinal; no layer first).
    let attribute (events: RunEvent seq) : (MeterKey * MeterTotals) list =
        let events = events |> Array.ofSeq
        let self = selfTimes events

        events
        |> Array.filter contributes
        |> Array.groupBy keyOf
        |> Array.map (fun (k, xs) -> k, xs |> Array.fold (fun acc e -> acc + totalsOf self e) MeterTotals.Zero)
        |> Array.sortBy (fun (k, _) -> k.Node, k.Component, k.Stage, k.Layer)
        |> List.ofArray

    /// Sums attributed rows under a coarser key, for example (fun k -> { k with Layer = None }) per component.
    let rollup (project: MeterKey -> MeterKey) (rows: (MeterKey * MeterTotals) list) =
        rows
        |> List.groupBy (fst >> project)
        |> List.map (fun (k, xs) -> k, xs |> List.fold (fun acc (_, t) -> acc + t) MeterTotals.Zero)
        |> List.sortBy (fun (k, _) -> k.Node, k.Component, k.Stage, k.Layer)

    /// Peak and last bytes per (node, arena), from mem.arena gauges in record order.
    let arenas (events: RunEvent seq) : ArenaUse list =
        events
        |> Seq.filter (fun e -> e.Kind = TelemetryKind.Gauge && Invariant.equals e.Name TelemetryNames.MemArena)
        |> Seq.groupBy (fun e -> e.Node, defaultArg (e.TryStr TelemetryKeys.Arena) "")
        |> Seq.map (fun ((node, arena), xs) ->
            let xs = Array.ofSeq xs

            {
                Node = node
                Arena = arena
                PeakBytes = xs |> Array.map _.Value |> Array.max
                LastBytes = xs[xs.Length - 1].Value
                Samples = xs.Length
            })
        |> Seq.sortBy (fun a -> a.Node, a.Arena)
        |> List.ofSeq
