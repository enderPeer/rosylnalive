/// A minimal run-log sink for the spike's telemetry, until T1's runtime lands. It writes
/// LRM_DATA/runs/<runId>/manifest.json and events.<node>.jsonl with the RunLog keys of the plan
/// (v, t, k, n, d, x, u, node, seq, a); T1's RunLog contract is authoritative and may differ in detail.
module VkHeapSpike.RunLog

open System
open System.IO
open System.Text
open System.Text.Json
open Lrm.Core

type JsonlSink(path: string, node: string) =
    let gate = obj ()
    let writer = new StreamWriter(path, false, UTF8Encoding(false))
    let mutable seq = 0L

    let attrValue (w: Utf8JsonWriter) (v: AttrValue) =
        match v with
        | AttrValue.Str s -> w.WriteStringValue s
        | AttrValue.Int i -> w.WriteNumberValue i
        | AttrValue.Float f -> w.WriteNumberValue f
        | AttrValue.Bool b -> w.WriteBooleanValue b

    member _.Path = path

    interface ITelemetrySink with
        member _.Write(r: TelemetryRecord) =
            try
                lock gate (fun () ->
                    seq <- seq + 1L
                    use ms = new MemoryStream()

                    do
                        use w = new Utf8JsonWriter(ms)
                        w.WriteStartObject()
                        w.WriteNumber("v", 1)
                        w.WriteNumber("t", r.TimeNs)
                        w.WriteString("k", TelemetryKind.name r.Kind)
                        w.WriteString("n", r.Name)
                        w.WriteNumber("d", r.DurationNs)
                        w.WriteNumber("x", r.Value)
                        w.WriteString("u", r.Unit)
                        w.WriteString("node", node)
                        w.WriteNumber("seq", seq)
                        w.WriteStartObject("a")

                        for a in r.Attrs do
                            w.WritePropertyName a.Key
                            attrValue w a.Value

                        if r.SpanId <> 0L then
                            w.WriteNumber("span", r.SpanId)

                        if r.ParentId <> 0L then
                            w.WriteNumber("parent", r.ParentId)

                        if r.GpuStartNs <> 0L then
                            w.WriteNumber("gpu_start", r.GpuStartNs)

                        w.WriteEndObject()
                        w.WriteEndObject()

                    writer.WriteLine(Encoding.UTF8.GetString(ms.ToArray()))
                    writer.Flush())
            with _ ->
                () // a production sink never throws

    interface IDisposable with
        member _.Dispose() = lock gate (fun () -> writer.Dispose())

/// Creates the run folder and manifest; returns the run id and the sink.
let start (tool: string) =
    let runId = RunId.fresh ()
    let dir = LrmPaths.run runId
    Directory.CreateDirectory dir |> ignore
    let manifest = JsonSerializer.Serialize(
        dict [
            "schema", box "lrm.run/1"
            "runId", box runId.Value
            "tool", box tool
            "provisional", box "spike sink; T1 owns the RunLog contract"
            "started", box (DateTimeOffset.UtcNow.ToString("o", Globalization.CultureInfo.InvariantCulture))
        ],
        JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(Path.Combine(dir, "manifest.json"), manifest + "\n")
    runId, new JsonlSink(Path.Combine(dir, "events.n0.jsonl"), "n0")

/// Adds the execution classes the run observed to the manifest.
let addClasses (runId: RunId) (classes: (string * ExecutionClass) list) =
    let path = Path.Combine(LrmPaths.run runId, "manifest.json")
    let node = Nodes.JsonNode.Parse(File.ReadAllText path)
    let arr = Nodes.JsonArray()

    for (label, c) in classes do
        let o = Nodes.JsonObject()
        o["label"] <- Nodes.JsonValue.Create label
        o["backend"] <- Nodes.JsonValue.Create(ExecutionBackend.name c.Backend)
        o["deviceId"] <- Nodes.JsonValue.Create c.DeviceId
        o["driver"] <- Nodes.JsonValue.Create c.Driver
        o["shaderSetHash"] <- Nodes.JsonValue.Create(c.ShaderSetHash |> Option.map (fun h -> h.Hex) |> Option.toObj)
        o["arch"] <- Nodes.JsonValue.Create c.Arch
        o["executionClassId"] <- Nodes.JsonValue.Create((ExecutionClass.id c).ToString())
        arr.Add o

    node["executionClasses"] <- arr
    File.WriteAllText(path, node.ToJsonString(JsonSerializerOptions(WriteIndented = true)) + "\n")
