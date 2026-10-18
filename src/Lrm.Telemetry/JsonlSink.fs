namespace Lrm.Telemetry

open System
open System.Collections.Concurrent
open System.IO
open System.Text
open System.Threading
open Lrm.Core

/// The production run-log sink (RunLog contract): appends one events.<node>.jsonl line per record.
///
/// - Thread-safe. Sequence numbers are assigned under the write lock, so file order is seq order.
/// - Never throws from Write. A record that breaks the naming rules (TelemetryNames.check) is dropped and counted as
///   invalid; a record that cannot be written (an I/O error, or the sink is closed) is dropped and counted as dropped.
/// - Buffered: lines reach the file when the 64 KiB buffer fills, on Flush, every `flushInterval` (1 s by default)
///   and on Dispose. A reader tolerates a cut-off last line.
/// - The file is created new; an existing file is never overwritten.
[<Sealed>]
type JsonlSink private (stream: Stream, node: NodeId, path: string, flushInterval: TimeSpan) as this =
    let gate = obj ()
    let sb = StringBuilder(512)
    // One encoder carries a surrogate pair across the builder's chunk boundaries; flushed at the end of every line.
    let encoder = UTF8Encoding(false).GetEncoder()
    let mutable bytes: byte[] = Array.zeroCreate 1024
    let mutable nextSeq = 0L
    let mutable dropped = 0L
    let mutable invalid = 0L
    let mutable closed = false

    // Names repeat, so their validity is cached (bounded); the rules are TelemetryNames.check's.
    static let names = ConcurrentDictionary<string, bool>(StringComparer.Ordinal)

    static let validName (s: string) =
        if isNull s then
            false
        else
            match names.TryGetValue s with
            | true, ok -> ok
            | _ ->
                let ok = TelemetryNames.isValid s

                if names.Count < 8192 then
                    names.TryAdd(s, ok) |> ignore

                ok

    static let isValid (r: TelemetryRecord) =
        validName r.Name
        && TelemetryNames.isValidUnit r.Unit
        && r.Attrs
           |> List.forall (fun a ->
               validName a.Key
               && (match a.Value with
                   | AttrValue.Str null -> false
                   | _ -> true))

    let timer =
        if flushInterval > TimeSpan.Zero then
            new Timer((fun _ -> this.Flush()), null, flushInterval, flushInterval)
        else
            null

    /// Creates `path` (it must not exist) and writes node `node`'s records to it.
    new(path: string, node: NodeId) = new JsonlSink(path, node, TimeSpan.FromSeconds 1.0)

    /// Creates `path` (it must not exist), flushing every `flushInterval` (TimeSpan.Zero: only when the buffer fills,
    /// on Flush and on Dispose).
    new(path: string, node: NodeId, flushInterval: TimeSpan) =
        let stream =
            new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 65536, FileOptions.None)

        new JsonlSink(stream, node, path, flushInterval)

    /// A sink over any writable stream (for tests). The sink owns and disposes the stream.
    static member OfStream(stream: Stream, node: NodeId) =
        new JsonlSink(stream, node, "", TimeSpan.Zero)

    /// The events file, or "" for a stream sink.
    member _.Path = path

    /// The node whose records this sink writes.
    member _.Node = node

    /// Records written so far, which is also the seq of the next one.
    member _.Records = Interlocked.Read(&nextSeq)

    /// Records lost to I/O errors or written after the sink closed.
    member _.Dropped = Interlocked.Read(&dropped)

    /// Records refused because they broke the naming rules.
    member _.Invalid = Interlocked.Read(&invalid)

    /// Writes buffered lines to the file. Never throws.
    member _.Flush() =
        lock gate (fun () ->
            if not closed then
                try
                    stream.Flush()
                with _ ->
                    ())

    /// Writes one record. Never throws.
    member _.Write(record: TelemetryRecord) =
        if not (isValid record) then
            Interlocked.Increment(&invalid) |> ignore
        else
            lock gate (fun () ->
                if closed then
                    dropped <- dropped + 1L
                else
                    try
                        sb.Clear() |> ignore
                        EventLine.append sb node.Value nextSeq record
                        sb.Append '\n' |> ignore
                        // Encode the builder's chunks straight into the byte buffer: no string per line. The encoder
                        // keeps a high surrogate that ends one chunk for the low surrogate that starts the next, so a
                        // line's bytes never depend on how the builder happened to split it.
                        let mutable count = 0
                        let mutable chunks = sb.GetChunks()
                        encoder.Reset()

                        while chunks.MoveNext() do
                            let span = chunks.Current.Span
                            // GetMaxByteCount(n) also covers a surrogate held over from the previous chunk.
                            let needed = count + Encoding.UTF8.GetMaxByteCount span.Length

                            if bytes.Length < needed then
                                Array.Resize(&bytes, max needed (bytes.Length * 2))

                            count <-
                                count
                                + encoder.GetBytes(span, Span<byte>(bytes, count, bytes.Length - count), false)

                        // The flush writes nothing for the lines EventLine builds (it never leaves a lone surrogate).
                        let needed = count + Encoding.UTF8.GetMaxByteCount 0

                        if bytes.Length < needed then
                            Array.Resize(&bytes, max needed (bytes.Length * 2))

                        count <-
                            count
                            + encoder.GetBytes(
                                ReadOnlySpan<char>.Empty,
                                Span<byte>(bytes, count, bytes.Length - count),
                                true
                            )

                        stream.Write(bytes, 0, count)
                        nextSeq <- nextSeq + 1L
                    with _ ->
                        dropped <- dropped + 1L)

    /// Flushes and closes the file. Later records are dropped and counted. A second call does nothing.
    member _.Close() =
        let wasOpen =
            lock gate (fun () ->
                if closed then
                    false
                else
                    closed <- true
                    true)

        if wasOpen then
            if not (isNull timer) then
                timer.Dispose()

            lock gate (fun () ->
                try
                    stream.Flush()
                with _ ->
                    ()

                try
                    stream.Dispose()
                with _ ->
                    ())

    interface ITelemetrySink with
        member x.Write(record) = x.Write record

    interface IDisposable with
        member x.Dispose() = x.Close()
