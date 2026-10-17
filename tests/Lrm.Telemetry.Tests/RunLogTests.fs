module Lrm.Telemetry.Tests.RunLogTests

open System
open System.IO
open System.Text
open System.Threading.Tasks
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Telemetry.Tests.Common

/// The manifest of the RunLog contract's vector M1.
let manifestM1 =
    {
        Schema = RunManifest.Schema
        RunId = "20261018-090000-0000002a"
        Kind = "parity"
        Title = "CPU parity"
        Status = RunStatus.Ok
        CreatedUtc = "2026-10-18T09:00:00.000Z"
        EndedUtc = Some "2026-10-18T09:05:00.000Z"
        Node = "n0"
        Synthetic = false
        GitSha = Some "0123abc"
        ExecutionClass = Some x2Class
        Driver = Some "616.92"
        Params = [ "seed", "7"; "model", "Qwen3-0.6B" ]
    }

let manifestM1Text =
    """{
  "schema": "lrm.run/1",
  "runId": "20261018-090000-0000002a",
  "kind": "parity",
  "title": "CPU parity",
  "status": "ok",
  "createdUtc": "2026-10-18T09:00:00.000Z",
  "endedUtc": "2026-10-18T09:05:00.000Z",
  "node": "n0",
  "synthetic": false,
  "gitSha": "0123abc",
  "executionClass": {
    "backend": "vulkan",
    "deviceId": "pci:10de:2704",
    "driver": "616.92",
    "shaderSetHash": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "arch": "win-x64",
    "id": "d79c93db855714890ff41356c0fd4dcb61e048b1fa346c87b7ca140506a3b057"
  },
  "driver": "616.92",
  "params": {
    "model": "Qwen3-0.6B",
    "seed": "7"
  }
}
"""

/// The events file of the RunLog contract's vector R1.
let eventsR1 =
    String.concat
        ""
        [
            """{"v":1,"t":2000,"k":"event","n":"run.node","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"role":"creator"}}"""
            "\n"
            """{"v":1,"t":3000,"k":"counter","n":"model.steps","d":0,"x":1,"u":"count","node":"n0","seq":1,"a":{}}"""
            "\n"
            """{"v":1,"t":5000,"k":"meter","n":"model.flops","d":0,"x":2048,"u":"flop","node":"n0","seq":2,"a":{"$parent":1}}"""
            "\n"
            """{"v":1,"t":4000,"k":"span","n":"model.layer","d":2000,"x":2000,"u":"ns","node":"n0","seq":3,"a":{"$span":1,"layer":0}}"""
            "\n"
            """{"v":1,"t":7000,"k":"event","n":"run.end","d":0,"x":0,"u":"","node":"n0","seq":4,"a":{"status":"ok","records":4,"dropped":0,"invalid":0}}"""
            "\n"
        ]

/// Writes the vector R1 run: a counter, then a span holding a meter, on a clock that steps 1000 ns per read.
let writeR1 (root: string) =
    use run =
        RunContext.Create(
            { RunOptions.create "parity" "t" with
                RunId = Some(RunId.parse "kat-run")
                RunsRoot = Some root
                Clock = Some(steppingClock 1000L 1000L)
                FlushInterval = TimeSpan.Zero
            }
        )

    do
        use _ = run.Install()
        Telemetry.counter "model.steps" 1.0 TelemetryUnits.Count []
        use _ = Telemetry.span "model.layer" [ Attr.int "layer" 0 ]
        Telemetry.meter "model.flops" 2048.0 TelemetryUnits.Flops []

    run.Finish RunStatus.Ok
    run.Directory

let private optionsIn (root: string) (id: string) =
    { RunOptions.create "inference" "test" with
        RunId = Some(RunId.parse id)
        RunsRoot = Some root
        FlushInterval = TimeSpan.Zero
    }

/// A stream whose writes always fail.
type private BrokenStream() =
    inherit Stream()
    override _.CanRead = false
    override _.CanSeek = false
    override _.CanWrite = true
    override _.Length = 0L

    override _.Position
        with get () = 0L
        and set _ = ()

    override _.Flush() = raise (IOException "flush failed")
    override _.Read(_, _, _) = 0
    override _.Seek(_, _) = 0L
    override _.SetLength _ = ()
    override _.Write(_, _, _) = raise (IOException "disk full")

let private eventsPath (dir: string) (node: string) =
    Path.Combine(dir, "events." + node + ".jsonl")

let tests =
    testList
        "runlog"
        [
            testCase "manifest vector M1 renders to its exact text"
            <| fun () ->
                Expect.equal (RunManifest.render manifestM1) manifestM1Text "M1"
                Expect.equal (withCulture "de-DE" (fun () -> RunManifest.render manifestM1)) manifestM1Text "de-DE"

            testCase "manifest text parses back to the same manifest, params sorted"
            <| fun () ->
                match RunManifest.parse manifestM1Text with
                | Ok m ->
                    Expect.equal
                        m
                        { manifestM1 with
                            Params = List.sortBy fst manifestM1.Params
                        }
                        "round trip"
                | Error e -> failtest e

            testCase "manifest parsing refuses another schema and a class id that does not match"
            <| fun () ->
                Expect.isError (RunManifest.parse (manifestM1Text.Replace("lrm.run/1", "lrm.run/2"))) "schema"

                Expect.isError
                    (RunManifest.parse (manifestM1Text.Replace("d79c93db", "d79c93dc")))
                    "tampered executionClass id"

                Expect.isError (RunManifest.parse (manifestM1Text.Replace("\"ok\"", "\"done\""))) "status"
                Expect.isError (RunManifest.parse "[]") "not an object"

            testCase "vector R1: a small run writes exactly the specified events file and manifest"
            <| fun () ->
                use tmp = new TempDir()
                let dir = writeR1 tmp.Path
                Expect.equal (File.ReadAllText(eventsPath dir "n0")) eventsR1 "events.n0.jsonl"

                match RunManifest.read dir with
                | Ok m ->
                    Expect.equal m.Status RunStatus.Ok "status"
                    Expect.equal m.CreatedUtc "1970-01-01T00:00:00.000Z" "created"
                    Expect.equal m.EndedUtc (Some "1970-01-01T00:00:00.000Z") "ended"
                    Expect.equal m.Kind "parity" "kind"
                | Error e -> failtest e

            testCase "the reader returns typed data records and node summaries"
            <| fun () ->
                use tmp = new TempDir()
                let dir = writeR1 tmp.Path
                let log = RunLogReader.loadOrFail dir
                Expect.isEmpty log.Problems "no problems"

                Expect.equal
                    (log.Events |> List.map _.Name)
                    [ "model.steps"; "model.flops"; "model.layer" ]
                    "data records"

                let span = log.Events |> List.find (fun e -> e.Kind = TelemetryKind.Span)
                Expect.equal (span.SpanId, span.DurationNs, span.TryInt "layer") (1L, 2000L, Some 0L) "span"
                Expect.equal (log.Named "model.flops").Head.ParentId 1L "parent"

                match log.Nodes with
                | [ n ] ->
                    Expect.equal
                        (n.Node, n.Role, n.Records, n.Ended, n.Status)
                        ("n0", "creator", 3, true, Some RunStatus.Ok)
                        "node"
                | _ -> failtest "one node"

                Expect.equal (RunLogReader.stream dir |> List.ofSeq) log.Events "stream equals load"

            testCase "the manifest and node header record the execution class and driver"
            <| fun () ->
                use tmp = new TempDir()

                use run =
                    RunContext.Create(
                        { optionsIn tmp.Path "xclass" with
                            ExecutionClass = Some x2Class
                        }
                    )

                run.Finish RunStatus.Ok
                let log = RunLogReader.loadOrFail run.Directory
                Expect.equal log.Manifest.ExecutionClass (Some x2Class) "manifest class"
                Expect.equal log.Manifest.Driver (Some "616.92") "manifest driver"
                Expect.equal (log.ClassOf "n0") (Some x2Class) "header class"
                let text = File.ReadAllText(Path.Combine(run.Directory, "manifest.json"))
                Expect.stringContains text (ExecutionClass.id x2Class).Value.Hex "class id in the manifest"

            testCase "a joined node writes its own file and class; the creator keeps the manifest"
            <| fun () ->
                use tmp = new TempDir()
                use run = RunContext.Create(optionsIn tmp.Path "joined")

                use other =
                    RunContext.Join(run.Directory, NodeId.parse "n1", Some x2Class, Some(steppingClock 0L 1L))

                Telemetry.counter "model.steps" 1.0 "count" [] // the ambient facade is still the no-op
                other.Telemetry.Counter("model.steps", 2.0, "count", [])
                other.Finish RunStatus.Ok
                run.Finish RunStatus.Ok
                let log = RunLogReader.loadOrFail run.Directory

                Expect.equal
                    (log.Nodes |> List.map (fun n -> n.Node, n.Role))
                    [ "n0", "creator"; "n1", "joined" ]
                    "nodes"

                Expect.equal (log.ClassOf "n1") (Some x2Class) "joined class"
                Expect.equal (log.Events |> List.map (fun e -> e.Node, e.Value)) [ "n1", 2.0 ] "records"

                Expect.throwsT<ArgumentException>
                    (fun () -> RunContext.Join(run.Directory, NodeId.parse "n0", None, None) |> ignore)
                    "the creator's node id"

            testCase "run folders and event files are never overwritten"
            <| fun () ->
                use tmp = new TempDir()
                use run = RunContext.Create(optionsIn tmp.Path "twice")

                Expect.throwsT<InvalidOperationException>
                    (fun () -> RunContext.Create(optionsIn tmp.Path "twice") |> ignore)
                    "folder"

                Expect.throws
                    (fun () -> new JsonlSink(eventsPath run.Directory "n0", NodeId.local) |> ignore)
                    "events file"

                run.Finish RunStatus.Ok

            testCase "a run disposed without Finish is recorded as aborted"
            <| fun () ->
                use tmp = new TempDir()

                let dir =
                    use run = RunContext.Create(optionsIn tmp.Path "aborted")
                    run.Directory

                match RunManifest.read dir with
                | Ok m -> Expect.equal m.Status RunStatus.Aborted "aborted"
                | Error e -> failtest e

            testCase "the sink is thread-safe: 8 writers, 16,000 records, gapless seq"
            <| fun () ->
                use tmp = new TempDir()
                use run = RunContext.Create(optionsIn tmp.Path "threads")

                do
                    use _ = run.Install()

                    Parallel.For(
                        0,
                        8,
                        fun w ->
                            for i in 1..2000 do
                                Telemetry.counter "model.steps" 1.0 "count" [ Attr.int "writer" w; Attr.int "i" i ]
                    )
                    |> ignore

                run.Finish RunStatus.Ok
                let log = RunLogReader.loadOrFail run.Directory
                Expect.isEmpty log.Problems "gapless, every line valid"
                Expect.equal log.Events.Length 16000 "records"
                Expect.equal (log.Events |> List.map _.Seq) [ 1L .. 16000L ] "seq in file order"

            testCase "the sink never throws: invalid names, writes after close and I/O errors are counted"
            <| fun () ->
                use tmp = new TempDir()
                use run = RunContext.Create(optionsIn tmp.Path "faults")
                run.Telemetry.Counter("Bad Name", 1.0, "count", [])
                run.Telemetry.Counter("ok.name", 1.0, "bad unit", [])
                run.Finish RunStatus.Ok
                run.Telemetry.Counter("late.record", 1.0, "count", [])
                Expect.equal run.Sink.Invalid 2L "invalid"
                Expect.equal run.Sink.Dropped 1L "dropped after close"
                let log = RunLogReader.loadOrFail run.Directory
                Expect.isTrue (log.Problems |> List.exists (fun p -> p.Contains "invalid names")) "reported"

                let broken = JsonlSink.OfStream(new BrokenStream(), NodeId.local)
                (broken :> ITelemetrySink).Write(record TelemetryKind.Event "e")
                broken.Flush()
                broken.Close()
                Expect.equal (broken.Dropped, broken.Records) (1L, 0L) "I/O error dropped, not thrown"

            testCase "the reader reports a cut-off line, a seq gap, a bad line and a missing trailer"
            <| fun () ->
                use tmp = new TempDir()
                let dir = writeR1 tmp.Path
                let path = eventsPath dir "n0"

                let lines =
                    File.ReadAllText(path).Split('\n', StringSplitOptions.RemoveEmptyEntries)
                // Drop the meter (seq 2) and the trailer, add a malformed line and a cut-off one.
                let damaged = [ lines[0]; lines[1]; lines[3]; "{not json" ] |> String.concat "\n"
                File.WriteAllText(path, damaged + "\n" + lines[4].Substring(0, 20), UTF8Encoding(false))
                let log = RunLogReader.loadOrFail dir
                let problems = String.concat "\n" log.Problems
                Expect.stringContains problems "seq 3 where 2 was expected" "gap"
                Expect.stringContains problems "malformed JSON" "bad line"
                Expect.stringContains problems "cut off" "cut-off line"
                Expect.stringContains problems "no run.end trailer" "missing trailer"
                Expect.equal (log.Events |> List.map _.Name) [ "model.steps"; "model.layer" ] "the good records survive"

            testCase "loading needs a valid manifest"
            <| fun () ->
                use tmp = new TempDir()
                Expect.isError (RunLogReader.load tmp.Path) "no manifest"
                File.WriteAllText(Path.Combine(tmp.Path, "manifest.json"), "{}")
                Expect.isError (RunLogReader.load tmp.Path) "invalid manifest"

            testCase "events files map to nodes"
            <| fun () ->
                Expect.equal (RunLogReader.nodeOfFile "events.n0.jsonl") (Some "n0") "n0"
                Expect.equal (RunLogReader.nodeOfFile "events.coord.jsonl") (Some "coord") "coord"
                Expect.equal (RunLogReader.nodeOfFile "events..jsonl") None "empty node"
                Expect.equal (RunLogReader.nodeOfFile "manifest.json") None "manifest"
        ]
