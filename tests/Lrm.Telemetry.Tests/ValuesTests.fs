/// Acceptance: telemetry on or off changes no value (TelemetryFacade rule 1, RunLog contract).
module Lrm.Telemetry.Tests.ValuesTests

open System
open Expecto
open Lrm.Core
open Lrm.Telemetry
open Lrm.Telemetry.Tests.Common

/// A small instrumented forward-like computation: spans per layer, metering through the FLOP registry, arena gauges
/// and a check. Its result must not depend on telemetry in any way.
let private compute () =
    let rng = DetRng(3UL, DetRng.streamOf "values-test")
    let x = Array.init 512 (fun _ -> rng.NextF32() * 2.0f - 1.0f)
    let w = Array.init 512 (fun _ -> Bf16.widen (Bf16.narrowRne (rng.NextF32())))

    use _ =
        Telemetry.scope [ Attr.str TelemetryKeys.Component "values"; Attr.int TelemetryKeys.Stage 0 ]

    use fwd = Telemetry.span "model.forward" [ Attr.str "phase" "decode" ]
    let mutable acc = 0.0f

    for layer in 0..5 do
        use _ = Telemetry.span "model.layer" [ Attr.int TelemetryKeys.Layer layer ]
        let mutable ss = 0.0f

        for i in 0 .. x.Length - 1 do
            ss <- ss + x[i] * x[i]

        let r = 1.0f / MathF.Sqrt(ss / float32 x.Length + 1e-6f)

        for i in 0 .. x.Length - 1 do
            x[i] <- x[i] * r * w[i] + 0.01f * float32 layer

        acc <- acc + x[layer]
        Metering.op "model" (Op.RmsNorm(1L, int64 x.Length, DType.BF16)) [ Attr.int TelemetryKeys.Layer layer ]
        Metering.arena "activations" (int64 (x.Length * 4)) []

    fwd.Add(Attr.float "acc" (float acc))
    Metering.tokens "model.tokens" 1.0 []

    let id =
        Canonical.hash "lrm.test/1" (fun cw ->
            cw.Seq(x, cw.F32)
            cw.F32 acc)

    Telemetry.check TelemetryNames.GoldenCompare true [ Attr.str "id" id.Short ]
    id

let tests =
    testList
        "values"
        [
            testCase "telemetry on or off changes no value"
            <| fun () ->
                let off = compute ()

                let inMemory =
                    let sink = InMemorySink()
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    let id = compute ()
                    Expect.isGreaterThan sink.Count 10 "records were made"
                    id

                use tmp = new TempDir()

                let runLog =
                    use run =
                        RunContext.Create(
                            { RunOptions.create "inference" "values" with
                                RunsRoot = Some tmp.Path
                                RunId = Some(RunId.parse "values")
                            }
                        )

                    let id =
                        use _ = run.Install()
                        compute ()

                    run.Finish RunStatus.Ok
                    id

                let virtualClock =
                    use run =
                        RunContext.Create(
                            { RunOptions.create "inference" "values" with
                                RunsRoot = Some tmp.Path
                                RunId = Some(RunId.parse "values-virtual")
                                Clock = Some(steppingClock 0L 7L)
                            }
                        )

                    let id =
                        use _ = run.Install()
                        compute ()

                    run.Finish RunStatus.Ok
                    id

                Expect.equal inMemory off "in-memory sink"
                Expect.equal runLog off "run-log sink"
                Expect.equal virtualClock off "run-log sink on a virtual clock"

                let log = RunLogReader.loadOrFail (System.IO.Path.Combine(tmp.Path, "values"))
                Expect.isEmpty log.Problems "the run log is clean"
                Expect.equal (log.Named "model.layer").Length 6 "layer spans"

            testCase "the same records on a virtual clock give byte-identical files"
            <| fun () ->
                use tmp = new TempDir()

                let write (id: string) =
                    use run =
                        RunContext.Create(
                            { RunOptions.create "inference" "values" with
                                RunsRoot = Some tmp.Path
                                RunId = Some(RunId.parse id)
                                Clock = Some(steppingClock 1_000L 1_000L)
                                FlushInterval = TimeSpan.Zero
                            }
                        )

                    do
                        use _ = run.Install()
                        compute () |> ignore

                    run.Finish RunStatus.Ok
                    System.IO.File.ReadAllBytes(System.IO.Path.Combine(run.Directory, "events.n0.jsonl"))

                let a = write "a"
                let b = withCulture "de-DE" (fun () -> write "b")
                Expect.equal (Lrm.Testing.Determinism.firstDifference a b) None "identical bytes under de-DE"
        ]
