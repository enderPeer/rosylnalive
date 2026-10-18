/// M2a's acceptance on the tiny model (dev plan, M2a row): the Qwen forward on CpuOps, from the ids, against O1's
/// goldens. CpuOps F64 against torch64p within 1e-12 on every tap; CpuOps F32 against torch32 and torch64p within the
/// CPU32 floors (relL2 1e-5 per tap, max |Δlogit| 2e-3, KL 1e-7); 16- and 64-step greedy decoding identical to both
/// oracles. Every check is a golden.compare record, which also goes to the parity run log (ParityRun).
module Lrm.Models.Qwen.Tests.ForwardTests

open System
open System.Collections.Generic
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

/// The tiny-model bound of cpu64-vs-torch64p (ToleranceSpec: fixed, 1e-12 on the tiny model).
let F64Bound = 1e-12

/// The CPU32 floors (ToleranceSpec), which bound the float32 forward on the tiny model.
let Cpu32RelL2 = 1e-5
let Cpu32LogitMaxAbs = 2e-3
let Cpu32LogitKl = 1e-7

/// The backend a CPU pair measures: cpu-f64 for cpu64-vs-torch64p, else cpu-f32.
let backendOfPair (pair: TolerancePair) =
    match pair with
    | TolerancePair.Cpu64VsTorch64p -> ExecutionBackend.name ExecutionBackend.CpuF64
    | _ -> ExecutionBackend.name ExecutionBackend.CpuF32

/// The golden.compare attributes of one comparison: the parity and gate1 templates' keys (pair, backend, prompt,
/// tap, layer, kind).
let attrs (pair: TolerancePair) (prompt: string) (tapName: string) =
    [
        Attr.str "pair" (TolerancePair.name pair)
        Attr.str "backend" (backendOfPair pair)
        Attr.str "prompt" prompt
        Attr.str "tap" tapName
        Attr.int TelemetryKeys.Layer (ActivationNames.layer tapName)
        Attr.str "kind" "chain"
    ]

/// A prefill of `ids` with every tap: the logits [T, vocab] and the taps by name, on the host.
let prefillTaps (ops: IForwardOps<'t>) (ids: int[]) =
    let model = tinyModel ()
    use cache = KvCache.create ops model (Stage.whole model.Config) ids.Length
    let store = TapStore<'t>(TapFilter.All)
    let logits = Forward.run ops model cache 0 ids (ForwardOptions.withTaps store)
    let taps = Dictionary<string, IHostTensor>(StringComparer.Ordinal)

    for name, x in store.Taps do
        taps[name] <- ops.Read x

    ops.Read logits, taps

let private failures (checks: CheckResult list) =
    checks |> List.filter (fun r -> not r.Passed)

let private tapOf (r: CheckResult) =
    r.Attrs
    |> List.tryPick (fun a ->
        match a.Value with
        | AttrValue.Str s when a.Key = "tap" -> Some s
        | _ -> None)
    |> Option.defaultValue "?"

/// Prints the worst measurement of each metric in a block of checks, then fails on the first one that did not pass.
let report (label: string) (checks: CheckResult list) =
    let worst =
        checks
        |> List.groupBy _.Metric
        |> List.map (fun (metric, xs) ->
            let w = xs |> List.maxBy _.Measured
            Invariant.format $"{metric} {Invariant.f64 w.Measured} at {tapOf w} (bound {Invariant.f64 w.Bound})")

    printfn "%s: %d checks; worst %s" label checks.Length (String.concat "; " worst)

    match failures checks with
    | [] -> ()
    | r :: _ -> failwithf "%s: %s %O" label (tapOf r) r

/// relL2 of every tap in `names` against the golden file `oracleFile.prompt`.
let chainChecks
    (taps: Dictionary<string, IHostTensor>)
    (oracleFile: string)
    (prompt: string)
    (pair: TolerancePair)
    (bound: float)
    (names: string[])
    =
    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"{oracleFile}.{prompt}.safetensors")

    [
        for name in names -> Compare.relL2 bound (attrs pair prompt name) taps[name] (f.Tensor name)
    ]

/// The logits' max |Δ| and KL against a golden file.
let logitChecks (logits: IHostTensor) (oracleFile: string) (prompt: string) (pair: TolerancePair) =
    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"{oracleFile}.{prompt}.safetensors")

    let golden = f.Tensor "logits"
    let a = attrs pair prompt "logits"

    [
        Compare.maxAbs Cpu32LogitMaxAbs a logits golden
        Compare.kl Cpu32LogitKl a logits golden
    ]

/// A greedy_match golden.compare check (ReportTemplates, parity; GoldenCheck.greedy): the steps that match the
/// golden before the first difference, against the steps generated; it passes only when every step matches.
let greedyCheck (pair: TolerancePair) (prompt: string) (got: int[]) (golden: int[]) =
    GoldenCheck.greedy
        [
            Attr.str "pair" (TolerancePair.name pair)
            Attr.str "backend" (backendOfPair pair)
            Attr.str "prompt" prompt
            Attr.str "tap" "greedy.ids"
            Attr.int TelemetryKeys.Layer -1
            Attr.str "kind" "chain"
        ]
        got
        golden

/// The greedy margin record the parity template charts (`goldens.greedy.min_margin`, ReportTemplates), with the
/// backend as the `oracle`.
let marginGauge (prompt: string) (backend: string) (margins: float[]) =
    Telemetry.gauge
        "goldens.greedy.min_margin"
        (Array.min margins)
        TelemetryUnits.NoUnit
        [
            Attr.str "prompt" prompt
            Attr.str "oracle" backend
            Attr.int "steps" margins.Length
        ]

let private greedyCase
    (backend: ExecutionBackend)
    (ops: unit -> IForwardOps<'t>)
    (pair: TolerancePair)
    (oracle: string)
    (prompt: string)
    =
    use _ = ParityRun.install backend
    let golden = goldenGreedy prompt oracle
    let ids = promptIds prompt
    use ops = ops ()
    let model = tinyModel ()

    let r64 =
        Greedy.run
            ops
            model
            ids
            { GreedyOptions.parity 64 with
                Margins = true
            }

    let r16 = Greedy.run ops model ids (GreedyOptions.parity 16)
    greedyCheck pair prompt r64.Ids golden |> ignore
    // 16 steps against the 64-step golden: its first 16 ids.
    Expect.isTrue (greedyCheck pair prompt r16.Ids golden).Passed "16 steps against the 64-id golden"
    marginGauge prompt (ExecutionBackend.name backend) r64.Margins
    printfn "%s greedy %s: min margin %s" (ExecutionBackend.name backend) prompt (Invariant.f64 (Array.min r64.Margins))
    Expect.sequenceEqual r64.Ids golden (sprintf "64 steps = %s" oracle)
    Expect.sequenceEqual r16.Ids golden[0..15] (sprintf "16 steps = the first 16 of %s" oracle)
    Expect.isFalse r64.StoppedAtEos "EOS ignored"

let tests =
    testList
        "tiny"
        [
            testList
                "chain"
                [
                    for prompt in [ "p0"; "p1" ] do
                        testCase (Invariant.format $"cpu-f64 vs torch64p, every tap <= 1e-12 ({prompt})")
                        <| fun () ->
                            use _ = ParityRun.install ExecutionBackend.CpuF64
                            let _, taps = prefillTaps (tinyF64 ()) (promptIds prompt)
                            Expect.equal taps.Count 39 "39 taps"

                            chainChecks
                                taps
                                "fwd_f64p"
                                prompt
                                TolerancePair.Cpu64VsTorch64p
                                F64Bound
                                (ActivationNames.all 2)
                            |> report (Invariant.format $"cpu-f64 vs torch64p, {prompt}")

                        testCase (
                            Invariant.format $"cpu-f32 vs torch32 and torch64p within the CPU32 floors ({prompt})"
                        )
                        <| fun () ->
                            use _ = ParityRun.install ExecutionBackend.CpuF32
                            let logits, taps = prefillTaps (tinyF32 ()) (promptIds prompt)
                            let names = ActivationNames.all 2

                            chainChecks taps "fwd_f32" prompt TolerancePair.Cpu32VsTorch32 Cpu32RelL2 names
                            @ logitChecks logits "fwd_f32" prompt TolerancePair.Cpu32VsTorch32
                            |> report (Invariant.format $"cpu-f32 vs torch32, {prompt}")

                            chainChecks taps "fwd_f64p" prompt TolerancePair.Cpu32VsTorch64p Cpu32RelL2 names
                            @ logitChecks logits "fwd_f64p" prompt TolerancePair.Cpu32VsTorch64p
                            |> report (Invariant.format $"cpu-f32 vs torch64p, {prompt}")

                    testCase "p2 (70 tokens, across a 64-key block): logits of both modes"
                    <| fun () ->
                        let ids = promptIds "p2"

                        do
                            use _ = ParityRun.install ExecutionBackend.CpuF64
                            let logits, _ = prefillTaps (tinyF64 ()) ids
                            let t = Dictionary<string, IHostTensor>(dict [ "logits", logits ])

                            chainChecks t "fwd_f64p" "p2" TolerancePair.Cpu64VsTorch64p F64Bound [| "logits" |]
                            |> report "cpu-f64 vs torch64p, p2"

                        do
                            use _ = ParityRun.install ExecutionBackend.CpuF32
                            let logits, _ = prefillTaps (tinyF32 ()) ids
                            let t = Dictionary<string, IHostTensor>(dict [ "logits", logits ])

                            chainChecks t "fwd_f32" "p2" TolerancePair.Cpu32VsTorch32 Cpu32RelL2 [| "logits" |]
                            @ logitChecks logits "fwd_f32" "p2" TolerancePair.Cpu32VsTorch32
                            |> report "cpu-f32 vs torch32, p2"

                    testCase "the taps come in ActivationNames order, [T, features], in the mode's dtype"
                    <| fun () ->
                        let ids = promptIds "p0"
                        use ops = tinyF32 ()
                        let model = tinyModel ()
                        use cache = KvCache.create ops model (Stage.whole model.Config) ids.Length
                        let store = TapStore(TapFilter.All)
                        Forward.run ops model cache 0 ids (ForwardOptions.withTaps store) |> ignore
                        Expect.sequenceEqual (store.Taps |> List.map fst) (ActivationNames.all 2) "order"

                        use f = tiny.Force().OpenSafetensors "fwd_f32.p0.safetensors"

                        for name, x in store.Taps do
                            Expect.equal (ops.Read x).Desc.Shape (f.Entry name).Desc.Shape name

                        let _, taps64 = prefillTaps (tinyF64 ()) ids
                        Expect.equal taps64["L1.attn.k_rope"].Desc.DType DType.F64 "F64 taps"

                    testCase "a tap filter records only the named taps"
                    <| fun () ->
                        use ops = tinyF32 ()
                        let model = tinyModel ()
                        use cache = KvCache.create ops model (Stage.whole model.Config) 13
                        let store = TapStore(TapFilter.Parse "L1.attn.q_rope, logits")

                        Forward.run ops model cache 0 (promptIds "p0") (ForwardOptions.withTaps store)
                        |> ignore

                        Expect.equal (store.Taps |> List.map fst) [ "L1.attn.q_rope"; "logits" ] "names"
                        Expect.equal (TapFilter.Parse "none") TapFilter.Off "none"
                        Expect.equal (TapFilter.Parse "all") TapFilter.All "all"
                ]

            testList
                "greedy"
                [
                    for prompt in [ "p0"; "p1" ] do
                        testCase (Invariant.format $"cpu-f64: 16 and 64 greedy steps equal torch64p ({prompt})")
                        <| fun () ->
                            greedyCase ExecutionBackend.CpuF64 tinyF64 TolerancePair.Cpu64VsTorch64p "torch64p" prompt

                        testCase (Invariant.format $"cpu-f32: 16 and 64 greedy steps equal torch32 ({prompt})")
                        <| fun () ->
                            greedyCase ExecutionBackend.CpuF32 tinyF32 TolerancePair.Cpu32VsTorch32 "torch32" prompt

                    testCase "StopAtEos stops p0 at its first EOS (token 7, step 9); parity runs ignore it"
                    <| fun () ->
                        let golden = goldenGreedy "p0" "torch32"
                        let ids = promptIds "p0"

                        for ops in
                            [
                                (fun () -> QwenRunner.ofOps "cpu-f32" (tinyModel ()) (tinyF32 ()))
                                (fun () -> QwenRunner.ofOps "cpu-f64" (tinyModel ()) (tinyF64 ()))
                            ] do
                            use runner = ops ()

                            let r =
                                runner.Greedy(
                                    ids,
                                    { GreedyOptions.parity 64 with
                                        StopAtEos = true
                                    }
                                )

                            Expect.isTrue r.StoppedAtEos "stopped"
                            Expect.equal r.Ids.Length 10 "10 ids, the last one EOS"
                            Expect.equal r.Ids[9] 7 "EOS"
                            Expect.sequenceEqual r.Ids golden[0..9] "the golden prefix"

                    testCase "the runner and both registry factories give the generic forward's bits"
                    <| fun () ->
                        let ids = promptIds "p0"
                        let w = tinyWeights.Value

                        for f in CpuBackends.all do
                            use runner = f.Create w (tinyOptions ())
                            let r = runner.Forward(ids, TapFilter.All, HeadRows.All)
                            Expect.equal runner.Backend f.Name "name"

                            let logits, taps =
                                if f.Name = "cpu-f32" then
                                    prefillTaps (tinyF32 ()) ids
                                else
                                    prefillTaps (tinyF64 ()) ids

                            expectSameBits (f.Name + " logits") r.Logits logits

                            for name, t in r.Taps do
                                expectSameBits (f.Name + " " + name) t taps[name]

                            Expect.equal
                                runner.ExecutionClass.Backend
                                (if f.Name = "cpu-f32" then
                                     ExecutionBackend.CpuF32
                                 else
                                     ExecutionBackend.CpuF64)
                                "class"
                ]
        ]
