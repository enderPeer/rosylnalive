/// M2b's acceptance, gate1-cpu (dev plan, M2b row; ForwardOpsAndQwenModel, "Full-model parity"): the Qwen forward on
/// CpuOps, from the ids, against G2's full-model goldens (goldens/qwen3-0.6b/g1) and tolerances. Model tier, local only.
///
/// Per gate prompt and CPU class (QwenParity.gatePrompt): the prompt rows of one teacher-forced prefill against the
/// prefill goldens (every tap of the two taps prompts, the logits of every prompt), the teacher-forced argmax, and 64
/// greedy steps on the canonical schedule against each oracle's ids. Per long prompt (QwenParity.longPrompt):
/// final_norm.out, the stored logits rows and the teacher-forced argmax. cpu-f64 is bounded by 1e-10 against torch64p;
/// cpu-f32 by the calibrated bounds against torch32 and torch64p. Self-tests: T_spec equals the goldens' table, and a
/// prefill equals the prefix rows of a longer prefill in other chunks, bit for bit. Every check is a golden.compare
/// record in the full-model parity run (ParityRun, key qwen3-0.6b); the last test records the gate1.check summary.
module Lrm.Models.Qwen.Tests.GateCpuTests

open System
open System.Collections.Generic
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

/// The CPU classes of gate1-cpu, in the order their tests run: cpu-f64 (node n0), then cpu-f32 (node n1).
let backends = [ ExecutionBackend.CpuF64; ExecutionBackend.CpuF32 ]

/// The rows of the self-test's reference prefill: small chunks, so it also proves chunked prefill.
[<Literal>]
let SelfTestChunk = 16

/// The full-model parity run (ParityRun).
let runSpec =
    lazy
        (let g = qwenGoldens.Value
         let tolerances = (g.Set.File "tolerances.json").Sha256.Hex

         let spec: ParityRun.RunSpec =
             {
                 Key = "qwen3-0.6b"
                 Title =
                     "Qwen3-0.6B gate1-cpu: the Qwen forward (M2b) on cpu-f64 and cpu-f32 against goldens/qwen3-0.6b/g1"
                 Params =
                     [
                         "golden", "goldens/qwen3-0.6b/g1"
                         "model", "LRM_DATA/models/Qwen3-0.6B at " + Qwen3Pins.qwen3_0_6b.Revision
                         "tolerances", "tolerances.json, SHA-256 " + tolerances
                         "source", "tests/Lrm.Models.Qwen.Tests (M2b gate1-cpu, model tier)"
                         "nodes", "n0 cpu-f64, n1 cpu-f32"
                         "threads", Invariant.i32 Threads
                         "chunk", Invariant.i32 ParityOptions.DefaultChunk + " rows per forward"
                         "prompts", Invariant.format $"{g.Gate.Length} gate, {g.Long.Length} long"
                         "op_spans", "cpuref.* spans left out (ParityRun.RunSpec.DropOpSpans)"
                     ]
                 DropOpSpans = true
             }

         spec)

let private install (backend: ExecutionBackend) =
    ParityRun.installFor runSpec.Value backend

/// The results of this process, for the summary: per (backend, prompt) the prompt's parity, and the self-tests.
let private results = Dictionary<string * string, PromptParity>()
let private selfTests = List<string * string * bool>()
let private gate = obj ()

let private recordResult (r: PromptParity) =
    lock gate (fun () -> results[(r.Backend, r.Prompt.Name)] <- { r with PromptRows = [] })

let private recordSelfTest (backend: string) (check: string) (passed: bool) =
    lock gate (fun () -> selfTests.Add(backend, check, passed))

let private tapOf (c: CheckResult) =
    c.Attrs
    |> List.tryPick (fun a ->
        match a.Key, a.Value with
        | "tap", AttrValue.Str s -> Some s
        | _ -> None)
    |> Option.defaultValue "?"

let private pairOf (c: CheckResult) =
    c.Attrs
    |> List.tryPick (fun a ->
        match a.Key, a.Value with
        | "pair", AttrValue.Str s -> Some s
        | _ -> None)
    |> Option.defaultValue "?"

/// measured / bound of an at-most check (NaN for greedy_match and for a bound of 0 with a pass).
let private ratio (c: CheckResult) =
    if c.Op = BoundOp.AtMost && c.Bound > 0.0 then
        c.Measured / c.Bound
    else
        nan

let private worst (checks: CheckResult list) =
    checks
    |> List.filter (fun c -> Double.IsFinite(ratio c))
    |> List.sortByDescending ratio
    |> List.tryHead

let private describe (c: CheckResult) =
    Invariant.format
        $"{pairOf c} {tapOf c} {c.Metric} {Invariant.f64 c.Measured} (bound {Invariant.f64 c.Bound}, ratio {Invariant.f64 (ratio c)})"

/// Fails with the first failing checks of a prompt's parity.
let private expectPassed (r: PromptParity) =
    match r.Failures with
    | [] -> ()
    | fails ->
        let shown =
            fails
            |> List.truncate 5
            |> List.map (fun c ->
                let detail =
                    c.Attrs
                    |> List.tryPick (fun a ->
                        match a.Key, a.Value with
                        | "detail", AttrValue.Str s -> Some(" (" + s + ")")
                        | _ -> None)
                    |> Option.defaultValue ""

                describe c + detail)

        failwithf
            "%s %s: %d of %d checks failed: %s"
            r.Backend
            r.Prompt.Name
            fails.Length
            r.Checks.Length
            (String.Join("; ", shown))

let private printResult (r: PromptParity) =
    let worstText =
        match worst r.Checks with
        | Some c -> describe c
        | None -> "n/a"

    let greedy =
        match r.Greedy with
        | Some g -> Invariant.format $", min margin {Invariant.f64 (Array.min g.Margins)}"
        | None -> ""

    printfn
        "%s"
        (Invariant.format
            $"{r.Backend} {r.Prompt.Name} ({r.Prompt.Ids.Length} tokens, {r.Rows} rows): prefill {r.PrefillSeconds:F1} s, greedy {r.GreedySeconds:F1} s, total {r.TotalSeconds:F1} s; {r.Checks.Length} checks, {r.Failures.Length} failed; worst {worstText}{greedy}")

/// Every row of a prefill of `ids` in chunks of `chunk` rows, with every tap: (name, [T, features]) in forward order.
let private allRows (ops: IForwardOps<'t>) (model: Qwen3Model) (ids: int[]) (chunk: int) =
    let blocks = Dictionary<string, List<IHostTensor>>(StringComparer.Ordinal)
    let order = List<string>()

    QwenParity.prefill ops model ids chunk TapFilter.All (fun _ _ taps ->
        for name, x in taps do
            if not (blocks.ContainsKey name) then
                blocks[name] <- List()
                order.Add name

            blocks[name].Add(ops.Read x))

    [
        for name in order ->
            let parts = blocks[name]
            let cols = int parts[0].Desc.Shape.Cols
            let rows = parts |> Seq.sumBy (fun b -> int b.Desc.Shape.Rows)
            let bytes = parts |> Seq.collect (fun b -> b.ToBytes()) |> Array.ofSeq

            let t =
                match parts[0].Desc.DType with
                | DType.F32 ->
                    HostTensor.ofBytes<float32> DType.F32 (Shape.matrix rows cols) (ReadOnlySpan bytes) :> IHostTensor
                | _ -> HostTensor.ofBytes<float> DType.F64 (Shape.matrix rows cols) (ReadOnlySpan bytes) :> IHostTensor

            name, t
    ]

/// The elements whose bits differ between two tensors of one dtype and shape (all of them when the shapes differ).
let private bitDiffs (a: IHostTensor) (b: IHostTensor) =
    if a.Desc.DType <> b.Desc.DType || not (a.Desc.Shape.Equals b.Desc.Shape) then
        max a.Desc.Shape.Elements b.Desc.Shape.Elements
    else
        let size = a.Desc.ElementSize
        let x = a.ToBytes()
        let y = b.ToBytes()
        let mutable n = 0L

        for i in 0..size .. x.Length - 1 do
            if not (ReadOnlySpan(x, i, size).SequenceEqual(ReadOnlySpan(y, i, size))) then
                n <- n + 1L

        n

/// The bit-identity self-test of a taps prompt: its own prefill, in chunks of SelfTestChunk rows, equals the prompt
/// rows of the teacher-forced prefill (prompt + 63 rows, one forward), every tap and the logits.
let private prefixSelfTest
    (backend: string)
    (ops: IForwardOps<'t>)
    (model: Qwen3Model)
    (p: GoldenPrompt)
    (promptRows: (string * IHostTensor) list)
    =
    let mine = allRows ops model p.Ids SelfTestChunk
    let names = promptRows |> List.map fst

    if (mine |> List.map fst) <> names then
        failwithf "%s %s: the prefill recorded other taps than the teacher-forced prefill" backend p.Name

    let diffs =
        List.zip mine promptRows |> List.sumBy (fun ((_, a), (_, b)) -> bitDiffs a b)

    let elements = mine |> List.sumBy (fun (_, t) -> t.Desc.Shape.Elements)

    let check =
        Invariant.format
            $"{backend} {p.Name}: a {p.Ids.Length}-row prefill in chunks of {SelfTestChunk} = the first {p.Ids.Length} rows of the {p.Ids.Length + qwenGoldens.Value.Steps - 1}-row prefill ({names.Length} tensors)"

    let r =
        Compare.check
            "exact"
            BoundOp.AtMost
            (float diffs)
            0.0
            elements
            [
                Attr.str "check" check
                Attr.str TelemetryKeys.Backend backend
                Attr.str "prompt" p.Name
                Attr.str "kind" "chain"
            ]

    recordSelfTest backend check r.Passed
    printfn "self-test %s: %d of %d elements differ" check diffs elements
    Expect.equal diffs 0L check

let private promptCase (backend: ExecutionBackend) (p: GoldenPrompt) =
    let name = ExecutionBackend.name backend
    use _ = install backend
    let w = qwenWeights.Value
    let g = qwenGoldens.Value

    let o =
        { ParityOptions.forBackend backend with
            KeepPromptRows = p.Taps
        }

    let plan = QwenParity.plan Threads w.Config g p o

    let run (ops: IForwardOps<'t>) =
        let r = QwenParity.run ops w.Model g p o
        recordResult r
        printResult r

        if p.Taps then
            prefixSelfTest name ops w.Model p r.PromptRows

        expectPassed r

    match backend with
    | ExecutionBackend.CpuF64 ->
        use ops = CpuOps.f64 w plan
        run ops
    | _ ->
        use ops = CpuOps.f32 w plan
        run ops

/// T_spec: CpuOps' RoPE table for positions 0..8191 and the pinned inv_freq bits equal the goldens' rope.safetensors,
/// bit for bit (ReferenceOracles, T_spec).
let private ropeSelfTest () =
    use _ = install ExecutionBackend.CpuF64
    let w = qwenWeights.Value
    let g = qwenGoldens.Value
    use f = g.Set.OpenSafetensors "rope.safetensors"
    let golden = (f.ReadU32 "inv_freq.bits").Data
    let cos = (f.ReadF32 "t_spec.cos").Data
    let sin = (f.ReadF32 "t_spec.sin").Data
    let positions = (f.Entry "t_spec.cos").Desc.Shape.Rows |> int
    let table = RopeConstants.table (CpuExec.WithThreads Threads) w.Config positions
    let bits = RopeConstants.forConfig w.Config

    let differ (a: float32[]) (b: float32[]) =
        if a.Length <> b.Length then
            int64 (max a.Length b.Length)
        else
            Array.map2 (fun (x: float32) y -> BitConverter.SingleToInt32Bits x <> BitConverter.SingleToInt32Bits y) a b
            |> Array.filter id
            |> Array.length
            |> int64

    let diffs =
        differ table.Cos cos
        + differ table.Sin sin
        + (if bits = golden then
               0L
           else
               int64 (max bits.Length golden.Length))

    let check =
        Invariant.format $"T_spec: inv_freq bits and the RoPE table for positions 0..{positions - 1} = rope.safetensors"

    let r =
        Compare.check
            "exact"
            BoundOp.AtMost
            (float diffs)
            0.0
            (int64 (cos.Length + sin.Length + golden.Length))
            [ Attr.str "check" check; Attr.str "kind" "chain" ]

    recordSelfTest "" check r.Passed
    Expect.equal diffs 0L check
    Expect.equal positions 8192 "T_spec covers 0..8191"

/// One gate1.check of the summary (ReportTemplates, gate1: group cpuref).
let private gateCheck (backend: string) (check: string) (passed: bool) (detail: string) =
    Telemetry.check
        "gate1.check"
        passed
        [
            Attr.str "group" "cpuref"
            Attr.str "check" check
            Attr.str TelemetryKeys.Backend backend
            Attr.str "detail" detail
        ]

let private summary () =
    let g = qwenGoldens.Value
    let all = lock gate (fun () -> List.ofSeq results.Values)
    let selfs = lock gate (fun () -> List.ofSeq selfTests)
    let failures = List<string>()

    // The oracles' own smallest margins over the gated steps, beside the candidates' (the parity report's chart).
    do
        use _ = install ExecutionBackend.CpuF64

        for p in g.Gate do
            for oracle in [ GoldenOracle.Torch32; GoldenOracle.Torch64p ] do
                let m = g.GreedyMargins(p.Name, oracle)

                Telemetry.gauge
                    "goldens.greedy.min_margin"
                    (Array.min m[0 .. p.GatedSteps - 1])
                    TelemetryUnits.NoUnit
                    [
                        Attr.str "prompt" p.Name
                        Attr.str "oracle" (GoldenOracle.name oracle)
                        Attr.int "steps" p.GatedSteps
                    ]

    let groupCheck
        (backend: ExecutionBackend)
        (check: string)
        (prompts: GoldenPrompt list)
        (pick: CheckResult -> bool)
        =
        let name = ExecutionBackend.name backend

        let missing =
            prompts |> List.filter (fun p -> not (results.ContainsKey((name, p.Name))))

        let checks =
            [
                for p in prompts do
                    match results.TryGetValue((name, p.Name)) with
                    | true, r -> yield! r.Checks |> List.filter pick
                    | _ -> ()
            ]

        let bad = checks |> List.filter (fun c -> not c.Passed)
        let passed = missing.IsEmpty && bad.IsEmpty && not checks.IsEmpty

        let detail =
            [
                Invariant.format $"{checks.Length} checks over {prompts.Length - missing.Length} prompts"
                (if bad.IsEmpty then
                     ""
                 else
                     Invariant.format $"{bad.Length} failed")
                (match worst checks with
                 | Some c -> "worst " + describe c
                 | None -> "")
                (if missing.IsEmpty then
                     ""
                 else
                     "not run: " + String.Join(", ", missing |> List.map _.Name))
            ]
            |> List.filter (fun s -> s <> "")
            |> String.concat "; "

        use _ = install backend
        gateCheck name check passed detail

        if not passed then
            failures.Add(name + ": " + check + ": " + detail)

    let isTf (c: CheckResult) =
        c.Metric = QwenParity.TeacherForcedMetric

    let isGreedy (c: CheckResult) = c.Metric = "greedy_match"
    let isLogits (c: CheckResult) = tapOf c = "logits"

    for b in backends do
        let label =
            match b with
            | ExecutionBackend.CpuF64 -> "within 1e-10 of torch64p"
            | _ -> "within the calibrated tolerances of torch32 and torch64p"

        groupCheck b ("gate prompts: logits " + label) g.Gate isLogits

        groupCheck b ("taps prompts: every tap " + label) (g.Gate |> List.filter _.Taps) (fun c ->
            not (isLogits c) && not (isTf c) && not (isGreedy c))

        groupCheck b "gate prompts: greedy ids identical over the gated steps" g.Gate isGreedy
        groupCheck b "gate prompts: teacher-forced argmax where the margin is at least 2·τ_logit" g.Gate isTf

        groupCheck b ("long prompts: final_norm.out and logits " + label + ", teacher-forced argmax") g.Long (fun _ ->
            true)

        let name = ExecutionBackend.name b
        let mine = selfs |> List.filter (fun (backend, _, _) -> backend = name)
        let ok = not mine.IsEmpty && mine |> List.forall (fun (_, _, p) -> p)

        do
            use _ = install b

            gateCheck
                name
                "bit identity: a prefill equals the prefix rows of a longer prefill in other chunks"
                ok
                (Invariant.format $"{mine.Length} self-tests")

        if not ok then
            failures.Add(name + ": bit identity")

    let rope = selfs |> List.filter (fun (backend, _, _) -> backend = "")
    let ropeOk = not rope.IsEmpty && rope |> List.forall (fun (_, _, p) -> p)

    do
        use _ = install ExecutionBackend.CpuF64
        gateCheck "" "T_spec equals the goldens' RoPE table" ropeOk "positions 0..8191, inv_freq bits"

    if not ropeOk then
        failures.Add "T_spec"

    // The time per prompt, for the PR and the README (host seconds at 8 threads).
    printfn "| Prompt | Tokens | Rows | %s |" (String.Join(" | ", backends |> List.map ExecutionBackend.name))
    printfn "| --- | ---: | ---: |%s" (String.replicate backends.Length " ---: |")

    for p in g.Prompts do
        let times =
            backends
            |> List.map (fun b ->
                match results.TryGetValue((ExecutionBackend.name b, p.Name)) with
                | true, r -> Invariant.format $"{r.TotalSeconds:F1} s"
                | _ -> "n/a")

        let rows =
            results.Values
            |> Seq.tryFind (fun r -> r.Prompt.Name = p.Name)
            |> Option.map (fun r -> Invariant.i32 r.Rows)
            |> Option.defaultValue "n/a"

        printfn "| %s | %d | %s | %s |" p.Name p.Ids.Length rows (String.Join(" | ", times))

    let checks = all |> List.sumBy _.Checks.Length
    printfn "gate1-cpu: %d prompt runs, %d checks, %d self-tests" all.Length checks selfs.Length

    if failures.Count > 0 then
        failwithf "gate1-cpu: %s" (String.Join(" | ", failures))

let tests =
    let cases =
        try
            let g = qwenGoldens.Value

            [
                testCase "T_spec: CpuOps' RoPE table and inv_freq bits equal the goldens' rope.safetensors"
                <| ropeSelfTest

                for p in g.Gate do
                    for b in backends do
                        let what =
                            match b with
                            | ExecutionBackend.CpuF64 -> "1e-10 of torch64p"
                            | _ -> "the calibrated tolerances"

                        let taps = if p.Taps then "every tap" else "logits"

                        testCase (
                            Invariant.format
                                $"{ExecutionBackend.name b} {p.Name} ({p.Ids.Length} tokens): {taps} within {what}, teacher-forced argmax, {p.GatedSteps} greedy ids"
                        )
                        <| fun () -> promptCase b p

                for p in g.Long do
                    for b in backends do
                        testCase (
                            Invariant.format
                                $"{ExecutionBackend.name b} {p.Name} ({p.Ids.Length} tokens): final_norm.out, logits rows and teacher-forced argmax"
                        )
                        <| fun () -> promptCase b p

                testCase "gate1-cpu: every prompt and self-test passed on both classes (gate1.check, group cpuref)"
                <| summary
            ]
        with e ->
            [
                testCase "gate1-cpu: the golden set goldens/qwen3-0.6b/g1 opens"
                <| fun () -> raise (Exception("the golden set does not open", e))
            ]

    testSequenced (testList "gate1-cpu" cases)
