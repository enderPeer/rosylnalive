/// The comparisons behind `lrm forward --compare` and `lrm generate --expect` (GoldenCheck), against the tiny goldens:
/// a tensor that cannot be compared is a failed check, never a skipped one; a last-row head compares with the golden's
/// last row; a missing tap fails when every tap is required; and greedy decoding of fewer steps than the golden is
/// compared with the golden's prefix.
module Lrm.Models.Qwen.Tests.GoldenCheckTests

open System
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

/// The CPU32 floors for every tensor and metric.
let private cpu32 (_: string) (metric: ToleranceMetric) =
    match metric with
    | ToleranceMetric.RelL2 -> Some ForwardTests.Cpu32RelL2
    | ToleranceMetric.MaxAbs -> Some ForwardTests.Cpu32LogitMaxAbs
    | ToleranceMetric.Kl -> Some ForwardTests.Cpu32LogitKl

let private attrsOf (name: string) =
    ForwardTests.attrs TolerancePair.Cpu32VsTorch32 "p0" name

let private detailOf (r: CheckResult) =
    r.Attrs
    |> List.tryPick (fun a ->
        match a.Value with
        | AttrValue.Str s when a.Key = GoldenCheck.DetailKey -> Some s
        | _ -> None)

let tests =
    testList
        "golden-check"
        [
            testCase "every tap and the logits of p0 compare and pass; a shape that differs is a failed check"
            <| fun () ->
                let logits, taps = ForwardTests.prefillTaps (tinyF32 ()) (promptIds "p0")
                use f = tiny.Force().OpenSafetensors "fwd_f32.p0.safetensors"
                let outputs = [ for KeyValue(n, t) in taps -> n, t ]
                expectSameBits "the logits tap" taps["logits"] logits

                let all = GoldenCheck.tensors f outputs cpu32 attrsOf true
                let checks = all |> List.collect _.Checks
                Expect.equal all.Length 39 "39 taps, the logits among them"
                Expect.equal checks.Length (39 + 2) "relL2 each, and max |Δ| and KL for the logits"
                Expect.isTrue (checks |> List.forall _.Passed) "within the CPU32 floors"

                // One row short: not comparable, so a failed relL2 NaN with the reason.
                let short = rowsOf taps["L0.out"] 0 12

                let r =
                    GoldenCheck.tensors f [ "L0.out", short ] cpu32 attrsOf false |> List.exactlyOne

                let c = List.exactlyOne r.Checks
                Expect.isFalse c.Passed "a shape mismatch fails"
                Expect.isTrue (Double.IsNaN c.Measured) "relL2 NaN: an invalid comparison"
                Expect.stringContains (detailOf c |> Option.defaultValue "") "not comparable" "the detail"

            testCase "a last-row head compares with the golden's last row; an unknown tensor has no checks"
            <| fun () ->
                let ids = promptIds "p0"
                use runner = CpuBackends.cpuF32.Create tinyWeights.Value (tinyOptions ())
                let r = runner.Forward(ids, TapFilter.Off, HeadRows.Last)
                use f = tiny.Force().OpenSafetensors "fwd_f32.p0.safetensors"

                match GoldenCheck.tensors f [ "logits", r.Logits; "extra", r.Logits ] cpu32 attrsOf false with
                | [ logits; extra ] ->
                    Expect.equal logits.Note " (last row)" "the last row"
                    Expect.equal logits.Checks.Length 3 "relL2, max |Δ| and KL"
                    Expect.isTrue (logits.Checks |> List.forall _.Passed) "passes"
                    Expect.isEmpty extra.Checks "not in the file"
                | xs -> failtestf "%A" xs

            testCase "with every tap required, each golden tap the run did not produce is a failed check"
            <| fun () ->
                use runner = CpuBackends.cpuF32.Create tinyWeights.Value (tinyOptions ())
                let r = runner.Forward(promptIds "p0", TapFilter.Off, HeadRows.All)
                use f = tiny.Force().OpenSafetensors "fwd_f32.p0.safetensors"
                let results = GoldenCheck.tensors f [ "logits", r.Logits ] cpu32 attrsOf true

                let failed =
                    results
                    |> List.filter (fun x -> x.Checks |> List.exists (fun c -> not c.Passed))

                Expect.equal failed.Length 38 "the 38 taps but logits"
                Expect.isFalse (failed |> List.exists (fun x -> x.Name = "input_ids")) "input_ids is the prompt"

                let without = GoldenCheck.tensors f [ "logits", r.Logits ] cpu32 attrsOf false
                Expect.equal without.Length 1 "only what the run wrote"

            testCase "greedy_match: 16 steps against the 64-step golden compare with its first 16 ids"
            <| fun () ->
                let golden = goldenGreedy "p0" "torch32"
                Expect.equal golden.Length 64 "the golden holds 64 ids"
                let ok = GoldenCheck.greedy [] golden[0..15] golden
                Expect.isTrue ok.Passed "16 of 16"
                Expect.equal (ok.Measured, ok.Bound, ok.Elements) (16.0, 16.0, 16L) "measured against the steps run"

                let wrong = Array.copy golden[0..15]
                wrong[3] <- (wrong[3] + 1) % 256
                let bad = GoldenCheck.greedy [] wrong golden
                Expect.isFalse bad.Passed "a different id fails"
                Expect.equal bad.Measured 3.0 "three steps match before it"

                // A greedy run of 16 steps on cpu-f32 matches the 64-id golden through the same check.
                use ops = tinyF32 ()
                let r = Greedy.run ops (tinyModel ()) (promptIds "p0") (GreedyOptions.parity 16)
                Expect.isTrue (GoldenCheck.greedy [] r.Ids golden).Passed "the run's 16 ids"

                Expect.throwsT<ArgumentException>
                    (fun () -> GoldenCheck.greedy [] (Array.append golden [| 1 |]) golden |> ignore)
                    "more ids than the golden holds"
        ]
