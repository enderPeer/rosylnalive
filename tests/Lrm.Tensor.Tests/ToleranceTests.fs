/// ToleranceSpec types (G2 owns the numbers) and KernelBounds.
module Lrm.Tensor.Tests.ToleranceTests

open System
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Tensor.Tests.Common

/// A tolerances.json in the lrm.tolerances/1 schema, with one calibrated and two fixed pairs.
let sample =
    """{
  "schema": "lrm.tolerances/1",
  "set": "qwen3-0.6b/g1",
  "pairs": {
    "cpu32-vs-torch32": {
      "rule": { "k": 3, "floors": { "relL2": 1e-05, "maxAbs": 0.002, "kl": 1e-07 } },
      "taps": {
        "attn.q_rope": { "relL2": { "bound": 1e-05, "e_ref": 1.6e-07 } },
        "mlp.down.out": { "relL2": { "bound": 6.000000000000001e-05, "e_ref": 2e-05 } },
        "logits": {
          "relL2": { "bound": 0.0001, "e_ref": 0.001 },
          "maxAbs": { "bound": 0.003, "e_ref": 0.001 },
          "kl": { "bound": 1e-07, "e_ref": 1e-09 }
        }
      }
    },
    "cpu64-vs-torch64p": { "default": { "relL2": { "bound": 1e-10 } } },
    "vulkan-vs-d3d12": { "default": { "relL2": { "bound": 0.0001 }, "maxAbs": { "bound": 0.01 } } }
  },
  "greedy": { "min_margin": "max(0.05, 4 * tau_logit)" }
}"""

let private refused (label: string) (json: string) =
    Expect.throwsT<ToleranceFormatException> (fun () -> ToleranceSpec.parse json |> ignore) label

let private gpu = Array.init 64 (fun i -> float i * 0.25)

let tests =
    testList
        "bounds"
        [
            testCase "calibrate is clamp(k * e_ref, floor, 10 * floor)"
            <| fun () ->
                Expect.equal (ToleranceSpec.calibrate 3.0 1e-5 1.6e-7) 1e-5 "floor"
                Expect.equal (ToleranceSpec.calibrate 3.0 1e-5 2e-5) 6.000000000000001e-05 "k * e_ref, rounded once"
                Expect.equal (ToleranceSpec.calibrate 3.0 1e-5 1e-3) 1e-4 "ceiling"
                Expect.equal (ToleranceSpec.calibrate 9.0 1e-4 5e-5) (9.0 * 5e-5) "GPU"

            testCase "a tolerances file parses, and lookups prefer the tap kind over the default"
            <| fun () ->
                let s = ToleranceSpec.parse sample
                Expect.equal s.Set "qwen3-0.6b/g1" "set"
                Expect.equal (s.Bound(TolerancePair.Cpu32VsTorch32, "attn.q_rope", ToleranceMetric.RelL2)) 1e-5 "tap"
                Expect.equal (s.Bound(TolerancePair.Cpu32VsTorch32, "logits", ToleranceMetric.Kl)) 1e-7 "kl"
                Expect.equal (s.Bound(TolerancePair.Cpu64VsTorch64p, "anything", ToleranceMetric.RelL2)) 1e-10 "default"
                Expect.isNone (s.TryBound(TolerancePair.Cpu64VsTorch64p, "logits", ToleranceMetric.Kl)) "absent"
                Expect.isNone (s.TryBound(TolerancePair.GpuVsTorch32, "logits", ToleranceMetric.RelL2)) "absent pair"

                Expect.equal
                    ((s.TryBound(TolerancePair.Cpu32VsTorch32, "logits", ToleranceMetric.MaxAbs)).Value.ERef)
                    (Some 0.001)
                    "e_ref"

                Expect.isSome s.Greedy "greedy kept"

                Expect.throwsT<Collections.Generic.KeyNotFoundException>
                    (fun () -> s.Bound(TolerancePair.GpuVsTorch32, "logits", ToleranceMetric.RelL2) |> ignore)
                    "missing raises"

            testCase "pair and metric names are frozen"
            <| fun () ->
                Expect.equal
                    (TolerancePair.all |> List.map TolerancePair.name)
                    [
                        "cpu64-vs-torch64p"
                        "cpu32-vs-torch32"
                        "cpu32-vs-torch64p"
                        "gpu-vs-cpu32"
                        "gpu-vs-torch32"
                        "gpu-vs-torch64p"
                        "vulkan-vs-d3d12"
                    ]
                    "pairs"

                Expect.equal
                    (ToleranceMetric.all |> List.map ToleranceMetric.name)
                    [ "relL2"; "maxAbs"; "kl" ]
                    "metrics"

            testList
                "malformed tolerance files are refused"
                [
                    testCase "schema" (fun () ->
                        refused "schema" (sample.Replace("lrm.tolerances/1", "lrm.tolerances/2")))
                    testCase "a bound that does not follow its rule" (fun () ->
                        refused "rule" (sample.Replace("\"bound\": 6.000000000000001e-05", "\"bound\": 6e-05")))
                    testCase "a calibrated bound without e_ref" (fun () ->
                        refused
                            "e_ref"
                            (sample.Replace("{ \"bound\": 1e-05, \"e_ref\": 1.6e-07 }", "{ \"bound\": 1e-05 }")))
                    testCase "an unknown pair" (fun () ->
                        refused "pair" (sample.Replace("vulkan-vs-d3d12", "vulkan-vs-metal")))
                    testCase "an unknown metric" (fun () ->
                        refused
                            "metric"
                            (sample.Replace("\"maxAbs\": { \"bound\": 0.01 }", "\"maxUlp\": { \"bound\": 0.01 }")))
                    testCase "a non-positive bound" (fun () ->
                        refused "bound" (sample.Replace("\"bound\": 1e-10", "\"bound\": 0")))
                    testCase "a duplicate key" (fun () ->
                        refused
                            "dup"
                            (sample.Replace(
                                "\"cpu64-vs-torch64p\": {",
                                "\"cpu64-vs-torch64p\": {}, \"cpu64-vs-torch64p\": {"
                            )))
                    testCase "not JSON" (fun () -> refused "json" "{")
                ]

            testCase "KernelBounds covers the 14 catalog kernels once each, with the contract's exact set"
            <| fun () ->
                let names = KernelBounds.all |> List.map _.Kernel
                Expect.equal names.Length 14 "14 kernels"
                Expect.equal (List.distinct names).Length 14 "unique"

                let exact =
                    KernelBounds.all
                    |> List.filter (fun b -> b.Kind = KernelBoundKind.BitExact)
                    |> List.map _.Kernel
                    |> Set.ofList

                Expect.equal
                    exact
                    (set
                        [
                            "scale_f32"
                            "table_echo"
                            "args_echo"
                            "groups_echo"
                            "bf16_unpack"
                            "embed_gather_bf16"
                            "add"
                            "copy_rows"
                            "argmax_rows"
                        ])
                    "bit-exact kernels"

                Expect.throwsT<ArgumentException>
                    (fun () -> KernelBounds.find "softmax_rows" |> ignore)
                    "unknown kernel"

            testCase "the relative bound is max(1e-6, 2 * relL2(CpuRef32, CpuRef64))"
            <| fun () ->
                Expect.equal (KernelBounds.relativeBound 1e-9) 1e-6 "floor"
                Expect.equal (KernelBounds.relativeBound 3e-6) 6e-6 "factor"
                Expect.isTrue (Double.IsNaN(KernelBounds.relativeBound nan)) "NaN"

            testCase "a case passes inside the bound and fails outside it; NaN fails"
            <| fun () ->
                let r64 = ReadOnlySpan(gpu)
                let r32 = gpu |> Array.mapi (fun i x -> if i = 3 then x * (1.0 + 1e-3) else x)
                let refErr = Metrics.relL2 (ReadOnlySpan r32) r64

                let ok =
                    KernelBounds.checkRelative "matmul_bf16w" (ReadOnlySpan r32) (ReadOnlySpan r32) r64

                Expect.isTrue ok.Passed "the reference itself passes"
                Expect.equal ok.ReferenceError refErr "reference error"
                Expect.equal ok.Bound (max 1e-6 (2.0 * refErr)) "bound"
                let worse = gpu |> Array.mapi (fun i x -> if i = 5 then x + 1.0 else x)

                Expect.isFalse
                    (KernelBounds.checkRelative "matmul_bf16w" (ReadOnlySpan worse) (ReadOnlySpan r32) r64).Passed
                    "fails"

                let bad = Array.copy gpu
                bad[0] <- nan

                Expect.isFalse
                    (KernelBounds.checkRelative "rmsnorm_rows" (ReadOnlySpan bad) (ReadOnlySpan r32) r64).Passed
                    "NaN"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        KernelBounds.checkRelative "add" (ReadOnlySpan gpu) (ReadOnlySpan gpu) (ReadOnlySpan gpu)
                        |> ignore)
                    "exact kernels have no relative bound"

            testCase "exact kernels count differing elements"
            <| fun () ->
                let a = HostTensor.f32 (Shape.vector 4) [| 1.0f; 2.0f; 3.0f; 4.0f |]
                let b = HostTensor.f32 (Shape.vector 4) [| 1.0f; 2.0f; -3.0f; 4.0f |]
                let same = KernelBounds.check "add" a a None
                Expect.isTrue same.Passed "equal"
                let diff = KernelBounds.check "add" b a None
                Expect.equal (diff.Measured, diff.Passed) (1.0, false) "one element"

                let nan1 =
                    HostTensor.f32 (Shape.vector 1) [| BitConverter.UInt32BitsToSingle 0x7FC00000u |]

                Expect.isTrue (KernelBounds.check "embed_gather_bf16" nan1 nan1 None).Passed "NaN bits compare as bits"

                Expect.throwsT<ArgumentException>
                    (fun () -> KernelBounds.check "matmul_bf16w" a a None |> ignore)
                    "relative kernels need CpuRef64"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        let b = (a :> IHostTensor).ToBytes()

                        KernelBounds.checkExact "matmul_bf16w" 4 (ReadOnlySpan b) (ReadOnlySpan b)
                        |> ignore)
                    "relative kernels have no exact check"

                Expect.equal
                    (KernelBounds.attrs diff |> List.map _.Key)
                    [ "kernel"; "bound_kind"; "measured"; "bound"; "ref_error"; "elements" ]
                    "conformance.case attributes"

            testCase "CpuRef F32 is within its own bound of CpuRef F64 on a matmul case"
            <| fun () ->
                let x = random<float> 3UL "kb/x" (17 * 256) 1.0
                let w = randomBf16 3UL "kb/w" (96 * 256) 0.1f
                let wt = HostTensor.bf16 (Shape.matrix 96 256) w

                let y64 =
                    CpuRef.matmulBf16W CpuExec.Machine (HostTensor.f64 (Shape.matrix 17 256) x) wt

                let y32 =
                    CpuRef.matmulBf16W
                        CpuExec.Machine
                        (HostTensor.f32 (Shape.matrix 17 256) (x |> Array.map float32))
                        wt

                let c = KernelBounds.check "matmul_bf16w" y32 y32 (Some y64)
                Expect.isTrue c.Passed (string c)
                Expect.isTrue (c.ReferenceError > 0.0 && c.ReferenceError < 1e-6) "float32 error, measured"
        ]
