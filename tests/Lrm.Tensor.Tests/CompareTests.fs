/// ComparatorsAndGoldenReader: the metrics, their known answers, and the golden.compare records.
module Lrm.Tensor.Tests.CompareTests

open System
open Expecto
open FsCheck
open Lrm.Core
open Lrm.Tensor
open Lrm.Testing
open Lrm.Tensor.Tests.Common

let private f64 (xs: float list) =
    HostTensor.f64 (Shape.vector xs.Length) (Array.ofList xs)

let private f32 (xs: float32 list) =
    HostTensor.f32 (Shape.vector xs.Length) (Array.ofList xs)

let private rows (r: int) (xs: float list) =
    HostTensor.f64 (Shape.matrix r (xs.Length / r)) (Array.ofList xs)

let private ro (xs: float list) = ReadOnlySpan(Array.ofList xs)

let tests =
    testList
        "compare"
        [
            testCase "relL2 known answers"
            <| fun () ->
                Expect.equal (Metrics.relL2 (ro [ 1.0; 2.0; 3.0 ]) (ro [ 1.0; 2.0; 4.0 ])) (1.0 / sqrt 21.0) "1/sqrt 21"
                Expect.equal (Metrics.relL2 (ro [ 0.0; 0.0 ]) (ro [ 0.0; 0.0 ])) 0.0 "both zero"
                Expect.equal (Metrics.relL2 (ro [ 1.0 ]) (ro [ 0.0 ])) infinity "zero reference"
                Expect.isTrue (Double.IsNaN(Metrics.relL2 (ro [ nan ]) (ro [ 1.0 ]))) "NaN"
                Expect.isTrue (Double.IsNaN(Metrics.relL2 (ro [ infinity ]) (ro [ infinity ]))) "inf - inf"

            testCase "maxAbs, cosine and ulp distances"
            <| fun () ->
                Expect.equal (Metrics.maxAbs (ro [ 1.0; -2.0; 3.0 ]) (ro [ 1.5; -2.0; 2.0 ])) 1.0 "maxAbs"
                Expect.isTrue (Double.IsNaN(Metrics.maxAbs (ro [ 1.0; nan ]) (ro [ 1.0; 2.0 ]))) "NaN"
                Expect.equal (Metrics.cosine (ro [ 1.0; 0.0 ]) (ro [ 0.0; 1.0 ])) 0.0 "orthogonal"
                Expect.floatClose Accuracy.veryHigh (Metrics.cosine (ro [ 2.0; 2.0 ]) (ro [ 1.0; 1.0 ])) 1.0 "parallel"
                Expect.equal (Metrics.cosine (ro [ 0.0 ]) (ro [ 0.0 ])) 1.0 "both zero"
                Expect.equal (Metrics.ulpDistance32 1.0f (MathF.BitIncrement 1.0f)) 1.0 "next float32"
                Expect.equal (Metrics.ulpDistance32 0.0f -0.0f) 0.0 "signed zeros"
                Expect.equal (Metrics.ulpDistance32 Single.Epsilon -Single.Epsilon) 2.0 "across zero"
                Expect.equal (Metrics.ulpDistance32 1.0f nanf) infinity "NaN"
                Expect.equal (Metrics.ulpDistance64 1.0 (Math.BitIncrement 1.0)) 1.0 "next float64"
                Expect.equal (Metrics.ulpDistance64 -Double.Epsilon Double.Epsilon) 2.0 "float64 across zero"
                Expect.equal (Metrics.ulpDistanceBf16 0x3F80us 0x3F82us) 2.0 "bf16"

            testCase "KL, top-k agreement and top-2 margin known answers"
            <| fun () ->
                // P = (1/2, 1/2), Q = (1/4, 3/4): KL = 1/2 ln 2 + 1/2 ln(2/3) = 1/2 ln(4/3).
                let kl = Metrics.klRow (ro [ 0.0; log 3.0 ]) (ro [ 0.0; 0.0 ])
                Expect.floatClose { absolute = 1e-15; relative = 1e-14 } kl (0.5 * log (4.0 / 3.0)) "KL"
                Expect.equal (Metrics.klRow (ro [ 1.0; 2.0; 3.0 ]) (ro [ 1.0; 2.0; 3.0 ])) 0.0 "same rows"
                Expect.sequenceEqual (Metrics.topK (ro [ 1.0; 3.0; 3.0; 2.0 ]) 3) [| 1; 2; 3 |] "ties keep index order"
                Expect.equal (Metrics.topKAgreementRow (ro [ 5.0; 4.0; 0.0 ]) (ro [ 5.0; 0.0; 4.0 ]) 2) 0.5 "agreement"
                Expect.equal (Metrics.top2MarginRow (ro [ 1.0; 4.0; 2.5 ])) 1.5 "margin"
                Expect.throwsT<ArgumentException> (fun () -> Metrics.topK (ro [ 1.0; nan ]) 1 |> ignore) "NaN"

            testCase "every Compare check returns a CheckResult and emits golden.compare"
            <| fun () ->
                let a = f64 [ 1.0; 2.0; 3.0 ]
                let b = f64 [ 1.0; 2.0; 4.0 ]
                let ctx = [ Attr.str "pair" "cpu32-vs-torch32"; Attr.str "tap" "logits" ]

                let results, records =
                    withSink (fun () ->
                        [
                            Compare.relL2 0.3 ctx a b
                            Compare.relL2 0.1 ctx a b
                            Compare.maxAbs 1.0 ctx a b
                            Compare.cosine 0.99 ctx a b
                            Compare.kl 1.0 ctx (rows 1 [ 0.0; 0.0 ]) (rows 1 [ 0.0; 1.0 ])
                            Compare.topKAgreement 1 1.0 ctx (rows 1 [ 0.0; 1.0 ]) (rows 1 [ 0.0; 2.0 ])
                            Compare.top2Margin 0.5 ctx (rows 2 [ 0.0; 1.0; 3.0; 0.0 ])
                            Compare.exact ctx a b
                        ])

                Expect.sequenceEqual
                    (results |> List.map _.Passed)
                    [ true; false; true; true; true; true; true; false ]
                    "passed"

                Expect.equal
                    (results |> List.map _.Metric)
                    [
                        "relL2"
                        "relL2"
                        "maxAbs"
                        "cosine"
                        "kl"
                        "topKAgreement"
                        "top2Margin"
                        "exact"
                    ]
                    "metrics"

                Expect.equal (results |> List.last).Measured 1.0 "one element differs"
                Expect.equal (results[6]).Measured 1.0 "smallest per-row margin"
                Expect.equal records.Length results.Length "one record each"

                for r, res in List.zip records results do
                    Expect.equal r.Name TelemetryNames.GoldenCompare "name"
                    Expect.equal r.Value (if res.Passed then 1.0 else 0.0) "value"
                    Expect.equal (strAttr "metric" r) (Some res.Metric) "metric"
                    Expect.equal (floatAttr "measured" r) res.Measured "measured"
                    Expect.equal (floatAttr "bound" r) res.Bound "bound"
                    Expect.equal (strAttr "pair" r) (Some "cpu32-vs-torch32") "caller attrs"

            testCase "a NaN measurement never passes, in either direction"
            <| fun () ->
                Expect.isFalse (Compare.check "relL2" BoundOp.AtMost nan 1.0 1L []).Passed "le"
                Expect.isFalse (Compare.check "cosine" BoundOp.AtLeast nan 0.0 1L []).Passed "ge"

            testCase "maxUlp works per dtype and widens nothing"
            <| fun () ->
                let r =
                    Compare.maxUlp 1.0 [] (f32 [ 1.0f; MathF.BitIncrement 2.0f ]) (f32 [ 1.0f; 2.0f ])

                Expect.equal r.Measured 1.0 "f32 ulps"

                Expect.throwsT<ArgumentException>
                    (fun () -> Compare.maxUlp 1.0 [] (f32 [ 1.0f ]) (f64 [ 1.0 ]) |> ignore)
                    "one dtype"

            testCase "shapes must match; dtypes may differ for value metrics"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () -> Compare.relL2 1.0 [] (f64 [ 1.0 ]) (f64 [ 1.0; 2.0 ]) |> ignore)
                    "shape"

                let r = Compare.relL2 0.0 [] (f32 [ 0.5f; 0.25f ]) (f64 [ 0.5; 0.25 ])
                Expect.isTrue r.Passed "f32 widened exactly"

            Props.testProperty "relL2 is 0 for identical inputs and invariant under power-of-two scaling"
            <| fun (xs: NormalFloat list) (e: int) ->
                let a = xs |> List.map (fun x -> Math.IEEERemainder(x.Get, 1e6)) |> Array.ofList
                let b = a |> Array.map (fun x -> x * 1.5 + 1.0)
                let s = Math.ScaleB(1.0, e % 32)
                let r = Metrics.relL2 (ReadOnlySpan a) (ReadOnlySpan b)

                let rs =
                    Metrics.relL2 (ReadOnlySpan(a |> Array.map ((*) s))) (ReadOnlySpan(b |> Array.map ((*) s)))

                Metrics.relL2 (ReadOnlySpan a) (ReadOnlySpan a) = 0.0
                && (Double.IsNaN r || Double.IsInfinity r || abs (rs - r) <= 1e-12 * max 1.0 r)
        ]
