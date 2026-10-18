/// The plan and the handle lifetimes of the op seam (ForwardOpsAndQwenModel, "Lifetimes and the plan"), which CpuOps
/// enforces exactly as a GPU executor's static plan needs them: every forward is bracketed by BeginForward and
/// EndForward; intermediates end with their forward, a result and its taps with the next forward of the same stage,
/// an upload with the next forward, a cache with its release; and a call outside the plan is refused before any op.
/// Also the KV cache's release, ReadRows, and runStages leaving consistent caches when a stage or a transfer fails.
module Lrm.Models.Qwen.Tests.SeamTests

open System
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

/// A decorator that keeps every handle MatMul returns (intermediates the forward never hands out) and counts the
/// BeginForward and EndForward calls.
type private CaptureOps<'t>(inner: IForwardOps<'t>) =
    let matmuls = ResizeArray<'t>()
    let calls = ResizeArray<string>()

    member _.MatMuls = matmuls
    member _.Calls = calls

    interface IForwardOps<'t> with
        member _.ExecutionClass = inner.ExecutionClass
        member _.DType = inner.DType
        member _.MaxContext = inner.MaxContext
        member _.Rows x = inner.Rows x
        member _.Cols x = inner.Cols x

        member _.BeginForward call =
            calls.Add(Invariant.format $"begin {call.Stage.Index} {call.Rows} {call.Pos0}")
            inner.BeginForward call

        member _.EndForward r =
            calls.Add(if r.IsSome then "end" else "end failed")
            inner.EndForward r

        member _.Embed ids = inner.Embed ids
        member _.RmsNorm(x, w, e) = inner.RmsNorm(x, w, e)

        member _.MatMul(x, w) =
            let y = inner.MatMul(x, w)
            matmuls.Add y
            y

        member _.Rope(x, h, p) = inner.Rope(x, h, p)
        member _.NewCache(s, l, r) = inner.NewCache(s, l, r)
        member _.ReleaseCache c = inner.ReleaseCache c
        member _.KvWrite(c, x, p) = inner.KvWrite(c, x, p)
        member _.Attention(q, k, v, s, p) = inner.Attention(q, k, v, s, p)
        member _.SiluMul(g, u) = inner.SiluMul(g, u)
        member _.Add(a, b) = inner.Add(a, b)
        member _.LastRow x = inner.LastRow x
        member _.Argmax x = inner.Argmax x
        member _.Tap(n, x) = inner.Tap(n, x)
        member _.Read x = inner.Read x
        member _.ReadRows(x, r, n) = inner.ReadRows(x, r, n)
        member _.Upload x = inner.Upload x

    interface IDisposable with
        member _.Dispose() = inner.Dispose()

/// Raises ForwardException with `text` in its message, or fails the test.
let private refused (label: string) (text: string) (f: unit -> unit) =
    let e = throws<ForwardException> label f
    Expect.stringContains e.Message text label

let tests =
    testList
        "seam"
        [
            testCase
                "runStage brackets its ops: BeginForward first, EndForward with the result, or None when an op raises"
            <| fun () ->
                let model = tinyModel ()
                use capture = new CaptureOps<_>(tinyF32 ())
                let ops = capture :> IForwardOps<_>
                use cache = KvCache.create ops model (Stage.whole model.Config) 16
                Forward.run ops model cache 0 [| 1; 2; 3 |] ForwardOptions.last |> ignore
                Forward.run ops model cache 3 [| 4 |] ForwardOptions.last |> ignore

                Expect.throwsT<ArgumentOutOfRangeException>
                    (fun () -> Forward.run ops model cache 4 [| 999 |] ForwardOptions.last |> ignore)
                    "an id outside the vocabulary raises inside the forward"

                Expect.sequenceEqual
                    capture.Calls
                    [ "begin 0 3 0"; "end"; "begin 0 1 3"; "end"; "begin 0 1 4"; "end failed" ]
                    "bracketed"

                Expect.equal cache.Length 4 "a failed forward leaves the cache at pos0"

                // A refused call changes nothing and reaches no op.
                refused "a gap" "a forward starts at most at" (fun () ->
                    Forward.run ops model cache 9 [| 1 |] ForwardOptions.last |> ignore)

                Expect.equal capture.Calls.Count 6 "no BeginForward for a refused call"

            testCase
                "an intermediate ends with its forward; the result and the taps stay until the stage's next forward"
            <| fun () ->
                let model = tinyModel ()
                use capture = new CaptureOps<_>(tinyF32 ())
                let ops = capture :> IForwardOps<_>
                use cache = KvCache.create ops model (Stage.whole model.Config) 16
                let store = TapStore(TapFilter.Names [ "L0.attn.q_proj.out"; "L1.out" ])

                let logits =
                    Forward.run ops model cache 0 [| 1; 2; 3 |] (ForwardOptions.withTaps store)

                // q_proj of layer 0 is a tap; k_proj of layer 0 is an intermediate.
                let q0 = capture.MatMuls[0]
                let k0 = capture.MatMuls[1]
                Expect.isTrue (obj.ReferenceEquals(snd store.Taps[0], q0)) "the q_proj tap is the MatMul result"
                refused "an intermediate after EndForward" "no longer valid" (fun () -> ops.Read k0 |> ignore)
                ops.Read logits |> ignore
                ops.Argmax logits |> ignore

                for _, x in store.Taps do
                    ops.Read x |> ignore

                Forward.run ops model cache 3 [| 4 |] ForwardOptions.last |> ignore

                refused "the result after the stage's next forward" "no longer valid" (fun () ->
                    ops.Read logits |> ignore)

                refused "a tap after the stage's next forward" "no longer valid" (fun () ->
                    ops.Read(snd store.Taps[1]) |> ignore)

            testCase "two stages on one ops: stage 0's result and taps outlive stage 1's forward"
            <| fun () ->
                let model = tinyModel ()
                use ops = CpuOps.f32 tinyWeights.Value (splitOptions [ 1 ])
                let stages = Stage.split model.Config [ 1 ]
                let caches = stages |> List.map (fun s -> KvCache.create ops model s 13)
                let store = TapStore(TapFilter.All)

                // The residual passes as the handle itself, without the host.
                let logits =
                    Forward.runStages ops model stages caches 0 (promptIds "p0") id (ForwardOptions.withTaps store)

                Expect.equal store.Count 39 "every tap of both stages"

                for _, x in store.Taps do
                    ops.Read x |> ignore

                ops.Read logits |> ignore

                for c in caches do
                    (c :> IDisposable).Dispose()

            testCase "ops outside a forward, and cache ops inside one, are refused"
            <| fun () ->
                let model = tinyModel ()
                use ops = tinyF32 ()
                refused "Embed outside a forward" "Embed outside a forward" (fun () -> ops.Embed [| 1 |] |> ignore)

                let call =
                    {
                        Stage = Stage.whole model.Config
                        Rows = 1
                        Pos0 = 0
                        Head = HeadRows.Last
                        Taps = TapFilter.Off
                    }

                ops.BeginForward call
                refused "a nested forward" "BeginForward inside a forward" (fun () -> ops.BeginForward call)

                refused "NewCache inside a forward" "NewCache inside a forward" (fun () ->
                    ops.NewCache(call.Stage, 0, 4) |> ignore)

                refused "Upload inside a forward" "Upload inside a forward" (fun () ->
                    ops.Upload(HostTensor.zeros<float32> DType.F32 (Shape.matrix 1 64)) |> ignore)

                let e = ops.Embed [| 1 |]

                refused "a tap the forward does not ask for" "neither the forward nor the plan" (fun () ->
                    ops.Tap("embed.out", e) |> ignore)

                ops.EndForward(Some e)
                refused "EndForward twice" "EndForward outside a forward" (fun () -> ops.EndForward None)

            testCase "the plan bounds every call: stages, rows, positions, taps and caches"
            <| fun () ->
                let model = tinyModel ()
                let c = model.Config

                use ops =
                    CpuOps.f32
                        tinyWeights.Value
                        { OpsOptions.defaults c with
                            Threads = Threads
                            MaxContext = 16
                            MaxRows = 4
                            Taps = TapFilter.Names [ "logits" ]
                        }

                use cache = KvCache.create ops model (Stage.whole c) 16

                refused "more rows than MaxRows" "rows per forward" (fun () ->
                    Forward.run ops model cache 0 (Array.create 5 1) ForwardOptions.last |> ignore)

                Expect.equal cache.Length 0 "a call the plan refuses changes nothing"

                refused "a tap the plan does not cover" "not in the ops' plan" (fun () ->
                    Forward.run ops model cache 0 [| 1 |] (ForwardOptions.withTaps (TapStore TapFilter.All))
                    |> ignore)

                let store = TapStore(TapFilter.Names [ "logits" ])
                Forward.run ops model cache 0 [| 1 |] (ForwardOptions.withTaps store) |> ignore
                Expect.equal store.Count 1 "a covered tap"

                refused "a second cache beyond Caches = 1" "the plan's limit" (fun () ->
                    KvCache.create ops model (Stage.whole c) 4 |> ignore)

                let s1 = (Stage.split c [ 1 ])[1]

                refused "a stage outside the plan" "not in the ops' plan" (fun () ->
                    KvCache.create ops model s1 4 |> ignore)

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        CpuOps.f32
                            tinyWeights.Value
                            { tinyOptions () with
                                Taps = TapFilter.Names [ "L0.bogus" ]
                            }
                        |> ignore)
                    "a plan naming a tap ActivationNames does not define"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        CpuOps.f32
                            tinyWeights.Value
                            { tinyOptions () with
                                Stages = [ Stage.whole c; (Stage.split c [ 1 ])[0] ]
                            }
                        |> ignore)
                    "two planned stages with one index"

            testCase "a cache's tensors live until it is disposed; Greedy and the runner release theirs"
            <| fun () ->
                let model = tinyModel ()
                let c = model.Config

                let oneCache =
                    { OpsOptions.defaults c with
                        Threads = Threads
                        MaxContext = 32
                        MaxRows = 13
                    }

                use ops = CpuOps.f32 tinyWeights.Value oneCache
                let ids = promptIds "p0"
                let a = Greedy.run ops model ids (GreedyOptions.parity 8)
                let b = Greedy.run ops model ids (GreedyOptions.parity 8)
                Expect.sequenceEqual a.Ids b.Ids "a second run in the one cache the plan allows"

                let cache = KvCache.create ops model (Stage.whole c) 8
                let k0 = cache.Keys 0
                (cache :> IDisposable).Dispose()
                (cache :> IDisposable).Dispose()
                refused "a released cache tensor" "no longer valid" (fun () -> ops.Read k0 |> ignore)
                Expect.throwsT<ObjectDisposedException> (fun () -> cache.Keys 0 |> ignore) "a disposed cache"

                refused "a forward into a disposed cache" "was disposed" (fun () ->
                    Forward.run ops model cache 0 [| 1 |] ForwardOptions.last |> ignore)

                use runner = CpuBackends.cpuF32.Create tinyWeights.Value oneCache
                let r1 = runner.Forward(ids, TapFilter.Off, HeadRows.Last)
                let r2 = runner.Forward(ids, TapFilter.Off, HeadRows.Last)
                expectSameBits "two runner forwards in one cache" r1.Logits r2.Logits

            testCase "an upload lasts until the next forward ends, or the next upload"
            <| fun () ->
                let model = tinyModel ()
                let c = model.Config
                use ops = CpuOps.f32 tinyWeights.Value (splitOptions [ 1 ])
                let stages = Stage.split c [ 1 ]
                use c0 = KvCache.create ops model stages[0] 8
                use c1 = KvCache.create ops model stages[1] 8
                let ids = [| 1; 2 |]

                let h =
                    Forward.runStage ops model stages[0] c0 0 (StageInput.Ids ids) ForwardOptions.last

                let u1 = Forward.hostRoundTrip ops h
                let u2 = Forward.hostRoundTrip ops h
                refused "an upload replaced by another" "no longer valid" (fun () -> ops.Read u1 |> ignore)

                Forward.runStage ops model stages[1] c1 0 (StageInput.Hidden u2) ForwardOptions.last
                |> ignore

                refused "an upload after the forward that consumed it" "no longer valid" (fun () ->
                    ops.Read u2 |> ignore)

            testCase "the ops are unusable after Dispose"
            <| fun () ->
                let ops = tinyF32 ()
                let model = tinyModel ()
                let cache = KvCache.create ops model (Stage.whole model.Config) 4
                ops.Dispose()

                Expect.throwsT<ObjectDisposedException>
                    (fun () -> Forward.run ops model cache 0 [| 1 |] ForwardOptions.last |> ignore)
                    "a forward"

                // Releasing after Dispose is a no-op: Dispose released everything.
                (cache :> IDisposable).Dispose()

            testCase "ReadRows reads exactly the rows asked for; the digest reads only a cache's valid rows"
            <| fun () ->
                let model = tinyModel ()
                use ops = tinyF64 ()
                use cache = KvCache.create ops model (Stage.whole model.Config) 32
                Forward.run ops model cache 0 (promptIds "p0") ForwardOptions.last |> ignore
                let k = cache.Keys 1
                expectSameBits "rows 3 .. 7" (ops.ReadRows(k, 3, 5)) (rowsOf (ops.Read k) 3 8)

                Expect.throwsT<ArgumentOutOfRangeException>
                    (fun () -> ops.ReadRows(k, 30, 3) |> ignore)
                    "beyond the rows"

                Expect.throwsT<ArgumentOutOfRangeException> (fun () -> ops.ReadRows(k, 0, 0) |> ignore) "no rows"

                // Rows beyond Length never change the digest.
                let d = KvCache.digest ops cache
                (cache.Keys 0).Data[ 20 * model.Config.KvDim ] <- 123.0
                Expect.equal (KvCache.digest ops cache) d "rows beyond Length"

            testCase "runStages: if a stage or the transfer raises, every cache's Length is pos0"
            <| fun () ->
                let model = tinyModel ()
                let c = model.Config
                use ops = CpuOps.f32 tinyWeights.Value (splitOptions [ 1 ])
                let stages = Stage.split c [ 1 ]
                let caches = stages |> List.map (fun s -> KvCache.create ops model s 16)
                let p0 = promptIds "p0"

                Forward.runStages ops model stages caches 0 p0 (Forward.hostRoundTrip ops) ForwardOptions.last
                |> ignore

                Expect.equal (caches |> List.map _.Length) [ 13; 13 ] "both stages ran"

                Expect.throwsT<InvalidOperationException>
                    (fun () ->
                        Forward.runStages
                            ops
                            model
                            stages
                            caches
                            13
                            [| 5 |]
                            (fun _ -> invalidOp "the link dropped")
                            ForwardOptions.last
                        |> ignore)
                    "a transfer that fails"

                Expect.equal (caches |> List.map _.Length) [ 13; 13 ] "rolled back to pos0 = 13"

                refused
                    "a cache that cannot take the call is refused before any stage runs"
                    "a forward starts at most at"
                    (fun () ->
                        Forward.runStages
                            ops
                            model
                            stages
                            caches
                            14
                            [| 5 |]
                            (Forward.hostRoundTrip ops)
                            ForwardOptions.last
                        |> ignore)

                Expect.equal (caches |> List.map _.Length) [ 13; 13 ] "nothing ran"

                for x in caches do
                    (x :> IDisposable).Dispose()
        ]
