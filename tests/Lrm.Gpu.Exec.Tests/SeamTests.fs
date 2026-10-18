/// The seam rules on the executor (ForwardOpsAndQwenModel, "Lifetimes and the plan"; GpuExecutor, "Pass two"), as
/// CpuOps enforces them: a call outside the plan is refused before any op; ops outside a forward and cache ops inside
/// one raise; intermediates end with their forward, the result and taps with the stage's next forward, a failed
/// forward's taps at once; the plan's cache limit; uploads; Dispose. And what only a static plan adds: a call that
/// departs from the static order, or a read of a result whose bytes the plan has reused, raises; a backend whose
/// kernel set lacks a kernel is refused at creation and disposed; the executor references no concrete backend.
module Lrm.Gpu.Exec.Tests.SeamTests

open System
open System.Collections.Generic
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Gpu.Exec.Tests.Common

/// Forwards every call to the executor and lets a test act before each compute op (by its seam name).
type private Spy(inner: IForwardOps<GpuTensor>, before: string -> unit) =
    let results = List<string * GpuTensor>()

    let keep (name: string) (x: GpuTensor) =
        results.Add(name, x)
        x

    member _.Results = List.ofSeq results

    interface IForwardOps<GpuTensor> with
        member _.ExecutionClass = inner.ExecutionClass
        member _.DType = inner.DType
        member _.MaxContext = inner.MaxContext
        member _.Rows x = inner.Rows x
        member _.Cols x = inner.Cols x
        member _.BeginForward call = inner.BeginForward call
        member _.EndForward r = inner.EndForward r

        member _.Embed ids =
            before "Embed"
            keep "Embed" (inner.Embed ids)

        member _.RmsNorm(x, w, eps) =
            before "RmsNorm"
            keep "RmsNorm" (inner.RmsNorm(x, w, eps))

        member _.MatMul(x, w) =
            before "MatMul"
            keep "MatMul" (inner.MatMul(x, w))

        member _.Rope(x, h, p) =
            before "Rope"
            keep "Rope" (inner.Rope(x, h, p))

        member _.NewCache(s, l, r) = inner.NewCache(s, l, r)
        member _.ReleaseCache x = inner.ReleaseCache x

        member _.KvWrite(c, x, p) =
            before "KvWrite"
            inner.KvWrite(c, x, p)

        member _.Attention(q, k, v, s, p) =
            before "Attention"
            keep "Attention" (inner.Attention(q, k, v, s, p))

        member _.SiluMul(g, u) =
            before "SiluMul"
            keep "SiluMul" (inner.SiluMul(g, u))

        member _.Add(a, b) =
            before "Add"
            keep "Add" (inner.Add(a, b))

        member _.LastRow x =
            before "LastRow"
            keep "LastRow" (inner.LastRow x)

        member _.Argmax x = inner.Argmax x
        member _.Tap(n, x) = inner.Tap(n, x)
        member _.Read x = inner.Read x
        member _.ReadRows(x, r, n) = inner.ReadRows(x, r, n)
        member _.Upload x = inner.Upload x

    interface IDisposable with
        member _.Dispose() = inner.Dispose()

/// A small plan: 16 positions, 4 rows, the logits tap, one cache.
let private small () =
    { OpsOptions.defaults tinyWeights.Value.Config with
        Threads = Threads
        MaxContext = 16
        MaxRows = 4
        Taps = TapFilter.Names [ "logits"; "embed.out" ]
        Caches = 1
    }

let private call (stage: Stage) (rows: int) (pos0: int) (taps: TapFilter) =
    {
        Stage = stage
        Rows = rows
        Pos0 = pos0
        Head = HeadRows.All
        Taps = taps
    }

let tests =
    testList
        "seam"
        [
            testCase "a call outside the plan is refused before any op, and changes nothing"
            <| fun () ->
                let model = tinyModel ()
                use ops = gpuWith GpuExecOptions.Default (small ())
                let g = ops :> IForwardOps<GpuTensor>
                use cache = KvCache.create g model (Stage.whole model.Config) 16
                let before = ops.Dispatches

                refused "more rows than MaxRows" "rows per forward" (fun () ->
                    Forward.run g model cache 0 [| 1; 2; 3; 4; 5 |] ForwardOptions.all |> ignore)

                refused "taps outside the plan" "not in the ops' plan" (fun () ->
                    Forward.run g model cache 0 [| 1; 2 |] (ForwardOptions.withTaps (TapStore(TapFilter.All)))
                    |> ignore)

                let second = List.last (Stage.split model.Config [ 1 ])

                refused "an unplanned stage" "not in the ops' plan" (fun () ->
                    g.BeginForward(call second 1 0 TapFilter.Off))

                refused "positions beyond MaxContext" "the ops hold" (fun () ->
                    g.BeginForward(call (Stage.whole model.Config) 4 13 TapFilter.Off))

                Expect.equal ops.Dispatches before "no dispatch"
                Expect.equal cache.Length 0 "the cache is untouched"

            testCase "ops outside a forward and cache ops inside one raise"
            <| fun () ->
                let model = tinyModel ()
                use g = gpu (small ())
                refused "Embed outside" "outside a forward" (fun () -> g.Embed [| 1 |] |> ignore)
                g.BeginForward(call (Stage.whole model.Config) 1 0 TapFilter.Off)

                refused "NewCache inside" "inside a forward" (fun () ->
                    g.NewCache(Stage.whole model.Config, 0, 4) |> ignore)

                refused "Upload inside" "inside a forward" (fun () ->
                    g.Upload(HostTensor.zeros<float32> DType.F32 (Shape.matrix 1 64)) |> ignore)

                g.EndForward None

            testCase "intermediates end with their forward; the result and taps live until the stage's next forward"
            <| fun () ->
                let model = tinyModel ()
                use g = gpu (small ())
                use cache = KvCache.create g model (Stage.whole model.Config) 16
                let spy = new Spy(g, ignore)
                let store = TapStore<GpuTensor>(TapFilter.Names [ "logits"; "embed.out" ])
                let l = Forward.run spy model cache 0 [| 1; 2; 3 |] (ForwardOptions.withTaps store)
                Expect.equal (g.Rows l, g.Cols l) (3, 256) "the logits"
                let taps = store.Taps |> List.map snd
                Expect.equal taps.Length 2 "two taps"

                for t in taps do
                    g.Read t |> ignore

                let intermediate = spy.Results |> List.find (fun (n, _) -> n = "Attention") |> snd
                refused "an intermediate after EndForward" "no longer valid" (fun () -> g.Read intermediate |> ignore)
                Forward.run g model cache 3 [| 4 |] ForwardOptions.all |> ignore
                refused "the result after the next forward" "no longer valid" (fun () -> g.Read l |> ignore)

                for t in taps do
                    refused "a tap after the next forward" "no longer valid" (fun () -> g.Read t |> ignore)

            testCase "a failed forward ends with EndForward None, and its taps are invalid at once"
            <| fun () ->
                let model = tinyModel ()
                use g = gpu (small ())
                g.BeginForward(call (Stage.whole model.Config) 2 0 (TapFilter.Names [ "embed.out" ]))
                let e = g.Embed [| 1; 2 |]
                let t = g.Tap("embed.out", e)
                g.EndForward None
                refused "the tap of a failed forward" "no longer valid" (fun () -> g.Read t |> ignore)

                Expect.throwsT<ArgumentOutOfRangeException>
                    (fun () ->
                        use cache = KvCache.create g model (Stage.whole model.Config) 4
                        Forward.run g model cache 0 [| 1; 999 |] ForwardOptions.all |> ignore)
                    "an id outside the vocabulary"

            testCase "two stages on one executor: stage 0's result feeds stage 1 without the host"
            <| fun () ->
                let model = tinyModel ()
                let ids = promptIds "p0"
                let stages = Stage.split model.Config [ 1 ]
                use g = gpu (splitOptions [ 1 ])

                let run (transfer: GpuTensor -> GpuTensor) =
                    let caches = stages |> List.map (fun s -> KvCache.create g model s ids.Length)

                    let l =
                        Forward.runStages g model stages caches 0 ids transfer ForwardOptions.all
                        |> g.Read

                    for c in caches do
                        (c :> IDisposable).Dispose()

                    l

                expectSameBits "logits" (run id) (run (Forward.hostRoundTrip g))

            testCase "the plan's cache limit, and caches released by Dispose"
            <| fun () ->
                let model = tinyModel ()
                use g = gpu (small ())
                let stage = Stage.whole model.Config
                let first = KvCache.create g model stage 8

                refused "a second cache beyond Caches = 1" "the plan's limit" (fun () ->
                    KvCache.create g model stage 8 |> ignore)

                (first :> IDisposable).Dispose()
                use second = KvCache.create g model stage 8
                let k = g.Read(second.Keys 0)
                Expect.equal (k.Desc.Shape.Rows, k.Desc.Shape.Cols) (8L, 64L) "[rows, KvDim]"
                Expect.isTrue (k.ToFloat64() |> Array.forall (fun v -> v = 0.0)) "a new cache holds zeros"

            testCase "uploads: f32 [rows, cols] within the buffer, valid until the next forward ends"
            <| fun () ->
                let model = tinyModel ()
                let stages = Stage.split model.Config [ 1 ]

                use g =
                    gpu
                        { splitOptions [ 1 ] with
                            MaxRows = 4
                            MaxContext = 8
                        }

                Expect.throwsT<ArgumentException>
                    (fun () -> g.Upload(HostTensor.zeros<float> DType.F64 (Shape.matrix 1 64)) |> ignore)
                    "f64"

                refused "beyond the buffer" "upload buffer" (fun () ->
                    g.Upload(HostTensor.zeros<float32> DType.F32 (Shape.matrix 5 64)) |> ignore)

                let x =
                    HostTensor.f32 (Shape.matrix 2 64) (Array.init 128 (fun i -> float32 i / 128.0f))

                let a = g.Upload x
                expectSameBits "the upload reads back" (g.Read a) x
                let b = g.Upload x
                refused "a second upload invalidates the first" "no longer valid" (fun () -> g.Read a |> ignore)
                use cache = KvCache.create g model stages[1] 8

                Forward.runStage g model stages[1] cache 0 (StageInput.Hidden b) ForwardOptions.all
                |> ignore

                refused "an upload after the next forward" "no longer valid" (fun () -> g.Read b |> ignore)

            testCase "a forward that departs from the static order is refused"
            <| fun () ->
                let model = tinyModel ()
                use g = gpu (small ())
                g.BeginForward(call (Stage.whole model.Config) 2 0 TapFilter.Off)
                let e = g.Embed [| 1; 2 |]

                refused "a MatMul where the plan has RmsNorm" "departs from the static plan" (fun () ->
                    g.MatMul(e, model.Params.Layers[0].Q) |> ignore)

                g.EndForward None
                g.BeginForward(call (Stage.whole model.Config) 2 0 TapFilter.Off)
                let e = g.Embed [| 1; 2 |]
                let h = g.RmsNorm(e, model.Params.Layers[0].InputNorm, model.Config.RmsNormEps)

                refused "the planned op on another input" "departs from the static plan" (fun () ->
                    g.MatMul(e, model.Params.Layers[0].Q) |> ignore)

                refused "an early EndForward" "not the plan's result" (fun () -> g.EndForward(Some h))

                Forward.run g model (KvCache.create g model (Stage.whole model.Config) 4) 0 [| 1 |] ForwardOptions.all
                |> ignore

            testCase "a result whose bytes the plan reused after its last use cannot be read"
            <| fun () ->
                let model = tinyModel ()
                use ops = gpuWith GpuExecOptions.Default (small ())
                let g = ops :> IForwardOps<GpuTensor>
                let v = (ops.Plan.Stages |> List.exactlyOne).Variants[0]
                Expect.isLessThan v.FirstClobber[0] Liveness.Never "the plan reuses the embedding's bytes"
                let clobberer = v.Ops |> Array.find (fun o -> o.Sequence = v.FirstClobber[0])
                let seen = ref 0
                let checkedBefore = ref false
                let refusedAfter = ref false
                let spy: Spy option ref = ref None

                let before (_: string) =
                    let embed = spy.Value.Value.Results |> List.tryFind (fun (n, _) -> n = "Embed")

                    match embed with
                    | Some(_, e) when seen.Value = clobberer.Index ->
                        // Before the op that reuses its bytes, the embedding still reads back.
                        g.Read e |> ignore
                        checkedBefore.Value <- true
                    | Some(_, e) when seen.Value = clobberer.Index + 1 ->
                        refusedAfter.Value <-
                            try
                                g.Read e |> ignore
                                false
                            with :? ForwardException as x ->
                                x.Message.Contains "overwritten"
                    | _ -> ()

                    seen.Value <- seen.Value + 1

                spy.Value <- Some(new Spy(g, before))

                use cache = KvCache.create g model (Stage.whole model.Config) 16

                Forward.run spy.Value.Value model cache 0 [| 1; 2; 3 |] ForwardOptions.all
                |> ignore

                Expect.isTrue checkedBefore.Value "read before the reuse"
                Expect.isTrue refusedAfter.Value "refused after the reuse"

            testCase "a backend whose kernel set lacks a kernel is refused at creation and disposed"
            <| fun () ->
                let shaders = Lrm.Shaders.Shaders.embedded ()

                let without =
                    { shaders with
                        Blobs = shaders.Blobs |> List.filter (fun b -> b.Kernel <> "rope_apply")
                    }

                let backend =
                    CpuDevice.create
                        { CpuDeviceOptions.Default with
                            MirrorShaderSet = true
                        }
                        (DeviceOptions.For ExecutionBackend.CpuDevice)

                let e =
                    throws<GpuContractException> "a missing kernel" (fun () ->
                        GpuForwardOps.createWith GpuExecOptions.Default backend without tinyWeights.Value (small ())
                        |> ignore)

                Expect.stringContains e.Message "rope_apply" "names the kernel"
                Expect.throwsT<ObjectDisposedException> (fun () -> backend.Info |> ignore) "the backend is disposed"

            testCase "Dispose: every op raises ObjectDisposedException, and releasing a cache afterwards does nothing"
            <| fun () ->
                let model = tinyModel ()
                let g = gpu (small ())
                let cache = KvCache.create g model (Stage.whole model.Config) 4
                let l = Forward.run g model cache 0 [| 1 |] ForwardOptions.all
                g.Dispose()
                Expect.throwsT<ObjectDisposedException> (fun () -> g.Read l |> ignore) "Read"

                Expect.throwsT<ObjectDisposedException>
                    (fun () -> g.BeginForward(call (Stage.whole model.Config) 1 0 TapFilter.Off))
                    "BeginForward"

                (cache :> IDisposable).Dispose()

            testCase "the registry's cpu-device runner equals cpu-f32 and owns its backend"
            <| fun () ->
                let factory =
                    GpuBackendFactory.create
                        ExecutionBackend.CpuDevice
                        "test"
                        GpuExecOptions.Default
                        Lrm.Shaders.Shaders.embedded
                        (fun o -> cpuDevice o.Threads)

                let ids = promptIds "p0"

                let o =
                    { tinyOptions () with
                        MaxContext = 32
                        MaxRows = 13
                    }

                use a = factory.Create tinyWeights.Value o
                use b = CpuBackends.cpuF32.Create tinyWeights.Value o
                Expect.equal a.Backend "cpu-device" "name"
                Expect.equal a.ExecutionClass.Backend ExecutionBackend.CpuDevice "class"
                let fa = a.Forward(ids, TapFilter.All, HeadRows.All)
                let fb = b.Forward(ids, TapFilter.All, HeadRows.All)
                expectSameBits "logits" fa.Logits fb.Logits

                for (n, x), (_, y) in List.zip fa.Taps fb.Taps do
                    expectSameBits n x y

                Expect.equal
                    (a.Greedy(ids, GreedyOptions.parity 8)).Ids
                    (b.Greedy(ids, GreedyOptions.parity 8)).Ids
                    "greedy"

            testCase "Lrm.Gpu.Exec references no concrete backend (ProjectGraph, exec-never-backend)"
            <| fun () ->
                let refs =
                    typeof<GpuOps>.Assembly.GetReferencedAssemblies()
                    |> Array.map _.Name
                    |> Set.ofArray

                for b in
                    [
                        "Lrm.Gpu.CpuDevice"
                        "Lrm.Gpu.D3D12"
                        "Lrm.Gpu.Vulkan"
                        "Lrm.Gpu.D3D12.Interop"
                        "Lrm.Gpu.Vulkan.Interop"
                    ] do
                    Expect.isFalse (refs.Contains b) b

                Expect.isTrue (refs.Contains "Lrm.Gpu.Core") "Lrm.Gpu.Core"
        ]
