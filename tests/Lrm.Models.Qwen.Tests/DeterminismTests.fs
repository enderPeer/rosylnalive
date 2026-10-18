/// The bit-identity proofs M2 owes (BatchInvariance, ForwardOpsAndQwenModel), on the tiny model in both modes: prefill
/// equals token-by-token decode and any chunked prefill, for logits, caches and every tap; the head on the last row
/// equals the last row of the full head; a two-stage split through the host equals one stage; any thread count gives
/// the same bits; a replay after a rewind equals the first run. Each proof is also a golden.compare self-test record
/// (a `check` and no `tap`) in the parity run, so the report lists it.
module Lrm.Models.Qwen.Tests.DeterminismTests

open System
open System.Collections.Generic
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

/// One self-test record: the differing elements of every comparison (bound 0) as one golden.compare `exact` check.
let private exactCheck (check: string) (pairs: (string * IHostTensor * IHostTensor) list) =
    let mutable differing = 0L
    let mutable elements = 0L

    for label, a, b in pairs do
        if not (a.Desc.Shape.Equals b.Desc.Shape) || a.Desc.DType <> b.Desc.DType then
            failwithf
                "%s: %s: %O %s and %O %s"
                check
                label
                a.Desc.Shape
                (DType.name a.Desc.DType)
                b.Desc.Shape
                (DType.name b.Desc.DType)

        let x = a.ToBytes()
        let y = b.ToBytes()
        let size = a.Desc.ElementSize

        for i in 0..size .. x.Length - 1 do
            if not (ReadOnlySpan(x, i, size).SequenceEqual(ReadOnlySpan(y, i, size))) then
                differing <- differing + 1L

        elements <- elements + a.Desc.Shape.Elements

    let r =
        Compare.check
            "exact"
            BoundOp.AtMost
            (float differing)
            0.0
            elements
            [ Attr.str "check" check; Attr.str "kind" "determinism" ]

    for label, a, b in pairs do
        expectSameBits (check + ": " + label) a b

    Expect.isTrue r.Passed check

let private backendOf (ops: IForwardOps<'t>) =
    ExecutionBackend.name ops.ExecutionClass.Backend

/// A fresh whole-model cache of `n` positions.
let private cacheOf (ops: IForwardOps<'t>) (n: int) =
    KvCache.create ops (tinyModel ()) (Stage.whole (tinyModel ()).Config) n

/// The valid rows of a cache, per layer: (label, K rows, V rows).
let private cacheRows (ops: IForwardOps<'t>) (label: string) (c: KvCache<'t>) =
    [
        for l in c.FirstLayer .. c.FirstLayer + c.LayerCount - 1 do
            yield Invariant.format $"{label} K L{l}", rowsOf (ops.Read(c.Keys l)) 0 c.Length
            yield Invariant.format $"{label} V L{l}", rowsOf (ops.Read(c.Values l)) 0 c.Length
    ]

/// Runs `ids` in chunks of the given sizes, recording every tap; returns the logits rows and taps concatenated by
/// position, and the cache.
let private chunked (ops: IForwardOps<'t>) (ids: int[]) (sizes: int list) =
    let model = tinyModel ()
    let cache = cacheOf ops ids.Length
    let logits = ResizeArray<IHostTensor>()
    let taps = Dictionary<string, ResizeArray<IHostTensor>>(StringComparer.Ordinal)
    let mutable pos = 0

    for n in sizes do
        let store = TapStore<'t>(TapFilter.All)

        let l =
            Forward.run ops model cache pos ids[pos .. pos + n - 1] (ForwardOptions.withTaps store)

        logits.Add(ops.Read l)

        for name, x in store.Taps do
            if not (taps.ContainsKey name) then
                taps[name] <- ResizeArray()

            taps[name].Add(ops.Read x)

        pos <- pos + n

    let rowsOfAll (xs: IHostTensor seq) =
        [
            for x in xs do
                for r in 0 .. int x.Desc.Shape.Rows - 1 -> rowsOf x r (r + 1)
        ]

    rowsOfAll logits, (taps |> Seq.map (fun kv -> kv.Key, rowsOfAll kv.Value) |> Map.ofSeq), cache

let private prefillEqualsChunks (ops: IForwardOps<'t>) (prompt: string) (sizes: int list) (what: string) =
    let ids = promptIds prompt
    let pl, pt, pc = chunked ops ids [ ids.Length ]
    let dl, dt, dc = chunked ops ids sizes
    let mode = backendOf ops

    exactCheck
        (Invariant.format $"{mode} prefill = {what} ({prompt}, {ids.Length} tokens: logits, caches, 39 taps)")
        [
            for t in 0 .. ids.Length - 1 do
                yield Invariant.format $"logits row {t}", dl[t], pl[t]
            yield!
                List.zip (cacheRows ops "decode" dc) (cacheRows ops "prefill" pc)
                |> List.map (fun ((l, a), (_, b)) -> l, a, b)
            for KeyValue(name, rows) in pt do
                for t in 0 .. rows.Length - 1 do
                    yield Invariant.format $"{name} row {t}", dt[name][t], rows[t]
        ]

    Expect.equal (KvCache.digest ops dc) (KvCache.digest ops pc) "cache digests"

/// The canonical schedule over two stages, the boundary residual through the host: prefill, then `steps` decode
/// steps feeding back the argmax. Returns every step's logits and both caches.
let private twoStage (ops: IForwardOps<'t>) (ids: int[]) (steps: int) (boundary: int) =
    let model = tinyModel ()
    let stages = Stage.split model.Config [ boundary ]
    let n = Greedy.positions ids.Length steps
    let caches = stages |> List.map (fun s -> KvCache.create ops model s n)
    let mutable next = ids
    let mutable pos = 0
    let logits = ResizeArray<IHostTensor>()
    let ids' = ResizeArray<int>()

    for s in 0 .. steps - 1 do
        let l =
            Forward.runStages
                ops
                model
                stages
                caches
                pos
                next
                (Forward.hostRoundTrip ops)
                { ForwardOptions.last with
                    Step = Some s
                }

        logits.Add(ops.Read l)
        let id = int (ops.Argmax l).[0]
        ids'.Add id
        pos <- pos + next.Length
        next <- [| id |]

    ids'.ToArray(), List.ofSeq logits, caches

/// The same schedule on one stage.
let private oneStage (ops: IForwardOps<'t>) (ids: int[]) (steps: int) =
    let model = tinyModel ()
    let cache = cacheOf ops (Greedy.positions ids.Length steps)
    let mutable next = ids
    let mutable pos = 0
    let logits = ResizeArray<IHostTensor>()
    let ids' = ResizeArray<int>()

    for s in 0 .. steps - 1 do
        let l =
            Forward.run
                ops
                model
                cache
                pos
                next
                { ForwardOptions.last with
                    Step = Some s
                }

        logits.Add(ops.Read l)
        let id = int (ops.Argmax l).[0]
        ids'.Add id
        pos <- pos + next.Length
        next <- [| id |]

    ids'.ToArray(), List.ofSeq logits, cache

/// Two stages on ops planned for the split (`split`) against one stage on ops planned for the whole model (`whole`),
/// both of one class.
let private stagesEqualOne (split: IForwardOps<'t>) (whole: IForwardOps<'t>) =
    let ids = promptIds "p0"
    let steps = 17
    let i2, l2, caches = twoStage split ids steps 1
    let i1, l1, cache = oneStage whole ids steps
    let mode = backendOf whole
    Expect.sequenceEqual i2 i1 "ids"
    let golden = goldenGreedy "p0" (if mode = "cpu-f32" then "torch32" else "torch64p")
    Expect.sequenceEqual i1 golden[0..16] "golden ids"

    let cacheTwo = caches |> List.collect (fun c -> cacheRows split "two-stage" c)

    exactCheck
        (Invariant.format
            $"{mode} two stages (embed + L0 | L1 + head, residual through the host) = one stage (p0: prefill + 16 decode steps, logits and caches)")
        [
            for s in 0 .. steps - 1 do
                yield Invariant.format $"step {s} logits", l2[s], l1[s]
            yield!
                List.zip cacheTwo (cacheRows whole "one-stage" cache)
                |> List.map (fun ((l, a), (_, b)) -> l, a, b)
        ]

/// A decorator that records every op call with its argument rows: the forward's op sequence.
type private RecordingOps<'t>(inner: IForwardOps<'t>, log: ResizeArray<string>) =
    let note (s: string) = log.Add s

    interface IForwardOps<'t> with
        member _.ExecutionClass = inner.ExecutionClass
        member _.DType = inner.DType
        member _.MaxContext = inner.MaxContext
        member _.Rows x = inner.Rows x
        member _.Cols x = inner.Cols x
        member _.BeginForward call = inner.BeginForward call
        member _.EndForward r = inner.EndForward r

        member _.Embed ids =
            note (Invariant.format $"embed {ids.Length}")
            inner.Embed ids

        member _.RmsNorm(x, w, eps) =
            note (Invariant.format $"rmsnorm {inner.Rows x} {w}")
            inner.RmsNorm(x, w, eps)

        member _.MatMul(x, w) =
            note (Invariant.format $"matmul {inner.Rows x} {w}")
            inner.MatMul(x, w)

        member _.Rope(x, h, p) =
            note (Invariant.format $"rope {inner.Rows x} {h} {p}")
            inner.Rope(x, h, p)

        member _.NewCache(s, l, r) = inner.NewCache(s, l, r)
        member _.ReleaseCache c = inner.ReleaseCache c

        member _.KvWrite(c, x, p) =
            note (Invariant.format $"kv_write {inner.Rows x} {p}")
            inner.KvWrite(c, x, p)

        member _.Attention(q, k, v, s, p) =
            note (Invariant.format $"attention {inner.Rows q} {p}")
            inner.Attention(q, k, v, s, p)

        member _.SiluMul(g, u) =
            note "silu_mul"
            inner.SiluMul(g, u)

        member _.Add(a, b) =
            note "add"
            inner.Add(a, b)

        member _.LastRow x =
            note "last_row"
            inner.LastRow x

        member _.Argmax x = inner.Argmax x
        member _.Tap(n, x) = inner.Tap(n, x)
        member _.Read x = inner.Read x
        member _.ReadRows(x, r, n) = inner.ReadRows(x, r, n)
        member _.Upload x = inner.Upload x

    interface IDisposable with
        member _.Dispose() = inner.Dispose()

/// A decorator whose lm_head returns NaN logits.
type private NanHeadOps(inner: IForwardOps<HostTensor<float32>>, head: string) =
    interface IForwardOps<HostTensor<float32>> with
        member _.ExecutionClass = inner.ExecutionClass
        member _.DType = inner.DType
        member _.MaxContext = inner.MaxContext
        member _.Rows x = inner.Rows x
        member _.Cols x = inner.Cols x
        member _.BeginForward call = inner.BeginForward call
        member _.EndForward r = inner.EndForward r
        member _.Embed ids = inner.Embed ids
        member _.RmsNorm(x, w, e) = inner.RmsNorm(x, w, e)

        member _.MatMul(x, w) =
            let y = inner.MatMul(x, w)

            if w = head && inner.Rows x = 1 && y.Cols > 1 then
                y.Data[3] <- Single.NaN

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

let tests =
    testList
        "determinism"
        [
            testCase "prefill equals token-by-token decode: logits, caches and every tap, bit for bit (both modes)"
            <| fun () ->
                do
                    use _ = ParityRun.install ExecutionBackend.CpuF64
                    prefillEqualsChunks (tinyF64 ()) "p2" (List.replicate 70 1) "decode"

                do
                    use _ = ParityRun.install ExecutionBackend.CpuF32
                    prefillEqualsChunks (tinyF32 ()) "p2" (List.replicate 70 1) "decode"

            testCase "prefill equals chunked prefills, across the 64-key block (both modes)"
            <| fun () ->
                do
                    use _ = ParityRun.install ExecutionBackend.CpuF64
                    prefillEqualsChunks (tinyF64 ()) "p2" [ 64; 6 ] "chunks 64 + 6"
                    prefillEqualsChunks (tinyF64 ()) "p2" [ 10; 31; 1; 28 ] "chunks 10 + 31 + 1 + 28"

                do
                    use _ = ParityRun.install ExecutionBackend.CpuF32
                    prefillEqualsChunks (tinyF32 ()) "p2" [ 64; 6 ] "chunks 64 + 6"
                    prefillEqualsChunks (tinyF32 ()) "p2" [ 10; 31; 1; 28 ] "chunks 10 + 31 + 1 + 28"

            testCase "the head on the last row equals the last row of the full head"
            <| fun () ->
                let check (ops: IForwardOps<'t>) =
                    let ids = promptIds "p2"
                    let model = tinyModel ()
                    use c1 = cacheOf ops 70
                    use c2 = cacheOf ops 70
                    // The full head's logits end when the next forward of the stage begins: read them first.
                    let all = ops.Read(Forward.run ops model c1 0 ids ForwardOptions.all)
                    let last = Forward.run ops model c2 0 ids ForwardOptions.last
                    Expect.equal (ops.Rows last) 1 "one row"
                    expectSameBits (backendOf ops) (ops.Read last) (rowsOf all 69 70)

                check (tinyF32 ())
                check (tinyF64 ())

            testCase "two stages through the host equal one stage, prefill and 16 decode steps (both modes)"
            <| fun () ->
                do
                    use _ = ParityRun.install ExecutionBackend.CpuF64
                    stagesEqualOne (CpuOps.f64 tinyWeights.Value (splitOptions [ 1 ])) (tinyF64 ())

                do
                    use _ = ParityRun.install ExecutionBackend.CpuF32
                    stagesEqualOne (CpuOps.f32 tinyWeights.Value (splitOptions [ 1 ])) (tinyF32 ())

            testCase "a stage's taps and boundary residual are the one-stage taps; the boundary is L{b-1}.out"
            <| fun () ->
                use ops = CpuOps.f32 tinyWeights.Value (splitOptions [ 1 ])
                let model = tinyModel ()
                let ids = promptIds "p0"
                let stages = Stage.split model.Config [ 1 ]
                let caches = stages |> List.map (fun s -> KvCache.create ops model s ids.Length)
                let s0 = TapStore(TapFilter.All)
                let s1 = TapStore(TapFilter.All)

                let hidden =
                    Forward.runStage ops model stages[0] caches[0] 0 (StageInput.Ids ids) (ForwardOptions.withTaps s0)

                Forward.runStage
                    ops
                    model
                    stages[1]
                    caches[1]
                    0
                    (StageInput.Hidden(Forward.hostRoundTrip ops hidden))
                    (ForwardOptions.withTaps s1)
                |> ignore

                let _, whole = ForwardTests.prefillTaps (tinyF32 ()) ids
                let names = (s0.Taps @ s1.Taps) |> List.map fst
                Expect.sequenceEqual names (ActivationNames.all 2) "stage 0 then stage 1 emit every tap once"
                Expect.isTrue (s0.Taps |> List.forall (fun (n, _) -> n = "embed.out" || n.StartsWith "L0.")) "stage 0"

                for name, x in s0.Taps @ s1.Taps do
                    expectSameBits name (ops.Read x) whole[name]

                expectSameBits "boundary = L0.out" (ops.Read hidden) whole["L0.out"]

            testCase "Stage.split covers the model in order; bad splits raise"
            <| fun () ->
                let c = Qwen3Config.parse ConfigTests.qwenConfigText
                let s = Stage.split c [ 14 ]

                Expect.equal
                    (s |> List.map (fun s -> s.Index, s.FirstLayer, s.LastLayer))
                    [ 0, 0, 13; 1, 14, 27 ]
                    "Qwen3-0.6B halves"

                Expect.isTrue (s[0].Embeds && not (Stage.hasHead c s[0])) "stage 0 embeds"
                Expect.isTrue (not s[1].Embeds && Stage.hasHead c s[1]) "stage 1 has the head"
                Expect.equal (Stage.split c []) [ Stage.whole c ] "one stage"
                Expect.equal (Stage.split c [ 7; 14; 21 ] |> List.map _.LayerCount) [ 7; 7; 7; 7 ] "four stages"
                Expect.throwsT<ArgumentException> (fun () -> Stage.split c [ 0 ] |> ignore) "boundary 0"
                Expect.throwsT<ArgumentException> (fun () -> Stage.split c [ 28 ] |> ignore) "boundary at the end"
                Expect.throwsT<ArgumentException> (fun () -> Stage.split c [ 14; 14 ] |> ignore) "repeated"

            testCase "phaseOf: decode is a single-token call after position 0, everything else a prefill"
            <| fun () ->
                Expect.equal
                    ([ 13, 0; 1, 0; 1, 13; 6, 64 ] |> List.map (fun (r, p) -> Forward.phaseOf r p))
                    [ "prefill"; "prefill"; "decode"; "prefill" ]
                    "phases"

            testCase "the forward is bit-identical at 1, 7 and 32 threads (both modes)"
            <| fun () ->
                let w = tinyWeights.Value
                let ids = promptIds "p2"

                let run (threads: int) (f64: bool) =
                    let o =
                        { tinyOptions () with
                            Threads = threads
                        }

                    if f64 then
                        ForwardTests.prefillTaps (CpuOps.f64 w o) ids
                    else
                        ForwardTests.prefillTaps (CpuOps.f32 w o) ids

                for f64 in [ false; true ] do
                    let l1, t1 = run 1 f64

                    use _ =
                        ParityRun.install (
                            if f64 then
                                ExecutionBackend.CpuF64
                            else
                                ExecutionBackend.CpuF32
                        )

                    for n in [ 7; 32 ] do
                        let ln, tn = run n f64

                        exactCheck
                            (Invariant.format
                                $"""{(if f64 then "cpu-f64" else "cpu-f32")} 1 thread = {n} threads (p2: logits, 39 taps)""")
                            [ yield "logits", ln, l1; for KeyValue(name, t) in t1 -> name, tn[name], t ]

            testCase "the caches hold exactly the k_rope and v_cache taps"
            <| fun () ->
                let check (ops: IForwardOps<'t>) =
                    let ids = promptIds "p0"
                    let model = tinyModel ()
                    let cache = cacheOf ops ids.Length
                    let store = TapStore<'t>(TapFilter.All)
                    Forward.run ops model cache 0 ids (ForwardOptions.withTaps store) |> ignore
                    let taps = dict [ for n, x in store.Taps -> n, ops.Read x ]
                    Expect.equal cache.Length ids.Length "length"

                    for l in 0..1 do
                        expectSameBits
                            "K"
                            (rowsOf (ops.Read(cache.Keys l)) 0 ids.Length)
                            taps[Invariant.format $"L{l}.attn.k_rope"]

                        expectSameBits
                            "V"
                            (rowsOf (ops.Read(cache.Values l)) 0 ids.Length)
                            taps[Invariant.format $"L{l}.attn.v_cache"]

                check (tinyF32 ())
                check (tinyF64 ())

                use ops64 = tinyF64 ()

                Expect.equal
                    (ops64.NewCache(Stage.whole (tinyModel ()).Config, 0, 4)
                     |> ops64.Read
                     |> _.Desc.DType)
                    DType.F64
                    "the F64 cache is float64"

            testCase "a replay after a rewind equals the first run; the digest sees one changed bit"
            <| fun () ->
                let ops = tinyF32 ()
                let model = tinyModel ()
                let ids = promptIds "p0"
                let cache = cacheOf ops 16
                Forward.run ops model cache 0 ids ForwardOptions.last |> ignore

                let steps (pos0: int) =
                    [
                        for s in 0..2 ->
                            ops.Read(Forward.run ops model cache (pos0 + s) [| 40 + s |] ForwardOptions.last)
                    ]

                let first = steps 13
                let d1 = KvCache.digest ops cache
                Expect.equal cache.Length 16 "length"
                let again = steps 13
                Expect.equal (KvCache.digest ops cache) d1 "replayed cache"

                for a, b in List.zip first again do
                    expectSameBits "replayed logits" a b

                cache.Keys(1).Data[ 16 * 64 - 1 ] <-
                    BitConverter.Int32BitsToSingle(
                        BitConverter.SingleToInt32Bits(cache.Keys(1).Data[16 * 64 - 1]) ^^^ 1
                    )

                Expect.notEqual (KvCache.digest ops cache) d1 "one bit"

            testCase "KvCache.digest matches the contract's vectors K1 to K4 (computed with Python hashlib)"
            <| fun () ->
                // Layers 3 and 4, two valid rows of two columns; rows 2 and 3 hold values the digest must ignore.
                let k3 = [| 1.0; 2.0; 3.0; 4.0; 99.0; 99.0; 99.0; 99.0 |]
                let v3 = [| -1.0; -2.0; -3.0; -4.0; 99.0; 99.0; 99.0; 99.0 |]
                let k4 = [| 0.5; 0.25; 0.0; -0.0; 99.0; 99.0; 99.0; 99.0 |]
                let v4 = [| 8.0; 16.0; 32.0; 64.0; 99.0; 99.0; 99.0; 99.0 |]
                let shape = Shape.matrix 4 2

                let f32 (xs: float[]) =
                    HostTensor.f32 shape (xs |> Array.map float32)

                use c32 =
                    new KvCache<HostTensor<float32>>(3, [| f32 k3; f32 k4 |], [| f32 v3; f32 v4 |], 4, 2, Length = 2)

                use c64 =
                    new KvCache<HostTensor<float>>(
                        3,
                        [| HostTensor.f64 shape k3; HostTensor.f64 shape k4 |],
                        [| HostTensor.f64 shape v3; HostTensor.f64 shape v4 |],
                        4,
                        2,
                        Length = 2
                    )

                let ops32 = tinyF32 ()

                Expect.equal
                    (KvCache.digest ops32 c32).Hex
                    "fe32a706cc6ab5e9638beaa88e0a2b4c9a589195986c4c493bd959b5f5abf893"
                    "K1"

                Expect.equal
                    (KvCache.digest (tinyF64 ()) c64).Hex
                    "a8b5f2a491a1feb61f56e5d9d83be920b69961bc2062350a2127ad8cb386b3c0"
                    "K2"

                c32.Length <- 1

                Expect.equal
                    (KvCache.digest ops32 c32).Hex
                    "6091fb9e00a1d9c26335f9bab1c10d1e157ab531de22bdada3a144d5f9de401f"
                    "K3"

                let empty =
                    KvCache.create ops32 (tinyModel ()) (Stage.whole (tinyModel ()).Config) 1

                Expect.equal
                    (KvCache.digest ops32 empty).Hex
                    "6512ddd1f2a5f8be07b058ced48aa376662e2564eddaecffbae06c89a9591f9b"
                    "K4"

                Expect.throwsT<ArgumentOutOfRangeException> (fun () -> c32.Length <- 5) "beyond MaxPos"

            testCase "the forward's op sequence depends only on T and pos0, never on the values (a static plan)"
            <| fun () ->
                let record (ids: int[]) (pos0: int) =
                    let log = ResizeArray<string>()
                    use ops = new RecordingOps<_>(tinyF32 (), log) :> IForwardOps<_>
                    let model = tinyModel ()
                    let cache = cacheOf ops 32

                    if pos0 > 0 then
                        Forward.run ops model cache 0 (Array.create pos0 3) ForwardOptions.last
                        |> ignore

                    log.Clear()
                    Forward.run ops model cache pos0 ids ForwardOptions.last |> ignore
                    List.ofSeq log

                let a = record [| 1; 2; 3 |] 5
                let b = record [| 200; 17; 9 |] 5
                Expect.sequenceEqual a b "same ops for other values"
                Expect.equal a.Length (1 + 2 * 19 + 3) "embed, 19 calls per layer, last row, final norm, lm_head"
                Expect.notEqual (record [| 1 |] 5) a "T changes the sequence"

            testCase
                "telemetry never changes a value: greedy ids and margins with the no-op facade, a strict sink and the run log"
            <| fun () ->
                let ids = promptIds "p0"

                let run () =
                    Greedy.run
                        (tinyF32 ())
                        (tinyModel ())
                        ids
                        { GreedyOptions.parity 16 with
                            Margins = true
                        }

                let quiet = run ()

                let sink = InMemorySink()

                let loud =
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    run ()

                let logged =
                    use _ = ParityRun.install ExecutionBackend.CpuF32
                    run ()

                Expect.isGreaterThan sink.Records.Length 100 "records"

                for r in [ loud; logged ] do
                    Expect.sequenceEqual r.Ids quiet.Ids "ids"

                    Expect.sequenceEqual
                        (r.Margins |> Array.map BitConverter.DoubleToInt64Bits)
                        (quiet.Margins |> Array.map BitConverter.DoubleToInt64Bits)
                        "margins"

            testCase "the forward's rules: inputs, positions and caches"
            <| fun () ->
                let model = tinyModel ()
                let c = model.Config
                let s1 = (Stage.split c [ 1 ])[1]
                // Planned for the whole model (stage 0) and for layer 1 alone as stage 1.
                use ops =
                    CpuOps.f32
                        tinyWeights.Value
                        { tinyOptions () with
                            Stages = [ Stage.whole c; s1 ]
                        }

                use cache = cacheOf ops 8

                let raises (label: string) (f: unit -> unit) =
                    Expect.throwsT<ForwardException> f label

                raises "a gap after the cache" (fun () ->
                    Forward.run ops model cache 1 [| 1 |] ForwardOptions.last |> ignore)

                raises "past the cache" (fun () ->
                    Forward.run ops model cache 0 (Array.create 9 1) ForwardOptions.last |> ignore)

                raises "no rows" (fun () -> Forward.run ops model cache 0 [||] ForwardOptions.last |> ignore)

                Expect.throwsT<ArgumentOutOfRangeException>
                    (fun () -> Forward.run ops model cache 0 [| 256 |] ForwardOptions.last |> ignore)
                    "an id outside the vocabulary"

                Expect.equal cache.Length 0 "a failed call leaves the cache at pos0"
                let hidden = ops.Upload(HostTensor.zeros<float32> DType.F32 (Shape.matrix 2 64))

                raises "hidden into the first stage" (fun () ->
                    Forward.runStage ops model (Stage.whole c) cache 0 (StageInput.Hidden hidden) ForwardOptions.last
                    |> ignore)

                use c1 = KvCache.create ops model s1 8

                raises "ids into a later stage" (fun () ->
                    Forward.runStage ops model s1 c1 0 (StageInput.Ids [| 1 |]) ForwardOptions.last
                    |> ignore)

                raises "a cache of other layers" (fun () ->
                    Forward.runStage ops model s1 cache 0 (StageInput.Hidden hidden) ForwardOptions.last
                    |> ignore)

                // A residual of twice the width would pass the first RmsNorm as two per-head norms; it is refused
                // before any op runs.
                let wide =
                    ops.Upload(HostTensor.zeros<float32> DType.F32 (Shape.matrix 2 (2 * c.Hidden)))

                let e =
                    throws<ForwardException> "a residual of another width" (fun () ->
                        Forward.runStage ops model s1 c1 0 (StageInput.Hidden wide) ForwardOptions.last
                        |> ignore)

                Expect.stringContains e.Message "128 columns" "the width"
                Expect.equal c1.Length 0 "nothing ran"

                let unknown =
                    throws<ForwardException> "a tap name ActivationNames does not define" (fun () ->
                        Forward.run
                            ops
                            model
                            cache
                            0
                            [| 1 |]
                            (ForwardOptions.withTaps (TapStore(TapFilter.Parse "L0.attn.qrope, logits")))
                        |> ignore)

                Expect.stringContains unknown.Message "'L0.attn.qrope'" "the unknown name"

                Expect.equal
                    (TapFilter.unknown 2 (TapFilter.Parse "L0.bogus, logits, L7.out"))
                    [ "L0.bogus"; "L7.out" ]
                    "unknown"

                Expect.isEmpty (TapFilter.unknown 2 TapFilter.All) "all"

                Expect.throwsT<ArgumentException>
                    (fun () -> KvCache.create ops model (Stage.whole c) 513 |> ignore)
                    "beyond MaxContext"

                Expect.throwsT<ArgumentException>
                    (fun () -> ops.Upload(HostTensor.zeros<float> DType.F64 (Shape.matrix 1 64)) |> ignore)
                    "an F64 residual into F32 ops"

                Expect.throwsT<ForwardException>
                    (fun () -> Greedy.run ops model (Array.create 500 1) (GreedyOptions.parity 20) |> ignore)
                    "a greedy run beyond MaxContext"

            testCase "a NaN in the logits stops greedy decoding with the step"
            <| fun () ->
                use ops = new NanHeadOps(tinyF32 (), (tinyModel ()).Params.LmHead) :> IForwardOps<_>

                let e =
                    throws<ForwardException> "NaN" (fun () ->
                        Greedy.run ops (tinyModel ()) [| 5 |] (GreedyOptions.parity 4) |> ignore)

                Expect.stringContains e.Message "step 0" "the step"
        ]
