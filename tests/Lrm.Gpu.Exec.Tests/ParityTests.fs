/// X1a's acceptance (GpuExecutor contract, "Verification"; BatchInvariance, X1's proof obligation): the tiny model on
/// the executor over a CpuDevice is bit-identical to CpuOps F32 — logits and every tap of a prefill, 16 greedy decode
/// steps, the caches — and on the executor prefill equals decode equals any chunked prefill, the last-row head equals
/// the last row of the full head, two stages through the host equal one stage, and splitting every matmul and
/// attention dispatch into small chunks changes no bit. Each proof is also a golden.compare `exact` check (a `check`
/// and no `tap`), which the parity template lists as a self-test.
module Lrm.Gpu.Exec.Tests.ParityTests

open System
open System.Collections.Generic
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Gpu.Exec.Tests.Common

/// One self-test record: the differing elements of every comparison (bound 0) as one golden.compare `exact` check,
/// then the comparisons themselves, which fail naming the first differing element.
let exactCheck (check: string) (pairs: (string * IHostTensor * IHostTensor) list) =
    let mutable differing = 0L
    let mutable elements = 0L

    for label, a, b in pairs do
        if a.Desc.Shape.Equals b.Desc.Shape && a.Desc.DType = b.Desc.DType then
            let x = a.ToBytes()
            let y = b.ToBytes()
            let size = a.Desc.ElementSize

            for i in 0..size .. x.Length - 1 do
                if not (ReadOnlySpan(x, i, size).SequenceEqual(ReadOnlySpan(y, i, size))) then
                    differing <- differing + 1L

            elements <- elements + a.Desc.Shape.Elements
        else
            differing <- differing + 1L

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

/// The canonical schedule with every tap: a prefill of `ids` (head on every row), then `steps` - 1 single-token
/// decode steps feeding back the argmax of the last row.
type Schedule =
    {
        Prefill: IHostTensor
        Taps: (string * IHostTensor) list
        Steps: IHostTensor list
        Ids: int[]
        Cache: (string * IHostTensor) list
        Digest: ContentId
    }

let schedule (ops: IForwardOps<'t>) (ids: int[]) (steps: int) =
    let model = tinyModel ()

    use cache =
        KvCache.create ops model (Stage.whole model.Config) (Greedy.positions ids.Length steps)

    let store = TapStore<'t>(TapFilter.All)
    let l0 = Forward.run ops model cache 0 ids (ForwardOptions.withTaps store)
    let prefill = ops.Read l0
    let taps = store.Taps |> List.map (fun (n, x) -> n, ops.Read x)
    let a = ops.Argmax l0
    let generated = List<int>([ int a[a.Length - 1] ])
    let logits = List<IHostTensor>()

    for s in 1 .. steps - 1 do
        let pos = ids.Length + s - 1
        let l = Forward.run ops model cache pos [| generated[s - 1] |] ForwardOptions.last
        logits.Add(ops.Read l)
        generated.Add(lastArgmax ops l)

    let rows =
        [
            for l in cache.FirstLayer .. cache.FirstLayer + cache.LayerCount - 1 do
                yield Invariant.format $"K L{l}", rowsOf (ops.Read(cache.Keys l)) 0 cache.Length
                yield Invariant.format $"V L{l}", rowsOf (ops.Read(cache.Values l)) 0 cache.Length
        ]

    {
        Prefill = prefill
        Taps = taps
        Steps = List.ofSeq logits
        Ids = generated.ToArray()
        Cache = rows
        Digest = KvCache.digest ops cache
    }

/// Every comparison of two schedules.
let pairsOf (a: Schedule) (b: Schedule) =
    [
        yield "prefill logits", a.Prefill, b.Prefill
        for (n, x), (m, y) in List.zip a.Taps b.Taps do
            if n <> m then
                failwithf "tap order differs: %s and %s" n m

            yield n, x, y
        for i, (x, y) in List.indexed (List.zip a.Steps b.Steps) do
            yield Invariant.format $"step {i + 1} logits", x, y
        for (n, x), (_, y) in List.zip a.Cache b.Cache do
            yield n, x, y
    ]

/// Runs `ids` in chunks of the given sizes with every tap; returns the logits rows and taps per position, and the
/// cache rows.
let chunked (ops: IForwardOps<'t>) (ids: int[]) (sizes: int list) =
    let model = tinyModel ()
    use cache = KvCache.create ops model (Stage.whole model.Config) ids.Length
    let logits = List<IHostTensor>()
    let taps = Dictionary<string, List<IHostTensor>>(StringComparer.Ordinal)
    let mutable pos = 0

    for n in sizes do
        let store = TapStore<'t>(TapFilter.All)

        let l =
            Forward.run ops model cache pos ids[pos .. pos + n - 1] (ForwardOptions.withTaps store)

        for r in 0 .. n - 1 do
            logits.Add(rowsOf (ops.Read l) r (r + 1))

        for name, x in store.Taps do
            if not (taps.ContainsKey name) then
                taps[name] <- List()

            let h = ops.Read x

            for r in 0 .. n - 1 do
                taps[name].Add(rowsOf h r (r + 1))

        pos <- pos + n

    let cacheRows =
        [
            for l in 0 .. model.Config.Layers - 1 do
                yield Invariant.format $"K L{l}", rowsOf (ops.Read(cache.Keys l)) 0 cache.Length
                yield Invariant.format $"V L{l}", rowsOf (ops.Read(cache.Values l)) 0 cache.Length
        ]

    List.ofSeq logits, (taps |> Seq.map (fun kv -> kv.Key, List.ofSeq kv.Value) |> List.ofSeq), cacheRows

let private chunkPairs (label: string) (a, ta, ca) (b, tb, cb) =
    [
        for i, (x, y) in List.indexed (List.zip a b) do
            yield Invariant.format $"{label} logits row {i}", x, y
        for (n, xs), (m, ys) in List.zip ta tb do
            if n <> m then
                failwithf "tap order differs: %s and %s" n m

            for i, (x, y) in List.indexed (List.zip xs ys) do
                yield Invariant.format $"{label} {n} row {i}", x, y
        for (n, x), (_, y) in List.zip ca cb do
            yield label + " " + n, x, y
    ]

/// A matmul budget that splits every tiny projection into dispatches of a few rows, and an attention budget of 7
/// query-key pairs (at most 3 queries from position 0, 1 query further on).
let tinyChunks =
    { GpuExecOptions.Default with
        MatMulMacsPerDispatch = 40000L
        AttentionPairsPerDispatch = 7L
    }

let tests =
    testList
        "acceptance"
        [
            for prompt in [ "p0"; "p1" ] do
                testCase (
                    Invariant.format $"{prompt}: prefill with every tap and 16 greedy steps on cpu-device = cpu-f32"
                )
                <| fun () ->
                    let ids = promptIds prompt
                    use g = gpu (tinyOptions ())
                    use c = cpu (tinyOptions ())
                    let a = schedule g ids 16
                    let b = schedule c ids 16
                    Expect.equal a.Taps.Length 39 "39 taps"
                    Expect.equal a.Ids b.Ids "greedy ids"
                    Expect.equal a.Digest b.Digest "cache digests"

                    exactCheck
                        (Invariant.format
                            $"cpu-device = cpu-f32 ({prompt}, {ids.Length} tokens + 15 decode steps: logits, 39 taps, caches)")
                        (pairsOf a b)

            testCase "greedy decoding on cpu-device: 16 steps of p0 and p1 equal cpu-f32 and the torch32 golden"
            <| fun () ->
                for prompt in [ "p0"; "p1" ] do
                    let ids = promptIds prompt

                    let o =
                        { tinyOptions () with
                            Taps = TapFilter.Off
                        }

                    use g = gpu o
                    use c = cpu o
                    let a = Greedy.run g (tinyModel ()) ids (GreedyOptions.parity 16)
                    let b = Greedy.run c (tinyModel ()) ids (GreedyOptions.parity 16)
                    Expect.equal a.Ids b.Ids (prompt + ": cpu-device = cpu-f32")

                    Expect.equal
                        a.Ids
                        (Array.truncate 16 (goldenGreedy prompt "torch32"))
                        (prompt + ": the torch32 golden's first 16")

            testCase "prefill = decode = chunked prefill on cpu-device, and = cpu-f32 (p2, 70 tokens)"
            <| fun () ->
                let ids = promptIds "p2"
                use g = gpu (tinyOptions ())
                let prefill = chunked g ids [ ids.Length ]

                for label, sizes in
                    [
                        "decode", List.replicate ids.Length 1
                        "64 + 6", [ 64; 6 ]
                        "10 + 31 + 1 + 28", [ 10; 31; 1; 28 ]
                    ] do
                    exactCheck
                        (Invariant.format $"cpu-device prefill = {label} (p2, 70 tokens: logits, caches, 39 taps)")
                        (chunkPairs label (chunked g ids sizes) prefill)

                use c = cpu (tinyOptions ())

                exactCheck
                    "cpu-device = cpu-f32 (p2, 70-token prefill: logits, caches, 39 taps)"
                    (chunkPairs "prefill" prefill (chunked c ids [ ids.Length ]))

            testCase "the last-row head equals the last row of the full head (p2)"
            <| fun () ->
                let ids = promptIds "p2"
                let model = tinyModel ()
                use g = gpu (tinyOptions ())

                let run (head: HeadRows) =
                    use cache = KvCache.create g model (Stage.whole model.Config) ids.Length
                    let l = Forward.run g model cache 0 ids { ForwardOptions.all with Head = head }
                    g.Read l

                let all = run HeadRows.All
                let last = run HeadRows.Last
                Expect.equal (int last.Desc.Shape.Rows) 1 "one row"

                exactCheck
                    "cpu-device last-row head = last row of the full head (p2)"
                    [ "logits", last, rowsOf all (ids.Length - 1) ids.Length ]

            testCase "two stages through the host = one stage on cpu-device, and = cpu-f32 (p0, prefill + 16 steps)"
            <| fun () ->
                let ids = promptIds "p0"
                let model = tinyModel ()

                let twoStage (ops: IForwardOps<'t>) =
                    let stages = Stage.split model.Config [ 1 ]
                    let n = Greedy.positions ids.Length 16
                    let caches = stages |> List.map (fun s -> KvCache.create ops model s n)
                    let logits = List<IHostTensor>()
                    let mutable next = ids
                    let mutable pos = 0

                    for _ in 0..15 do
                        let l =
                            Forward.runStages
                                ops
                                model
                                stages
                                caches
                                pos
                                next
                                (Forward.hostRoundTrip ops)
                                ForwardOptions.last

                        logits.Add(ops.Read l)
                        pos <- pos + next.Length
                        next <- [| lastArgmax ops l |]

                    let digests = caches |> List.map (KvCache.digest ops)

                    for c in caches do
                        (c :> IDisposable).Dispose()

                    List.ofSeq logits, digests

                let oneStage (ops: IForwardOps<'t>) =
                    use cache =
                        KvCache.create ops model (Stage.whole model.Config) (Greedy.positions ids.Length 16)

                    let logits = List<IHostTensor>()
                    let mutable next = ids
                    let mutable pos = 0

                    for _ in 0..15 do
                        let l = Forward.run ops model cache pos next ForwardOptions.last
                        logits.Add(ops.Read l)
                        pos <- pos + next.Length
                        next <- [| lastArgmax ops l |]

                    List.ofSeq logits

                use g2 = gpu (splitOptions [ 1 ])
                use c2 = cpu (splitOptions [ 1 ])
                use g1 = gpu (tinyOptions ())
                let a, da = twoStage g2
                let b, db = twoStage c2
                let one = oneStage g1
                Expect.equal da db "stage cache digests: cpu-device = cpu-f32"

                exactCheck
                    "cpu-device two stages = one stage = cpu-f32 two stages (p0, prefill + 15 steps)"
                    [
                        for i in 0 .. a.Length - 1 do
                            yield Invariant.format $"step {i} two = one", a[i], one[i]
                            yield Invariant.format $"step {i} cpu-device = cpu-f32", a[i], b[i]
                    ]

            testCase "small dispatch budgets split every matmul and attention and change no bit (p2)"
            <| fun () ->
                let ids = promptIds "p2"
                use small = gpuWith tinyChunks (tinyOptions ())
                use plain = gpuWith GpuExecOptions.Default (tinyOptions ())
                let a = schedule (small :> IForwardOps<GpuTensor>) ids 16
                let b = schedule (plain :> IForwardOps<GpuTensor>) ids 16

                Expect.isGreaterThan
                    small.Dispatches
                    (plain.Dispatches + 300L)
                    "the prefill splits into many more dispatches"

                exactCheck "cpu-device with split dispatches = without (p2, prefill + 15 steps)" (pairsOf a b)

            testCase "the CpuDevice thread count changes no bit (p2)"
            <| fun () ->
                let ids = promptIds "p2"

                let on (threads: int) =
                    use ops =
                        GpuForwardOps.createWith
                            GpuExecOptions.Default
                            (cpuDevice threads)
                            (Lrm.Shaders.Shaders.embedded ())
                            tinyWeights.Value
                            (tinyOptions ())

                    schedule (ops :> IForwardOps<GpuTensor>) ids 4

                exactCheck "cpu-device 1 thread = 8 threads (p2, prefill + 3 steps)" (pairsOf (on 1) (on 8))
        ]
