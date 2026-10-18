/// K2's matmul_bf16w measurements on a GPU backend, outside the per-case harness:
///
/// - **Throughput** (bench tier): every Qwen3-0.6B weight shape at m from 1 (decode) to 4,000 (the longest prefill
///   chunk), lm_head up to 1,024 rows. The sweep starts after `WarmUpMs` of load and runs back to back; each point is
///   one warm-up dispatch and `Reps` timed ones in one command list, and the fastest decides (warm). Records: the
///   backend's `gpu.dispatch` intervals, and one `gpu.matmul.throughput` gauge (flop/s) and one `gpu.matmul.bandwidth`
///   gauge (B/s) per point, from the FLOP registry's cost of the matmul (OpCost.ofOp, Op.MatMul: 2mnk and the minimum
///   traffic), with `kernel`, `shape`, `m`, `n`, `k`, `time_ns` (warm), `budget_ns` and `x1_dispatch` (whether X1
///   dispatches this many rows, ADR K2-0001).
/// - **Cold dispatches** (bench tier): the points closest to the budget (lm_head at every m, the projections at
///   m = 4,000) are also timed, after the whole warm sweep, as one dispatch alone after the GPU has idled `IdleMs`,
///   so it starts from idle clocks, as a conformance case or X1's first dispatch after host work does. Whether the
///   clocks have dropped after the gap varies, so each point takes `ColdSamples` such dispatches and keeps the
///   slowest. The gauges carry it as `cold_ns`, with `cold_fast_ns`, `cold_samples` and `idle_ms`. The budget is
///   judged on the slower of warm and cold, for the points X1 dispatches (`dispatched`: lm_head only up to
///   `LmHeadMaxRows`).
/// - **Batch invariance across calls** (gpu tier): per shape, one activation matrix and one weight; rows computed by
///   an m = 512 call, an m = 37 call and m = 1 calls (views of the same rows) must be bit-identical, and equal to
///   CpuRef F32 (Rows.dotBf16) bit for bit as measured on the class. Records: `gpu.matmul.batch_invariance` and
///   `gpu.matmul.cpuref_exact` checks.
/// - **Arithmetic probes** (both): whether the class's mad is fused (one rounding per step, as CpuRef's
///   FusedMultiplyAdd) and whether the steps run in increasing p, each from a two- or three-element dot product whose
///   result tells them apart; `gpu.matmul.probe` checks with `probe`, `expected` and `result` (informational: the
///   bound is KernelBounds, not these). The fp32 denormal probe of GpuBackendInterface runs first, so the run log has
///   the class's `gpu.fp32_denormals`.
///
/// `gpu ()` (the cross-call check and the probes) and `bench ()` (all of it) are TestTiers lists, so Lrm.Testing's
/// runner holds the backend's mutex; bench.fsx runs them until the conformance exe's tiers list them. `runOn` writes a
/// run log of kind `kernels` that `lrm report` renders, and report.fsx renders templates/matmul.md from the same log.
/// This module has no `cases` value, so Discovery does not read it.
module Lrm.Kernels.Conformance.Cases.MatmulBench

open System
open System.Reflection
open System.Runtime.InteropServices
open System.Threading.Tasks
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Telemetry
open Lrm.Testing
open Lrm.Tensor.CpuReference
open Lrm.Kernels.Conformance

/// A Qwen3-0.6B weight shape: n output features of k inputs.
type Shape = { Name: string; N: int; K: int }

let shapes =
    [
        { Name = "q_proj"; N = 2048; K = 1024 }
        { Name = "kv_proj"; N = 1024; K = 1024 }
        { Name = "o_proj"; N = 1024; K = 2048 }
        {
            Name = "gate_up_proj"
            N = 3072
            K = 1024
        }
        {
            Name = "down_proj"
            N = 1024
            K = 3072
        }
        {
            Name = "lm_head"
            N = 151936
            K = 1024
        }
    ]

[<Literal>]
let private MaxM = 4000

[<Literal>]
let private Budget = 100_000_000L

/// The most rows X1 gives one lm_head dispatch (ADR K2-0001). From idle clocks a dispatch can take about 18 times its
/// warm time on the RTX 4080: 64 rows about 23 ms, a quarter of the budget, as the projections' 4,000-row chunks; 128
/// rows up to 50 ms; 256 rows nearly all of it. The bench measures past this cap but judges the budget only up to it.
[<Literal>]
let LmHeadMaxRows = 64

/// How long the warm sweep's warm-up keeps the GPU busy first, so its clocks are up.
[<Literal>]
let WarmUpMs = 1000

/// How long the GPU idles before a cold dispatch. On the RTX 4080 a dispatch after 1 s idle still ran at full clocks
/// and one after 3 s did not (ADR K2-0001).
[<Literal>]
let IdleMs = 3000

/// Cold dispatches per point, each after its own idle gap; the slowest is the point's cold time.
[<Literal>]
let ColdSamples = 2

/// The m values measured per shape: decode, small batches, prefill chunks up to 4,000; lm_head up to 1,024 rows
/// (its 1,024 x 151,936 output is 622 MB), past X1's cap of LmHeadMaxRows.
let rowsOf (s: Shape) =
    if s.Name = "lm_head" then
        [ 1; 4; 16; LmHeadMaxRows; 128; 256; 512; 1024 ]
    else
        [ 1; 4; 16; 37; 64; 128; 256; 512; 1024; 2048; MaxM ]

/// The m values also timed cold: lm_head at every m, the projections at the longest prefill chunk.
let coldRowsOf (s: Shape) =
    if s.Name = "lm_head" then rowsOf s else [ MaxM ]

/// Whether X1 dispatches m rows of this shape (ADR K2-0001): the projections up to 4,000 rows, lm_head up to
/// LmHeadMaxRows. Only these points are held to the budget.
let dispatched (s: Shape) (m: int) =
    if s.Name = "lm_head" then m <= LmHeadMaxRows else m <= MaxM

/// Timed dispatches per point (after one warm-up).
[<Literal>]
let Reps = 5

/// The rows of the cross-call check, alone (m = 1); those below 37 are also compared with the m = 37 call.
let invarianceRows = [ 0; 1; 15; 16; 36; 63; 64; 100; 511 ]

/// Deterministic data by element index (SplitMix64 of seed and index), as the case data of Cases/matmul.
module private Data =
    let private word (seed: uint64) (i: int64) =
        SplitMix64.Mix(seed + uint64 i * SplitMix64.Gamma)

    let f32 (seed: uint64) (n: int64) : float32[] =
        let a = Array.zeroCreate<float32>(int n)

        Parallel.For(
            0,
            8,
            fun part ->
                let lo = int (int64 part * n / 8L)
                let hi = int (int64 (part + 1) * n / 8L)

                for i in lo .. hi - 1 do
                    let u = uint32 (word seed (int64 i) >>> 40)
                    a[i] <- float32 (int u - (1 <<< 23)) / 8388608.0f
        )
        |> ignore

        a

    let bf16 (seed: uint64) (n: int64) : uint16[] =
        let h = Array.zeroCreate<uint16>(int n)

        Parallel.For(
            0,
            8,
            fun part ->
                let lo = int (int64 part * n / 8L)
                let hi = int (int64 (part + 1) * n / 8L)

                for i in lo .. hi - 1 do
                    let z = word seed (int64 i)
                    let sign = uint16 ((z >>> 63) <<< 15)
                    let exponent = uint16 (126UL - (z >>> 32) % 8UL)
                    h[i] <- sign ||| (exponent <<< 7) ||| uint16 (z &&& 0x7FUL)
        )
        |> ignore

        h

/// One throughput point: the faster of the timed dispatches, and the FLOP registry's work and traffic.
type Point =
    {
        Shape: Shape
        M: int
        /// The fastest of the warm dispatches.
        BestNs: int64
        MedianNs: int64
        /// The slowest of ColdSamples dispatches, each alone after the GPU idled IdleMs, for the points of coldRowsOf.
        ColdNs: int64 option
        /// The fastest of those cold dispatches.
        ColdFastNs: int64 option
        Flops: int64
        Bytes: int64
    }

    /// The time the budget is judged on: the slower of warm and cold.
    member p.BudgetNs = max p.BestNs (defaultArg p.ColdNs 0L)

    member p.FlopsPerSecond = float p.Flops / (float p.BestNs / 1e9)
    member p.BytesPerSecond = float p.Bytes / (float p.BestNs / 1e9)

/// One cross-call check of a shape.
type Invariance =
    {
        Shape: Shape
        /// Rows compared (each alone, and inside the m = 512 call; those below 37 also inside the m = 37 call).
        Rows: int
        /// Elements that differ between the calls (0 when batch-invariant).
        Differences: int64
        /// Elements of the m = 1 rows that differ from CpuRef F32's bits.
        CpuRefDifferences: int64
        Detail: string
    }

/// A dot product whose result tells two arithmetic choices apart (m = n = 1).
type Probe =
    {
        Name: string
        X: uint32[]
        W: uint16[]
        /// CpuRef F32's bits: one fused multiply-add per step, in increasing p, from +0.
        Expected: uint32
        /// The bits of the other choice.
        Other: uint32
        Doc: string
    }

/// The arithmetic probes. `fused`: x = (-(1 + 2^-7 + 2^-23), 1 + 2^-23), w = (1, 1 + 2^-7); the second product is
/// 1 + 2^-7 + 2^-23 + 2^-30, which needs 31 bits, so a fused step gives 2^-30 and a rounded product gives 0. `order`:
/// x = (1, 2^24, -2^24), w = (1, 1, 1); in increasing p, 1 + 2^24 rounds to 2^24 (ties to even) and the sum is +0; in
/// decreasing p it is 1.
let probes =
    [
        {
            Name = "fused"
            X = [| 0xBF810001u; 0x3F800001u |]
            W = [| 0x3F80us; 0x3F81us |]
            Expected = 0x30800000u
            Other = 0x00000000u
            Doc = "mad is one fused multiply-add per step (2^-30), not a rounded product and a sum (0)"
        }
        {
            Name = "order"
            X = [| 0x3F800000u; 0x4B800000u; 0xCB800000u |]
            W = [| 0x3F80us; 0x3F80us; 0x3F80us |]
            Expected = 0x00000000u
            Other = 0x3F800000u
            Doc = "the steps run in increasing p (+0), not in decreasing p (1)"
        }
    ]

type private Planned =
    {
        Heap: IGpuHeap
        X: TensorToken
        Weights: Map<string, TensorToken>
        Y: TensorToken
        YFull: TensorToken
        Y37: TensorToken
        /// Per shape and invariance row: the x row view and the y row view.
        RowViews: Map<string * int, TensorToken * TensorToken>
        Table: SealedTable
        Inputs: Map<string, uint16[]>
        XData: float32[]
        /// The fp32 denormal probe's input and output (scale_f32).
        DenormalX: TensorToken
        DenormalY: TensorToken
        /// The arithmetic probes with their x, w and y (m = n = 1).
        Probes: (Probe * TensorToken * TensorToken * TensorToken) list
    }

let private maxN = shapes |> List.map _.N |> List.max
let private maxK = shapes |> List.map _.K |> List.max

/// Plans and fills the heap: one x [4000, 3072] (Activations), the six weights (Weights), the measurement output
/// (Scratch, the largest m * n) and the cross-call outputs (Scratch), with one x and one y view per invariance row.
let private plan (backend: IGpuBackend) =
    let planner = HeapPlanner()
    let xs = int64 MaxM * int64 maxK
    let px = planner.Allocate(ArenaKind.Activations, 4L * xs)

    let pw =
        shapes
        |> List.map (fun s -> s.Name, planner.Allocate(ArenaKind.Weights, 2L * int64 s.N * int64 s.K))

    let yElements =
        shapes
        |> List.map (fun s -> int64 (List.max (rowsOf s)) * int64 s.N)
        |> List.max

    let py = planner.Allocate(ArenaKind.Scratch, 4L * yElements)
    let pFull = planner.Allocate(ArenaKind.Scratch, 4L * 512L * int64 maxN)
    let p37 = planner.Allocate(ArenaKind.Scratch, 4L * 37L * int64 maxN)

    let pRows =
        planner.Allocate(ArenaKind.Scratch, 4L * int64 invarianceRows.Length * int64 maxN)

    let denormalN = int64 Fp32Denormals.probeInputs.Length
    let pdx = planner.Allocate(ArenaKind.Scratch, 4L * denormalN)
    let pdy = planner.Allocate(ArenaKind.Scratch, 4L * denormalN)

    let pProbes =
        probes
        |> List.map (fun pr ->
            pr,
            planner.Allocate(ArenaKind.Scratch, 4L * int64 pr.X.Length),
            planner.Allocate(ArenaKind.Scratch, Heap.alignUp 4L (2L * int64 pr.W.Length)),
            planner.Allocate(ArenaKind.Scratch, 4L))

    let layout = planner.Build()
    let tt = TensorTable layout
    let x = tt.Add("k2.bench/x", px, DType.F32, [ xs ], TensorFlags.ReadOnly)

    let weights =
        pw
        |> List.map (fun (name, p) ->
            let s = shapes |> List.find (fun s -> s.Name = name)
            name, tt.Add("k2.bench/w." + name, p, DType.BF16, [ int64 s.N; int64 s.K ], TensorFlags.ReadOnly))
        |> Map.ofList

    let y = tt.Add("k2.bench/y", py, DType.F32, [ yElements ], TensorFlags.None)

    let yFull =
        tt.Add("k2.bench/y512", pFull, DType.F32, [ 512L * int64 maxN ], TensorFlags.None)

    let y37 =
        tt.Add("k2.bench/y37", p37, DType.F32, [ 37L * int64 maxN ], TensorFlags.None)

    let yRows =
        tt.Add("k2.bench/yrows", pRows, DType.F32, [ int64 invarianceRows.Length * int64 maxN ], TensorFlags.None)

    let views =
        [
            for s in shapes do
                for idx, r in List.indexed invarianceRows do
                    let xv =
                        tt.AddView(Invariant.format $"k2.bench/x.{s.Name}.r{r}", x, [ int64 s.K ], int64 r * int64 s.K)

                    let yv =
                        tt.AddView(
                            Invariant.format $"k2.bench/y.{s.Name}.r{r}",
                            yRows,
                            [ int64 s.N ],
                            int64 idx * int64 s.N
                        )

                    (s.Name, r), (xv, yv)
        ]
        |> Map.ofList

    let dx =
        tt.Add("k2.bench/denormal.x", pdx, DType.F32, [ denormalN ], TensorFlags.ReadOnly)

    let dy =
        tt.Add("k2.bench/denormal.y", pdy, DType.F32, [ denormalN ], TensorFlags.None)

    let probeTokens =
        pProbes
        |> List.map (fun (pr, px, pw, py) ->
            pr,
            tt.Add("k2.bench/probe." + pr.Name + ".x", px, DType.F32, [ int64 pr.X.Length ], TensorFlags.ReadOnly),
            tt.Add("k2.bench/probe." + pr.Name + ".w", pw, DType.BF16, [ int64 pr.W.Length ], TensorFlags.ReadOnly),
            tt.Add("k2.bench/probe." + pr.Name + ".y", py, DType.F32, [ 1L ], TensorFlags.None))

    let table = tt.Seal()
    let heap = backend.CreateHeap layout
    heap.WriteTable table

    let upload (p: Placement) (words: uint32[]) =
        heap.Upload(layout.SlotOf(p.Arena, p.Page), p.Offset, MemoryMarshal.AsBytes(ReadOnlySpan words))

    upload pdx Fp32Denormals.probeInputs

    for pr, px, pw, _ in pProbes do
        upload px pr.X
        // bf16 pairs in u32 words, element 2e in the low half; an odd count is padded with +0.
        upload
            pw
            (Array.init ((pr.W.Length + 1) / 2) (fun e ->
                let hi =
                    if 2 * e + 1 < pr.W.Length then
                        uint32 pr.W[2 * e + 1]
                    else
                        0u

                uint32 pr.W[2 * e] ||| (hi <<< 16)))

    let xData = Data.f32 0x4B32424E43480000UL xs
    heap.Upload(layout.SlotOf(px.Arena, px.Page), px.Offset, MemoryMarshal.AsBytes(ReadOnlySpan xData))

    let inputs =
        pw
        |> List.mapi (fun i (name, p) ->
            let s = shapes |> List.find (fun s -> s.Name = name)
            let w = Data.bf16 (0x4B32424E43480100UL + uint64 i) (int64 s.N * int64 s.K)
            heap.Upload(layout.SlotOf(p.Arena, p.Page), p.Offset, MemoryMarshal.AsBytes(ReadOnlySpan w))
            name, w)
        |> Map.ofList

    {
        Heap = heap
        X = x
        Weights = weights
        Y = y
        YFull = yFull
        Y37 = y37
        RowViews = views
        Table = table
        Inputs = inputs
        XData = xData
        DenormalX = dx
        DenormalY = dy
        Probes = probeTokens
    }

let private args (x: TensorToken) (w: TensorToken) (y: TensorToken) (m: int) (s: Shape) : Kernels.MatmulBf16wArgs =
    {
        X = x
        W = w
        Y = y
        M = uint32 m
        N = uint32 s.N
        K = uint32 s.K
    }

/// The dispatch times of one completion, in submission order.
let private dispatchNs (c: Completion) =
    c.Timers
    |> List.filter (fun t -> t.Name = GpuTelemetry.Dispatch)
    |> List.map _.DurationNs

/// Keeps the GPU busy for WarmUpMs with q_proj at m = 4,000, so the warm sweep starts at full clocks.
let private warmUp (backend: IGpuBackend) (p: Planned) =
    let s = shapes.Head
    let a = args p.X p.Weights[s.Name] p.Y MaxM s
    let clock = Diagnostics.Stopwatch.StartNew()

    while clock.ElapsedMilliseconds < int64 WarmUpMs do
        let commands = backend.Begin()

        for i in 0..19 do
            if i > 0 then
                commands.Barrier()

            Kernels.MatmulBf16w.dispatch commands a

        backend.Wait(backend.Submit commands) |> ignore

/// ColdSamples dispatches, each alone and submitted after the GPU has idled IdleMs (nothing of this process runs on it
/// meanwhile), sorted.
let private measureCold (backend: IGpuBackend) (p: Planned) (s: Shape) (m: int) : int64 list =
    [
        for _ in 1..ColdSamples do
            Threading.Thread.Sleep IdleMs
            let commands = backend.Begin()
            Kernels.MatmulBf16w.dispatch commands (args p.X p.Weights[s.Name] p.Y m s)

            match dispatchNs (backend.Wait(backend.Submit commands)) with
            | [ t ] -> t
            | ts -> failwithf "a cold dispatch gave %d dispatch timers (TimeDispatches is off?)" ts.Length
    ]
    |> List.sort

/// One point of the warm sweep: one warm-up dispatch and Reps timed ones in one command list.
let private measure (backend: IGpuBackend) (p: Planned) (s: Shape) (m: int) : Point =
    let commands = backend.Begin()
    let a = args p.X p.Weights[s.Name] p.Y m s

    for i in 0..Reps do
        if i > 0 then
            commands.Barrier()

        Kernels.MatmulBf16w.dispatch commands a

    let times = dispatchNs (backend.Wait(backend.Submit commands))

    if times.Length <> Reps + 1 then
        failwithf
            "%s m = %d: %d dispatch timers for %d dispatches (TimeDispatches is off?)"
            s.Name
            m
            times.Length
            (Reps + 1)

    let timed = times |> List.tail |> List.sort

    // The FLOP registry's cost (RunLog contract): 2mnk, and every element of x, w and y once.
    let cost =
        OpCost.ofOp (Op.MatMul(int64 m, int64 s.N, int64 s.K, DType.F32, DType.BF16, DType.F32))

    {
        Shape = s
        M = m
        BestNs = timed.Head
        MedianNs = timed[timed.Length / 2]
        ColdNs = None
        ColdFastNs = None
        Flops = cost.Flops
        Bytes = cost.BytesMoved
    }

/// The point's gauges: throughput and traffic over the warm time, with the cold time when it was measured.
let private record (backend: IGpuBackend) (point: Point) =
    let s = point.Shape

    let attrs =
        [
            Attr.str TelemetryKeys.Kernel Kernels.MatmulBf16w.Name
            Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend.ExecutionClass.Backend)
            Attr.str "shape" s.Name
            Attr.int "m" point.M
            Attr.int "n" s.N
            Attr.int "k" s.K
            Attr.int64 "time_ns" point.BestNs
            Attr.int64 "median_ns" point.MedianNs
            Attr.int "reps" Reps
            Attr.int64 "budget_ns" Budget
            Attr.bool "x1_dispatch" (dispatched s point.M)
            match point.ColdNs, point.ColdFastNs with
            | Some c, Some f ->
                Attr.int64 "cold_ns" c
                Attr.int64 "cold_fast_ns" f
                Attr.int "cold_samples" ColdSamples
                Attr.int "idle_ms" IdleMs
            | _ -> ()
        ]

    Telemetry.gauge "gpu.matmul.throughput" point.FlopsPerSecond "flop/s" attrs
    Telemetry.gauge "gpu.matmul.bandwidth" point.BytesPerSecond "B/s" attrs

let private download (p: Planned) (token: TensorToken) (elements: int64) : uint32[] =
    let row = p.Table.Row token
    let bytes = Array.zeroCreate<byte>(int (4L * elements))
    p.Heap.Download(row.Slot, int64 row.ByteOffset, Span bytes)
    MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()

/// A signalling-NaN word no matmul of finite inputs produces: the outputs of the cross-call check hold it before the
/// dispatches, so a word still holding it was never written.
[<Literal>]
let private Poison = 0x7FBADBADu

let private poison (p: Planned) (token: TensorToken) (elements: int64) =
    let row = p.Table.Row token
    let words = Array.create (int elements) Poison
    p.Heap.Upload(row.Slot, int64 row.ByteOffset, MemoryMarshal.AsBytes(ReadOnlySpan words))

/// CpuRef F32's bits of row r of x times w (Rows.dotBf16, the reference of matmul_bf16w).
let private cpuRefRow (p: Planned) (s: Shape) (r: int) : uint32[] =
    let w = p.Inputs[s.Name]
    let out = Array.zeroCreate<uint32> s.N

    Parallel.For(
        0,
        8,
        fun part ->
            let xr = ReadOnlySpan(p.XData, r * s.K, s.K)

            for j in part * s.N / 8 .. (part + 1) * s.N / 8 - 1 do
                out[j] <- BitConverter.SingleToUInt32Bits(Rows.dotBf16<float32> xr (ReadOnlySpan(w, j * s.K, s.K)))
    )
    |> ignore

    out

let private invariance (backend: IGpuBackend) (p: Planned) (s: Shape) : Invariance =
    let w = p.Weights[s.Name]
    poison p p.YFull (512L * int64 s.N)
    poison p p.Y37 (37L * int64 s.N)

    for r in invarianceRows do
        poison p (snd p.RowViews[s.Name, r]) (int64 s.N)

    let commands = backend.Begin()
    Kernels.MatmulBf16w.dispatch commands (args p.X w p.YFull 512 s)
    Kernels.MatmulBf16w.dispatch commands (args p.X w p.Y37 37 s)

    for r in invarianceRows do
        let xv, yv = p.RowViews[s.Name, r]
        Kernels.MatmulBf16w.dispatch commands (args xv w yv 1 s)

    backend.Wait(backend.Submit commands) |> ignore
    let full = download p p.YFull (512L * int64 s.N)
    let y37 = download p p.Y37 (37L * int64 s.N)
    let mutable diffs = 0L
    let mutable refDiffs = 0L
    let notes = ResizeArray<string>()

    for r in invarianceRows do
        let _, yv = p.RowViews[s.Name, r]
        let alone = download p yv (int64 s.N)
        let inFull = Array.sub full (r * s.N) s.N

        let unwritten =
            Array.append alone inFull |> Array.filter ((=) Poison) |> Array.length

        let d512 =
            Array.fold2 (fun acc a b -> if a = b then acc else acc + 1L) 0L alone inFull

        let d37 =
            if r < 37 then
                Array.fold2 (fun acc a b -> if a = b then acc else acc + 1L) 0L alone (Array.sub y37 (r * s.N) s.N)
            else
                0L

        let reference = cpuRefRow p s r

        let dRef =
            Array.fold2 (fun acc a b -> if a = b then acc else acc + 1L) 0L alone reference

        diffs <- diffs + d512 + d37 + int64 unwritten
        refDiffs <- refDiffs + dRef

        if unwritten > 0 then
            notes.Add(Invariant.format $"row {r}: {unwritten} word(s) still hold the poison (never written)")

        if d512 + d37 > 0L then
            notes.Add(Invariant.format $"row {r}: {d512} differ from m = 512, {d37} from m = 37")

        if dRef > 0L then
            notes.Add(Invariant.format $"row {r}: {dRef} differ from CpuRef F32")

    let detail =
        if notes.Count = 0 then
            Invariant.format
                $"rows {String.Join(',', invarianceRows)} alone equal the m = 512 call (and the m = 37 call below 37) and CpuRef F32, bit for bit"
        else
            String.Join("; ", notes)

    let attrs =
        [
            Attr.str TelemetryKeys.Kernel Kernels.MatmulBf16w.Name
            Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend.ExecutionClass.Backend)
            Attr.str "shape" s.Name
            Attr.int "n" s.N
            Attr.int "k" s.K
            Attr.str "ms" "1,37,512"
            Attr.int "rows" invarianceRows.Length
            Attr.str "detail" detail
        ]

    Telemetry.check "gpu.matmul.batch_invariance" (diffs = 0L) (Attr.int64 "differences" diffs :: attrs)
    Telemetry.check "gpu.matmul.cpuref_exact" (refDiffs = 0L) (Attr.int64 "differences" refDiffs :: attrs)

    {
        Shape = s
        Rows = invarianceRows.Length
        Differences = diffs
        CpuRefDifferences = refDiffs
        Detail = detail
    }

/// What one bench run measured.
type Result =
    {
        RunId: string
        RunDirectory: string
        Device: string
        Points: Point list
        Invariance: Invariance list
    }

/// The commit this assembly was built from (the SDK stamps it after '+' in the informational version). Env.commit
/// reads the entry assembly, which is fsi when bench.fsx runs this module.
let private commit () =
    match typeof<Shape>.Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>() with
    | null -> None
    | v ->
        let s = v.InformationalVersion
        let plus = s.IndexOf '+'

        if plus >= 0 && plus < s.Length - 1 then
            Some(s.Substring(plus + 1))
        else
            None

/// The fp32 denormal probe of GpuBackendInterface (scale_f32, a = 0.5), recorded as gpu.fp32_denormals.
let private denormalProbe (backend: IGpuBackend) (p: Planned) =
    let commands = backend.Begin()

    Kernels.ScaleF32.dispatch
        commands
        {
            X = p.DenormalX
            Y = p.DenormalY
            N = uint32 Fp32Denormals.probeInputs.Length
            A = Fp32Denormals.ProbeScale
        }

    backend.Wait(backend.Submit commands) |> ignore
    let row = p.Table.Row p.DenormalY
    let bytes = Array.zeroCreate<byte>(4 * Fp32Denormals.probeInputs.Length)
    p.Heap.Download(row.Slot, int64 row.ByteOffset, Span bytes)
    let words = MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()
    Fp32Denormals.recordProbe backend.ExecutionClass.Backend words

/// The arithmetic probes: each result against CpuRef F32's bits and the other choice's.
let private runProbes (backend: IGpuBackend) (p: Planned) =
    let commands = backend.Begin()

    for pr, x, w, y in p.Probes do
        Kernels.MatmulBf16w.dispatch
            commands
            {
                X = x
                W = w
                Y = y
                M = 1u
                N = 1u
                K = uint32 pr.X.Length
            }

    backend.Wait(backend.Submit commands) |> ignore

    [
        for pr, _, _, y in p.Probes do
            let row = p.Table.Row y
            let bytes = Array.zeroCreate<byte> 4
            p.Heap.Download(row.Slot, int64 row.ByteOffset, Span bytes)
            let result = BitConverter.ToUInt32(bytes, 0)

            let verdict =
                if result = pr.Expected then "as CpuRef F32"
                elif result = pr.Other then "the other choice"
                else "neither"

            Telemetry.check
                "gpu.matmul.probe"
                (result = pr.Expected)
                [
                    Attr.str TelemetryKeys.Kernel Kernels.MatmulBf16w.Name
                    Attr.str TelemetryKeys.Backend (ExecutionBackend.name backend.ExecutionClass.Backend)
                    Attr.str "probe" pr.Name
                    Attr.str "expected" (pr.Expected.ToString("x8"))
                    Attr.str "other" (pr.Other.ToString("x8"))
                    Attr.str "result" (result.ToString("x8"))
                    Attr.str "detail" (pr.Doc + ": " + verdict)
                ]

            pr, result
    ]

/// A device of a GPU backend, through its provider (Discovery, as the conformance suites find it).
let private create (provider: IGpuBackendProvider) : IGpuBackend =
    let registry = BackendRegistry()
    registry.Register provider

    match
        registry.TryCreate
            { DeviceOptions.For provider.Backend with
                TimeDispatches = true
            }
    with
    | Ok b -> b
    | Error e -> failwithf "no %s device: %s" (ExecutionBackend.name provider.Backend) e

/// Runs the bench on one backend (throughput unless `invarianceOnly`, then the cross-call checks), in a run log of
/// kind kernels that `lrm report` renders. `loaded` sees the backend once its kernels are loaded (its class is known
/// then). The caller holds the backend's mutex.
let runOn (backend: IGpuBackend) (invarianceOnly: bool) (loaded: IGpuBackend -> unit) : Result =
    let _ = backend.LoadKernels(Lrm.Shaders.Shaders.embedded ())
    loaded backend
    let cls = backend.ExecutionClass
    let name = ExecutionBackend.name cls.Backend

    let run =
        RunContext.Create(
            { RunOptions.create
                  "kernels"
                  (if invarianceOnly then
                       "matmul_bf16w batch invariance on " + name
                   else
                       "matmul_bf16w throughput on " + name) with
                ExecutionClass = Some cls
                GitSha = commit ()
                Params =
                    [
                        "source", "tests/Lrm.Kernels.Conformance/Cases/matmul/MatmulBench.fs (K2)"
                        "backend", name
                        "device", backend.Info.Name
                        "reps", Invariant.i32 Reps
                    ]
            }
        )

    printfn "matmul bench run log (%s): %s" name run.Directory
    use _ = run.Install()
    let p = plan backend
    let mode = denormalProbe backend p
    printfn "  fp32 denormals: %s" (Fp32Denormals.name mode)

    for pr, result in runProbes backend p do
        printfn "  probe %-6s %08x (CpuRef F32 %08x, the other choice %08x)" pr.Name result pr.Expected pr.Other

    // The warm sweep runs first and back to back, after WarmUpMs of load; then the cold dispatches, each after an
    // idle gap (interleaving them would leave the next warm points to start from idle clocks).
    let warm =
        if invarianceOnly then
            []
        else
            warmUp backend p

            [
                for s in shapes do
                    for m in rowsOf s -> measure backend p s m
            ]

    let points =
        [
            for pt in warm ->
                if List.contains pt.M (coldRowsOf pt.Shape) then
                    let cold = measureCold backend p pt.Shape pt.M

                    { pt with
                        ColdNs = Some(List.last cold)
                        ColdFastNs = Some(List.head cold)
                    }
                else
                    pt
        ]

    for pt in points do
        record backend pt

        printfn
            "  %-13s m=%5d n=%6d k=%5d  %9.3f ms  %8.1f GFLOP/s  %7.1f GB/s%s"
            pt.Shape.Name
            pt.M
            pt.Shape.N
            pt.Shape.K
            (float pt.BestNs / 1e6)
            (pt.FlopsPerSecond / 1e9)
            (pt.BytesPerSecond / 1e9)
            (match pt.ColdNs, pt.ColdFastNs with
             | Some c, Some f -> Invariant.format $"  cold {float c / 1e6:F3} ms (fastest {float f / 1e6:F3} ms)"
             | _ -> "")

    let inv = shapes |> List.map (invariance backend p)

    for i in inv do
        printfn
            "  invariance %-13s differences %d, against CpuRef F32 %d"
            i.Shape.Name
            i.Differences
            i.CpuRefDifferences

    let ok = inv |> List.forall (fun i -> i.Differences = 0L)
    run.Finish(if ok then RunStatus.Ok else RunStatus.Failed)

    {
        RunId = run.RunId.Value
        RunDirectory = run.Directory
        Device = backend.Info.Name
        Points = points
        Invariance = inv
    }

/// One backend's tests: the cross-call invariance check (gpu tier) or the throughput sweep with the same checks
/// (bench tier, which never gates), each with its run log.
let private backendTests (backend: ExecutionBackend) (testing: Backend) (invarianceOnly: bool) =
    match Discovery.provider backend with
    | None -> []
    | Some provider ->
        [
            testing,
            [
                testCase (
                    if invarianceOnly then
                        "matmul_bf16w rows alone equal the rows of m = 37 and m = 512 calls and CpuRef F32"
                    else
                        "matmul_bf16w throughput at the Qwen3-0.6B shapes, m = 1 to 4,000"
                )
                <| fun () ->
                    use b = create provider

                    let r =
                        runOn b invarianceOnly (fun b ->
                            let c = b.ExecutionClass
                            Devices.record testing b.Info.Name c.Driver (c.ShaderSetHash |> Option.map _.Hex))

                    for i in r.Invariance do
                        Expect.equal i.Differences 0L (i.Shape.Name + ": " + i.Detail)

                    // The budget holds for what X1 dispatches, warm and cold; lm_head past its cap is only measured.
                    for pt in r.Points |> List.filter (fun pt -> dispatched pt.Shape pt.M) do
                        Expect.isLessThan
                            pt.BudgetNs
                            Budget
                            (Invariant.format
                                $"{pt.Shape.Name} m = {pt.M} is over the 100 ms dispatch budget (warm or cold)")
            ]
        ]

let private backends =
    [
        ExecutionBackend.D3D12, Backend.D3D12
        ExecutionBackend.Vulkan, Backend.Vulkan
    ]

/// The gpu-tier tests (TestTiers): the cross-call batch-invariance check on every GPU backend with a provider.
let gpu () : (Backend * Test list) list =
    backends |> List.collect (fun (b, t) -> backendTests b t true)

/// The bench-tier tests: the throughput sweep with its cold dispatches, then the same checks, on every GPU backend
/// with a provider.
let bench () : Test list =
    backends
    |> List.collect (fun (b, t) -> backendTests b t false)
    |> List.collect snd
