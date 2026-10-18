namespace Lrm.Serving

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Threading
open Lrm.Core
open Lrm.Tensor
open Lrm.Cluster
open Lrm.Models.Qwen

/// A unit or call the Qwen stage executor refuses: another job, chain, model, stage or class, a chain that is not at
/// the unit's position, or a payload of the wrong size (StagePipeline, "The executor").
type StageException(message: string) =
    inherit InvalidOperationException("stage executor: " + message)

/// The settings of a QwenStageExecutor (StagePipeline, "The executor"). They decide which units the executor runs and
/// how much it holds, never the bits it computes.
type StageExecutorOptions =
    {
        /// The split (StageLayout): the midpoint by default.
        Boundaries: int list
        /// The indices of the stages this executor hosts (IStageExecutor.CanHost).
        Hosted: int list
        /// The positions each chain's KV cache holds; a job must fit (ServingJob.maxPositions).
        MaxContext: int
        /// The chains (lane, stage) alive at once; one KV cache each.
        MaxChains: int
        /// CPU threads of CPU ops; the bits are the same for every count.
        Threads: int
        /// Allow IFastRebuild.Rebuild on the chain states (off by default: exact replay is the primary rebuild).
        FastRebuild: bool
        /// The most rows of one fast-rebuild forward; 0 for the automatic limit (MaxContext, and on the head stage at
        /// most 64 MiB of logits). A unit with more rows is a forward of its own.
        RebuildChunkRows: int
        /// Milliseconds each unit sleeps before it runs, to give a test run a cost; 0 for none. It changes no value.
        CostMs: int
    }

[<RequireQualifiedAccess>]
module StageExecutorOptions =
    /// The midpoint split, every stage hosted, min(1024, max_position_embeddings) positions, 16 chains, 8 threads (or
    /// fewer cores), no fast rebuild, no cost.
    let defaults (c: Qwen3Config) =
        {
            Boundaries = StageLayout.midpoint c.Layers
            Hosted = StageLayout.stages c.Layers (StageLayout.midpoint c.Layers) |> List.map _.Index
            MaxContext = min 1024 c.MaxPositionEmbeddings
            MaxChains = 16
            Threads = min 8 Environment.ProcessorCount
            FastRebuild = false
            RebuildChunkRows = 0
            CostMs = 0
        }

    /// The options when they fit the model: a valid split, hosted stages of it (at least one, each once), 1 ..
    /// max_position_embeddings positions, at least one chain and one thread, and a cost of 0 or more.
    let validate (c: Qwen3Config) (o: StageExecutorOptions) : Result<StageExecutorOptions, string> =
        match StageLayout.validate c.Layers o.Boundaries with
        | Error e -> Error e
        | Ok bs ->
            let count = bs.Length + 1

            if o.Hosted.IsEmpty then
                Error "the executor hosts no stage"
            elif o.Hosted |> List.exists (fun i -> i < 0 || i >= count) then
                Error(sprintf "hosted stages %A are not all stages of a %d-stage split" o.Hosted count)
            elif (List.distinct o.Hosted).Length <> o.Hosted.Length then
                Error(sprintf "hosted stages %A repeat a stage" o.Hosted)
            elif o.MaxContext < 1 || o.MaxContext > c.MaxPositionEmbeddings then
                Error(sprintf "max context %d is outside 1 .. %d" o.MaxContext c.MaxPositionEmbeddings)
            elif o.MaxChains < 1 then
                Error "at least one chain"
            elif o.Threads < 1 then
                Error "at least one thread"
            elif o.CostMs < 0 || o.RebuildChunkRows < 0 then
                Error "a negative cost or rebuild chunk"
            else
                Ok o

    /// The hosted stages, in index order.
    let stages (c: Qwen3Config) (o: StageExecutorOptions) =
        let all = StageLayout.stages c.Layers o.Boundaries |> Array.ofList
        o.Hosted |> List.sort |> List.map (fun i -> all[i])

    /// The ops plan of an executor (IForwardOps, "The plan"): the hosted stages, MaxContext positions and rows, no
    /// taps, and MaxChains caches per stage.
    let opsOptions (c: Qwen3Config) (o: StageExecutorOptions) : OpsOptions =
        {
            Threads = o.Threads
            MaxContext = o.MaxContext
            MaxRows = o.MaxContext
            Stages = stages c o
            Taps = TapFilter.Off
            Caches = o.MaxChains
        }

/// A chain state that can rebuild its KV cache from its chain's units in few forwards instead of one per unit (the
/// fast rebuild). It must equal exact replay bit for bit: the same outputs and the same digest after every unit.
type IFastRebuild =
    /// On a fresh chain state, applies `units` (step 0 onwards, a hash chain, with their inputs) in forwards of many
    /// rows each and returns every unit's output and the digest after it. Raises StageException when fast rebuild is
    /// off or the state is not fresh.
    abstract Rebuild: units: (WorkUnit * byte[]) list -> (byte[] * ContentId) list

/// The Qwen stage executor as nodes and tests see it, with its ops type hidden. It owns its ops: Dispose releases them.
type IQwenStageExecutor =
    inherit IStageExecutor
    inherit IDisposable

    /// The ModelId of the weights it runs.
    abstract Model: ContentId
    /// The hosted stages, in index order.
    abstract Stages: Stage list
    abstract Options: StageExecutorOptions
    /// The chains that hold a KV cache now.
    abstract LiveChains: int
    /// Units applied so far (Apply and Rebuild).
    abstract Applied: int64

[<Sealed; AllowNullLiteral>]
type internal ChainSlot<'t>(job: ContentId, chain: ChainKey) =
    member _.Job = job
    member _.Chain = chain
    member val Digest = ChainDigest.initial job chain with get, set
    member val Cache: KvCache<'t> option = None with get, set
    /// No unit applied yet.
    member val Fresh = true with get, set
    member val Disposed = false with get, set

/// A chain state's slot, for the executor's own inspection.
type internal IChainSlotView<'t> =
    abstract Slot: ChainSlot<'t>

/// IStageExecutor for the Qwen pipeline over any IForwardOps (StagePipeline, "The executor"). Each chain (lane, stage)
/// keeps one KV cache, created at its first unit; Apply runs Forward.runStage on the unit's rows and returns the
/// residual or, on the head stage, the token and the last row's logits. One lock serializes every use of the ops,
/// because a node opens and disposes chain states on its actor thread while units run on a worker; a cache whose
/// state is disposed during another chain's forward is released at the next use of the ops.
[<Sealed>]
type QwenStageExecutor<'t>(ops: IForwardOps<'t>, model: Qwen3Model, modelId: ContentId, options: StageExecutorOptions) =
    let c = model.Config

    do
        match StageExecutorOptions.validate c options with
        | Ok _ -> ()
        | Error e -> invalidArg "options" e

    let hosted = StageExecutorOptions.stages c options
    let classId = ExecutionClass.id ops.ExecutionClass
    let dtype = ops.DType
    let gate = obj ()
    let pending = ConcurrentQueue<ChainSlot<'t>>()
    let mutable live = 0
    let mutable applied = 0L
    let mutable disposed = false

    let fail (message: string) = raise (StageException message)

    /// Releases a slot's cache (under the gate).
    let releaseNow (slot: ChainSlot<'t>) =
        match slot.Cache with
        | Some cache ->
            slot.Cache <- None
            (cache :> IDisposable).Dispose()
            live <- live - 1
        | None -> ()

    let drain () =
        let mutable slot = null

        while pending.TryDequeue(&slot) do
            releaseNow slot

    /// Runs f with the ops to itself, after releasing the caches of states disposed meanwhile.
    let exclusive (f: unit -> 'r) : 'r =
        lock gate (fun () ->
            if disposed then
                raise (ObjectDisposedException("QwenStageExecutor", "the executor was disposed"))

            drain ()
            f ())

    let disposeSlot (slot: ChainSlot<'t>) =
        if not slot.Disposed then
            slot.Disposed <- true

            if Monitor.TryEnter gate then
                try
                    if not disposed then
                        releaseNow slot
                finally
                    Monitor.Exit gate
            else
                pending.Enqueue slot

    let stageOf (index: uint32) =
        match hosted |> List.tryFind (fun s -> uint32 s.Index = index) with
        | Some s -> s
        | None -> fail (sprintf "stage %d is not hosted here (hosted: %A)" index (hosted |> List.map _.Index))

    /// The unit's args and stage, after every check that needs no ops.
    let checkUnit (slot: ChainSlot<'t>) (work: WorkUnit) (input: byte[]) : StageArgs * Stage =
        if not (Invariant.equals work.Kind WorkUnit.FwdStage) || work.KindVersion <> 1us then
            fail (sprintf "unit kind %s v%d is not fwd.stage v1" work.Kind work.KindVersion)

        if work.Job <> slot.Job || WorkUnit.chain work <> slot.Chain then
            fail "the unit belongs to another job or chain than this state"

        if work.Class <> classId then
            fail "the unit's class is not this executor's class"

        if isNull input || BlobId.ofBytes input <> work.Input then
            fail "the input bytes do not match the unit's Input"

        let a =
            try
                StageArgs.decode work.Args
            with :? FormatException as e ->
                fail e.Message

        if a.Model <> modelId then
            fail (sprintf "the unit is for model %s; this executor runs %s" a.Model.Short modelId.Short)

        if a.Layers <> c.Layers || a.DType <> dtype then
            fail (
                sprintf
                    "the unit is for %d layers in %s; this executor runs %d in %s"
                    a.Layers
                    (DType.name a.DType)
                    c.Layers
                    (DType.name dtype)
            )

        let stage = stageOf work.Stage

        if stage.FirstLayer <> a.FirstLayer || stage.LastLayer <> a.LastLayer then
            fail (
                sprintf
                    "the unit's stage %d is layers %d-%d; here it is %d-%d"
                    stage.Index
                    a.FirstLayer
                    a.LastLayer
                    stage.FirstLayer
                    stage.LastLayer
            )

        let expected =
            if stage.Embeds then
                4 * a.Rows
            else
                StagePayload.residualBytes dtype a.Rows c.Hidden

        if input.Length <> expected then
            fail (sprintf "%d input bytes; %d rows at stage %d need %d" input.Length a.Rows stage.Index expected)

        a, stage

    let cacheOf (slot: ChainSlot<'t>) (stage: Stage) =
        match slot.Cache with
        | Some cache -> cache
        | None ->
            if live >= options.MaxChains then
                fail (sprintf "%d chains are alive, the executor's limit (MaxChains)" live)

            let cache = KvCache.create ops model stage options.MaxContext
            live <- live + 1
            slot.Cache <- Some cache
            cache

    /// The stage input of `rows` rows from payload bytes (outside a forward: Upload).
    let stageInput (stage: Stage) (rows: int) (bytes: byte[]) : StageInput<'t> =
        if stage.Embeds then
            StageInput.Ids(StagePayload.ids bytes rows)
        else
            let shape = Shape.matrix rows c.Hidden

            let host: IHostTensor =
                match dtype with
                | DType.F64 -> HostTensor.ofBytes<float> DType.F64 shape (ReadOnlySpan bytes)
                | _ -> HostTensor.ofBytes<float32> DType.F32 shape (ReadOnlySpan bytes)

            StageInput.Hidden(ops.Upload host)

    let kvRows (stage: Stage) (cache: KvCache<'t>) (pos0: int) (rows: int) =
        ChainDigest.kvRows stage.FirstLayer stage.LayerCount pos0 rows dtype (fun layer isValue ->
            let t = if isValue then cache.Values layer else cache.Keys layer
            (ops.ReadRows(t, pos0, rows)).ToBytes())

    let advance (slot: ChainSlot<'t>) (stage: Stage) (cache: KvCache<'t>) (work: WorkUnit) (a: StageArgs) output =
        let kv = kvRows stage cache a.Pos0 a.Rows
        slot.Digest <- ChainDigest.after slot.Digest work.Step work.Input (BlobId.ofBytes output) kv
        slot.Fresh <- false
        Interlocked.Increment(&applied) |> ignore

    let apply (slot: ChainSlot<'t>) (work: WorkUnit) (input: byte[]) : byte[] =
        if options.CostMs > 0 then
            Thread.Sleep options.CostMs

        exclusive (fun () ->
            if slot.Disposed then
                raise (ObjectDisposedException("chain state", "the chain state was disposed"))

            let a, stage = checkUnit slot work input

            let length =
                match slot.Cache with
                | Some cache -> cache.Length
                | None -> 0

            if length <> a.Pos0 then
                fail (
                    sprintf
                        "chain (%d, %d) holds %d positions; unit step %d starts at %d"
                        slot.Chain.Lane
                        slot.Chain.Stage
                        length
                        work.Step
                        a.Pos0
                )

            if a.Pos0 + a.Rows > options.MaxContext then
                fail (
                    sprintf
                        "positions %d .. %d do not fit the executor's %d (MaxContext)"
                        a.Pos0
                        (a.Pos0 + a.Rows - 1)
                        options.MaxContext
                )

            let cache = cacheOf slot stage
            let x = stageInput stage a.Rows input

            let result =
                Forward.runStage
                    ops
                    model
                    stage
                    cache
                    a.Pos0
                    x
                    ({
                        Head = HeadRows.Last
                        Taps = None
                        Step = Some(int work.Step)
                    }
                    : ForwardOptions<'t>)

            let output =
                if a.Head then
                    let token = (ops.Argmax result)[0]
                    StagePayload.ofHead token ((ops.Read result).ToBytes())
                else
                    (ops.Read result).ToBytes()

            advance slot stage cache work a output
            output)

    /// The most rows of one rebuild forward: MaxContext, and on the head stage at most 64 MiB of logits.
    let rebuildRows (stage: Stage) =
        if options.RebuildChunkRows > 0 then
            options.RebuildChunkRows
        elif Stage.hasHead c stage then
            max 1 (min options.MaxContext ((64 * 1024 * 1024) / (c.Vocab * DType.size dtype)))
        else
            options.MaxContext

    let rebuild (slot: ChainSlot<'t>) (units: (WorkUnit * byte[]) list) : (byte[] * ContentId) list =
        if not options.FastRebuild then
            fail "fast rebuild is off (StageExecutorOptions.FastRebuild)"

        exclusive (fun () ->
            if slot.Disposed then
                raise (ObjectDisposedException("chain state", "the chain state was disposed"))

            if not slot.Fresh then
                fail "fast rebuild needs a fresh chain state"

            // Every check before the first forward: a hash chain from step 0 with contiguous positions.
            let checkedUnits = ResizeArray<WorkUnit * byte[] * StageArgs * Stage>()
            let mutable prev = UnitId.none
            let mutable pos = 0

            for i, (work, input) in List.indexed units do
                if int work.Step <> i || work.Prev <> prev then
                    fail (sprintf "unit %d of the rebuild does not follow the chain (step %d)" i work.Step)

                let a, stage = checkUnit slot work input

                if a.Pos0 <> pos then
                    fail (sprintf "unit step %d starts at %d; the chain holds %d positions" work.Step a.Pos0 pos)

                prev <- WorkUnit.id work
                pos <- pos + a.Rows
                checkedUnits.Add((work, input, a, stage))

            if pos > options.MaxContext then
                fail (
                    sprintf "the chain's %d positions do not fit the executor's %d (MaxContext)" pos options.MaxContext
                )

            match List.ofSeq checkedUnits with
            | [] -> []
            | ((_, _, _, stage) :: _) as checkedUnits ->
                let cache = cacheOf slot stage
                let limit = rebuildRows stage

                // Consecutive units in chunks of at most `limit` rows (a longer unit is a chunk of its own).
                let chunks =
                    let acc = ResizeArray<ResizeArray<WorkUnit * byte[] * StageArgs * Stage>>()

                    for u in checkedUnits do
                        let _, _, a, _ = u

                        let fits =
                            acc.Count > 0
                            && (acc[acc.Count - 1] |> Seq.sumBy (fun (_, _, x, _) -> x.Rows)) + a.Rows <= limit

                        if fits then
                            acc[acc.Count - 1].Add u
                        else
                            acc.Add(ResizeArray [ u ])

                    acc |> Seq.map List.ofSeq |> List.ofSeq

                [
                    for chunk in chunks do
                        let _, _, first, _ = chunk.Head
                        let rows = chunk |> List.sumBy (fun (_, _, a, _) -> a.Rows)
                        let bytes = chunk |> List.map (fun (_, input, _, _) -> input) |> Array.concat
                        let x = stageInput stage rows bytes

                        let result =
                            Forward.runStage
                                ops
                                model
                                stage
                                cache
                                first.Pos0
                                x
                                ({
                                    Head = HeadRows.All
                                    Taps = None
                                    Step = None
                                }
                                : ForwardOptions<'t>)

                        let tokens = if first.Head then ops.Argmax result else [||]

                        for work, _, a, _ in chunk do
                            let row0 = a.Pos0 - first.Pos0

                            let output =
                                if a.Head then
                                    let last = row0 + a.Rows - 1
                                    StagePayload.ofHead tokens[last] ((ops.ReadRows(result, last, 1)).ToBytes())
                                else
                                    (ops.ReadRows(result, row0, a.Rows)).ToBytes()

                            advance slot stage cache work a output
                            yield output, slot.Digest
                ])

    /// The ops (for tests that compare caches with KvCache.digest).
    member _.Ops = ops

    /// KvCache.digest of a chain state's cache, or None when it holds none (tests).
    member _.CacheDigest(state: IChainState) : ContentId option =
        match box state with
        | :? IChainSlotView<'t> as v ->
            exclusive (fun () -> v.Slot.Cache |> Option.map (fun cache -> KvCache.digest ops cache))
        | _ -> None

    interface IQwenStageExecutor with
        member _.Class = ops.ExecutionClass
        member _.CanHost = hosted |> List.map (fun s -> uint32 s.Index)
        member _.Model = modelId
        member _.Stages = hosted
        member _.Options = options
        member _.LiveChains = live
        member _.Applied = Interlocked.Read(&applied)

        member _.Open(job: ContentId, chain: ChainKey) =
            let slot = ChainSlot<'t>(job, chain)

            { new IChainState with
                member _.Digest = slot.Digest
                member _.Apply(work, input) = apply slot work input
                member _.Dispose() = disposeSlot slot
              interface IFastRebuild with
                  member _.Rebuild units = rebuild slot units
              interface IChainSlotView<'t> with
                  member _.Slot = slot
            }

        member _.Dispose() =
            lock gate (fun () ->
                if not disposed then
                    drain ()
                    disposed <- true
                    ops.Dispose())

/// Constructors and the serving backend registry (StagePipeline, "Backends").
[<RequireQualifiedAccess>]
module QwenStageExecutor =
    /// An executor over `ops`, which it owns, for the model of `weights`. The ops must have been created with
    /// StageExecutorOptions.opsOptions (or a plan that covers it).
    let create (ops: IForwardOps<'t>) (weights: Qwen3Weights) (options: StageExecutorOptions) : IQwenStageExecutor =
        new QwenStageExecutor<'t>(ops, weights.Model, ModelId.ofWeights weights, options) :> IQwenStageExecutor

    /// One serving backend: a name and how to build an executor on verified weights.
    type Backend =
        {
            Name: string
            Create: Qwen3Weights -> StageExecutorOptions -> IQwenStageExecutor
        }

    /// cpu-f32 and cpu-f64: CpuOps (CpuRef) under the executor's plan.
    let cpu: Backend list =
        [
            {
                Name = ExecutionBackend.name ExecutionBackend.CpuF32
                Create = fun w o -> create (CpuOps.f32 w (StageExecutorOptions.opsOptions w.Config o)) w o
            }
            {
                Name = ExecutionBackend.name ExecutionBackend.CpuF64
                Create = fun w o -> create (CpuOps.f64 w (StageExecutorOptions.opsOptions w.Config o)) w o
            }
        ]

    let private gate = obj ()
    let private registry = List<Backend>(cpu)

    /// Adds a backend (C2b registers the GPU executors). Raises ArgumentException for a name registered already.
    let register (backend: Backend) =
        lock gate (fun () ->
            if registry |> Seq.exists (fun b -> b.Name = backend.Name) then
                invalidArg "backend" (sprintf "serving backend '%s' is registered already" backend.Name)

            registry.Add backend)

    /// Every registered backend, in registration order.
    let backends () =
        lock gate (fun () -> List.ofSeq registry)

    /// A backend by name.
    let tryBackend (name: string) =
        backends () |> List.tryFind (fun b -> b.Name = name)

    /// The ExecutionClass a backend's executors have, without building one: ExecutionClass.forCpu for the CPU
    /// backends; None for others (their class depends on the device).
    let classOf (name: string) : ExecutionClass option =
        match ExecutionBackend.tryParse name with
        | Some(ExecutionBackend.CpuF32 as b)
        | Some(ExecutionBackend.CpuF64 as b) -> Some(ExecutionClass.forCpu b)
        | _ -> None

    /// KvCache.digest of the cache behind a chain state from `executor` (None when it holds no cache, or when the
    /// executor is not a QwenStageExecutor over host tensors of float32 or float).
    let cacheDigest (executor: IQwenStageExecutor) (state: IChainState) : ContentId option =
        match executor with
        | :? QwenStageExecutor<HostTensor<float32>> as e -> e.CacheDigest state
        | :? QwenStageExecutor<HostTensor<float>> as e -> e.CacheDigest state
        | _ -> None
