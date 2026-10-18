namespace Lrm.Models.Qwen

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// A forward pass, a stage, a cache or a handle that breaks the forward's rules (ForwardOpsAndQwenModel contract).
type ForwardException(message: string) =
    inherit InvalidOperationException("forward: " + message)

/// A contiguous range of decoder layers of one pipeline stage (StagePipeline: C2 splits the model at a boundary). The
/// stage that starts at layer 0 also embeds; the stage that ends at the last layer also runs the final norm and the
/// head.
type Stage =
    {
        /// The stage's index in its pipeline: the `stage` attribute of its telemetry, and the key of its handles'
        /// lifetimes on one ops instance (IForwardOps, "Handle lifetimes").
        Index: int
        FirstLayer: int
        /// Inclusive.
        LastLayer: int
    }

    member s.LayerCount = s.LastLayer - s.FirstLayer + 1

    /// True for the stage that starts at layer 0: its input is token ids.
    member s.Embeds = s.FirstLayer = 0

[<RequireQualifiedAccess>]
module Stage =
    /// The whole model as one stage.
    let whole (c: Qwen3Config) =
        {
            Index = 0
            FirstLayer = 0
            LastLayer = c.Layers - 1
        }

    /// True for the stage that ends at the last layer: its output is logits.
    let hasHead (c: Qwen3Config) (s: Stage) = s.LastLayer = c.Layers - 1

    /// Raises ArgumentException unless the stage lies inside the model.
    let validate (c: Qwen3Config) (s: Stage) =
        if
            s.Index < 0
            || s.FirstLayer < 0
            || s.LastLayer < s.FirstLayer
            || s.LastLayer >= c.Layers
        then
            invalidArg "stage" (sprintf "%A is not a layer range of a %d-layer model" s c.Layers)

    /// The stages that start at layer 0 and at each boundary, in order: `split c [ 14 ]` is layers 0-13 (with the
    /// embedding) and 14 to the last (with the head). Boundaries must increase strictly inside 1 .. layers - 1.
    let split (c: Qwen3Config) (boundaries: int list) =
        let starts = 0 :: boundaries

        for a, b in List.pairwise starts do
            if b <= a then
                invalidArg "boundaries" (sprintf "boundaries %A do not increase from layer 1" boundaries)

        match List.tryLast boundaries with
        | Some b when b >= c.Layers -> invalidArg "boundaries" (sprintf "boundary %d is not below %d layers" b c.Layers)
        | _ -> ()

        let ends = (boundaries |> List.map (fun b -> b - 1)) @ [ c.Layers - 1 ]

        List.zip starts ends
        |> List.mapi (fun i (a, b) ->
            {
                Index = i
                FirstLayer = a
                LastLayer = b
            })

/// Which rows the head (final norm and lm_head) runs on.
[<RequireQualifiedAccess>]
type HeadRows =
    /// Every row: logits [T, vocab] (parity runs; the final_norm.out and logits taps of a prefill).
    | All
    /// The last row only: logits [1, vocab] (greedy decoding), bit-equal to the last row of All (BatchInvariance). The
    /// final norm still runs on every row, so the final_norm.out tap is [T, hidden] either way.
    | Last

/// Which taps a forward records (ActivationNames).
[<RequireQualifiedAccess>]
type TapFilter =
    /// No taps.
    | Off
    /// Every ActivationNames tap the stage produces.
    | All
    /// Exact names, such as `L0.attn.q_rope` or `logits`. Every name must be an ActivationNames tap of the model
    /// (`TapFilter.unknown`); the forward refuses a filter with any other name.
    | Names of string list

    /// True when the filter asks for `name`.
    member f.Wants(name: string) =
        match f with
        | TapFilter.Off -> false
        | TapFilter.All -> true
        | TapFilter.Names names -> List.contains name names

    /// True when every tap `other` asks for, this filter asks for too: the plan's filter covers a forward's.
    member f.Covers(other: TapFilter) =
        match f, other with
        | _, TapFilter.Off -> true
        | TapFilter.All, _ -> true
        | TapFilter.Names mine, TapFilter.Names theirs -> theirs |> List.forall (fun n -> List.contains n mine)
        | _ -> false

    /// Parses `none`, `all` or a comma-separated list of tap names. The names are checked against a model by
    /// `TapFilter.unknown`, which the forward, the ops' plan and `lrm forward --taps` apply.
    static member Parse(text: string) =
        match text.Trim() with
        | "none"
        | "" -> TapFilter.Off
        | "all" -> TapFilter.All
        | list ->
            list.Split(',')
            |> Array.map (fun s -> s.Trim())
            |> Array.filter (fun s -> s.Length > 0)
            |> List.ofArray
            |> TapFilter.Names

[<RequireQualifiedAccess>]
module TapFilter =
    /// The names of a filter that are not ActivationNames taps of a model with `layers` layers, once each, in the
    /// filter's order (empty for Off and All).
    let unknown (layers: int) (filter: TapFilter) =
        match filter with
        | TapFilter.Off
        | TapFilter.All -> []
        | TapFilter.Names names ->
            let known = HashSet<string>(ActivationNames.all layers, StringComparer.Ordinal)
            names |> List.filter (fun n -> not (known.Contains n)) |> List.distinct

/// One forward call, as `Forward.runStage` announces it to the ops before its first op (IForwardOps.BeginForward). The
/// forward's op sequence depends only on the model and these (the static order).
type ForwardCall =
    {
        Stage: Stage
        /// T, the rows of the call.
        Rows: int
        /// The position of the first row.
        Pos0: int
        /// The rows the head runs on; the op sequence depends on it only on the stage with the head.
        Head: HeadRows
        /// The taps the call records (Off when it records none).
        Taps: TapFilter
    }

/// The op seam of the Qwen3 forward (ForwardOpsAndQwenModel contract). The forward is written once over this
/// interface; CpuOps implements it on CpuRef (F32 and F64), and the GPU executor (X1) on the kernel catalog.
///
/// 't is the backend's activation handle: a host tensor for CpuOps, a tensor token for a GPU. Every handle is a
/// row-major [rows, cols] tensor in the ops' DType (float32, or float64 for cpu-f64). An op returns its result handle
/// at once and never reads back to the host; only Argmax, Read and ReadRows return host data. The forward calls the
/// ops in one fixed order that depends only on the model and the ForwardCall, so a backend can record a static plan.
/// Weights are named by their safetensors names (Qwen3Params, aliases resolved); the ops are created over one
/// verified Qwen3Weights and resolve names themselves.
///
/// The plan. The ops are created with OpsOptions, which bound every call: the stages they run, the rows of one
/// forward, the positions, the taps and the caches alive at once. A GPU sizes and seals its heap and tensor table from
/// them once, before the first forward; a call outside the plan raises ForwardException.
///
/// Handle lifetimes. Each forward is bracketed by BeginForward and EndForward (runStage does both).
/// - The compute ops (Embed to LastRow, and Tap) are called only inside a forward; NewCache, ReleaseCache and Upload
///   only outside one. Rows, Cols, Argmax, Read and ReadRows work on any live handle at any time.
/// - Intermediates, the handles the ops return inside a forward, are valid until its EndForward, except the forward's
///   result (the handle EndForward receives) and its taps (the handles Tap returns). Those stay valid until the next
///   BeginForward of a stage with the same Index on these ops: a stage's result can be the next stage's input, and a
///   greedy step's logits are read before the next step begins.
/// - A forward that raises ends with EndForward None: every handle it made, taps included, is then invalid.
/// - A handle from Upload stays valid until the next forward ends; a second Upload before then invalidates it.
/// - A cache tensor from NewCache stays valid until ReleaseCache (KvCache.Dispose releases a cache's tensors).
/// - Dispose releases everything; the ops are unusable after it.
/// One ops instance serves one caller at a time.
type IForwardOps<'t> =
    inherit IDisposable

    /// The class whose bits these ops produce. Bit-identity holds only within it (ExecutionClass contract).
    abstract ExecutionClass: ExecutionClass

    /// The element type of activations and caches: F32, or F64 for cpu-f64 (CpuReferenceSemantics).
    abstract DType: DType

    /// The positions the RoPE table covers; a cache holds at most this many.
    abstract MaxContext: int

    /// The rows of a handle.
    abstract Rows: x: 't -> int

    /// The columns of a handle.
    abstract Cols: x: 't -> int

    /// Starts a forward: every compute op until EndForward belongs to it. Raises ForwardException for a call outside
    /// the plan or inside another forward. A GPU picks or records the call's plan here, and may reuse the previous
    /// result and taps of the call's stage.
    abstract BeginForward: call: ForwardCall -> unit

    /// Ends the forward: Some result when it completed (the result and the taps stay valid), None when it raised.
    abstract EndForward: result: 't option -> unit

    /// Embedding rows of ids: [ids, hidden], each weight widened exactly. An id outside the vocabulary raises
    /// ArgumentOutOfRangeException.
    abstract Embed: ids: int[] -> 't

    /// RMSNorm over rows of the weight's length (CpuRef.rmsNorm): [T, heads * head_dim] with a head_dim weight is the
    /// per-head q_norm and k_norm. eps is rounded once to DType.
    abstract RmsNorm: x: 't * weight: string * eps: float -> 't

    /// x [T, k] times the weight [n, k] transposed: [T, n] (CpuRef.matmulBf16W).
    abstract MatMul: x: 't * weight: string -> 't

    /// NeoX RoPE of x [T, heads * head_dim] from the pinned T_spec table; row t is at position pos0 + t.
    abstract Rope: x: 't * heads: int * pos0: int -> 't

    /// A new cache tensor of `rows` rows of zeros for model layer `layer` of a planned stage: its K or V,
    /// [rows, KvHeads * HeadDim], in DType (a GPU takes it from its KvCache arena).
    abstract NewCache: stage: Stage * layer: int * rows: int -> 't

    /// Returns a cache tensor from NewCache to the ops; the handle is invalid afterwards.
    abstract ReleaseCache: cache: 't -> unit

    /// Copies the rows of x into cache rows pos0 .. pos0 + T - 1, bit for bit.
    abstract KvWrite: cache: 't * x: 't * pos0: int -> unit

    /// Causal GQA attention (CpuRef.attention) of q [T, QHeads * HeadDim], after norm and RoPE, over cache rows
    /// 0 .. past + T - 1; query row i is at position past + i.
    abstract Attention: q: 't * k: 't * v: 't * shape: AttentionShape * past: int -> 't

    /// silu(gate) * up, element-wise.
    abstract SiluMul: gate: 't * up: 't -> 't

    /// a + b, element-wise.
    abstract Add: a: 't * b: 't -> 't

    /// The last row of x as [1, cols].
    abstract LastRow: x: 't -> 't

    /// Each row's lowest maximal index, or 0xFFFFFFFF for a row that holds a NaN (CpuRef.argmaxRows, the argmax_rows
    /// kernel). Reads back to the host.
    abstract Argmax: x: 't -> uint32[]

    /// A snapshot of x that later ops never change, valid as long as the forward's result ("Handle lifetimes"); a GPU
    /// copies it into its Taps arena when the op is recorded. Called only for the taps the forward's filter wants,
    /// which the plan's filter covers.
    abstract Tap: name: string * x: 't -> 't

    /// The elements of x on the host, [rows, cols] in DType: a copy the caller owns.
    abstract Read: x: 't -> IHostTensor

    /// Rows row0 .. row0 + count - 1 of x on the host, [count, cols] in DType, count >= 1: a copy the caller owns.
    /// Only those rows are read back (KvCache.digest reads a cache's valid rows).
    abstract ReadRows: x: 't * row0: int * count: int -> IHostTensor

    /// A handle holding a host tensor [rows, cols] in DType (a stage boundary's residual, for example).
    abstract Upload: x: IHostTensor -> 't

/// The options a backend's ops are created with (OpsFactory, Commands/Backends.fs): the threads and the plan, which
/// bounds every call (IForwardOps, "The plan").
type OpsOptions =
    {
        /// Host threads, for CPU backends (CpuExec); the bits are the same for every count.
        Threads: int
        /// Positions the RoPE table covers (T_spec for 0 .. MaxContext - 1); caches hold at most this many.
        MaxContext: int
        /// The most rows (T) one forward runs; MaxContext bounds a forward too.
        MaxRows: int
        /// The stages these ops run, with distinct indices.
        Stages: Stage list
        /// The taps a forward may record; a GPU sizes its Taps arena for them.
        Taps: TapFilter
        /// The KV caches (KvCache.create) of each planned stage alive at once.
        Caches: int
    }

[<RequireQualifiedAccess>]
module OpsOptions =
    /// 8 threads (the plan's limit for long CPU jobs on the shared machine), the model's max_position_embeddings for
    /// the positions and the rows, the whole model as stage 0, no taps and one cache.
    let defaults (c: Qwen3Config) =
        {
            Threads = min 8 Environment.ProcessorCount
            MaxContext = c.MaxPositionEmbeddings
            MaxRows = c.MaxPositionEmbeddings
            Stages = [ Stage.whole c ]
            Taps = TapFilter.Off
            Caches = 1
        }

    /// Raises ArgumentException unless the options are a plan for the model: at least one thread, position, row and
    /// cache; valid stages with distinct indices; only ActivationNames taps.
    let validate (c: Qwen3Config) (o: OpsOptions) =
        if o.Threads < 1 then
            invalidArg "options" (sprintf "%d threads" o.Threads)

        if o.MaxContext < 1 then
            invalidArg "options" (sprintf "max context %d" o.MaxContext)

        if o.MaxRows < 1 then
            invalidArg "options" (sprintf "max rows %d" o.MaxRows)

        if o.Caches < 1 then
            invalidArg "options" (sprintf "%d caches per stage" o.Caches)

        if o.Stages.IsEmpty then
            invalidArg "options" "the plan has no stages"

        for s in o.Stages do
            Stage.validate c s

        if (o.Stages |> List.distinctBy _.Index).Length <> o.Stages.Length then
            invalidArg "options" (sprintf "the planned stages %A repeat an index" o.Stages)

        match TapFilter.unknown c.Layers o.Taps with
        | [] -> ()
        | names ->
            invalidArg
                "options"
                (sprintf "the plan names taps that are not ActivationNames taps: %s" (String.concat ", " names))

/// The KV cache of one stage's layers (CpuReferenceSemantics, "The KV cache"): per layer, K after k_norm and RoPE and
/// V raw, each [MaxPos, KvHeads * HeadDim] in the ops' DType, row p for position p. Rows 0 .. Length - 1 are valid;
/// rows beyond may hold anything and are never read. A cache from KvCache.create holds tensors of its ops, and
/// Dispose returns them (IForwardOps.ReleaseCache); one built over tensors the caller owns releases nothing.
[<Sealed>]
type KvCache<'t>(firstLayer: int, keys: 't[], values: 't[], maxPos: int, cols: int, release: 't -> unit) =
    let mutable length = 0
    let mutable disposed = false

    let check () =
        if disposed then
            raise (ObjectDisposedException("KvCache", "the cache was disposed"))

    /// A cache over tensors the caller owns: Dispose releases nothing.
    new(firstLayer: int, keys: 't[], values: 't[], maxPos: int, cols: int) =
        new KvCache<'t>(firstLayer, keys, values, maxPos, cols, ignore)

    /// The first model layer this cache holds.
    member _.FirstLayer = firstLayer

    member _.LayerCount = keys.Length

    /// Rows per layer.
    member _.MaxPos = maxPos

    /// KvHeads * HeadDim.
    member _.Cols = cols

    /// The positions written so far: the next forward starts at a position at most this. The forward sets it; set it
    /// directly only to rewind (a replay from an earlier position) or after filling rows another way, such as
    /// restoring a snapshot. 0 .. MaxPos.
    member _.Length
        with get () = length
        and set v =
            if v < 0 || v > maxPos then
                raise (ArgumentOutOfRangeException("Length", sprintf "length %d outside 0 .. %d" v maxPos))

            length <- v

    /// K of model layer `layer`.
    member _.Keys(layer: int) =
        check ()

        if layer < firstLayer || layer >= firstLayer + keys.Length then
            raise (ArgumentOutOfRangeException("layer", sprintf "layer %d is not in this cache" layer))

        keys[layer - firstLayer]

    /// V of model layer `layer`.
    member _.Values(layer: int) =
        check ()

        if layer < firstLayer || layer >= firstLayer + values.Length then
            raise (ArgumentOutOfRangeException("layer", sprintf "layer %d is not in this cache" layer))

        values[layer - firstLayer]

    /// True once Dispose has run.
    member _.IsDisposed = disposed

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true

                for t in keys do
                    release t

                for t in values do
                    release t

[<RequireQualifiedAccess>]
module KvCache =
    /// An empty cache for a planned stage's layers with `maxPos` rows each (at most ops.MaxContext). Dispose it to
    /// return its tensors to the ops.
    let create (ops: IForwardOps<'t>) (model: Qwen3Model) (stage: Stage) (maxPos: int) : KvCache<'t> =
        let c = model.Config
        Stage.validate c stage

        if maxPos < 1 || maxPos > ops.MaxContext then
            invalidArg "maxPos" (sprintf "%d positions; the ops hold 1 .. %d" maxPos ops.MaxContext)

        let made = List<'t>()

        let make () =
            Array.init stage.LayerCount (fun i ->
                let t = ops.NewCache(stage, stage.FirstLayer + i, maxPos)
                made.Add t
                t)

        try
            let keys = make ()
            let values = make ()
            new KvCache<'t>(stage.FirstLayer, keys, values, maxPos, c.KvDim, ops.ReleaseCache)
        with _ ->
            for t in made do
                ops.ReleaseCache t

            reraise ()

    /// SHA-256 of the valid rows: tag `lrm.kvcache/1`, then the first layer, layer count, length and dtype code as
    /// u32 little-endian, then per layer the K rows and the V rows 0 .. Length - 1 as their little-endian bytes. Two
    /// caches of one class have the same digest exactly when their valid rows are bit-identical (C2: fast rebuild
    /// against exact replay). Only the valid rows are read back (IForwardOps.ReadRows).
    let digest (ops: IForwardOps<'t>) (cache: KvCache<'t>) : ContentId =
        use h = new Hash.Incremental()
        h.Append(Text.Encoding.ASCII.GetBytes "lrm.kvcache/1\000")

        let u32 (v: int) =
            h.Append(BitConverter.GetBytes(uint32 v))

        u32 cache.FirstLayer
        u32 cache.LayerCount
        u32 cache.Length
        u32 (int (DType.code ops.DType))

        if cache.Length > 0 then
            for l in cache.FirstLayer .. cache.FirstLayer + cache.LayerCount - 1 do
                for t in [ cache.Keys l; cache.Values l ] do
                    h.Append((ops.ReadRows(t, 0, cache.Length)).ToBytes())

        h.Finish()

/// The input of a stage: token ids for the first stage, the residual stream [T, hidden] for the others.
[<RequireQualifiedAccess>]
type StageInput<'t> =
    | Ids of int[]
    | Hidden of 't

/// The taps of a forward, in the order the forward produces them: snapshot handles (IForwardOps.Tap). They stay valid
/// until the next forward of the same stage on the ops ("Handle lifetimes"), so read them before that.
[<Sealed>]
type TapStore<'t>(filter: TapFilter) =
    let taps = List<string * 't>()

    member _.Filter = filter

    member _.Wants(name: string) = filter.Wants name

    member _.Add(name: string, x: 't) = taps.Add(name, x)

    /// The recorded taps, in forward order.
    member _.Taps = List.ofSeq taps

    member _.Count = taps.Count

/// The options of one forward call.
type ForwardOptions<'t> =
    {
        /// Which rows the head runs on (ignored by a stage without the head).
        Head: HeadRows
        /// Where taps go; None records nothing and asks the ops for no snapshot.
        Taps: TapStore<'t> option
        /// The greedy step this forward is (0 for the prefill), for telemetry only.
        Step: int option
    }

[<RequireQualifiedAccess>]
module ForwardOptions =
    /// Logits for every row, no taps.
    let all<'t> : ForwardOptions<'t> =
        {
            Head = HeadRows.All
            Taps = None
            Step = None
        }

    /// Logits for the last row, no taps.
    let last<'t> : ForwardOptions<'t> =
        {
            Head = HeadRows.Last
            Taps = None
            Step = None
        }

    /// Logits for every row, recording the taps into `store`.
    let withTaps (store: TapStore<'t>) =
        {
            Head = HeadRows.All
            Taps = Some store
            Step = None
        }
