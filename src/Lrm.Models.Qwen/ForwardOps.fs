namespace Lrm.Models.Qwen

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// The op seam of the Qwen3 forward (ForwardOpsAndQwenModel contract). The forward is written once over this
/// interface; CpuOps implements it on CpuRef (F32 and F64), and the GPU executor (X1) on the kernel catalog.
///
/// 't is the backend's activation handle: a host tensor for CpuOps, a tensor token for a GPU. Every handle is a
/// row-major [rows, cols] tensor in the ops' DType (float32, or float64 for cpu-f64). An op returns its result
/// handle at once and never reads back to the host; only Argmax and Read return host data. The forward calls the
/// ops in one fixed order that depends only on the model, the stage, T and pos0, so a backend can record a static
/// plan. Weights are named by their safetensors names (Qwen3Params, aliases resolved); the ops are created over one
/// verified Qwen3Weights and resolve names themselves.
type IForwardOps<'t> =
    /// The class whose bits these ops produce. Bit-identity holds only within it (ExecutionClass contract).
    abstract ExecutionClass: ExecutionClass

    /// The element type of activations and caches: F32, or F64 for cpu-f64 (CpuReferenceSemantics).
    abstract DType: DType

    /// The positions the RoPE table covers; a cache holds at most this many.
    abstract MaxContext: int

    /// The rows of a handle.
    abstract Rows: x: 't -> int

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

    /// A new cache tensor of `rows` rows of zeros: one layer's K or V, [rows, KvHeads * HeadDim], in DType.
    abstract NewCache: rows: int * cols: int -> 't

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

    /// A snapshot of x that stays valid, unchanged, after later ops (a GPU copies it into its Taps arena when the op
    /// is recorded). Called only for the taps a caller asked for.
    abstract Tap: name: string * x: 't -> 't

    /// The elements of x on the host, [rows, cols] in DType: a copy the caller owns.
    abstract Read: x: 't -> IHostTensor

    /// A handle holding a host tensor [rows, cols] in DType (a stage boundary's residual, for example).
    abstract Upload: x: IHostTensor -> 't

/// The options a backend's ops are created with (OpsFactory, Commands/Backends.fs).
type OpsOptions =
    {
        /// Host threads, for CPU backends (CpuExec); the bits are the same for every count.
        Threads: int
        /// Positions the RoPE table covers (T_spec for 0 .. MaxContext - 1); caches hold at most this many.
        MaxContext: int
    }

[<RequireQualifiedAccess>]
module OpsOptions =
    /// 8 threads (the plan's limit for long CPU jobs on the shared machine) and the model's max_position_embeddings.
    let defaults (c: Qwen3Config) =
        {
            Threads = min 8 Environment.ProcessorCount
            MaxContext = c.MaxPositionEmbeddings
        }

/// A forward pass, a stage or a cache that breaks the forward's rules (ForwardOpsAndQwenModel contract).
type ForwardException(message: string) =
    inherit InvalidOperationException("forward: " + message)

/// A contiguous range of decoder layers of one pipeline stage (StagePipeline: C2 splits the model at a boundary). The
/// stage that starts at layer 0 also embeds; the stage that ends at the last layer also runs the final norm and the
/// head.
type Stage =
    {
        /// The stage's index in its pipeline: the `stage` attribute of its telemetry.
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

/// The KV cache of one stage's layers (CpuReferenceSemantics, "The KV cache"): per layer, K after k_norm and RoPE and
/// V raw, each [MaxPos, KvHeads * HeadDim] in the ops' DType, row p for position p. Rows 0 .. Length - 1 are valid;
/// rows beyond may hold anything and are never read.
[<Sealed>]
type KvCache<'t>(firstLayer: int, keys: 't[], values: 't[], maxPos: int, cols: int) =
    let mutable length = 0

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
        if layer < firstLayer || layer >= firstLayer + keys.Length then
            raise (ArgumentOutOfRangeException("layer", sprintf "layer %d is not in this cache" layer))

        keys[layer - firstLayer]

    /// V of model layer `layer`.
    member _.Values(layer: int) =
        if layer < firstLayer || layer >= firstLayer + values.Length then
            raise (ArgumentOutOfRangeException("layer", sprintf "layer %d is not in this cache" layer))

        values[layer - firstLayer]

[<RequireQualifiedAccess>]
module KvCache =
    /// An empty cache for a stage's layers with `maxPos` rows each (at most ops.MaxContext).
    let create (ops: IForwardOps<'t>) (model: Qwen3Model) (stage: Stage) (maxPos: int) : KvCache<'t> =
        let c = model.Config
        Stage.validate c stage

        if maxPos < 1 || maxPos > ops.MaxContext then
            invalidArg "maxPos" (sprintf "%d positions; the ops hold 1 .. %d" maxPos ops.MaxContext)

        let make () =
            Array.init stage.LayerCount (fun _ -> ops.NewCache(maxPos, c.KvDim))

        KvCache(stage.FirstLayer, make (), make (), maxPos, c.KvDim)

    /// SHA-256 of the valid rows: tag `lrm.kvcache/1`, then the first layer, layer count, length and dtype code as
    /// u32 little-endian, then per layer the K rows and the V rows 0 .. Length - 1 as their little-endian bytes. Two
    /// caches of one class have the same digest exactly when their valid rows are bit-identical (C2: fast rebuild
    /// against exact replay).
    let digest (ops: IForwardOps<'t>) (cache: KvCache<'t>) : ContentId =
        use h = new Hash.Incremental()
        h.Append(Text.Encoding.ASCII.GetBytes "lrm.kvcache/1\000")

        let u32 (v: int) =
            h.Append(BitConverter.GetBytes(uint32 v))

        u32 cache.FirstLayer
        u32 cache.LayerCount
        u32 cache.Length
        u32 (int (DType.code ops.DType))
        let size = DType.size ops.DType
        let rowBytes = cache.Cols * size

        for l in cache.FirstLayer .. cache.FirstLayer + cache.LayerCount - 1 do
            for t in [ cache.Keys l; cache.Values l ] do
                let bytes = (ops.Read t).ToBytes()
                h.Append(ReadOnlySpan(bytes, 0, cache.Length * rowBytes))

        h.Finish()

/// The input of a stage: token ids for the first stage, the residual stream [T, hidden] for the others.
[<RequireQualifiedAccess>]
type StageInput<'t> =
    | Ids of int[]
    | Hidden of 't

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
    /// Exact names, such as `L0.attn.q_rope` or `logits`.
    | Names of string list

    /// True when the filter asks for `name`.
    member f.Wants(name: string) =
        match f with
        | TapFilter.Off -> false
        | TapFilter.All -> true
        | TapFilter.Names names -> List.contains name names

    /// Parses `none`, `all` or a comma-separated list of tap names.
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

/// The taps one forward records, in the order the forward produces them: snapshot handles (IForwardOps.Tap).
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
