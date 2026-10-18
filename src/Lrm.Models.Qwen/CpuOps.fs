namespace Lrm.Models.Qwen

open System
open System.Collections.Generic
open System.Numerics
open System.Runtime.CompilerServices
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// IForwardOps on the F# CPU reference (ForwardOpsAndQwenModel contract): cpu-f32 with 'T = float32 and cpu-f64 with
/// 'T = float. Handles are HostTensor<'T>; every op is the CpuRef op of the same name, so the results are CpuRef's
/// bits, the same at every thread count. Weights come from a verified Qwen3Weights, copied out of the mapping once
/// and shared. The KV cache is in 'T (fp64 in cpu-f64, CpuReferenceSemantics).
///
/// CpuOps enforces the plan and the handle lifetimes of IForwardOps exactly, although host memory would not need
/// them: a call outside the plan, an op outside a forward, or a handle used after its lifetime raises
/// ForwardException. Code that runs on CpuOps therefore runs on a GPU executor's static plan too.
[<Sealed>]
type CpuOps<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (weights: Qwen3Weights, options: OpsOptions) =
    let dtype, backend =
        if typeof<'T> = typeof<float32> then
            DType.F32, ExecutionBackend.CpuF32
        elif typeof<'T> = typeof<float> then
            DType.F64, ExecutionBackend.CpuF64
        else
            invalidArg "'T" (sprintf "CpuOps computes in float32 or float, not %s" typeof<'T>.Name)

    let config = weights.Config
    do OpsOptions.validate config options
    let exec = CpuExec.WithThreads options.Threads
    let table = RopeConstants.table exec config options.MaxContext
    let scale = CpuRef.attentionScale<'T> config.HeadDim
    let xclass = ExecutionClass.forCpu backend
    let w (name: string) = weights.Bf16 name

    // The lifetime state (IForwardOps, "Handle lifetimes"). Handles are compared by reference.
    let tombstone = obj ()
    let dead = ConditionalWeakTable<HostTensor<'T>, obj>()

    let refSet () =
        HashSet<HostTensor<'T>>(HashIdentity.Reference)

    let made = List<HostTensor<'T>>()
    let tapped = refSet ()
    let kept = Dictionary<int, HostTensor<'T> list>()
    let caches = Dictionary<HostTensor<'T>, int>(HashIdentity.Reference)
    let cacheCount = Dictionary<int, int>()
    let mutable current: ForwardCall option = None
    let mutable upload: HostTensor<'T> option = None
    let mutable disposed = false

    let kill (x: HostTensor<'T>) = dead.AddOrUpdate(x, tombstone)

    let live () =
        if disposed then
            raise (ObjectDisposedException("CpuOps", "the ops were disposed"))

    let alive (what: string) (x: HostTensor<'T>) =
        live ()

        if isNull (box x) then
            nullArg what

        let mutable found: obj = null

        if dead.TryGetValue(x, &found) then
            raise (
                ForwardException(
                    sprintf "%s: the handle is no longer valid (ForwardOpsAndQwenModel, \"Handle lifetimes\")" what
                )
            )

        x

    let check (what: string) (x: HostTensor<'T>) =
        alive what x |> ignore

        if x.Shape.Rank <> 2 then
            invalidArg what (sprintf "%s is %O; handles are [rows, cols]" what x.Shape)

        x

    let inside (op: string) =
        live ()

        if current.IsNone then
            raise (ForwardException(sprintf "%s outside a forward (BeginForward .. EndForward)" op))

    let outside (op: string) =
        live ()

        if current.IsSome then
            raise (ForwardException(sprintf "%s inside a forward" op))

    let made' (x: HostTensor<'T>) =
        made.Add x
        x

    let cacheTensor (what: string) (x: HostTensor<'T>) =
        let x = check what x

        if not (caches.ContainsKey x) then
            raise (ForwardException(sprintf "%s is not a cache tensor of these ops (NewCache)" what))

        x

    let planned (stage: Stage) =
        if not (options.Stages |> List.contains stage) then
            raise (
                ForwardException(
                    sprintf
                        "stage %d (layers %d .. %d) is not in the ops' plan (OpsOptions.Stages)"
                        stage.Index
                        stage.FirstLayer
                        stage.LastLayer
                )
            )

    /// The weights these ops read.
    member _.Weights = weights

    member _.Options = options

    /// The pinned RoPE table (T_spec) for positions 0 .. MaxContext - 1.
    member _.RopeTable = table

    interface IForwardOps<HostTensor<'T>> with
        member _.ExecutionClass = xclass
        member _.DType = dtype
        member _.MaxContext = options.MaxContext
        member _.Rows x = (check "x" x).Rows
        member _.Cols x = (check "x" x).Cols

        member _.BeginForward call =
            outside "BeginForward"
            planned call.Stage

            if call.Rows < 1 || call.Rows > options.MaxRows then
                raise (
                    ForwardException(
                        sprintf "%d rows; the ops' plan runs 1 .. %d rows per forward" call.Rows options.MaxRows
                    )
                )

            if call.Pos0 < 0 || call.Pos0 + call.Rows > options.MaxContext then
                raise (
                    ForwardException(
                        sprintf
                            "positions %d .. %d; the ops hold 0 .. %d"
                            call.Pos0
                            (call.Pos0 + call.Rows - 1)
                            (options.MaxContext - 1)
                    )
                )

            if not (options.Taps.Covers call.Taps) then
                raise (ForwardException(sprintf "the taps %A are not in the ops' plan (%A)" call.Taps options.Taps))

            // The previous result and taps of this stage end here.
            match kept.TryGetValue call.Stage.Index with
            | true, xs ->
                for x in xs do
                    kill x

                kept.Remove call.Stage.Index |> ignore
            | _ -> ()

            made.Clear()
            tapped.Clear()
            current <- Some call

        member _.EndForward result =
            inside "EndForward"
            let call = current.Value
            let keep = refSet ()

            match result with
            | Some r ->
                keep.Add r |> ignore

                for t in tapped do
                    keep.Add t |> ignore
            | None -> ()

            for x in made do
                if not (keep.Contains x) then
                    kill x

            kept[call.Stage.Index] <- List.ofSeq keep

            match upload with
            | Some u ->
                kill u
                upload <- None
            | None -> ()

            made.Clear()
            tapped.Clear()
            current <- None

        member _.Embed ids =
            inside "Embed"
            made' (CpuRef.embed<'T> exec (w weights.Params.Embed) ids)

        member _.RmsNorm(x, weight, eps) =
            inside "RmsNorm"
            made' (CpuRef.rmsNorm exec (check "x" x) (w weight) ('T.CreateTruncating eps))

        member _.MatMul(x, weight) =
            inside "MatMul"
            made' (CpuRef.matmulBf16W exec (check "x" x) (w weight))

        member _.Rope(x, heads, pos0) =
            inside "Rope"
            made' (CpuRef.ropeApply exec table (check "x" x) heads pos0)

        member _.NewCache(stage, layer, rows) =
            outside "NewCache"
            planned stage

            if layer < stage.FirstLayer || layer > stage.LastLayer then
                invalidArg "layer" (sprintf "layer %d is not in stage %d" layer stage.Index)

            if rows < 1 || rows > options.MaxContext then
                invalidArg "rows" (sprintf "%d rows; the ops hold 1 .. %d positions" rows options.MaxContext)

            let n =
                match cacheCount.TryGetValue stage.Index with
                | true, n -> n
                | _ -> 0

            if n >= 2 * options.Caches * stage.LayerCount then
                raise (
                    ForwardException(
                        sprintf
                            "stage %d holds %d caches already, the plan's limit (OpsOptions.Caches); dispose one first"
                            stage.Index
                            options.Caches
                    )
                )

            let t = HostTensor.zeros<'T> dtype (Shape.matrix rows config.KvDim)
            caches[t] <- stage.Index
            cacheCount[stage.Index] <- n + 1
            t

        member _.ReleaseCache x =
            // After Dispose every cache is released already.
            if not disposed then
                outside "ReleaseCache"

                match caches.TryGetValue x with
                | true, s ->
                    caches.Remove x |> ignore
                    cacheCount[s] <- cacheCount[s] - 1
                    kill x
                | _ -> raise (ForwardException "ReleaseCache: not a live cache tensor of these ops")

        member _.KvWrite(cache, x, pos0) =
            inside "KvWrite"
            CpuRef.copyRows (check "x" x) 0 (cacheTensor "cache" cache) pos0 x.Rows

        member _.Attention(q, k, v, shape, past) =
            inside "Attention"
            made' (CpuRef.attention exec (check "q" q) (cacheTensor "k" k) (cacheTensor "v" v) shape past scale)

        member _.SiluMul(gate, up) =
            inside "SiluMul"
            made' (CpuRef.siluMul exec (alive "gate" gate) (alive "up" up))

        member _.Add(a, b) =
            inside "Add"
            made' (CpuRef.add exec (alive "a" a) (alive "b" b))

        member _.LastRow x =
            inside "LastRow"
            let x = check "x" x
            made' (HostTensor.create dtype (Shape.matrix 1 x.Cols) (Array.sub x.Data ((x.Rows - 1) * x.Cols) x.Cols))

        member _.Argmax x = CpuRef.argmaxRows (check "x" x)

        // CpuRef results are fresh arrays that no later op changes. Only the cache tensors are written in place, and
        // the forward never taps a cache, so a tap is the handle itself; it is kept past EndForward like the result.
        member _.Tap(name, x) =
            inside "Tap"
            let x = alive name x

            if not (current.Value.Taps.Wants name) || not (options.Taps.Wants name) then
                raise (ForwardException(sprintf "Tap '%s': neither the forward nor the plan asks for it" name))

            tapped.Add x |> ignore
            x

        member _.Read x = (check "x" x).Clone() :> IHostTensor

        member _.ReadRows(x, row0, count) =
            let x = check "x" x

            if count < 1 || row0 < 0 || row0 + count > x.Rows then
                raise (
                    ArgumentOutOfRangeException(
                        "count",
                        sprintf "rows %d .. %d of a tensor of %d rows" row0 (row0 + count - 1) x.Rows
                    )
                )

            HostTensor.create dtype (Shape.matrix count x.Cols) (Array.sub x.Data (row0 * x.Cols) (count * x.Cols))
            :> IHostTensor

        member _.Upload x =
            outside "Upload"

            if x.Desc.DType <> dtype then
                invalidArg
                    "x"
                    (sprintf
                        "%s is %s; these ops hold %s"
                        (string x.Desc.Shape)
                        (DType.name x.Desc.DType)
                        (DType.name dtype))

            if x.Desc.Shape.Rank <> 2 then
                invalidArg "x" (sprintf "%O is not [rows, cols]" x.Desc.Shape)

            upload |> Option.iter kill
            let t = HostTensor.ofBytes<'T> dtype x.Desc.Shape (ReadOnlySpan(x.ToBytes()))
            upload <- Some t
            t

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                made.Clear()
                tapped.Clear()
                kept.Clear()
                caches.Clear()
                cacheCount.Clear()
                upload <- None
                current <- None

/// Constructors for the CPU ops (cpu-f32 and cpu-f64). The options are the plan (OpsOptions.validate).
[<RequireQualifiedAccess>]
module CpuOps =
    /// cpu-f32: CpuRef F32.
    let f32 (weights: Qwen3Weights) (options: OpsOptions) =
        new CpuOps<float32>(weights, options) :> IForwardOps<HostTensor<float32>>

    /// cpu-f64: CpuRef F64.
    let f64 (weights: Qwen3Weights) (options: OpsOptions) =
        new CpuOps<float>(weights, options) :> IForwardOps<HostTensor<float>>
