namespace Lrm.Models.Qwen

open System
open System.Numerics
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// IForwardOps on the F# CPU reference (ForwardOpsAndQwenModel contract): cpu-f32 with 'T = float32 and cpu-f64 with
/// 'T = float. Handles are HostTensor<'T>; every op is the CpuRef op of the same name, so the results are CpuRef's
/// bits, the same at every thread count. Weights come from a verified Qwen3Weights, copied out of the mapping once
/// and shared. The KV cache is in 'T (fp64 in cpu-f64, CpuReferenceSemantics).
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

    do
        if options.Threads < 1 then
            invalidArg "options" (sprintf "%d threads" options.Threads)

        if options.MaxContext < 1 then
            invalidArg "options" (sprintf "max context %d" options.MaxContext)

    let exec = CpuExec.WithThreads options.Threads
    let config = weights.Config
    let table = RopeConstants.table exec config options.MaxContext
    let scale = CpuRef.attentionScale<'T> config.HeadDim
    let xclass = ExecutionClass.forCpu backend
    let w (name: string) = weights.Bf16 name

    let check (what: string) (x: HostTensor<'T>) =
        if x.Shape.Rank <> 2 then
            invalidArg what (sprintf "%s is %O; handles are [rows, cols]" what x.Shape)

        x

    /// The weights these ops read.
    member _.Weights = weights

    member _.Options = options

    /// The pinned RoPE table (T_spec) for positions 0 .. MaxContext - 1.
    member _.RopeTable = table

    interface IForwardOps<HostTensor<'T>> with
        member _.ExecutionClass = xclass
        member _.DType = dtype
        member _.MaxContext = options.MaxContext
        member _.Rows x = x.Rows

        member _.Embed ids =
            CpuRef.embed<'T> exec (w weights.Params.Embed) ids

        member _.RmsNorm(x, weight, eps) =
            CpuRef.rmsNorm exec (check "x" x) (w weight) ('T.CreateTruncating eps)

        member _.MatMul(x, weight) =
            CpuRef.matmulBf16W exec (check "x" x) (w weight)

        member _.Rope(x, heads, pos0) =
            CpuRef.ropeApply exec table (check "x" x) heads pos0

        member _.NewCache(rows, cols) =
            HostTensor.zeros<'T> dtype (Shape.matrix rows cols)

        member _.KvWrite(cache, x, pos0) =
            CpuRef.copyRows (check "x" x) 0 cache pos0 x.Rows

        member _.Attention(q, k, v, shape, past) =
            CpuRef.attention exec (check "q" q) k v shape past scale

        member _.SiluMul(gate, up) = CpuRef.siluMul exec gate up
        member _.Add(a, b) = CpuRef.add exec a b

        member _.LastRow x =
            let x = check "x" x
            HostTensor.create dtype (Shape.matrix 1 x.Cols) (Array.sub x.Data ((x.Rows - 1) * x.Cols) x.Cols)

        member _.Argmax x = CpuRef.argmaxRows (check "x" x)

        // CpuRef results are fresh arrays that no later op changes. Only the cache tensors are written in place, and
        // the forward never taps a cache, so a tap is the handle itself.
        member _.Tap(_, x) = x

        member _.Read x = x.Clone() :> IHostTensor

        member _.Upload x =
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

            HostTensor.ofBytes<'T> dtype x.Desc.Shape (ReadOnlySpan(x.ToBytes()))

/// Constructors for the CPU ops (cpu-f32 and cpu-f64).
[<RequireQualifiedAccess>]
module CpuOps =
    /// cpu-f32: CpuRef F32.
    let f32 (weights: Qwen3Weights) (options: OpsOptions) =
        CpuOps<float32>(weights, options) :> IForwardOps<HostTensor<float32>>

    /// cpu-f64: CpuRef F64.
    let f64 (weights: Qwen3Weights) (options: OpsOptions) =
        CpuOps<float>(weights, options) :> IForwardOps<HostTensor<float>>
