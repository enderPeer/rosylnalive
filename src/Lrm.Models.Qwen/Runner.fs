namespace Lrm.Models.Qwen

open System
open Lrm.Core
open Lrm.Tensor

/// The host results of one prefill.
type ForwardResult =
    {
        /// [T, vocab] for HeadRows.All, [1, vocab] for HeadRows.Last, in the backend's DType.
        Logits: IHostTensor
        /// The taps asked for, in forward (ActivationNames) order.
        Taps: (string * IHostTensor) list
    }

/// A model bound to one backend's ops, with the handle type hidden: what `lrm forward`, `lrm generate` and the
/// backend registry (Commands/Backends.fs) work with. Code that needs the op seam itself uses IForwardOps. The runner
/// owns its ops: Dispose releases them (a GPU's weights and heap), so callers `use` it.
type IQwenRunner =
    inherit IDisposable

    /// The registry name, such as cpu-f32.
    abstract Backend: string
    abstract ExecutionClass: ExecutionClass
    abstract DType: DType
    abstract Model: Qwen3Model
    abstract MaxContext: int

    /// A prefill of `ids` from position 0 into a new cache of exactly ids.Length positions, released before it
    /// returns; the results are on the host. The ops' plan must cover the rows and the taps.
    abstract Forward: ids: int[] * taps: TapFilter * head: HeadRows -> ForwardResult

    /// Greedy.run on these ops.
    abstract Greedy: prompt: int[] * options: GreedyOptions -> GreedyResult

[<RequireQualifiedAccess>]
module QwenRunner =
    /// A runner over any ops; it owns them and disposes them with itself.
    let ofOps (backend: string) (model: Qwen3Model) (ops: IForwardOps<'t>) : IQwenRunner =
        { new IQwenRunner with
            member _.Dispose() = ops.Dispose()

            member _.Backend = backend
            member _.ExecutionClass = ops.ExecutionClass
            member _.DType = ops.DType
            member _.Model = model
            member _.MaxContext = ops.MaxContext

            member _.Forward(ids, taps, head) =
                use cache = KvCache.create ops model (Stage.whole model.Config) (max 1 ids.Length)
                let store = TapStore<'t>(taps)

                let logits =
                    Forward.run
                        ops
                        model
                        cache
                        0
                        ids
                        {
                            Head = head
                            Taps = Some store
                            Step = None
                        }

                {
                    Logits = ops.Read logits
                    Taps = store.Taps |> List.map (fun (n, x) -> n, ops.Read x)
                }

            member _.Greedy(prompt, options) = Greedy.run ops model prompt options
        }

/// One entry of the forward-ops registry (Commands/Backends.fs): a backend name and how to bind it to verified
/// weights.
type OpsFactory =
    {
        /// The registry name, such as cpu-f32 or vulkan; for a CPU or GPU backend it is the ExecutionBackend name.
        Name: string
        Summary: string
        /// Binds the backend to verified weights under a plan (OpsOptions); the caller disposes the runner.
        Create: Qwen3Weights -> OpsOptions -> IQwenRunner
    }

/// The CPU backends of the registry.
[<RequireQualifiedAccess>]
module CpuBackends =
    /// cpu-f32: CpuRef F32 (the fp32 class every GPU is bounded against).
    let cpuF32 =
        {
            Name = ExecutionBackend.name ExecutionBackend.CpuF32
            Summary = "F# CPU reference in float32 (CpuRef F32)"
            Create = fun w o -> QwenRunner.ofOps "cpu-f32" w.Model (CpuOps.f32 w o)
        }

    /// cpu-f64: CpuRef F64 (the fp64 truth on our side, against torch64p).
    let cpuF64 =
        {
            Name = ExecutionBackend.name ExecutionBackend.CpuF64
            Summary = "F# CPU reference in float64 (CpuRef F64)"
            Create = fun w o -> QwenRunner.ofOps "cpu-f64" w.Model (CpuOps.f64 w o)
        }

    let all = [ cpuF32; cpuF64 ]
