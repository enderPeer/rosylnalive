namespace Lrm.Tensor.CpuReference

open System
open System.Runtime.ExceptionServices
open System.Threading.Tasks
open Lrm.Core

/// How many threads a CpuRef op may use (CpuReferenceSemantics contract). Work is split only over independent output
/// elements (rows, (row, head) pairs, output tiles); every reduction runs whole on one thread in its fixed order, so
/// results are bit-identical for every thread count.
[<Struct>]
type CpuExec =
    {
        /// At least 1. With 1, every op runs as a plain loop on the calling thread.
        Threads: int
    }

    /// One thread.
    static member Serial = { Threads = 1 }

    /// One thread per logical processor.
    static member Machine = { Threads = Environment.ProcessorCount }

    /// `n` threads (at least 1).
    static member WithThreads(n: int) =
        if n < 1 then
            invalidArg "n" (sprintf "%d threads" n)

        { Threads = n }

/// Parallel loops over independent outputs.
module internal Par =
    /// Runs body i for i in [0, count). With more than one thread the indices run concurrently, each exactly once.
    /// An exception from a body is rethrown as itself (not wrapped in AggregateException).
    let forEach (exec: CpuExec) (count: int) (body: int -> unit) =
        if exec.Threads <= 1 || count <= 1 then
            for i in 0 .. count - 1 do
                body i
        else
            try
                Parallel.For(0, count, ParallelOptions(MaxDegreeOfParallelism = exec.Threads), (fun i -> body i))
                |> ignore
            with :? AggregateException as e when e.InnerExceptions.Count >= 1 ->
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw()

    /// Runs body lo hi over consecutive chunks of [0, count) of at most `chunk` indices.
    let forChunks (exec: CpuExec) (count: int) (chunk: int) (body: int -> int -> unit) =
        let chunks = (count + chunk - 1) / chunk
        forEach exec chunks (fun c -> body (c * chunk) (min count ((c + 1) * chunk)))

/// The element types CpuRef computes in.
module internal Elt =
    /// F32 for float32, F64 for float; raises for any other type.
    let dtype<'T> () =
        if typeof<'T> = typeof<float32> then
            DType.F32
        elif typeof<'T> = typeof<float> then
            DType.F64
        else
            invalidArg "'T" (sprintf "CpuRef computes in float32 or float, not %s" typeof<'T>.Name)
