namespace Lrm.Tensor

open System
open Lrm.Core

/// How a kernel's output is judged (KernelBounds contract).
[<RequireQualifiedAccess>]
type KernelBoundKind =
    /// The output bits must equal CpuRef F32's (and so CpuDevice's) bit for bit.
    | BitExact
    /// relL2(gpu, CpuRef64) <= max(Floor, Factor * relL2(CpuRef32, CpuRef64)), per case.
    | Relative

[<RequireQualifiedAccess>]
module KernelBoundKind =
    /// "exact" or "relative", as written to telemetry.
    let name =
        function
        | KernelBoundKind.BitExact -> "exact"
        | KernelBoundKind.Relative -> "relative"

/// The bound of one catalog kernel.
type KernelBound =
    {
        /// The catalog name, for example "matmul_bf16w".
        Kernel: string
        Kind: KernelBoundKind
        /// Why the kernel has this kind.
        Reason: string
    }

/// The judgement of one conformance case.
type KernelCheck =
    {
        Kernel: string
        Kind: KernelBoundKind
        /// BitExact: the number of elements whose bits differ. Relative: relL2(gpu, CpuRef64).
        Measured: float
        /// Relative: relL2(CpuRef32, CpuRef64), the reference's own float32 error, which E1 logs in conformance.case.
        /// BitExact: 0.
        ReferenceError: float
        /// BitExact: 0. Relative: max(Floor, Factor * ReferenceError).
        Bound: float
        Passed: bool
        Elements: int64
    }

/// Per-kernel conformance bounds (KernelBounds contract). The factor and floor change only through a contract-change
/// PR with evidence.
[<RequireQualifiedAccess>]
module KernelBounds =
    /// The factor on the reference's own float32 error.
    [<Literal>]
    let Factor = 2.0

    /// The smallest relative bound.
    [<Literal>]
    let Floor = 1e-6

    let private exact kernel reason =
        {
            Kernel = kernel
            Kind = KernelBoundKind.BitExact
            Reason = reason
        }

    let private relative kernel reason =
        {
            Kernel = kernel
            Kind = KernelBoundKind.Relative
            Reason = reason
        }

    /// Every kernel of the v1 catalog (KernelAbiAndCatalog), in catalog order. A kernel missing here has no bound, and
    /// the conformance harness refuses it.
    let all =
        [
            exact "scale_f32" "one float32 multiply per element"
            exact "table_echo" "copies table words"
            exact "args_echo" "copies argument words"
            exact "groups_echo" "writes group indices"
            exact "bf16_unpack" "bf16 widening is exact"
            exact "embed_gather_bf16" "gathers rows and widens bf16 exactly"
            relative "rmsnorm_rows" "a reduction plus rsqrt"
            relative "rope_apply" "products and a sum per element from the pinned table"
            relative "silu_mul" "exp"
            exact "add" "one float32 add per element"
            exact "copy_rows" "copies 4-byte elements"
            exact "argmax_rows" "the index is exact; ties take the lowest index and a NaN row gives 0xFFFFFFFF"
            relative "matmul_bf16w" "a length-k reduction"
            relative "attention_gqa_causal" "online softmax over 64-key blocks"
        ]

    let tryFind (kernel: string) =
        all |> List.tryFind (fun b -> b.Kernel = kernel)

    /// The bound of a catalog kernel; raises ArgumentException for a kernel without one.
    let find (kernel: string) =
        match tryFind kernel with
        | Some b -> b
        | None -> invalidArg "kernel" (sprintf "kernel '%s' has no KernelBounds entry" kernel)

    /// max(Floor, Factor * referenceError). NaN when referenceError is NaN, so the case fails.
    let relativeBound (referenceError: float) =
        if Double.IsNaN referenceError then
            nan
        else
            max Floor (Factor * referenceError)

    /// Judges a Relative kernel's case from the GPU output and both CpuRef outputs, widened to float64.
    let checkRelative
        (kernel: string)
        (gpu: ReadOnlySpan<float>)
        (cpuRef32: ReadOnlySpan<float>)
        (cpuRef64: ReadOnlySpan<float>)
        =
        let b = find kernel

        if b.Kind <> KernelBoundKind.Relative then
            invalidArg "kernel" (sprintf "%s is bit-exact; use checkExact" kernel)

        let referenceError = Metrics.relL2 cpuRef32 cpuRef64
        let bound = relativeBound referenceError
        let measured = Metrics.relL2 gpu cpuRef64

        {
            Kernel = kernel
            Kind = b.Kind
            Measured = measured
            ReferenceError = referenceError
            Bound = bound
            Passed = not (Double.IsNaN measured) && not (Double.IsNaN bound) && measured <= bound
            Elements = int64 gpu.Length
        }

    /// Judges a BitExact kernel's case: the outputs' bytes, compared per element of `elementSize` bytes.
    let checkExact (kernel: string) (elementSize: int) (gpu: ReadOnlySpan<byte>) (cpuRef32: ReadOnlySpan<byte>) =
        let b = find kernel

        if b.Kind <> KernelBoundKind.BitExact then
            invalidArg "kernel" (sprintf "%s has a relative bound; use checkRelative" kernel)

        if
            gpu.Length <> cpuRef32.Length
            || elementSize < 1
            || gpu.Length % elementSize <> 0
        then
            invalidArg "gpu" (sprintf "%d and %d bytes in %d-byte elements" gpu.Length cpuRef32.Length elementSize)

        let mutable differing = 0L

        for i in 0..elementSize .. gpu.Length - 1 do
            if not (gpu.Slice(i, elementSize).SequenceEqual(cpuRef32.Slice(i, elementSize))) then
                differing <- differing + 1L

        {
            Kernel = kernel
            Kind = b.Kind
            Measured = float differing
            ReferenceError = 0.0
            Bound = 0.0
            Passed = (differing = 0L)
            Elements = int64 (gpu.Length / elementSize)
        }

    /// Judges a case by the kernel's kind. Exact kernels compare against cpuRef32's bits (cpuRef64 is ignored and may
    /// be None); relative kernels need cpuRef64.
    let check (kernel: string) (gpu: IHostTensor) (cpuRef32: IHostTensor) (cpuRef64: IHostTensor option) =
        if not (gpu.Desc.Shape.Equals cpuRef32.Desc.Shape) then
            invalidArg "cpuRef32" (sprintf "shapes differ: %O and %O" gpu.Desc.Shape cpuRef32.Desc.Shape)

        match (find kernel).Kind with
        | KernelBoundKind.BitExact ->
            if gpu.Desc.DType <> cpuRef32.Desc.DType then
                invalidArg "cpuRef32" "a bit-exact comparison needs one dtype"

            checkExact kernel gpu.Desc.ElementSize (ReadOnlySpan(gpu.ToBytes())) (ReadOnlySpan(cpuRef32.ToBytes()))
        | KernelBoundKind.Relative ->
            match cpuRef64 with
            | None -> invalidArg "cpuRef64" (sprintf "%s needs the CpuRef F64 output" kernel)
            | Some r64 ->
                checkRelative
                    kernel
                    (ReadOnlySpan(gpu.ToFloat64()))
                    (ReadOnlySpan(cpuRef32.ToFloat64()))
                    (ReadOnlySpan(r64.ToFloat64()))

    /// The attributes a conformance.case record carries for a check: kernel, bound_kind, measured, bound, ref_error and
    /// elements.
    let attrs (c: KernelCheck) =
        [
            Attr.str TelemetryKeys.Kernel c.Kernel
            Attr.str "bound_kind" (KernelBoundKind.name c.Kind)
            Attr.float "measured" c.Measured
            Attr.float "bound" c.Bound
            Attr.float "ref_error" c.ReferenceError
            Attr.int64 "elements" c.Elements
        ]
