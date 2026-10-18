namespace Lrm.Tensor

open System
open System.Buffers.Binary
open Lrm.Core

/// How a kernel's output is judged (KernelBounds contract).
[<RequireQualifiedAccess>]
type KernelBoundKind =
    /// The output bits must equal CpuRef F32's (and so CpuDevice's) bit for bit; for the fp32 arithmetic kernels, CpuRef
    /// F32 in the class's measured denormal mode, with any NaN matching any NaN.
    | BitExact
    /// relL2(gpu, CpuRef64) <= max(Floor, Factor * relL2(CpuRef32, CpuRef64)) + the flushed-denormal allowance, per
    /// case.
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
        /// True for the bit-exact kernels that do fp32 arithmetic (scale_f32, add): a NaN element matches any NaN, whatever
        /// its sign and payload, because GPUs return 0x7FFFFFFF for any NaN operand while CpuRef propagates the payload.
        /// The other bit-exact kernels compare every bit, NaN payloads included. The reference of every kernel that
        /// computes on fp32 values (these two, argmax_rows and the relative kernels) is CpuRef in the class's measured
        /// fp32 denormal mode; building it is the harness's job (KernelBounds contract).
        Fp32Arithmetic: bool
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
        /// BitExact: 0. Relative: max(Floor, Factor * ReferenceError) plus the flushed-denormal allowance (0 when no
        /// reference result is below 2^-126).
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

    /// The smallest normal float32, 2^-126. A class whose fp32 denormal mode is flush returns every result below it as
    /// zero of the same sign: an absolute error below 2^-126 per flushed value.
    [<Literal>]
    let SmallestNormalF32 = 1.1754943508222875e-38

    let private exact kernel reason =
        {
            Kernel = kernel
            Kind = KernelBoundKind.BitExact
            Fp32Arithmetic = false
            Reason = reason
        }

    let private exactArithmetic kernel reason =
        { exact kernel reason with
            Fp32Arithmetic = true
        }

    let private relative kernel reason =
        {
            Kernel = kernel
            Kind = KernelBoundKind.Relative
            Fp32Arithmetic = false
            Reason = reason
        }

    /// Every kernel of the v1 catalog (KernelAbiAndCatalog), in catalog order. A kernel missing here has no bound, and
    /// the conformance harness refuses it.
    let all =
        [
            exactArithmetic "scale_f32" "one float32 multiply per element"
            exact "table_echo" "copies table words"
            exact "args_echo" "copies argument words"
            exact "groups_echo" "writes group indices"
            exact "bf16_unpack" "bf16 widening is exact"
            exact "embed_gather_bf16" "gathers rows and widens bf16 exactly"
            relative "rmsnorm_rows" "a reduction plus rsqrt"
            relative "rope_apply" "products and a sum per element from the pinned table"
            relative "silu_mul" "exp"
            exactArithmetic "add" "one float32 add per element"
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

    /// The number of reference results a flushing class returns as zero: the elements with 0 < |r| < 2^-126.
    let flushableCount (cpuRef64: ReadOnlySpan<float>) =
        let mutable d = 0L

        for i in 0 .. cpuRef64.Length - 1 do
            let a = abs cpuRef64[i]

            if a > 0.0 && a < SmallestNormalF32 then
                d <- d + 1L

        d

    /// The flushed-denormal allowance as a relL2 term: 2^-126 * sqrt d / ||cpuRef64||2, where d is flushableCount. Each
    /// flushed result is an absolute error below 2^-126, so their L2 norm is below 2^-126 * sqrt d. It is 0 when d is 0,
    /// so it loosens nothing for results at or above 2^-126, and it is the same in every class. The sum of squares runs
    /// sequentially in increasing index order in float64.
    let flushAllowance (cpuRef64: ReadOnlySpan<float>) =
        let d = flushableCount cpuRef64

        if d = 0L then
            0.0
        else
            let mutable sq = 0.0

            for i in 0 .. cpuRef64.Length - 1 do
                sq <- sq + cpuRef64[i] * cpuRef64[i]

            SmallestNormalF32 * sqrt (float d) / sqrt sq

    /// Judges a Relative kernel's case from the GPU output and both CpuRef outputs, widened to float64. The bound is
    /// relativeBound referenceError plus flushAllowance cpuRef64.
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
        let bound = relativeBound referenceError + flushAllowance cpuRef64
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

    let private isNaN32 (element: ReadOnlySpan<byte>) =
        let bits = BinaryPrimitives.ReadUInt32LittleEndian element
        bits &&& 0x7F800000u = 0x7F800000u && bits &&& 0x007FFFFFu <> 0u

    /// Judges a BitExact kernel's case: the outputs' bytes, compared per element of `elementSize` bytes. For the fp32
    /// arithmetic kernels (Fp32Arithmetic) with 4-byte elements, an element where both sides are NaN matches whatever
    /// the bits; `cpuRef32` is then CpuRef F32 computed in the class's measured denormal mode (the caller's job).
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

        let nanByClass = b.Fp32Arithmetic && elementSize = 4
        let mutable differing = 0L

        for i in 0..elementSize .. gpu.Length - 1 do
            let g = gpu.Slice(i, elementSize)
            let r = cpuRef32.Slice(i, elementSize)

            if not (g.SequenceEqual r) && not (nanByClass && isNaN32 g && isNaN32 r) then
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
