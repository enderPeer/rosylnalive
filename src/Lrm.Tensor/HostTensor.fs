namespace Lrm.Tensor

open System
open System.Runtime.InteropServices
open Lrm.Core

/// A dtype-erased view of a host tensor, for code that handles every element type alike: comparators, writers and
/// reports (HostTensorAndSafetensors contract).
type IHostTensor =
    /// The element type, shape and (packed row-major) strides.
    abstract Desc: TensorDesc
    /// Every element widened exactly to float64. Integers convert exactly up to 2^53 in magnitude.
    abstract ToFloat64: unit -> float[]
    /// The elements as little-endian bytes, in row-major order.
    abstract ToBytes: unit -> byte[]

/// The host element type that carries each DType.
[<RequireQualifiedAccess>]
module ElementType =
    /// The .NET type of one element: float32, float, uint16 (BF16 and F16 bits), int32, uint32, byte or int64.
    let ofDType (dtype: DType) : Type =
        match dtype with
        | DType.F32 -> typeof<float32>
        | DType.F64 -> typeof<float>
        | DType.BF16
        | DType.F16 -> typeof<uint16>
        | DType.I32 -> typeof<int32>
        | DType.U32 -> typeof<uint32>
        | DType.U8 -> typeof<byte>
        | DType.I64 -> typeof<int64>

    /// Raises ArgumentException unless 'T carries `dtype`.
    let check<'T> (dtype: DType) =
        if ofDType dtype <> typeof<'T> then
            invalidArg "dtype" (sprintf "%s elements are not carried by %s" (DType.name dtype) typeof<'T>.Name)

/// A contiguous row-major tensor in host memory (HostTensorAndSafetensors contract): a dtype, a shape and an array of
/// elements. The array is owned by the tensor and is not copied; callers that keep it must not change it while the
/// tensor is in use. BF16 and F16 elements are their raw uint16 bits.
[<Sealed>]
type HostTensor<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (dtype: DType, shape: Shape, data: 'T[]) =
    do
        if isNull data then
            nullArg "data"

        ElementType.check<'T> dtype

        if int64 data.Length <> shape.Elements then
            invalidArg "data" (sprintf "%d elements for shape %O" data.Length shape)

    let desc = TensorDesc.Contiguous(dtype, shape)

    member _.DType = dtype
    member _.Shape = shape
    member _.Desc = desc

    /// The elements, row-major. Not a copy.
    member _.Data = data

    /// The element count.
    member _.Length = data.Length

    /// The first dimension (1 for a scalar).
    member _.Rows = int shape.Rows

    /// The product of the other dimensions: the row length.
    member _.Cols = int shape.Cols

    /// The elements as a span.
    member _.Span = Span<'T>(data)

    /// Row r of a row-major tensor, as a span of Cols elements.
    member this.Row(r: int) =
        let cols = this.Cols

        if r < 0 || r >= this.Rows then
            raise (ArgumentOutOfRangeException("r", sprintf "row %d of %d" r this.Rows))

        Span<'T>(data, r * cols, cols)

    /// The same elements under another shape with the same element count.
    member _.Reshape(newShape: Shape) = HostTensor<'T>(dtype, newShape, data)

    /// A tensor with a copy of the elements.
    member _.Clone() =
        HostTensor<'T>(dtype, shape, Array.copy data)

    override _.ToString() = desc.ToString()

    interface IHostTensor with
        member _.Desc = desc

        member _.ToBytes() =
            MemoryMarshal.AsBytes(ReadOnlySpan<'T>(data)).ToArray()

        member _.ToFloat64() =
            let n = data.Length
            let r = Array.zeroCreate<float> n
            let bytes = MemoryMarshal.AsBytes(ReadOnlySpan<'T>(data))

            match dtype with
            | DType.F32 ->
                let s = MemoryMarshal.Cast<byte, float32> bytes

                for i in 0 .. n - 1 do
                    r[i] <- float s[i]
            | DType.F64 -> MemoryMarshal.Cast<byte, float>(bytes).CopyTo(Span r)
            | DType.BF16 ->
                let s = MemoryMarshal.Cast<byte, uint16> bytes

                for i in 0 .. n - 1 do
                    r[i] <- float (Bf16.widen s[i])
            | DType.F16 ->
                let s = MemoryMarshal.Cast<byte, uint16> bytes

                for i in 0 .. n - 1 do
                    r[i] <- float (F16.widen s[i])
            | DType.I32 ->
                let s = MemoryMarshal.Cast<byte, int32> bytes

                for i in 0 .. n - 1 do
                    r[i] <- float s[i]
            | DType.U32 ->
                let s = MemoryMarshal.Cast<byte, uint32> bytes

                for i in 0 .. n - 1 do
                    r[i] <- float s[i]
            | DType.U8 ->
                for i in 0 .. n - 1 do
                    r[i] <- float bytes[i]
            | DType.I64 ->
                let s = MemoryMarshal.Cast<byte, int64> bytes

                for i in 0 .. n - 1 do
                    if abs s[i] > (1L <<< 53) then
                        invalidOp (sprintf "int64 element %d is not exact in float64" s[i])

                    r[i] <- float s[i]

            r

/// Constructors for host tensors.
[<RequireQualifiedAccess>]
module HostTensor =
    /// A tensor over `data` (not copied).
    let create<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (dtype: DType)
        (shape: Shape)
        (data: 'T[])
        =
        HostTensor<'T>(dtype, shape, data)

    /// A tensor of zero bits.
    let zeros<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (dtype: DType)
        (shape: Shape)
        =
        HostTensor<'T>(dtype, shape, Array.zeroCreate shape.ElementCount)

    let f32 (shape: Shape) (data: float32[]) = create DType.F32 shape data
    let f64 (shape: Shape) (data: float[]) = create DType.F64 shape data

    /// bfloat16 bits.
    let bf16 (shape: Shape) (bits: uint16[]) = create DType.BF16 shape bits

    let i32 (shape: Shape) (data: int32[]) = create DType.I32 shape data
    let u32 (shape: Shape) (data: uint32[]) = create DType.U32 shape data
    let i64 (shape: Shape) (data: int64[]) = create DType.I64 shape data
    let u8 (shape: Shape) (data: byte[]) = create DType.U8 shape data

    /// A tensor whose elements are copied from little-endian bytes. Raises ArgumentException when the byte count is
    /// not elements times the element size, or 'T does not carry `dtype`.
    let ofBytes<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (dtype: DType)
        (shape: Shape)
        (bytes: ReadOnlySpan<byte>)
        =
        ElementType.check<'T> dtype

        if int64 bytes.Length <> shape.Elements * int64 (DType.size dtype) then
            invalidArg "bytes" (sprintf "%d bytes for %s%O" bytes.Length (DType.name dtype) shape)

        let data = Array.zeroCreate<'T> shape.ElementCount
        bytes.CopyTo(MemoryMarshal.AsBytes(Span<'T>(data)))
        HostTensor<'T>(dtype, shape, data)

    /// Every element widened exactly to float64.
    let toFloat64 (t: #IHostTensor) = t.ToFloat64()

    /// The elements as little-endian bytes.
    let toBytes (t: #IHostTensor) = t.ToBytes()
