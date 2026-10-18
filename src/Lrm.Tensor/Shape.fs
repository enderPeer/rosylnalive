namespace Lrm.Tensor

open System
open Lrm.Core

/// A tensor shape (HostTensorAndSafetensors contract): dimensions, outermost first, each at least 0. Rank 0 is a scalar
/// with one element. A Shape is immutable: it keeps its own copy of the dimensions.
[<Sealed>]
type Shape private (dims: int64[]) =
    static let checkedElements (dims: int64[]) =
        let mutable n = 1L

        for d in dims do
            if d < 0L then
                invalidArg "dims" (sprintf "negative dimension %d" d)

            n <-
                try
                    Checked.(*) n d
                with :? OverflowException ->
                    invalidArg "dims" "the element count overflows int64"

        n

    let elements = checkedElements dims

    /// A shape from dimensions; raises ArgumentException for a negative dimension or an element count beyond int64.
    static member Of(dims: int64 seq) = Shape(Array.ofSeq dims)

    /// A shape from int dimensions.
    static member Of(dims: int seq) =
        Shape(dims |> Seq.map int64 |> Array.ofSeq)

    /// The scalar shape [].
    static member Scalar = Shape([||])

    /// Number of dimensions.
    member _.Rank = dims.Length

    /// Dimension i, outermost first.
    member _.Item(i: int) = dims[i]

    /// A copy of the dimensions.
    member _.Dims = Array.copy dims

    /// Product of the dimensions (1 for a scalar).
    member _.Elements = elements

    /// The element count as an int, for host arrays; raises when it does not fit.
    member _.ElementCount =
        if elements > int64 Int32.MaxValue then
            invalidOp (sprintf "%d elements do not fit a host array" elements)
        else
            int elements

    /// Packed row-major strides in elements: the last dimension has stride 1.
    member _.RowMajorStrides =
        let s = Array.zeroCreate<int64> dims.Length
        let mutable acc = 1L

        for i in dims.Length - 1 .. -1 .. 0 do
            s[i] <- acc
            acc <- acc * max 1L dims[i]

        s

    /// The first dimension, or 1 for a scalar: the row count of a row-major tensor.
    member _.Rows = if dims.Length = 0 then 1L else dims[0]

    /// The product of every dimension but the first (1 for rank 0 and 1): the row length of a row-major tensor.
    member _.Cols = if dims.Length <= 1 then 1L else Array.fold (*) 1L dims[1..]

    /// "[13, 64]" (invariant digits).
    override _.ToString() =
        "[" + String.Join(", ", dims |> Array.map Invariant.i64) + "]"

    override _.Equals(other: obj) =
        match other with
        | :? Shape as s -> s.Rank = dims.Length && Seq.forall2 (=) (s.Dims) dims
        | _ -> false

    override _.GetHashCode() =
        let mutable h = HashCode()

        for d in dims do
            h.Add d

        h.ToHashCode()

    interface IEquatable<Shape> with
        member this.Equals(other: Shape) = this.Equals(box other)

[<RequireQualifiedAccess>]
module Shape =
    /// A shape from int dimensions.
    let ofList (dims: int list) = Shape.Of(dims)

    /// A shape from int64 dimensions.
    let ofArray (dims: int64[]) = Shape.Of(dims)

    /// [rows, cols].
    let matrix (rows: int) (cols: int) = Shape.Of([ rows; cols ])

    /// [n].
    let vector (n: int) = Shape.Of([ n ])

/// The description of a tensor (HostTensorAndSafetensors contract): element type, shape and element strides. Strides
/// count elements, not bytes; the packed row-major strides make a tensor contiguous.
[<Sealed>]
type TensorDesc private (dtype: DType, shape: Shape, strides: int64[]) =
    do
        if strides.Length <> shape.Rank then
            invalidArg "strides" (sprintf "%d strides for a rank-%d shape" strides.Length shape.Rank)

        if strides |> Array.exists (fun s -> s < 0L) then
            invalidArg "strides" "negative stride"

    /// A contiguous row-major description.
    static member Contiguous(dtype: DType, shape: Shape) =
        TensorDesc(dtype, shape, shape.RowMajorStrides)

    /// A description with explicit element strides (one per dimension, each at least 0).
    static member Strided(dtype: DType, shape: Shape, strides: int64 seq) =
        TensorDesc(dtype, shape, Array.ofSeq strides)

    member _.DType = dtype
    member _.Shape = shape

    /// A copy of the element strides.
    member _.Strides = Array.copy strides

    /// Bytes per element.
    member _.ElementSize = DType.size dtype

    /// True when the strides are the packed row-major strides of the shape.
    member _.IsContiguous =
        let packed = shape.RowMajorStrides
        // A dimension of size 1 (or 0) never moves, so its stride does not matter.
        Seq.forall2 (fun i (s: int64) -> shape[i] <= 1L || s = packed[i]) (seq { 0 .. strides.Length - 1 }) strides

    /// Elements the tensor spans from its first element: 1 + sum (dims[i] - 1) * strides[i]; 0 for an empty tensor.
    member _.ExtentElements =
        if shape.Elements = 0L then
            0L
        else
            let mutable e = 1L

            for i in 0 .. strides.Length - 1 do
                e <- e + (shape[i] - 1L) * strides[i]

            e

    /// Bytes of a contiguous tensor: elements times the element size.
    member this.ByteSize = shape.Elements * int64 this.ElementSize

    override _.ToString() =
        sprintf "%s%s" (DType.name dtype) (shape.ToString())

    override _.Equals(other: obj) =
        match other with
        | :? TensorDesc as d -> d.DType = dtype && d.Shape.Equals shape && d.Strides = strides
        | _ -> false

    override _.GetHashCode() = HashCode.Combine(dtype, shape)

    interface IEquatable<TensorDesc> with
        member this.Equals(other: TensorDesc) = this.Equals(box other)
