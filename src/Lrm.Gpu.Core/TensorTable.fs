namespace Lrm.Gpu.Core

open System
open System.Buffers.Binary
open System.Collections.Generic
open Lrm.Core

/// A 32-bit tensor token: the index of a row in the tensor table (HeapLayoutAndTensorTable contract). Kernels receive
/// tokens in their argument block. Token 0 is the null token; real tokens are 1 to 16,383.
[<Struct>]
type TensorToken =
    | TensorToken of uint32

    /// The row index.
    member x.Value =
        let (TensorToken v) = x
        v

    /// True for token 0.
    member x.IsNull = x.Value = 0u

    /// Token 0, whose row is all zeros.
    static member Null = TensorToken 0u

    override x.ToString() = sprintf "t%d" x.Value

/// Tensor-row flags (word 13). Bits 3 to 31 are reserved and must be 0.
[<Flags>]
type TensorFlags =
    | None = 0u
    /// The strides are the packed row-major strides of the dims. Every v1 kernel requires it.
    | Contiguous = 1u
    /// Kernels never write this tensor (weights, RoPE tables). A write token on it is a contract violation.
    | ReadOnly = 2u
    /// The row aliases bytes that another row owns.
    | View = 4u

[<RequireQualifiedAccess>]
module TensorFlags =
    /// Every defined bit.
    let known = TensorFlags.Contiguous ||| TensorFlags.ReadOnly ||| TensorFlags.View

    /// True when `flags` has every bit of `bit`.
    let has (bit: TensorFlags) (flags: TensorFlags) = flags &&& bit = bit

/// One 64-byte row of the tensor table: 16 little-endian u32 words.
///
/// | Word | Field |
/// | 0 | slot (1..63) |
/// | 1 | byte offset in the page (a multiple of 4 and of the element size) |
/// | 2 | dtype code (LrmCore DType) |
/// | 3 | rank (1..4) |
/// | 4..7 | dims[0..3], outermost first; unused dims are 1 |
/// | 8..11 | strides[0..3] in elements; unused strides are 0 |
/// | 12 | element count, the product of the dims |
/// | 13 | flags (TensorFlags) |
/// | 14 | debug id (TensorRow.debugIdOf name); never read for computation |
/// | 15 | reserved, 0 |
[<Sealed>]
type TensorRow private (words: uint32[]) =
    /// u32 words per row.
    static member WordCount = 16

    /// The all-zero row: the null token's row, and every unused row. Its dtype code 0 never decodes as valid.
    static member Zero = TensorRow(Array.zeroCreate 16)

    /// A row from exactly 16 words (copied).
    static member OfWords(w: uint32[]) =
        if isNull w || w.Length <> 16 then
            Fail.contract "a tensor row is 16 words"

        TensorRow(Array.copy w)

    /// A row from exactly 64 little-endian bytes.
    static member OfBytes(bytes: ReadOnlySpan<byte>) =
        if bytes.Length <> Heap.RowBytes then
            Fail.contract "a tensor row is 64 bytes, got %d" bytes.Length

        let w = Array.zeroCreate<uint32> 16

        for i in 0..15 do
            w[i] <- BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(4 * i, 4))

        TensorRow w

    /// Word `i` (0..15).
    member _.Word(i: int) = words[i]

    member _.Slot = int words[0]
    member _.ByteOffset = words[1]
    member _.DTypeCode = words[2]

    /// The dtype, or None when the code is not a DType (the zero row).
    member _.DType = DType.tryOfCode (int words[2])

    member _.Rank = int words[3]

    /// dims[i], 0..3.
    member _.Dim(i: int) = words[4 + i]

    /// strides[i] in elements, 0..3.
    member _.Stride(i: int) = words[8 + i]

    /// The first Rank dims.
    member x.Dims = [ for i in 0 .. (min 4 (max 0 x.Rank)) - 1 -> words[4 + i] ]

    member _.ElemCount = words[12]
    member _.Flags: TensorFlags = LanguagePrimitives.EnumOfValue words[13]
    member _.DebugId = words[14]
    member _.Reserved = words[15]

    /// True for the all-zero row.
    member _.IsZero = words |> Array.forall ((=) 0u)

    /// A copy of the 16 words.
    member _.ToWords() = Array.copy words

    /// Writes the 64 bytes to the start of `destination`.
    member _.WriteTo(destination: Span<byte>) =
        if destination.Length < Heap.RowBytes then
            Fail.contract "a tensor row needs 64 bytes"

        for i in 0..15 do
            BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(4 * i, 4), words[i])

    /// The 64 bytes.
    member x.ToBytes() =
        let b = Array.zeroCreate<byte> Heap.RowBytes
        x.WriteTo(Span b)
        b

    override _.Equals(other: obj) =
        match other with
        | :? TensorRow as o -> o.ToWords() = words
        | _ -> false

    override _.GetHashCode() = hash words

    override x.ToString() =
        match x.DType with
        | None when x.IsZero -> "TensorRow(null)"
        | dt ->
            sprintf
                "TensorRow(slot %d +%d %s %A flags %A)"
                x.Slot
                x.ByteOffset
                (dt
                 |> Option.map DType.name
                 |> Option.defaultValue (sprintf "dtype?%d" x.DTypeCode))
                x.Dims
                x.Flags

[<RequireQualifiedAccess>]
module TensorRow =
    /// The packed row-major strides of `dims` (outermost first), in elements.
    let packedStrides (dims: int64 list) =
        List.foldBack (fun d (acc: int64 list) -> (List.head acc * d) :: acc) dims [ 1L ]
        |> List.tail

    /// The debug id of a tensor name: the first 4 bytes of SHA-256(UTF-8 name), little-endian. 0 for "".
    let debugIdOf (name: string) =
        if String.IsNullOrEmpty name then
            0u
        else
            let h = (Hash.sha256Utf8 name).ToBytes()
            BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(h, 0, 4))

    let private build
        (slot: int)
        (byteOffset: int64)
        (dtype: DType)
        (dims: int64 list)
        (strides: int64 list)
        (flags: TensorFlags)
        (debugId: uint32)
        =
        let rank = dims.Length

        if rank < 1 || rank > 4 then
            Fail.contract "tensor rank %d is outside 1..4" rank

        if strides.Length <> rank then
            Fail.contract "%d strides for rank %d" strides.Length rank

        if dims |> List.exists (fun d -> d < 1L || d > int64 UInt32.MaxValue) then
            Fail.contract "tensor dims %A must each be 1..2^32-1" dims

        if strides |> List.exists (fun s -> s < 0L || s > int64 UInt32.MaxValue) then
            Fail.contract "tensor strides %A must each be 0..2^32-1" strides

        if byteOffset < 0L || byteOffset > int64 UInt32.MaxValue then
            Fail.contract "byte offset %d does not fit in u32" byteOffset

        let count = dims |> List.fold (*) 1L

        if count > int64 UInt32.MaxValue then
            Fail.contract "tensor of %d elements does not fit a u32 count" count

        let contiguous =
            if strides = packedStrides dims then
                TensorFlags.Contiguous
            else
                TensorFlags.None

        let w = Array.zeroCreate<uint32> 16
        w[0] <- uint32 slot
        w[1] <- uint32 byteOffset
        w[2] <- uint32 (DType.code dtype)
        w[3] <- uint32 rank

        for i in 0..3 do
            w[4 + i] <- if i < rank then uint32 dims[i] else 1u
            w[8 + i] <- if i < rank then uint32 strides[i] else 0u

        w[12] <- uint32 count
        w[13] <- LanguagePrimitives.EnumToValue((flags &&& ~~~TensorFlags.Contiguous) ||| contiguous)
        w[14] <- debugId
        TensorRow.OfWords w

    /// A contiguous row: packed row-major strides and the Contiguous flag, plus `flags`.
    let create
        (slot: int)
        (byteOffset: int64)
        (dtype: DType)
        (dims: int64 list)
        (flags: TensorFlags)
        (debugId: uint32)
        =
        build slot byteOffset dtype dims (packedStrides dims) flags debugId

    /// A row with explicit strides (in elements). Contiguous is set exactly when they are the packed strides.
    let strided
        (slot: int)
        (byteOffset: int64)
        (dtype: DType)
        (dims: int64 list)
        (strides: int64 list)
        (flags: TensorFlags)
        (debugId: uint32)
        =
        build slot byteOffset dtype dims strides flags debugId

    /// Elements spanned from the row's start: 1 + sum((dims[i] - 1) * strides[i]).
    let spanElements (row: TensorRow) =
        1L
        + ([ 0 .. row.Rank - 1 ]
           |> List.sumBy (fun i -> (int64 (row.Dim i) - 1L) * int64 (row.Stride i)))

    /// Bytes spanned from ByteOffset, rounded up to 4 because kernels load whole u32 words. Requires a valid dtype.
    let extentBytes (row: TensorRow) =
        match row.DType with
        | Some t -> Heap.alignUp 4L (spanElements row * int64 (DType.size t))
        | None -> 0L

    /// Checks a row against the contract and, for its slot, against `layout`:
    /// slot 1..63 with a page; a valid dtype; rank 1..4 with unused dims 1 and unused strides 0; the count equal to the
    /// product of the dims; the byte offset a multiple of 4 and of the element size; only known flags, with Contiguous
    /// exactly when the strides are packed; the reserved word 0; and the extent inside the page.
    let validate (layout: HeapLayout) (row: TensorRow) : Result<unit, string> =
        let problems = ResizeArray<string>()
        let add (s: string) = problems.Add s

        if row.Slot < 1 || row.Slot >= Heap.SlotCount then
            add (sprintf "slot %d is outside 1..63" row.Slot)
        elif (layout.TryPage row.Slot).IsNone then
            add (sprintf "slot %d holds no page in the layout" row.Slot)

        match row.DType with
        | None -> add (sprintf "dtype code %d is not a DType" row.DTypeCode)
        | Some t ->
            if int64 row.ByteOffset % int64 (max 4 (DType.size t)) <> 0L then
                add (sprintf "byte offset %d is not a multiple of %d" row.ByteOffset (max 4 (DType.size t)))

        if row.Rank < 1 || row.Rank > 4 then
            add (sprintf "rank %d is outside 1..4" row.Rank)
        else
            for i in 0..3 do
                if i < row.Rank && row.Dim i < 1u then
                    add (sprintf "dims[%d] is 0" i)

                if i >= row.Rank && (row.Dim i <> 1u || row.Stride i <> 0u) then
                    add (sprintf "unused dims[%d] must be 1 with stride 0" i)

            let product =
                [ 0 .. row.Rank - 1 ] |> List.fold (fun acc i -> acc * int64 (row.Dim i)) 1L

            if product <> int64 row.ElemCount then
                add (sprintf "element count %d is not the product of the dims (%d)" row.ElemCount product)

            let packed = packedStrides [ for i in 0 .. row.Rank - 1 -> int64 (row.Dim i) ]

            let actual = [ for i in 0 .. row.Rank - 1 -> int64 (row.Stride i) ]

            if TensorFlags.has TensorFlags.Contiguous row.Flags <> (packed = actual) then
                add "the Contiguous flag must be set exactly when the strides are packed row-major"

        if
            LanguagePrimitives.EnumToValue row.Flags
            &&& ~~~(LanguagePrimitives.EnumToValue TensorFlags.known)
            <> 0u
        then
            add (sprintf "flags 0x%x use reserved bits" (LanguagePrimitives.EnumToValue row.Flags))

        if row.Reserved <> 0u then
            add "word 15 is reserved and must be 0"

        if problems.Count = 0 then
            let page = layout.Page row.Slot
            let endByte = int64 row.ByteOffset + extentBytes row

            if endByte > page.Bytes then
                add (sprintf "extent ends at byte %d, past the %d-byte page in slot %d" endByte page.Bytes row.Slot)

        if problems.Count = 0 then
            Ok()
        else
            Error(String.concat "; " problems)

/// The tensor table after Seal: 16,384 rows (1 MiB), immutable, written to slot 0 once.
[<Sealed>]
type SealedTable internal (layout: HeapLayout, rows: TensorRow[], names: string[]) =
    let image =
        let b = Array.zeroCreate<byte>(int Heap.TableBytes)

        rows
        |> Array.iteri (fun i r -> r.WriteTo(Span(b, i * Heap.RowBytes, Heap.RowBytes)))

        b

    let id = Hash.sha256 image
    let byName = Dictionary<string, TensorToken>(StringComparer.Ordinal)

    do
        for i in 1 .. names.Length - 1 do
            byName[names[i]] <- TensorToken(uint32 i)

    /// The layout the rows were checked against.
    member _.Layout = layout

    /// Real rows (tokens 1..Count).
    member _.Count = rows.Length - 1

    /// The row of `token`: the zero row for the null token and for unused tokens. Raises GpuContractException for a
    /// token outside 0..16,383.
    member _.Row(token: TensorToken) =
        let t = token.Value

        if t >= uint32 Heap.TableRows then
            Fail.contract "token %d is outside 0..%d" t (Heap.TableRows - 1)
        elif t < uint32 rows.Length then
            rows[int t]
        else
            TensorRow.Zero

    /// True when `token` names a real row (not null, not unused).
    member _.IsAssigned(token: TensorToken) =
        token.Value >= 1u && token.Value < uint32 rows.Length

    /// The name given to `token` when it was added ("" for the null token).
    member _.Name(token: TensorToken) =
        if token.Value < uint32 names.Length then
            names[int token.Value]
        else
            ""

    /// The token added under `name`.
    member _.TryFind(name: string) =
        match byName.TryGetValue name with
        | true, t -> Some t
        | _ -> None

    /// The 1 MiB table image, row t at byte 64 t.
    member _.Image = ReadOnlyMemory image

    /// SHA-256 of the image.
    member _.Id = id

/// Pass two of a static plan: adds one row per tensor, checked against the layout, then seals the table once.
[<Sealed>]
type TensorTable(layout: HeapLayout) =
    let rows = ResizeArray<TensorRow>([ TensorRow.Zero ])
    let names = ResizeArray<string>([ "" ])
    let byName = HashSet<string>(StringComparer.Ordinal)
    let mutable sealedTable: SealedTable option = None

    let checkOpen () =
        if sealedTable.IsSome then
            Fail.contract "the tensor table is sealed"

    member _.Layout = layout

    /// Real rows so far.
    member _.Count = rows.Count - 1

    /// The row of a token added so far (the zero row for the null token).
    member _.Row(token: TensorToken) =
        if token.Value < uint32 rows.Count then
            rows[int token.Value]
        else
            Fail.contract "token %d has not been added" token.Value

    /// Adds a row under a unique, non-empty name and returns its token (1, 2, ...). The row must validate against the
    /// layout. Raises GpuContractException when the table is sealed or full (16,383 real rows).
    member _.Add(name: string, row: TensorRow) : TensorToken =
        checkOpen ()

        if String.IsNullOrEmpty name then
            Fail.contract "a tensor needs a name"

        if byName.Contains name then
            Fail.contract "tensor name '%s' is already in the table" name

        if rows.Count >= Heap.TableRows then
            Fail.contract "the tensor table is full (%d rows)" (Heap.TableRows - 1)

        match TensorRow.validate layout row with
        | Error e -> Fail.contract "tensor '%s': %s" name e
        | Ok() -> ()

        rows.Add row
        names.Add name
        byName.Add name |> ignore
        TensorToken(uint32 (rows.Count - 1))

    /// Adds a contiguous tensor at a planner placement. The placement must hold the tensor's bytes.
    member x.Add(name: string, placement: Placement, dtype: DType, dims: int64 list, flags: TensorFlags) =
        let slot = layout.SlotOf(placement.Arena, placement.Page)

        let row =
            TensorRow.create slot placement.Offset dtype dims flags (TensorRow.debugIdOf name)

        if TensorRow.extentBytes row > Heap.alignUp 4L placement.Bytes then
            Fail.contract
                "tensor '%s' needs %d bytes but its placement has %d"
                name
                (TensorRow.extentBytes row)
                placement.Bytes

        x.Add(name, row)

    /// Adds a contiguous view of `source`'s bytes: `dims` starting `elementOffset` elements into it, with the View
    /// flag and the source's ReadOnly flag. The view must lie inside the source's extent.
    member x.AddView(name: string, source: TensorToken, dims: int64 list, elementOffset: int64) =
        let src = x.Row source

        if src.IsZero then
            Fail.contract "cannot view the null token"

        let dtype = src.DType.Value
        let offset = int64 src.ByteOffset + elementOffset * int64 (DType.size dtype)

        let row =
            TensorRow.create
                src.Slot
                offset
                dtype
                dims
                (TensorFlags.View ||| (src.Flags &&& TensorFlags.ReadOnly))
                (TensorRow.debugIdOf name)

        if
            elementOffset < 0L
            || offset + TensorRow.extentBytes row > int64 src.ByteOffset + TensorRow.extentBytes src
        then
            Fail.contract "view '%s' does not lie inside its source %O" name source

        x.Add(name, row)

    /// Seals the table. The second call raises GpuContractException.
    member _.Seal() : SealedTable =
        checkOpen ()
        let s = SealedTable(layout, rows.ToArray(), names.ToArray())
        sealedTable <- Some s
        s
