namespace Lrm.Tensor

#nowarn "9" // NativePtr over the memory-mapped view

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.IO
open System.IO.MemoryMappedFiles
open System.Text
open System.Text.Json
open Microsoft.FSharp.NativeInterop
open Lrm.Core

/// A safetensors file that breaks the format rules (HostTensorAndSafetensors contract).
type SafetensorsFormatException(message: string) =
    inherit FormatException("safetensors: " + message)

/// A file whose size or SHA-256 differs from the value it is pinned to, for example by a golden manifest
/// (ComparatorsAndGoldenReader contract).
type IntegrityException(path: string, expected: string, actual: string) =
    inherit Exception(sprintf "integrity check failed for %s: expected %s, found %s" path expected actual)

    /// The file.
    member _.Path = path

    /// The pinned value ("<bytes> bytes" or "sha256 <hex>").
    member _.Expected = expected

    /// The value found.
    member _.Actual = actual

/// One tensor of a safetensors file. Begin and End are byte offsets from the start of the data section (the byte after
/// the header), End exclusive.
type SafetensorsEntry =
    {
        Name: string
        Desc: TensorDesc
        Begin: int64
        End: int64
    }

    /// The tensor's byte count.
    member e.ByteLength = e.End - e.Begin

/// One tensor to write: its name, dtype, shape and little-endian bytes.
type SafetensorsTensor =
    {
        Name: string
        DType: DType
        Shape: Shape
        Data: ReadOnlyMemory<byte>
    }

[<RequireQualifiedAccess>]
module SafetensorsTensor =
    /// A tensor to write from a host tensor (the bytes are copied).
    let ofHost (name: string) (t: #IHostTensor) =
        {
            Name = name
            DType = t.Desc.DType
            Shape = t.Desc.Shape
            Data = ReadOnlyMemory(t.ToBytes())
        }

    /// A tensor to write from raw little-endian bytes.
    let ofBytes (name: string) (dtype: DType) (shape: Shape) (bytes: byte[]) =
        {
            Name = name
            DType = dtype
            Shape = shape
            Data = ReadOnlyMemory bytes
        }

/// The deterministic writer and the format rules (HostTensorAndSafetensors contract).
[<RequireQualifiedAccess>]
module Safetensors =
    /// The reserved header key for string metadata.
    [<Literal>]
    let MetadataKey = "__metadata__"

    /// The largest header the reader accepts, as in the reference implementation (100,000,000 bytes).
    [<Literal>]
    let MaxHeaderBytes = 100_000_000L

    let private strictUtf8 = UTF8Encoding(false, true)

    /// The UTF-8 bytes of a name; raises ArgumentException for a lone surrogate.
    let private utf8 (what: string) (s: string) =
        try
            strictUtf8.GetBytes s
        with :? EncoderFallbackException ->
            invalidArg what (sprintf "'%s' is not valid Unicode (lone surrogate)" s)

    /// Code point order: the order of the UTF-8 bytes, which is also Python's str order.
    let codePointCompare (a: string) (b: string) =
        ReadOnlySpan(utf8 "name" a).SequenceCompareTo(ReadOnlySpan(utf8 "name" b))

    /// A JSON string literal escaped exactly as Python's json.dumps with ensure_ascii=True: \" \\ \n \r \t \b \f,
    /// \u00xx for other control characters and DEL, and \uxxxx (lowercase hex) for every UTF-16 unit above U+007E,
    /// so astral characters become surrogate pairs.
    let jsonString (s: string) =
        let sb = StringBuilder(s.Length + 2)
        sb.Append('"') |> ignore

        for c in s do
            match c with
            | '"' -> sb.Append("\\\"") |> ignore
            | '\\' -> sb.Append("\\\\") |> ignore
            | '\n' -> sb.Append("\\n") |> ignore
            | '\r' -> sb.Append("\\r") |> ignore
            | '\t' -> sb.Append("\\t") |> ignore
            | '\b' -> sb.Append("\\b") |> ignore
            | '\f' -> sb.Append("\\f") |> ignore
            | c when c >= ' ' && c <= '~' -> sb.Append(c) |> ignore
            | c -> sb.Append("\\u").Append((int c).ToString("x4", Invariant.culture)) |> ignore

        sb.Append('"').ToString()

    /// The 8-byte header length plus the padded header JSON for (name, dtype, shape, byte count) entries:
    /// "__metadata__" first (keys in code point order, omitted when empty), then the tensors in code point order of
    /// their names, as {"dtype","shape","data_offsets"} with offsets in that order; compact ASCII JSON; spaces pad the
    /// header to a multiple of 8 bytes. Raises ArgumentException for duplicate or reserved names, lone surrogates and
    /// byte counts that do not match the shape.
    let header (metadata: (string * string) seq) (entries: (string * DType * Shape * int64) seq) : byte[] =
        let metadata = List.ofSeq metadata
        let entries = List.ofSeq entries

        let duplicates (names: string list) =
            names |> List.countBy id |> List.filter (fun (_, n) -> n > 1) |> List.map fst

        match duplicates (metadata |> List.map fst) with
        | [] -> ()
        | d -> invalidArg "metadata" (sprintf "duplicate metadata keys %A" d)

        match duplicates (entries |> List.map (fun (n, _, _, _) -> n)) with
        | [] -> ()
        | d -> invalidArg "entries" (sprintf "duplicate tensor names %A" d)

        for (k, v) in metadata do
            utf8 "metadata" k |> ignore
            utf8 "metadata" v |> ignore

        for (name, dtype, shape, bytes) in entries do
            utf8 "entries" name |> ignore

            if String.IsNullOrEmpty name then
                invalidArg "entries" "empty tensor name"

            if name = MetadataKey then
                invalidArg "entries" "the tensor name __metadata__ is reserved"

            let expected = shape.Elements * int64 (DType.size dtype)

            if bytes <> expected then
                invalidArg
                    "entries"
                    (sprintf "%s: %d bytes for %s%O (expected %d)" name bytes (DType.name dtype) shape expected)

        let sortByCodePoint (key: 'a -> string) (xs: 'a list) =
            xs |> List.sortWith (fun a b -> codePointCompare (key a) (key b))

        let parts = ResizeArray<string>()

        if not metadata.IsEmpty then
            let body =
                metadata
                |> sortByCodePoint fst
                |> List.map (fun (k, v) -> jsonString k + ":" + jsonString v)
                |> String.concat ","

            parts.Add(jsonString MetadataKey + ":{" + body + "}")

        let mutable offset = 0L

        for (name, dtype, shape, bytes) in sortByCodePoint (fun (n, _, _, _) -> n) entries do
            let dims = shape.Dims |> Array.map Invariant.i64 |> String.concat ","

            parts.Add(
                jsonString name
                + ":"
                + Invariant.format
                    $"""{{"dtype":"{DType.safetensorsName dtype}","shape":[{dims}],"data_offsets":[{offset},{offset + bytes}]}}"""
            )

            offset <- offset + bytes

        let json = "{" + String.Join(",", parts) + "}"
        let padded = json + String(' ', (8 - json.Length % 8) % 8)
        let result = Array.zeroCreate<byte>(8 + padded.Length)
        BinaryPrimitives.WriteUInt64LittleEndian(Span(result, 0, 8), uint64 padded.Length)
        Encoding.ASCII.GetBytes(padded, 0, padded.Length, result, 8) |> ignore
        result

    /// Writes a deterministic safetensors file to `stream`: the header (see `header`), then each tensor's bytes in
    /// the same name order, with no gaps and nothing after. The same inputs in any order give the same bytes.
    let writeTo (stream: Stream) (metadata: (string * string) seq) (tensors: SafetensorsTensor seq) =
        let tensors = List.ofSeq tensors

        let head =
            header metadata (tensors |> List.map (fun t -> t.Name, t.DType, t.Shape, int64 t.Data.Length))

        stream.Write(head, 0, head.Length)

        for t in tensors |> List.sortWith (fun a b -> codePointCompare a.Name b.Name) do
            stream.Write(t.Data.Span)

    /// The bytes of a deterministic safetensors file.
    let toBytes (metadata: (string * string) seq) (tensors: SafetensorsTensor seq) =
        use ms = new MemoryStream()
        writeTo ms metadata tensors
        ms.ToArray()

    /// Writes a deterministic safetensors file to `path`, through a temporary file in the same folder that replaces
    /// the target only when complete.
    let writeFile (path: string) (metadata: (string * string) seq) (tensors: SafetensorsTensor seq) =
        let full = Path.GetFullPath path
        let temp = full + ".tmp-" + Guid.NewGuid().ToString("N")

        try
            do
                use fs = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None)
                writeTo fs metadata tensors

            File.Move(temp, full, true)
        finally
            if File.Exists temp then
                File.Delete temp

/// The validated header of a safetensors file.
type internal SafetensorsHeader =
    {
        HeaderLength: int64
        Metadata: (string * string)[]
        Entries: SafetensorsEntry[]
    }

/// A read-only memory mapping of a whole file, with its pointer acquired.
[<Sealed>]
type internal FileMapping(stream: FileStream) =
    let length = stream.Length

    let mmf =
        MemoryMappedFile.CreateFromFile(stream, null, 0L, MemoryMappedFileAccess.Read, HandleInheritability.None, false)

    let view =
        try
            mmf.CreateViewAccessor(0L, 0L, MemoryMappedFileAccess.Read)
        with _ ->
            mmf.Dispose()
            reraise ()

    let basePtr =
        let mutable p = NativePtr.nullPtr<byte>
        view.SafeMemoryMappedViewHandle.AcquirePointer(&p)
        NativePtr.toNativeInt p + nativeint view.PointerOffset

    let mutable disposed = 0

    member _.Length = length

    member _.IsDisposed = disposed <> 0

    /// `count` bytes at `offset` of the file.
    member _.Bytes(offset: int64, count: int) =
        if offset < 0L || count < 0 || offset + int64 count > length then
            raise (ArgumentOutOfRangeException("offset"))

        ReadOnlySpan<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte>(basePtr + nativeint offset)), count)

    /// A pointer to byte `offset` of the file.
    member _.Pointer(offset: int64) =
        NativePtr.ofNativeInt<byte>(basePtr + nativeint offset)

    /// The raw SHA-256 of the mapped bytes.
    member this.Sha256() : ContentId =
        use h = new Hash.Incremental()
        let chunk = 1L <<< 30
        let mutable at = 0L

        while at < length do
            let n = int (min chunk (length - at))
            h.Append(this.Bytes(at, n))
            at <- at + int64 n

        h.Finish()

    interface IDisposable with
        member _.Dispose() =
            if Threading.Interlocked.Exchange(&disposed, 1) = 0 then
                view.SafeMemoryMappedViewHandle.ReleasePointer()
                view.Dispose()
                mmf.Dispose()
                stream.Dispose()

[<RequireQualifiedAccess>]
module internal SafetensorsParse =
    let private fail (msg: string) = raise (SafetensorsFormatException msg)

    let private nonNegative (what: string) (e: JsonElement) =
        match e.ValueKind with
        | JsonValueKind.Number ->
            match e.TryGetInt64() with
            | true, v when v >= 0L -> v
            | _ -> fail (sprintf "%s: %s is not a non-negative integer" what (e.GetRawText()))
        | _ -> fail (sprintf "%s: %s is not a non-negative integer" what (e.GetRawText()))

    let private entry (dataLength: int64) (name: string) (value: JsonElement) =
        if name.Length = 0 then
            fail "empty tensor name"

        if value.ValueKind <> JsonValueKind.Object then
            fail (sprintf "%s: the entry is not an object" name)

        let fields = HashSet<string>(StringComparer.Ordinal)
        let mutable dtype = None
        let mutable dims = None
        let mutable offsets = None

        for f in value.EnumerateObject() do
            if not (fields.Add f.Name) then
                fail (sprintf "%s: duplicate field '%s'" name f.Name)

            match f.Name with
            | "dtype" ->
                if f.Value.ValueKind <> JsonValueKind.String then
                    fail (sprintf "%s: dtype is not a string" name)

                match DType.tryOfSafetensorsName (f.Value.GetString()) with
                | Some t -> dtype <- Some t
                | None -> fail (sprintf "%s: unsupported dtype '%s'" name (f.Value.GetString()))
            | "shape" ->
                if f.Value.ValueKind <> JsonValueKind.Array then
                    fail (sprintf "%s: shape is not an array" name)

                dims <-
                    Some(
                        f.Value.EnumerateArray()
                        |> Seq.map (nonNegative (name + ": shape"))
                        |> Array.ofSeq
                    )
            | "data_offsets" ->
                if f.Value.ValueKind <> JsonValueKind.Array || f.Value.GetArrayLength() <> 2 then
                    fail (sprintf "%s: data_offsets is not a pair" name)

                let b = nonNegative (name + ": data_offsets") f.Value[0]
                let e = nonNegative (name + ": data_offsets") f.Value[1]
                offsets <- Some(b, e)
            | other -> fail (sprintf "%s: unknown field '%s'" name other)

        match dtype, dims, offsets with
        | Some dtype, Some dims, Some(b, e) ->
            let shape =
                try
                    Shape.Of(dims)
                with :? ArgumentException as ex ->
                    fail (sprintf "%s: %s" name ex.Message)

            let bytes =
                try
                    Checked.(*) shape.Elements (int64 (DType.size dtype))
                with :? OverflowException ->
                    fail (sprintf "%s: the byte count overflows" name)

            if e < b then
                fail (sprintf "%s: data_offsets [%d, %d] are reversed" name b e)

            if e - b <> bytes then
                fail (sprintf "%s: %d bytes for %s%O (expected %d)" name (e - b) (DType.name dtype) shape bytes)

            if e > dataLength then
                fail (sprintf "%s: data_offsets end %d is beyond the data section (%d bytes)" name e dataLength)

            {
                Name = name
                Desc = TensorDesc.Contiguous(dtype, shape)
                Begin = b
                End = e
            }
        | _ -> fail (sprintf "%s: the entry needs dtype, shape and data_offsets" name)

    /// Validates the header of a file of `length` bytes whose bytes `bytesAt` returns.
    let header (length: int64) (bytesAt: int64 -> int -> byte[]) : SafetensorsHeader =
        if length < 8L then
            fail (sprintf "%d bytes is shorter than the 8-byte header length" length)

        let n = BinaryPrimitives.ReadUInt64LittleEndian(ReadOnlySpan(bytesAt 0L 8))

        if n > uint64 Safetensors.MaxHeaderBytes then
            fail (sprintf "header length %d exceeds %d" n Safetensors.MaxHeaderBytes)

        if int64 n > length - 8L then
            fail (sprintf "header length %d exceeds the file (%d bytes)" n length)

        let headerLength = int64 n
        let dataLength = length - 8L - headerLength
        let headerBytes = bytesAt 8L (int headerLength)

        if headerLength = 0L || headerBytes[0] <> byte '{' then
            fail "the header is not a JSON object"

        let doc =
            try
                JsonDocument.Parse(ReadOnlyMemory headerBytes, JsonDocumentOptions(MaxDepth = 8))
            with :? JsonException as e ->
                fail ("the header is not valid JSON: " + e.Message)

        use doc = doc

        if doc.RootElement.ValueKind <> JsonValueKind.Object then
            fail "the header is not a JSON object"

        let seen = HashSet<string>(StringComparer.Ordinal)
        let meta = List<string * string>()
        let list = List<SafetensorsEntry>()

        for p in doc.RootElement.EnumerateObject() do
            if not (seen.Add p.Name) then
                fail (sprintf "duplicate key '%s'" p.Name)

            if p.Name = Safetensors.MetadataKey then
                if p.Value.ValueKind <> JsonValueKind.Object then
                    fail "__metadata__ is not an object"

                let keys = HashSet<string>(StringComparer.Ordinal)

                for m in p.Value.EnumerateObject() do
                    if not (keys.Add m.Name) then
                        fail (sprintf "duplicate metadata key '%s'" m.Name)

                    if m.Value.ValueKind <> JsonValueKind.String then
                        fail (sprintf "metadata '%s' is not a string" m.Name)

                    meta.Add((m.Name, m.Value.GetString()))
            else
                list.Add(entry dataLength p.Name p.Value)

        // The tensors must tile the data section exactly: no holes, no overlaps, nothing after the last one.
        let mutable position = 0L

        for e in list |> Seq.sortBy (fun e -> struct (e.Begin, e.End)) do
            if e.Begin < position then
                fail (
                    sprintf
                        "%s overlaps the previous tensor (begins at %d, previous ends at %d)"
                        e.Name
                        e.Begin
                        position
                )

            if e.Begin > position then
                fail (sprintf "hole before %s (begins at %d, previous ends at %d)" e.Name e.Begin position)

            position <- e.End

        if position <> dataLength then
            fail (sprintf "%d bytes after the last tensor" (dataLength - position))

        {
            HeaderLength = headerLength
            Metadata =
                meta
                |> Seq.sortWith (fun (a, _) (b, _) -> Safetensors.codePointCompare a b)
                |> Array.ofSeq
            Entries =
                list
                |> Seq.sortWith (fun a b -> Safetensors.codePointCompare a.Name b.Name)
                |> Array.ofSeq
        }

/// A memory-mapped, validated safetensors file (HostTensorAndSafetensors contract). Open checks the header length,
/// the JSON, every dtype, shape and offset range, and that the tensors tile the data section exactly, before anything
/// is returned. While it is open nobody can write the file (it is opened with FileShare.Read), so bytes a caller has
/// verified stay the bytes it reads. Spans are valid until Dispose. Reads are thread-safe.
[<Sealed>]
type SafetensorsFile private (path: string, map: FileMapping, header: SafetensorsHeader) =
    let entries = header.Entries
    let metadata = header.Metadata
    let dataOffset = 8L + header.HeaderLength

    let index =
        let d = Dictionary<string, SafetensorsEntry>(StringComparer.Ordinal)

        for e in entries do
            d[e.Name] <- e

        d

    let checkOpen () =
        if map.IsDisposed then
            raise (ObjectDisposedException("SafetensorsFile", path))

    static member private OpenCore(path: string, expected: (int64 * ContentId) option) : SafetensorsFile =
        let full = Path.GetFullPath path

        let stream =
            new FileStream(full, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.RandomAccess)

        let length = stream.Length

        let early (e: exn) =
            stream.Dispose()
            raise e

        match expected with
        | Some(bytes, _) when bytes <> length ->
            early (IntegrityException(full, Invariant.format $"{bytes} bytes", Invariant.format $"{length} bytes"))
        | _ -> ()

        if length < 8L then
            early (
                SafetensorsFormatException(sprintf "%s: %d bytes is shorter than the 8-byte header length" full length)
            )

        let map = new FileMapping(stream)

        try
            // Verify the pinned hash over the mapped bytes before parsing anything: these are the bytes later reads
            // return, and nobody can write the file while it is open.
            match expected with
            | Some(_, sha) ->
                let actual = map.Sha256()

                if actual <> sha then
                    raise (IntegrityException(full, "sha256 " + sha.Hex, "sha256 " + actual.Hex))
            | None -> ()

            let header =
                try
                    SafetensorsParse.header map.Length (fun offset count -> map.Bytes(offset, count).ToArray())
                with :? SafetensorsFormatException as e ->
                    raise (SafetensorsFormatException(full + ": " + e.Message.Substring("safetensors: ".Length)))

            new SafetensorsFile(full, map, header)
        with _ ->
            (map :> IDisposable).Dispose()
            reraise ()

    /// Opens and validates a file. Raises SafetensorsFormatException for a malformed file, and the usual IO exceptions
    /// when it cannot be read.
    static member Open(path: string) : SafetensorsFile = SafetensorsFile.OpenCore(path, None)

    /// Opens a file pinned to a size and raw SHA-256: both are checked over the mapped bytes before the header is
    /// parsed, and IntegrityException is raised on a mismatch. Then the file is validated as by Open.
    static member OpenVerified(path: string, bytes: int64, sha256: ContentId) : SafetensorsFile =
        SafetensorsFile.OpenCore(path, Some(bytes, sha256))

    /// Validates safetensors bytes held in memory and returns the header length, metadata and entries.
    static member Validate(bytes: byte[]) =
        let h =
            SafetensorsParse.header (int64 bytes.Length) (fun offset count -> Array.sub bytes (int offset) count)

        h.HeaderLength, h.Metadata, h.Entries

    /// The full path of the file.
    member _.Path = path

    /// The file size in bytes.
    member _.Length = map.Length

    /// N, the header length from the first 8 bytes.
    member _.HeaderLength = header.HeaderLength

    /// The byte offset of the data section: 8 + N.
    member _.DataOffset = dataOffset

    /// The string metadata, in code point order of the keys.
    member _.Metadata: (string * string)[] = Array.copy metadata

    /// The metadata value for `key`, if present.
    member _.TryMetadata(key: string) =
        metadata |> Array.tryFind (fun (k, _) -> k = key) |> Option.map snd

    /// Every tensor, in code point order of the names.
    member _.Entries: SafetensorsEntry[] = Array.copy entries

    /// The names, in code point order.
    member _.Names = entries |> Array.map _.Name

    /// True when the data section starts at a multiple of 8 and every tensor starts at a multiple of its element size
    /// relative to it (the GoldenSet alignment rule).
    member _.IsAligned =
        dataOffset % 8L = 0L
        && entries |> Array.forall (fun e -> e.Begin % int64 e.Desc.ElementSize = 0L)

    member _.Contains(name: string) = index.ContainsKey name

    member _.TryEntry(name: string) =
        match index.TryGetValue name with
        | true, e -> Some e
        | _ -> None

    /// The entry for `name`; raises KeyNotFoundException.
    member this.Entry(name: string) =
        match this.TryEntry name with
        | Some e -> e
        | None -> raise (KeyNotFoundException(sprintf "%s: no tensor '%s'" path name))

    /// The tensor's bytes, straight from the mapped file. Valid until Dispose. Raises for a tensor over 2 GiB (use
    /// OpenRead).
    member this.Span(name: string) : ReadOnlySpan<byte> =
        checkOpen ()
        let e = this.Entry name

        if e.ByteLength > int64 Int32.MaxValue then
            invalidOp (sprintf "%s is %d bytes; use OpenRead" name e.ByteLength)

        map.Bytes(dataOffset + e.Begin, int e.ByteLength)

    /// A read-only stream over the tensor's bytes. Valid until Dispose.
    member this.OpenRead(name: string) : Stream =
        checkOpen ()
        let e = this.Entry name
        new UnmanagedMemoryStream(map.Pointer(dataOffset + e.Begin), e.ByteLength, e.ByteLength, FileAccess.Read)

    /// The raw SHA-256 of the whole file, for SHA256SUMS and golden manifests.
    member _.Sha256() : ContentId =
        checkOpen ()
        map.Sha256()

    /// A copy of the tensor as a HostTensor<'T>; 'T must carry the tensor's dtype.
    member this.Read<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (name: string)
        : HostTensor<'T> =
        let e = this.Entry name

        if ElementType.ofDType e.Desc.DType <> typeof<'T> then
            invalidArg "name" (sprintf "%s is %s, not readable as %s" name (DType.name e.Desc.DType) typeof<'T>.Name)

        HostTensor.ofBytes<'T> e.Desc.DType e.Desc.Shape (this.Span name)

    member private this.Expect(name: string, dtype: DType) =
        let e = this.Entry name

        if e.Desc.DType <> dtype then
            invalidArg "name" (sprintf "%s is %s, not %s" name (DType.name e.Desc.DType) (DType.name dtype))

    /// A F32 tensor.
    member this.ReadF32(name: string) =
        this.Expect(name, DType.F32)
        this.Read<float32> name

    /// A F64 tensor.
    member this.ReadF64(name: string) =
        this.Expect(name, DType.F64)
        this.Read<float> name

    /// A BF16 tensor (raw bits).
    member this.ReadBf16(name: string) =
        this.Expect(name, DType.BF16)
        this.Read<uint16> name

    /// An I64 tensor.
    member this.ReadI64(name: string) =
        this.Expect(name, DType.I64)
        this.Read<int64> name

    /// An I32 tensor.
    member this.ReadI32(name: string) =
        this.Expect(name, DType.I32)
        this.Read<int32> name

    /// A U32 tensor.
    member this.ReadU32(name: string) =
        this.Expect(name, DType.U32)
        this.Read<uint32> name

    /// A U8 tensor.
    member this.ReadU8(name: string) =
        this.Expect(name, DType.U8)
        this.Read<byte> name

    /// Any tensor, dtype-erased (a copy).
    member this.Tensor(name: string) : IHostTensor =
        match (this.Entry name).Desc.DType with
        | DType.F32 -> this.Read<float32> name :> IHostTensor
        | DType.F64 -> this.Read<float> name :> IHostTensor
        | DType.BF16
        | DType.F16 -> this.Read<uint16> name :> IHostTensor
        | DType.I32 -> this.Read<int32> name :> IHostTensor
        | DType.U32 -> this.Read<uint32> name :> IHostTensor
        | DType.U8 -> this.Read<byte> name :> IHostTensor
        | DType.I64 -> this.Read<int64> name :> IHostTensor

    /// The header the deterministic writer builds from this file's metadata and entries. It equals HeaderBytes()
    /// exactly when the file follows the writer's convention.
    member _.RebuildHeader() =
        Safetensors.header metadata (entries |> Array.map (fun e -> e.Name, e.Desc.DType, e.Desc.Shape, e.ByteLength))

    /// The file's own first 8 + N bytes.
    member _.HeaderBytes() =
        checkOpen ()
        map.Bytes(0L, int dataOffset).ToArray()

    /// Unmaps and closes the file. Spans and streams from it become invalid.
    member _.Dispose() = (map :> IDisposable).Dispose()

    interface IDisposable with
        member this.Dispose() = this.Dispose()
