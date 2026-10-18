namespace Lrm.Core

open System
open System.Buffers
open System.Buffers.Binary
open System.Security.Cryptography
open System.Text

/// Strict UTF-8 shared by the canonical writer and reader: no BOM, and invalid text raises instead of being replaced.
module internal CanonicalText =
    let utf8 = UTF8Encoding(false, true)

    /// Strict UTF-8 bytes of `text`; raises ArgumentException (not a bare EncoderFallbackException) for a lone
    /// surrogate.
    let encode (what: string) (text: string) =
        try
            utf8.GetBytes text
        with :? EncoderFallbackException as e ->
            raise (
                ArgumentException(sprintf "%s is not valid UTF-16 (lone surrogate at index %d)" what e.Index, what, e)
            )

    /// Orders UTF-8 byte strings lexicographically, which is Unicode code point order.
    let compareBytes (a: byte[]) (b: byte[]) =
        ReadOnlySpan(a).SequenceCompareTo(ReadOnlySpan b)

/// Writer for the canonical encoding (LrmCore contract).
///
/// The encoding is positional: a value is its fields in declared order, with no names, tags or padding. Integers are
/// little-endian; floats are their raw IEEE 754 bits (negative zero and every NaN payload are kept); strings are a
/// u32 byte length plus strict UTF-8; byte strings and sequences carry a u32 length or count; an option is 0x00, or
/// 0x01 followed by the value; a map is a u32 count plus its entries sorted by the UTF-8 bytes of their keys, with
/// no duplicates. There is deliberately no encoder for times, dates or durations: timestamps never enter a hash.
[<Sealed>]
type CanonicalWriter() =
    let buffer = ArrayBufferWriter<byte>(256)

    let checkLength (n: int) (what: string) =
        // Lengths and counts are u32; an int is never above Int32.MaxValue, so only negatives are impossible here.
        if n < 0 then
            invalidArg what "negative length"

    /// Bytes written so far.
    member _.Length = buffer.WrittenCount

    /// The bytes written so far. Valid until the next write.
    member _.Written: ReadOnlySpan<byte> = buffer.WrittenSpan

    /// A copy of the bytes written so far.
    member _.ToArray() = buffer.WrittenSpan.ToArray()

    /// One byte.
    member _.U8(value: byte) =
        let span = buffer.GetSpan 1
        span[0] <- value
        buffer.Advance 1

    /// 0x00 or 0x01.
    member w.Bool(value: bool) = w.U8(if value then 1uy else 0uy)

    /// u16, little-endian.
    member _.U16(value: uint16) =
        BinaryPrimitives.WriteUInt16LittleEndian(buffer.GetSpan 2, value)
        buffer.Advance 2

    /// u32, little-endian.
    member _.U32(value: uint32) =
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.GetSpan 4, value)
        buffer.Advance 4

    /// i32, little-endian two's complement.
    member _.I32(value: int) =
        BinaryPrimitives.WriteInt32LittleEndian(buffer.GetSpan 4, value)
        buffer.Advance 4

    /// u64, little-endian.
    member _.U64(value: uint64) =
        BinaryPrimitives.WriteUInt64LittleEndian(buffer.GetSpan 8, value)
        buffer.Advance 8

    /// i64, little-endian two's complement.
    member _.I64(value: int64) =
        BinaryPrimitives.WriteInt64LittleEndian(buffer.GetSpan 8, value)
        buffer.Advance 8

    /// The raw IEEE 754 bits of a float32, little-endian. Negative zero and NaN payloads are kept as they are.
    member w.F32(value: float32) =
        w.U32(BitConverter.SingleToUInt32Bits value)

    /// The raw IEEE 754 bits of a float, little-endian. Negative zero and NaN payloads are kept as they are.
    member w.F64(value: float) =
        w.U64(BitConverter.DoubleToUInt64Bits value)

    /// Raw bytes with no length prefix, for fixed-size fields such as a ContentId.
    member _.Raw(value: ReadOnlySpan<byte>) =
        value.CopyTo(buffer.GetSpan value.Length)
        buffer.Advance value.Length

    /// A u32 length, then the bytes.
    member w.Bytes(value: ReadOnlySpan<byte>) =
        checkLength value.Length "value"
        w.U32(uint32 value.Length)
        w.Raw value

    /// A u32 length, then the bytes.
    member w.Bytes(value: byte[]) =
        if isNull value then
            nullArg "value"

        w.Bytes(ReadOnlySpan value)

    /// A u32 byte length, then strict UTF-8. Lone surrogates raise; the text is not normalized (callers that need NFC
    /// normalize first).
    member w.String(value: string) =
        if isNull value then
            nullArg "value"

        w.Bytes(CanonicalText.encode "value" value)

    /// The 32 bytes of a ContentId, with no length prefix.
    member w.Id(value: ContentId) = w.Raw(ReadOnlySpan(value.ToBytes()))

    /// 0x00 for None, or 0x01 followed by the value.
    member w.Option(value: 'a option, write: 'a -> unit) =
        match value with
        | None -> w.U8 0uy
        | Some v ->
            w.U8 1uy
            write v

    /// A u32 count, then each item in order. Every item must encode to at least one byte (a zero-width item, such as
    /// unit, an empty record or `Raw` of nothing, raises), so that the reader can bound a count by the bytes left.
    member w.Seq(items: 'a seq, write: 'a -> unit) =
        if isNull (box items) then
            nullArg "items"

        let items = Array.ofSeq items
        w.U32(uint32 items.Length)

        for item in items do
            let before = buffer.WrittenCount
            write item

            if buffer.WrittenCount = before then
                invalidArg "write" "a canonical sequence item must encode to at least one byte"

    /// A u32 count, then (key, value) entries sorted by the UTF-8 bytes of the key (Unicode code point order).
    /// Duplicate keys raise. Values may be zero-width: every entry carries at least its key's u32 length.
    member w.Map(entries: (string * 'a) seq, write: 'a -> unit) =
        if isNull (box entries) then
            nullArg "entries"

        let sorted =
            entries
            |> Seq.map (fun (k, v) ->
                if isNull k then
                    nullArg "key"

                CanonicalText.encode "key" k, v)
            |> Array.ofSeq
            |> Array.sortWith (fun (a, _) (b, _) -> CanonicalText.compareBytes a b)

        for i in 1 .. sorted.Length - 1 do
            if CanonicalText.compareBytes (fst sorted[i - 1]) (fst sorted[i]) = 0 then
                invalidArg "entries" (sprintf "duplicate map key '%s'" (CanonicalText.utf8.GetString(fst sorted[i])))

        w.U32(uint32 sorted.Length)

        for k, v in sorted do
            w.Bytes k
            write v

/// Reader for the canonical encoding. It accepts only canonical input: booleans and option tags other than 0x00 and
/// 0x01, invalid UTF-8, unsorted or duplicate map keys, and truncated data raise FormatException.
[<Sealed>]
type CanonicalReader(data: ReadOnlyMemory<byte>) =
    let mutable pos = 0

    let take (n: int) =
        if n < 0 || n > data.Length - pos then
            raise (
                FormatException(sprintf "canonical data truncated: need %d bytes at offset %d of %d" n pos data.Length)
            )

        let s = data.Span.Slice(pos, n)
        pos <- pos + n
        s

    new(data: byte[]) = CanonicalReader(ReadOnlyMemory data)

    /// Bytes consumed so far.
    member _.Position = pos

    /// Bytes left.
    member _.Remaining = data.Length - pos

    /// Raises FormatException unless every byte was consumed.
    member _.ExpectEnd() =
        if pos <> data.Length then
            raise (FormatException(sprintf "canonical data has %d trailing bytes" (data.Length - pos)))

    member _.U8() = (take 1).[0]

    member r.Bool() =
        match r.U8() with
        | 0uy -> false
        | 1uy -> true
        | b -> raise (FormatException(sprintf "canonical bool must be 0x00 or 0x01, got 0x%02x" b))

    member _.U16() =
        BinaryPrimitives.ReadUInt16LittleEndian(take 2)

    member _.U32() =
        BinaryPrimitives.ReadUInt32LittleEndian(take 4)

    member _.I32() =
        BinaryPrimitives.ReadInt32LittleEndian(take 4)

    member _.U64() =
        BinaryPrimitives.ReadUInt64LittleEndian(take 8)

    member _.I64() =
        BinaryPrimitives.ReadInt64LittleEndian(take 8)

    member r.F32() =
        BitConverter.UInt32BitsToSingle(r.U32())

    member r.F64() =
        BitConverter.UInt64BitsToDouble(r.U64())

    /// `n` raw bytes.
    member _.Raw(n: int) = (take n).ToArray()

    /// A u32 length or count.
    member r.Count() =
        let n = r.U32()

        if n > uint32 Int32.MaxValue then
            raise (FormatException(sprintf "canonical length %d is too large" n))

        int n

    member r.Bytes() = r.Raw(r.Count())

    member r.String() =
        let bytes = r.Bytes()

        try
            CanonicalText.utf8.GetString bytes
        with :? DecoderFallbackException as e ->
            raise (FormatException("canonical string is not valid UTF-8", e))

    member r.Id() = ContentId.OfBytes(r.Raw 32)

    member r.Option(read: unit -> 'a) : 'a option =
        match r.U8() with
        | 0uy -> None
        | 1uy -> Some(read ())
        | b -> raise (FormatException(sprintf "canonical option tag must be 0x00 or 0x01, got 0x%02x" b))

    /// A u32 count of items that each take at least one byte (the writer enforces this), so a count beyond the
    /// remaining bytes is malformed. Checking it first keeps hostile input from forcing a huge allocation.
    member r.ItemCount() =
        let n = r.Count()

        if n > data.Length - pos then
            raise (FormatException(sprintf "canonical count %d exceeds the %d remaining bytes" n (data.Length - pos)))

        n

    /// A u32 count, then the items. An item that reads no bytes raises, as the writer never produces one.
    member r.Seq(read: unit -> 'a) : 'a list =
        let n = r.ItemCount()

        List.init n (fun _ ->
            let before = pos
            let item = read ()

            if pos = before then
                raise (FormatException "canonical sequence item took no bytes (zero-width items are not canonical)")

            item)

    /// A u32 count, then the entries. Every entry carries its key's u32 length, so a map value may be zero-width.
    member r.Map(read: unit -> 'a) : (string * 'a) list =
        let n = r.ItemCount()
        let previous = ref (null: byte[])

        List.init n (fun _ ->
            let key = r.Bytes()

            if
                not (isNull previous.Value)
                && CanonicalText.compareBytes previous.Value key >= 0
            then
                raise (FormatException "canonical map keys must be strictly increasing (sorted, no duplicates)")

            previous.Value <- key

            let text =
                try
                    CanonicalText.utf8.GetString key
                with :? DecoderFallbackException as e ->
                    raise (FormatException("canonical map key is not valid UTF-8", e))

            text, read ())

/// Canonical encoding and domain-separated hashing (LrmCore contract).
[<RequireQualifiedAccess>]
module Canonical =
    /// A tag is non-empty printable ASCII without spaces (0x21 to 0x7E), conventionally "lrm.<thing>/<version>".
    let isValidTag (tag: string) =
        not (String.IsNullOrEmpty tag)
        && tag.Length <= 128
        && tag |> Seq.forall (fun c -> c >= '!' && c <= '~')

    let private checkTag (tag: string) =
        if not (isValidTag tag) then
            invalidArg "tag" (sprintf "invalid canonical tag '%s' (expected 1-128 printable ASCII characters)" tag)

    /// Runs `write` on a fresh writer and returns the bytes.
    let encode (write: CanonicalWriter -> unit) : byte[] =
        let w = CanonicalWriter()
        write w
        w.ToArray()

    /// SHA-256(tag || 0x00 || payload): the domain-separated hash of an already encoded payload.
    let hashPayload (tag: string, payload: ReadOnlySpan<byte>) : ContentId =
        checkTag tag
        use h = IncrementalHash.CreateHash HashAlgorithmName.SHA256
        h.AppendData(Encoding.ASCII.GetBytes tag)
        h.AppendData(ReadOnlySpan [| 0uy |])
        h.AppendData payload
        ContentId.OfBytes(h.GetHashAndReset())

    /// SHA-256(tag || 0x00 || payload) of an encoded payload held in an array.
    let hashBytes (tag: string) (payload: byte[]) : ContentId =
        if isNull payload then
            nullArg "payload"

        hashPayload (tag, ReadOnlySpan payload)

    /// Canonical.hash tag write = SHA-256(tag || 0x00 || encode write).
    let hash (tag: string) (write: CanonicalWriter -> unit) : ContentId =
        checkTag tag
        let w = CanonicalWriter()
        write w
        hashPayload (tag, w.Written)
