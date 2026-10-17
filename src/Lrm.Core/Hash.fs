namespace Lrm.Core

open System
open System.Buffers.Binary
open System.IO
open System.Security.Cryptography
open System.Text

/// A 32-byte content address, the SHA-256 of something (LrmCore contract).
///
/// Equality is byte equality, and ordering is lexicographic over the 32 bytes (the four words hold the bytes
/// big-endian, so the generated structural comparison is exactly that). The text form is 64 lowercase hex digits.
[<Struct; CustomEquality; CustomComparison>]
type ContentId =
    private
        {
            W0: uint64
            W1: uint64
            W2: uint64
            W3: uint64
        }

    /// Number of bytes in a ContentId.
    static member Size = 32

    /// The id whose 32 bytes are all zero, for "no predecessor" fields. It is never the hash of real content.
    static member Zero =
        {
            W0 = 0UL
            W1 = 0UL
            W2 = 0UL
            W3 = 0UL
        }

    /// Builds an id from exactly 32 bytes.
    static member OfSpan(bytes: ReadOnlySpan<byte>) =
        if bytes.Length <> 32 then
            invalidArg "bytes" (sprintf "a ContentId is 32 bytes, got %d" bytes.Length)

        {
            W0 = BinaryPrimitives.ReadUInt64BigEndian(bytes.Slice(0, 8))
            W1 = BinaryPrimitives.ReadUInt64BigEndian(bytes.Slice(8, 8))
            W2 = BinaryPrimitives.ReadUInt64BigEndian(bytes.Slice(16, 8))
            W3 = BinaryPrimitives.ReadUInt64BigEndian(bytes.Slice(24, 8))
        }

    /// Builds an id from exactly 32 bytes.
    static member OfBytes(bytes: byte[]) =
        if isNull bytes then
            nullArg "bytes"

        ContentId.OfSpan(ReadOnlySpan bytes)

    /// Writes the 32 bytes to the start of `destination`.
    member x.WriteTo(destination: Span<byte>) =
        if destination.Length < 32 then
            invalidArg "destination" "a ContentId needs 32 bytes"

        BinaryPrimitives.WriteUInt64BigEndian(destination.Slice(0, 8), x.W0)
        BinaryPrimitives.WriteUInt64BigEndian(destination.Slice(8, 8), x.W1)
        BinaryPrimitives.WriteUInt64BigEndian(destination.Slice(16, 8), x.W2)
        BinaryPrimitives.WriteUInt64BigEndian(destination.Slice(24, 8), x.W3)

    /// The 32 bytes, in order.
    member x.ToBytes() =
        let bytes = Array.zeroCreate<byte> 32
        x.WriteTo(Span bytes)
        bytes

    /// 64 lowercase hex digits.
    member x.Hex = Convert.ToHexStringLower(x.ToBytes())

    /// The first 12 hex digits, for logs and reports. Never use it as a key.
    member x.Short = x.Hex.Substring(0, 12)

    override x.ToString() = x.Hex

    member private x.Order(y: ContentId) =
        let c0 = compare x.W0 y.W0

        if c0 <> 0 then
            c0
        else
            let c1 = compare x.W1 y.W1

            if c1 <> 0 then
                c1
            else
                let c2 = compare x.W2 y.W2
                if c2 <> 0 then c2 else compare x.W3 y.W3

    override x.Equals(other: obj) =
        match other with
        | :? ContentId as y -> x.W0 = y.W0 && x.W1 = y.W1 && x.W2 = y.W2 && x.W3 = y.W3
        | _ -> false

    override x.GetHashCode() = int (x.W0 >>> 32) ^^^ int x.W0

    interface IEquatable<ContentId> with
        member x.Equals(y: ContentId) =
            x.W0 = y.W0 && x.W1 = y.W1 && x.W2 = y.W2 && x.W3 = y.W3

    interface IComparable<ContentId> with
        member x.CompareTo(y: ContentId) = x.Order y

    interface IComparable with
        member x.CompareTo(other: obj) =
            match other with
            | :? ContentId as y -> x.Order y
            | _ -> invalidArg "other" "not a ContentId"

/// Functions over ContentId.
[<RequireQualifiedAccess>]
module ContentId =
    /// The all-zero id.
    let zero = ContentId.Zero

    /// Builds an id from exactly 32 bytes.
    let ofBytes (bytes: byte[]) = ContentId.OfBytes bytes

    /// The 32 bytes.
    let toBytes (id: ContentId) = id.ToBytes()

    /// 64 lowercase hex digits.
    let hex (id: ContentId) = id.Hex

    /// Parses 64 hex digits (either case). None on any other text.
    let tryParse (text: string) : ContentId option =
        if isNull text || text.Length <> 64 then
            None
        else
            Invariant.tryOfHex text |> Option.map ContentId.OfBytes

    /// Parses 64 hex digits; raises FormatException on any other text.
    let parse (text: string) =
        match tryParse text with
        | Some id -> id
        | None -> raise (FormatException(sprintf "not a ContentId (64 hex digits): '%s'" text))

/// SHA-256, the only hash in LRM (LrmCore contract). Domain-separated hashing of structured values is Canonical.hash.
[<RequireQualifiedAccess>]
module Hash =
    /// SHA-256 of a byte span.
    let sha256Span (data: ReadOnlySpan<byte>) : ContentId =
        let digest = Array.zeroCreate<byte> 32
        SHA256.HashData(data, Span digest) |> ignore
        ContentId.OfBytes digest

    /// SHA-256 of a byte array.
    let sha256 (data: byte[]) : ContentId =
        if isNull data then
            nullArg "data"

        sha256Span (ReadOnlySpan data)

    /// SHA-256 of the UTF-8 bytes of a string (strict: lone surrogates raise).
    let sha256Utf8 (text: string) : ContentId =
        if isNull text then
            nullArg "text"

        sha256 (UTF8Encoding(false, true).GetBytes text)

    /// SHA-256 of the rest of a stream.
    let sha256Stream (stream: Stream) : ContentId =
        if isNull stream then
            nullArg "stream"

        ContentId.OfBytes(SHA256.HashData stream)

    /// SHA-256 of a file's bytes.
    let sha256File (path: string) : ContentId =
        use stream =
            new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 <<< 20, FileOptions.SequentialScan)

        sha256Stream stream

    /// Incremental SHA-256 for data that arrives in pieces.
    [<Sealed>]
    type Incremental() =
        let hash = IncrementalHash.CreateHash HashAlgorithmName.SHA256

        /// Appends bytes.
        member _.Append(data: ReadOnlySpan<byte>) = hash.AppendData data

        /// Appends a byte array.
        member _.Append(data: byte[]) = hash.AppendData data

        /// Returns the hash of everything appended and resets to the empty state.
        member _.Finish() : ContentId =
            ContentId.OfBytes(hash.GetHashAndReset())

        interface IDisposable with
            member _.Dispose() = hash.Dispose()
