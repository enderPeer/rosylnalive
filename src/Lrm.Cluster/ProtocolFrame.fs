namespace Lrm.Cluster

open System
open System.Buffers.Binary
open System.IO
open System.Security.Cryptography
open System.Threading
open System.Threading.Tasks
open Lrm.Core

/// The message kinds of the ClusterProtocol: the `kind` field of a frame header (docs/contracts/cluster.md).
type MessageKind =
    | Hello = 1us
    | Welcome = 2us
    | Heartbeat = 3us
    | Lease = 4us
    | Membership = 5us
    | Assign = 6us
    | Result = 7us
    | Fetch = 8us
    | Log = 9us
    | Notice = 10us
    | Ready = 11us

[<RequireQualifiedAccess>]
module MessageKind =
    /// Every kind, in code order.
    let all =
        [
            MessageKind.Hello
            MessageKind.Welcome
            MessageKind.Heartbeat
            MessageKind.Lease
            MessageKind.Membership
            MessageKind.Assign
            MessageKind.Result
            MessageKind.Fetch
            MessageKind.Log
            MessageKind.Notice
            MessageKind.Ready
        ]

    /// True for the assigned codes, 1 to 11.
    let isDefined (code: uint16) = code >= 1us && code <= 11us

    /// The lowercase name used in telemetry and logs.
    let name (kind: MessageKind) =
        match kind with
        | MessageKind.Hello -> "hello"
        | MessageKind.Welcome -> "welcome"
        | MessageKind.Heartbeat -> "heartbeat"
        | MessageKind.Lease -> "lease"
        | MessageKind.Membership -> "membership"
        | MessageKind.Assign -> "assign"
        | MessageKind.Result -> "result"
        | MessageKind.Fetch -> "fetch"
        | MessageKind.Log -> "log"
        | MessageKind.Notice -> "notice"
        | MessageKind.Ready -> "ready"
        | _ -> "unknown"

/// The flag bits of a frame header. Bits that are not defined here must be zero.
[<RequireQualifiedAccess>]
module FrameFlags =
    /// No flags.
    [<Literal>]
    let Empty = 0u

    /// The frame repeats an earlier message (a re-sent Assign, Hello or Fetch). Informational only.
    [<Literal>]
    let Retransmit = 1u

    /// Every defined bit.
    [<Literal>]
    let Known = 1u

/// One frame of the ClusterProtocol: the header fields and the payload. The version is always FrameCodec.Version; the
/// magic, the reserved words, the payload length and the trailer are produced and checked by FrameCodec.
type Frame =
    {
        Kind: MessageKind
        /// FrameFlags bits.
        Flags: uint32
        /// The sender's membership epoch.
        Epoch: uint64
        /// The sender's member id: 0 for the coordinator, the node id otherwise.
        Sender: uint64
        /// The sender's frame sequence number, increasing per sender process.
        Seq: uint64
        /// The work unit the frame is about (Assign, Result, Fetch, Ready), or ContentId.zero.
        UnitId: ContentId
        /// The message payload (canonical encoding, see ProtocolMessages.fs).
        Payload: byte[]
    }

/// Why a byte sequence is not a valid frame.
[<RequireQualifiedAccess>]
type FrameError =
    | Truncated of needed: int * got: int
    | BadMagic of magic: uint32
    | BadVersion of version: uint16
    | UnknownKind of kind: uint16
    | UnknownFlags of flags: uint32
    | ReservedNotZero of offset: int
    | PayloadTooLarge of length: uint32
    | LengthMismatch of expected: int * actual: int
    | BadTrailer

[<RequireQualifiedAccess>]
module FrameError =
    /// One line for logs and exceptions.
    let describe (e: FrameError) =
        match e with
        | FrameError.Truncated(needed, got) -> sprintf "frame truncated: need %d bytes, got %d" needed got
        | FrameError.BadMagic m -> sprintf "bad frame magic 0x%08x (expected \"LRM1\")" m
        | FrameError.BadVersion v -> sprintf "unsupported frame version %d" v
        | FrameError.UnknownKind k -> sprintf "unknown message kind %d" k
        | FrameError.UnknownFlags f -> sprintf "unknown frame flags 0x%08x" f
        | FrameError.ReservedNotZero o -> sprintf "reserved header word at offset %d is not zero" o
        | FrameError.PayloadTooLarge n -> sprintf "payload length %d exceeds the limit" n
        | FrameError.LengthMismatch(expected, actual) -> sprintf "frame is %d bytes, header says %d" actual expected
        | FrameError.BadTrailer -> "frame trailer does not match the SHA-256 of header and payload"

/// The frame codec (ClusterProtocol contract): an 80-byte little-endian header, the payload, and a 32-byte trailer
/// that is the SHA-256 of header and payload.
///
/// | Offset | Size | Field |
/// | 0 | 4 | magic "LRM1" (4C 52 4D 31) |
/// | 4 | 2 | version (1) |
/// | 6 | 2 | kind |
/// | 8 | 4 | flags |
/// | 12 | 4 | reserved, zero |
/// | 16 | 8 | epoch |
/// | 24 | 8 | sender |
/// | 32 | 8 | seq |
/// | 40 | 4 | payload length |
/// | 44 | 4 | reserved, zero |
/// | 48 | 32 | unit id |
[<RequireQualifiedAccess>]
module FrameCodec =
    [<Literal>]
    let HeaderSize = 80

    [<Literal>]
    let TrailerSize = 32

    [<Literal>]
    let Version = 1us

    /// "LRM1" as the bytes 4C 52 4D 31, read as a little-endian u32.
    [<Literal>]
    let Magic = 0x314D524Cu

    /// The largest payload, 64 MiB.
    [<Literal>]
    let MaxPayload = 67108864

    /// Header, payload and trailer.
    let frameSize (payloadLength: int) =
        HeaderSize + payloadLength + TrailerSize

    let private check (f: Frame) =
        if isNull f.Payload then
            nullArg "Payload"

        if f.Payload.Length > MaxPayload then
            invalidArg "Payload" (sprintf "payload of %d bytes exceeds the %d-byte limit" f.Payload.Length MaxPayload)

        if not (MessageKind.isDefined (uint16 f.Kind)) then
            invalidArg "Kind" (sprintf "unknown message kind %d" (uint16 f.Kind))

        if f.Flags &&& ~~~FrameFlags.Known <> 0u then
            invalidArg "Flags" (sprintf "unknown frame flags 0x%08x" f.Flags)

    /// The encoded frame: header, payload, trailer.
    let encode (f: Frame) : byte[] =
        check f
        let n = f.Payload.Length
        let bytes = Array.zeroCreate<byte>(frameSize n)
        let span = Span bytes
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(0, 4), Magic)
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(4, 2), Version)
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(6, 2), uint16 f.Kind)
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(8, 4), f.Flags)
        BinaryPrimitives.WriteUInt64LittleEndian(span.Slice(16, 8), f.Epoch)
        BinaryPrimitives.WriteUInt64LittleEndian(span.Slice(24, 8), f.Sender)
        BinaryPrimitives.WriteUInt64LittleEndian(span.Slice(32, 8), f.Seq)
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(40, 4), uint32 n)
        f.UnitId.WriteTo(span.Slice(48, 32))
        f.Payload.CopyTo(span.Slice HeaderSize)

        SHA256.HashData(ReadOnlySpan(bytes, 0, HeaderSize + n), span.Slice(HeaderSize + n, TrailerSize))
        |> ignore

        bytes

    /// Checks a header and returns the payload length it announces.
    let tryParseHeader (header: ReadOnlySpan<byte>) : Result<int, FrameError> =
        if header.Length < HeaderSize then
            Error(FrameError.Truncated(HeaderSize, header.Length))
        else
            let magic = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(0, 4))
            let version = BinaryPrimitives.ReadUInt16LittleEndian(header.Slice(4, 2))
            let kind = BinaryPrimitives.ReadUInt16LittleEndian(header.Slice(6, 2))
            let flags = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(8, 4))
            let reserved0 = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(12, 4))
            let length = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(40, 4))
            let reserved1 = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(44, 4))

            if magic <> Magic then
                Error(FrameError.BadMagic magic)
            elif version <> Version then
                Error(FrameError.BadVersion version)
            elif not (MessageKind.isDefined kind) then
                Error(FrameError.UnknownKind kind)
            elif flags &&& ~~~FrameFlags.Known <> 0u then
                Error(FrameError.UnknownFlags flags)
            elif reserved0 <> 0u then
                Error(FrameError.ReservedNotZero 12)
            elif reserved1 <> 0u then
                Error(FrameError.ReservedNotZero 44)
            elif length > uint32 MaxPayload then
                Error(FrameError.PayloadTooLarge length)
            else
                Ok(int length)

    /// Decodes exactly one whole frame and checks its trailer.
    let tryDecode (bytes: ReadOnlySpan<byte>) : Result<Frame, FrameError> =
        match tryParseHeader bytes with
        | Error e -> Error e
        | Ok n ->
            let total = frameSize n

            if bytes.Length <> total then
                Error(FrameError.LengthMismatch(total, bytes.Length))
            else
                let expected = SHA256.HashData(bytes.Slice(0, HeaderSize + n))

                if not (bytes.Slice(HeaderSize + n, TrailerSize).SequenceEqual(ReadOnlySpan expected)) then
                    Error FrameError.BadTrailer
                else
                    Ok
                        {
                            Kind =
                                LanguagePrimitives.EnumOfValue<uint16, MessageKind>(
                                    BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(6, 2))
                                )
                            Flags = BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(8, 4))
                            Epoch = BinaryPrimitives.ReadUInt64LittleEndian(bytes.Slice(16, 8))
                            Sender = BinaryPrimitives.ReadUInt64LittleEndian(bytes.Slice(24, 8))
                            Seq = BinaryPrimitives.ReadUInt64LittleEndian(bytes.Slice(32, 8))
                            UnitId = ContentId.OfSpan(bytes.Slice(48, 32))
                            Payload = bytes.Slice(HeaderSize, n).ToArray()
                        }

    /// tryDecode over a whole array.
    let tryDecodeArray (bytes: byte[]) : Result<Frame, FrameError> =
        if isNull bytes then
            nullArg "bytes"

        tryDecode (ReadOnlySpan bytes)

    /// tryParseHeader over the start of an array.
    let tryParseHeaderArray (bytes: byte[]) : Result<int, FrameError> =
        if isNull bytes then
            nullArg "bytes"

        tryParseHeader (ReadOnlySpan bytes)

    /// Decodes one frame; raises FormatException when it is not valid.
    let decode (bytes: byte[]) : Frame =
        match tryDecodeArray bytes with
        | Ok f -> f
        | Error e -> raise (FormatException(FrameError.describe e))

    /// The trailer of an encoded frame: the SHA-256 of its header and payload. It is an integrity check, not a
    /// content address, and never a key in a ContentId store.
    let trailer (bytes: byte[]) : ContentId =
        if isNull bytes || bytes.Length < HeaderSize + TrailerSize then
            invalidArg "bytes" "not an encoded frame"

        ContentId.OfSpan(ReadOnlySpan(bytes, bytes.Length - TrailerSize, TrailerSize))

/// Frames over byte streams (TCP): frames are self-delimiting, since the header carries the payload length.
[<RequireQualifiedAccess>]
module FrameStream =
    /// Reads the next whole frame from `stream` and checks it. Returns its bytes, or None at a clean end of stream
    /// before the first header byte. Raises FormatException for a malformed or truncated frame; the stream cannot be
    /// resynchronized after that, so the caller closes it.
    let readAsync (stream: Stream) (ct: CancellationToken) : Task<byte[] option> =
        task {
            let header = Array.zeroCreate<byte> FrameCodec.HeaderSize

            let! got =
                stream.ReadAtLeastAsync(header.AsMemory(), FrameCodec.HeaderSize, false, ct)

            if got = 0 then
                return None
            elif got < FrameCodec.HeaderSize then
                return raise (FormatException(sprintf "stream ended inside a frame header (%d bytes)" got))
            else
                match FrameCodec.tryParseHeaderArray header with
                | Error e -> return raise (FormatException(FrameError.describe e))
                | Ok n ->
                    let bytes = Array.zeroCreate<byte>(FrameCodec.frameSize n)
                    Buffer.BlockCopy(header, 0, bytes, 0, FrameCodec.HeaderSize)
                    let rest = n + FrameCodec.TrailerSize

                    let! got2 =
                        stream.ReadAtLeastAsync(bytes.AsMemory(FrameCodec.HeaderSize, rest), rest, false, ct)

                    if got2 < rest then
                        return raise (FormatException(sprintf "stream ended inside a frame (%d of %d bytes)" got2 rest))
                    else
                        match FrameCodec.tryDecodeArray bytes with
                        | Error e -> return raise (FormatException(FrameError.describe e))
                        | Ok _ -> return Some bytes
        }
