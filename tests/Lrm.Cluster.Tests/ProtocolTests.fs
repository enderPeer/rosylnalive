/// Frames, work units and messages (ClusterProtocol contract): known answers, round trips and rejection of every
/// malformed input.
module Lrm.Cluster.Tests.ProtocolTests

open System
open System.IO
open System.Threading
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Cluster.Tests.Common
open Lrm.Testing

let private leaseFrame =
    Message.toFrame
        {
            Epoch = 7UL
            Sender = 0UL
            Seq = 42UL
            Flags = FrameFlags.Empty
            Message = Message.Lease { DurationMs = 1000u; LeaseEcho = 41UL }
        }

let private sampleMessages: Message list =
    let chain = { Lane = 1u; Stage = 2u }
    let digest = Hash.sha256Utf8 "digest"

    [
        Message.Hello
            {
                Name = "n3"
                Incarnation = 0x0123456789ABCDEFUL
                Class = FakeJob.classId
                CanHost = [ 0u; 2u ]
                Heads =
                    [
                        {
                            Chain = chain
                            Head = WorkUnit.id u1
                            Digest = digest
                        }
                    ]
            }
        Message.Welcome
            {
                HeartbeatMs = 100u
                LeaseMs = 1000u
                EchoSeq = 5UL
            }
        Message.Heartbeat
            {
                IsReady = true
                Active = [ WorkUnit.id u1, UnitPhase.Positioned; WorkUnit.id u2, UnitPhase.Fetching ]
            }
        Message.Lease { DurationMs = 1000u; LeaseEcho = 41UL }
        Message.Membership
            {
                Members =
                    [
                        {
                            MemberId = 1UL
                            MemberName = "n1"
                            MemberClass = FakeJob.classId
                        }
                    ]
                Roles = [ { RoleStage = 0u; Nodes = [ 1UL; 2UL ] } ]
            }
        Message.Assign { Work = u1; WorkInput = "abc"B }
        Message.Result
            {
                ResultUnit = WorkUnit.id u1
                ResultOutput = [| 1uy; 2uy; 3uy; 4uy |]
                ResultDigest = digest
            }
        Message.Fetch
            {
                FetchChain = chain
                From = UnitId.none
                To = WorkUnit.id u1
            }
        Message.Log
            {
                LogChain = chain
                Reset = true
                Final = false
                Entries =
                    [
                        {
                            Unit = u1
                            Input = "abc"B
                            Output = BlobId.ofBytes [| 9uy |]
                            StateDigest = digest
                        }
                    ]
            }
        Message.Notice
            {
                Code = NoticeCode.StaleEpoch
                Value = 12UL
                Detail = "stale epoch"
            }
        Message.Ready
            {
                ReadyChain = Some chain
                ReadyHead = WorkUnit.id u1
                ReadyDigest = digest
                ReadyOutput = BlobId.ofBytes [| 9uy |]
            }
        Message.Ready
            {
                ReadyChain = None
                ReadyHead = UnitId.none
                ReadyDigest = ContentId.zero
                ReadyOutput = BlobId.none
            }
    ]

let private envelope (m: Message) =
    {
        Epoch = 9UL
        Sender = 3UL
        Seq = 77UL
        Flags = FrameFlags.Empty
        Message = m
    }

/// Re-encodes a frame after changing its bytes, with a fresh trailer, so only the header check can reject it.
let private retrail (bytes: byte[]) =
    let n = bytes.Length - FrameCodec.TrailerSize
    let copy = Array.copy bytes
    let digest = Security.Cryptography.SHA256.HashData(ReadOnlySpan(copy, 0, n))
    Array.blit digest 0 copy n FrameCodec.TrailerSize
    copy

/// A stream that returns at most one byte per read, to test reassembly of fragmented frames.
type private Trickle(data: byte[]) =
    inherit Stream()
    let mutable pos = 0
    override _.CanRead = true
    override _.CanSeek = false
    override _.CanWrite = false
    override _.Length = int64 data.Length

    override _.Position
        with get () = int64 pos
        and set _ = raise (NotSupportedException())

    override _.Flush() = ()

    override _.Read(buffer: byte[], offset: int, count: int) =
        if pos >= data.Length || count = 0 then
            0
        else
            buffer[offset] <- data[pos]
            pos <- pos + 1
            1

    override _.Seek(_, _) = raise (NotSupportedException())
    override _.SetLength _ = raise (NotSupportedException())
    override _.Write(_, _, _) = raise (NotSupportedException())

let frameTests =
    testList
        "frame"
        [
            testCase "F1 known answer: a Lease frame encodes to the contract bytes"
            <| fun () ->
                Expect.equal (hex leaseFrame.Payload) Vectors.f1Payload "payload"
                let bytes = FrameCodec.encode leaseFrame
                Expect.equal bytes.Length 124 "80 + 12 + 32 bytes"
                Expect.equal (hex bytes) Vectors.f1Frame "frame bytes"

                Expect.equal
                    (FrameCodec.trailer bytes).Hex
                    (Vectors.f1Frame.Substring(Vectors.f1Frame.Length - 64))
                    "trailer"

            testCase "F2 known answer: an Assign frame carries the unit id in its header"
            <| fun () ->
                let bytes =
                    Message.encode
                        {
                            Epoch = 3UL
                            Sender = 0UL
                            Seq = 9UL
                            Flags = FrameFlags.Retransmit
                            Message = Message.Assign { Work = u1; WorkInput = "abc"B }
                        }

                Expect.equal (hex bytes) Vectors.f2Frame "frame bytes"

            testCase "the header is 80 bytes and the trailer 32"
            <| fun () ->
                Expect.equal FrameCodec.HeaderSize 80 "header"
                Expect.equal FrameCodec.TrailerSize 32 "trailer"
                Expect.equal (FrameCodec.frameSize 0) 112 "empty payload"

            testCase "kinds 1 to 10 plus Ready = 11"
            <| fun () ->
                Expect.equal [ for k in MessageKind.all -> uint16 k ] [ 1us .. 11us ] "codes"
                Expect.equal (uint16 MessageKind.Ready) 11us "Ready"
                Expect.isFalse (MessageKind.isDefined 0us) "0"
                Expect.isFalse (MessageKind.isDefined 12us) "12"

            Props.testProperty "any frame round-trips"
            <| fun (kind: byte) (flags: bool) (epoch: uint64) (sender: uint64) (seq: uint64) (payload: byte[]) ->
                let payload = if isNull payload then [||] else payload

                let f =
                    {
                        Kind = LanguagePrimitives.EnumOfValue<uint16, MessageKind>(1us + uint16 (kind % 11uy))
                        Flags = (if flags then FrameFlags.Retransmit else FrameFlags.Empty)
                        Epoch = epoch
                        Sender = sender
                        Seq = seq
                        UnitId = Hash.sha256 payload
                        Payload = payload
                    }

                FrameCodec.decode (FrameCodec.encode f) = f

            testCase "every single-bit flip of a frame is rejected"
            <| fun () ->
                let bytes = FrameCodec.encode leaseFrame

                for i in 0 .. bytes.Length - 1 do
                    for bit in 0..7 do
                        let copy = Array.copy bytes
                        copy[i] <- copy[i] ^^^ (1uy <<< bit)

                        match FrameCodec.tryDecodeArray copy with
                        | Ok _ -> failtestf "flipping bit %d of byte %d was not detected" bit i
                        | Error _ -> ()

            testCase "header checks name the problem"
            <| fun () ->
                let bytes = FrameCodec.encode leaseFrame

                let mutate (offset: int) (value: byte) =
                    let copy = Array.copy bytes
                    copy[offset] <- value
                    FrameCodec.tryDecodeArray (retrail copy)

                Expect.equal (mutate 0 0uy) (Error(FrameError.BadMagic 0x314D5200u)) "magic"
                Expect.equal (mutate 4 2uy) (Error(FrameError.BadVersion 2us)) "version"
                Expect.equal (mutate 6 12uy) (Error(FrameError.UnknownKind 12us)) "kind"
                Expect.equal (mutate 6 0uy) (Error(FrameError.UnknownKind 0us)) "kind 0"
                Expect.equal (mutate 8 2uy) (Error(FrameError.UnknownFlags 2u)) "flags"
                Expect.equal (mutate 12 1uy) (Error(FrameError.ReservedNotZero 12)) "reserved 12"
                Expect.equal (mutate 44 1uy) (Error(FrameError.ReservedNotZero 44)) "reserved 44"
                Expect.equal (mutate 40 13uy) (Error(FrameError.LengthMismatch(125, 124))) "length"
                Expect.equal (mutate 43 0x10uy) (Error(FrameError.PayloadTooLarge 0x1000000Cu)) "too large"

                Expect.equal
                    (FrameCodec.tryDecode (ReadOnlySpan(bytes, 0, 50)))
                    (Error(FrameError.Truncated(80, 50)))
                    "short"

                let tail = Array.copy bytes
                tail[tail.Length - 1] <- tail[tail.Length - 1] ^^^ 1uy
                Expect.equal (FrameCodec.tryDecodeArray tail) (Error FrameError.BadTrailer) "trailer"

            testCase "the encoder refuses what the decoder would reject"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () ->
                        FrameCodec.encode
                            { leaseFrame with
                                Kind = LanguagePrimitives.EnumOfValue 12us
                            }
                        |> ignore)
                    "kind"

                Expect.throwsT<ArgumentException>
                    (fun () -> FrameCodec.encode { leaseFrame with Flags = 4u } |> ignore)
                    "flags"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        FrameCodec.encode
                            { leaseFrame with
                                Payload = Array.zeroCreate (FrameCodec.MaxPayload + 1)
                            }
                        |> ignore)
                    "payload limit"

            testCase "a 64 MiB payload is the largest that encodes and decodes"
            <| fun () ->
                let f =
                    { leaseFrame with
                        Payload = Array.zeroCreate FrameCodec.MaxPayload
                    }

                let bytes = FrameCodec.encode f
                Expect.equal (FrameCodec.decode bytes).Payload.Length FrameCodec.MaxPayload "decoded"

            testCase "frames reassemble from a stream that returns one byte per read"
            <| fun () ->
                let a = FrameCodec.encode leaseFrame
                let b = Message.encode (envelope (Message.Assign { Work = u1; WorkInput = "abc"B }))
                use s = new Trickle(Array.append a b)

                let next () =
                    (FrameStream.readAsync s CancellationToken.None).Result

                let first = next ()
                let second = next ()
                let third = next ()
                Expect.equal (first |> Option.map fst) (Some a) "first"
                Expect.equal (first |> Option.map snd) (Some leaseFrame) "first, decoded"
                Expect.equal (second |> Option.map fst) (Some b) "second"
                Expect.equal third None "clean end"

            testCase "a stream that ends inside a frame raises"
            <| fun () ->
                let a = FrameCodec.encode leaseFrame

                for cut in [ 1; 79; 80; 100; a.Length - 1 ] do
                    use s = new MemoryStream(Array.sub a 0 cut)

                    Expect.throws
                        (fun () ->
                            (FrameStream.readAsync s CancellationToken.None).GetAwaiter().GetResult()
                            |> ignore)
                        (sprintf "cut at %d" cut)

            testCase "a header that announces 64 MiB on a stream that then ends raises"
            <| fun () ->
                // A valid header whose length says MaxPayload, then 10 bytes and the end: the reader grows its
                // buffer with the bytes that arrive, and reports the short frame.
                let header = Array.sub (FrameCodec.encode leaseFrame) 0 FrameCodec.HeaderSize

                Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(
                    Span(header, 40, 4),
                    uint32 FrameCodec.MaxPayload
                )

                Expect.equal (FrameCodec.tryParseHeaderArray header) (Ok FrameCodec.MaxPayload) "a valid header"
                use s = new MemoryStream(Array.append header (Array.zeroCreate 10))

                let e =
                    Expect.throwsC
                        (fun () ->
                            (FrameStream.readAsync s CancellationToken.None).GetAwaiter().GetResult()
                            |> ignore)
                        id

                Expect.stringContains e.Message "10 of 67108896 bytes" "how much arrived"
        ]

let unitTests =
    testList
        "unit"
        [
            testCase "the fake execution class id is the contract value"
            <| fun () -> Expect.equal FakeJob.classId.Value.Hex Vectors.fakeClass "class id"

            testCase "U1 known answer: payload and UnitId"
            <| fun () ->
                Expect.equal (hex (WorkUnit.toBytes u1)) Vectors.u1Payload "payload"
                Expect.equal (WorkUnit.id u1).Value.Hex Vectors.u1Id "unit id"

            testCase "U2 known answer: the next unit of the chain names U1 as Prev"
            <| fun () ->
                Expect.equal (hex (WorkUnit.toBytes u2)) Vectors.u2Payload "payload"
                Expect.equal (WorkUnit.id u2).Value.Hex Vectors.u2Id "unit id"
                Expect.equal u2.Prev (WorkUnit.id u1) "chained"

            testCase "UnitId = Canonical.hash lrm.unit/1 over the encoding"
            <| fun () ->
                let manual = Canonical.hashBytes WorkUnit.Tag (WorkUnit.toBytes u1)
                Expect.equal (WorkUnit.id u1).Value manual "tagged hash"
                Expect.notEqual (WorkUnit.id u1).Value (Hash.sha256 (WorkUnit.toBytes u1)) "not a raw SHA-256"

            testCase "BlobId is lrm.blob/1, the S1 keyspace for raw bytes"
            <| fun () ->
                Expect.equal
                    (BlobId.ofBytes "abc"B).Value.Hex
                    "43dc986338811e8f6ba04bbba1545fdb098e2e138a1392cb276961ebfe901a2c"
                    "lrm-core.md vector C4"

            testCase "every field changes the id"
            <| fun () ->
                let variants =
                    [
                        { u2 with Kind = "fwd.other" }
                        { u2 with KindVersion = 2us }
                        { u2 with Job = Hash.sha256Utf8 "job" }
                        { u2 with
                            Class = ExecutionClass.id (ExecutionClass.forCpu ExecutionBackend.CpuF32)
                        }
                        { u2 with Lane = 0u }
                        { u2 with Stage = 0u }
                        { u2 with Step = 2u }
                        { u2 with Prev = WorkUnit.id u2 }
                        { u2 with
                            Input = BlobId.ofBytes [| 0uy |]
                        }
                        { u2 with Args = [| 0uy |] }
                    ]

                let ids = (WorkUnit.id u2) :: (variants |> List.map WorkUnit.id)
                Expect.equal (List.distinct ids).Length ids.Length "all ids differ"

            testCase "units round-trip through their bytes"
            <| fun () ->
                for u in [ u1; u2 ] do
                    Expect.equal (WorkUnit.ofBytes (WorkUnit.toBytes u)) u "round trip"

            testCase "validation: kind, args and the fwd.stage chain rule"
            <| fun () ->
                Expect.isError (WorkUnit.validate { u1 with Kind = "Fwd" }) "upper case kind"
                Expect.isError (WorkUnit.validate { u1 with Kind = "" }) "empty kind"
                Expect.isError (WorkUnit.validate { u1 with Step = 1u }) "step 1 without Prev"
                Expect.isError (WorkUnit.validate { u2 with Step = 0u }) "step 0 with Prev"

                Expect.isError
                    (WorkUnit.validate
                        { u1 with
                            Args = Array.zeroCreate (WorkUnit.MaxArgs + 1)
                        })
                    "args too large"

                Expect.throwsT<ArgumentException> (fun () -> WorkUnit.id { u1 with Step = 1u } |> ignore) "id refuses"

                let bytes = WorkUnit.toBytes u1

                Expect.throwsT<FormatException>
                    (fun () -> WorkUnit.ofBytes (Array.append bytes [| 0uy |]) |> ignore)
                    "trailing bytes"
        ]

let messageTests =
    testList
        "message"
        [
            testCase "every message kind round-trips through a frame"
            <| fun () ->
                Expect.equal
                    (sampleMessages |> List.map Message.kind |> List.distinct |> List.length)
                    11
                    "all kinds covered"

                for m in sampleMessages do
                    let e = envelope m
                    let decoded = Message.tryDecode (Message.encode e)
                    Expect.equal decoded (Ok e) (sprintf "%A" (Message.kind m))

            testCase "the header UnitId is the unit for Assign and Result, To for Fetch, Head for Ready"
            <| fun () ->
                for m in sampleMessages do
                    let f = Message.toFrame (envelope m)

                    let expected =
                        match m with
                        | Message.Assign _
                        | Message.Result _
                        | Message.Fetch _ -> (WorkUnit.id u1).Value
                        | Message.Ready r -> r.ReadyHead.Value
                        | _ -> ContentId.zero

                    Expect.equal f.UnitId expected (sprintf "%A" f.Kind)

            testCase "a frame whose UnitId disagrees with its payload is rejected"
            <| fun () ->
                // A Result names its unit only in the header (the trailer protects it), so it has nothing to disagree with.
                for m in sampleMessages |> List.filter (fun m -> Message.kind m <> MessageKind.Result) do
                    let f =
                        { Message.toFrame (envelope m) with
                            UnitId = Hash.sha256Utf8 "other"
                        }

                    Expect.isError (Message.tryOfFrame f) (sprintf "%A" f.Kind)

            testCase "an Assign whose input does not match the unit is refused and rejected"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () -> Message.payload (Message.Assign { Work = u1; WorkInput = "abd"B }) |> ignore)
                    "encoder"

                let good =
                    Message.toFrame (envelope (Message.Assign { Work = u1; WorkInput = "abc"B }))

                let payload = Array.copy good.Payload
                payload[payload.Length - 1] <- byte 'd'
                Expect.isError (Message.tryOfFrame { good with Payload = payload }) "decoder"

            testCase "payloads must be canonical and complete"
            <| fun () ->
                for m in sampleMessages do
                    let f = Message.toFrame (envelope m)

                    Expect.isError
                        (Message.tryOfFrame
                            { f with
                                Payload = Array.append f.Payload [| 0uy |]
                            })
                        (sprintf "%A trailing byte" f.Kind)

                    if f.Payload.Length > 0 then
                        Expect.isError
                            (Message.tryOfFrame
                                { f with
                                    Payload = Array.sub f.Payload 0 (f.Payload.Length - 1)
                                })
                            (sprintf "%A truncated" f.Kind)

            testCase "unknown phases and notice codes are rejected"
            <| fun () ->
                let hb =
                    Message.toFrame (
                        envelope (
                            Message.Heartbeat
                                {
                                    IsReady = false
                                    Active = [ WorkUnit.id u1, UnitPhase.Queued ]
                                }
                        )
                    )

                let p = Array.copy hb.Payload
                p[p.Length - 1] <- 4uy
                Expect.isError (Message.tryOfFrame { hb with Payload = p }) "phase 4"

                let n =
                    Message.toFrame (
                        envelope (
                            Message.Notice
                                {
                                    Code = NoticeCode.Shutdown
                                    Value = 0UL
                                    Detail = ""
                                }
                        )
                    )

                let q = Array.copy n.Payload
                q[0] <- 6uy
                Expect.isError (Message.tryOfFrame { n with Payload = q }) "notice 6"
        ]

let tests = testList "protocol" [ frameTests; unitTests; messageTests ]
