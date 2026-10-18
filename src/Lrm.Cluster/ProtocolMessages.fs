namespace Lrm.Cluster

open System
open Lrm.Core

/// Where a node is with an assigned unit, as its heartbeat reports it.
type UnitPhase =
    /// Queued behind other work, chain not yet examined.
    | Queued = 0uy
    /// Waiting for the unit log to replay its chain up to the unit's Prev.
    | Fetching = 1uy
    /// Replaying its chain.
    | Replaying = 2uy
    /// The chain is at the unit's Prev: executing, or next in line. Deadlines may run.
    | Positioned = 3uy

/// The code of a Notice message.
type NoticeCode =
    /// Coordinator to node: the frame's epoch is not current; Value is the current epoch.
    | StaleEpoch = 1us
    /// Coordinator to node: the sender is not a member; it must send Hello.
    | UnknownMember = 2us
    /// Coordinator to node: the run is over; the node stops.
    | Shutdown = 3us
    /// Node to coordinator: the node leaves the cluster.
    | Leave = 4us
    /// Node to coordinator: the node cannot run the assigned unit (other class or stage).
    | WrongClass = 5us

/// The position of one chain on a node.
type ChainHead =
    {
        Chain: ChainKey
        /// The last unit applied to the chain, or UnitId.none.
        Head: UnitId
        /// The chain state's digest after Head (IChainState.Digest).
        Digest: ContentId
    }

/// One member in a Membership message.
type MemberInfo =
    {
        MemberId: uint64
        MemberName: string
        MemberClass: ExecutionClassId
    }

/// The replicas of one stage.
type StageRole =
    {
        RoleStage: uint32
        Nodes: uint64 list
    }

/// One accepted unit in a Log message: the unit, its input bytes, and the accepted output id and state digest.
type LogEntry =
    {
        Unit: WorkUnit
        Input: byte[]
        Output: BlobId
        StateDigest: ContentId
    }

/// 1. Node to coordinator: join (or rejoin) the cluster.
type HelloMsg =
    {
        Name: string
        /// Distinguishes restarts of one node id; a new incarnation replaces the old member.
        Incarnation: uint64
        Class: ExecutionClassId
        /// The stages this node can host.
        CanHost: uint32 list
        /// The chains the node already holds (after a rejoin), so replay starts from them.
        Heads: ChainHead list
    }

/// 2. Coordinator to node: the node is a member of the current epoch.
type WelcomeMsg =
    {
        HeartbeatMs: uint32
        LeaseMs: uint32
        /// The Seq of the Hello being answered; the node's lease runs from that Hello's send time.
        EchoSeq: uint64
    }

/// 3. Node to coordinator, every heartbeat interval.
type HeartbeatMsg =
    {
        /// The node's executor is loaded (the node-level Ready).
        IsReady: bool
        /// Every unit the node has queued or in progress, with its phase.
        Active: (UnitId * UnitPhase) list
    }

/// 4. Coordinator to node: the reply to a heartbeat, granting a lease.
type LeaseMsg =
    {
        DurationMs: uint32
        /// The Seq of the heartbeat being answered; the lease runs from its send time.
        LeaseEcho: uint64
    }

/// 5. Coordinator to all members: the membership of the frame's epoch and the stage replicas.
type MembershipMsg =
    {
        Members: MemberInfo list
        Roles: StageRole list
    }

/// 6. Coordinator to node: run a unit. The frame's UnitId is the unit's id.
type AssignMsg = { Work: WorkUnit; WorkInput: byte[] }

/// 7. Node to coordinator: the output of a unit and the chain digest after it. The frame's UnitId is the unit's id.
type ResultMsg =
    {
        ResultUnit: UnitId
        ResultOutput: byte[]
        ResultDigest: ContentId
    }

/// 8. Node to coordinator: send the accepted units of a chain after From (exclusive; UnitId.none = from the start) up
/// to To (inclusive). The frame's UnitId is To.
type FetchMsg =
    {
        FetchChain: ChainKey
        From: UnitId
        To: UnitId
    }

/// 9. Coordinator to node: a batch of the unit log of one chain, in step order.
type LogMsg =
    {
        LogChain: ChainKey
        /// The node must drop its state of the chain first and replay from the start (From was not on the chain).
        Reset: bool
        /// The last batch of this reply.
        Final: bool
        Entries: LogEntry list
    }

/// 10. Either direction: a NoticeCode with a value and a detail text.
type NoticeMsg =
    {
        Code: NoticeCode
        Value: uint64
        Detail: string
    }

/// 11. Node to coordinator: Ready. With no chain, the node's executor is loaded; with a chain, the node holds that
/// chain at Head (after a replay), or reports a replayed unit whose recomputation disagrees with the log. Deadlines of
/// a node's assignments count only from Ready. The frame's UnitId is Head.
type ReadyMsg =
    {
        ReadyChain: ChainKey option
        ReadyHead: UnitId
        /// The chain's state digest after Head.
        ReadyDigest: ContentId
        /// The blob id of Head's output as this node computed it; BlobId.none when there is no head.
        ReadyOutput: BlobId
    }

/// A ClusterProtocol message: one case per MessageKind.
[<RequireQualifiedAccess>]
type Message =
    | Hello of HelloMsg
    | Welcome of WelcomeMsg
    | Heartbeat of HeartbeatMsg
    | Lease of LeaseMsg
    | Membership of MembershipMsg
    | Assign of AssignMsg
    | Result of ResultMsg
    | Fetch of FetchMsg
    | Log of LogMsg
    | Notice of NoticeMsg
    | Ready of ReadyMsg

/// A message with the header fields it travels with.
type Envelope =
    {
        Epoch: uint64
        Sender: uint64
        Seq: uint64
        Flags: uint32
        Message: Message
    }

/// Message payloads (canonical encoding, LrmCore contract) and their frames.
[<RequireQualifiedAccess>]
module Message =
    /// The MessageKind of a message.
    let kind (m: Message) =
        match m with
        | Message.Hello _ -> MessageKind.Hello
        | Message.Welcome _ -> MessageKind.Welcome
        | Message.Heartbeat _ -> MessageKind.Heartbeat
        | Message.Lease _ -> MessageKind.Lease
        | Message.Membership _ -> MessageKind.Membership
        | Message.Assign _ -> MessageKind.Assign
        | Message.Result _ -> MessageKind.Result
        | Message.Fetch _ -> MessageKind.Fetch
        | Message.Log _ -> MessageKind.Log
        | Message.Notice _ -> MessageKind.Notice
        | Message.Ready _ -> MessageKind.Ready

    let private writeChain (w: CanonicalWriter) (c: ChainKey) =
        w.U32 c.Lane
        w.U32 c.Stage

    let private readChain (r: CanonicalReader) : ChainKey =
        let lane = r.U32()
        let stage = r.U32()
        { Lane = lane; Stage = stage }

    let private writeHead (w: CanonicalWriter) (h: ChainHead) =
        writeChain w h.Chain
        w.Id h.Head.Value
        w.Id h.Digest

    let private readHead (r: CanonicalReader) : ChainHead =
        let chain = readChain r
        let head = UnitId(r.Id())
        let digest = r.Id()

        {
            Chain = chain
            Head = head
            Digest = digest
        }

    let private checkData (what: string) (bytes: byte[]) =
        if isNull bytes then
            nullArg what

        if bytes.Length > WorkUnit.MaxData then
            invalidArg what (sprintf "%s of %d bytes exceeds %d" what bytes.Length WorkUnit.MaxData)

    let private writeEntry (w: CanonicalWriter) (e: LogEntry) =
        checkData "Input" e.Input
        WorkUnit.encode w e.Unit
        w.Bytes e.Input
        w.Id e.Output.Value
        w.Id e.StateDigest

    /// The encoded size of one Log entry in bytes: the unit's encoding, the u32-prefixed input, the output id and the
    /// state digest. A Log payload is 14 bytes (chain, reset, final, entry count) plus its entries.
    let logEntrySize (e: LogEntry) =
        (WorkUnit.toBytes e.Unit).Length + 4 + e.Input.Length + 64

    let private readEntry (r: CanonicalReader) : LogEntry =
        let u = WorkUnit.read r
        let input = r.Bytes()
        let output = BlobId(r.Id())
        let digest = r.Id()

        if BlobId.ofBytes input <> u.Input then
            raise (FormatException "log entry input does not match the unit's Input blob id")

        {
            Unit = u
            Input = input
            Output = output
            StateDigest = digest
        }

    let private phaseOf (b: byte) =
        if b > 3uy then
            raise (FormatException(sprintf "unknown unit phase %d" b))

        LanguagePrimitives.EnumOfValue<byte, UnitPhase> b

    let private noticeOf (v: uint16) =
        if v < 1us || v > 5us then
            raise (FormatException(sprintf "unknown notice code %d" v))

        LanguagePrimitives.EnumOfValue<uint16, NoticeCode> v

    /// The canonical payload of a message.
    let payload (m: Message) : byte[] =
        Canonical.encode (fun w ->
            match m with
            | Message.Hello h ->
                w.String h.Name
                w.U64 h.Incarnation
                w.Id h.Class.Value
                w.Seq(h.CanHost, w.U32)
                w.Seq(h.Heads, writeHead w)
            | Message.Welcome x ->
                w.U32 x.HeartbeatMs
                w.U32 x.LeaseMs
                w.U64 x.EchoSeq
            | Message.Heartbeat x ->
                w.Bool x.IsReady

                w.Seq(
                    x.Active,
                    fun (id, phase) ->
                        w.Id id.Value
                        w.U8(byte phase)
                )
            | Message.Lease x ->
                w.U32 x.DurationMs
                w.U64 x.LeaseEcho
            | Message.Membership x ->
                w.Seq(
                    x.Members,
                    fun mi ->
                        w.U64 mi.MemberId
                        w.String mi.MemberName
                        w.Id mi.MemberClass.Value
                )

                w.Seq(
                    x.Roles,
                    fun r ->
                        w.U32 r.RoleStage
                        w.Seq(r.Nodes, w.U64)
                )
            | Message.Assign x ->
                checkData "WorkInput" x.WorkInput

                if BlobId.ofBytes x.WorkInput <> x.Work.Input then
                    invalidArg "WorkInput" "input bytes do not match the unit's Input blob id"

                WorkUnit.encode w x.Work
                w.Bytes x.WorkInput
            | Message.Result x ->
                checkData "ResultOutput" x.ResultOutput
                w.Bytes x.ResultOutput
                w.Id x.ResultDigest
            | Message.Fetch x ->
                writeChain w x.FetchChain
                w.Id x.From.Value
                w.Id x.To.Value
            | Message.Log x ->
                writeChain w x.LogChain
                w.Bool x.Reset
                w.Bool x.Final
                w.Seq(x.Entries, writeEntry w)
            | Message.Notice x ->
                w.U16(uint16 x.Code)
                w.U64 x.Value
                w.String x.Detail
            | Message.Ready x ->
                w.Option(x.ReadyChain, writeChain w)
                w.Id x.ReadyHead.Value
                w.Id x.ReadyDigest
                w.Id x.ReadyOutput.Value)

    /// The frame header's UnitId for a message: the unit for Assign and Result, To for Fetch, Head for Ready, and
    /// ContentId.zero otherwise.
    let headerUnitId (m: Message) : ContentId =
        match m with
        | Message.Assign x -> (WorkUnit.id x.Work).Value
        | Message.Result x -> x.ResultUnit.Value
        | Message.Fetch x -> x.To.Value
        | Message.Ready x -> x.ReadyHead.Value
        | _ -> ContentId.zero

    /// The frame of an envelope.
    let toFrame (e: Envelope) : Frame =
        {
            Kind = kind e.Message
            Flags = e.Flags
            Epoch = e.Epoch
            Sender = e.Sender
            Seq = e.Seq
            UnitId = headerUnitId e.Message
            Payload = payload e.Message
        }

    let private parse (f: Frame) : Message =
        let r = CanonicalReader f.Payload

        let m =
            match f.Kind with
            | MessageKind.Hello ->
                let name = r.String()
                let incarnation = r.U64()
                let cls = ExecutionClassId(r.Id())
                let canHost = r.Seq(fun () -> r.U32())
                let heads = r.Seq(fun () -> readHead r)

                Message.Hello
                    {
                        Name = name
                        Incarnation = incarnation
                        Class = cls
                        CanHost = canHost
                        Heads = heads
                    }
            | MessageKind.Welcome ->
                let hb = r.U32()
                let lease = r.U32()
                let echo = r.U64()

                Message.Welcome
                    {
                        HeartbeatMs = hb
                        LeaseMs = lease
                        EchoSeq = echo
                    }
            | MessageKind.Heartbeat ->
                let ready = r.Bool()

                let active =
                    r.Seq(fun () ->
                        let id = UnitId(r.Id())
                        let phase = phaseOf (r.U8())
                        id, phase)

                Message.Heartbeat { IsReady = ready; Active = active }
            | MessageKind.Lease ->
                let duration = r.U32()
                let echo = r.U64()

                Message.Lease
                    {
                        DurationMs = duration
                        LeaseEcho = echo
                    }
            | MessageKind.Membership ->
                let members =
                    r.Seq(fun () ->
                        let id = r.U64()
                        let name = r.String()
                        let cls = ExecutionClassId(r.Id())

                        {
                            MemberId = id
                            MemberName = name
                            MemberClass = cls
                        })

                let roles =
                    r.Seq(fun () ->
                        let stage = r.U32()
                        let nodes = r.Seq(fun () -> r.U64())
                        { RoleStage = stage; Nodes = nodes })

                Message.Membership { Members = members; Roles = roles }
            | MessageKind.Assign ->
                let u = WorkUnit.read r
                let input = r.Bytes()

                if BlobId.ofBytes input <> u.Input then
                    raise (FormatException "Assign input does not match the unit's Input blob id")

                if (WorkUnit.id u).Value <> f.UnitId then
                    raise (FormatException "Assign frame UnitId is not the unit's id")

                Message.Assign { Work = u; WorkInput = input }
            | MessageKind.Result ->
                let output = r.Bytes()
                let digest = r.Id()

                Message.Result
                    {
                        ResultUnit = UnitId f.UnitId
                        ResultOutput = output
                        ResultDigest = digest
                    }
            | MessageKind.Fetch ->
                let chain = readChain r
                let from = UnitId(r.Id())
                let upTo = UnitId(r.Id())

                if upTo.Value <> f.UnitId then
                    raise (FormatException "Fetch frame UnitId is not To")

                Message.Fetch
                    {
                        FetchChain = chain
                        From = from
                        To = upTo
                    }
            | MessageKind.Log ->
                let chain = readChain r
                let reset = r.Bool()
                let final = r.Bool()
                let entries = r.Seq(fun () -> readEntry r)

                Message.Log
                    {
                        LogChain = chain
                        Reset = reset
                        Final = final
                        Entries = entries
                    }
            | MessageKind.Notice ->
                let code = noticeOf (r.U16())
                let value = r.U64()
                let detail = r.String()

                Message.Notice
                    {
                        Code = code
                        Value = value
                        Detail = detail
                    }
            | MessageKind.Ready ->
                let chain = r.Option(fun () -> readChain r)
                let head = UnitId(r.Id())
                let digest = r.Id()
                let output = BlobId(r.Id())

                if head.Value <> f.UnitId then
                    raise (FormatException "Ready frame UnitId is not Head")

                Message.Ready
                    {
                        ReadyChain = chain
                        ReadyHead = head
                        ReadyDigest = digest
                        ReadyOutput = output
                    }
            | k -> raise (FormatException(sprintf "unknown message kind %d" (uint16 k)))

        r.ExpectEnd()

        match m with
        | Message.Assign _
        | Message.Result _
        | Message.Fetch _
        | Message.Ready _ -> ()
        | _ ->
            if f.UnitId <> ContentId.zero then
                raise (FormatException(sprintf "%s frame must have a zero UnitId" (MessageKind.name f.Kind)))

        m

    /// The envelope carried by a decoded frame, or why its payload is not a valid message of its kind.
    let tryOfFrame (f: Frame) : Result<Envelope, string> =
        try
            Ok
                {
                    Epoch = f.Epoch
                    Sender = f.Sender
                    Seq = f.Seq
                    Flags = f.Flags
                    Message = parse f
                }
        with
        | :? FormatException as e -> Error e.Message
        | :? ArgumentException as e -> Error e.Message

    /// The encoded frame of an envelope.
    let encode (e: Envelope) : byte[] = FrameCodec.encode (toFrame e)

    /// Decodes and parses one encoded frame.
    let tryDecode (bytes: byte[]) : Result<Envelope, string> =
        match FrameCodec.tryDecodeArray bytes with
        | Error e -> Error(FrameError.describe e)
        | Ok f -> tryOfFrame f
