namespace Lrm.Cluster

open System
open Lrm.Core

/// The id of a work unit: Canonical.hash "lrm.unit/1" over its canonical encoding (ClusterProtocol contract).
[<Struct>]
type UnitId =
    | UnitId of ContentId

    /// The underlying ContentId.
    member x.Value =
        let (UnitId v) = x
        v

    override x.ToString() = x.Value.Hex

[<RequireQualifiedAccess>]
module UnitId =
    /// "No unit": the predecessor of the first unit of a chain, and the head of an empty chain.
    let none = UnitId ContentId.zero

    /// True for UnitId.none.
    let isNone (id: UnitId) = id.Value = ContentId.zero

    /// The underlying ContentId.
    let value (id: UnitId) = id.Value

/// The id of a blob of raw bytes: Canonical.hashBytes "lrm.blob/1" over the bytes (LrmCore keyspace rule), never a raw
/// SHA-256, so blobs and typed ids can share a store.
[<Struct>]
type BlobId =
    | BlobId of ContentId

    /// The underlying ContentId.
    member x.Value =
        let (BlobId v) = x
        v

    override x.ToString() = x.Value.Hex

[<RequireQualifiedAccess>]
module BlobId =
    /// The domain tag of blob ids (registered by S1 in lrm-core.md).
    [<Literal>]
    let Tag = "lrm.blob/1"

    /// The blob id of `bytes`.
    let ofBytes (bytes: byte[]) =
        if isNull bytes then
            nullArg "bytes"

        BlobId(Canonical.hashBytes Tag bytes)

    /// "No blob": 32 zero bytes, where a message has no output to name (a Ready without a head).
    let none = BlobId ContentId.zero

    /// True for BlobId.none.
    let isNone (id: BlobId) = id.Value = ContentId.zero

    /// The underlying ContentId.
    let value (id: BlobId) = id.Value

/// One hash chain of units: the units of one lane (sequence) at one stage, in step order. For the Qwen pipeline a
/// chain's state on a node is that stage's KV cache for that sequence.
[<Struct>]
type ChainKey = { Lane: uint32; Stage: uint32 }

/// A work unit (ClusterProtocol contract): a pure, content-addressed piece of work. Its output is a deterministic
/// function of the unit and of the state its chain reaches after Prev, which the unit id commits to through the hash
/// chain, so any node of the same execution class can recompute it and duplicates are harmless.
type WorkUnit =
    {
        /// The unit kind, such as "fwd.stage": 1 to 64 characters of a-z, 0-9, '.' and '_', starting with a letter.
        Kind: string
        /// The version of the kind's semantics and Args layout.
        KindVersion: uint16
        /// The job: what is being computed (model, weights, configuration, request). Defined by the workload.
        Job: ContentId
        /// The execution class that must run the unit. Units are only assigned to nodes of this class.
        Class: ExecutionClassId
        /// The lane (sequence or request) within the job.
        Lane: uint32
        /// The pipeline stage.
        Stage: uint32
        /// The position in the chain, counting from 0.
        Step: uint32
        /// PrevUnitId: the unit before this one in the chain (same job, lane and stage), or UnitId.none at step 0.
        Prev: UnitId
        /// The blob id of the input bytes.
        Input: BlobId
        /// Kind-specific arguments in their own canonical encoding; may be empty.
        Args: byte[]
    }

[<RequireQualifiedAccess>]
module WorkUnit =
    /// The domain tag of unit ids.
    [<Literal>]
    let Tag = "lrm.unit/1"

    /// The pipeline-stage forward kind.
    [<Literal>]
    let FwdStage = "fwd.stage"

    /// The largest Args, 64 KiB.
    [<Literal>]
    let MaxArgs = 65536

    /// The largest input or output of one unit: 1 MiB less than a frame payload, so that an Assign, a Result or a
    /// one-entry Log always fits in one frame.
    [<Literal>]
    let MaxData = 66060288

    let private isKindChar (c: char) =
        (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c = '.' || c = '_'

    /// True for a well-formed kind name.
    let isValidKind (kind: string) =
        not (isNull kind)
        && kind.Length >= 1
        && kind.Length <= 64
        && kind[0] >= 'a'
        && kind[0] <= 'z'
        && kind |> Seq.forall isKindChar

    /// The unit when it is well formed: a valid kind, Args of at most 64 KiB, and for fwd.stage the chain rule
    /// (Prev is none exactly at step 0).
    let validate (u: WorkUnit) : Result<WorkUnit, string> =
        if not (isValidKind u.Kind) then
            Error(sprintf "invalid unit kind '%s'" u.Kind)
        elif isNull u.Args then
            Error "Args is null"
        elif u.Args.Length > MaxArgs then
            Error(sprintf "Args of %d bytes exceed %d" u.Args.Length MaxArgs)
        elif Invariant.equals u.Kind FwdStage && (u.Step = 0u) <> UnitId.isNone u.Prev then
            Error(sprintf "fwd.stage step %d must have %s Prev" u.Step (if u.Step = 0u then "no" else "a"))
        else
            Ok u

    let private checkValid (u: WorkUnit) =
        match validate u with
        | Ok u -> u
        | Error e -> invalidArg "u" (sprintf "invalid WorkUnit: %s" e)

    /// The canonical payload, fields in declared order: Kind (string), KindVersion (u16), Job (32 raw bytes), Class
    /// (32 raw bytes), Lane, Stage, Step (u32), Prev (32 raw bytes), Input (32 raw bytes), Args (bytes).
    let encode (w: CanonicalWriter) (u: WorkUnit) =
        w.String u.Kind
        w.U16 u.KindVersion
        w.Id u.Job
        w.Id u.Class.Value
        w.U32 u.Lane
        w.U32 u.Stage
        w.U32 u.Step
        w.Id u.Prev.Value
        w.Id u.Input.Value
        w.Bytes u.Args

    /// Reads a unit written by encode and validates it (FormatException otherwise).
    let read (r: CanonicalReader) : WorkUnit =
        let u =
            {
                Kind = r.String()
                KindVersion = r.U16()
                Job = r.Id()
                Class = ExecutionClassId(r.Id())
                Lane = r.U32()
                Stage = r.U32()
                Step = r.U32()
                Prev = UnitId(r.Id())
                Input = BlobId(r.Id())
                Args = r.Bytes()
            }

        match validate u with
        | Ok u -> u
        | Error e -> raise (FormatException(sprintf "invalid WorkUnit: %s" e))

    /// The canonical bytes of a unit.
    let toBytes (u: WorkUnit) =
        let u = checkValid u
        Canonical.encode (fun w -> encode w u)

    /// A unit from exactly its canonical bytes.
    let ofBytes (bytes: byte[]) =
        let r = CanonicalReader bytes
        let u = read r
        r.ExpectEnd()
        u

    /// UnitId = Canonical.hash "lrm.unit/1" (encode unit). Raises ArgumentException for an invalid unit.
    let id (u: WorkUnit) : UnitId =
        let u = checkValid u
        UnitId(Canonical.hash Tag (fun w -> encode w u))

    /// The chain the unit belongs to.
    let chain (u: WorkUnit) : ChainKey = { Lane = u.Lane; Stage = u.Stage }

    /// A fwd.stage v1 unit. `prev` is the previous unit of the same lane and stage (UnitId.none at step 0).
    let fwdStage
        (job: ContentId)
        (cls: ExecutionClassId)
        (lane: uint32)
        (stage: uint32)
        (step: uint32)
        (prev: UnitId)
        (input: BlobId)
        (args: byte[])
        =
        checkValid
            {
                Kind = FwdStage
                KindVersion = 1us
                Job = job
                Class = cls
                Lane = lane
                Stage = stage
                Step = step
                Prev = prev
                Input = input
                Args = args
            }
