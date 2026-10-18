namespace Lrm.Core

open System
open System.Security.Cryptography

/// Path-safe names for runs and nodes (LrmCore contract): 1 to 64 characters from a-z, 0-9, '.', '_' and '-',
/// starting with a letter or digit and not ending with '.'. Lowercase only, so names never collide on
/// case-insensitive file systems, and never "." or "..".
[<RequireQualifiedAccess>]
module SafeName =
    let private isLowerAlnum (c: char) =
        (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')

    let private isAllowed (c: char) =
        isLowerAlnum c || c = '.' || c = '_' || c = '-'

    /// True when `text` is a safe name.
    let isValid (text: string) =
        not (isNull text)
        && text.Length >= 1
        && text.Length <= 64
        && isLowerAlnum text[0]
        && text[text.Length - 1] <> '.'
        && text |> Seq.forall isAllowed

    /// Returns `text` when it is a safe name; raises ArgumentException naming `what` otherwise.
    let check (what: string) (text: string) =
        if not (isValid text) then
            invalidArg
                what
                (sprintf
                    "'%s' is not a valid %s (1-64 characters of a-z, 0-9, '.', '_', '-', starting with a letter or digit)"
                    text
                    what)

        text

/// The id of one run: its folder under LRM_DATA/runs. Run ids are labels; they never enter a content hash.
[<Struct>]
type RunId =
    private
    | RunId of string

    /// The id text.
    member x.Value =
        let (RunId v) = x
        v

    override x.ToString() = x.Value

[<RequireQualifiedAccess>]
module RunId =
    /// A run id from a safe name; None otherwise.
    let tryParse (text: string) =
        if SafeName.isValid text then Some(RunId text) else None

    /// A run id from a safe name; raises ArgumentException otherwise.
    let parse (text: string) = RunId(SafeName.check "run id" text)

    /// The id text.
    let value (id: RunId) = id.Value

    /// "yyyyMMdd-HHmmss-xxxxxxxx": the UTC start time, then 32 bits of `suffix` in hex. Ids made this way sort by
    /// start time.
    let create (utcNow: DateTimeOffset) (suffix: uint32) =
        let t = utcNow.ToUniversalTime()

        RunId(
            t.ToString("yyyyMMdd-HHmmss", Invariant.culture)
            + "-"
            + suffix.ToString("x8", Invariant.culture)
        )

    /// A fresh run id from the current UTC time and 32 random bits from the operating system.
    let fresh () =
        create DateTimeOffset.UtcNow (BitConverter.ToUInt32(RandomNumberGenerator.GetBytes 4, 0))

/// The id of one process in a run, such as "n0" or "coord" (the node part of events.<node>.jsonl).
[<Struct>]
type NodeId =
    private
    | NodeId of string

    /// The id text.
    member x.Value =
        let (NodeId v) = x
        v

    override x.ToString() = x.Value

[<RequireQualifiedAccess>]
module NodeId =
    /// A node id from a safe name; None otherwise.
    let tryParse (text: string) =
        if SafeName.isValid text then Some(NodeId text) else None

    /// A node id from a safe name; raises ArgumentException otherwise.
    let parse (text: string) = NodeId(SafeName.check "node id" text)

    /// The id text.
    let value (id: NodeId) = id.Value

    /// The id of a single-process run: "n0".
    let local = NodeId "n0"
