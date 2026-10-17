namespace Lrm.Cluster

open System.Diagnostics

/// A monotonic clock in nanoseconds. Cluster logic reads time only through a clock, so the simulator can run it on
/// virtual time. Clock values order events and set deadlines; they never enter a content hash.
type IClock =
    /// The current time in ns. Only differences are meaningful.
    abstract NowNs: int64

/// Virtual time for the deterministic simulator: it only moves when told to, and never backwards.
[<Sealed>]
type VirtualClock(startNs: int64) =
    let mutable now = startNs

    new() = VirtualClock 0L

    /// The current virtual time in ns.
    member _.NowNs = now

    /// Moves the clock to `t`; raises ArgumentException when `t` is in the past.
    member _.AdvanceTo(t: int64) =
        if t < now then
            invalidArg "t" (sprintf "virtual time cannot go back from %d to %d ns" now t)

        now <- t

    /// Moves the clock forward by `delta` ns (at least 0).
    member c.Advance(delta: int64) =
        if delta < 0L then
            invalidArg "delta" "negative delta"

        c.AdvanceTo(now + delta)

    /// A telemetry clock (Telemetry.ofSinkWithClock) on this virtual time: virtual ns since the Unix epoch.
    member _.TelemetryClock: unit -> int64 = fun () -> now

    interface IClock with
        member _.NowNs = now

/// The process's monotonic clock: Stopwatch time since the first use.
[<Sealed>]
type SystemClock private () =
    static let origin = Stopwatch.GetTimestamp()

    /// The shared instance.
    static member Instance = SystemClock()

    /// ns since the first use in this process.
    member _.NowNs =
        let elapsed = Stopwatch.GetTimestamp() - origin
        let f = Stopwatch.Frequency
        (elapsed / f) * 1_000_000_000L + (elapsed % f) * 1_000_000_000L / f

    interface IClock with
        member c.NowNs = c.NowNs

/// Time constants in ns.
[<RequireQualifiedAccess>]
module Ns =
    [<Literal>]
    let PerMs = 1_000_000L

    [<Literal>]
    let PerSecond = 1_000_000_000L

    /// Milliseconds to ns.
    let ms (n: int64) = n * PerMs

    /// Seconds to ns.
    let seconds (n: int64) = n * PerSecond

    /// ns to whole milliseconds (rounded down, at least 0), for the u32 durations in messages.
    let toMs (ns: int64) = uint32 (max 0L (ns / PerMs))
