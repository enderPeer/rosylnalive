namespace Lrm.Cluster

open System.Collections.Generic
open Lrm.Core

/// Per-frame network faults. Probabilities are in permille (0 to 1000).
type ChaosConfig =
    {
        DropPermille: int
        DuplicatePermille: int
        /// Frames that get ReorderDelayNs extra, so later frames overtake them.
        ReorderPermille: int
        /// Every frame is delayed uniformly in [MinDelayNs, MaxDelayNs].
        MinDelayNs: int64
        MaxDelayNs: int64
        ReorderDelayNs: int64
    }

[<RequireQualifiedAccess>]
module ChaosConfig =
    /// A perfect network: no loss, no delay.
    let none =
        {
            DropPermille = 0
            DuplicatePermille = 0
            ReorderPermille = 0
            MinDelayNs = 0L
            MaxDelayNs = 0L
            ReorderDelayNs = 0L
        }

    /// A fast, clean network: 0.2 to 2 ms, nothing lost.
    let lan =
        { none with
            MinDelayNs = Ns.ms 1L / 5L
            MaxDelayNs = Ns.ms 2L
        }

    /// The simulator's standard chaos: 2% drops, 2% duplicates, 5% reordered by 40 ms, 1 to 15 ms of delay.
    let standard =
        {
            DropPermille = 20
            DuplicatePermille = 20
            ReorderPermille = 50
            MinDelayNs = Ns.ms 1L
            MaxDelayNs = Ns.ms 15L
            ReorderDelayNs = Ns.ms 40L
        }

    /// Mild chaos for real processes: 1% drops, 1% duplicates, 2% reordered by 10 ms, up to 3 ms of delay.
    let proxy =
        {
            DropPermille = 10
            DuplicatePermille = 10
            ReorderPermille = 20
            MinDelayNs = 0L
            MaxDelayNs = Ns.ms 3L
            ReorderDelayNs = Ns.ms 10L
        }

    /// The config when its values are in range.
    let validate (c: ChaosConfig) : Result<ChaosConfig, string> =
        let permille v = v >= 0 && v <= 1000

        if
            not (
                permille c.DropPermille
                && permille c.DuplicatePermille
                && permille c.ReorderPermille
            )
        then
            Error "probabilities are permille, 0 to 1000"
        elif c.MinDelayNs < 0L || c.MaxDelayNs < c.MinDelayNs || c.ReorderDelayNs < 0L then
            Error "delays must satisfy 0 <= MinDelayNs <= MaxDelayNs and ReorderDelayNs >= 0"
        else
            Ok c

/// A scripted fault at a point in time (ns from the start of the run).
[<RequireQualifiedAccess>]
type Fault =
    /// The node's process dies.
    | Kill of node: uint64 * atNs: int64
    /// A dead node starts again as a new incarnation with no state.
    | Restart of node: uint64 * atNs: int64
    /// Every frame to or from the node is lost in [fromNs, untilNs).
    | Partition of node: uint64 * fromNs: int64 * untilNs: int64
    /// The node's executor runs `factor` times slower in [fromNs, untilNs).
    | Lag of node: uint64 * fromNs: int64 * untilNs: int64 * factor: int
    /// The node's executor gets the output of (stage, step) wrong in every lane, and its state digest follows the
    /// wrong output as the digest rule requires (simulator only; tripwire tests).
    | Corrupt of node: uint64 * stage: uint32 * step: uint32
    /// Like Corrupt, but the digest stays honest: an executor that breaks the digest rule (simulator only).
    | CorruptOutput of node: uint64 * stage: uint32 * step: uint32
    /// The node's TCP connection is cut at atNs (ChaosProxy only); the node reconnects.
    | Sever of node: uint64 * atNs: int64

/// What happens to one frame: the delays of the copies that get through. Empty means the frame is lost.
type FrameFate = { Copies: int64 list }

/// Chaos decisions (ClusterProtocol contract: ChaosPolicy is seeded by DetRng). Each directed link (src, dst) draws
/// from its own DetRng stream, DetRng(seed, DetRng.streamOf "lrm.cluster.chaos.link/<src>/<dst>"), in the order of the
/// frames on that link, so the same frames on the same links always meet the same fate. Every frame draws exactly
/// five decisions: drop, duplicate, reorder, then two delays. Not thread-safe; callers lock.
[<Sealed>]
type ChaosPolicy(seed: uint64, config: ChaosConfig, faults: Fault list) =
    do
        match ChaosConfig.validate config with
        | Ok _ -> ()
        | Error e -> invalidArg "config" e

    let links = Dictionary<struct (uint64 * uint64), DetRng>()

    let rngFor (src: uint64) (dst: uint64) =
        let key = struct (src, dst)

        match links.TryGetValue key with
        | true, r -> r
        | _ ->
            let r =
                DetRng(seed, DetRng.streamOf ("lrm.cluster.chaos.link/" + Invariant.u64 src + "/" + Invariant.u64 dst))

            links[key] <- r
            r

    let delay (r: DetRng) =
        let span = config.MaxDelayNs - config.MinDelayNs
        let draw = r.NextU64()

        if span <= 0L then
            config.MinDelayNs
        else
            config.MinDelayNs + int64 (draw % uint64 (span + 1L))

    member _.Seed = seed
    member _.Config = config
    member _.Faults = faults

    /// True when the node is cut off at `now` by a Partition fault.
    member _.Partitioned(node: uint64, now: int64) =
        faults
        |> List.exists (function
            | Fault.Partition(n, a, b) -> n = node && a <= now && now < b
            | _ -> false)

    /// The executor slowdown of the node at `now` (1 when no Lag fault is active).
    member _.LagFactor(node: uint64, now: int64) =
        faults
        |> List.fold
            (fun f fault ->
                match fault with
                | Fault.Lag(n, a, b, factor) when n = node && a <= now && now < b -> max f factor
                | _ -> f)
            1

    /// How the node's fake executor computes this unit: Corrupt and CorruptOutput faults make it wrong (Corrupt wins
    /// when both name the unit).
    member _.FakeFault(node: uint64, u: WorkUnit) =
        let hits (n: uint64) (stage: uint32) (step: uint32) =
            n = node && stage = u.Stage && step = u.Step

        faults
        |> List.fold
            (fun f fault ->
                match fault with
                | Fault.Corrupt(n, stage, step) when hits n stage step -> FakeFault.WrongOutput
                | Fault.CorruptOutput(n, stage, step) when hits n stage step && f = FakeFault.Honest ->
                    FakeFault.WrongOutputHonestDigest
                | _ -> f)
            FakeFault.Honest

    /// The fate of the next frame on link (src, dst) sent at `now`. Partitions drop it without drawing.
    member p.Decide(src: uint64, dst: uint64, now: int64) : FrameFate =
        if p.Partitioned(src, now) || p.Partitioned(dst, now) then
            { Copies = [] }
        else
            let r = rngFor src dst
            let drop = r.NextBelow 1000u < uint32 config.DropPermille
            let duplicate = r.NextBelow 1000u < uint32 config.DuplicatePermille
            let reorder = r.NextBelow 1000u < uint32 config.ReorderPermille
            let d1 = delay r + (if reorder then config.ReorderDelayNs else 0L)
            let d2 = delay r

            if drop then { Copies = [] }
            elif duplicate then { Copies = [ d1; d1 + d2 ] }
            else { Copies = [ d1 ] }

    /// A random fault schedule for one seed, from DetRng(seed, DetRng.streamOf "lrm.cluster.chaos.faults"): up to two
    /// kills of distinct nodes in [20%, 80%] of the horizon (each restarted after 0.5 to 3 s with probability 1/2), up
    /// to two partitions of 0.3 to 3 s starting in [10%, 80%], and up to two lag windows of 0.5 to 4 s with a factor of
    /// 5 to 50.
    static member RandomFaults(seed: uint64, nodes: uint64 list, horizonNs: int64) : Fault list =
        if nodes.IsEmpty then
            []
        else
            let r = DetRng(seed, DetRng.streamOf "lrm.cluster.chaos.faults")

            let pick () =
                nodes[int (r.NextBelow(uint32 nodes.Length))]

            let at (lo: int) (hi: int) =
                horizonNs * int64 (lo + int (r.NextBelow(uint32 (hi - lo + 1)))) / 100L

            let ms (lo: int) (hi: int) =
                Ns.ms (int64 (lo + int (r.NextBelow(uint32 (hi - lo + 1)))))

            let faults = ResizeArray<Fault>()
            let killed = HashSet<uint64>()

            for _ in 1 .. int (r.NextBelow 3u) do
                let n = pick ()
                let t = at 20 80
                let restart = r.NextBelow 2u = 1u
                let after = ms 500 3000

                if killed.Add n then
                    faults.Add(Fault.Kill(n, t))

                    if restart then
                        faults.Add(Fault.Restart(n, t + after))

            for _ in 1 .. int (r.NextBelow 3u) do
                let n = pick ()
                let t = at 10 80
                let d = ms 300 3000
                faults.Add(Fault.Partition(n, t, t + d))

            for _ in 1 .. int (r.NextBelow 3u) do
                let n = pick ()
                let t = at 10 80
                let d = ms 500 4000
                let factor = 5 + int (r.NextBelow 46u)
                faults.Add(Fault.Lag(n, t, t + d, factor))

            List.ofSeq faults
