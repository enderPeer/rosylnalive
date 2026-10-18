namespace Lrm.Cluster

open Lrm.Core

/// How much executor work a node asks its host to run: `Units` units of stage `Stage`. The simulator turns it into
/// virtual time; a real host ignores it.
[<Struct>]
type WorkCost = { CostStage: uint32; CostUnits: int }

/// What an actor (the coordinator or a node) can do. The same actor code runs in the deterministic simulator and in a
/// real process: the context supplies time, sending, timers and executor work. Every call comes from the actor's own
/// single thread of events.
type IActorContext =
    /// The current time in ns on the host's clock.
    abstract Now: int64
    /// Sends a frame to member `dest` (0 is the coordinator). Best effort: frames can be lost, delayed, duplicated or
    /// reordered, and the protocol tolerates all four.
    abstract Send: dest: uint64 * frame: Frame -> unit
    /// Calls OnTimer with `tag` once the clock reaches `dueNs`.
    abstract SetTimer: dueNs: int64 * tag: uint64 -> unit
    /// Runs `work` away from the actor's thread and then calls OnWorkDone with `token` and its result (or the
    /// exception it raised). At most one work item per actor runs at a time.
    abstract Execute: cost: WorkCost * work: (unit -> obj) * token: uint64 -> unit

/// A single-threaded event handler.
type IActor =
    /// The member id: 0 for the coordinator.
    abstract Id: uint64
    abstract Start: ctx: IActorContext -> unit
    abstract OnFrame: ctx: IActorContext * frame: Frame -> unit
    abstract OnTimer: ctx: IActorContext * tag: uint64 -> unit
    abstract OnWorkDone: ctx: IActorContext * token: uint64 * result: obj -> unit

/// The cluster's reserved telemetry names (TelemetryFacade contract: `cluster.*` belongs to C1 and C2). Every name
/// and its attributes are listed in docs/contracts/cluster.md, "Telemetry".
[<RequireQualifiedAccess>]
module ClusterNames =
    /// Gauge: the coordinator's epoch after a membership change (attrs reason, members).
    [<Literal>]
    let Epoch = "cluster.epoch"

    /// Event: a node joined (attrs node, incarnation, reason).
    [<Literal>]
    let Join = "cluster.member.join"

    /// Event: a node left the membership (attrs node, reason: lease, restart, leave).
    [<Literal>]
    let Leave = "cluster.member.leave"

    /// Event: a node's lease ran out on the node's side (attrs node).
    [<Literal>]
    let LeaseLost = "cluster.lease.lost"

    /// Event: a node reported Ready (attrs node, scope, lane, stage).
    [<Literal>]
    let Ready = "cluster.ready"

    /// Counter: Assign frames sent (attrs stage, node, retransmit).
    [<Literal>]
    let Assign = "cluster.assign"

    /// Event: a unit was accepted (attrs lane, stage, step, node, start_ns, end_ns, latency_ns): the stage Gantt.
    [<Literal>]
    let UnitDone = "cluster.unit.done"

    /// Histogram: ns from deadline start to the accepted result (attrs stage, node).
    [<Literal>]
    let UnitLatency = "cluster.unit.latency"

    /// Histogram: ns by which a duplicate result trailed the accepted one (attrs stage, node).
    [<Literal>]
    let Lag = "cluster.lag"

    /// Gauge: the stage's current deadline in ns, whenever it changes (attrs stage, samples).
    [<Literal>]
    let Deadline = "cluster.deadline"

    /// Counter: an assignment missed its deadline (attrs stage, node).
    [<Literal>]
    let DeadlineMiss = "cluster.deadline.miss"

    /// Counter: lagging work recomputed elsewhere (attrs lane, stage, step, node).
    [<Literal>]
    let Reissue = "cluster.reissue"

    /// Event: no node was free to take lagging work (attrs lane, stage, step).
    [<Literal>]
    let ReissueUnavailable = "cluster.reissue.unavailable"

    /// Counter: a duplicate result that matched the accepted one (attrs stage, node).
    [<Literal>]
    let Duplicate = "cluster.duplicate"

    /// Event: replicas disagree (attrs unit, lane, stage, step, first_node, other_node, source).
    [<Literal>]
    let Divergence = "cluster.divergence"

    /// Counter: log entries the coordinator sent for replay (attrs lane, stage, node).
    [<Literal>]
    let ReplayServed = "cluster.replay.served"

    /// Counter: units a node replayed (attrs lane, stage, node).
    [<Literal>]
    let ReplayUnits = "cluster.replay.units"

    /// Event: a lagging replica was dropped from a stage (attrs stage, node).
    [<Literal>]
    let Demote = "cluster.demote"

    /// Counter: frames the receiver rejected (attrs reason, kind).
    [<Literal>]
    let FrameRejected = "cluster.frame.rejected"

    /// Counter: faults injected by chaos (attrs fault, node).
    [<Literal>]
    let Chaos = "cluster.chaos"

    /// Event: the run finished (attrs status, units, epoch, reissues, duplicates, divergences, run_digest).
    [<Literal>]
    let Run = "cluster.run"

    /// Meter: payload bytes sent (attrs kind).
    [<Literal>]
    let BytesSent = "cluster.bytes_sent"
