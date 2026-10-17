# ClusterProtocol contract

Owner: C1. Status: v1, to be frozen at the contracts-v1 tag. Consumers: the serving pipeline (C2: `QwenStageExecutor`, `lrm coord`, `lrm node`), the Gate 1 cluster suite (I2), the cluster report and dashboard (T1, T2). Code: `src/Lrm.Cluster` (contract files `Protocol*.fs`). Tests: `tests/Lrm.Cluster.Tests` (cpu tier).

After contracts-v1, this text and its vectors change only through a `contract-change` PR, with a CHANGELOG entry and updated vectors.

The cluster runs pure, content-addressed work units on networked nodes that lag, drop and die. Results stay exact anyway: any node of the same execution class can recompute any unit, the first result wins, and replicas that disagree trip a divergence alarm. A chaos run and a fault-free run of one job in one class give the same bits.

## Contents

| Module | What it provides |
| --- | --- |
| `ProtocolFrame.fs` | `MessageKind`, `FrameFlags`, `Frame`, `FrameCodec` (the 80-byte header and SHA-256 trailer), `FrameStream` |
| `ProtocolUnit.fs` | `UnitId`, `BlobId`, `ChainKey`, `WorkUnit` and its id |
| `ProtocolMessages.fs` | The eleven messages, their canonical payloads, `Envelope` and `Message.toFrame` / `tryOfFrame` |
| `Clock.fs` | `IClock`, `VirtualClock`, `SystemClock` |
| `BlobStore.fs` | `IBlobStore`, `MemoryBlobStore`, `DirectoryBlobStore` |
| `Executor.fs` | `IStageExecutor`, `IChainState`, `IWorkload`, `RunDigest`, `DeadlinePolicy`, `DeadlineEstimator` |
| `FakePipeline.fs` | The fake 3-stage pipeline: `FakeJob`, `FakeExecutor`, `FakeWorkload`, `FakeReference` (the oracle) |
| `Actor.fs` | `IActor`, `IActorContext`, the reserved telemetry names (`ClusterNames`) |
| `Coordinator.fs`, `Node.fs` | The two actors |
| `ChaosPolicy.fs` | `ChaosConfig`, `Fault`, `ChaosPolicy` |
| `Simulation.fs` | The deterministic simulator and its trace hash |
| `Transport.fs`, `ChaosProxy.fs`, `Runtime.fs` | InProc, TCP and Chaos transports, the frame-aware TCP proxy, the real-time `ActorHost` |
| `Harness.fs` | Node processes, the local multi-process run, the standalone proxy |

## Frames

Every message travels as one frame: an 80-byte little-endian header, the payload, and a 32-byte trailer.

| Offset | Size | Field | Rule |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | the bytes `4C 52 4D 31` ("LRM1") |
| 4 | 2 | version | 1 |
| 6 | 2 | kind | 1 to 11 (table below) |
| 8 | 4 | flags | bit 0 = Retransmit; every other bit is zero |
| 12 | 4 | reserved | zero |
| 16 | 8 | epoch | the sender's membership epoch |
| 24 | 8 | sender | the sender's member id: 0 is the coordinator, nodes are 1 or more |
| 32 | 8 | seq | the sender's frame counter, increasing per process |
| 40 | 4 | payload length | at most 64 MiB (67,108,864) |
| 44 | 4 | reserved | zero |
| 48 | 32 | unit id | the unit the frame is about, or 32 zero bytes |

- **Trailer.** The last 32 bytes are SHA-256 over the header and the payload. It is an integrity check, not a content address, and never a key in a ContentId store.
- **Strict decoding.** A frame is rejected for a wrong magic or version, an unknown kind or flag, a nonzero reserved word, a payload length over the limit, a total size that differs from 80 + length + 32, or a trailer mismatch. Every single-bit flip of a frame is detected (a test flips each bit of the F1 vector).
- **Streams.** Frames are self-delimiting, so TCP carries them back to back. A malformed frame closes the connection; the protocol retransmits.
- **Retransmit** is informational: it marks a re-sent Assign. Receivers treat every message idempotently anyway.
- **No timestamps.** Frames carry durations in milliseconds (heartbeat and lease), never wall-clock times.

### Message kinds

| Kind | Name | Direction | Header unit id |
| ---: | --- | --- | --- |
| 1 | Hello | node to coordinator | zero |
| 2 | Welcome | coordinator to node | zero |
| 3 | Heartbeat | node to coordinator | zero |
| 4 | Lease | coordinator to node | zero |
| 5 | Membership | coordinator to members | zero |
| 6 | Assign | coordinator to node | the unit's id |
| 7 | Result | node to coordinator | the unit's id |
| 8 | Fetch | node to coordinator | To |
| 9 | Log | coordinator to node | zero |
| 10 | Notice | either | zero |
| 11 | Ready | node to coordinator | Head |

### Payloads

Payloads use the canonical encoding (LrmCore contract): fields in order, little-endian integers, `u32`-prefixed strings, bytes and sequences, 32 raw bytes per id, options as `00` or `01` plus the value. A payload must be read completely; trailing bytes, unknown enum values or a header unit id that disagrees with the payload reject the frame.

| Kind | Payload |
| --- | --- |
| Hello | name (string), incarnation (u64), class (ExecutionClassId), can-host stages (seq of u32), held chains (seq of chain heads) |
| Welcome | heartbeat ms (u32), lease ms (u32), echo seq (u64): the Seq of the Hello answered |
| Heartbeat | ready (bool), active units (seq of: unit id, phase u8) |
| Lease | duration ms (u32), echo seq (u64): the Seq of the heartbeat answered |
| Membership | members (seq of: id u64, name string, class id), roles (seq of: stage u32, nodes seq of u64) |
| Assign | the unit (its canonical encoding), the input (bytes). BlobId of the input must equal the unit's Input, and the header unit id must equal the unit's id |
| Result | the output (bytes), the chain's state digest after the unit (id) |
| Fetch | chain (lane u32, stage u32), From (unit id, zero = from the start), To (unit id) |
| Log | chain (lane u32, stage u32), reset (bool), final (bool), entries (seq of: unit, input bytes, output blob id, state digest). Each entry's input must match its unit's Input |
| Notice | code (u16), value (u64), detail (string) |
| Ready | chain (option of lane u32, stage u32), head (unit id), state digest (id) |

A chain head is lane (u32), stage (u32), head unit id, state digest. Unit phases: 0 Queued, 1 Fetching, 2 Replaying, 3 Positioned. Notice codes: 1 StaleEpoch (value = current epoch), 2 UnknownMember, 3 Shutdown, 4 Leave, 5 WrongClass.

## Work units

A work unit is a pure piece of work: its output is a deterministic function of the unit and of the state its chain reaches after the previous unit. Any node of the unit's execution class can recompute it, and duplicates are harmless.

| # | Field | Encoding | Meaning |
| ---: | --- | --- | --- |
| 1 | Kind | string | `fwd.stage` in Phase 1; 1 to 64 characters of `a-z`, `0-9`, `.`, `_`, starting with a letter |
| 2 | KindVersion | u16 | 1 for fwd.stage v1 |
| 3 | Job | 32 bytes | what is computed: model, weights, configuration, request. Defined by the workload |
| 4 | Class | 32 bytes | the ExecutionClassId that must run the unit |
| 5 | Lane | u32 | the sequence or request |
| 6 | Stage | u32 | the pipeline stage |
| 7 | Step | u32 | the position in the chain, from 0 |
| 8 | Prev | 32 bytes | PrevUnitId: the previous unit of the same job, lane and stage; zero at step 0 |
| 9 | Input | 32 bytes | the BlobId of the input bytes |
| 10 | Args | bytes | kind-specific arguments in their own canonical encoding, at most 64 KiB |

- **UnitId** = `Canonical.hash "lrm.unit/1"` over the encoding above (tag registered in `lrm-core.md`). It is never a raw SHA-256.
- **Hash chain.** A chain is the units of one (job, lane, stage) in step order. For fwd.stage, Prev is zero exactly at step 0 and otherwise the id of step − 1, so a unit id commits to the whole history of its chain: the stage's state (for Qwen, the KV cache of that sequence) is determined by the id of the unit that produced it.
- **Blob ids.** Raw bytes are keyed by `BlobId` = `Canonical.hashBytes "lrm.blob/1" bytes` (the S1 keyspace rule), so blobs and typed ids can share a store. Inputs and outputs are at most 63 MiB (`WorkUnit.MaxData`), so an Assign, a Result or a one-entry Log always fits in one frame.
- **Class.** A unit is assigned only to a node whose executor class has the unit's ExecutionClassId. Bit-identity claims hold only within one class.

## Membership, epochs and leases

The coordinator is member 0 and the single authority (coordinator replication is Phase 2). Nodes talk only to it.

- **Join.** A node sends Hello every heartbeat interval until it gets Welcome. A Hello from a new id adds a member; a Hello from a member with a new incarnation replaces it (a restart, with fresh state); a Hello from a member with the same incarnation is a resync and changes nothing.
- **Epochs.** Every membership change (join, restart, lease expiry, Leave) increments the epoch. The coordinator then refills the stage replicas, broadcasts Membership, and re-sends every pending Assign under the new epoch.
- **Stale epochs are rejected.** Every node frame except Hello must carry the current epoch. Otherwise its content is ignored, and the sender gets Notice StaleEpoch (value = current epoch) and the Membership. A stale heartbeat from a member still counts as proof of life for its lease. A frame from a non-member (except Hello) gets Notice UnknownMember, and the node sends Hello. A node adopts any higher epoch it sees in a coordinator frame and ignores Assign, Log and Membership frames of a lower epoch.
- **Heartbeats and leases.** Nodes heartbeat every 100 ms. Each heartbeat renews the member's lease for 1 s from its receipt and is answered by Lease, which echoes the heartbeat's Seq. The node's own lease runs 1 s from the send time of the last answered heartbeat, so it never outlives the coordinator's view. A member whose lease runs out is removed in a new epoch. A node whose own lease runs out stops counting itself a member and sends Hello again, with the chains it holds.
- **Stage replicas.** Each stage has `Replicas` members (2 by default); every unit of a stage is assigned to all of them. Missing slots are filled from members that can host the stage and have the unit class: fewest roles first, then Ready ones, then the lowest id. Survivors keep their roles.

## Assignment, Ready gating and deadlines

- **Ready.** A node sends Ready with no chain when its executor is loaded, and Ready with a chain when a replay has brought that chain to Head. Heartbeats also carry the ready flag and the phase of every active unit, so a lost Ready is repaired.
- **Deadlines count only after Ready.** An assignment's deadline starts when the node is Ready and is not replaying the unit's chain (from a Fetch for the chain until the chain's Ready, a Result on it, or a Positioned phase). Cold start and replay therefore never cause reissues.
- **The rule** (`DeadlinePolicy.standard`): 30 s until the stage has 8 completion samples, then clamp(4·p95 + 50 ms, 250 ms, 30 s). A sample is the time from the winning assignment's deadline start to the accepted result. p95 is the nearest-rank 95th percentile (the ⌈0.95 n⌉-th smallest) of the stage's last 64 samples.
- **A missed deadline** re-sends the Assign (the Assign or the Result may be lost) and re-arms for another period. An assignment is live while its deadline has not started, or has started and not yet been missed.
- **Reissue.** When a pending unit has no live assignment left, the coordinator recomputes it elsewhere: it assigns the unit to the Ready, eligible member not yet assigned with the fewest roles (then the lowest id), which joins the stage's replicas. If nobody is free, it waits and emits `cluster.reissue.unavailable` once.
- **Demotion.** A replica whose assignment had missed its deadline when another replica's result was accepted gains a lag count; any other outcome resets it. While a stage has more than `Replicas` replicas, those with a lag count of 3 or more are dropped from it, the worst first.
- **Retransmission.** An assignment that the node's heartbeat does not list is re-sent after 500 ms. A node asked for a unit it already applied as its chain head answers with the cached result.

## Results, divergence and the unit log

- **First result wins.** The first Result for a pending unit is accepted: its output enters the blob store and the unit log, the workload plans the units it unlocks, and the unit is done.
- **Duplicates.** A later Result for an accepted unit is compared by output blob id and state digest. Equal: it confirms the unit (`cluster.duplicate`). Different: `cluster.divergence`, recorded in the outcome.
- **Replica digests.** A chain Ready after a replay carries the replayed state digest. If it differs from the accepted digest at the same head, that is a divergence too.
- **The tripwire.** With `HaltOnDivergence` (the default), the first divergence ends the run with status Diverged.
- **Replay of the unit log before execution.** A node that is asked to run a unit whose Prev it does not hold sends Fetch (From = its head, To = the unit's Prev). The coordinator answers with Log batches of at most 16 MiB of inputs: the accepted units after From up to To, with their inputs, outputs and digests. If From is not on the chain, the first batch says Reset and the node replays from the start. The node applies the entries in order, reports the chain Ready, then executes. Fetch is retried every 1 s and on every epoch change until the log arrives.
- **End of run.** When the workload is complete and nothing is pending, the coordinator waits until every member that was assigned a chain's final unit has confirmed it, or 2 s, then sends Shutdown to every member. Afterwards it answers every frame with Shutdown, so a node that lost the notice stops at its next heartbeat.

### Digests

- **RunDigest** = `Canonical.hash "lrm.cluster.run/1"` over the job id (32 bytes), then a sequence of the accepted units sorted by (lane, step, stage), each written as lane, step, stage (u32) and unit id, output blob id, state digest (32 bytes each).
- **ReplicaDigest** = `Canonical.hash "lrm.cluster.replicas/1"` over a sequence of chain heads sorted by (lane, stage), each written as lane, stage (u32), head unit id and state digest. A node's report carries the replica digest of the chains it holds; the outcome carries the digest of every chain's final head.
- Both hold only content, never times, nodes or schedules.

| Tag | Used for |
| --- | --- |
| `lrm.unit/1` | UnitId |
| `lrm.cluster.run/1` | RunDigest |
| `lrm.cluster.replicas/1` | ReplicaDigest |
| `lrm.cluster.trace/1` | The simulator's trace hash (not a content address) |
| `lrm.cluster.chaos.link/<src>/<dst>`, `lrm.cluster.chaos.faults`, `lrm.cluster.sim.coldstart`, `lrm.cluster.sim.cost/<node>/<incarnation>` | DetRng stream names (`DetRng.streamOf`) |
| `lrm.fake.job/1`, `lrm.fake.prompt/1`, `lrm.fake.init/1`, `lrm.fake.out/1`, `lrm.fake.state/1`, `lrm.fake.expand/1` | The fake pipeline |

## Executor interface

C2's `QwenStageExecutor` implements this; the fake executor is the reference implementation.

```fsharp
type IChainState =
    inherit IDisposable
    abstract Digest: ContentId                              // after the last Apply, or the initial digest
    abstract Apply: work: WorkUnit * input: byte[] -> byte[] // returns the unit's output

type IStageExecutor =
    abstract Class: ExecutionClass
    abstract CanHost: uint32 list
    abstract Open: job: ContentId * chain: ChainKey -> IChainState

type IWorkload =                                            // the coordinator's plan
    abstract Job: ContentId
    abstract Class: ExecutionClassId
    abstract Stages: uint32
    abstract Start: unit -> PlannedUnit list
    abstract Accept: work: WorkUnit * output: byte[] -> PlannedUnit list  // once per unit, with the first result
    abstract IsComplete: bool
```

- A node calls `Apply` only with the unit whose Prev is the chain's head, so a chain state always follows the hash chain; replay is `Apply` over logged units.
- `Digest` must be a deterministic function of the units applied within one class, and it must change whenever an output changes: replica digests are compared to detect divergence.
- `Apply` runs on a worker thread, never the actor's, so a long unit does not stop heartbeats.
- A workload must be deterministic: the units it plans may depend only on accepted outputs.

## Chaos

`ChaosPolicy(seed, config, faults)` decides every injected fault from DetRng.

- **Per-frame fate.** Each directed link (src, dst) draws from `DetRng(seed, DetRng.streamOf "lrm.cluster.chaos.link/<src>/<dst>")`, in the order of the frames on the link, exactly five decisions per frame: drop, duplicate, reorder (each `NextBelow 1000 < permille`), then two delays uniform in [MinDelay, MaxDelay]. A reordered frame gets ReorderDelay extra; a duplicate's second copy arrives one more delay later.
- **Faults.** `Kill(node, t)`, `Restart(node, t)` (a new incarnation with no state), `Partition(node, from, until)` (every frame to or from the node is lost), `Lag(node, from, until, factor)` (executor cost times factor), `Corrupt(node, stage, step)` (simulator only: wrong outputs, honest digests) and `Sever(node, t)` (ChaosProxy only: the TCP connection is cut once).
- **Random schedules.** `ChaosPolicy.RandomFaults(seed, nodes, horizon)` draws from `lrm.cluster.chaos.faults`: up to two kills of distinct nodes in [20%, 80%] of the horizon, each restarted after 0.5 to 3 s with probability 1/2; up to two partitions of 0.3 to 3 s; up to two lag windows of 0.5 to 4 s with a factor of 5 to 50.
- **Configs.** `ChaosConfig.standard` (simulator): 2% drops, 2% duplicates, 5% reordered by 40 ms, 1 to 15 ms of delay. `ChaosConfig.proxy` (real processes): 1%, 1%, 2% by 10 ms, 0 to 3 ms.

## Simulation

`Simulation.run` executes the real coordinator and node actors on a `VirtualClock`. Events (deliveries, timers, executor completions, kills, restarts) are processed in (virtual time, scheduling order); every frame is encoded, passed through the policy and decoded. Nothing reads the wall clock, and every coordinator decision iterates ordered collections, so a configuration always produces the same schedule.

- **Trace hash** = SHA-256(`lrm.cluster.trace/1` || `00` || records): one 57-byte record per processed event, `u8` kind (1 deliver, 2 timer, 3 work done, 4 kill, 5 restart), `i64` virtual time, `u64` actor, `u64` other (the sender, timer tag or work token), and 32 bytes (the frame trailer for a delivery, zero otherwise). It includes virtual times on purpose; it is a test fingerprint, never a content address, and no wall-clock time enters it.
- **Executor cost** is units × the stage's cost + 0 to CostJitter (from `lrm.cluster.sim.cost/<node>/<incarnation>`), times the active lag factor.
- **`Simulation.standard seed`**: the standard fake job on 7 nodes (3 stages × 2 replicas plus a spare), cold starts of 300 to 800 ms, `ChaosConfig.standard`, and `RandomFaults` over the expected run time.

## Transports

All transports are best effort: frames can be lost, and the protocol retransmits.

- **InProc** (`InProcHub`): endpoints of one process; frames are decoded on delivery, exactly as over TCP.
- **TCP**: `TcpServerTransport` (the coordinator, listening on 127.0.0.1 in tests) routes a frame to member `dest` over the connection that last delivered a frame from it; `TcpClientTransport` (a node) reconnects every 100 ms. Each connection has a bounded send queue (4,096 frames; full means dropped).
- **Chaos** (`ChaosTransport`): a decorator that applies a policy to outgoing frames and drops everything while its endpoint is partitioned.
- **ChaosProxy**: a TCP proxy between nodes and the coordinator. It parses whole frames in both directions, learns the node id from the first frame, applies the fate on link (node, 0) or (0, node), drops frames of a partitioned or killed node, and cuts connections at Sever and Kill times. Frames are forwarded byte for byte, so trailers stay valid. `lrm chaos-proxy` runs it standalone.
- **ActorHost** runs one actor on a dedicated thread, taking frames, timers and executor results one at a time; executor work runs on the thread pool.

## The fake pipeline

The fake job `{ Seed; Lanes; Steps; Stages; PromptBytes; HiddenBytes }` mimics the Qwen pipeline with SHA-256 only, so its digests are the same on every host and are known answers.

- **Job id** = `Canonical.hash "lrm.fake.job/1"` (Seed u64, then Lanes, Steps, Stages, PromptBytes, HiddenBytes as u32).
- **Class**: `{ cpu-f32; "fake"; "lrm.fake/1"; None; "any" }`, id `86d7e37b8867c60d8b79289e75bd715066544d2036790ebac258eb909107ee04`.
- **expand(seed, n)**: the blocks `Canonical.hash "lrm.fake.expand/1"` (seed, u32 block) for blocks 0, 1, …, concatenated and cut to n bytes.
- **Prompt** of a lane = expand(`Canonical.hash "lrm.fake.prompt/1"` (job id, u32 lane), PromptBytes).
- **Units** are fwd.stage v1, with Args = the output size as one u32: HiddenBytes, or 4 at the last stage. Stage 0 at step 0 reads the prompt; stage s+1 reads stage s's output; stage 0 at step t+1 reads the last stage's output at step t (the "token", read as a little-endian u32).
- **A chain's initial digest** = `Canonical.hash "lrm.fake.init/1"` (job id, u32 lane, u32 stage).
- **One step** with digest d: h = `Canonical.hash "lrm.fake.out/1"` (job id, u32 lane, u32 stage, u32 step, d, input as bytes); output = expand(h, output size); new digest = `Canonical.hash "lrm.fake.state/1"` (d, u32 step, input blob id, output blob id).
- **`FakeReference.run`** runs every unit in plan order on one honest executor. Its RunDigest is the oracle that every cluster run of the job must equal.

## Telemetry

The cluster reports through the Lrm.Core facade with the reserved `cluster.*` names. The coordinator's records carry `component = cluster.coordinator`, `node = coord`; a node's carry `component = cluster.node` and its name. In the simulator, records are timed on virtual time.

| Name | Kind | Unit | Attributes | Meaning |
| --- | --- | --- | --- | --- |
| `cluster.epoch` | gauge | | reason, members | the epoch after a membership change |
| `cluster.member.join` | event | | node, incarnation, reason (join, restart) | a node joined |
| `cluster.member.leave` | event | | node, reason (lease, restart, leave) | a node left the membership |
| `cluster.lease.lost` | event | | | a node's own lease ran out |
| `cluster.ready` | event | | node, scope (node, chain), lane, stage | a Ready |
| `cluster.assign` | counter | count | stage, node, retransmit | an Assign sent |
| `cluster.unit.done` | event | | lane, stage, step, node, start_ns, end_ns, latency_ns | a unit accepted: the stage Gantt chart |
| `cluster.unit.latency` | histogram | ns | stage, node | deadline start to accepted result |
| `cluster.lag` | histogram | ns | stage, node | how far a duplicate trailed the accepted result |
| `cluster.deadline` | gauge | ns | stage, samples | the stage's deadline, when it changes |
| `cluster.deadline.miss` | counter | count | stage, node | an assignment missed its deadline |
| `cluster.reissue` | counter | count | lane, stage, step, node | lagging work recomputed elsewhere |
| `cluster.reissue.unavailable` | event | | lane, stage, step | no node was free to take it |
| `cluster.duplicate` | counter | count | stage, node | a matching duplicate |
| `cluster.divergence` | event | | unit, lane, stage, step, first_node, other_node, source (result, replay) | replicas disagree |
| `cluster.replay.served` | counter | count | lane, stage, node | log entries sent for replay |
| `cluster.replay.units` | counter | count | lane, stage | units a node replayed |
| `cluster.demote` | event | | stage, node | a lagging replica left a stage |
| `cluster.frame.rejected` | counter | count | reason, kind, node | a rejected frame |
| `cluster.chaos` | counter | count | fault (drop, duplicate, kill, restart), node | an injected fault (simulator) |
| `cluster.run` | event | | status, units, epoch, reissues, duplicates, divergences, replays, run_digest, end_ns | the run finished |
| `cluster.bytes_sent` | meter | B | kind | coordinator payload bytes sent |

## Limits and design choices

- **One coordinator, star topology.** Nodes talk only to the coordinator, which holds the unit log and the blob store. Losing it ends the run; replay recovers every worker's state, and coordinator replication (3 replicas, majority leases) is Phase 2.
- **Active replication.** Every replica of a stage executes every unit of that stage, so a failover needs no catch-up while one replica survives; replay rebuilds a replica that has none.
- **Pull-based replay.** The node that lacks a chain asks for exactly the missing suffix; the coordinator does not track node state beyond replays in progress.
- **One job per coordinator run.** Fetch, Log and chain heads name chains by (lane, stage) within that job.
- **Unauthenticated localhost.** The sender field is self-declared. Phase 1 runs on trusted links.

## Verification

All in the cpu tier, so CI runs them on every PR:

- the frame, unit and fake-pipeline vectors below; round trips of every message kind; rejection of every malformed header, payload and single-bit flip; frames reassembled from one-byte reads;
- the coordinator frame by frame: epochs, StaleEpoch and UnknownMember, lease expiry, deadlines only after Ready (30 s, then reissue), first result wins, the divergence tripwire, replay divergence, Fetch, late Shutdown;
- the simulator: **200 chaos seeds bit-identical** to `FakeReference` with 2 replicas per stage; the **trace hash stable over 100 runs** of one seed; **loss of both replicas** of a stage recovered by replay (with and without a spare); kill and restart; the tripwire on a corrupt replica; 40 s cold starts with no reissue; lagging work reissued; a partition expelled and rejoined; 20% frame loss; telemetry changing nothing; every reserved name emitted on virtual time;
- real time: an in-process cluster with chaos transports, TCP with an 8 MiB payload, the proxy byte for byte and Sever, a TCP cluster through the proxy, and **3 node processes behind a ChaosProxy with one killed**. These wait at most 60 s and assert digests only.

### Vectors

`tests/Lrm.Cluster.Tests/vectors/cluster_vectors.py` re-implements the encodings from this text with Python's `hashlib` and prints every value below; the cpu tier checks the F# code against them.

**F1, a Lease frame:** kind 4, flags 0, epoch 7, sender 0, seq 42, no unit, payload DurationMs 1000 and LeaseEcho 41 (`e8030000 2900000000000000`). 124 bytes:

```
4c524d31 0100 0400 00000000 00000000 0700000000000000 0000000000000000 2a00000000000000 0c000000 00000000
0000000000000000000000000000000000000000000000000000000000000000
e80300002900000000000000
ca9a4fb535a0703a6464a32b4a939f809c9251f439361d91d3babd83e962f74a
```

**U1, a fwd.stage unit:** job = the bytes `01 02 … 20`, the fake class, lane 1, stage 2, step 0, no Prev, Input = BlobId("abc") = `43dc9863…1a2c`, Args = u32 4. Payload:

```
09000000 6677642e7374616765 0100
0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20
86d7e37b8867c60d8b79289e75bd715066544d2036790ebac258eb909107ee04
01000000 02000000 00000000
0000000000000000000000000000000000000000000000000000000000000000
43dc986338811e8f6ba04bbba1545fdb098e2e138a1392cb276961ebfe901a2c
04000000 04000000
```

UnitId = `b40822f7972990ed40b8c5d94d2bacb84df0586d900b5c2b81fc7a296bea44b3`.

**U2, the next unit of U1's chain:** step 1, Prev = U1's id, Input = BlobId of no bytes (`ad6b5a67…42c1`), everything else as U1. UnitId = `27e7e84f456d780f4e22a0504941bf186cf0746a3110e35b00701bf9759dbcc9`.

**F2, an Assign of U1** with input "abc": kind 6, flags 1 (Retransmit), epoch 3, sender 0, seq 9, header unit id = U1's id, payload = U1's payload then `03000000 616263`. Trailer `d1d88a5a72ac401416cf63a4352c18a6403e16d27bb7851868a929009077b5e9`.

**Fake pipeline.**

| Job | Job id | RunDigest | ReplicaDigest | Units |
| --- | --- | --- | --- | ---: |
| small: seed 1, 2 lanes, 3 steps, 3 stages, 16-byte prompts, 64-byte hidden | `0d93494f91cde7827594a012c9ff0461e53a2c95a37d8f0fb96c5152fe568a77` | `2f358424718b1c4c3b4bb1efb4cc4d7dcde1bb54acff4695e8f4b7f77960c14c` | `d55283da804392ea3875f0686c394ff211e87d70f2234b611a0134f0cd1dbf2a` | 18 |
| standard (`FakeJob.standard`): seed 1, 3 lanes, 8 steps, 3 stages, 64, 256 | `77b04d909bce35e504e0249e2fa172f5d50ff8d6421f476e09cc3370395d578f` | `507fb323a4f9e5efda3e7a0ce78bd9ff1c1ed07d6a95b8520850357bf90eec16` | `b0a5410211781844069c973e94a169274ba43db3f50ba3b44a6976b5068f344f` | 72 |

The small job's tokens are `[3360429926, 1564617581, 3831387546]` for lane 0 and `[3456249224, 414529344, 200954126]` for lane 1.

## Commands

| Command | What it does |
| --- | --- |
| `lrm chaos sim [--seeds 200] [--from 1]` | runs `Simulation.standard` for each seed and compares every RunDigest with the reference |
| `lrm chaos trace [--seed 7] [--runs 100]` | runs one seed many times and prints the distinct trace hashes |
| `lrm chaos local [--seed 1] [--nodes 3] [--kill 2\|none] [--kill-after N] [--cost-ms 2] [--timeout-ms 60000] [chaos options]` | a coordinator and a ChaosProxy here, node processes behind them, one killed part-way |
| `lrm chaos node --connect HOST:PORT --id N --name NAME [--cost-ms C] [--stages 0,1,2]` | one fake-executor node process |
| `lrm chaos-proxy --listen PORT --target HOST:PORT [--seed N] [--chaos none\|lan\|proxy\|standard] [--drop P] [--duplicate P] [--reorder P] [--min-delay-ms] [--max-delay-ms] [--reorder-delay-ms] [--partition node:fromMs:untilMs] [--sever node:ms] [--kill node:ms] [--restart node:ms] [--duration-ms N]` | the standalone proxy |
