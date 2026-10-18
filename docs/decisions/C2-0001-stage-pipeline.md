# ADR C2-0001: Stage pipeline digests, payloads and rebuild

- **Status:** proposed, 2026-10-18 (with PR #32).
- **Owner:** C2.
- **Scope:** StagePipeline v1 (`docs/contracts/stage-pipeline.md`); binds C2b (GPU executors) and I2 (Gate 1 cluster suite).

## Context

C1's cluster core runs pure, hash-chained `fwd.stage` units and leaves three things to the serving layer: what a chain's state digest commits to, what crosses the network, and how a node that lacks a chain rebuilds it. The plan asks for chaos and failover on the tiny model that are bit-identical to the single-process CpuOps run, logits and greedy ids included, and allows a fast KV rebuild only if it equals exact replay bit for bit. C1's node replays a chain by calling `IChainState.Apply` on each logged unit; it has no other hook. C1's node also opens and disposes chain states on its actor thread while a unit runs on a worker, and CpuOps (like a GPU's command lists) serves one caller at a time.

## Decision

1. **The digest commits to the cache.** A chain's digest after a unit is `Canonical.hash "lrm.serving.state/1"` over the previous digest, the step, the input and output blob ids, and the digest of the KV-cache rows the unit wrote (`lrm.serving.kvrows/1`, read back with `ReadRows`, those rows only). This keeps C1's rule (it changes whenever an output changes and stays changed) and also commits to every valid cache row, so a replica whose cache differs is caught at its next chain Ready even before an output differs.
2. **The head ships its token and its logits.** The head stage's output is the token (the node's own `IForwardOps.Argmax` of the last row, `0xFFFFFFFF` for a NaN, as `Greedy.run` picks it) followed by the last row's logits in the class's dtype. The coordinator plans the next step from the token; the logits make the run digest, and the logits digest, a bit-for-bit comparison with the single-process run.
3. **Exact replay is the rebuild.** A node rebuilds a chain only through C1's replay, which recomputes every logged unit with `Apply` and checks each output and digest. The fast rebuild (`IFastRebuild.Rebuild`: few forwards of many rows, the head with `HeadRows.All`) is a chain-state API behind `StageExecutorOptions.FastRebuild`, off by default, proven equal to replay (every output, every digest, `KvCache.digest`) on cpu-f32 and cpu-f64. C1's node does not call it; a GPU class enables it only after the same proof.
4. **One ops lock per executor.** `Open` touches no ops (a cache is created at a chain's first unit); every use of the ops holds one lock; a state disposed during another chain's forward is released at the next use of the ops, so the actor thread never waits for a forward.
5. **Cluster settings.** A two-node split runs one replica per stage (C1: with two, every stage needs two eligible hosts); failover runs two. The drain may last 10 s, because a verifier may replay a chain; it ends as soon as every final unit is settled.
6. **Where CI runs what.** The 50 proxy seeds run node actors in the test process behind C1's ChaosProxy on real TCP (each seed with its own coordinator, proxy and nodes), four at a time, so the gate takes about 40 s. Four runs use `node` and `chaos-proxy` processes, one also a `coord run` process. `lrm coord local --seeds 1-50` runs all 50 with processes.
7. **The seeded partitions avoid one C1 drain case** (below), which C1 is asked to fix.

## Evidence

Measured on the development machine (RTX 4080 host, 32 threads, Windows 11, .NET SDK 10.0.402, runtime 10.0.13), at the PR's head:

- `pwsh tools/test.ps1 -Tier cpu -Project Lrm.Serving.Tests`: 39 tests passed in 68 s, among them the fast rebuild against replay on both CPU classes (one forward per chain, and chunks of 1, 7 and 16 rows), the plan-order pipeline against the single-process run (every head output equal, element for element), and 50 of 50 proxy seeds bit-identical (run digest `e3ce3176…2844`, logits digest `a5c08eb2…79bc` on cpu-f32).
- `pwsh tools/test.ps1 -Tier model -Project Lrm.Serving.Tests`: Qwen3-0.6B split at layer 14 on two node actors with the stage-1 node killed and restarted, ids `34208 916 279`, bit-identical to the single-process run, 5 s.
- `lrm coord local --seeds 1-50 --parallel 2` (node and proxy processes): the PR body gives the count.
- **The drain case.** Before the partition rule, sweeps ended Unverified twice (`replicated3` seeds 5 and 32), each time with every digest equal to the oracles. The trace (`cluster.*` records of the coordinator, and `Describe()` of the coordinator and nodes) showed: n2 killed for good; n3 partitioned while it owed the final units, expelled at lease expiry and rejoined; then `pickVerifier` never engaged n3, because it takes the union of `Expected`, `Late` and `Verifiers` by member id across memberships, while `unconfirmedOf` drops the old membership's entries. The finals kept one confirmation (n1) until the drain timed out.

## Consequences

- **C2b** registers the GPU executors with `QwenStageExecutor.register`, proves the fast rebuild equal to replay per class before enabling it, and gives a GPU coordinator the nodes' class (a GPU class depends on the device).
- **I2** needs a class per stage for a mixed Vulkan | D3D12 split: C1's `IWorkload.Class` is one class per job.
- **C1** is asked (PR #32, requests): to count only the current membership's (member, serial) pairs as asked in `pickVerifier`; to offer a batch-replay hook in the node (an optional chain-state interface that replays a log suffix), so that a node can use a proven fast rebuild; and to make the simulator take any workload and executor, so the Qwen pipeline can run C1's 200 virtual-time seeds.
- Reopen this if a class cannot read back KV rows cheaply, or if C1's digest rule changes.

## Alternatives

- **A digest over outputs only**, as C1's fake executor does. Rejected: a divergent cache would show only when an output differs, and the cost of hashing the written rows is small.
- **Argmax on the coordinator.** Rejected: the token must come from the class's own argmax kernel (`argmax_rows` on a GPU), as in `Greedy.run`.
- **Tokens without logits.** Rejected: equal ids are a weaker claim than equal logits, and the plan asks for both.
- **The fast rebuild inside `Apply`** (batching the replay's units). Impossible without lookahead: `Apply` must return each unit's output before the next unit arrives.
- **A serving node actor of our own** that calls `Rebuild` during replay. Rejected: it would duplicate C1's node (membership, leases, replay checks) for a speed-up nobody needs on the tiny model; a hook in C1's node is the right place.
- **Process nodes for all 50 CI seeds.** Possible (about 3.5 s per run), but about three times slower in CI and more exposed to runner load; the proxy, TCP and the protocol are the same either way.
