# StagePipeline contract

Owner: C2 (C2a wrote v1 for the CPU classes; C2b adds the GPU executors). Status: v1, proposed with PR #32; it changes only through a `contract-change` PR once it is tagged. Consumers: C2b (GPU executors behind the same executor), I2 (the Gate 1 cluster suite: same-backend and mixed splits, GPU chaos), I3 (`lrm gate1`), T1 and T2 (the cluster report and the dashboard read the records below). Code: `src/Lrm.Serving/` and `apps/Lrm.Cli/Commands/{Coord,Node}.fs`. Tests: `tests/Lrm.Serving.Tests/` (cpu tier on the tiny model, which CI runs; model tier on Qwen3-0.6B, local). Decision record: `docs/decisions/C2-0001-stage-pipeline.md`.

The pipeline runs the Qwen forward (ForwardOpsAndQwenModel) as C1's hash-chained `fwd.stage` work units (ClusterProtocol) on networked nodes: each node hosts whole stages of a split, each chain (lane, stage) keeps that stage's KV cache for one sequence, the residual stream crosses each stage boundary, and the coordinator turns the head's tokens into the next decode step. A chaos run (lag, drops, duplicates, partitions, kills, restarts) and the whole model in one process give the same bits within one ExecutionClass: the same logits and the same greedy ids.

It consumes ClusterProtocol (`cluster.md`: units, the executor interface, the coordinator, nodes, transports, ChaosProxy), ForwardOpsAndQwenModel (`qwen-model.md`: `IForwardOps`, `Forward.runStage`, `KvCache`, `Stage`), ExecutionClass, LrmCore (canonical encoding, DetRng), TelemetryFacade, RunLog and ReportTemplates, and BatchInvariance (for the fast rebuild only).

## The pieces

| Type or module | What it is |
| --- | --- |
| `ModelId` | The identity of a model folder: its verified SHA256SUMS lines |
| `StageLayout` | The split into stages, `--layers` and which stages a node hosts |
| `ServingJob` | A request: greedy decoding of one prompt per lane, split, in one class; its id |
| `StageArgs`, `StagePayload`, `ServingUnit` | The Args, the input and output bytes, and the planning of one unit |
| `ChainDigest` | The state digests of a chain, and the logits digest of a run |
| `QwenWorkload` | C1's `IWorkload`: the coordinator's plan |
| `QwenStageExecutor<'t>`, `IQwenStageExecutor`, `StageExecutorOptions`, `IFastRebuild` | C1's `IStageExecutor` over any `IForwardOps`, its options, and the fast rebuild |
| `PlanOrder`, `QwenReference` | The oracles: every unit in plan order on one executor, and the whole model in one process |
| `PipelineConfig`, `PipelineCoordinator`, `ServingOutcome` | The coordinator of one job on TCP and its outcome |
| `ServingNode`, `ServingCoordinator`, `ServingLocal`, `ServingWorker` | `lrm node`, `lrm coord run`, `lrm coord local`, and the worker entry of `lrm` and test exes |
| `ServingHarness`, `HarnessScenario`, `HarnessMode`, `Topology` | The localhost harness: scenarios, seeded fault schedules, runs, sweeps and the verdict |

## Stages

- **The split.** Boundaries are the first layers of every stage but the first, the rule of `Stage.split`: they increase strictly inside 1 .. layers − 1. `StageLayout.stages layers boundaries` equals `Stage.split config boundaries`. The default is the midpoint, `[layers / 2]`: Qwen3-0.6B splits at `[14]` (stage 0 is the embedding and layers 0 to 13, stage 1 is layers 14 to 27, the final norm and the head), the tiny model at `[1]`. Any split works; Gate 1 uses the midpoint.
- **The boundary carries the residual stream** `[T, hidden]` in the ops' DType: fp32 on every fp32 class (cpu-f32, cpu-device, the GPUs), which is the plan's fp32 residual, and fp64 on cpu-f64.
- **Hosting.** A node hosts whole stages. `--layers` takes ranges (`0-13`, `14-27`, `0-13,14-27`, one layer `5`, or `all`), each of which must start at a stage's first layer and end at a stage's last layer; the node hosts every stage inside a range, and `CanHost` lists only those. Anything else is refused.

## The job

`ServingJob` is `{ Model; Layers; Hidden; Vocab; Boundaries; Class; Prompts; Steps; StopAtEos; Eos }`.

- **Model** is `ModelId` = `Canonical.hash "lrm.serving.model/1"` over the sequence of the verified SHA256SUMS lines (file name as a string, SHA-256 as 32 raw bytes), sorted by the name's UTF-8 bytes. It names content, never a path, so every host of one checkpoint agrees on it.
- **Class** is the ExecutionClass every unit runs in; its backend fixes the payload dtype (F64 for cpu-f64, F32 otherwise). The CPU classes are `ExecutionClass.forCpu`, which a coordinator computes without building ops.
- **Prompts** are one int array per lane (1 to 4,096 lanes, each at least one id inside the vocabulary). **Steps** ≥ 1 greedy ids per lane. **StopAtEos** stops a lane after the first id in **Eos** (the config's EOS ids), kept as its last id; parity runs pass false.
- **Job id** = `Canonical.hash "lrm.serving.job/1"`: model id and class id (32 raw bytes each); layers, hidden and vocab (u32); the boundaries (seq of u32); the prompts (seq of seq of i32); steps (u32); stop at EOS (bool); the EOS ids (seq of i32). It is the units' Job.
- `ServingJob.validate` refuses an invalid split, no lanes or more than 4,096, an empty prompt, an id outside the vocabulary and zero steps. `maxPositions` (prompt + steps − 1, over the lanes) is the cache size every node needs; `maxRows` (the longest prompt) the most rows of one forward.

## Units

Every unit is `fwd.stage` with KindVersion 1 (ClusterProtocol): Job = the job id, Class = the job's class id, Lane = the lane, Stage = the stage index, Step = the greedy step (0 is the prefill), Prev = the previous unit of the chain, Input = the input's blob id, Args = `StageArgs`.

**Args** (`StageArgs`, 54 bytes, canonical): model id (32 raw bytes); layers, first layer, last layer, pos0, rows (u32); head (bool); the dtype code (u8, F32 = 1 or F64 = 7). Decoding refuses another dtype, a layer range outside the model, zero rows, a head flag that is not set exactly on the stage with the last layer, and trailing or missing bytes.

**The plan** (`QwenWorkload`), per lane with a prompt of P ids:

| Unit | Rows | pos0 | Input |
| --- | ---: | ---: | --- |
| stage 0, step 0 | P | 0 | the prompt's ids |
| stage s + 1, step t | as stage s | as stage s | stage s's output at step t (the residual) |
| stage 0, step t + 1 | 1 | P + t | the token of the head's output at step t |

- The head's output at step t ends the lane when its size is wrong (`Invalid`), its token is `0xFFFFFFFF` (`NaN`), StopAtEos is set and the token is an EOS id (`Eos`), or t + 1 = Steps (`Done`); otherwise the lane decodes on (`LaneRule.after`, which the single-process oracle shares).
- A lane has Steps × stages units when it runs to the end; the tiny job below has 96.
- The plan depends only on accepted outputs, so it is deterministic (ClusterProtocol, "Executor interface").

## Payloads

All little-endian.

| Payload | Bytes |
| --- | --- |
| The first stage's input | the ids, i32 each (`StagePayload.ofIds`) |
| A residual (any stage's output but the head's, every later stage's input) | rows × hidden elements of the dtype, row-major: `IForwardOps.Read` of the stage's result |
| The head's output | the token (u32), then the logits of the unit's last row, vocab elements of the dtype (`StagePayload.ofHead`) |

The token is `IForwardOps.Argmax` of the logits row on the node's own backend: the lowest maximal index, `0xFFFFFFFF` for a row that holds a NaN, as `Greedy.run` picks it. The head runs with `HeadRows.Last` on every unit.

## The executor

`QwenStageExecutor<'t>` implements C1's `IStageExecutor` over any `IForwardOps<'t>`, which it owns (`IQwenStageExecutor` hides `'t` and adds `Model`, `Stages`, `Options`, `LiveChains`, `Applied` and `Dispose`). `QwenStageExecutor.create ops weights options` builds one; the serving backend registry starts with cpu-f32 and cpu-f64 (CpuOps), and C2b registers the GPU executors with `QwenStageExecutor.register`.

**Options** (`StageExecutorOptions`; they decide what an executor runs and holds, never the bits):

| Field | Meaning | Default |
| --- | --- | --- |
| `Boundaries` | the split | the midpoint |
| `Hosted` | the indices of the hosted stages (`CanHost`) | every stage |
| `MaxContext` | positions per chain cache; a job must fit | min(1024, max_position_embeddings) |
| `MaxChains` | chains (lane, stage) holding a cache at once | 16 |
| `Threads` | CPU threads of CPU ops (the bits are the same at every count) | 8, or fewer cores |
| `FastRebuild` | allow `IFastRebuild.Rebuild` | false |
| `RebuildChunkRows` | the most rows of one rebuild forward; 0 is MaxContext, and on the head stage at most 64 MiB of logits | 0 |
| `CostMs` | a sleep before each unit, to give test runs a cost | 0 |

The ops plan is `StageExecutorOptions.opsOptions`: the hosted stages, MaxContext positions and rows, no taps, MaxChains caches per stage.

**A chain state** (`Open job chain`) is cheap: it computes the initial digest and touches no ops, because a node opens and disposes states on its actor thread. Its KV cache is created (`KvCache.create`, MaxContext rows) at its first unit. **Apply** then:

1. checks, before any op, that the unit is `fwd.stage` v1 of the state's job and chain and of the executor's class; that the input's blob id is the unit's Input; that the Args name the executor's model, layer count and dtype and a hosted stage with the same layers; that the input has the size of its rows; and that the chain holds exactly pos0 positions (the unit follows the chain head). Each refusal is a `StageException` and changes nothing;
2. runs `Forward.runStage` on the unit's rows at pos0 (ids on stage 0, the uploaded residual otherwise; `HeadRows.Last`; `Step` = the unit's step for telemetry);
3. returns the residual, or on the head stage the token and the logits row;
4. advances the digest (below).

**Threading.** One lock serializes every use of the ops (CpuOps serves one caller at a time; a GPU's command lists too). A state disposed while another chain's unit runs is queued and its cache released at the next use of the ops, so the actor thread never waits for a forward. `MaxChains` bounds the caches; a unit that would exceed it is refused. `Dispose` of the executor releases every cache and the ops; later units raise `ObjectDisposedException`.

## Chain digests

A chain's digest (`IChainState.Digest`) chains over the previous digest, the unit's input and output ids and the digest of the KV-cache rows the unit wrote. It therefore changes whenever an output or a cache row changes and stays changed: the final digest of a chain commits to every output on it and to every valid row of its cache (ClusterProtocol's digest rule, made stronger).

- **Initial** = `Canonical.hash "lrm.serving.init/1"` (job id, lane u32, stage u32).
- **KV rows** = SHA-256(`lrm.serving.kvrows/1` ‖ `00` ‖ payload): first layer, layer count, pos0, rows and the dtype code (u32 each), then for each layer of the stage in order its K rows pos0 .. pos0 + rows − 1 and then its V rows, as their little-endian element bytes (`IForwardOps.ReadRows`, only the rows the unit wrote).
- **After a unit** = `Canonical.hash "lrm.serving.state/1"` (digest, step u32, input blob id, output blob id, KV rows digest).
- Within one class, two states hold the same digest exactly when they applied the same units with the same outputs and cache rows; `KvCache.digest` of their caches is then equal too (the tests check both).
- **Logits digest** of a run = `Canonical.hash "lrm.serving.logits/1"` over the sequence of lanes in order, each its lane (u32) and the sequence of its head outputs (bytes), step by step. Equal logits digests mean equal logits and equal greedy ids, bit for bit.

## KV rebuild

- **Exact replay is the rebuild.** A node that lacks a chain (a restart with no state, a refilled replica, a verifier) fetches the coordinator's unit log and applies the logged units in order with `Apply` (C1's node). Every unit is recomputed, and C1's replay check compares each recomputed output id and digest with the log, so a rebuilt cache is the original bit for bit, and any difference is a replay divergence.
- **The fast rebuild** (`IFastRebuild.Rebuild units`, on a fresh state, with `FastRebuild` on) applies a chain's units (a hash chain from step 0 with contiguous positions, with their inputs) as few forwards of many rows: consecutive units in chunks of at most `RebuildChunkRows` rows (a longer unit is a chunk of its own), the head with `HeadRows.All`, each unit's output taken from its rows (its last row's logits and argmax on the head stage), and the digest advanced unit by unit. It returns every unit's output and digest. Every check runs before the first forward; a refused rebuild changes nothing.
- **It must equal replay bit for bit**: every output, every digest and `KvCache.digest`. The cpu tier proves it on cpu-f32 and cpu-f64 for every chain of the tiny job, in one forward and in chunks of 1, 7 and 16 rows, and that a rebuilt chain continues exactly as the replayed one. It rests on BatchInvariance (prefill = decode = chunked prefill, and a last-row head = the last row of an all-rows head), which M2a proved for CpuOps. A GPU class enables it only after the same proof (C2b); failover never relies on it.
- C1's node replays with `Apply` and does not call `Rebuild`: nodes run exact replay whatever the flag says, until the node offers a batch-replay hook (requested of C1). The harness can run nodes with the flag on (the tests do), which changes no bit.

## The oracles

- **The plan-order run** (`PlanOrder.run workload executor`): every unit in plan order (first in, first out) on one executor that hosts every stage. Its RunDigest and ReplicaDigest are what every cluster run of the job in this class must equal (ClusterProtocol, "Digests").
- **The single-process run** (`QwenReference.single ops model job`): the whole model as one stage in one process, per lane a prefill with the head on its last row and then single-token decode steps, each feeding back `Argmax` of the previous step, stopped by the same lane rule. Its head outputs (token and logits row per step) and logits digest are what the pipeline must equal bit for bit.
- `QwenReference.cpu weights job threads` computes both on a CPU class. The cpu tier proves that the plan-order pipeline's head outputs equal the single-process run's, element for element, on cpu-f32 and cpu-f64; that its ids are `Greedy.run`'s; and that p0's and p1's equal the first 16 golden torch32 ids.

## The coordinator

`PipelineCoordinator job config port` is C1's coordinator with a `QwenWorkload`, a memory blob store and a TCP server transport (port 0 picks a free one).

- **`PipelineConfig`** sets C1's `Replicas`, `MinNodes` and `DrainNs`; everything else is `CoordinatorConfig.standard`. A two-node split (one node per stage) uses `Replicas` = 1, as C1 requires (with 2, every stage needs two eligible hosts). Failover without catch-up uses 2. `DrainNs` is 10 s, because a verifier may replay a chain; the drain ends as soon as every final unit is settled.
- **`ServingOutcome`** is C1's `RunOutcome` plus the workload's tokens, lane ends and logits digest, as one JSON line (`ServingOutcome.toJson`, prefix `LRM-COORD-OUTCOME `).
- **Bit-identical** (`ServingOutcome.differences`) means: status completed, no divergence, the job id, the plan-order run's RunDigest and ReplicaDigest, and the single-process run's logits digest and greedy ids.

## Commands

| Command | What it does |
| --- | --- |
| `lrm coord run --listen PORT [job options] [--replicas 1] [--nodes 2] [--drain-ms 10000] [--timeout-ms 600000] [--verify] [--threads T] [--run-log] [--title TEXT] [--progress]` | The coordinator of one job as its own process. It prints `LRM-COORD-LISTEN <port>`, `LRM-COORD-JOB <job id>`, `LRM-COORD-RUN <run folder>` (with `--run-log`), `LRM-COORD-PROGRESS <accepted units>` (with `--progress`), a summary and the outcome line. `--verify` computes both oracles here and compares. Exit 0 when completed (and bit-identical with `--verify`), 1 otherwise, 2 for bad options |
| `lrm coord local [job options] [--topology split2\|replicated3\|bothlost5] [--seed S \| --seeds A-B,C] [--kill N:AFTER[:RESTART_MS]] [--partition N:FROM_MS:UNTIL_MS] [--sever N:MS] [--no-faults] [--chaos none\|lan\|proxy\|standard] [--drop P] [--duplicate P] [--reorder P] [--min-delay-ms] [--max-delay-ms] [--reorder-delay-ms] [--nodes process\|here] [--proxy process\|here] [--coordinator here\|process] [--cost-ms 2] [--threads 1] [--timeout-ms 60000] [--fast-rebuild] [--run-log] [--title TEXT] [--parallel 1]` | The harness on localhost (below): by default the coordinator here, `lrm chaos-proxy` and `lrm node` processes. Each seed's scenario comes from its topology and schedule unless `--kill`, `--partition` or `--sever` give the faults. It prints one line per run and a summary; exit 0 when every run is bit-identical |
| `lrm node --connect HOST:PORT --id N --name nN [--incarnation X] [--model tiny\|qwen3-0.6b\|DIR] [--backend cpu-f32\|cpu-f64] [--split B] [--layers A-B[,C-D]\|all] [--threads T] [--max-context M] [--chains C] [--fast-rebuild] [--cost-ms C] [--timeout-ms 120000] [--run-dir DIR [--run-node ID] [--restart-of ID]]` | One node: C1's node actor over a QwenStageExecutor of the hosted stages, behind TCP. It prints C1's report line (`LRM-NODE-REPORT`) when the coordinator shuts the run down. Exit 0 done, 1 the node failed, 2 bad options, 3 no shutdown in time |
| `lrm chaos-proxy ...` | C1's frame-aware TCP proxy (ClusterProtocol, "Commands"), reused as is |

**Job options**: `--model` (`tiny` is `goldens/tiny/g1/model`, `qwen3-0.6b` is `LRM_DATA/models/Qwen3-0.6B`, else a folder), `--backend` (cpu-f32), `--split` (the midpoint; `none` for one stage), `--steps` (16), `--stop-at-eos`, and the lanes' prompts in order: `--ids 1,2,3`, `--ids-from FILE[:TENSOR]` (an integer tensor, `input_ids` by default), `--prompt TEXT` (UTF-8 bytes, byte-level models only) and `--prompt-set tiny` (the tiny golden prompts p0, p1 and p2), each repeatable; the tiny set by default. A harness passes the same options to a coordinator process (`JobSpec.toArgs`), which builds the same job.

**Workers.** `ServingWorker.main` runs `node ...`, `coord run ...` or `chaos-proxy ...`; `lrm` dispatches them as commands, and a test exe as `--serving-worker <command>` (with `Env.workerEnvironment`, so a worker never takes a test mutex).

## The harness

`ServingHarness.run mode scenario` runs one job on localhost: a coordinator, C1's ChaosProxy, and the nodes connected to the proxy. `HarnessMode` puts each piece in this process or in a process: node actors or `node` processes, the proxy object or a `chaos-proxy` process, the coordinator here or a `coord run` process. The proxy applies the frame chaos (ChaosConfig.proxy by default: 1% drops, 1% duplicates, 2% reordered by 10 ms, 0 to 3 ms of delay) and its partitions and severs; the harness kills nodes (a process kill, or an actor stopped with its executor) once the coordinator has accepted a number of units, and restarts them as new incarnations with no state (incarnation = id + 1000 × attempt). Nodes that start later than the others (`StartAfterUnits`) are spares.

| Topology | Nodes | Replicas | Seeded faults |
| --- | --- | ---: | --- |
| `split2` | n1 hosts stage 0, n2 stage 1 | 1 | one node killed after 15–60% of the units and restarted 0.2–1.2 s later: the loss of every replica of a stage, rebuilt by replay after the restart |
| `replicated3` | n1, n2, n3 host every stage | 2 | one node killed after 15–60% of the units, restarted 0.2–1.5 s later with probability 1/2: failover with no catch-up, and a refilled replica rebuilt by replay |
| `bothlost5` | n1, n2 stage 0; n3, n4 stage 1; spare n5 every stage, started after the first unit | 2 | n3 killed after 15–40% of the units and n4 3–15% later, one of them restarted: both replicas of the last stage lost, the spare takes it over by replay |

- **The seeded schedule** (`ServingHarness.chaos spec topology seed startMs`) draws the faults from `DetRng(seed, DetRng.streamOf "lrm.serving.chaos.faults")`: the kills above, then with probability 1/2 a partition of 0.3 to 1 s starting 0.1 to 0.7 s after `startMs`, then with probability 1/2 a severed connection 0.1 to 2 s after `startMs`. A partition targets any node of `split2`, a stage-0 node of `bothlost5`, and a node of `replicated3` only when its killed node comes back ("Limits"). `Topology.ofSeed` takes the three topologies in turn (seed mod 3 = 1, 2, 0). Each unit costs 2 ms (`CostMs`) so that the faults land mid-run, and every run must finish within 60 s.
- **The verdict** (`ServingHarness.verdict reference result`) is empty exactly when the run is bit-identical: no harness failure, `ServingOutcome.differences` empty, every surviving node exited cleanly and reports the reference digest for each chain it holds at its final head, and every chain is held at its final head by a surviving node.
- `ServingHarness.sweep` runs scenarios a few at a time.

## Telemetry

The pipeline emits only reserved names (TelemetryFacade) that existing templates chart:

- **The coordinator** emits C1's `cluster.*` records (ClusterProtocol, "Telemetry"); the cluster template draws the stage Gantt chart from `cluster.unit.done` and the rest.
- **Each node** emits C1's node records and, from `Forward.runStage`, the `model.forward`, `model.layer`, `model.flops`, `model.bytes_moved` and `model.tokens` records with `stage` (and `step`), into its own events file; the cluster template's metering table attributes them per node and stage.
- **The harness** adds `cluster.chaos` counters on the coordinator's run: `fault` = `kill` or `restart` with `node` = the node's name (the Gantt chart's markers), and at the end `fault` = `drop` and `duplicate` with `node` = `proxy` (the proxy's totals).
- **The comparison with the oracles**, when the harness or `lrm coord run --verify` makes one, is four `golden.compare` checks with metric `exact`, a `check` and no `tap` (the parity report's self-test table): `cluster_run_digest_vs_plan_order`, `cluster_replica_digest_vs_plan_order`, `cluster_logits_vs_single_process` and `cluster_greedy_ids_vs_single_process`, with `kind` = `determinism`.

**The run log** (RunLog): kind `cluster`, created by the coordinator's process as node `coord` with the job's class and its parameters (model, model id, job id, backend, split, lanes, prompt tokens, steps, replicas, and the harness's scenario, topology, seed, kills and proxy faults). Every node joins it with its own class (`--run-dir`); a restarted node joins as `<name>-r<attempt>` (`RunLogNames.restartedNode`) with `RestartOf` = its previous id. A killed process leaves its events file without a trailer, which the reader reports. On one host the run folder is shared; across hosts each node would need the manifest from the coordinator (RunLog, "Runs across hosts"), which ClusterProtocol does not carry yet.

**Reports.** `lrm report <run>` renders a pipeline run with the cluster template; `--template parity` lists the comparisons. The committed reports are in `docs/reports/cluster-dev/`.

## Tags

| Tag | Used for |
| --- | --- |
| `lrm.serving.model/1` | ModelId |
| `lrm.serving.job/1` | The job id |
| `lrm.serving.init/1` | The digest of an empty chain |
| `lrm.serving.state/1` | The digest after a unit |
| `lrm.serving.kvrows/1` | The digest of the KV rows a unit wrote |
| `lrm.serving.logits/1` | The logits digest of a run |
| `lrm.serving.chaos.faults` | The DetRng stream of the harness's fault schedules (`DetRng.streamOf`) |

## Limits and design choices

- **One class per job.** C1's `IWorkload.Class` gives every unit of a job one class, so a node of another class answers WrongClass. A mixed split (Vulkan | D3D12, I2) needs a class per stage, which ClusterProtocol does not have yet.
- **The coordinator's class.** A CPU class is computed from the backend name; a GPU class depends on the device and driver, so a GPU coordinator needs the nodes' class (C2b).
- **C1's drain verifier rule.** `pickVerifier` counts a member asked in an earlier membership as asked by its id. A node that is partitioned while it owes a chain's final unit, expelled and rejoined, is never engaged again as a verifier, so with no other replica left the run ends Unverified although every digest equals the oracles (seen in sweeps before the rule below, in `replicated3` seeds 5 and 32: a permanent kill, then a partition of the third node over the last steps). The seeded partitions avoid that combination; C1 is asked to compare (member, membership) pairs of the current membership.
- **The head output carries the logits row**: 1 KiB per step on the tiny model, 594 KiB on Qwen3-0.6B, kept by the coordinator's in-memory blob store with every other output, for the comparison with the single-process run.
- **Digest cost.** Each unit reads back the KV rows it wrote (112 KiB per row of a 14-layer Qwen3-0.6B stage), never the whole cache.
- **The fast rebuild is not wired into C1's node** (above).
- **One coordinator**, as in ClusterProtocol: losing it ends the run.

## Vectors

`tests/Lrm.Serving.Tests/vectors/serving_vectors.py` re-implements the encodings from this text with Python's `hashlib` and `struct` and prints every value below; the cpu tier checks the F# code against them. The vector job is: model = the bytes `01 02 … 20`, the fake executor's class (ClusterProtocol, `86d7e37b…ee04`), 2 layers, hidden 64, vocabulary 256, split `[1]`, lanes `[1, 2, 3]` and `[4]`, 4 steps, EOS `[7]` ignored.

| Vector | Value |
| --- | --- |
| M1: ModelId of (`model.safetensors`, 32 × `11`), (`REVISION`, 32 × `22`), (`config.json`, 32 × `33`) | `ee4be75724ddf15bb8355336c14b7b236ba502b565e936b0658cb3efd3bc067a` |
| J1: the vector job's id | `ca0e63adc155bf3494bfd8dbc86c13ee9899db27f1a859e183c1f175dec6218c` |
| J2: the same job with StopAtEos | `32826f379e592d5ca236116421c33be5411adbe777a75e087cc678dd0894ef7a` |
| A1: Args of stage 0 (layer 0), pos0 0, 3 rows, F32 | `0102…1f20 02000000 00000000 00000000 00000000 03000000 00 01` |
| A2: Args of stage 1 (layer 1), pos0 3, 1 row, head, F32 | `0102…1f20 02000000 01000000 01000000 03000000 01000000 01 01` |
| P1: the ids `[1, -1]` | `01000000ffffffff` |
| P2: a head output, token 7 and the logits `[0.5]` | `070000000000003f` |
| U1: lane 0, stage 0, step 0 of J1, input the ids `[1, 2, 3]`, Args A1 | `2d17dac3e03fdabc60aa9f49ad1d24807ddd3b20e2be951f5f949f213a610a68` |
| D0: the initial digest of J1's chain (lane 1, stage 1) | `5abefe293d5d6edd36288bb4d94ba4b14d0bee4a3e129dc7384761a0d1665301` |
| K1: KV rows of layer 1 (one layer), pos0 2, 1 row, F32, K = (1.5, −2), V = (0.25, 8) | `49a80723ee6cbe18b3a31a10017d8772ab50fff97a819719e9d3e671f1874c6f` |
| D1: after D0, step 3, input "abc", output of no bytes, KV rows K1 | `d336f89922584e44d681fadecad6bbd426a7ddf2e931d93d5498effc855f5174` |
| L1: logits digest of lane 0 = [head output token 5, logits (1, 2)] and an empty lane 1 | `21bdec2fd4ce03bf97abbbe86e0286995322cb4bdac042c1a16be68aa8f4ebd2` |

**Decision tables.**

| Input | Result |
| --- | --- |
| `StageLayout.midpoint` 2; 28; 1 | `[1]`; `[14]`; `[]` |
| `StageLayout.stages 28 [14]` | (0, 0..13), (1, 14..27) |
| boundaries `[0]`, `[28]`, `[14; 14]`, `[14; 7]` of 28 layers | refused |
| `--layers` on Qwen3-0.6B's `[14]`: `0-13`; `14-27`; `all`; `0-27`; `0-13,14-27` | stage 0; stage 1; both; both; both |
| `--layers` `0-6`, `5-20`, `0-28` | refused (part of a stage, parts of two, beyond the model) |
| `ServingJob.maxPositions` of the vector job | 6 (3 + 4 − 1) |

**The tiny pipeline** (measured, informative). The tiny job (`JobSpec.tiny`: p0, p1 and p2 of `goldens/tiny/g1`, 13, 1 and 70 ids, 16 steps, split `[1]`) has 96 units. Its greedy ids, on both CPU classes, are p0 `76 170 135 201 174 203 135 232 232 7 49 255 217 76 246 89` (the golden torch32 ids' first 16), p1 `165 49 165 49 209 49 209 209 209 209 209 209 209 209 209 209` (likewise) and p2 `129 121 124 188 149 149 149 149 149 149 149 149 95 149 95 119`. The job ids, run digests and logits digests depend on the class, whose driver is the .NET runtime; on the development machine (.NET 10.0.13, win-x64+avx2+fma):

| Class | Job id | RunDigest | Logits digest |
| --- | --- | --- | --- |
| cpu-f32 (`7cc3fe8c…7405`) | `da6f4c6d6e6f4e0907cb835885b1c2b56f37d5cedd0e7dbf55f4c9fd318139eb` | `e3ce317691dc2c47be6ac505535aae009c3b87064dd54e26f2d5b2112e672844` | `a5c08eb28110f3f74e74d321f75f028937546d7181ae2782bfb41fd4259c79bc` |
| cpu-f64 (`f705487c…9b6d`) | `21cefe333118d3770e907c0543bcb38d46f6c69975117f35b882003d8103739c` | `65aa3917bb81b3ef222369e114ecc22197df87f9856ef2d1f66ff84c8c295057` | `4a842f1565bc2c63b892efb848dd0ad2b77319390290c3c7a9d98076c7f63fb2` |

**Qwen3-0.6B** (model tier, measured): split `[14]` on two node actors behind the chaos proxy, the stage-1 node killed after 2 units and restarted, greedy decoding after `785 3974 13876 38835` on cpu-f32 gives `34208 916 279` (M2a's single-process ids), bit-identical to the single-process run.

## Verification

The cpu tier of `Lrm.Serving.Tests` (CI runs it on every PR):

- **Encodings** (`JobTests`): the vectors above; Args round trips and refusals; payloads; the split against `Stage.split` and the decision tables; `--layers`; job refusals; the lane rule; the job spec through another process's command line; the tiny prompt set.
- **The executor** (`ExecutorTests`): the plan-order pipeline equals the single-process run on cpu-f32 (every head output, element for element) and cpu-f64; the ids are `Greedy.run`'s and p0's and p1's the golden torch32 ids; the digest changes with every unit and stays changed after one flipped input; two executors at 2 and 1 threads give equal outputs, digests and `KvCache.digest`; every refusal, with nothing changed; MaxChains and releases; states opened and disposed on another thread while units run; a cluster in one process on InProc transports.
- **Rebuild** (`RebuildTests`): exact replay gives every output and digest of the plan-order run; the fast rebuild equals replay on both CPU classes, in one forward and in chunks of 1, 7 and 16 rows, and continues like it; its refusals.
- **The cluster** (`ClusterTests`, real time, 60 s per run, digests and ids only): a clean split; **50 proxy seeds** (1 to 50, the three topologies in turn, node actors in the test process behind a ChaosProxy on TCP, four at a time) each bit-identical, with every seed killing a node and rebuilding a chain by replay, restarts, dropped, duplicated and severed frames and partitions; three seeds on cpu-f64; nodes with fast rebuild on.
- **Processes** (`ProcessTests`): `node` processes of the test exe and C1's `chaos-proxy` process, with real kills and restarts, for `split2` (a kill, a restart and a sever), `replicated3` (failover) and `bothlost5` (both replicas lost, the spare replays), and once with the coordinator as a `coord run` process too; `lrm node`'s refusals.
- **Telemetry** (`RunLogTests`): telemetry changes no value; a multi-process run writes one cluster run log (coord, n1, n2, n2-r1 with its `RestartOf`; only the killed process's missing trailer as a problem) with C1's records, the nodes' `model.forward` records per stage, the kill and restart markers and the four passed comparisons, and the cluster template renders PASS and a Gantt chart with the markers.

The model tier runs Qwen3-0.6B split in two with a kill and a restart (above). `lrm coord local --seeds 1-50` runs the same 50 scenarios with node and proxy processes.
