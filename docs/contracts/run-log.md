# RunLog contract

Owner: T1. Status: v1, proposed with T1's PR; it changes only through a `contract-change` PR once it is tagged. Consumers: every host that records a run (test exes, `lrm` commands, cluster nodes), the reports (T1), the dashboard (T2) and the gates (I1 to I3). Code: `src/Lrm.Telemetry`. Tests: `tests/Lrm.Telemetry.Tests` (cpu tier; timing in the bench tier).

A run log is the only input of reports and the dashboard. It is written through the TelemetryFacade (`telemetry-facade.md`): components report through the ambient `ITelemetry`, and the host installs a `RunContext` whose sink writes the files below.

## Layout

```
LRM_DATA/runs/<runId>/
  manifest.json          the run, written by the node that created it (schema lrm.run/1)
  events.<node>.jsonl    one file per node, written only by that node (line schema v1)
  artifacts/             free-form files the run produces (RunContext.ArtifactsDirectory)
  report/                rendered reports, the default output of `lrm report`
```

- **Ids.** `runId` and `node` are safe names (LrmCore contract): `RunId.fresh ()` gives `yyyyMMdd-HHmmss-xxxxxxxx`, and a single-process run uses node `n0`. Ids are labels and never enter a hash.
- **Never overwritten.** `RunContext.Create` refuses an existing run folder, and every events file is opened with `FileMode.CreateNew`.
- **Never committed.** Run logs stay under `LRM_DATA` (DataRootAndToolchain contract). Reports rendered from them may be committed.
- **One writer per file.** The creating node owns `manifest.json`; every node, the creator included, owns only its own events file. Nodes that join a run declare themselves in their events file (the `run.node` header below), so no two processes ever write one file.

## manifest.json

Schema `lrm.run/1`. The creating node writes it when the run starts (status `running`) and again when it finishes. Each write goes to `manifest.json.tmp` and is then renamed over the old file.

| Key | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Always `lrm.run/1` |
| `runId` | string | The run id; the folder name |
| `kind` | string | What the run is for, a safe name. A report template name (`parity`, `inference`, `kernels`, `cluster`, `gate1`, `training`, `routing`, `thinking`) makes `lrm report` pick that template by default |
| `title` | string | One line for people |
| `status` | string | `running`, `ok`, `failed` or `aborted` (disposed without `Finish`, for example after an exception) |
| `createdUtc` | string | Start time, ISO 8601 UTC with milliseconds. A label only |
| `endedUtc` | string or null | End time, once finished |
| `node` | string | The creating node |
| `synthetic` | bool | True for runs made by a synthetic generator. Reports then carry a banner |
| `gitSha` | string or null | The commit the run was built from, when the host knows it |
| `executionClass` | object or null | The creating node's ExecutionClass: `backend`, `deviceId`, `driver`, `shaderSetHash` (64 hex digits or null), `arch`, and `id`, the ExecutionClassId in hex |
| `driver` | string or null | The class's driver again, because gates pin the driver recorded in the manifest (dev plan, gate 8) |
| `params` | object | Free-form string parameters (model, prompt set, seed, ...), keys sorted ordinally |

**Text form.** JSON indented by two spaces, keys in the order of the table, `": "` after keys, LF line ends, a final LF, UTF-8 without a BOM. Strings follow "JSON text" below. The same manifest always renders to the same bytes, under any culture.

**Reading.** `RunManifest.parse` requires `schema` = `lrm.run/1` exactly, a known status, and an `executionClass` that validates and whose recorded `id` equals `ExecutionClass.id` of its fields. Anything else is an error.

### Vector M1

The manifest `{ runId 20261018-090000-0000002a; kind parity; title "CPU parity"; status ok; createdUtc 2026-10-18T09:00:00.000Z; endedUtc 2026-10-18T09:05:00.000Z; node n0; synthetic false; gitSha 0123abc; executionClass X2; params seed=7, model=Qwen3-0.6B }`, where X2 is the GPU class of the ExecutionClass contract, renders to exactly:

```json
{
  "schema": "lrm.run/1",
  "runId": "20261018-090000-0000002a",
  "kind": "parity",
  "title": "CPU parity",
  "status": "ok",
  "createdUtc": "2026-10-18T09:00:00.000Z",
  "endedUtc": "2026-10-18T09:05:00.000Z",
  "node": "n0",
  "synthetic": false,
  "gitSha": "0123abc",
  "executionClass": {
    "backend": "vulkan",
    "deviceId": "pci:10de:2704",
    "driver": "616.92",
    "shaderSetHash": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "arch": "win-x64",
    "id": "d79c93db855714890ff41356c0fd4dcb61e048b1fa346c87b7ca140506a3b057"
  },
  "driver": "616.92",
  "params": {
    "model": "Qwen3-0.6B",
    "seed": "7"
  }
}
```

## events.<node>.jsonl

One telemetry record per line: compact JSON with exactly the keys `v, t, k, n, d, x, u, node, seq, a`, always present and always in that order, then LF. UTF-8 without a BOM.

| Key | Type | From the TelemetryRecord |
| --- | --- | --- |
| `v` | integer | The line schema version, `1` |
| `t` | integer | `TimeNs`: Unix ns, the start of a span, otherwise when the record was made |
| `k` | string | `TelemetryKind.name`: `span`, `counter`, `gauge`, `histogram`, `event`, `gpu`, `meter`, `check` |
| `n` | string | `Name` |
| `d` | integer | `DurationNs`: span host duration, gpu duration, otherwise 0 |
| `x` | number or string | `Value`. A finite value is its invariant `R` text (`1500`, `3.2E-07`, `-0`); NaN and the infinities are the strings `"NaN"`, `"Infinity"` and `"-Infinity"` |
| `u` | string | `Unit` |
| `node` | string | The node that wrote the file |
| `seq` | integer | 0, 1, 2, ... in file order: the sink assigns it under its write lock |
| `a` | object | The attributes (below) |

**Structural attributes** come first in `a`, and their `$` prefix cannot clash with an attribute key (TelemetryNames allows only `a-z`, `0-9`, `_` and `.`):
- `$span`: the span id, on `span` records only (always present there);
- `$parent`: the enclosing span's id, when it is not 0;
- `$gpu_start`: the GPU start in ns, on `gpu` records only (always present there, even when 0).

**Attributes** follow in the order of their first occurrence in `Attrs`. A key that occurs more than once is written once, at its first position, with its last value, so the innermost scope wins. Values:

| AttrValue | JSON |
| --- | --- |
| `Str s` | a string |
| `Int i` | an integer |
| `Float f` | the `R` text, plus `.0` when that text has neither a `.` nor an `E` (`1.0`, `-0.0`, `1E-05`). NaN and the infinities are `{"$f":"NaN"}`, `{"$f":"Infinity"}` and `{"$f":"-Infinity"}` |
| `Bool b` | `true` or `false` |

A reader therefore types numbers by their text: a number with `.`, `e` or `E` is a Float, any other is an Int (int64).

**JSON text.** Strings escape `"` and `\`, the short escapes `\b \f \n \r \t`, every other character below U+0020 as `\u00xx` (lowercase hex), and U+2028 and U+2029 as `\u2028` and `\u2029`. A lone surrogate becomes U+FFFD. Every other character is written as itself in UTF-8, a surrogate pair as its one 4-byte sequence. The bytes of a line are the UTF-8 of its text and nothing else: they never depend on the lines written before it or on how the writer buffers the text. Numbers use the invariant culture only.

**Header and trailer.** Every events file starts with a header record and a finished one ends with a trailer, both of kind `event`:

| Record | seq | Attributes |
| --- | --- | --- |
| `run.node` | 0 | `role` (`creator` or `joined`); when the node has an execution class: `backend`, `device_id`, `driver`, `shader_set_hash` (hex, or `""`), `arch` and `xclass` (the ExecutionClassId in hex) |
| `run.end` | last | `status` (as in the manifest), `records` (the records before it, which is its own seq), `dropped`, `invalid` |

A file without a trailer belongs to a node that is still running, or one that stopped without finishing (killed, crashed). Readers report it; it is not an error in itself.

### Vectors E1 to E5

All at `t` = 1760000000000000000 unless stated.

| Vector | Record | Line |
| --- | --- | --- |
| E1 | span `model.layer`, d 1500, x 1500, u `ns`, span 3, parent 1, attrs `component=forward`, `layer=3`; node `n0`, seq 7 | `{"v":1,"t":1760000000000000000,"k":"span","n":"model.layer","d":1500,"x":1500,"u":"ns","node":"n0","seq":7,"a":{"$span":3,"$parent":1,"component":"forward","layer":3}}` |
| E2 | gpu `gpu.dispatch`, d 9400, x 9400, u `ns`, gpu start 0, attr `kernel=matmul_bf16w`; node `n1`, seq 0 | `{"v":1,"t":1760000000000000000,"k":"gpu","n":"gpu.dispatch","d":9400,"x":9400,"u":"ns","node":"n1","seq":0,"a":{"$gpu_start":0,"kernel":"matmul_bf16w"}}` |
| E3 | check `golden.compare`, x 1, float attrs `measured` 3.2e-7, `bound` 1e-5, `exact` 1.0, `neg` -0.0, `bad` NaN, then `ok=true`, `tap=L3.q_proj`; node `n0`, seq 1 | `{"v":1,"t":1760000000000000000,"k":"check","n":"golden.compare","d":0,"x":1,"u":"","node":"n0","seq":1,"a":{"measured":3.2E-07,"bound":1E-05,"exact":1.0,"neg":-0.0,"bad":{"$f":"NaN"},"ok":true,"tap":"L3.q_proj"}}` |
| E4 | meter `model.flops`, x +infinity, u `flop`, attrs `layer=1`, `note` = tab, `here "q" \ é 😀 ` and U+2028, then `layer=2`; node `n0`, seq 2 | `{"v":1,"t":1760000000000000000,"k":"meter","n":"model.flops","d":0,"x":"Infinity","u":"flop","node":"n0","seq":2,"a":{"layer":2,"note":"tab\there \"q\" \\ é 😀 \u2028"}}` |
| E5 | event `cluster.node_lost` at `t` = -5, attrs `target=s0a`, `reason` = `a`, U+0001, `b`, a lone U+D800; node `coord`, seq 12 | `{"v":1,"t":-5,"k":"event","n":"cluster.node_lost","d":0,"x":0,"u":"","node":"coord","seq":12,"a":{"target":"s0a","reason":"a\u0001b�"}}` |

### Vector R1: a whole events file

A run created with run id `kat-run`, no execution class, and a clock that returns 1000, 2000, 3000, ... on successive reads; then, with its facade installed, `Telemetry.counter "model.steps" 1.0 "count" []`, a span `model.layer` with `layer=0` around `Telemetry.meter "model.flops" 2048.0 "flop" []`, and `Finish RunStatus.Ok`. Its `events.n0.jsonl` is exactly (the manifest took the clock's first read):

```
{"v":1,"t":2000,"k":"event","n":"run.node","d":0,"x":0,"u":"","node":"n0","seq":0,"a":{"role":"creator"}}
{"v":1,"t":3000,"k":"counter","n":"model.steps","d":0,"x":1,"u":"count","node":"n0","seq":1,"a":{}}
{"v":1,"t":5000,"k":"meter","n":"model.flops","d":0,"x":2048,"u":"flop","node":"n0","seq":2,"a":{"$parent":1}}
{"v":1,"t":4000,"k":"span","n":"model.layer","d":2000,"x":2000,"u":"ns","node":"n0","seq":3,"a":{"$span":1,"layer":0}}
{"v":1,"t":7000,"k":"event","n":"run.end","d":0,"x":0,"u":"","node":"n0","seq":4,"a":{"status":"ok","records":4,"dropped":0,"invalid":0}}
```

A span's line is written when it ends, so file order is end order: the meter inside the span comes before the span.

## Writing: JsonlSink and RunContext

```fsharp
use run = RunContext.Create { RunOptions.create "parity" "CPU parity" with ExecutionClass = Some cls }
use _ = run.Install()            // the ambient facade now writes events.n0.jsonl
// ... work that reports through Telemetry ...
run.Finish RunStatus.Ok          // trailer, close, manifest status and end time
```

- **`RunOptions`**: `Kind`, `Title`, `Node` (default `n0`), `ExecutionClass`, `Params`, `Synthetic`, `GitSha`, `RunId` (default `RunId.fresh ()`), `RunsRoot` (default `LrmPaths.runs ()`), `Clock` (default `TelemetryClock.system`; tests and simulations pass a virtual clock) and `FlushInterval` (default 1 s).
- **`RunContext.Join(runDir, options)`** opens another node's file in an existing run; it refuses the creator's node id. `JoinOptions` holds `Node`, `ExecutionClass`, `Clock`, `FlushInterval` (default 1 s), `Manifest` and `RestartOf` (see "Runs across hosts"); `Join(runDir, node, class, clock)` is the shared-folder shorthand. The cluster (C1, C2) gives every process of a run its own node id under one `runId`.
- **`Finish status`** writes the trailer, closes the file and, on the creator, rewrites the manifest with the status and end time. Later calls do nothing. Disposing without `Finish` finishes with `aborted`.
- **The sink never throws** (TelemetryFacade contract). A record whose name, unit or attribute keys break the naming rules (`TelemetryNames.check`) is refused and counted as `invalid`; a record that cannot be written (an I/O error, or the sink is closed) is dropped and counted as `dropped`. The trailer reports both, and so does every report.
- **Thread-safe.** Records from any thread are serialized under one lock, which also assigns `seq`.
- **Buffered.** Lines reach the file when the 64 KiB buffer fills, every `FlushInterval`, on `Flush` and on `Finish`. A reader tolerates a cut-off last line, so a live run can be read while it is written (the dashboard's `--watch`).
- **Telemetry never changes values.** Nothing the sink or `RunContext` does feeds back into a computation; the cpu tier runs an instrumented computation with the no-op facade, an in-memory sink, the run-log sink and the run-log sink on a virtual clock, and requires identical bits.
- **Determinism.** The same records with the same clock give byte-identical files under any culture. Timestamps never enter a hash.

## Runs across hosts

A cluster run spans hosts that share no file system and no clock. Four rules keep one run folder readable as one run.

- **The manifest travels with the run.** A node on another host joins with `JoinOptions.Manifest` set to the creator's manifest (the cluster sends it when the node registers). `Join` creates `LRM_DATA/runs/<runId>/` on that host if it is missing, checks that the folder is named after the run, and writes the manifest there when the folder has none. That copy is the node's local reference only; it is never collected back, and the creator's `manifest.json` stays the one that counts.
- **Events files are collected, never merged.** Each node writes only `events.<node>.jsonl` in its local folder. The creator's host collects every node's file into its own run folder under the same name, byte for byte, during the run or after it. Collection never rewrites, merges or reorders lines. A file collected while its node is still writing is a file with a cut-off last line and no trailer, which the reader already tolerates; a later collection replaces the copy with the longer file.
- **A restarted process is a new node.** Events files are never reopened (`FileMode.CreateNew`), so a process that restarts takes a new id, `RunLogNames.restartedNode node attempt` = `<node>-r<attempt>` (`n1-r1`, `n1-r2`, ...), and joins with `JoinOptions.RestartOf` = its old id. Its header records `restart_of`, and `NodeLog.RestartOf` reads it back. The old node's file keeps what it wrote before the restart.
- **One time axis: the creator's clock.** The creating node measures each other node's clock offset (for example from a ping's round trip) and records it with `RunContext.RecordClockOffset(target, offsetNs, rttNs)`: an event `run.clock_offset` in the creator's own file, with `target` (the node), `offset_ns` (the target's clock minus the creator's, in ns) and `rtt_ns` (the round trip the measurement took, its uncertainty). Only the creator may record offsets; the reader reports offsets in any other file as a problem and ignores them. The reader subtracts the last offset recorded for a node from every `TimeNs` of that node, in `load` and `stream`, and reports it as `NodeLog.ClockOffsetNs`. A node with no recorded offset keeps its own times. GPU timestamps (`$gpu_start`) are never shifted: they belong to their device's own domain.

## Reading: RunLogReader

- `RunLogReader.load dir` returns a `RunLog`: the manifest, one `NodeLog` per events file (ordered by node id, ordinal) and every data record as a typed `RunEvent`, nodes in id order and each node in seq order, times on the creator's clock. Structural records (`run.node`, `run.end`, `run.clock_offset`) are not in `Events`; they fill the `NodeLog` (role, execution class, status, dropped and invalid counts, `RestartOf`, `ClockOffsetNs`). `RunLog.ClassOf node` is the node's class from its header, else the manifest's for the creator.
- `RunLogReader.stream dir` streams the same data records without holding them, and `readLines path` streams one file as (line number, record or error).
- **Bytes first.** A file is split into lines on LF bytes, and each line is checked as UTF-8 on its own. A line with invalid UTF-8 (a stray byte, or a file cut inside a multi-byte character) is a problem of that line, `invalid UTF-8`; it never stops the read. A UTF-8 byte order mark at the start of the file is skipped.
- **Tolerance.** Only a missing or invalid manifest is an error. Inside an events file, a bad line is skipped and reported, and the rest is read. `RunLog.Problems` lists, per node: malformed lines (with the reason), invalid UTF-8, a cut-off last line, a `seq` that skips (records lost), a record of another node, a missing or misplaced header, a missing trailer, a trailer whose count disagrees, records after the trailer, clock offsets from a node that did not create the run, and non-zero dropped or invalid counts. Every report prints them.
- **Reasons in fixed words.** A reason never quotes the .NET runtime's own message, which changes with its version and UI culture: a JSON syntax error is `malformed JSON at byte <n>` (`at line <l>, byte <n>` in a manifest, lines from 1), a bad number `bad number`, a decoding error `invalid UTF-8`. The same damaged log gives the same problems, and so the same report bytes, on every machine.
- **Strict line parsing** (`EventLine.decode`): every key present exactly once, no other key, `v` = 1, a known kind, integer `t`, `d` and `seq`, `$span` exactly on span records and `$gpu_start` exactly on gpu records, no other `$` key, no duplicate attribute, and attribute values of the four types only.
- `RunEvent` offers `TryStr`, `TryInt`, `TryFloat` (an Int widens), `TryBool` and `TryText` (any value as display text).

## Metering

Metering is attribution of resource use, computed from ordinary records; there is no separate metering stream.

**Emitting.** `Metering` has helpers that build the record shapes the reports read. They only report:

| Helper | Record |
| --- | --- |
| `Metering.flops name amount attrs` | `meter`, unit `flop` |
| `Metering.bytes name amount attrs` | `meter`, unit `B` |
| `Metering.tokens name amount attrs` | `meter`, unit `token` |
| `Metering.time name ns attrs` | `meter`, unit `ns` (time that is not a span, such as a queue wait) |
| `Metering.arena arenaName bytes attrs` | `gauge` `mem.arena`, unit `B`, with `arena` |
| `Metering.op prefix op attrs` | `meter` `<prefix>.flops` (unit `flop`) and `meter` `<prefix>.bytes_moved` (unit `B`), both with `op` = the registry name, from the FLOP registry below. Nothing is computed when the facade is the no-op |

The reserved names keep their owners (TelemetryFacade contract): the model uses `model.*` (for example `Metering.op "model" ...`), the GPU backends and executor `gpu.*`, the heap and executor `mem.arena`. Wall time comes from spans and GPU time from `gpu` records; nothing else is needed.

**Attribution** (`Metering.attribute events`). Each record is keyed by (`node` of the line, `component` attribute or `""`, `stage` attribute as text or `""`, `layer` attribute or none). Components set these with `Telemetry.scope` or `Child`. Per key:

| Total | Sum of |
| --- | --- |
| `WallNs` | the self time of `span` records: duration minus the durations of the direct child spans, at least 0. Self times of a single-threaded run add up to the wall time of its root spans, with no double counting |
| `GpuNs` | the durations of `gpu` records |
| `Bytes`, `Flops`, `Tokens`, `MeterNs` | `meter` records with unit `B`, `flop`, `token` and `ns` |
| `Records` | the span, gpu and meter records that contributed |

Rows are sorted by node, component, stage and layer (ordinal; no layer first). `Metering.rollup project rows` sums under a coarser key, for example per component with `{ k with Layer = None }`. `Metering.arenas events` gives the peak and last `mem.arena` value per (node, arena), in record order.

## FLOP registry

`OpCost.ofOp` gives the cost of one operation as exact integers, so the CPU reference, the CPU device and both GPU backends meter the same work identically. Conventions: a multiply-add is 2 FLOPs; exp, rsqrt, division and comparison are 1 each. Bytes are the minimum traffic: every input element read once and every output element written once. Activations are fp32.

| Op (registry name) | FLOPs | Bytes read | Bytes written |
| --- | --- | --- | --- |
| `MatMul(m, n, k, act, w, out)` (`matmul`) | 2mnk | mk·size(act) + nk·size(w) | mn·size(out) |
| `RmsNorm(rows, cols, w)` (`rmsnorm`) | 4·rows·cols + 2·rows | 4·rows·cols + cols·size(w) | 4·rows·cols |
| `Rope(rows, heads, d)` (`rope`) | 3·rows·heads·d | 4·rows·heads·d + 4·rows·d | 4·rows·heads·d |
| `Attention(rows, past, heads, kv, d)` (`attention`) | heads·P·(4d + 3), P = rows·past + rows(rows+1)/2 | 4·rows·heads·d + 8·kv·(past+rows)·d | 4·rows·heads·d |
| `SiluMul(n)` (`silu_mul`) | 5n | 8n | 4n |
| `Add(n)` (`add`) | n | 8n | 4n |
| `Embed(rows, cols, w)` (`embed`) | 0 | rows·cols·size(w) + 4·rows | 4·rows·cols |
| `CopyRows(rows, cols, t)` (`copy_rows`) | 0 | rows·cols·size(t) | rows·cols·size(t) |
| `Argmax(rows, cols)` (`argmax`) | rows·cols | 4·rows·cols | 4·rows |

`OpCost.layerOps dims rows past` lists one Qwen3 decoder layer's operations, each named by the ActivationNames tap it produces (`input_layernorm`, `q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`, `q_rope`, `k_rope`, `attention`, `o_proj`, `resid_mid`, `post_attention_layernorm`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.silu_mul`, `mlp.down_proj`, `out`). `OpCost.forwardOps dims rows past headRows` adds `embed`, every layer as `L<i>.<tap>`, `final_norm`, and `lm_head` and `argmax` on `headRows` rows. `TransformerDims.qwen3_0_6b` holds hidden 1024, 16 heads, 8 KV heads, head dim 128, intermediate 3072, 28 layers, vocabulary 151,936, bf16 weights.

**Vectors.**

| Operation | FLOPs | Read | Written |
| --- | ---: | ---: | ---: |
| MatMul(1, 1024, 1024, f32, bf16, f32) | 2,097,152 | 2,101,248 | 4,096 |
| RmsNorm(1, 1024, bf16) | 4,098 | 6,144 | 4,096 |
| RmsNorm(16, 128, bf16) | 8,224 | 8,448 | 8,192 |
| Rope(1, 16, 128) | 6,144 | 8,704 | 8,192 |
| Attention(1, 0, 16, 8, 128) | 8,240 | 16,384 | 8,192 |
| Attention(4, 10, 16, 8, 128) | 412,000 | 147,456 | 32,768 |
| SiluMul(3072) | 15,360 | 24,576 | 12,288 |
| Argmax(1, 151936) | 151,936 | 607,744 | 4 |
| One Qwen3-0.6B layer, rows 1, past 0 | 31,512,676 | | |
| A Qwen3-0.6B decode step at position 0 (rows 1, past 0, head rows 1) | 1,193,675,890 | | |

## Names this contract adds

The `run.*` prefix is reserved for this contract in the TelemetryFacade contract (`TelemetryNames.RunPrefix`, owner T1), so no component can emit a record that the reader would take for a structural one.

| Name | Kind | Emitted by |
| --- | --- | --- |
| `run.node`, `run.end` | event | `RunContext` only: the header and trailer of every events file |
| `run.clock_offset` | event | `RunContext.RecordClockOffset`, on the creating node only |
| `report.render` | span | `Lrm.Reports`, one per rendered report |
| `telemetry.*` | gauge | Telemetry's own measurements (bench tier) |

## Verification

The cpu tier of `Lrm.Telemetry.Tests` checks:
- vectors E1 to E5 (also under de-DE), M1 and R1, byte for byte, and that E1 to E5 decode back;
- that 1,000 random records of every kind and attribute type, and every float bit pattern (FsCheck, fixed seed), round-trip through a line, NaN staying NaN;
- that 15 kinds of malformed line are rejected with a reason;
- manifest round trip, and the rejection of another schema, an unknown status and a class id that does not match its fields;
- the execution class and driver in the manifest and the node header, joined nodes, and that folders and events files are never overwritten;
- an aborted run, 8 threads writing 16,000 records with gapless `seq`, and a sink that never throws (invalid names, writes after close, a failing stream);
- that the reader reports a cut-off line, a `seq` gap, a malformed line and a missing trailer, and keeps the good records;
- that a surrogate pair at any chunk boundary of the sink's buffer, after any earlier line, is written as the line's own UTF-8;
- that a file cut inside a multi-byte character, or holding a stray byte, reads with a per-line problem instead of an exception;
- that malformed-JSON reasons are fixed words, the same under de-DE and tr-TR;
- that clock offsets put a joined node on the creator's clock in `load` and `stream`, that only the creator records them, and that a node on another host joins a copy of the run folder whose collected events file reads back with its `restart_of`;
- the FLOP registry vectors, attribution with self time, GPU time and every meter unit, arena peaks, and `Metering.op`;
- that telemetry on or off changes no value, and that the same records on a virtual clock give byte-identical files under de-DE.

The bench tier measures the sink's cost per record and the read speed. Throughput is measured there only.
