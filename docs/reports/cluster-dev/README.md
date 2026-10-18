# Pipeline cluster reports (C2)

The C2 family's reports (StagePipeline, `docs/contracts/stage-pipeline.md`), rendered by `lrm report` from real multi-process runs on localhost. The Markdown and SVG files are never edited by hand; this index is.

Each run is the tiny model's standard job (p0, p1 and p2 of `goldens/tiny/g1`, 16 greedy steps each, split at layer 1, cpu-f32: 96 work units) on `lrm node` processes behind an `lrm chaos-proxy` process, with the coordinator in the `lrm coord local` process, which created the run log; every node process joined it. Each run was compared with both oracles, and the four comparisons are the `golden.compare` checks that `parity.md` lists as self-tests: all four pass in every run (run digest `e3ce3176…2844`, logits digest `a5c08eb2…79bc`, the plan-order and single-process values).

| Report | Run | What happened |
| --- | --- | --- |
| [tiny-split2/cluster.md](tiny-split2/cluster.md) ([parity](tiny-split2/parity.md)) | `20261018-154112-e8cca8ca` | Two nodes, one stage each, one replica per stage. n2 (stage 1) killed after 40 units and restarted 400 ms later as a new incarnation (`n2-r1` in the log), which rebuilt its three chains by replay; n1's connection severed at 1.5 s. Stage 1's units waited for the restart (`cluster.reissue.unavailable`) |
| [tiny-replicated3/cluster.md](tiny-replicated3/cluster.md) ([parity](tiny-replicated3/parity.md)) | `20261018-154116-32c002b6` | Three nodes hosting both stages, two replicas per stage. n1 killed after 40 units and never restarted: its stages failed over to the surviving replicas with no catch-up, and the refilled replicas replayed. Every unit cross-checked by a second replica |
| [tiny-bothlost5/cluster.md](tiny-bothlost5/cluster.md) ([parity](tiny-bothlost5/parity.md)) | `20261018-154120-1dd7d965` | n1 and n2 host stage 0, n3 and n4 stage 1, the spare n5 both (started after the first unit). n3 killed after 20 units and n4 after 30: both stage-1 replicas lost; n5 took stage 1 over by replay, and n4 came back after 300 ms as a new incarnation (`n4-r1` in the log), a second stage-1 replica again. n1 partitioned from 1.2 to 1.7 s |

To make them again from a Release build (the run ids change):

```bash
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll coord local --topology split2 --seed 101 --kill 2:40:400 --sever 1:1500 \
  --run-log --title "Tiny model on two node processes behind lrm chaos-proxy: n2 (stage 1) killed after 40 units and restarted (C2a)"
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll coord local --topology replicated3 --seed 102 --kill 1:40 \
  --run-log --title "Tiny model on three node processes, two replicas per stage: n1 killed after 40 units, failover (C2a)"
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll coord local --topology bothlost5 --seed 103 --kill 3:20 --kill 4:30:300 \
  --partition 1:1200:1700 --run-log \
  --title "Tiny model on five node processes: both stage-1 replicas killed, the spare n5 takes over, n4 restarted (C2a)"
# each prints "run log <dir> (lrm report <runId>)"
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <runId> --out docs/reports/cluster-dev/<name>
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <runId> --template parity --out docs/reports/cluster-dev/<name>
```

Reading them:

- The digests and ids are deterministic within the class; the times are host times on the shared development machine and informative only. Each unit sleeps 2 ms on its node (`--cost-ms 2`) so that the faults land mid-run; on Windows a 2 ms sleep lasts one timer tick (about 15.6 ms), which is most of the 30 to 50 ms per unit the Gantt charts show.
- A killed process writes no `run.end` trailer and may leave a cut-off last line, so each report lists those as log problems of the killed nodes; nothing else is wrong with the logs.
- The cluster template's PASS sentence says every unit was cross-checked by a second replica. The split2 run has one replica per stage by design, so its table says 0 of 96: the sentence does not apply there (a wording request to T1).
- The parity reports hold only the four self-tests; their error-by-layer charts are empty because a cluster run compares no taps.
