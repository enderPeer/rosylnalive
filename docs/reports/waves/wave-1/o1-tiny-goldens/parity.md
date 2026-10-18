# Parity: make_tiny (tools/goldens)


**PASS.** All 106 comparisons are within their bounds.

- Comparisons: 106, failed: 0
- Pairs: oracle-selftest, torch32-vs-torch64p; cases: 4
- Closest to its bound: oracle-selftest   rel_l2: 0.118 of the bound

## Error by layer

Every per-layer tap's rel_l2 against the golden, summarized per layer over all cases and taps. A pair passes when every dot and whisker stays under its bound line.

![rel_l2 by layer](charts/parity-rel_l2_by_layer.svg)

![Distance to the bound, every comparison](charts/parity-ratio_histogram.svg)

## Per pair

### oracle-selftest: PASS

3 comparisons, 0 failed, worst ratio 0.118 of the bound.

![Worst measured/bound, oracle-selftest](charts/parity-pairs-0-heatmap.svg)

### torch32-vs-torch64p: PASS

103 comparisons, 0 failed, worst ratio n/a of the bound.

![Worst measured/bound, torch32-vs-torch64p](charts/parity-pairs-1-heatmap.svg)

## Per case

| Case | Checks | Failed | Worst comparison | Worst ratio | Logits rel_l2 | Greedy match |
| --- | ---: | ---: | --- | ---: | --- | --- |
|  | 27 | 0 | oracle-selftest  rel_l2 | 0.118 | n/a | n/a |
| p0 | 39 | 0 |  |  | torch32-vs-torch64p 5.44e-07 | n/a |
| p1 | 39 | 0 |  |  | torch32-vs-torch64p 1.92e-07 | n/a |
| p2 | 1 | 0 |  |  | torch32-vs-torch64p 6.69e-07 | n/a |

## Closest to the bound

| Pair | Case | Tap | Metric | Measured | Bound | Ratio | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| oracle-selftest |  |  | rel_l2 | 1.18e-16 | 1.00e-15 | 0.118 | ✓ pass |
| oracle-selftest |  |  | rel_l2 | 1.59e-15 | 1.00e-13 | 0.016 | ✓ pass |
| oracle-selftest |  |  | rel_l2 | 4.54e-16 | 1.00e-13 | 0.005 | ✓ pass |

## Run

| Field | Value |
| --- | --- |
| Run | `20261018-032001-6e21cd70` |
| Title | make_tiny (tools/goldens) |
| Kind | parity |
| Status | ok |
| Started | 2026-10-18T03:20:01.323Z |
| Ended | 2026-10-18T03:20:07.640Z |
| Execution class | not recorded |
| Class id | n/a |
| Driver | n/a |
| Commit | `ad5d24292344ad2c1bf262fa0f2f3a5c2781bef0` |
| Nodes | 1 |
| Records | 143 |
| Synthetic | no |

| Parameter | Value |
| --- | --- |
| tool | make_tiny |

### Nodes

| Node | Role | Execution class | Records | Finished | Dropped |
| --- | --- | --- | ---: | --- | ---: |
| n0 | creator | not recorded | 143 | ok | 0 |

### Log problems

None.
