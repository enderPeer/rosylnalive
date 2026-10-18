# ToleranceSpec contract

Owner: O2 owns the types, the file schema and the loader. G2 owns the numbers in `goldens/qwen3-0.6b/g1/tolerances.json`. Status: v1, frozen at the contracts-v1 tag. Consumers: M2b (gate1-cpu), X1b, I1, I2 and I3 (every model-level check), and T1 (the bounds in parity reports). Code: `src/Lrm.Tensor/ToleranceSpec.fs`. Tests: `tests/Lrm.Tensor.Tests/ToleranceTests.fs` (cpu tier).

After contracts-v1, the schema, the pair and metric names, and the calibration rule change only through a `contract-change` PR. G2 changes numbers by writing a new golden version, `g<N+1>`. Per-kernel bounds are KernelBounds (`kernel-bounds.md`), not this contract.

## Pairs, metrics and tap kinds

A **pair** names a candidate and a reference. Its name is frozen; it is also the `pair` attribute of `golden.compare` records.

| Pair | Candidate | Reference |
| --- | --- | --- |
| `cpu64-vs-torch64p` | CpuRef F64 | torch64p |
| `cpu32-vs-torch32` | CpuRef F32 | torch32 |
| `cpu32-vs-torch64p` | CpuRef F32 | torch64p |
| `gpu-vs-cpu32` | a GPU backend | CpuRef F32 |
| `gpu-vs-torch32` | a GPU backend | torch32 |
| `gpu-vs-torch64p` | a GPU backend | torch64p |
| `vulkan-vs-d3d12` | Vulkan | D3D12 |

A **metric** is one of three comparators (ComparatorsAndGoldenReader):

| Metric | Comparator |
| --- | --- |
| `relL2` | relL2 per tap |
| `maxAbs` | max \|Δ\|; for logits, max \|Δlogit\| |
| `kl` | the largest per-row KL(P_reference ‖ P_candidate) of the logits |

A **tap kind** is `ActivationNames.kind`: the part of a tap name after `L{i}.`, or the whole model-level name. A bound for `attn.q_rope` therefore holds for every layer.

## The calibration rule

```
e_ref(m)  = m(torch32, torch64p)               the metric m between the two oracles, per tap (or per tensor)
bound     = clamp(k * e_ref, floor, 10 * floor) = min(max(k * e_ref, floor), 10 * floor), in float64
```

- **Exact equality.** `ToleranceSpec.calibrate k floor eRef` is that expression. A stored bound must equal it exactly: G2 computes it the same IEEE way and writes it as the shortest round-trip float. For example, 3 · 2e-05 is `6.000000000000001e-05`, not `6e-05`.
- **The plan's starting values** (dev plan, section 5) are given below. G2 calibrates the final values and records each one in the file:

  | Pair | k | relL2 floor | maxAbs floor | kl floor |
  | --- | ---: | ---: | ---: | ---: |
  | `cpu32-vs-torch64p` | 2 | 1e-5 | 2e-3 | 1e-7 |
  | `cpu32-vs-torch32` | 3 | 1e-5 | 2e-3 | 1e-7 |
  | `gpu-vs-torch64p` | 9 | 1e-4 | 1e-2 | 1e-6 |
  | `gpu-vs-torch32` | 10 | 1e-4 | 1e-2 | 1e-6 |

- **Fixed bounds** have no rule:
  - `cpu64-vs-torch64p`: relL2 ≤ 1e-10 on the full model, and 1e-12 on the tiny model (O2's own acceptance);
  - `vulkan-vs-d3d12`: relL2 ≤ 1e-4 and maxAbs ≤ 1e-2.
- **`gpu-vs-cpu32`** is calibrated by G2 under the same rule, or left out.

## The file: lrm.tolerances/1

```json
{
  "schema": "lrm.tolerances/1",
  "set": "qwen3-0.6b/g1",
  "pairs": {
    "cpu32-vs-torch32": {
      "rule": { "k": 3, "floors": { "relL2": 1e-05, "maxAbs": 0.002, "kl": 1e-07 } },
      "taps": {
        "attn.q_rope": { "relL2": { "bound": 1e-05, "e_ref": 1.6e-07 } },
        "logits": {
          "relL2": { "bound": 0.0001, "e_ref": 0.001 },
          "maxAbs": { "bound": 0.003, "e_ref": 0.001 },
          "kl": { "bound": 1e-07, "e_ref": 1e-09 }
        }
      }
    },
    "cpu64-vs-torch64p": { "default": { "relL2": { "bound": 1e-10 } } },
    "vulkan-vs-d3d12": { "default": { "relL2": { "bound": 0.0001 }, "maxAbs": { "bound": 0.01 } } }
  },
  "greedy": { "...": "G2's gate rules, kept as JSON" }
}
```

**Fields.** Readers ignore unknown top-level fields; inside `pairs` every field is known, and a `rule` holds only `k` and `floors`.

| Field | Meaning |
| --- | --- |
| `schema` | `lrm.tolerances/1` |
| `set` | The golden set the numbers belong to, for example `qwen3-0.6b/g1` |
| `pairs.<pair>.rule` | Optional: `k` and `floors` per metric. With a rule, **every** bound of the pair must carry its `e_ref` and equal the calibration |
| `pairs.<pair>.default` | Optional: bounds per metric for tap kinds without an entry of their own |
| `pairs.<pair>.taps.<kind>.<metric>` | `{ "bound": b, "e_ref": e }`. `e_ref` is optional for pairs without a rule, where it is informative |
| `greedy` | G2's greedy and gate rules, such as the minimum margin `max(0.05, 4·τ_logit)`, the 16-step prefix fallback, and the teacher-forcing threshold `2·τ_logit`. The loader keeps them as raw JSON |

## Loading and lookup

`ToleranceSpec.load path` (or `parse json`) returns a `ToleranceSpec`. It raises `ToleranceFormatException` for any of the following:
- invalid JSON, or a root that is not an object;
- a duplicate key anywhere: at the top level, inside `pairs`, inside `greedy`, inside an ignored unknown field, and inside arrays. JSON parsers disagree on duplicates (System.Text.Json's `TryGetProperty` returns the last one), so a second `pairs` object would otherwise silently drop every bound of the first;
- text that is not valid Unicode: a lone surrogate, written as an escape such as `\ud800` or present in the string passed to `parse`;
- a schema other than `lrm.tolerances/1`;
- an unknown pair, metric or field, including a field of `rule` other than `k` and `floors`;
- a bound that is not positive and finite, or an `e_ref` that is negative or not finite;
- in a pair with a rule: a missing floor for a metric, a bound without `e_ref`, or a bound that is not exactly `clamp(k · e_ref, floor, 10 · floor)`.

**Lookup.** `spec.TryBound(pair, tapKind, metric)` returns the tap kind's own entry, else the pair's default, else None. `spec.Bound` returns the value, or raises `KeyNotFoundException`. Each entry keeps its `ERef`, so reports can show the bound next to the oracle distance it came from.

## Vectors

| Input | Result |
| --- | --- |
| `calibrate 3 1e-5 1.6e-7` | 1e-5 (the floor) |
| `calibrate 3 1e-5 2e-5` | 6.000000000000001e-05 |
| `calibrate 3 1e-5 1e-3` | 1e-4 (the ceiling, 10 · floor) |
| `calibrate 9 1e-4 5e-5` | 9 · 5e-5 = 0.00045000000000000004 |
| The file above: `cpu32-vs-torch32`, `attn.q_rope`, `relL2` | 1e-5 |
| The same file: `cpu64-vs-torch64p`, any tap, `relL2` | 1e-10 (the default) |
| The same file: `cpu64-vs-torch64p`, `logits`, `kl` | none |

## Verification

The cpu tier checks:
- the calibration vectors;
- the sample file's lookups, defaults and `e_ref` values;
- the frozen pair and metric names;
- one refusal for each rule above:
  - the schema, a bound off its rule, a calibrated bound without `e_ref`;
  - an unknown pair, an unknown metric, an unknown field in a rule, a zero bound;
  - duplicate keys: a pair, `schema`, a second `pairs`, inside `greedy` (in an array) and inside an unknown top-level field;
  - a lone surrogate, as an escape and in the text;
  - a root that is not an object, and invalid JSON;
- that unknown top-level fields are ignored.
