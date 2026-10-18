# ComparatorsAndGoldenReader contract

Owner: O2. Status: v1, frozen at the contracts-v1 tag. Consumers: every check against a golden or a reference, among them M2 (tiny and full-model parity), E1 and KernelBounds, X1, C2, G2's tools on the F# side, I1 (`lrm bisect`, gate reports) and T1 (reports chart `golden.compare`). Code: `src/Lrm.Tensor/Compare.fs` and `Goldens.fs`. Tests: `tests/Lrm.Tensor.Tests/CompareTests.fs` and `GoldensTests.fs` (cpu tier).

After contracts-v1, this text and its vectors change only through a `contract-change` PR, with a CHANGELOG entry and updated vectors. The file layout that `Goldens.Open` reads is GoldenSet (`golden-set.md`). The bounds that comparisons use come from ToleranceSpec (`tolerance-spec.md`) and KernelBounds (`kernel-bounds.md`).

## Metrics

`Metrics` holds the metrics as pure float64 functions. Their rules:
- **Order of arguments.** The candidate comes first and the reference second.
- **Determinism.** Every sum runs sequentially in increasing index order, in float64, so a metric is one exact number on every machine.
- **Widening.** Inputs of other dtypes are widened exactly first (`IHostTensor.ToFloat64`).

| Metric | Definition | Edge cases |
| --- | --- | --- |
| `relL2 a b` | `sqrt(Σ (a−b)²) / sqrt(Σ b²)` | 0 when both are all zero; +∞ when only `b` is; NaN when either holds a NaN, or infinities cancel |
| `maxAbs a b` | `max |a − b|` | 0 for empty input; NaN when any difference is NaN |
| `ulpDistance32`, `ulpDistance64`, `ulpDistanceBf16` | The distance between two values on the ordered integer line of their format: 1 for neighbours | 0 for +0 and −0; +∞ when either is NaN |
| `maxUlp32`, `maxUlp64` | The largest ulp distance | |
| `cosine a b` | `Σ ab / (sqrt(Σ a²) · sqrt(Σ b²))` | 1 when both are all zero; 0 when exactly one is |
| `logSoftmax row` | `x − (m + log Σ exp(x − m))`, with `m` the row maximum | A NaN makes the whole row NaN |
| `klRow cand ref` | `KL(P_ref ‖ P_cand) = Σ p_i (log p_i − log q_i)` over the softmaxes of two logit rows; terms with `p_i = 0` are skipped | NaN when either row holds a NaN |
| `topK row k` | The indices of the k largest values, largest first, equal values in index order | A NaN raises |
| `topKAgreementRow cand ref k` | `|topK(cand) ∩ topK(ref)| / k` | |
| `top2MarginRow row` | top-1 value minus top-2 value | Needs 2 or more elements; a NaN raises |

## Checks

`Compare.*` runs one metric over two tensors of equal shape and returns a `CheckResult`. A shape mismatch raises `ArgumentException`. Row metrics treat a tensor as `[Rows, Cols]`.

```fsharp
type CheckResult =
    { Metric: string; Measured: float; Bound: float; Op: BoundOp
      Passed: bool; Elements: int64; Attrs: Attr list }
```

| Function | Metric | Measured | Passes when |
| --- | --- | --- | --- |
| `Compare.relL2 bound attrs cand ref` | `relL2` | relL2 | ≤ bound |
| `Compare.maxAbs bound …` | `maxAbs` | max \|Δ\| | ≤ bound |
| `Compare.maxUlp bound …` | `maxUlp` | the largest ulp distance, in the common dtype (F32, F64 or BF16; other dtypes or mixed dtypes raise) | ≤ bound |
| `Compare.cosine minimum …` | `cosine` | cosine | ≥ minimum |
| `Compare.kl bound …` | `kl` | the largest per-row KL | ≤ bound |
| `Compare.topKAgreement k minimum …` | `topKAgreement` | the smallest per-row agreement | ≥ minimum |
| `Compare.top2Margin minimum attrs logits` | `top2Margin` | the smallest per-row margin of one tensor | ≥ minimum |
| `Compare.exact attrs cand ref` | `exact` | the number of elements whose bits differ (same dtype required) | ≤ 0 |
| `Compare.check metric op measured bound elements attrs` | any | a value computed elsewhere | by `op` |

**A NaN measurement never passes**, whatever the direction.

### Telemetry: golden.compare

Every check emits one `golden.compare` Check through the ambient facade (TelemetryFacade). Its value is 1 for a pass and 0 for a failure. The attributes are, in this order:
1. `metric` (string), `measured` (float), `bound` (float), `direction` (`le` or `ge`) and `elements` (int);
2. then the caller's `attrs`.

Callers use these conventional keys, so reports can group results:

| Key | Meaning | Examples |
| --- | --- | --- |
| `pair` | The ToleranceSpec pair name | `cpu64-vs-torch64p`, `cpu32-vs-torch32`, `gpu-vs-torch32` |
| `tap` | The ActivationNames tap | `L1.attn.q_rope`, `logits` |
| `layer` | The tap's layer, or −1 for model-level taps (`ActivationNames.layer`) | `1` |
| `prompt` | The golden prompt name | `p0` |
| `op` | The op under test, for per-op checks | `matmul`, `attention` |
| `kind` | `op` (an op fed golden inputs), `chain` (a full forward), `measurement` (an e_ref) | |

The same record shape is what O1's golden tools emit for their self-tests and e_ref measurements, so T1's parity report charts them all from one name.

## Goldens.Open

`Goldens.Open(dir, ?dataRoot)` opens `goldens/<set>/g<N>/` and validates its `manifest.json` before returning a `GoldenSet`. Any violation raises `GoldenFormatException`:

1. `schema` is `lrm.golden/1`. Unknown schemas are refused.
2. `set` and `version` equal the folder names, and the version is `g<N>` with N ≥ 1 and no leading zero.
3. `files` is an array of objects, each with `path`, `location`, `bytes` (a non-negative integer), `sha256` (64 lowercase hex digits) and `role`.
4. **Paths stay inside the set.** A `repo` path is `goldens/<set>/<version>/<name>`, and a `data` path is `<set>/<version>/<name>` under `LRM_DATA/golden`. The name has no empty, `.` or `..` segment, no `\` and no `:`, and it is never `manifest.json`.
5. Paths are unique and sorted in code point order, and no repo file shares a name with a data file.

Fields the reader does not know are ignored, so G2 can add its own. `GoldenSet.Manifest` exposes the whole manifest as JSON.

**Files are verified before use.** The verified bytes are the bytes used:

- **`OpenSafetensors name`** opens the file with `SafetensorsFile.OpenVerified`, so the size and the SHA-256 are checked over the mapped bytes before the header is parsed, and nobody can write the file while it is open (HostTensorAndSafetensors).
- **`ReadAllBytes name`** reads the whole file, checks its size and SHA-256, and returns those same bytes.
- **`VerifyAll()`** checks every listed file.
- **Results.** A mismatch raises `IntegrityException`, which names the file and the expected and actual value. Every verification emits one `goldens.verify` Check with `set`, `file`, `bytes` and `sha256`.
- **Data files** resolve under `dataRoot/golden`. The default data root is `LRM_DATA` (`LrmPaths.golden`). A data file is touched only when it is read, so a set whose large files live elsewhere still opens on CI.

`Goldens.TinyDir repoRoot` is `goldens/tiny/g1`.

## ActivationNames helpers

The ActivationNames contract (O1) defines the names. `ActivationNames` in Lrm.Tensor follows it:
- `all layers` gives the 18·L + 3 names in canonical order;
- `layer name` gives the layer of `L{i}.…`, or −1 for model-level taps;
- `kind name` gives the part after `L{i}.`, or the whole model-level name. This is the key ToleranceSpec bounds use.

## Vectors

Each value below was computed independently with Python floats or math, and the tests check it:

| Case | Result |
| --- | --- |
| `relL2((1, 2, 3), (1, 2, 4))` | `1/√21` = 0.2182178902359924 |
| `relL2(0, 0)`; `relL2(1, 0)` | 0; +∞ |
| `maxAbs((1, −2, 3), (1.5, −2, 2))` | 1 |
| `ulpDistance32(1, nextUp 1)`; `(+0, −0)`; `(ε, −ε)` (the smallest subnormals) | 1; 0; 2 |
| `ulpDistanceBf16(3f80, 3f82)` | 2 |
| `klRow((0, ln 3), (0, 0))`: P = (½, ½), Q = (¼, ¾) | ½ ln(4/3) = 0.14384103622589042 |
| `topK((1, 3, 3, 2), 3)` | (1, 2, 3) |
| `topKAgreementRow((5, 4, 0), (5, 0, 4), 2)` | 0.5 |
| `top2MarginRow((1, 4, 2.5))` | 1.5 |

**The tiny set** `goldens/tiny/g1` opens as `tiny`/`g1`/`tiny-model` with 15 files, and every file verifies. Its tap names equal `ActivationNames.all 2` (39 names).

## Verification

The cpu tier of `Lrm.Tensor.Tests` covers:
- **The metric vectors** above, including the NaN and zero cases.
- **Every Compare function:** its pass and fail sides, the CheckResult fields, and one `golden.compare` record per check with metric, measured, bound and the caller's attributes, checked with the strict InMemorySink.
- **Edge cases:** NaN measurements fail in both directions; maxUlp refuses mixed dtypes.
- **A property:** relL2 is 0 for identical inputs and unchanged under power-of-two scaling.
- **`Goldens.Open`:**
  - the tiny set and `VerifyAll`, with 15 passing `goldens.verify` records;
  - one flipped byte and one appended byte, each caught with `IntegrityException` and a failing `goldens.verify`;
  - data-file resolution under a given data root;
  - ten malformed manifests: schema, set, version, order, an escaping path, another set's path, the manifest itself, an unknown location, an uppercase hash and a negative size.
