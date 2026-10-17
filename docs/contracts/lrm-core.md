# LrmCore contract

Owner: S1. Status: v1, frozen at the contracts-v1 tag. Consumers: every project. Code: `src/Lrm.Core`. Tests: `tests/Lrm.Core.Tests` (cpu tier).

After contracts-v1, this text and its vectors change only through a `contract-change` PR, with a CHANGELOG entry and updated vectors. The telemetry facade and the ExecutionClass type also live in Lrm.Core; their texts are `telemetry-facade.md` and `execution-class.md`.

## Contents

| Module | What it provides |
| --- | --- |
| `ContentId`, `Hash` | 32-byte content addresses; SHA-256 of spans, arrays, UTF-8 text, streams and files; incremental hashing |
| `CanonicalWriter`, `CanonicalReader`, `Canonical` | The canonical encoding, and `Canonical.hash`, the domain-separated hash |
| `DType` | Element-type codes, names and sizes |
| `Bf16`, `F16` | bfloat16 and binary16 conversions |
| `SplitMix64`, `Philox4x32`, `DetRng` | Deterministic random numbers |
| `SafeName`, `RunId`, `NodeId` | Path-safe identifiers |
| `Invariant` | Culture-independent formatting and parsing |
| `LrmPaths` | Locations under `LRM_DATA` |
| `Process` | `Process.init`, which every exe calls first |

## ContentId and Hash

- **SHA-256 is the only hash** in LRM.
- **ContentId** is 32 bytes.
  - Equality is byte equality, and ordering is lexicographic over the bytes.
  - The text form (`ToString`, `Hex`) is 64 lowercase hex digits. `ContentId.parse` accepts either case.
  - `Short` is the first 12 hex digits, for logs and reports only. Never use it as a key.
  - `ContentId.zero` (all zero bytes) means "none", for example a first unit's predecessor. It is never the hash of real content.
- **Hash** offers `Hash.sha256` (array), `sha256Span`, `sha256Utf8` (strict UTF-8; a lone surrogate raises), `sha256Stream`, `sha256File` and `Hash.Incremental`.

## Canonical encoding

The canonical encoding is how structured values become bytes for hashing and content addressing. It is positional: a value is its fields in declared order, with no field names, type tags or padding. The schema is fixed by the encoding function together with the tag (below).

| Value | Encoding |
| --- | --- |
| `u8` | 1 byte |
| `bool` | `0x00` or `0x01` |
| `u16`, `u32`, `u64` | little-endian |
| `i32`, `i64` | little-endian two's complement |
| `f32`, `f64` | the raw IEEE 754 bits, little-endian. Negative zero and every NaN payload are kept; nothing is canonicalized |
| bytes | `u32` length, then the bytes |
| string | `u32` byte length, then strict UTF-8 with no BOM. A lone surrogate raises `ArgumentException`. Text is not normalized: callers that need NFC normalize first |
| `ContentId` | the 32 raw bytes, with no length prefix |
| option | `0x00` for None, or `0x01` followed by the value |
| sequence | `u32` count, then the items in order |
| map with string keys | `u32` count, then (key as a string, value) entries sorted by the UTF-8 bytes of the key, which is Unicode code point order. Duplicate keys raise |
| `DType` | its code as one `u8` |
| record | its fields in declared order |

- **Map order is code point order, not UTF-16 order.** The two differ for astral characters. For example, U+FF61 sorts before U+1F600 here, but after it under .NET's `String.CompareOrdinal`.
- **Lengths and counts are `u32`.** A blob too large for that is hashed on its own and embedded as a ContentId.
- **No time encoder.** There is no encoder for times, dates or durations, so timestamps cannot enter a hash (TelemetryFacade contract). A test checks this by reflection.
- **The reader accepts only canonical input.** `CanonicalReader` raises `FormatException` for any of these:
  - a bool or option tag other than `0x00` or `0x01`;
  - invalid UTF-8;
  - map keys that are unsorted or duplicated;
  - truncated input;
  - trailing bytes, when `ExpectEnd` is called.

### Domain-separated hash

`Canonical.hash tag write` = SHA-256(ASCII(tag) || `0x00` || payload), where payload is the canonical encoding written by `write`. `Canonical.hashBytes tag payload` does the same for bytes that are already encoded.

- **Tags.** A tag is 1 to 128 printable ASCII characters (0x21 to 0x7E, so no spaces). By convention it is `lrm.<thing>/<version>`.
- **Versioning.** A change to a value's encoding changes the version in its tag, so old and new ids never collide.
- **The separator.** The `0x00` byte keeps tag and payload apart: (tag `ab`, payload `c`) and (tag `a`, payload `bc`) hash differently.

| Tag | Owner | Used for |
| --- | --- | --- |
| `lrm.xclass/1` | S1 | ExecutionClassId (`execution-class.md`) |
| `lrm.rng.stream/1` | S1 | `DetRng.streamOf` |
| `lrm.test/1` | S1 | Test vectors only |
| `lrm.unit/1` | C1 | WorkUnit ids (ClusterProtocol) |

Other owners register their tags in their own contracts.

### Vectors

These hashes were computed independently with `sha256sum` over the hex bytes shown, and the cpu tier checks the F# encoder against them.

**C1: empty payload.** SHA-256(`lrm.test/1` || 00) = `7390733e381c00f7903c802456310b720fdec8109fe53d67f28df45d3db9aad4`.

**C2: one value of every kind.** The table lists the writes in order and the bytes each one produces.

| Write | Bytes (hex) |
| --- | --- |
| `U8 0x2A` | `2a` |
| `Bool true` | `01` |
| `U16 0x1234` | `3412` |
| `U32 0xDEADBEEF` | `efbeadde` |
| `I32 -2` | `feffffff` |
| `U64 0x0123456789ABCDEF` | `efcdab8967452301` |
| `I64 -1` | `ffffffffffffffff` |
| `F32 1.5` | `0000c03f` |
| `F64 -0.0` | `0000000000000080` |
| `String "h" U+00E9 "llo"` | `06000000 68c3a96c6c6f` |
| `Bytes [1; 2; 3]` | `03000000 010203` |
| `Option None` | `00` |
| `Option (Some 7u)` | `01 07000000` |
| `Seq [1u; 2u]` | `02000000 01000000 02000000` |
| `Map [b: 2uy; a: 1uy]` | `02000000 01000000 61 01 01000000 62 02` |
| `Id SHA-256("abc")` | `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad` |

The payload is 123 bytes. `Canonical.hash "lrm.test/1"` over it is `bcaeed69c2d44298c10172d350032c34ab1e5eb45e12051ba7c3869a4933b23e`.

**C3: map key order.** `Map [U+1F600: 2uy; U+FF61: 1uy]` encodes as `02000000 03000000 efbda1 01 04000000 f09f9880 02`. Its `lrm.test/1` hash is `6658a5c6133053fc57c29797182eae1f2fb6f773c658691de0cf468dad570edb`.

## DType

The codes are frozen. They appear in the GPU tensor table, in canonical encodings and in hashes. Code 0 is invalid, so zeroed memory never reads as a valid type.

| DType | Code | Name | safetensors | Bytes |
| --- | ---: | --- | --- | ---: |
| (invalid) | 0 | | | |
| F32 | 1 | `f32` | `F32` | 4 |
| BF16 | 2 | `bf16` | `BF16` | 2 |
| F16 | 3 | `f16` | `F16` | 2 |
| I32 | 4 | `i32` | `I32` | 4 |
| U32 | 5 | `u32` | `U32` | 4 |
| U8 | 6 | `u8` | `U8` | 1 |
| F64 | 7 | `f64` | `F64` | 8 |
| I64 | 8 | `i64` | `I64` | 8 |

- **Names are matched ordinally.** `DType.tryParse` takes the lowercase LRM name, and `DType.tryOfSafetensorsName` takes the uppercase safetensors string. Types LRM does not use, such as `BOOL`, give None.
- **Unknown codes.** `DType.tryOfCode` gives None for 0 and for every unassigned code.

## bf16 and binary16

Both types are carried as their raw `uint16` bits.

- **`Bf16.widen`** is exact: the bits become the top half of a float32. NaN payloads are kept.
- **`Bf16.narrowRne`** rounds to nearest, ties to even:
  - for a non-NaN input `u`, the result is `(u + 0x7FFF + ((u >> 16) & 1)) >> 16`;
  - results beyond the largest finite bf16 become infinities;
  - subnormals are rounded like any other value, never flushed;
  - a NaN narrows to `0x7FC0`, or to `0xFFC0` when its sign bit is set, whatever its payload.
- **Round trip.** Every non-NaN bf16 round-trips through `widen` and `narrowRne`. The 254 NaN patterns cannot, by design.
- **Compared with PyTorch.** The result equals PyTorch's `c10::BFloat16` round-to-nearest-even for every non-NaN input. c10 maps every NaN to `0x7FC0`, while LRM keeps the sign.
- **`F16.widen`** is exact. Subnormals become normal float32 values. A NaN keeps its payload, shifted into the top of the float32 mantissa.
- **`F16.narrowRne`** rounds to nearest, ties to even:
  - subnormal results are kept, never flushed;
  - values of 65520 or more become infinities (65520 is the tie above 65504 and rounds to even, which is infinity);
  - a NaN narrows to `0x7E00`, or to `0xFE00` when its sign bit is set.
- **Bulk forms.** `Bf16.widenInto` and `narrowInto` work on spans, and `widenArray` and `narrowArray` on arrays.

**Verification.**
- Both `narrowRne` functions are checked against independent oracles on every one of the 2^32 float32 bit patterns:
  - bf16 against an exact-distance oracle, which compares both bf16 neighbours in float64 and breaks ties to even;
  - f16 against a float64 `Math.Round(ToEven)` oracle and against `System.Half`.
- The full sweep takes 1.1 s (bf16) and 2.0 s (f16) on 32 threads, and 2.9 s and 9.2 s with 4 processors. It therefore runs in full on CI, with no sampling by classes.
- All 65,536 patterns of each type are checked for widening and round trips.

**bf16 vectors** (float32 bits to bf16 bits):

| float32 | bf16 | Case |
| --- | --- | --- |
| `00000000` | `0000` | +0 |
| `80000000` | `8000` | -0 |
| `3f800000` | `3f80` | 1.0 |
| `3f807fff` | `3f80` | just below a tie |
| `3f808000` | `3f80` | tie, even neighbour below |
| `3f808001` | `3f81` | just above a tie |
| `3f818000` | `3f82` | tie, even neighbour above |
| `bf818000` | `bf82` | negative tie |
| `40490fdb` | `4049` | pi |
| `7f7f7fff` | `7f7f` | largest value that stays finite |
| `7f7f8000` | `7f80` | tie above the largest bf16 rounds to infinity |
| `7f7fffff` | `7f80` | FLT_MAX rounds to infinity |
| `7f800000` | `7f80` | +infinity |
| `ff800000` | `ff80` | -infinity |
| `00000001` | `0000` | smallest subnormal rounds to zero |
| `00008000` | `0000` | subnormal tie rounds to even zero |
| `00008001` | `0001` | subnormal just above the tie |
| `00018000` | `0002` | subnormal tie rounds up to even |
| `007fffff` | `0080` | largest subnormal rounds to the smallest normal |
| `7f800001` | `7fc0` | signalling NaN (truncation would give infinity) |
| `7fc00000` | `7fc0` | quiet NaN |
| `7fffffff` | `7fc0` | NaN with every payload bit set |
| `ff800001` | `ffc0` | negative signalling NaN |
| `ffffffff` | `ffc0` | negative NaN |

**binary16 vectors** (float32 bits to f16 bits):

| float32 | f16 | Case |
| --- | --- | --- |
| `3f800000` | `3c00` | 1.0 |
| `c0000000` | `c000` | -2.0 |
| `3dcccccd` | `2e66` | 0.1f |
| `477fe000` | `7bff` | 65504, the largest half |
| `477fefff` | `7bff` | just below 65520 |
| `477ff000` | `7c00` | 65520 ties to infinity |
| `7f800000` | `7c00` | +infinity |
| `38800000` | `0400` | 2^-14, the smallest normal |
| `387fc000` | `03ff` | 1023 * 2^-24, the largest subnormal |
| `387fe000` | `0400` | 1023.5 * 2^-24 ties to the smallest normal |
| `33800000` | `0001` | 2^-24, the smallest subnormal |
| `33000000` | `0000` | 2^-25 ties to even zero |
| `33000001` | `0001` | just above 2^-25 |
| `33c00000` | `0002` | 1.5 * 2^-24 ties to 2 |
| `34200000` | `0002` | 2.5 * 2^-24 ties to 2 |
| `80000000` | `8000` | -0 |
| `7f800001` | `7e00` | signalling NaN |
| `7fc00000` | `7e00` | quiet NaN |
| `ff800001` | `fe00` | negative NaN |

## Random numbers

### SplitMix64

Each call adds `0x9E3779B97F4A7C15` to the state and returns `Mix(state)`, where `Mix` is:

1. `z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9`;
2. `z = (z ^ (z >> 27)) * 0x94D049BB133111EB`;
3. return `z ^ (z >> 31)`.

This is the generator behind `java.util.SplittableRandom`. Use it for seeding and mixing. Anything that must be reproducible across shards and replays uses DetRng.

**Vectors** (the published sequences):
- Seed 1234567 gives 6457827717110365317, 3203168211198807973, 9817491932198370423, 4593380528125082431, 16408922859458223821.
- Seed 0 gives `e220a8397b1dcdaf`, then `6e789e6aa1b965f4`.

### Philox4x32

`Philox4x32.generate rounds (c0, c1, c2, c3) (k0, k1)` is Random123's `philox4x32_R`. `Philox4x32.block10` is the 10-round variant used everywhere in LRM.

- **Constants.** `M0 = 0xD2511F53` and `M1 = 0xCD9E8D57` are the multipliers; `W0 = 0x9E3779B9` and `W1 = 0xBB67AE85` are the key bumps.
- **One round.** Compute `(hi0, lo0) = M0 * c0` and `(hi1, lo1) = M1 * c2` as full 64-bit products. The output is `(hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)`.
- **Key schedule.** The first round uses the key as given. Before each later round, `(k0, k1)` grows by `(W0, W1)`, modulo 2^32.

**Vectors** (Random123 `kat_vectors`, philox4x32 10):

| Counter | Key | Output |
| --- | --- | --- |
| `00000000 00000000 00000000 00000000` | `00000000 00000000` | `6627e8d5 e169c58d bc57ac4c 9b00dbd8` |
| `ffffffff ffffffff ffffffff ffffffff` | `ffffffff ffffffff` | `408f276d 41c83b0e a20bc7c6 6d5451fd` |
| `243f6a88 85a308d3 13198a2e 03707344` | `a4093822 299f31d0` | `d16cfe09 94fdcceb 5001e420 24126ea1` |

The tests also check Random123's 7-round vectors, to exercise the round count.

### DetRng

`DetRng(seed: u64, stream: u64[, position: u64])` is Philox4x32-10 with:
- key = (seed low 32 bits, seed high 32 bits);
- counter = (block low 32 bits, block high 32 bits, stream low 32 bits, stream high 32 bits).

The sequence:
- **Draws.** Draw `i` is a 32-bit word, counted from 0: word `i mod 4` of block `i / 4`.
- **Position** is the number of words drawn so far. `DetRng(seed, stream, position)` continues a sequence at any point in O(1), so a shard or a replay on another node draws exactly the same numbers. `Block(index)` returns a whole Philox block directly.
- **Instances are not thread-safe.** Give each worker its own stream, or its own instance at a known position.

| Method | Definition | Words |
| --- | --- | ---: |
| `NextU32` | the next word | 1 |
| `NextU64` | `lo + 2^32 * hi`, where the first word drawn is `lo` | 2 |
| `NextF32` | `(word >> 8) * 2^-24`, in [0, 1) | 1 |
| `NextF64` | `(NextU64 >> 11) * 2^-53`, in [0, 1) | 2 |
| `NextBelow bound` | rejection sampling: redraw while `word < (2^32 - bound) mod bound`, then `word mod bound`. Unbiased and deterministic | 1 or more |
| `NextInt(lo, hi)` | `lo + NextBelow(hi - lo)` | 1 or more |
| `Fill span` | consecutive words | n |
| `WithStream s` | the same seed on stream `s`, at position 0 | 0 |

`DetRng.streamOf name` is the first 8 bytes, read little-endian, of `Canonical.hash "lrm.rng.stream/1"` over the name written as a string. It gives components stable, distinct streams.

**Vectors.**
- `DetRng(0, 0)`: its first four words are the all-zero Philox vector above.
- `DetRng(2^64-1, 2^64-1).Block(2^64-1)` is the all-ones vector.
- `DetRng(0x299f31d0a4093822, 0x0370734413198a2e).Block(0x85a308d3243f6a88)` is the pi vector.
- `DetRng(0, 0)` gives:
  - `NextU64()` = `e169c58d6627e8d5`;
  - `NextF32()` = 6694888 * 2^-24;
  - `NextF64()` = 7931020870206717 * 2^-53.
- `DetRng(42, 7)`: its first eight words are `67ee6f2c e55410cc 6c7eca35 557398d3 e5dde940 600f6196 8fcdf8f1 2c8ed839`.
- `DetRng.streamOf "test"` = `73c5a94594114dd9`.

## Identifiers

- **Safe names.** A `SafeName` is 1 to 64 characters from `a-z`, `0-9`, `.`, `_` and `-`.
  - It starts with a letter or digit and does not end with `.`.
  - It is lowercase only, so names never collide on case-insensitive file systems.
- **RunId** is a safe name; it names the folder `LRM_DATA/runs/<runId>`.
  - `RunId.create utc suffix` gives `yyyyMMdd-HHmmss-xxxxxxxx`: the UTC start time, then 32 bits in hex.
  - `RunId.fresh ()` uses the current time and 32 bits from the operating system's random number generator.
  - Ids made this way sort by start time.
- **NodeId** is a safe name for one process in a run, for example `n0` or `coord`; it is the node part of `events.<node>.jsonl`. `NodeId.local` is `n0`.
- **Not content.** Run ids, node ids and paths are labels. They never enter a content hash.
- **Other ids.** Numeric ids such as shards, stages and epochs belong to the contracts that define them, for example ClusterProtocol.

## Invariant helpers

Library code formats and parses numbers only through `Invariant`, or with an explicit `CultureInfo.InvariantCulture`. It never relies on the current culture, even though `Process.init` sets it.

- **Floats.** `Invariant.f32` and `Invariant.f64` use the `R` format, the shortest text that parses back to the same bits. Examples: `1.5`, `0.1`, `1E-05`, `3.4028235E+38`, `-0`, `NaN`, `Infinity` and `-Infinity`. `parseF32` and `parseF64` invert them exactly, and a property test checks this for random float32 bit patterns.
- **Integers.** `i32`, `i64`, `u32` and `u64` format; `tryParseI32`, `tryParseI64` and `tryParseU64` parse, together with their raising forms.
- **Interpolation.** `Invariant.format $"..."` formats an interpolated string with the invariant culture. Plain F# interpolation without a format specifier uses the current culture.
- **Hex.** `hex` writes lowercase hex. `ofHex` and `tryOfHex` read either case, with no separators or prefix.
- **Text.** `equals` is ordinal equality. `lower` lowercases with the invariant rules, never the Turkish dotless i.

## LrmPaths

- **Default.** `LrmPaths.defaultDataRoot` comes from the `Lrm.DefaultDataRoot` assembly metadata, which Directory.Build.props sets from the `dataRoot` in `tools/bootstrap/toolchain.json`. It is the same source Lrm.Testing reads.
- **Override.** `dataRoot ()` returns `LRM_DATA` when it is set and not blank, else the default, always as a full path. `resolveDataRoot` is the same rule as a pure function.
- **Fixed locations.** `models ()`, `model name`, `toolchain ()`, `downloads ()`, `golden ()`, `runs ()`, `run runId`, `locks ()` and `venvGolden ()`.
- **Other paths.** `under relative` builds any other path under the data root, and `underRoot` does the same under a given root. Both reject rooted paths and paths that climb out with `..`.

## Process.init

Every exe calls `Lrm.Core.Process.init ()` first (ProjectGraph contract). It runs `Process.check ()`, then sets the invariant culture as the default and current culture and UI culture. It is safe to call more than once.

`Process.check ()` fails in these cases:
1. **Globalization-invariant mode.** This mode silently turns Unicode normalization into a no-op, which would corrupt tokenization. The check follows the runtime's own precedence, measured on .NET 10.0.13:
   - `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT` decides when it is `1`, `true`, `0` or `false` (in any case), even against the runtimeconfig;
   - otherwise the `System.Globalization.Invariant` switch decides;
   - otherwise the mode is off.
2. **The NFC canary.** `e` followed by U+0301 COMBINING ACUTE ACCENT must normalize (FormC) to U+00E9. This catches invariant mode, and a missing ICU, without reading any switch.
3. **The host** must be little-endian and 64-bit.

A failure raises `InvalidOperationException("LRM Process.init: <cause>")`. Unhandled, it ends the process with a nonzero exit code and the message on stderr.

## Verification

The cpu tier of `Lrm.Core.Tests` covers all of the following:
- **Known answers:** SHA-256 (FIPS 180-2 vectors plus one million `a`), Canonical C1 to C3, ExecutionClass, Philox4x32-10 and -7, and SplitMix64.
- **bf16 and f16:** the exhaustive sweeps and the vector tables.
- **Round-trip properties:** canonical values, strings and collections; `R` floats; hex; ContentId; and DetRng random access against sequential draws.
- **Process.init:** child processes show that it succeeds normally and fails in invariant mode, whether set by switch or by variable. A further child run shows that the NFC canary alone detects invariant mode.
- **Culture:** every module's output is identical under the invariant culture, de-DE and tr-TR. The test first confirms that each culture is really active.
