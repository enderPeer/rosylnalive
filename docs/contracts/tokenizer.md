# TokenizerApi contract

Owner: M1. Status: v1, proposed with M1's PR; frozen by convention once merged, and through a `contract-change` PR after the contracts tag. Consumers: M2 (prompts and decoding in Models.Qwen and `lrm generate`), C2 (serving), I1 and I3 (the Gate 1 tokenizer check), T1 and T2 (reports and dashboard over `tokenizer.*` records). Code: `src/Lrm.Tokenizer`. Tests: `tests/Lrm.Tokenizer.Tests` (cpu and bench tiers). Oracle data: `tests/Lrm.Tokenizer.Tests/oracle` (ADR M1-0001).

The specification is exact id parity with the oracle, HF tokenizers 0.23.3 on the pinned Qwen3-0.6B `tokenizer.json`, as O1's corpus records it (GoldenSet, "The tokenizer corpus"). No `System.Text.RegularExpressions`, no third-party tokenizer and no Python are used at run time.

## API

```fsharp
namespace Lrm.Tokenizer

type AddedToken = { Id: int; Content: string; Special: bool }

[<Sealed>]
type QwenTokenizer =
    member Encode: text: string -> int[]
    member EncodeBytes: bytes: byte[] -> int[]
    member Decode: ids: IReadOnlyList<int> -> string
    member DecodeBytes: ids: IReadOnlyList<int> -> byte[]
    member IdOf: token: string -> int option
    member TokenOf: id: int -> string option
    member VocabSize: int                  // real ids: 151,669
    member MergeCount: int                 // 151,387
    member AddedTokens: AddedToken list    // the 26 added tokens, by id
    member Sha256: string                  // of the file it was read from
    member ClearCache: unit -> unit
    member CachedPieces: int
    static member ModelRows: int           // 151,936, Qwen3-0.6B's embedding and lm_head rows

module QwenTokenizer =
    val PinnedSha256: string               // aeb13307…, from $(LrmTokenizerSha256)
    val PinnedPath: string                 // models/Qwen3-0.6B/tokenizer.json, $(LrmTokenizerJson) under $(LrmData)
    val RealIds: int                       // 151,669
    val defaultPath: unit -> string        // PinnedPath under LRM_DATA
    val load: path: string -> QwenTokenizer
    val tryLoadDefault: unit -> QwenTokenizer option
    val parse: json: byte[] -> QwenTokenizer
```

The stages are public too, for diagnostics and tests: `ByteLevel` (the byte map), `CharClass` (classes and case folding), `Nfc.normalize`, and `PreTokenizer.split`, `PreTokenizer.pieces` and `PreTokenizer.SplitPattern`.

## Loading

1. **The pin.** `load path` reads the file and requires its SHA-256 to be `PinnedSha256`, else it raises `InvalidDataException` naming both hashes. The pin and the path come from `tools/bootstrap/toolchain.json` through `build/Toolchain.g.props` (`$(LrmTokenizerSha256)`, `$(LrmTokenizerJson)`), built into the assembly as metadata by `Lrm.Tokenizer.lrm.targets`. The code repeats neither (DataRootAndToolchain).
2. **The structure.** `load` and `parse` accept only the configuration Qwen3 uses, and raise `InvalidDataException` for anything else:
   - normalizer `NFC`;
   - pre-tokenizer `Sequence[Split, ByteLevel]`: Split with exactly `PreTokenizer.SplitPattern`, behaviour `Isolated`, `invert` false; ByteLevel with `add_prefix_space` false and `use_regex` false (tokenizers' default for an absent `use_regex` is true);
   - decoder `ByteLevel` with `add_prefix_space` false; post-processor `ByteLevel` or none;
   - model `BPE` with no dropout and no unknown token, empty `continuing_subword_prefix` and `end_of_word_suffix`, and `fuse_unk`, `byte_fallback` and `ignore_merges` false;
   - a vocabulary with ids 0..n−1, each token a byte-level string, every single byte present;
   - merges as `[a, b]` pairs or `"a b"` strings, with `a`, `b` and `ab` in the vocabulary and no pair twice; the list index is the rank;
   - added tokens with `normalized`, `lstrip`, `rstrip` and `single_word` false, and ids n, n+1, … in order.
3. **The counts.** `load` also requires `VocabSize` = 151,669: 151,643 vocabulary tokens and 26 added tokens (151,643 to 151,668). The model has 151,936 rows (`ModelRows`); rows 151,669 to 151,935 have no token.
4. `parse json` skips the hash check, for tests with small vocabularies. Production code calls `load` or `tryLoadDefault`.
5. **Where the file is.** `defaultPath ()` is `PinnedPath` under the data root (`LrmPaths`). `tryLoadDefault ()` returns None when the file is absent, as on CI runners until the user approves its download (AP-1).

## Encode

`Encode text` runs these stages in order and returns the ids. It adds no BOS and no EOS, so the result holds only ids that the text spells.

1. **Scalars only.** A lone surrogate raises `ArgumentException` with its index. The oracle refuses such text too: Python cannot pass it to Rust. A null text raises `ArgumentNullException`.
2. **Added tokens, on the raw text.** A character trie over the 26 contents scans left to right. At the first position where some added token starts, the longest one there is taken (leftmost-longest, as tokenizers' Aho-Corasick matcher), and its id is emitted. All 26, special or not, are matched the same way; Encode has no "special tokens off" mode. The stretches between matches go on separately, so nothing below crosses an added token.
3. **NFC, per stretch.** `Nfc.normalize` is the oracle's NFC, which uses Unicode 9.0 tables (ADR M1-0001):
   - a stretch is split at the *split points* (`OracleNfc.g.fs`, 10,403 scalars in 110 ranges, plus U+FFFE). Each is inert in the oracle (combining class 0, no decomposition, in no composition) but possibly not in the runtime's newer tables;
   - the parts between split points go through `string.Normalize(NormalizationForm.FormC)`, and the split points stay as they are.

   `Process.init` guarantees that normalization works (no globalization-invariant mode). `string.Normalize` alone would differ from the oracle: it reorders 108 combining marks newer than Unicode 9, composes U+11935 U+11930 to U+11938, and throws on U+FFFE.
4. **The Split pre-tokenizer.** `PreTokenizer.split` runs a state machine over Unicode scalar values that matches the Oniguruma regex

   ```
   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
   ```

   with behaviour Isolated: first alternative that matches, greedy and backtracking quantifiers. Every scalar is in exactly one class (`CharClass`): a letter (`\p{L}`: Lu, Ll, Lt, Lm, Lo), a number (`\p{N}`: Nd, Nl, No), white space (`\s`: the White_Space property) or other. So the matches tile the text. A match starting at p ends:
   1. at p+2 when U+0027 is followed by s, t, m or d, and at p+3 when it is followed by re, ve or ll, compared under simple case folding: the ASCII cases, plus U+017F for s. Nothing else matches, and only U+0027 opens the group;
   2. at the end of the letter run, when p is a letter, or when p is neither \r, \n nor a number and p+1 is a letter;
   3. at p+1, when p is a number (each digit is its own piece);
   4. when p is other, or a U+0020 space followed by other: after the run of others, then the run of \r and \n;
   5. when p starts a white-space run [p, q) holding \r or \n: just after the last \r or \n in the run;
   6. otherwise, at q when the run ends the text, or at q−1 when the run has two or more scalars;
   7. otherwise at q.

   Classes come from the runtime's tables (`Rune.GetUnicodeCategory`, `Rune.IsWhiteSpace`). On .NET 10 they are Unicode 16.0 and equal Oniguruma's on all 1,112,064 scalars (ADR M1-0001). `System.Text.RegularExpressions` cannot run this pattern correctly: it matches UTF-16 code units, so astral letters and numbers fall into `[^\s\p{L}\p{N}]` (vector P13; it differs on 890 of the 4,000 probe split cases).
5. **Byte level.** Each piece's UTF-8 bytes map to GPT-2's byte characters (`ByteLevel`):
   - the 188 bytes 0x21–0x7E, 0xA1–0xAC and 0xAE–0xFF stand for themselves;
   - the other 68, in increasing order, become U+0100 to U+0143.

   The vocabulary is spelled in these characters.
6. **BPE.** The piece starts as one symbol per byte. While some adjacent pair has a merge, the pair with the lowest rank is merged, and among equal ranks the leftmost, exactly as tokenizers' `Word::merge_all` pops its heap by (rank, position). The symbols' ids are the piece's ids. A whole piece that is itself a vocabulary token is not looked up directly (`ignore_merges` is false).
7. **The cache.** Results are cached per piece, up to 65,536 pieces. The cache never changes a result.

`EncodeBytes bytes` is `Encode` of `bytes` decoded as UTF-8 with one U+FFFD per maximal invalid subpart (errors=replace, which Python's `bytes.decode` and .NET's `Encoding.UTF8` both follow).

## Decode

- **DecodeBytes.** `DecodeBytes ids` concatenates, for each id:
  - a vocabulary id's bytes;
  - an added token's content as the ByteLevel decoder gives it: its byte-map bytes when every character is in the map, else its UTF-8. Both are the same for all 26 tokens, which are ASCII;
  - nothing for ids from 151,669 to 151,935, which are lm_head rows with no token, even between the bytes of one character.

  An id below 0, or at 151,936 or above, raises `ArgumentOutOfRangeException`. The oracle drops it silently, but in this system such an id is a bug. The empty list gives no bytes.
- **Decode.** `Decode ids` is `DecodeBytes ids` decoded as UTF-8 with errors=replace (one U+FFFD per maximal invalid subpart). This is tokenizers' `decode(ids, skip_special_tokens=False)`.
- **Round trip.** `Decode (Encode text)` = the oracle's NFC of `text` for every scalar-valued text.

## Ids and tokens

- **IdOf.** `IdOf token` looks up an added token's content first, then a vocabulary token in byte-level form: `"Ġthe"` is " the", and plain `" the"` is not a token.
- **TokenOf.** `TokenOf id` is the inverse, and None for ids with no token.
- **AddedTokens.** The 26 added tokens with their ids and special flags, as in the corpus manifest (vector A1).

## Threads and determinism

- **Threads.** Every member may be called from any number of threads at once. The tables are immutable; the piece cache is concurrent and capped; work arrays are per thread.
- **Determinism.** A result is a pure function of the text and the pinned file, given the runtime's Unicode tables and its ICU normalization data. The cpu tier checks those against the oracle scalar by scalar on every machine it runs on, CI runners included (`unicode` tests). So a runtime or OS update that changes them fails there, before it can change an id.

## Telemetry

Through the Lrm.Core facade (TelemetryFacade), component `tokenizer`:

| Name | Kind | Emitted by | Attributes |
| --- | --- | --- | --- |
| `tokenizer.load` | span | `load` | `bytes`, `vocab`, `merges`, `added`, `sha256` |
| `tokenizer.encode` | span | `Encode`, when telemetry is enabled | `chars`, `tokens`, `pieces`, `added` |
| `tokenizer.tokens` | meter (`token`) | `Encode` | |
| `tokenizer.decode` | event | `DecodeBytes`, when enabled | `ids`, `bytes` |
| `tokenizer.case` (reserved) | check | the corpus tests, one per corpus or probe line | **`case`** (the corpus id), **`cat`**, **`kind`**, `tokens` (expected ids), `detail` (on failure: the first differing index and the ids around it), and the scope's `source` (`corpus` or `probe`) |
| `tokenizer.corpus` | event | after each corpus file | `source`, `cases`, `passed`, `failed` |
| `tokenizer.unicode` | check | the oracle-probe tests | `probe` (`classes`, `contraction_letters`, `apostrophes`, `split_points`, `nfc.<family>`), `checked`, `mismatches` |
| `tokenizer.split` | check | the split-case test | `cases`, `mismatches` |
| `tokenizer.regex_regression` | check | the Regex regression test | `cases`, `regex_mismatches`, `state_machine_mismatches` |
| `tokenizer.skipped` | event | a test skipped without `tokenizer.json` | `test`, `reason` |
| `tokenizer.bench.tokens_per_s` | gauge (`token/s`) | the bench tier | `mode` (`encode_cold`, `encode_warm`, `decode`), `threads`, `input` (`corpus`, `csharp`) |
| `tokenizer.bench.load` | gauge (`ns`) | the bench tier | |

The tests write these records to a run of kind `tokenizer` (`LRM_DATA/runs/<runId>`, RunLog). Its params record the runtime, OS and ICU versions. Telemetry never changes an id: a cpu-tier test encodes with and without a sink and compares.

## Vectors

All values are from the oracle (tokenizers 0.23.3, `probe_oracle.py`, or a direct `Tokenizer.encode` / `decode` / `Split.pre_tokenize_str`), computed independently of this code.

**Pins**

| Pin | Value |
| --- | --- |
| `tokenizer.json` SHA-256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` (11,422,654 bytes, revision `c1899de289a04d12100db370d81485cdf75e47ca`) |
| `PinnedPath` | `models/Qwen3-0.6B/tokenizer.json` |
| Real ids, merges, model rows | 151,669; 151,387; 151,936 |
| Corpus `cases.jsonl` SHA-256 | `35ff0c43a1bc4325588b1a22953f861784e4ca76e0f3f863a5f70ff9ceb16c66` (5,597 cases) |
| `oracle/unicode-oracle.json` SHA-256 | `f9339ab0e55acabd0509b4f562ae831d97d167398aaeacbd47c4a49324acbb89` |
| `oracle/split-cases.jsonl` SHA-256 | `39bb2555c870896ecb4bc4c907c90a1dcb912aaf8ed75a143aee114c35bb3b95` (4,000 cases) |
| `oracle/encode-cases.jsonl` SHA-256 | `3ce0e0e7473f4e6465b92ba3482c31682d61d7c48645b535a987940bdd6743dd` (1,800 cases) |
| `src/Lrm.Tokenizer/OracleNfc.g.fs` SHA-256 | `52e1ea35744313d89115f904babf8ce0c37b957e1b2eb264fb20217bab74301c` |

**B1, the byte map:** 0x00 → U+0100, 0x09 → U+0109, 0x0A → U+010A, 0x20 → U+0120, 0x21 → `!`, 0x7E → `~`, 0x7F → U+0121, 0xA0 → U+0142, 0xA1 → U+00A1, 0xAC → U+00AC, 0xAD → U+0143, 0xFF → U+00FF.

**P1 to P18, the pre-tokenizer** (pieces separated by `|`; `\u` escapes as in F#):

| # | Text | Pieces |
| ---: | --- | --- |
| P1 | `Hello world` | `Hello` \| ` world` |
| P2 | `it's` | `it` \| `'s` |
| P3 | `WE'RE 'Sa` | `WE` \| `'RE` \| ` '` \| `Sa` |
| P4 | `x'ſa` | `x` \| `'ſ` \| `a` |
| P5 | two spaces, `hi` | ` ` \| ` hi` |
| P6 | `a  \n\n  b` | `a` \| `  \n\n` \| ` ` \| ` b` |
| P7 | `end` and three spaces | `end` \| three spaces |
| P8 | `12345` | `1` \| `2` \| `3` \| `4` \| `5` |
| P9 | `x½Ⅻ` | `x` \| `½` \| `Ⅻ` |
| P10 | ` !!\r\n\r\nx` | ` !!\r\n\r\n` \| `x` |
| P11 | `\tfoo` | `\tfoo` |
| P12 | `　中文` | `　中文` |
| P13 | U+1D400 U+1D401 | one piece (.NET Regex: other) |
| P14 | `a\r\nb` | `a` \| `\r\n` \| `b` |
| P15 | `x ́y` | `x` \| ` ́` \| `y` |
| P16 | `'ll've` | `'ll` \| `'ve` |
| P17 | `don't'` | `don` \| `'t` \| `'` |
| P18 | U+1D7CE U+1D7CF | U+1D7CE \| U+1D7CF |

**E1 to E11, Encode:**

| # | Text | Ids |
| ---: | --- | --- |
| E1 | `Hello world` | 9707 1879 |
| E2 | `using System;` | 970 739 26 |
| E3 | `<\|im_start\|>user\nHi<\|im_end\|>` | 151644 872 198 13048 151645 |
| E4 | `é` | 963 |
| E5 | U+1F600 | 141334 |
| E6 | `a̴߽` (no reordering) | 64 155 121 136 112 |
| E7 | `x￾́` | 87 5564 122 53839 |
| E8 | `public static int Add(int a, int b) => a + b;` | 888 1099 526 2691 1548 264 11 526 293 8 589 264 488 293 26 |
| E9 | the empty text | (none) |
| E10 | one space | 220 |
| E11 | `中文` | 104811 |

**D1 to D6, Decode:** [151643] → `<|endoftext|>`; [9707, 1879] → `Hello world`; [160, 151700, 116, 255] → `中`; [5691] → U+FFFD; [151669] → the empty text; [] → the empty text.

**A1, the added tokens:** 151643 `<|endoftext|>` to 151656 `<|video_pad|>` are special (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`, `<|quad_start|>`, `<|quad_end|>`, `<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`, `<|video_pad|>`); 151657 to 151668 are not (`<tool_call>`, `</tool_call>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`).

**U1, the Unicode facts** (`unicode-oracle.json`):
- 1,112,064 scalars classify as 141,028 letters, 1,911 numbers, 25 white space and the rest other.
- The contraction letters are `D M S T d m s t U+017F`, then `R V r v`, `E e` and `L l`.
- The NFC probe families have these counts of changed inputs: F1 1,120, F2 1,898, F3 1,929, F5 1,269, F6 1,136, F7 1,188, F8 1,141, F9 1,147, P 3,291,616 and K 940.

## Verification

`tests/Lrm.Tokenizer.Tests`:

- **CI-safe cpu tier** (needs no `tokenizer.json`):
  - `pretokenizer`: P1 to P18; all 4,000 split cases; pieces tile the text; B1 and the byte map's bijection.
  - `unicode`, checked on every scalar value against the oracle probe:
    - the class of every scalar;
    - the contraction letters at each position, and the apostrophe;
    - the NFC families F1 to F9, P and K, by count and SHA-256;
    - the split points;
    - U+FFFE, and the U+07FD reordering case.
  - `unicode.regex regression`: .NET Regex splits astral letters and numbers, and differs from the oracle on 890 split cases; the state machine on none.
  - `toy`, on a 266-id tokenizer.json with Qwen's configuration:
    - BPE rank order and leftmost ties;
    - added tokens, leftmost-longest;
    - NFC per stretch;
    - lone surrogates rejected;
    - EncodeBytes replacement;
    - Decode rules and errors=replace;
    - out-of-range ids;
    - IdOf and TokenOf;
    - 18 refused configurations, and the hash check;
    - the pins against `toolchain.json`;
    - telemetry changes no id;
    - 8 threads equal sequential;
    - Decode(Encode) = NFC.
  - The oracle data files and the generated split points have the SHA-256 values above.
- **Needs `tokenizer.json`** (cpu tier, skipped visibly without it):
  - every one of the 5,597 corpus cases is exact, one `tokenizer.case` check each;
  - the 1,800 probe cases are exact;
  - E1 to E11 and D1 to D6;
  - A1: the 26 added tokens round-trip;
  - no BOS;
  - 151,669 real ids, 151,387 merges, and every id maps to its token and back.

  Without the file each test is ignored, prints `SKIPPED <test>: tokenizer.json is not at …` to stderr and records `tokenizer.skipped`, so the CI summary shows six ignored tests, not passes. The acceptance "100% of the corpus exact in CI" waits for the user's approval of the CI download (AP-1).
- **Bench tier:** load time, and Encode (cold and warm cache, 1 and 8 threads) and Decode throughput in tokens per second.
