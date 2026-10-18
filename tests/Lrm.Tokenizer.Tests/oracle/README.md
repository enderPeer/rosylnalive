# Tokenizer oracle probes (M1)

Test data that `probe_oracle.py` writes from the oracle, HF tokenizers 0.23.3 on the pinned Qwen3-0.6B `tokenizer.json` (SHA-256 `aeb13307…`), and from Python's `unicodedata` (14.0.0). The TokenizerApi contract (`docs/contracts/tokenizer.md`) and ADR M1-0001 explain what the data pins. These files are derived, never edited by hand.

| File | Contents | Needs tokenizer.json to test |
| --- | --- | --- |
| `unicode-oracle.json` | The class of every scalar value in the Split regex (L, N, S, O as runs), the letters the case-insensitive contraction group matches at each position, the scalars that open it (only U+0027), and per NFC probe family the count and SHA-256 of the oracle's changed outputs. It also lists the reordering marks, the composition inputs and the split points | no |
| `split-cases.jsonl` | 4,000 strings with the oracle's Split pieces: every tricky scalar next to a letter, a space, an apostrophe, a digit, a newline, `!` and U+3000, then seeded random mixtures | no |
| `encode-cases.jsonl` | 1,500 seeded random texts (added tokens, NFC-sensitive tails, astral and newer scalars) with the oracle's ids, and 300 random id lists with the oracle's decoded text | yes |
| `../../../src/Lrm.Tokenizer/OracleNfc.g.fs` | The split points of `Nfc.normalize` | no |

## Regenerating

From the repo root, in the golden venv, isolated and offline:

```
C:/Users/end/dev/lrm-data/venv-golden/Scripts/python.exe -I tests/Lrm.Tokenizer.Tests/oracle/probe_oracle.py --tokenizer C:/Users/end/dev/lrm-data/models/Qwen3-0.6B/tokenizer.json
```

It takes about 45 s and needs nothing else: no network, no package install. A second run writes the same bytes (checked when the data was committed). The tool refuses another `tokenizer.json` hash or another tokenizers version.

## What the probes found (ADR M1-0001)

- **Classes.** Oniguruma's `\p{L}`, `\p{N}` and `\s` in tokenizers 0.23.3 agree with .NET 10's Unicode 16.0 tables (`Rune.GetUnicodeCategory`, `Rune.IsWhiteSpace`) on all 1,112,064 scalar values. The Unicode 15 and 16 letters of the corpus's `unicode-version` category are letters on both sides.
- **Case folding.** The contraction group matches s, t, m, d, r, v, e and l in both ASCII cases, plus U+017F LATIN SMALL LETTER LONG S for s. Nothing else, and only U+0027 opens it.
- **NFC.** The oracle normalizes with Unicode 9.0 tables (the `unicode-normalization-alignments` crate). 101 scalars that Unicode 14 normalizes (newer combining marks such as U+07FD, and U+11935, U+11930, U+11938) are inert there, and so is everything assigned later. On this machine (.NET 10.0.402, ICU 72.1) `string.Normalize` treats 111 scalars differently from the oracle (108 combining marks it reorders, and the three scalars of the Dives Akuru composite U+11938, which it composes) and throws on U+FFFE, which three corpus cases hold. `Nfc.normalize` splits at those scalars, so it equals the oracle on every probe family.
