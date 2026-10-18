# ADR M1-0001: The tokenizer follows the oracle's Unicode: runtime classes, Unicode 9 NFC through split points

- **Status:** accepted, 2026-10-18.
- **Owner:** M1.
- **Scope:** TokenizerApi (`docs/contracts/tokenizer.md`), Encode's NFC and pre-tokenizer stages, and anyone who runs the tokenizer on another runtime or OS.

## Context

The tokenizer must give the oracle's ids exactly: HF tokenizers 0.23.3 on the pinned `tokenizer.json`. Two of its stages read Unicode data, and the oracle's data are not the .NET runtime's:

- **The Split regex** reads `\p{L}`, `\p{N}` and `\s` from Oniguruma's tables. The plan expected these to differ from .NET's for new letters, and O1's corpus has a `unicode-version` category for that.
- **The NFC normalizer** uses the Rust crate `unicode-normalization-alignments`. .NET's `string.Normalize` uses ICU.

`System.Text.RegularExpressions` is out: it matches UTF-16 code units, so astral letters fall into `[^\s\p{L}\p{N}]`. This was found while planning, and is pinned by the Regex regression test.

## Decision

1. **Classes.** The pre-tokenizer's classes come from the runtime: `Rune.GetUnicodeCategory` and `Rune.IsWhiteSpace`, with the BMP in a table built once. There is no exception table, because there is no difference to correct (Evidence).
2. **Case folding.** The contraction group compares scalars by simple case folding restricted to its letters: the two ASCII cases of s, t, r, e, v, m, l and d, plus U+017F LATIN SMALL LETTER LONG S for s. Only U+0027 opens the group.
3. **NFC.** `Nfc.normalize` splits a stretch at every *split point* and runs `string.Normalize(FormC)` on the parts. The split points stay as they are. A split point is a scalar that the oracle's NFC treats as inert: combining class 0, no decomposition, in no composition. For such a scalar b, the oracle's NFC(a b c) = NFC(a) b NFC(c). The split points are:
   - the 101 scalars that are normalization-relevant in Python's Unicode 14.0 tables but inert in the oracle: newer combining marks such as U+07FD and U+1DF6, and the Dives Akuru composite U+11938 with its parts U+11935 and U+11930;
   - the 10,301 scalars assigned after Unicode 14.0 (Oniguruma's `\P{Cn}` minus Python's assigned set), which a newer ICU may know;
   - U+FFFE, which `string.Normalize` rejects with "invalid Unicode code points" although NFC leaves it alone.

   `probe_oracle.py` generates the list as `src/Lrm.Tokenizer/OracleNfc.g.fs`, after checking that each split point is inert in the oracle in 144 contexts.
4. **Checked where it runs.** The cpu tier compares the classes, the case folding and the NFC with the oracle on every scalar value, wherever it runs, CI included. NFC is compared by family digests. A runtime or ICU whose data change an id fails these tests before it reaches a model run.

## Evidence

Measured on the development machine: .NET 10.0.402 (runtime 10.0.13), ICU 72.1 (from the runtime), Windows 10.0.26200, tokenizers 0.23.3, Python 3.11.9 with unicodedata 14.0.0. The tool is `tests/Lrm.Tokenizer.Tests/oracle/probe_oracle.py`, about 45 s; a second run writes the same bytes.

- **Classes.** Each scalar was classified from the oracle's Split pieces of `a·c·a`, `c·!` and `c·c·a`. All three probes agree for every scalar, and the class equals .NET's for all 1,112,064 scalars: 141,028 letters, 1,911 numbers, 25 white space. Oniguruma in tokenizers 0.23.3 has Unicode 16.0 letters (U+1C89, U+A7CB, U+105C0 and others are letters there). So does .NET 10. Python 3.11's Unicode 14.0 would differ on 9,393 scalars. So the `unicode-version` corpus category passes with the runtime's tables, and no Unicode-version difference exists between .NET 10 and the oracle for the regex.
- **Case folding.** Every letter was probed at each position of the group (`'·c·a`, `'·c·ea`, `'r·c·a` and so on). Only `D M S T d m s t U+017F`, `R V r v`, `E e` and `L l` match. Over every scalar, only U+0027 opens the group.
- **NFC.** The oracle's NFC, compared with .NET's, over probe families covering every scalar: alone; between `a` and U+0334 (ccc 1); after `a` U+0345 (ccc 240); before U+0301 and U+0308; after `a`, U+1100 and U+AC00; every pair of the 1,930 marks the oracle reorders; and every canonical decomposition, recomposed.
  - The oracle's tables are Unicode 9.0. U+07FD (Unicode 11) and U+1DF6 (Unicode 10) are inert in the oracle, while Adlam's U+1E944 (Unicode 9) reorders.
  - `string.Normalize` with ICU 72.1 (Unicode 15.0) reorders 108 marks that the oracle does not, and composes U+11935 U+11930 to U+11938. It throws on U+FFFE.
  - With the split points, `Nfc.normalize` equals the oracle on every family: the counts and SHA-256 values in `unicode-oracle.json`.
- **The corpus.** O1's 5,597 cases, and the probe's 1,800 random texts and id lists, are all exact. Three corpus cases hold U+FFFE: plain `string.Normalize` throws on them. No corpus case depends on the reordering difference. The probe's `encode-cases.jsonl` adds such texts on purpose (E6 is `a U+07FD U+0334`).
- **Regex.** On the 4,000 probe split cases, .NET Regex with the same pattern differs from the oracle on 890 and the state machine on 0.

## Consequences

- **M2 and C2** get the oracle's ids for any text on this runtime, not just the corpus.
- **Another runtime or OS.** Encode is exact as long as the runtime's tables are at least Unicode 9 and its newer data stay in the split points. That holds for every ICU up to Unicode 16. A future runtime with Unicode 17 tables could add letters or numbers that Oniguruma 0.23.3 lacks. The `classes` test would then fail, and M1 adds an exception table.
- **Reopening.** Another tokenizers version, or a `tokenizer.json` with another normalizer or pattern, means rerunning the probe tool. The loader refuses such a file anyway (TokenizerApi, "Loading"). A Unicode-version difference found by the tests reopens decision 1.

## Alternatives

- **An NFC implementation of our own with the oracle's Unicode 9 tables.** This is exact without ICU, but it means about 2,000 decompositions, 900 combining classes and the composition table to extract and maintain. The plan names `string.Normalize(FormC)` (`Process.init` guarantees ICU). The split points get the oracle's result from ICU with a 110-range table.
- **Plain `string.Normalize`.** It throws on three corpus cases (U+FFFE), and it differs from the oracle on texts with post-Unicode-9 marks.
- **A class table of our own, extracted from the oracle.** This would decouple the classes from the runtime. But it is identical to the runtime's today, and the exhaustive test would catch a drift, so it adds data without adding exactness.
- **System.Text.RegularExpressions.** Wrong on astral scalars (P13, P18), and 890 of 4,000 split cases differ.
