"""Oracle probes for M1 (TokenizerApi contract, docs/contracts/tokenizer.md).

Asks the oracle, HF tokenizers 0.23.3 on the pinned Qwen3-0.6B tokenizer.json, how it treats every Unicode scalar
value, and writes what it says as test data and as one generated source file:

  tests/Lrm.Tokenizer.Tests/oracle/unicode-oracle.json   character classes of the Split regex (L, N, S, O) for every
                                                         scalar, the case-folded contraction letters, the apostrophe,
                                                         and digests of the oracle's NFC over probe families
  tests/Lrm.Tokenizer.Tests/oracle/split-cases.jsonl     strings and the oracle's Split pieces (no tokenizer.json
                                                         needed to test them, so they run in CI)
  tests/Lrm.Tokenizer.Tests/oracle/encode-cases.jsonl    random texts with the oracle's ids, and random id lists with
                                                         the oracle's decoded text (need tokenizer.json)
  src/Lrm.Tokenizer/OracleNfc.g.fs                       the code points at which Lrm.Tokenizer splits a string
                                                         before string.Normalize(FormC) (ADR M1-0001)

Run it in the golden venv, isolated and offline, from the repo root:

  <LRM_DATA>/venv-golden/Scripts/python.exe -I tests/Lrm.Tokenizer.Tests/oracle/probe_oracle.py \
      --tokenizer <LRM_DATA>/models/Qwen3-0.6B/tokenizer.json

Every output is a pure function of the oracle, Python's unicodedata and a fixed seed; a second run writes the same
bytes. Every file is ASCII with LF line ends.
"""

import argparse
import hashlib
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import unicodedata  # noqa: E402

import tokenizers  # noqa: E402
from tokenizers import Regex, Tokenizer, normalizers  # noqa: E402
from tokenizers.pre_tokenizers import Split  # noqa: E402

TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZERS_VERSION = "0.23.3"
SEED = 0x4D31544F4B  # "M1TOK"
MAX_SCALAR = 0x110000

HERE = os.path.dirname(os.path.abspath(__file__))


def scalars():
    for c in range(MAX_SCALAR):
        if 0xD800 <= c <= 0xDFFF:
            continue
        yield c


def hexs(s):
    return " ".join("%x" % ord(ch) for ch in s)


def ranges(sorted_points):
    out = []
    for c in sorted_points:
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


class Oracle:
    def __init__(self, tokenizer_path):
        raw = open(tokenizer_path, "rb").read()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != TOKENIZER_SHA256:
            sys.exit("tokenizer.json has SHA-256 %s, expected %s" % (sha, TOKENIZER_SHA256))
        if tokenizers.__version__ != TOKENIZERS_VERSION:
            sys.exit("tokenizers %s, expected %s" % (tokenizers.__version__, TOKENIZERS_VERSION))
        self.sha256 = sha
        self.json = json.loads(raw.decode("utf-8"))
        self.pattern = self.json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.split = Split(Regex(self.pattern), behavior="isolated", invert=False)
        self.unassigned = Split(Regex(r"\p{Cn}"), behavior="removed", invert=False)
        self.nfc = normalizers.NFC()
        self.nfd = normalizers.NFD()
        self.tok = Tokenizer.from_str(raw.decode("utf-8"))

    def pieces(self, s):
        return [p for p, _ in self.split.pre_tokenize_str(s)]


# ---------------------------------------------------------------------------------------------------------------------
# Character classes of the Split regex


def classify(o, c):
    """L, N, S (\\s) or O for one scalar, from the pieces of two probe strings."""
    if c in (0x0A, 0x0D, 0x20):
        return "S"
    ch = chr(c)
    n1 = len(o.pieces("a" + ch + "a"))
    if n1 == 1:
        k = "L"
    elif n1 == 3:
        k = "N"
    else:
        k = "O" if len(o.pieces(ch + "!")) == 1 else "S"
    # Cross-check with a third probe: L -> one piece, N -> [c, c, a], S -> [c, ca], O -> [cc, a].
    p = o.pieces(ch + ch + "a")
    want = {"L": [ch + ch + "a"], "N": [ch, ch, "a"], "S": [ch, ch + "a"], "O": [ch + ch, "a"]}[k]
    if p != want:
        sys.exit("class probe disagrees for U+%04X: %s, pieces %r" % (c, k, p))
    return k


def class_runs(o):
    runs = []
    for c in range(MAX_SCALAR):
        k = "X" if 0xD800 <= c <= 0xDFFF else classify(o, c)
        if not runs or runs[-1][1] != k:
            runs.append([c, k])
    return runs


def contraction_letters(o, classes):
    """Every letter that the case-insensitive contraction group matches, per pattern position."""
    probes = {
        "s|t|m|d": lambda ch: ("'" + ch + "a", 2),
        "r|v (first of 're 've)": lambda ch: ("'" + ch + "ea", 3),
        "e (second of 're)": lambda ch: ("'r" + ch + "a", 3),
        "e (second of 've)": lambda ch: ("'v" + ch + "a", 3),
        "l (first of 'll)": lambda ch: ("'" + ch + "la", 3),
        "l (second of 'll)": lambda ch: ("'l" + ch + "a", 3),
    }
    out = {}
    for name, fn in probes.items():
        hits = []
        for c in scalars():
            if classes[c] != "L":
                continue
            s, k = fn(chr(c))
            p = o.pieces(s)
            if len(p) == 2 and len(p[0]) == k:
                hits.append(c)
        out[name] = hits
    apostrophes = [c for c in scalars() if o.pieces("x" + chr(c) + "sa") == ["x", chr(c) + "s", "a"]]
    return out, apostrophes


# ---------------------------------------------------------------------------------------------------------------------
# NFC

# Probe families: (name, input builder). The digest of a family is the SHA-256 of the UTF-8 lines
# "<family> <hex scalar> <hex scalars of the output>\n" for every scalar, in increasing order, whose input the oracle's
# NFC changes. The cpu tier computes the same lines with Lrm.Tokenizer's NFC and compares the digests.
FAMILIES = [
    ("F1", lambda ch: ch),
    ("F2", lambda ch: "a" + ch + "\u0334"),
    ("F3", lambda ch: "a\u0345" + ch),
    ("F5", lambda ch: ch + "\u0301"),
    ("F6", lambda ch: "a" + ch),
    ("F7", lambda ch: ch + "\u0308"),
    ("F8", lambda ch: "\u1100" + ch),
    ("F9", lambda ch: "\uac00" + ch),
]


def family_lines(o, name, fn):
    lines = []
    changed = []
    for c in scalars():
        s = fn(chr(c))
        out = o.nfc.normalize_str(s)
        if out != s:
            lines.append("%s %x %s\n" % (name, c, hexs(out)))
            changed.append(c)
    return lines, changed


def digest(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def nfc_probes(o):
    families = {}
    changed_by = {}
    for name, fn in FAMILIES:
        lines, changed = family_lines(o, name, fn)
        families[name] = {"count": len(lines), "sha256": digest(lines)}
        changed_by[name] = set(changed)

    # Oracle NFD of every scalar, and Python's (unicodedata) NFD.
    oracle_nfd = {}
    py_nfd = {}
    for c in scalars():
        ch = chr(c)
        d = o.nfd.normalize_str(ch)
        if d != ch:
            oracle_nfd[c] = d
        p = unicodedata.normalize("NFD", ch)
        if p != ch:
            py_nfd[c] = p

    # Pairwise ordering of the marks the oracle reorders: 'a' + m1 + m2.
    marks = sorted(changed_by["F2"] | changed_by["F3"])
    pair_lines = []
    for m1 in marks:
        for m2 in marks:
            s = "a" + chr(m1) + chr(m2)
            out = o.nfc.normalize_str(s)
            if out != s:
                pair_lines.append("P %x %x %s\n" % (m1, m2, hexs(out)))
    families["P"] = {"count": len(pair_lines), "sha256": digest(pair_lines)}

    # Composition: the oracle's NFC of every canonical decomposition that the oracle or Python knows. Hangul syllables
    # compose by arithmetic, which F8 and F9 cover, so their 11,172 decompositions are left out.
    def hangul(c):
        return 0xAC00 <= c <= 0xD7A3

    seqs = sorted(
        {d for c, d in oracle_nfd.items() if not hangul(c)} | {d for c, d in py_nfd.items() if not hangul(c)},
        key=lambda s: [ord(x) for x in s],
    )
    k_lines = []
    for i, s in enumerate(seqs):
        out = o.nfc.normalize_str(s)
        if out != s:
            k_lines.append("K %d %s\n" % (i, hexs(out)))
    families["K"] = {"count": len(k_lines), "sha256": digest(k_lines)}

    # Non-inert scalars (ccc != 0, a decomposition, or part of a composition) per the oracle and per Python.
    oracle_non_inert = set()
    for name, _ in FAMILIES:
        oracle_non_inert |= changed_by[name]
    for c, d in oracle_nfd.items():
        oracle_non_inert.add(c)
        oracle_non_inert |= {ord(x) for x in d}
    py_non_inert = set()
    for c in scalars():
        if unicodedata.combining(chr(c)) != 0:
            py_non_inert.add(c)
    for c, d in py_nfd.items():
        py_non_inert.add(c)
        py_non_inert |= {ord(x) for x in d}

    assigned_py = {c for c in scalars() if unicodedata.category(chr(c)) != "Cn"}
    def assigned_in_oniguruma(c):
        # \p{Cn} with behaviour "removed" drops an unassigned scalar, so only then do the pieces join to "ab".
        return "".join(p for p, _ in o.unassigned.pre_tokenize_str("a" + chr(c) + "b")) != "ab"

    assigned_onig = {c for c in scalars() if assigned_in_oniguruma(c)}

    newer = py_non_inert - oracle_non_inert
    after_py = assigned_onig - assigned_py
    split_points = sorted(newer | after_py | {0xFFFE})
    if set(split_points) & oracle_non_inert:
        sys.exit("a split point is non-inert in the oracle")

    # Every split point must be inert in the oracle: NFC(b + d + a) == NFC(b) + d + NFC(a).
    contexts = ["", "a", "e", "\u0301", "\u0334", "\u0345", "\u1100", "\u1161", "\uac00", "\u0b47", "\u0dd9", "\u05d0"]
    for d in split_points:
        ch = chr(d)
        for b in contexts:
            for a in contexts:
                got = o.nfc.normalize_str(b + ch + a)
                want = o.nfc.normalize_str(b) + ch + o.nfc.normalize_str(a)
                if got != want:
                    sys.exit("split point U+%04X is not inert in the oracle (%r)" % (d, b + ch + a))

    stats = {
        "oracle_non_inert": len(oracle_non_inert),
        "python_non_inert": len(py_non_inert),
        "python_non_inert_unknown_to_oracle": len(newer),
        "assigned_oniguruma": len(assigned_onig),
        "assigned_python": len(assigned_py),
        "assigned_after_python": len(after_py),
        "split_points": len(split_points),
        "oracle_nfd": len(oracle_nfd),
        "python_nfd": len(py_nfd),
        "marks": len(marks),
    }
    return families, marks, [hexs(s) for s in seqs], split_points, stats


# ---------------------------------------------------------------------------------------------------------------------
# Split cases and encode cases


SPECIAL = (
    [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x85", "\xa0", "\u1680", "\u2000", "\u200a", "\u2028", "\u2029"]
    + ["\u202f", "\u205f", "\u3000", "\u180e", "\u200b", "\ufeff", "\x00", "\x1f", "\x7f", "\ufffe", "\uffff"]
    + ["'", "\u2019", "s", "S", "\u017f", "t", "T", "re", "RE", "ve", "Ve", "m", "M", "ll", "LL", "lL", "d", "D"]
    + ["a", "Z", "\xe9", "\xdf", "\u01c5", "\u02b0", "\u4e2d", "\u3042", "\uac00", "\u0628", "\u0928", "\u094d"]
    + ["\U0001d400", "\U0001d49c", "\U00020000", "\U0001e900", "\u1c89", "\U000105c0", "\U00011f04"]
    + ["0", "9", "\u0663", "\uff10", "\U0001d7ce", "\u216b", "\u3007", "\U00010140", "\xbd", "\xb2", "\u2460"]
    + ["\u2080", "\u0f33", "\u2189", "\u0301", "\u0308", "\u0334", "\u20dd", "\u07fd", "\U0001e4ec"]
    + ["\U0001f600", "\U0001f44d", "\U0001f3fd", "\u200d", "\U0001f1e9", "\U0001f1ea", "\ufe0f", "\u20e3"]
    + ["!", ".", ",", ";", "(", ")", "{", "}", "<", ">", "=", "_", "-", "#", "@", "$", "\u20ac", "\xa9", "\u2122"]
    + ["\U0010ffff", "\U000e0001", "\ue000"]
)


def random_text(rng, max_len):
    n = rng.randint(1, max_len)
    parts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.55:
            parts.append(rng.choice(SPECIAL))
        elif r < 0.75:
            parts.append(rng.choice(["the", "The", "isn't", "we'RE", "I'M", "class", "int", "x1", "IL_00", "  "]))
        elif r < 0.9:
            c = rng.randrange(0x20, 0x3000)
            parts.append(chr(c))
        else:
            c = rng.randrange(0x10000, 0x110000)
            parts.append(chr(c))
    return "".join(parts)


def split_cases(o):
    rng = random.Random(SEED)
    texts = []
    seen = set()
    tricky = [s for s in SPECIAL if len(s) == 1]
    for a in tricky:
        for b in ["a", " ", "'", "1", "\n", "!", "\u3000"]:
            for s in (a + b, b + a, a + a + b):
                if s not in seen:
                    seen.add(s)
                    texts.append(s)
    while len(texts) < 4000:
        s = random_text(rng, 16)
        if s not in seen:
            seen.add(s)
            texts.append(s)
    return [{"pieces": o.pieces(s), "text": s} for s in texts]


def encode_cases(o):
    rng = random.Random(SEED + 1)
    added = [t["content"] for t in o.json["added_tokens"]]
    out = []
    seen = set()
    while len(out) < 1500:
        s = random_text(rng, 32)
        if rng.random() < 0.25:
            i = rng.randint(0, len(s))
            s = s[:i] + rng.choice(added) + s[i:]
        if rng.random() < 0.1:
            s = s + rng.choice(["e\u0301", "\u1100\u1161\u11a8", "a\u07fd\u0334", "\U00011935\U00011930", "\ufffe\u0301"])
        if s in seen:
            continue
        seen.add(s)
        out.append({"ids": o.tok.encode(s, add_special_tokens=False).ids, "kind": "encode", "text": s})
    rows = o.tok.get_vocab_size(with_added_tokens=True)
    while len(out) < 1800:
        n = rng.randint(0, 12)
        ids = []
        for _ in range(n):
            r = rng.random()
            if r < 0.7:
                ids.append(rng.randrange(0, 151643))
            elif r < 0.85:
                ids.append(rng.randrange(151643, rows))
            else:
                ids.append(rng.randrange(151669, 151936))
        out.append({"ids": ids, "kind": "decode", "text": o.tok.decode(ids, skip_special_tokens=False)})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Writers


def write_ascii(path, text):
    text.encode("ascii")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(text)


def jdump(v):
    return json.dumps(v, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_jsonl(path, rows):
    write_ascii(path, "".join(jdump(r) + "\n" for r in rows))


def write_fsharp(path, split_points, stats):
    rs = ranges(split_points)
    lines = [
        "// GENERATED by tests/Lrm.Tokenizer.Tests/oracle/probe_oracle.py from the oracle (HF tokenizers %s) and Python's"
        % TOKENIZERS_VERSION,
        "// unicodedata %s. Do not edit; rerun the tool. ADR M1-0001 explains the set." % unicodedata.unidata_version,
        "module internal Lrm.Tokenizer.OracleNfcData",
        "",
        "/// Scalars that the oracle's NFC (Unicode 9 tables) treats as inert but a newer Unicode does not: %d scalars in"
        % len(split_points),
        "/// %d ranges, as inclusive [first; last] pairs in increasing order. %d are normalization-relevant in Python's"
        % (len(rs), stats["python_non_inert_unknown_to_oracle"]),
        "/// Unicode %s tables and unknown to the oracle, %d were assigned after Unicode %s, and U+FFFE, which"
        % (unicodedata.unidata_version, stats["assigned_after_python"], unicodedata.unidata_version),
        "/// string.Normalize rejects.",
        "let splitRanges: int[] =",
        "    [|",
    ]
    row = []
    for a, b in rs:
        row.append("0x%X; 0x%X" % (a, b))
        if len(row) == 6:
            lines.append("        " + "; ".join(row))
            row = []
    if row:
        lines.append("        " + "; ".join(row))
    lines.append("    |]")
    write_ascii(path, "\n".join(lines) + "\n")


def main():
    if not sys.flags.isolated:
        sys.exit("run with python -I")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path of the pinned Qwen3-0.6B tokenizer.json")
    ap.add_argument("--out", default=HERE, help="folder for the test data (default: this folder)")
    ap.add_argument("--src", default=os.path.join(HERE, "..", "..", "..", "src", "Lrm.Tokenizer"))
    args = ap.parse_args()

    o = Oracle(args.tokenizer)
    runs = class_runs(o)
    classes = []
    for i, (start, k) in enumerate(runs):
        end = runs[i + 1][0] if i + 1 < len(runs) else MAX_SCALAR
        classes.extend([k] * (end - start))
    letters, apostrophes = contraction_letters(o, classes)
    families, marks, seqs, split_points, stats = nfc_probes(o)

    meta = {
        "about": "Oracle probes for M1 (docs/contracts/tokenizer.md, ADR M1-0001), written by probe_oracle.py. "
        "classes: [first scalar, class] runs over 0..0x10FFFF (L letter, N number, S white space, O other, X surrogate). "
        "contraction_letters: the letters the case-insensitive group matches at each position; apostrophes: the "
        "scalars that open it. nfc: per probe family the count and SHA-256 of the lines '<family> <hex> <hex output>' "
        "for every input the oracle's NFC changes; P runs over marks x marks, K over composition_inputs.",
        "pattern": o.pattern,
        "python": sys.version.split()[0],
        "seed": SEED,
        "tokenizer_sha256": o.sha256,
        "tokenizers": tokenizers.__version__,
        "unicodedata": unicodedata.unidata_version,
    }
    sections = [
        ("meta", meta),
        ("classes", runs),
        ("contraction_letters", letters),
        ("apostrophes", apostrophes),
        ("nfc", families),
        ("nfc_stats", stats),
        ("marks", ranges(marks)),
        ("split_points", ranges(split_points)),
        ("composition_inputs", seqs),
    ]
    body = "{\n" + ",\n".join("%s: %s" % (json.dumps(k), jdump(v)) for k, v in sections) + "\n}\n"
    write_ascii(os.path.join(args.out, "unicode-oracle.json"), body)
    write_jsonl(os.path.join(args.out, "split-cases.jsonl"), split_cases(o))
    write_jsonl(os.path.join(args.out, "encode-cases.jsonl"), encode_cases(o))
    write_fsharp(os.path.join(args.src, "OracleNfc.g.fs"), split_points, stats)
    print(json.dumps(stats, sort_keys=True))
    print("families", json.dumps({k: v["count"] for k, v in families.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
