"""Builds goldens/tokenizer/g1: the Qwen3 tokenizer corpus with ids from HF tokenizers (GoldenSet contract).

    pwsh tools/goldens/run.ps1 make_tokenizer.py            # regenerate goldens/tokenizer/g1
    pwsh tools/goldens/run.ps1 make_tokenizer.py --check    # regenerate in memory, compare with the committed bytes

The oracle is tokenizers 0.23.3 loading the pinned tokenizer.json (SHA-256 aeb13307...). Cases are JSON lines in
cases.jsonl, one per line, with sorted keys and ASCII-only escapes:

- {"kind": "encode", "text": s, "ids": [...]}: Encode(s) = tok.encode(s, add_special_tokens=False).ids;
- {"kind": "encode_bytes", "bytes": hex, "text": s, "ids": [...]}: raw bytes, possibly invalid UTF-8. s is the bytes
  decoded with U+FFFD for each maximal invalid subpart (Python errors="replace", the Unicode recommended practice that
  .NET's UTF8Encoding also follows); ids = Encode(s);
- {"kind": "decode", "ids": [...], "text": s, "bytes": hex}: bytes are the concatenated token bytes (DecodeBytes);
  s = tok.decode(ids, skip_special_tokens=False), which equals the bytes decoded with errors="replace". Ids from
  151,669 to 151,935 are lm_head rows with no token: they contribute no bytes, as in HF;
- {"kind": "encode_reject", "utf16": hex, "error": "lone surrogate"}: the text is given only as its UTF-16 code units
  (big-endian, lowercase hex, 4 digits per unit), because a JSON string with a lone-surrogate escape is not readable
  by every JSON library (System.Text.Json's GetString throws). The text holds a lone surrogate; Encode must raise.

Every case also has "id" (unique, "<category>-<n>") and "cat" (its category).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402  (runs the process guards first)
from verify_goldens import SET_BUDGETS  # noqa: E402  (the committed-size budgets CI checks)

SET, VERSION = "tokenizer", "g1"
OUT_REL = f"goldens/{SET}/{VERSION}"
SRC = os.path.join(c.TOOLS_DIR, "tokenizer")
TOOL_FILES = ["common.py", "make_tokenizer.py", "tokenizer/csharp.txt", "tokenizer/il.txt", "tokenizer/multilingual.txt", "tokenizer/prose.txt"]
SEED = 0x70C
REAL_IDS = 151669
MODEL_ROWS = 151936  # Qwen3-0.6B vocab_size: lm_head rows. Ids REAL_IDS..MODEL_ROWS-1 have no token.
MIN_CASES = 2000


def _rng(name: str) -> c.DetRng:
    return c.DetRng(SEED, c.stream_of("tokenizer-g1/" + name))


class Corpus:
    def __init__(self):
        self.cases: list[dict] = []
        self._seen: set = set()
        self._count: dict = {}

    def _add(self, cat: str, case: dict, key) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        n = self._count.get(cat, 0) + 1
        self._count[cat] = n
        case["cat"] = cat
        case["id"] = f"{cat}-{n:04d}"
        self.cases.append(case)

    def text(self, cat: str, s: str) -> None:
        self._add(cat, {"kind": "encode", "text": s}, ("e", s))

    def raw(self, cat: str, b: bytes) -> None:
        self._add(cat, {"kind": "encode_bytes", "bytes": b.hex()}, ("b", b))

    def decode(self, cat: str, ids: list[int]) -> None:
        self._add(cat, {"kind": "decode", "ids": list(ids)}, ("d", tuple(ids)))

    def reject(self, cat: str, s: str) -> None:
        self._add(cat, {"kind": "encode_reject", "utf16": utf16_hex(s)}, ("r", s))


def utf16_hex(s: str) -> str:
    """The UTF-16 code units of s, big-endian, as lowercase hex: 4 digits per unit. Lone surrogates pass through."""
    return s.encode("utf-16-be", errors="surrogatepass").hex()


def from_utf16_hex(h: str) -> str:
    """The inverse of utf16_hex. A high-low surrogate pair becomes one astral character, as in .NET strings."""
    return bytes.fromhex(h).decode("utf-16-be", errors="surrogatepass")


# ---------------------------------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------------------------------


def snippets(name: str) -> list[str]:
    with open(os.path.join(SRC, name), encoding="utf-8", newline="") as f:
        text = f.read()
    if "\r" in text:
        raise c.GoldenError(f"{name} must be LF only; CR variants are generated")
    parts = text.split("\n%%\n")
    return [p.rstrip("\n") for p in parts[1:]]  # parts[0] is the header comment


def lines(name: str) -> list[str]:
    with open(os.path.join(SRC, name), encoding="utf-8", newline="") as f:
        return [ln for ln in f.read().split("\n") if ln and not ln.startswith("#")]


def code_cases(corpus: Corpus, cat: str, snips: list[str]) -> None:
    rng = _rng(cat)
    for s in snips:
        corpus.text(cat, s)
        corpus.text(cat, s + "\n")
        corpus.text(cat, s.replace("\n", "\r\n") + "\r\n")
        corpus.text(cat, s.replace("    ", "\t"))
        corpus.text(cat, s.replace("    ", "  "))
    for s in snips:
        ls = s.split("\n")
        for ln in ls:
            corpus.text(cat, ln)
        for i in range(len(ls) - 2):
            eol = rng.choice(["\n", "\r\n", "\n", "\r"])
            corpus.text(cat, eol.join(ls[i:i + 3]) + rng.choice(["", eol]))


def prose_cases(corpus: Corpus) -> None:
    for ln in lines("prose.txt"):
        corpus.text("prose", ln)
        corpus.text("prose", ln + "\n")


def multilingual_cases(corpus: Corpus) -> None:
    for ln in lines("multilingual.txt"):
        corpus.text("multilingual", ln)
        nfd = unicodedata.normalize("NFD", ln)
        if nfd != ln:
            corpus.text("nfc", nfd)
        for w in ln.split(" "):
            if w:
                corpus.text("multilingual", w)


# ---------------------------------------------------------------------------------------------------------------------
# Targeted categories
# ---------------------------------------------------------------------------------------------------------------------

UNICODE_SPACES = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x85", "\xa0", " "] + [chr(cp) for cp in range(0x2000, 0x200B)] + [
    " ", " ", " ", " ", "　"]
NOT_SPACES = ["᠎", "​", "‌", "‍", "⁠", "﻿", "\x1c", "\x1d", "\x1e", "\x1f"]


def whitespace_cases(corpus: Corpus) -> None:
    cat = "whitespace"
    for n in range(1, 33):
        for unit in (" ", "\t", "\n"):
            corpus.text(cat, unit * n)
    for n in range(1, 9):
        for unit in ("\r\n", "\r", " \n", "\t\n", "\n\n "):
            corpus.text(cat, unit * n)
    for n in range(1, 21):
        sp = " " * n
        for t in (f"a{sp}b", f"a{sp}", f"{sp}a", f"a{sp}\nb", f"a\n{sp}b", f"x{chr(9) * n}y", f"a{sp}\r\nb", f"{sp}\r{sp}",
                  f"if (x){sp}{{", f"a,{sp}b"):
            corpus.text(cat, t)
    for t in (" \t \t", "\t \n", " \r\n ", "\r\n\r\n", "\n\r", "a\rb", "a\r\nb", "a\n\rb", "\r", "\r\r\n", " \r", "\r ",
              "a \r\n b", "\n \n", " \n \n ", "foo   \nbar", "foo\t\nbar", "line1\rline2", "\r\n\r", "a\r\r\nb", "end\r",
              "\rstart", "tab\t\r", "  \r  ", "\r\r\r\n\n\n", "x\r\n\r\n\r\ny", "}\r\n\r\n    return x;\r\n}", "\n\n\n\nx",
              "x\n\n\n\n", " \n", "\n ", "a  \n  b", "a\t\t\nb", "Trailing spaces at the end of this line.   "):
        corpus.text(cat, t)
    for ws in UNICODE_SPACES + NOT_SPACES:
        for t in (ws, f"a{ws}b", ws + ws, f"a{ws}", f"{ws}a", f"a {ws} b", f"{ws}\n", f"\n{ws}", f"x{ws * 3}y", f"'{ws}s",
                  f"1{ws}2", f"{ws}{ws} x"):
            corpus.text("unicode-space", t)


def digit_cases(corpus: Corpus) -> None:
    cat = "digits"
    for d in range(10):
        corpus.text(cat, str(d))
    for k in range(1, 26):
        corpus.text(cat, "1" + "0" * k)
    rng = _rng(cat)
    for n in range(1, 41):
        corpus.text(cat, "".join(str(rng.next_below(10)) for _ in range(n)))
    for t in ("12345678901234567890", "3.14159265358979", "-42", "+7", "1,234,567.89", "1e-10", "6.02e23", "0x7FFF", "0b1010",
              "1_000_000", "v1.2.3", "2026-10-18", "192.168.0.1", "14:30:00", "1/2", "100%", "$5", "€20", "5€", "¥100", "£3",
              "123abc", "abc123", "a1b2c3", "1st 2nd 3rd 4th", "x1 y22 z333", "0.1f", "1e6", "1E+06", "9_223_372_036_854_775_807L",
              "½", "¼", "¾", "⅓", "²", "³", "¹", "x²+y²", "H₂O", "Ⅻ", "ⅻ", "①②③", "٠١٢٣٤٥٦٧٨٩", "۰۱۲۳", "०१२३४५६७८९", "३.१४",
              "০১২", "๐๑๒", "０１２３４５６７８９", "１２３abc", "𝟎𝟏𝟐𝟑", "𝟘𝟙", "𑁦𑁧", "〇一二三", "٣٫١٤", "1٢3", "⑴⒈", "⓪", "₀₁₂"):
        corpus.text(cat, t)


# \p{N} has three classes, Nd, Nl and No, and the Split regex treats them alike: each is one pre-token, and a space
# before one is its own pre-token (` ?[^\s\p{L}\p{N}]+` excludes \p{N}). An engine that classifies only Nd as a
# number merges ' ½' into one pre-token. These contexts catch a classifier that misses Nl or No.
NUMBER_NL = ["Ⅻ", "ⅻ", "〇", "Ⅰ", "〡", "\U00010140", "ᛮ", "ↂ"]
NUMBER_NO = ["½", "²", "①", "₀", "⑴", "༳", "↉", "⓪", "¼", "³", "\U0001d360", "⁵"]


def number_class_cases(corpus: Corpus) -> None:
    for cls, chars in (("Nl", NUMBER_NL), ("No", NUMBER_NO)):
        for n in chars:
            if unicodedata.category(n) != cls:
                raise c.GoldenError(f"U+{ord(n):04X} is {unicodedata.category(n)}, not {cls}")
            for ctx in (" {n}", "x {n}", "{n}x", "{n}{n}", "({n})", " {n}.", "{n}1", "1{n}", " {n}1", "a {n} b", "{n} {n}",
                        "\n{n}", "'{n}"):
                corpus.text("number-class", ctx.format(n=n))


CONTRACTIONS = ["s", "t", "re", "ve", "m", "ll", "d"]


def _case_variants(s: str) -> list[str]:
    return ["".join(p) for p in itertools.product(*[(ch.lower(), ch.upper()) for ch in s])]


def contraction_cases(corpus: Corpus) -> None:
    cat = "contractions"
    contexts = ["{a}{c}", "it{a}{c}", "IT{a}{c}", " {a}{c}", "x {a}{c} y", "{a}{c}x", "{a}{c} ", "{a}{c}\n", "{a}{c}{a}{c}",
                "word{a}{c}word", "1{a}{c}", "{a}{c}1", "\n{a}{c}", "\t{a}{c}"]
    for base in CONTRACTIONS:
        for v in _case_variants(base):
            for ctx in contexts:
                corpus.text(cat, ctx.format(a="'", c=v))
    for apo in ("’", "ʼ", "＇", "`", "´"):
        for base in CONTRACTIONS:
            for ctx in ("{a}{c}", "it{a}{c}", " {a}{c} "):
                corpus.text(cat, ctx.format(a=apo, c=base))
    # Unicode case folding in the (?i:...) group: long s and other folding look-alikes.
    folds = ["ſ", "ſt", "İ", "ı", "K", "ﬅ", "ﬆ", "ß", "ẞ", "ǆ", "Ｓ",
             "ｓ", "ꜱ", "ˢ", "ⓢ", "ʀᴇ", "ʟʟ", "ᴅ", "ᴍ", "ᴛ", "ѕ",
             "σ", "Σ", "ſS", "Sſ", "ſſ", "Å", "́s", "ś", "İ̇", "Le",
             "lL", "rE", "Ve", "vE", "M", "D"]
    for f in folds:
        for ctx in ("'{f}", "it'{f}", " '{f} ", "'{f}\n", "x'{f}y", "'{f}'{f}"):
            corpus.text("casefold", ctx.format(f=f))
    for t in ("''s", "'''", "'", "' s", "'S'", "'sss", "'tis", "'Twas", "'twas", "I'm", "you're", "we've", "they'll", "she'd",
              "it's", "don't", "DON'T", "Don'T", "won't", "o'clock", "rock'n'roll", "'n'", "y'all", "ma'am", "l'élève", "aujourd'hui",
              "Bob's", "BOB'S", "James'", "the '90s", "'80s", "'s'", "'S's", "x='s'", "char c = 's';", "@'s", "#'t", "''"):
        corpus.text(cat, t)


EMOJI = ["😀", "😀😃😄", "👍", "👍🏽", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🏴‍☠️", "🇩🇪", "🇯🇵🇨🇳", "1️⃣",
         "#️⃣", "❤️", "❤", "☺️", "✔️", "🫠", "🪿", "🧑‍💻", "👩🏾‍🚀", "🤷‍♂️", "🦀",
         "🐍", "🚀✨", "🏻", "‍", "😀‍😃", "🧑🏿‍🤝‍🧑🏻", "🇺🇳🇪🇺", "🔟", "🆗", "Ⓜ️", "🀄", "🃏", "🎴",
         "©️®️™️", "↔️", "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"]


def emoji_cases(corpus: Corpus) -> None:
    for e in EMOJI:
        for ctx in ("{e}", "a{e}b", "{e} a", "a {e}", "{e}\n", "123{e}", "{e}123", "'{e}", "x={e};", "// {e} note", '"{e}"', "{e}{e}{e}"):
            corpus.text("emoji", ctx.format(e=e))
    corpus.text("emoji", "😀" * 10)


ASTRAL_LETTERS = ["𝐀𝐁𝐂", "𝐚𝐛𝐜", "𝒜𝒷𝒸", "𝔄𝔅", "𝕏𝕐", "𝖆", "𝗔", "𝘈", "𝙰", "𝚨𝚩", "𐐀𐐁𐐨", "𐌰𐌱𐌲", "𐎀𐎁", "𠀀𠀁𠮷", "𪚥",
                  "𓀀𓀁", "𞤀𞤁𞤢", "𖹀𖹠", "𑀅𑀆", "𐒀", "𐤀", "𐀀", "𐐷"]
ASTRAL_OTHER = ["𝄞", "𐄀", "𝟎", "🀄", "𑁇", "\U0001f3fb", "𝅘𝅥𝅮", "\U000e0001", "\U000f0000", "\U0010fffd", "\U0010ffff"]
# Characters added in Unicode 15 and 16 (Kawi, Nag Mundari, Cyrillic Extended-D, Garay, Kirat Rai, Ol Onal, Sunuwar,
# Tulu-Tigalari); engines with older Unicode tables classify them differently from newer ones.
UNICODE_NEW = [0x11F04, 0x11F05, 0x1E4D0, 0x1E4D1, 0x1E030, 0x1E031, 0x10D50, 0x10D51, 0x16D43, 0x16D44, 0x1E5D0, 0x1E5D1,
               0x11BC0, 0x11BC1, 0x11380, 0x11381, 0x1CCF0, 0x1FAE9]


def astral_cases(corpus: Corpus) -> None:
    for a in ASTRAL_LETTERS:
        for ctx in ("{a}", "a{a}b", " {a}", "{a} ", "'{a}", "x{a}y z", "{a}1", "1{a}", "{a}\n{a}", "Ab{a}Cd", "{a}{a}{a}", "'s{a}"):
            corpus.text("astral", ctx.format(a=a))
    corpus.text("astral", "𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙")
    corpus.text("astral", "𠮷野家で𠮷田さんと𩸽を食べた")
    for o in ASTRAL_OTHER:
        for ctx in ("{o}", "a{o}b", " {o} ", "{o}{o}"):
            corpus.text("astral", ctx.format(o=o))
    for cp in UNICODE_NEW:
        ch = chr(cp)
        for ctx in ("{x}", "a{x}b", " {x}{x}", "1{x}"):
            corpus.text("unicode-version", ctx.format(x=ch))


def added_token_cases(corpus: Corpus, added: list[dict]) -> None:
    cat = "added"
    for t in (a["content"] for a in added):
        for s in (t, f"a{t}b", f" {t} ", f"{t}\n", f"\n{t}", t + t, f"{t} {t}", f"x{t}", f"{t}x", f"1{t}2", t[:-1], t[1:],
                  t.upper(), t[: len(t) // 2] + " " + t[len(t) // 2:], f'var s = "{t}";', f"{t}\r\n", f"\t{t}\t", f"'{t}'s",
                  f"{t}😀", f"中{t}文"):
            corpus.text(cat, s)
    corpus.text(cat, "".join(a["content"] for a in added))
    corpus.text(cat, " ".join(a["content"] for a in added))
    chatml = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a C# method that adds two "
              "ints.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n```csharp\npublic static int Add(int a, int b) => a + b;\n"
              "```<|im_end|>\n")
    corpus.text("chatml", chatml)
    corpus.text("chatml", chatml.replace("\n", "\r\n"))
    corpus.text("chatml", "<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n")
    corpus.text("chatml", '<tool_call>\n{"name": "compile", "arguments": {"code": "class A {}"}}\n</tool_call>')
    corpus.text("chatml", "<tool_response>\n{\"diagnostics\": []}\n</tool_response>")
    corpus.text("chatml", "Some text.<|endoftext|>")
    corpus.text("chatml", "<|endoftext|><|endoftext|>Next document")
    corpus.text("chatml", "<|im_start|>assistant\n<think>\nThe user asks for 'Add'. It's simple.\n</think>\n\nHere it is.<|im_end|>")
    corpus.text("chatml", "<|fim_prefix|>int Add(int a, int b) {<|fim_suffix|>}<|fim_middle|>return a + b;")


def unicode_cases(corpus: Corpus) -> None:
    for cp in range(0x00, 0x100):
        corpus.text("codepoint", chr(cp))
    for cp in range(0x00, 0x100, 7):
        corpus.text("codepoint", f"a{chr(cp)}b")
    nfc = ["é", "é̂", "́", "ä", "Å", "Å", "Ω", "K", "Å", "ﬁ",
           "Ǆ", "ǅ", "ǆ", "각", "가", "각", "क्ष", "क़",
           "क़", "豈", " ", " ", "̀", "ʹ", ";", "·", "ά", "­", "‎",
           "‮", "﻿BOM at start", "�", "￾", "￿", "", "naïve café résumé", "naïve café",
           "ﬁnancial ﬂow", "ộ", "ộ", "ṩ", "ṩ", "ṩ",
           "ୋ", "เา", "x͏y", "é́", "Ǻ", "ﾞ", "ｶﾞ", "が", "が", "ⷵ"]
    for t in nfc:
        corpus.text("nfc", t)
        corpus.text("nfc", f"word{t}word")
        corpus.text("nfc", f" {t} ")


INVALID = [b"\x80", b"\xbf", b"\xc0\x80", b"\xc1\xbf", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xed\xbf\xbf", b"\xf4\x90\x80\x80",
           b"\xf5\x80\x80\x80", b"\xf8\x88\x80\x80\x80", b"\xfc\x84\x80\x80\x80\x80", b"\xfe", b"\xff", b"\xff\xfe", b"\xe4\xb8",
           b"\xe4", b"\xf0\x9f\x98", b"\xf0\x9f", b"\xc3", b"a\x80b", b"abc\xe4\xb8", b"\xe4\xb8abc", b"\xe4\xb8\xad\xe4",
           b"\xc3\xa9\xc3", b"\xc3\x28", b"\xa0\xa1", b"\xe2\x28\xa1", b"\xe2\x82\x28", b"\xf0\x28\x8c\xbc", b"\xf0\x90\x28\xbc",
           b"\xf0\x28\x8c\x28", b" \x80 ", b"\r\n\x80\r\n", b"'\xc5", b"it'\x80s", b"<|im_end|>\x80<|im_start|>", b"\x80<|endoftext|>",
           b"\xed\xa0\xbd\xed\xb8\x80", b"\xef\xbb\xbfint x;", b"hello", "héllo".encode(), "中文".encode(), "😀".encode(),
           b"\xe0\xa0", b"\xf0\x90\x80", b"\xf4\x8f\xbf\xbf", b"\xf4\x8f\xbf", b"\xc2\x80", b"\xdf\xbf", b"\xe0\xa0\x80",
           b"\xef\xbf\xbf", b"\xf0\x90\x80\x80", b"\x00", b"\x7f", b"\x80\x80\x80", b"\xc0", b"\xc0\xc0", b"\xe1\x80\xe1\x80",
           b"x\xf0\x9f\x98\x80\x80y", b"   \xff   ", b"\t\xfe\n"]


def invalid_utf8_cases(corpus: Corpus) -> None:
    for b in INVALID:
        corpus.raw("utf8", b)
    rng = _rng("utf8-random")
    for _ in range(40):
        n = rng.next_int(1, 17)
        corpus.raw("utf8", bytes(rng.next_below(256) for _ in range(n)))
    base = "Grüße, 世界! 😀 public int X => 42;".encode("utf-8")
    for _ in range(40):
        b = bytearray(base)
        b[rng.next_below(len(b))] = rng.next_below(256)
        corpus.raw("utf8", bytes(b))


def reject_cases(corpus: Corpus) -> None:
    for s in ("\ud800", "\udc00", "a\ud800b", "\udc00\ud800", "x\ud83d", "\ud83d x", "\udfff", "ok \udbff"):
        corpus.reject("surrogate", s)


def long_cases(corpus: Corpus, cs: list[str]) -> None:
    for s in ("\n\n".join(cs[:6]), "a" * 1000, " " * 1000, "1" * 300, "中" * 300, "😀" * 100, "ab" * 500, "\n" * 300, "\t" * 200,
              "-" * 500, "=" * 300, "/*" + "*" * 400 + "*/", "x" + " " * 500 + "y", "'s" * 200, "\r\n" * 200):
        corpus.text("long", s)
    rng = _rng("long")
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    corpus.text("long", "".join(rng.choice(letters) for _ in range(2000)))
    corpus.text("long", "".join(chr(0x4E00 + rng.next_below(0x5000)) for _ in range(500)))


def mixture_cases(corpus: Corpus, pool: list[str]) -> None:
    rng = _rng("mixture")
    seps = [" ", "", "\n", "\t", "  ", "\r\n", ", ", ".", "'", "_", "\r", " \n", "(", ")", "::"]
    for _ in range(400):
        k = rng.next_int(1, 13)
        parts = []
        for _ in range(k):
            parts.append(rng.choice(pool))
            parts.append(rng.choice(seps))
        corpus.text("mixture", "".join(parts[:-1] if rng.next_below(2) else parts))


def decode_cases(corpus: Corpus, tok, added: list[dict]) -> None:
    byte_ids = byte_token_ids(tok)
    for b in range(256):
        corpus.decode("decode", [byte_ids[b]])
    for a in added:
        corpus.decode("decode", [a["id"]])
        corpus.decode("decode", [a["id"], byte_ids[ord("x")], a["id"]])
    utf8_seqs = ["中".encode(), "😀".encode(), "é".encode(), "€".encode()]
    for seq in utf8_seqs:
        ids = [byte_ids[b] for b in seq]
        for cut in range(1, len(ids) + 1):
            corpus.decode("decode", ids[:cut])
            corpus.decode("decode", ids[:cut] + [byte_ids[ord("a")]])
        corpus.decode("decode", ids[::-1])
        corpus.decode("decode", ids[:1] + [added[2]["id"]] + ids[1:])
    rng = _rng("decode")
    for _ in range(120):
        n = rng.next_int(1, 21)
        corpus.decode("decode", [rng.next_below(151643) for _ in range(n)])
    for _ in range(40):
        n = rng.next_int(1, 11)
        corpus.decode("decode", [rng.choice([rng.next_below(151643), rng.choice(added)["id"], byte_ids[rng.next_below(256)]]) for _ in range(n)])
    # The empty list, and lm_head rows with no token (REAL_IDS..MODEL_ROWS-1): HF drops them and they add no bytes.
    zh = [byte_ids[b] for b in "中".encode()]
    for ids in ([], [REAL_IDS], [MODEL_ROWS - 1], [REAL_IDS, MODEL_ROWS - 1], [byte_ids[ord("I")], 151700, byte_ids[ord("J")]],
                zh[:1] + [151700] + zh[1:], [151700, added[0]["id"]], [added[0]["id"], MODEL_ROWS - 1, added[1]["id"]]):
        corpus.decode("decode", ids)
    for _ in range(16):
        n = rng.next_int(1, 9)
        corpus.decode("decode", [rng.choice([rng.next_below(151643), REAL_IDS + rng.next_below(MODEL_ROWS - REAL_IDS), byte_ids[rng.next_below(256)]])
                                 for _ in range(n)])


# ---------------------------------------------------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------------------------------------------------


def bytes_to_unicode() -> dict:
    """GPT-2 byte-level map: byte -> printable character (what ByteLevel uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def byte_token_ids(tok) -> list[int]:
    m = bytes_to_unicode()
    ids = [tok.token_to_id(m[b]) for b in range(256)]
    if any(i is None for i in ids):
        raise c.GoldenError("a single-byte token is missing from the vocabulary")
    return ids


def token_bytes(tok, i: int, added_by_id: dict) -> bytes:
    if i in added_by_id:
        return added_by_id[i].encode("utf-8")
    if REAL_IDS <= i < MODEL_ROWS:
        if tok.id_to_token(i) is not None:
            raise c.GoldenError(f"id {i} has a token, but ids from {REAL_IDS} on should have none")
        return b""
    if not 0 <= i < REAL_IDS:
        raise c.GoldenError(f"decode case id {i} is outside the model's {MODEL_ROWS} rows")
    inv = {v: k for k, v in bytes_to_unicode().items()}
    return bytes(inv[ch] for ch in tok.id_to_token(i))


def build_corpus(tok, added: list[dict]) -> Corpus:
    corpus = Corpus()
    cs, il = snippets("csharp.txt"), snippets("il.txt")
    code_cases(corpus, "csharp", cs)
    code_cases(corpus, "il", il)
    prose_cases(corpus)
    multilingual_cases(corpus)
    whitespace_cases(corpus)
    digit_cases(corpus)
    number_class_cases(corpus)
    contraction_cases(corpus)
    emoji_cases(corpus)
    astral_cases(corpus)
    added_token_cases(corpus, added)
    unicode_cases(corpus)
    invalid_utf8_cases(corpus)
    reject_cases(corpus)
    long_cases(corpus, cs)
    pool = sorted({w for s in cs + il for w in s.replace("\n", " ").split(" ") if w} |
                  {w for ln in lines("multilingual.txt") + lines("prose.txt") for w in ln.split(" ") if w} |
                  set(EMOJI) | set(ASTRAL_LETTERS) | {"'s", "'T", "'ll", "'ſ", "42", "3.14", " ", "\x85", "\xa0", "　"} |
                  {a["content"] for a in added[:4]})
    mixture_cases(corpus, pool)
    decode_cases(corpus, tok, added)
    return corpus


def run_oracle(tok, corpus: Corpus, added: list[dict]) -> tuple[bytes, dict]:
    added_by_id = {a["id"]: a["content"] for a in added}
    checks = {"no_bos": 0, "round_trip_nfc": 0, "round_trip_mismatch": [], "decode_equals_replace": 0, "rejected": 0, "cases": 0}
    out_lines = []
    for case in corpus.cases:
        kind = case["kind"]
        if kind in ("encode", "encode_bytes"):
            if kind == "encode_bytes":
                case["text"] = bytes.fromhex(case["bytes"]).decode("utf-8", errors="replace")
            text = case["text"]
            ids = tok.encode(text, add_special_tokens=False).ids
            if tok.encode(text, add_special_tokens=True).ids != ids:
                raise c.GoldenError(f"{case['id']}: add_special_tokens changes the ids (BOS?)")
            checks["no_bos"] += 1
            case["ids"] = ids
            if tok.decode(ids, skip_special_tokens=False) == unicodedata.normalize("NFC", text):
                checks["round_trip_nfc"] += 1
            else:
                checks["round_trip_mismatch"].append(case["id"])
        elif kind == "decode":
            ids = case["ids"]
            raw = b"".join(token_bytes(tok, i, added_by_id) for i in ids)
            text = tok.decode(ids, skip_special_tokens=False)
            if text != raw.decode("utf-8", errors="replace"):
                raise c.GoldenError(f"{case['id']}: tokenizers decode differs from bytes decoded with errors=replace")
            checks["decode_equals_replace"] += 1
            case["text"] = text
            case["bytes"] = raw.hex()
        elif kind == "encode_reject":
            text = from_utf16_hex(case["utf16"])
            if utf16_hex(text) != case["utf16"]:
                raise c.GoldenError(f"{case['id']}: utf16 does not round-trip")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                pass
            else:
                raise c.GoldenError(f"{case['id']}: the text holds no lone surrogate")
            try:
                tok.encode(text, add_special_tokens=False)
            except Exception as e:  # noqa: BLE001  (PyO3 raises UnicodeEncodeError or a TypeError subclass)
                case["error"] = "lone surrogate"
                checks["rejected"] += 1
                checks.setdefault("reject_exception", type(e).__name__)
            else:
                raise c.GoldenError(f"{case['id']}: tokenizers accepted a lone surrogate")
        checks["cases"] += 1
        out_lines.append(json.dumps(case, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
    return ("\n".join(out_lines) + "\n").encode("ascii"), checks


def transformers_crosscheck(model_dir: str, corpus: Corpus) -> dict:
    """The same ids through transformers' own tokenizer class, as a second opinion on the encode cases."""
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    agree, differ = 0, []
    for case in corpus.cases:
        if case["kind"] in ("encode", "encode_bytes"):
            if t.encode(case["text"], add_special_tokens=False) == case["ids"]:
                agree += 1
            else:
                differ.append(case["id"])
    return {"class": type(t).__name__, "agree": agree, "differ": differ}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    ap.add_argument("--out", default=os.path.join(c.REPO_ROOT, *OUT_REL.split("/")), help="output directory")
    ap.add_argument("--no-telemetry", action="store_true", help="do not write a run log under LRM_DATA/runs")
    args = ap.parse_args()

    log = c.RunLog("make_tokenizer", enabled=not args.no_telemetry, kind="goldens")
    c.setup_torch(8)
    guards = c.check_effective_guards()
    from tokenizers import Tokenizer

    tc = c.toolchain()["model"]
    model_dir = os.path.join(c.data_root(), *tc["path"].split("/"))
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok_sha = c.sha256_file(tok_path)
    if tok_sha != tc["sha256"]["tokenizer.json"]:
        raise c.GoldenError(f"tokenizer.json SHA-256 {tok_sha} is not the pinned {tc['sha256']['tokenizer.json']}")
    with open(os.path.join(model_dir, "REVISION"), encoding="ascii") as f:
        revision = f.read().strip()
    if revision != tc["revision"]:
        raise c.GoldenError(f"model REVISION {revision} is not the pinned {tc['revision']}")
    tok = Tokenizer.from_file(tok_path)
    with open(tok_path, encoding="utf-8") as f:
        tj = json.load(f)
    added = [{"id": a["id"], "content": a["content"], "special": a["special"]} for a in tj["added_tokens"]]
    if len(added) != 26 or tok.get_vocab_size(with_added_tokens=True) != REAL_IDS:
        raise c.GoldenError("unexpected added tokens or vocabulary size")
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        if json.load(f)["vocab_size"] != MODEL_ROWS:
            raise c.GoldenError(f"the model's vocab_size is not {MODEL_ROWS}")
    for a in added:
        if tok.encode(a["content"], add_special_tokens=False).ids != [a["id"]]:
            raise c.GoldenError(f"added token {a['content']} does not encode to its id")

    def build(_threads):
        corpus = build_corpus(tok, added)
        data, checks = run_oracle(tok, corpus, added)
        build.corpus, build.checks = corpus, checks
        return {"cases.jsonl": data}

    with log.span("goldens.tokenizer.build", set="tokenizer"):
        files, dbl = c.double_run(build)
    corpus, checks = build.corpus, build.checks
    with log.span("goldens.tokenizer.crosscheck", set="tokenizer"):
        cross = transformers_crosscheck(model_dir, corpus)
    if len(corpus.cases) < MIN_CASES:
        raise c.GoldenError(f"only {len(corpus.cases)} cases; the corpus needs at least {MIN_CASES}")
    by_cat, by_kind = {}, {}
    for case in corpus.cases:
        by_cat[case["cat"]] = by_cat.get(case["cat"], 0) + 1
        by_kind[case["kind"]] = by_kind.get(case["kind"], 0) + 1
    rt_bad = [i for i in checks["round_trip_mismatch"] if not i.startswith(("utf8-",))]
    man = {
        "schema": c.GOLDEN_SCHEMA,
        "set": SET,
        "version": VERSION,
        "kind": "tokenizer-corpus",
        "about": "Qwen3 tokenizer corpus for M1 (TokenizerApi): ids from HF tokenizers for C# and IL text, CRLF and lone CR, tabs, "
                 "space runs, Unicode spaces, digits, contractions with case folding, de/zh/ja and other scripts, emoji, astral "
                 "letters, newer Unicode, every added token, NFC, invalid UTF-8, lone surrogates and decode cases.",
        "generator": c.generator_record([os.path.join(c.TOOLS_DIR, *f.split("/")) for f in TOOL_FILES],
                                        "pwsh tools/goldens/run.ps1 make_tokenizer.py"),
        "env": c.env_snapshot(dbl["threads"]),
        "guards": guards,
        "model": {"path": "LRM_DATA/" + tc["path"], "revision": revision, "files": {"tokenizer.json": tok_sha}},
        "oracle": {"library": "tokenizers 0.23.3", "load": "Tokenizer.from_file(tokenizer.json)",
                   "encode": "encode(text, add_special_tokens=False).ids", "decode": "decode(ids, skip_special_tokens=False)",
                   "normalizer": tj["normalizer"], "pre_tokenizer": tj["pre_tokenizer"], "decoder": tj["decoder"]},
        "semantics": {
            "encode": "ids = Encode(text): added tokens first, NFC, the Split regex, byte-level BPE; no BOS",
            "encode_bytes": "text = bytes decoded as UTF-8 with U+FFFD per maximal invalid subpart (errors=replace); ids = Encode(text)",
            "decode": "bytes = concatenated token bytes (DecodeBytes); text = Decode(ids) = bytes decoded with errors=replace; "
                      f"ids {REAL_IDS}..{MODEL_ROWS - 1} (lm_head rows with no token) add no bytes",
            "encode_reject": "text = the UTF-16 code units in utf16 (big-endian hex, 4 digits per unit); Encode(text) raises: "
                             "the text holds a lone surrogate",
        },
        "added_tokens": added,
        "vocab_real_ids": REAL_IDS,
        "vocab_model_rows": MODEL_ROWS,
        "counts": {"cases": len(corpus.cases), "by_kind": dict(sorted(by_kind.items())), "by_category": dict(sorted(by_cat.items()))},
        "checks": {"no_bos": checks["no_bos"], "round_trip_nfc": checks["round_trip_nfc"],
                   "round_trip_mismatch": checks["round_trip_mismatch"], "decode_equals_errors_replace": checks["decode_equals_replace"],
                   "rejected": checks["rejected"], "reject_exception": checks.get("reject_exception"),
                   "transformers": cross},
        "threads": dbl["threads"],
        "double_run": dbl,
        "eos": {"ignored": True, "note": "the tokenizer adds no BOS or EOS; EOS ids are ordinary added tokens here"},
        "files": [c.file_entry(f"{OUT_REL}/cases.jsonl", files["cases.jsonl"], "cases")],
    }
    if rt_bad:
        raise c.GoldenError(f"decode(encode(text)) != NFC(text) for {rt_bad[:10]}")
    if cross["differ"]:
        print("note: transformers' tokenizer differs on", len(cross["differ"]), "cases:", cross["differ"][:10])
    files["manifest.json"] = c.json_bytes(man)
    for cat, n in sorted(by_cat.items()):
        log.gauge("goldens.tokenizer.cases", n, "count", category=cat)
    for kind, n in sorted(by_kind.items()):
        log.gauge("goldens.tokenizer.case_kinds", n, "count", kind=kind)
    log.check("goldens.tokenizer.transformers", not cross["differ"], agree=cross["agree"], differ=len(cross["differ"]),
              tokenizer_class=cross["class"])
    log.check("goldens.double_run_identical", dbl["identical"], set="tokenizer", threads=dbl["threads"])
    total = sum(len(b) for b in files.values())
    for n, b in sorted(files.items()):
        log.gauge("goldens.file_bytes", len(b), "B", file=f"{OUT_REL}/{n}")
    log.gauge("goldens.set_bytes", total, "B", set="tokenizer", budget=SET_BUDGETS["tokenizer"])
    print(f"{len(corpus.cases)} cases ({', '.join(f'{k} {v}' for k, v in sorted(by_kind.items()))}); "
          f"{len(by_cat)} categories; {total} bytes; transformers agrees on {cross['agree']}")
    if args.check:
        problems = c.compare_tree(args.out, files)
        log.check("goldens.regen_identical", not problems, set="tokenizer")
        log.close("failed" if problems else "ok")
        if problems:
            print("CHECK FAILED:", *problems, sep="\n  ")
            return 1
        print("check: regenerated bytes are identical to", args.out)
        return 0
    c.write_tree(args.out, files)
    log.close()
    print("wrote", args.out, "| run log:", log.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
