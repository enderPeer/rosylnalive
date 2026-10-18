"""The committed corpora, the seeded prompt pool and the fixed prompt rules of goldens/qwen3-0.6b/g1 (workstream G2).

Everything here is a pure function of the committed corpus files in tools/goldens/qwen/corpus/, Lrm.Core's DetRng
(tools/goldens/common.py) and the pinned tokenizer, so the pool regenerates bit for bit. Nothing here runs a model:

- the corpora: C# snippets written for this repo, the Apache License 2.0 paragraphs (the LICENSE file shipped with the
  Qwen3-0.6B weights, copied byte for byte), design-doc paragraphs (a snapshot of docs/LRM-design.md), multilingual
  lines and ChatML wrappers;
- the pool: POOL_SIZE seeded candidates in fixed category quotas, each 8..900 tokens;
- the calibration subset: 12 candidates chosen by category and length only, before any model runs;
- the long prompts: two fixed concatenations of about 1,000 and at most 4,000 tokens.

The gate rule, which needs the torch32 margins, is in make_qwen.py (select_gate).
"""

from __future__ import annotations

import json
import os
import textwrap

import common as c

SET = "qwen3-0.6b"
VERSION = "g1"
SEED = 0x6732  # "g2"
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
CORPUS_FILES = ("apache-2.0.txt", "chatml.json", "csharp.txt", "design-doc.md", "multilingual.txt")

# Category quotas, in pool order. The pool is the concatenation of each category's candidates k = 0, 1, ...
CATEGORIES = ("csharp", "license", "design", "multilingual", "chatml")
POOL_QUOTAS = {"csharp": 48, "license": 32, "design": 32, "multilingual": 32, "chatml": 56}
POOL_SIZE = sum(POOL_QUOTAS.values())  # 200
MIN_TOKENS = 8
MAX_TOKENS = 900  # the gate prompts' limit (dev plan, section 6)
MAX_ATTEMPTS = 200  # redraws per candidate before the pool builder gives up

# Calibration: per category, the first candidate below SHORT_TOKENS and the first at or above it, plus the two longest
# candidates of the pool: 12 prompts, fixed before any model runs.
SHORT_TOKENS = 160
CALIBRATION_SIZE = 12

# Long prompts (dev plan, section 6: about 1,000 and at most 4,000 tokens; logits only).
LONG_MAX_TOKENS = 4000
LONG_SPECS = (
    ("long-design", "design", 1000),  # design-doc paragraphs from the start until at least 1,000 tokens
    ("long-csharp", "csharp", 3600),  # C# snippets in file order until at least 3,600 tokens
)

ENGLISH = ("license", "design")  # the English categories (the gate needs English, C# and ChatML)


def _read(name: str) -> str:
    with open(os.path.join(CORPUS_DIR, name), "rb") as f:
        raw = f.read()
    if b"\r" in raw:
        raise c.GoldenError(f"corpus/{name}: CR in a corpus file (LF only)")
    return raw.decode("utf-8")


def _paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks, dedented, trailing spaces removed, blank blocks dropped."""
    out = []
    for block in text.split("\n\n"):
        lines = [ln.rstrip() for ln in textwrap.dedent(block).split("\n")]
        p = "\n".join(lines).strip("\n")
        if p.strip():
            out.append(p)
    return out


def load_corpus() -> dict:
    """The five corpora as lists of texts (and the ChatML wrappers)."""
    snippets = []
    name, buf = None, []
    for line in _read("csharp.txt").split("\n"):
        if line.startswith("//// snippet: "):
            if name is not None:
                snippets.append((name, "\n".join(buf).strip("\n")))
            name, buf = line[len("//// snippet: "):].strip(), []
        else:
            buf.append(line)
    if name is not None:
        snippets.append((name, "\n".join(buf).strip("\n")))
    if len({n for n, _ in snippets}) != len(snippets):
        raise c.GoldenError("corpus/csharp.txt: duplicate snippet names")

    license_paras = _paragraphs(_read("apache-2.0.txt"))
    # Design-doc prose: every block except headings and tables.
    design_paras = [p for p in _paragraphs(_read("design-doc.md")) if not p.startswith(("#", "|"))]
    lines = []
    for ln in _read("multilingual.txt").split("\n"):
        if ln:
            lang, text = ln.split("\t", 1)
            lines.append((lang, text))
    chat = json.loads(_read("chatml.json"))
    return {"csharp": snippets, "license": license_paras, "design": design_paras, "multilingual": lines, "chatml": chat}


def corpus_record(corpus: dict) -> dict:
    """What the manifest says about the corpora: files with their SHA-256 (CRLF folded) and item counts."""
    return {
        "files": {f"tools/goldens/qwen/corpus/{n}": c.sha256_source(os.path.join(CORPUS_DIR, n)) for n in CORPUS_FILES},
        "items": {"csharp_snippets": len(corpus["csharp"]), "license_paragraphs": len(corpus["license"]),
                  "design_paragraphs": len(corpus["design"]), "multilingual_lines": len(corpus["multilingual"]),
                  "multilingual_languages": len({lang for lang, _ in corpus["multilingual"]}),
                  "chatml_wrappers": len(corpus["chatml"]["wrappers"])},
        "sources": {
            "csharp": "C# snippets written for this repo (G2)",
            "license": "Apache License 2.0, the LICENSE file of Qwen3-0.6B at the pinned revision, byte for byte",
            "design": "docs/LRM-design.md as of commit f480fef, byte for byte; prose blocks only (no headings, no tables)",
            "multilingual": "lines written for this repo (G2), one language code and one line each",
            "chatml": "Qwen3 ChatML wrappers written for this repo (G2)",
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# Seeded drawing
# ---------------------------------------------------------------------------------------------------------------------


def _rng(name: str) -> c.DetRng:
    return c.DetRng(SEED, c.stream_of(f"{SET}-{VERSION}/{name}"))


def _draw_csharp(rng: c.DetRng, corpus: dict) -> tuple[str, str]:
    snippets = corpus["csharp"]
    mode = rng.next_below(4)
    if mode <= 1:
        i = rng.next_below(len(snippets))
        return snippets[i][1], f"csharp:{snippets[i][0]}"
    if mode == 2:
        i = rng.next_below(len(snippets))
        n = 2 + rng.next_below(3)
        picked = [snippets[(i + j) % len(snippets)] for j in range(n)]
        return "\n\n".join(t for _, t in picked), "csharp:" + "+".join(nm for nm, _ in picked)
    i = rng.next_below(len(snippets))
    lines = snippets[i][1].split("\n")
    if len(lines) < 6:
        return snippets[i][1], f"csharp:{snippets[i][0]}"
    start = rng.next_below(len(lines) - 2)
    count = 3 + rng.next_below(len(lines) - start - 2)
    return "\n".join(lines[start:start + count]), f"csharp:{snippets[i][0]}[{start}:{start + count}]"


def _draw_paragraphs(rng: c.DetRng, paras: list[str], tag: str) -> tuple[str, str]:
    start = rng.next_below(len(paras))
    count = 1 + rng.next_below(7)
    end = min(len(paras), start + count)
    return "\n\n".join(paras[start:end]), f"{tag}:{start}..{end - 1}"


def _draw_multilingual(rng: c.DetRng, corpus: dict) -> tuple[str, str]:
    lines = corpus["multilingual"]
    want = 1 + rng.next_below(6)
    picked: list[int] = []
    while len(picked) < want:
        i = rng.next_below(len(lines))
        if i not in picked:
            picked.append(i)
    return "\n".join(lines[i][1] for i in picked), "multilingual:" + ",".join(f"{lines[i][0]}{i}" for i in picked)


def _draw_content(kind: str, rng: c.DetRng, corpus: dict) -> tuple[str, str]:
    if kind == "csharp":
        return _draw_csharp(rng, corpus)
    if kind == "license":
        return _draw_paragraphs(rng, corpus["license"], "license")
    if kind == "design":
        return _draw_paragraphs(rng, corpus["design"], "design")
    if kind == "multilingual":
        return _draw_multilingual(rng, corpus)
    raise ValueError(kind)


def chatml_text(wrapper: dict, fmt: dict, content: str | None) -> str:
    """A ChatML prompt that ends with an open assistant turn (and Qwen3's empty thinking block when 'think')."""
    out = []
    for turn in wrapper["turns"]:
        text = turn["text"]
        if "{content}" in text:
            if content is None:
                raise c.GoldenError(f"chatml wrapper {wrapper['id']}: a slot without content")
            text = text.replace("{content}", content)
        out.append(fmt["turn"].replace("{role}", turn["role"]).replace("{text}", text))
    out.append(fmt["assistant_open"])
    if wrapper["think"]:
        out.append(fmt["think_empty"])
    return "".join(out)


def _draw_chatml(rng: c.DetRng, corpus: dict) -> tuple[str, str]:
    chat = corpus["chatml"]
    w = chat["wrappers"][rng.next_below(len(chat["wrappers"]))]
    if w["content"] == "none":
        return chatml_text(w, chat["format"], None), f"chatml:{w['id']}"
    content, src = _draw_content(w["content"], rng, corpus)
    return chatml_text(w, chat["format"], content), f"chatml:{w['id']}({src})"


def _draw(category: str, rng: c.DetRng, corpus: dict) -> tuple[str, str]:
    if category == "chatml":
        return _draw_chatml(rng, corpus)
    return _draw_content(category, rng, corpus)


def encode(tok, text: str) -> list[int]:
    """TokenizerApi Encode: tokenizers' ids with no special tokens added (Qwen3 adds no BOS either way)."""
    ids = tok.encode(text, add_special_tokens=False).ids
    if tok.encode(text, add_special_tokens=True).ids != ids:
        raise c.GoldenError("the tokenizer adds special tokens; TokenizerApi expects none")
    return ids


def build_pool(tok, corpus: dict, quotas: dict | None = None) -> list[dict]:
    """The seeded candidate pool, in pool order. Candidate k of a category draws from
    DetRng(SEED, streamOf("qwen3-0.6b-g1/pool/<category>/<k>")) until a text of MIN_TOKENS..MAX_TOKENS tokens that is
    not already in the pool comes up."""
    quotas = quotas or POOL_QUOTAS
    pool, seen = [], set()
    for cat in CATEGORIES:
        for k in range(quotas[cat]):
            rng = _rng(f"pool/{cat}/{k}")
            for attempt in range(MAX_ATTEMPTS):
                text, source = _draw(cat, rng, corpus)
                ids = encode(tok, text)
                if MIN_TOKENS <= len(ids) <= MAX_TOKENS and text not in seen:
                    break
            else:
                raise c.GoldenError(f"pool {cat}/{k}: no draw of {MIN_TOKENS}..{MAX_TOKENS} tokens in {MAX_ATTEMPTS} attempts")
            seen.add(text)
            pool.append({"name": f"{cat}-{k:03d}", "category": cat, "k": k, "attempts": attempt + 1, "source": source,
                         "text": text, "ids": ids, "length": len(ids), "text_sha256": c.sha256_bytes(text.encode("utf-8"))})
    return pool


def calibration_subset(pool: list[dict], size: int = CALIBRATION_SIZE) -> list[str]:
    """The fixed calibration subset, chosen by category and length only (no model output): per category in order, the
    first candidate shorter than SHORT_TOKENS and the first at or above it (the next unused one of the category when a
    bin is empty), then the longest candidates of the pool until `size` (ties: the earlier candidate)."""
    chosen: list[str] = []
    for cat in CATEGORIES:
        mine = [p for p in pool if p["category"] == cat]
        for want_short in (True, False):
            hit = next((p for p in mine if (p["length"] < SHORT_TOKENS) == want_short and p["name"] not in chosen), None)
            if hit is None:
                hit = next((p for p in mine if p["name"] not in chosen), None)
            if hit is not None and len(chosen) < size:
                chosen.append(hit["name"])
    for p in sorted(pool, key=lambda p: (-p["length"], pool.index(p))):
        if len(chosen) >= size:
            break
        if p["name"] not in chosen:
            chosen.append(p["name"])
    return chosen


def long_prompts(tok, corpus: dict, max_tokens: int = LONG_MAX_TOKENS, specs=LONG_SPECS) -> list[dict]:
    """The fixed long prompts: corpus items in file order, joined by a blank line, until the target length; an item
    that would pass max_tokens ends the prompt."""
    out = []
    for name, cat, target in specs:
        items = [t for _, t in corpus["csharp"]] if cat == "csharp" else corpus[cat]
        text, used = "", 0
        for item in items:
            nxt = item if not text else text + "\n\n" + item
            if len(encode(tok, nxt)) > max_tokens:
                break
            text, used = nxt, used + 1
            if len(encode(tok, text)) >= target:
                break
        ids = encode(tok, text)
        if not (target <= len(ids) <= max_tokens):
            raise c.GoldenError(f"long prompt {name}: {len(ids)} tokens, wanted {target}..{max_tokens}")
        out.append({"name": name, "category": cat, "source": f"{cat}: the first {used} items in file order",
                    "text": text, "ids": ids, "length": len(ids), "text_sha256": c.sha256_bytes(text.encode("utf-8"))})
    return out
