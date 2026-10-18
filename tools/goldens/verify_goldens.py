"""Verifies every committed golden set against its manifest (GoldenSet contract), with the standard library only.

    python -I tools/goldens/verify_goldens.py            # any Python 3.8+; no venv needed
    pwsh tools/goldens/run.ps1 verify_goldens.py

For each goldens/<set>/g<N>/manifest.json it checks: the schema; that every listed repo file exists with the listed
size and SHA-256; that nothing unlisted sits in the set folder; that committed files are at most 2 MiB; that a
model folder's SHA256SUMS matches its files and REVISION is a local: revision; that every safetensors file follows the
GoldenSet rules (sorted names, padded header, contiguous and naturally aligned data); that a tokenizer corpus follows its
line rules (check_cases: fields per kind, unique ids, no lone surrogate in any JSON string); and the size budgets below.
Files with location "data" live under LRM_DATA/golden and are checked only when --data is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = "lrm.golden/1"
MAX_FILE = 2 * 1024 * 1024
MIB = 1024 * 1024
# Committed-size budgets in bytes: per set, and for O1's sets together (tiny, tokenizer and goldens/README.md).
SET_BUDGETS = {"tiny": 5 * MIB, "tokenizer": 2 * MIB, "qwen3-0.6b": 10 * MIB}
O1_BUDGET = 5 * MIB
O1_SETS = ("tiny", "tokenizer")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_ST_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "U32": 4, "U8": 1}


def check_safetensors(path: str, rel: str) -> list[str]:
    """GoldenSet safetensors rules: header length a multiple of 8, __metadata__ first, names sorted, data contiguous
    in name order with no holes or trailing bytes, and every tensor naturally aligned."""
    import struct

    out = []
    with open(path, "rb") as f:
        raw = f.read()
    (n,) = struct.unpack("<Q", raw[:8])
    if n % 8 or 8 + n > len(raw):
        return [f"{rel}: header length {n} is not a multiple of 8 or overruns the file"]
    text = raw[8:8 + n].decode("ascii")
    pairs = json.loads(text, object_pairs_hook=list)
    names = [k for k, _ in pairs if k != "__metadata__"]
    if any(k == "__metadata__" for k, _ in pairs[1:]):
        out.append(f"{rel}: __metadata__ is not first")
    if names != sorted(names):
        out.append(f"{rel}: tensor names are not sorted")
    pos, data = 0, len(raw) - 8 - n
    entries = dict(pairs)
    for name in names:
        e = dict(entries[name])
        b, end = e["data_offsets"]
        size = _ST_SIZES.get(e["dtype"])
        count = 1
        for d in e["shape"]:
            count *= d
        if size is None or b != pos or end - b != count * size:
            out.append(f"{rel}: entry {name} is not contiguous or has the wrong size")
        elif b % size:
            out.append(f"{rel}: entry {name} at offset {b} is not aligned to {size}")
        pos = end
    if pos != data:
        out.append(f"{rel}: {data - pos} trailing or missing data bytes")
    return out


_CASE_FIELDS = {
    "encode": {"text", "ids"},
    "encode_bytes": {"bytes", "text", "ids"},
    "decode": {"ids", "bytes", "text"},
    "encode_reject": {"utf16", "error"},
}


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield k
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def check_cases(path: str, rel: str) -> list[str]:
    """GoldenSet tokenizer-corpus rules: ASCII, LF lines with a final LF, one compact JSON object per line with sorted
    keys, a unique id, the fields of its kind and no other, and no string holding a lone surrogate (System.Text.Json
    cannot read one), so encode_reject carries its text only as utf16."""
    out, seen = [], set()
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return [f"{rel}: not ASCII"]
    if "\r" in text or not text.endswith("\n"):
        return [f"{rel}: lines must be LF with a final LF"]
    for n, line in enumerate(text[:-1].split("\n"), 1):
        case = json.loads(line)
        if json.dumps(case, sort_keys=True, ensure_ascii=True, separators=(",", ":")) != line:
            out.append(f"{rel}:{n}: not compact JSON with sorted keys")
        want = _CASE_FIELDS.get(case.get("kind"))
        if want is None or set(case) != want | {"kind", "id", "cat"}:
            out.append(f"{rel}:{n}: kind {case.get('kind')!r} with fields {sorted(case)}")
        if case.get("id") in seen:
            out.append(f"{rel}:{n}: duplicate id {case.get('id')!r}")
        seen.add(case.get("id"))
        for s in _strings(case):
            try:
                s.encode("utf-8")
            except UnicodeEncodeError:
                out.append(f"{rel}:{n}: a string holds a lone surrogate")
                break
        if case.get("kind") == "encode_reject":
            h = case.get("utf16", "")
            if len(h) % 4 or h != h.lower() or any(ch not in "0123456789abcdef" for ch in h):
                out.append(f"{rel}:{n}: utf16 is not lowercase hex in 4-digit units")
    return out


def verify_set(set_dir: str, data_root: str | None) -> tuple[list[str], int]:
    problems, committed = [], 0
    man_path = os.path.join(set_dir, "manifest.json")
    with open(man_path, "rb") as f:
        raw = f.read()
    man = json.loads(raw)
    rel_dir = os.path.relpath(set_dir, REPO).replace(os.sep, "/")
    committed += len(raw)
    if man.get("schema") != SCHEMA:
        problems.append(f"{rel_dir}: schema is {man.get('schema')!r}, not {SCHEMA}")
    if f"goldens/{man.get('set')}/{man.get('version')}" != rel_dir:
        problems.append(f"{rel_dir}: set/version {man.get('set')}/{man.get('version')} do not match the folder")
    listed = set()
    for e in man.get("files", []):
        path = e["path"]
        if e["location"] == "repo":
            full = os.path.join(REPO, *path.split("/"))
            listed.add(os.path.normcase(os.path.abspath(full)))
            if not os.path.isfile(full):
                problems.append(f"missing: {path}")
                continue
            size = os.path.getsize(full)
            committed += size
            if size != e["bytes"]:
                problems.append(f"{path}: {size} bytes, manifest says {e['bytes']}")
            if size > MAX_FILE:
                problems.append(f"{path}: {size} bytes is over the 2 MiB limit for committed golden files")
            if sha256(full) != e["sha256"]:
                problems.append(f"{path}: SHA-256 differs from the manifest")
            if path.endswith(".safetensors"):
                problems += check_safetensors(full, path)
            if man.get("kind") == "tokenizer-corpus" and path.endswith(".jsonl"):
                problems += check_cases(full, path)
        elif e["location"] == "data":
            if data_root:
                full = os.path.join(data_root, "golden", *path.split("/"))
                if not os.path.isfile(full) or sha256(full) != e["sha256"]:
                    problems.append(f"LRM_DATA/golden/{path}: missing or SHA-256 differs")
        else:
            problems.append(f"{path}: unknown location {e['location']!r}")
    for dirpath, _dirs, names in os.walk(set_dir):
        for n in names:
            full = os.path.normcase(os.path.abspath(os.path.join(dirpath, n)))
            if n != "manifest.json" or os.path.dirname(full) != os.path.normcase(os.path.abspath(set_dir)):
                if full not in listed:
                    problems.append(f"unlisted file: {os.path.relpath(full, REPO)}")
    model_dir = os.path.join(set_dir, "model")
    if os.path.isdir(model_dir):
        sums = {}
        with open(os.path.join(model_dir, "SHA256SUMS"), encoding="ascii") as f:
            for line in f:
                digest, name = line.rstrip("\n").split(" *", 1)
                sums[name] = digest
        for name in sorted(os.listdir(model_dir)):
            if name != "SHA256SUMS" and sums.get(name) != sha256(os.path.join(model_dir, name)):
                problems.append(f"{rel_dir}/model/{name}: not matched by SHA256SUMS")
        with open(os.path.join(model_dir, "REVISION"), encoding="ascii") as f:
            rev = f.read().strip()
        if not rev.startswith("local:") or rev != man.get("model", {}).get("revision"):
            problems.append(f"{rel_dir}/model/REVISION is {rev!r}")
    return problems, committed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="also verify location=data files under this LRM_DATA root")
    args = ap.parse_args()
    root = os.path.join(REPO, "goldens")
    problems, sizes = [], {}
    for set_name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        sdir = os.path.join(root, set_name)
        if not os.path.isdir(sdir):
            continue
        for ver in sorted(os.listdir(sdir)):
            vdir = os.path.join(sdir, ver)
            if os.path.isfile(os.path.join(vdir, "manifest.json")):
                p, n = verify_set(vdir, args.data)
                problems += p
                sizes[set_name] = sizes.get(set_name, 0) + n
                print(f"goldens/{set_name}/{ver}: {n} bytes committed, {'ok' if not p else str(len(p)) + ' problem(s)'}")
    for set_name, n in sorted(sizes.items()):
        if n > SET_BUDGETS.get(set_name, MAX_FILE):
            problems.append(f"goldens/{set_name}: {n} bytes is over its budget of {SET_BUDGETS.get(set_name, MAX_FILE)}")
    readme = os.path.join(root, "README.md")
    o1 = sum(sizes.get(s, 0) for s in O1_SETS) + (os.path.getsize(readme) if os.path.isfile(readme) else 0)
    print(f"O1 sets and goldens/README.md: {o1} bytes of {O1_BUDGET} ({o1 / MIB:.2f} MiB)")
    if o1 > O1_BUDGET:
        problems.append(f"O1 goldens are {o1} bytes, over {O1_BUDGET}")
    for p in problems:
        print("PROBLEM:", p)
    print("verify_goldens:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
