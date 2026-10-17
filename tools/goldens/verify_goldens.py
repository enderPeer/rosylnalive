"""Verifies every committed golden set against its manifest (GoldenSet contract), with the standard library only.

    python -I tools/goldens/verify_goldens.py            # any Python 3.8+; no venv needed
    pwsh tools/goldens/run.ps1 verify_goldens.py

For each goldens/<set>/g<N>/manifest.json it checks: the schema; that every listed repo file exists with the listed
size and SHA-256; that nothing unlisted sits in the set folder; that committed files are at most 2 MiB; that a
model folder's SHA256SUMS matches its files and REVISION is a local: revision; and the size budgets below. Files with
location "data" live under LRM_DATA/golden and are checked only when --data is given.
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
