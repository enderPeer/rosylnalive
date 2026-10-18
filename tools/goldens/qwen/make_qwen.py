"""Builds goldens/qwen3-0.6b/g1: the full-model goldens and the ToleranceSpec numbers (workstream G2).

    pwsh tools/goldens/run.ps1 qwen/make_qwen.py             # regenerate; writes goldens/qwen3-0.6b/g1 and LRM_DATA/golden
    pwsh tools/goldens/run.ps1 qwen/make_qwen.py --check     # regenerate in memory, compare with the committed bytes
                                                             # and the data files' SHA-256 in the committed manifest
    pwsh tools/goldens/run.ps1 qwen/make_qwen.py --force     # replace a data folder whose files differ (never for g1)
    pwsh tools/goldens/run.ps1 qwen/make_qwen.py --quick --out <dir> --data-out <dir>   # a small smoke run

The stages, in order (README.md in this folder says what each file holds):
1. model: the pinned revision, SHA256SUMS, the toolchain pins, lm_head.weight equal to embed_tokens;
2. self-tests: O1's oracle self-tests (tools/goldens/selftest.py) and a hand-written numpy fp64 forward of the real
   model against torch64p on a short prompt;
3. rope: torch's inv_freq bits for head_dim 128 and T_spec for positions 0..8191, checked against the
   ReferenceOracles vectors and recomputed with mpmath in every entry;
4. pool: torch32 64-step greedy (EOS ignored) with top-2 margins on every seeded candidate;
5. calibration: e_ref = relL2(torch32, torch64p) per tap on the fixed calibration subset (and max |delta logit| and KL
   for logits), plus a bf16 run for calibration only; tolerances.json follows from it;
6. selection: the gate prompts (margin at least max(0.05, 4 * tau_logit) over 64 steps, at most 900 tokens, English,
   C# and ChatML among them; the prefix fallback otherwise), each verified in torch64p;
7. gate: per gate prompt and oracle, the prefill logits (all taps for two short prompts), 64 greedy steps through
   the manual loop, generate() (common.greedy_generate) and a no-cache recompute, and the teacher-forced summaries;
8. long: two long prompts (about 1,000 and at most 4,000 tokens), final_norm.out, logits at fixed rows and per-row
   summaries.
Every stage that produces files runs twice at 8 threads and compares SHA-256 digests (1 thread if they differ).

The data folder is shared by every worktree on the machine, and golden content never changes in place (GoldenSet). So a
run whose data folder already holds files writes no data: it hashes what it produces, compares that with the files in
the folder, and writes the committed files only if every data file is byte-identical; otherwise it stops with the
differences. --force replaces the folder instead (for a new version or a smoke folder, never for a published one).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common as c  # noqa: E402  (runs the process guards first)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import qwen_corpus as qc  # noqa: E402
import qwen_oracles as qo  # noqa: E402
import qwen_tolerances as qt  # noqa: E402
import selftest  # noqa: E402
import tiny_model as tm  # noqa: E402
from verify_goldens import SET_BUDGETS  # noqa: E402

transformers.utils.logging.disable_progress_bar()

SET_NAME = f"{qc.SET}/{qc.VERSION}"
OUT_REL = f"goldens/{qc.SET}/{qc.VERSION}"
DATA_REL = f"{qc.SET}/{qc.VERSION}"
STEPS = qt.GREEDY_STEPS
MAX_POS = 8192
TAPS_PROMPTS = 2
PER_CATEGORY = 2
GATE_MIN = 8
SELFTEST_TOKENS = 16
# A gate prompt's 64 greedy ids hold at least this many distinct tokens, so no gate prompt is a degenerate repetition
# (the first g1 run picked csharp-000, whose continuation is 64 newlines).
MIN_DISTINCT = 8
QWEN_F64_BOUND = 1e-13

# ReferenceOracles vectors for Qwen3-0.6B (docs/contracts/reference-oracles.md, T_spec vectors).
INV_FREQ_SHA256 = "138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e"
T_SPEC_SHA256 = "da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf"
TORCH32_TABLE_MISMATCHES = {"cos": 27369, "sin": 19437}

TOOL_FILES = ["common.py", "selftest.py", "tiny_model.py", "verify_goldens.py", "qwen/make_qwen.py", "qwen/qwen_corpus.py",
              "qwen/qwen_oracles.py", "qwen/qwen_tolerances.py"] + [f"qwen/corpus/{n}" for n in qc.CORPUS_FILES]


def say(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Params:
    """The canonical configuration, or a small one for smoke runs (--quick never writes the canonical folders)."""

    def __init__(self, quick: bool):
        self.quick = quick
        self.quotas = {k: 2 for k in qc.CATEGORIES} if quick else dict(qc.POOL_QUOTAS)
        self.calibration_size = 3 if quick else qc.CALIBRATION_SIZE
        self.long_specs = (("long-design", "design", 150), ("long-csharp", "csharp", 300)) if quick else qc.LONG_SPECS
        self.long_max = 600 if quick else qc.LONG_MAX_TOKENS
        self.gate_min = 2 if quick else GATE_MIN
        self.per_category = 1 if quick else PER_CATEGORY

    def record(self) -> dict:
        return {"quick": self.quick, "pool_quotas": self.quotas, "calibration_size": self.calibration_size,
                "long_specs": [list(s) for s in self.long_specs], "long_max_tokens": self.long_max,
                "gate_min": self.gate_min, "per_category": self.per_category, "steps": STEPS}


# ---------------------------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------------------------


def st_file(arrays: dict, meta: dict) -> bytes:
    """A golden tensor file; every tensor starts at a multiple of its element size (GoldenSet alignment rule)."""
    tensors = {n: (c.st_bf16(a) if a.dtype == np.uint16 else c.st(a)) for n, a in arrays.items()}
    offset = 0
    for n in sorted(tensors):
        if offset % c._ST_SIZES[tensors[n].dtype]:
            raise c.GoldenError(f"tensor {n} would start at a misaligned offset {offset}")
        offset += len(tensors[n].data)
    return c.safetensors_bytes(tensors, meta)


def meta(**kw) -> dict:
    return {"lrm.golden": SET_NAME, **{k: str(v) for k, v in kw.items()}}


def digest_of_shas(shas: dict) -> str:
    """common.digest_of_files over precomputed SHA-256 digests: sha256 of name, 0x00 and the file's digest, by name."""
    h = hashlib.sha256()
    for name in sorted(shas):
        h.update(name.encode() + b"\x00" + bytes.fromhex(shas[name]))
    return h.hexdigest()


class Sink:
    """Where a stage's files go. Run A only hashes them; run B hashes them, keeps repo files in memory, and writes data
    files to the staging folder (or, with --check or an existing data folder, only hashes them)."""

    def __init__(self, keep: bool, data_dir: str | None):
        self.keep, self.data_dir = keep, data_dir
        self.shas: dict = {}
        self.entries: dict = {}
        self.repo: dict = {}

    def __call__(self, name: str, data: bytes, role: str, location: str) -> None:
        if name in self.shas:
            raise c.GoldenError(f"{name} produced twice")
        self.shas[name] = c.sha256_bytes(data)
        if not self.keep:
            return
        rel = f"{OUT_REL}/{name}" if location == "repo" else f"{DATA_REL}/{name}"
        self.entries[name] = c.file_entry(rel, data, role, location)
        if location == "repo":
            self.repo[name] = data
        elif self.data_dir is not None:
            c.write_bytes(os.path.join(self.data_dir, name), data)


def double_run_stream(produce, data_dir: str | None, stage: str):
    """common.double_run for stages whose files are too large to hold twice: run A hashes every file and runs the
    cross-checks (produce(sink, first=True)), run B recomputes and hashes every file and keeps it. The digests must
    agree at 8 threads, else at 1 thread. Returns run A's results, run B's sink and the double-run record."""
    attempts = []
    for threads in (8, 1):
        c.setup_torch(threads)
        a = Sink(False, None)
        results = produce(a, True)
        c.setup_torch(threads)
        b = Sink(True, data_dir)
        produce(b, False)
        da, db = digest_of_shas(a.shas), digest_of_shas(b.shas)
        diff = sorted(n for n in a.shas if a.shas[n] != b.shas.get(n))
        attempts.append({"threads": threads, "digest_a": da, "digest_b": db, "identical": da == db})
        say(f"{stage}: double run at {threads} thread(s): {'identical' if da == db else 'DIFFERENT ' + str(diff)} ({da[:16]})")
        if da == db:
            return results, b, {"threads": threads, "identical": True, "digest": db, "attempts": attempts}
    raise c.GoldenError(f"{stage}: double runs differ even at 1 thread: {attempts}")


# ---------------------------------------------------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------------------------------------------------


def qwen_weights_f64(model_dir: str) -> dict:
    """Every Qwen3-0.6B weight except lm_head (tied) as float64, read from model.safetensors through a memory map and
    widened from the bf16 bits: independent of transformers' loading path."""
    path = os.path.join(model_dir, "model.safetensors")
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    out = {}
    for name, e in header.items():
        if name in ("__metadata__", "lm_head.weight"):
            continue
        if e["dtype"] != "BF16":
            raise c.GoldenError(f"{name}: dtype {e['dtype']}")
        b, end = e["data_offsets"]
        bits = np.memmap(path, dtype="<u2", mode="r", offset=8 + n + b, shape=tuple(e["shape"]))
        out[name] = c.bf16_widen(np.asarray(bits)).astype(np.float64)
    return out


def selftest_qwen_fp64(model_dir: str, m64, ids: list[int], tables) -> dict:
    """Every torch64p tap of a short prompt on the real model against selftest.np_forward (plain numpy fp64)."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    w = qwen_weights_f64(model_dir)
    with c.Torch64pPatches():
        taps, _ = qo.forward_taps_t(m64, ids)
    ref = selftest.np_forward(ids, w, tables[0], tables[1], cfg)
    per = {n: c.rel_l2(taps[n].numpy(), ref[n]) for n in c.activation_names(cfg["num_hidden_layers"])}
    worst = max(per, key=lambda n: per[n])
    del w
    return {"passed": per[worst] <= QWEN_F64_BOUND, "metric": "relL2", "measured": per[worst], "bound": QWEN_F64_BOUND,
            "worst_tap": worst, "tokens": len(ids),
            "what": "torch64p Qwen3-0.6B taps vs a hand-written numpy fp64 forward over weights read from model.safetensors"}


def run_selftests(model_dir: str, m64, ids: list[int], tables) -> dict:
    with tempfile.TemporaryDirectory(prefix="lrm-g2-tiny-") as d:
        for name, data in tm.model_files(eos_token_id=1).items():
            c.write_bytes(os.path.join(d, name), data)
        m32t = c.load_model(d, "torch32")
        inv = c.torch_inv_freq_bits(m32t.config)
        del m32t
        cos_t, sin_t, _ = c.t_spec(inv, tm.MAX_POS)
        res = selftest.run_all(d, inv, (cos_t, sin_t), model_dir)
    res["qwen_fp64_forward"] = selftest_qwen_fp64(model_dir, m64, ids, tables)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------------------------------------------------


def rope_stage(config) -> tuple[np.ndarray, tuple, dict]:
    inv = c.torch_inv_freq_bits(config)
    inv_sha = c.sha256_bytes(inv.astype("<u4").tobytes())
    if inv_sha != INV_FREQ_SHA256:
        raise c.GoldenError(f"inv_freq bits SHA-256 {inv_sha} differs from the ReferenceOracles vector")
    cos, sin, info = c.t_spec(inv, MAX_POS)
    digest = c.sha256_bytes(np.ascontiguousarray(cos).tobytes() + np.ascontiguousarray(sin).tobytes())
    if digest != T_SPEC_SHA256:
        raise c.GoldenError(f"T_spec 0..{MAX_POS - 1} digest {digest} differs from the ReferenceOracles vector")
    cross = c.t_spec_crosscheck(inv, MAX_POS, cos, sin, full_mpmath=True)
    for fn in ("cos", "sin"):
        r = cross[fn]
        if r["full_mpmath_mismatches"] or r["torch_f64_mismatches"] or r["numpy_f64_mismatches"] or r["crt_f64_mismatches"]:
            raise c.GoldenError(f"T_spec {fn}: an f64 source or mpmath disagrees: {r}")
        if r["torch32_f32_mismatches"] != TORCH32_TABLE_MISMATCHES[fn]:
            raise c.GoldenError(f"T_spec {fn}: torch32 differs in {r['torch32_f32_mismatches']} entries, the contract says "
                                f"{TORCH32_TABLE_MISMATCHES[fn]}")
    rec = {"inv_freq_sha256": inv_sha, "t_spec_sha256": digest, "resolved_with_mpmath": info["stats"], "crosscheck": cross}
    return inv, (cos, sin), rec


def rope_file(inv: np.ndarray, tables) -> bytes:
    return st_file({"inv_freq.bits": inv.astype(np.uint32), "t_spec.cos": tables[0], "t_spec.sin": tables[1]},
                   meta(theta=1000000, head_dim=128, max_pos=MAX_POS))


# ---------------------------------------------------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------------------------------------------------


def build_pool_file(m32, pool: list[dict]) -> dict:
    """torch32 manual greedy (64 steps, EOS ignored) for every candidate: ids, margins and lengths."""
    c.assert_unpatched()
    ids, margins = [], []
    t0 = time.time()
    for i, p in enumerate(pool):
        det = c.greedy_manual_detail(m32, p["ids"], STEPS)
        ids.append(det["ids"])
        margins.append(det["margins"])
        if i % 20 == 19:
            say(f"pool: {i + 1}/{len(pool)} candidates ({time.time() - t0:.0f} s)")
    arrays = {"greedy_ids": np.array(ids, dtype=np.int64), "margins": np.array(margins, dtype=np.float64),
              "lengths": np.array([p["length"] for p in pool], dtype=np.int64)}
    return {"pool.safetensors": st_file(arrays, meta(oracle="torch32", steps=STEPS, eos="ignored",
                                                      order="pool.json candidates"))}


def prefix_len(margins, theta: float) -> int:
    """The number of leading greedy steps whose margin is at least theta (64 when every step is)."""
    return next((s for s, m in enumerate(margins) if m < theta), len(margins))


def distinct(p: dict) -> int:
    """How many distinct tokens the candidate's 64 torch32 greedy ids hold."""
    return len(set(p["greedy32"]))


def eligible(p: dict, theta: float) -> bool:
    """A full gate prompt: at most 900 tokens, a torch32 margin of at least theta at all 64 steps, and at least
    MIN_DISTINCT distinct tokens among the 64 greedy ids."""
    return p["length"] <= qc.MAX_TOKENS and prefix_len(p["margins"], theta) == STEPS and distinct(p) >= MIN_DISTINCT


# ---------------------------------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------------------------------


def bf16_taps(m16, ids: list[int]) -> dict:
    """Taps of the bf16 calibration model (torch tensors in bfloat16; qo.as_numpy widens them exactly)."""
    c.assert_unpatched()
    taps, _ = qo.forward_taps_t(m16, ids)
    return taps


def build_calibration(m32, m64, m16, prompts: list[dict]) -> dict:
    """e_ref = relL2(torch32, torch64p) per tap on each calibration prompt (torch64p is the reference), the logits'
    max |delta| and largest per-row KL(P_torch64p || P_torch32); the same for bf16 against torch64p."""
    out = []
    for p in prompts:
        t0 = time.time()
        with c.Torch64pPatches():
            t64, chk64 = qo.forward_taps_t(m64, p["ids"])
        c.assert_unpatched()
        t32, chk32 = qo.forward_taps_t(m32, p["ids"])
        rec = {"name": p["name"], "category": p["category"], "length": p["length"], "ids": p["ids"],
               "cache_check": {"torch32": chk32, "torch64p": chk64},
               "e_ref": {"relL2": {n: c.rel_l2(t32[n].numpy(), t64[n].numpy()) for n in t64},
                         "logits": qo.logits_metrics(t32["logits"], t64["logits"])}}
        del t32
        t16 = bf16_taps(m16, p["ids"])
        rec["bf16"] = {"relL2": {n: c.rel_l2(qo.as_numpy(t16[n]), t64[n].numpy()) for n in t64},
                       "logits": qo.logits_metrics(t16["logits"], t64["logits"])}
        del t16, t64
        out.append(rec)
        say(f"calibration: {p['name']} ({p['length']} tokens) in {time.time() - t0:.0f} s; logits e_ref "
            f"{rec['e_ref']['logits']['relL2']:.3e}")
    doc = {"schema": "lrm.qwen-calibration/1", "set": SET_NAME, "pair": "torch32-vs-torch64p",
           "about": "e_ref per prompt and tap: relL2(torch32, torch64p) with torch64p as the reference; for logits also "
                    "max |delta logit| and the largest per-row KL(P_torch64p || P_torch32). bf16: the same distances of a "
                    "bf16 model (calibration only, never an oracle) to torch64p.",
           "prompts": out}
    return {"calibration.json": c.json_bytes(doc)}


# ---------------------------------------------------------------------------------------------------------------------
# Gate selection
# ---------------------------------------------------------------------------------------------------------------------


def select_gate(cands: list[dict], theta: float, verify, params: Params) -> tuple[list[dict], dict]:
    """Gate prompts: at most 900 tokens and a torch32 margin of at least theta at all 64 steps. Per category in order,
    the first such candidate and the longest one (then the next by k when torch64p does not confirm one), at most
    params.per_category each. If fewer than params.gate_min, or English, C# or ChatML is missing, the prefix fallback
    fills in: candidates whose first low-margin step comes at step 16 or later, by the longest prefix."""
    full = [p for p in cands if eligible(p, theta)]
    chosen, tried = [], []

    def take(p, steps):
        tried.append(p["name"])
        if verify(p, steps):
            chosen.append({**p, "gated_steps": steps, "fallback": steps < STEPS})
            return True
        return False

    for cat in qc.CATEGORIES:
        mine = [p for p in full if p["category"] == cat]
        if not mine:
            continue
        longest = max(mine, key=lambda p: (p["length"], -p["k"]))
        order = [mine[0]] + ([longest] if longest is not mine[0] else [])
        order += [p for p in mine if p not in order]
        picks = 0
        for p in order:
            if picks == params.per_category:
                break
            picks += take(p, STEPS)

    def satisfied():
        cats = {p["category"] for p in chosen}
        return len(chosen) >= params.gate_min and cats & set(qc.ENGLISH) and {"csharp", "chatml"} <= cats

    fallback_used = False
    if not satisfied():
        fallback_used = True
        rest = [p for p in cands if p["length"] <= qc.MAX_TOKENS and p["name"] not in tried and distinct(p) >= MIN_DISTINCT
                and qt.PREFIX_MIN_STEPS <= prefix_len(p["margins"], theta) < STEPS]
        rest.sort(key=lambda p: (-prefix_len(p["margins"], theta), p["k"], p["category"]))
        for need in (qc.ENGLISH, ("csharp",), ("chatml",)):
            if not any(p["category"] in need for p in chosen):
                for p in [q for q in rest if q["category"] in need and q["name"] not in tried]:
                    if take(p, prefix_len(p["margins"], theta)):
                        break
        for p in rest:
            if len(chosen) >= params.gate_min:
                break
            if p["name"] not in tried:
                take(p, prefix_len(p["margins"], theta))
    if not satisfied():
        raise c.GoldenError(f"gate selection: only {[p['name'] for p in chosen]} pass, even with the prefix fallback")
    position = {p["name"]: i for i, p in enumerate(cands)}
    chosen.sort(key=lambda p: position[p["name"]])
    rule = {
        "eligible": f"at most {qc.MAX_TOKENS} tokens, a torch32 top-2 margin of at least the margin rule at all {STEPS} steps, "
                    f"and at least {MIN_DISTINCT} distinct tokens among the {STEPS} greedy ids",
        "min_distinct_tokens": MIN_DISTINCT,
        "per_category": f"in category order {list(qc.CATEGORIES)}: the first eligible candidate (lowest k) and the longest "
                        f"(ties: lowest k), then the next by k when torch64p does not confirm one; at most {params.per_category} each",
        "torch64p": "the torch64p manual loop must give the same ids over the gated steps, with the same minimum margin rule",
        "requires": f"at least {params.gate_min} prompts, including English (license or design), csharp and chatml",
        "fallback": f"otherwise candidates whose first low-margin step is at step {qt.PREFIX_MIN_STEPS} or later fill in, "
                    "longest prefix first; their greedy gate is that prefix, and teacher-forced argmax covers the rest "
                    "wherever the reference margin is at least 2 * tau_logit",
        "fallback_used": fallback_used,
        "tried": tried,
    }
    return chosen, rule


def taps_prompts(gate: list[dict]) -> list[str]:
    """The two shortest gate prompts of different categories (at least 16 tokens when possible)."""
    order = sorted(gate, key=lambda p: (p["length"] < 16, p["length"], p["name"]))
    out, cats = [], set()
    for p in order:
        if p["category"] not in cats:
            out.append(p["name"])
            cats.add(p["category"])
        if len(out) == TAPS_PROMPTS:
            break
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Gate and long-prompt files
# ---------------------------------------------------------------------------------------------------------------------


def summaries_arrays(s: dict, prefix: str = "") -> dict:
    return {f"{prefix}argmax": s["argmax"], f"{prefix}margin": s["margin"], f"{prefix}lse": s["lse"],
            f"{prefix}top64.ids": s["top_ids"], f"{prefix}top64.values": s["top_values"]}


def f32_record(det: dict) -> dict:
    agrees = det["f32_argmax_agrees"]
    first = next((s for s, a in enumerate(agrees) if not a), None)
    return {"f32_argmax_agrees_every_step": all(agrees), "first_f32_disagreement": first,
            "min_margin_f32_ulps": min(det["margin_f32_ulps"])}


def produce_gate(m32, m64, gate: list[dict], taps_names: list[str], inv, tables, eos_ids: list[int], tau: float):
    """The gate stage: per prompt and oracle, the prefill (fwd_*), the manual greedy loop, and the teacher-forced
    summaries (tf_*); on the first run also generate(), the no-cache recompute and the oracle comparisons."""

    def produce(sink, first: bool):
        sink("rope.safetensors", rope_file(inv, tables), "rope", "data")
        results, greedy_arrays = {}, {}
        for p in gate:
            t0 = time.time()
            name, ids = p["name"], p["ids"]
            taps = name in taps_names
            res = {"taps": taps}
            per = {}
            for oracle, model in (("torch32", m32), ("torch64p", m64)):
                ctx = c.Torch64pPatches() if oracle == "torch64p" else _Unpatched()
                tag = "f32" if oracle == "torch32" else "f64p"
                with ctx:
                    if taps:
                        tp, chk = qo.forward_taps_t(model, ids)
                        arrays = {n: qo.as_numpy(t) for n, t in tp.items()}
                        res[f"cache_check_{oracle}"] = chk
                        logits = tp["logits"]
                    else:
                        logits = qo.forward_logits(model, ids)
                        arrays = {"logits": qo.as_numpy(logits)}
                    arrays["input_ids"] = np.array(ids, dtype=np.int64)
                    sink(f"fwd_{tag}.{name}.safetensors", st_file(arrays, meta(oracle=oracle, prompt=name)), f"fwd_{tag}", "data")
                    del arrays
                    det = c.greedy_manual_detail(model, ids, STEPS)
                    teacher = ids + (per["torch32"]["det"]["ids"] if oracle == "torch64p" else det["ids"])[: STEPS - 1]
                    tf = qo.row_summaries(qo.forward_logits(model, teacher))
                    sink(f"tf_{tag}.{name}.safetensors",
                         st_file({"input_ids": np.array(teacher, dtype=np.int64), **summaries_arrays(tf)},
                                 meta(oracle=oracle, prompt=name, teacher="prompt + torch32 greedy ids[0:63]")), f"tf_{tag}", "data")
                    per[oracle] = {"det": det, "tf": tf, "logits": logits if first else None,
                                   "taps": tp if (first and taps) else None}
                    if first:
                        gen, info = c.greedy_generate(model, ids, STEPS)
                        nocache = c.greedy_no_cache(model, ids, STEPS)
                        per[oracle].update({"generate": gen, "generate_info": info, "no_cache": nocache})
                        if oracle == "torch32" and any(t in eos_ids for t in det["ids"]):
                            leak = c.greedy_generate_without_neutral_config(model, ids, STEPS)
                            per[oracle]["control"] = {"new_tokens": len(leak),
                                                      "stops_at_eos": len(leak) < STEPS and leak[-1] in eos_ids}
                    if not taps:
                        del logits
                greedy_arrays[f"{name}.ids.{oracle}"] = np.array(det["ids"], dtype=np.int64)
                greedy_arrays[f"{name}.margin.{oracle}"] = np.array(det["margins"], dtype=np.float64)
            if first:
                results[name] = {**gate_checks(p, per, tau, eos_ids), **res}
                say(f"gate: {name} ({p['length']} tokens) in {time.time() - t0:.0f} s")
            per.clear()
        sink("greedy.safetensors", st_file(greedy_arrays, meta(steps=STEPS, eos="ignored", oracles="torch32,torch64p")),
             "greedy", "repo")
        return results

    return produce


class _Unpatched:
    def __enter__(self):
        c.assert_unpatched()
        return self

    def __exit__(self, *exc):
        return False


def gate_checks(p: dict, per: dict, tau: float, eos_ids: list[int]) -> dict:
    """The greedy agreement checks of one gate prompt, and its holdout e_ref."""
    steps = p["gated_steps"]
    d32, d64 = per["torch32"]["det"], per["torch64p"]["det"]
    if d32["ids"] != p["greedy32"] or d32["margins"] != p["margins"]:
        raise c.GoldenError(f"{p['name']}: the torch32 greedy loop differs from the pool stage's")
    out = {"gated_steps": steps, "fallback": p["fallback"], "distinct_tokens": len(set(d32["ids"]))}
    for oracle in ("torch32", "torch64p"):
        r, det = per[oracle], per[oracle]["det"]
        f32 = f32_record(det)
        manual = det["ids"]
        upto = STEPS if oracle == "torch32" or f32["first_f32_disagreement"] is None else f32["first_f32_disagreement"]
        gen_ok = r["generate"][:upto] == manual[:upto] and len(r["generate"]) == STEPS
        nc_ok = r["no_cache"] == manual
        if oracle == "torch32" and not (gen_ok and nc_ok) and steps == STEPS:
            raise c.GoldenError(f"{p['name']} torch32: manual loop, generate() and no-cache recompute disagree")
        if not (r["generate"][:steps] == manual[:steps] and r["no_cache"][:steps] == manual[:steps]):
            raise c.GoldenError(f"{p['name']} {oracle}: the greedy paths disagree within the gated {steps} steps")
        out[oracle] = {
            "manual_equals_generate": r["generate"] == manual, "generate_compared_steps": upto,
            "manual_equals_generate_where_f32_agrees": gen_ok, "manual_equals_no_cache": nc_ok,
            "min_margin": min(det["margins"]), "min_margin_gated": min(det["margins"][:steps]),
            "first_step_below_rule": next((s for s, m in enumerate(det["margins"]) if m < qt.margin_rule(tau)), None),
            "eos_first_step": next((s for s, t in enumerate(manual) if t in eos_ids), None),
            "generate_effective_config": r["generate_info"]["effective"], "generate_argmax_dtype": r["generate_info"]["argmax_dtype"],
            **f32,
        }
        if "control" in r:
            out[oracle]["control_without_neutral_model_config"] = r["control"]
    if d64["ids"][:steps] != d32["ids"][:steps]:
        raise c.GoldenError(f"{p['name']}: torch32 and torch64p choose different tokens within the gated steps")
    out["torch64p_equals_torch32"] = d64["ids"] == d32["ids"]
    t32, t64 = per["torch32"]["tf"], per["torch64p"]["tf"]
    strong = np.maximum(t32["margin"], t64["margin"]) >= 2 * tau
    out["teacher_forced"] = {"rows": int(len(t32["argmax"])), "rows_with_margin_ge_2tau": int((t32["margin"] >= 2 * tau).sum()),
                             "oracles_differ_rows": int((t32["argmax"] != t64["argmax"]).sum()),
                             "oracles_agree_where_margin_ge_2tau": bool(np.array_equal(t32["argmax"][strong], t64["argmax"][strong]))}
    if not out["teacher_forced"]["oracles_agree_where_margin_ge_2tau"]:
        raise c.GoldenError(f"{p['name']}: torch32 and torch64p teacher-forced argmax differ where the margin is >= 2 tau")
    out["holdout"] = {"logits": qo.logits_metrics(per["torch32"]["logits"], per["torch64p"]["logits"])}
    if per["torch32"]["taps"] is not None:
        a, b = per["torch32"]["taps"], per["torch64p"]["taps"]
        out["holdout"]["relL2"] = {n: c.rel_l2(a[n].numpy(), b[n].numpy()) for n in b}
    return out


def long_rows(t_len: int) -> list[int]:
    """The rows whose full logits a long prompt stores: the last row of every 64-token block, and the last 64 rows."""
    return sorted(set(range(63, t_len, 64)) | set(range(max(0, t_len - 64), t_len)))


def produce_long(m32, m64, longs: list[dict]):
    def produce(sink, first: bool):
        results = {}
        for p in longs:
            t0 = time.time()
            name, ids = p["name"], p["ids"]
            rows = long_rows(len(ids))
            logits_by = {}
            for oracle, model in (("torch32", m32), ("torch64p", m64)):
                tag = "f32" if oracle == "torch32" else "f64p"
                ctx = c.Torch64pPatches() if oracle == "torch64p" else _Unpatched()
                captured = {}
                with ctx:
                    handle = model.model.norm.register_forward_hook(lambda _m, _a, o: captured.__setitem__("f", o[0].detach().clone()))
                    try:
                        logits = qo.forward_logits(model, ids)
                    finally:
                        handle.remove()
                s = qo.row_summaries(logits)
                arrays = {"input_ids": np.array(ids, dtype=np.int64), "final_norm.out": qo.as_numpy(captured["f"]),
                          "logits.rows": np.array(rows, dtype=np.int64), "logits": qo.as_numpy(logits[rows]).copy(),
                          **summaries_arrays(s)}
                sink(f"long_{tag}.{name}.safetensors", st_file(arrays, meta(oracle=oracle, prompt=name)), f"long_{tag}", "data")
                del arrays
                if first:
                    logits_by[oracle] = (logits, captured["f"], s)
                else:
                    del logits
            if first:
                (l32, f32, s32), (l64, f64, s64) = logits_by["torch32"], logits_by["torch64p"]
                results[name] = {"rows_stored": len(rows),
                                 "holdout": {"logits": qo.logits_metrics(l32, l64),
                                             "logits_stored_rows": qo.logits_metrics(l32[rows], l64[rows]),
                                             "relL2": {"final_norm.out": c.rel_l2(f32.numpy(), f64.numpy())}},
                                 "argmax_oracles_differ_rows": int((s32["argmax"] != s64["argmax"]).sum()),
                                 "min_margin_torch32": float(s32["margin"].min())}
                logits_by.clear()
                say(f"long: {name} ({len(ids)} tokens) in {time.time() - t0:.0f} s")
        return results

    return produce


# ---------------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------------


def oracle_records(patches: list) -> dict:
    return {
        "torch32": {"role": "spec", "dtype": "float32", "attn_implementation": "eager", "load": "from_pretrained(dtype=float32)", "patches": []},
        "torch64p": {"role": "truth", "dtype": "float64", "attn_implementation": "eager",
                     "load": "from_pretrained(dtype=float32), then .double()", "patches": patches},
        "bf16": {"role": "calibration only, never an oracle", "dtype": "bfloat16", "attn_implementation": "eager",
                 "load": "from_pretrained(dtype=bfloat16)", "patches": []},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed files and data hashes instead of writing")
    ap.add_argument("--out", default=os.path.join(c.REPO_ROOT, *OUT_REL.split("/")), help="the committed set folder")
    ap.add_argument("--data-out", default=os.path.join(c.data_root(), "golden", *DATA_REL.split("/")), help="the data folder")
    ap.add_argument("--quick", action="store_true", help="a small smoke run (needs --out and --data-out elsewhere)")
    ap.add_argument("--no-telemetry", action="store_true", help="do not write a run log under LRM_DATA/runs")
    ap.add_argument("--force", action="store_true",
                    help="replace a data folder that already holds files, even if they differ (golden content then "
                         "changes in place; a published set gets a new version g<N+1> instead)")
    args = ap.parse_args()
    params = Params(args.quick)
    canonical_out = os.path.normcase(os.path.abspath(os.path.join(c.REPO_ROOT, *OUT_REL.split("/"))))
    canonical_data = os.path.normcase(os.path.abspath(os.path.join(c.data_root(), "golden", *DATA_REL.split("/"))))
    if args.quick and (os.path.normcase(os.path.abspath(args.out)) == canonical_out
                       or os.path.normcase(os.path.abspath(args.data_out)) == canonical_data):
        raise SystemExit("--quick never writes the canonical folders: pass --out and --data-out")
    # An existing data folder is only compared with, never rewritten, unless --force (module docstring).
    keep_data = not args.check and not args.force and data_present(args.data_out)
    if keep_data:
        say(f"{args.data_out} already holds files: this run writes no data; it compares every data file it produces with "
            "the files there and writes the committed files only if all are byte-identical (--force replaces them)")
    committed_manifest = None
    if args.check:
        with open(os.path.join(args.out, "manifest.json"), encoding="utf-8") as f:
            committed_manifest = json.load(f)

    t_start = time.time()
    log = c.RunLog("make_qwen", enabled=not args.no_telemetry,
                   title="make_qwen (tools/goldens/qwen): Qwen3-0.6B goldens and tolerances",
                   params={"set": SET_NAME, "quick": params.quick, "check": args.check, "force": args.force,
                           "keep_data": keep_data})
    c.setup_torch(8)
    guards = c.check_effective_guards()
    model_dir = os.path.join(c.data_root(), "models", "Qwen3-0.6B")
    timings = {}

    def phase(name):
        return log.span(f"goldens.{name}", set=qc.SET)

    with phase("check_model"):
        t0 = time.time()
        model_rec = check_model_and_tokenizer(model_dir)
        timings["check_model"] = time.time() - t0
    tok = model_rec.pop("_tokenizer")
    corpus = qc.load_corpus()
    pool = qc.build_pool(tok, corpus, params.quotas)
    longs = qc.long_prompts(tok, corpus, params.long_max, params.long_specs)
    calib_names = qc.calibration_subset(pool, params.calibration_size)
    say(f"pool: {len(pool)} candidates; calibration {calib_names}; long {[(p['name'], p['length']) for p in longs]}")
    with open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8") as f:
        gen_cfg = json.load(f)
    eos_ids = list(gen_cfg["eos_token_id"])

    m32 = c.load_model(model_dir, "torch32")
    with phase("rope"):
        t0 = time.time()
        inv, tables, rope_rec = rope_stage(m32.config)
        timings["rope"] = time.time() - t0
    m64 = c.load_model(model_dir, "torch64p", tables)

    with phase("selftests"):
        t0 = time.time()
        selftests = run_selftests(model_dir, m64, pool[0]["ids"][:SELFTEST_TOKENS] if len(pool[0]["ids"]) >= SELFTEST_TOKENS
                                  else pool[0]["ids"], tables)
        timings["selftests"] = time.time() - t0
    for k, v in sorted(selftests.items()):
        if "bound" in v:
            log.compare(v["passed"], v["metric"], v["measured"], v["bound"], check=k, pair="oracle-selftest")
        else:
            log.check("goldens.selftest", v["passed"], check=k)
    failed = [k for k, v in selftests.items() if not v["passed"]]
    if failed:
        raise c.GoldenError(f"oracle self-tests failed: {failed}")
    say("self-tests passed:", ", ".join(sorted(selftests)))
    ties = c.tie_probe()
    if not ties["lowest_index_on_ties"]:
        raise c.GoldenError("torch.argmax does not return the lowest index on ties")
    log.check("goldens.tie_probe", True, set=qc.SET, cases=ties["cases"])

    double = {}
    with phase("pool"):
        t0 = time.time()
        pool_files, double["pool"] = c.double_run(lambda _t: build_pool_file(m32, pool), log=say)
        timings["pool"] = time.time() - t0
    arrs, _ = c.read_safetensors(pool_files["pool.safetensors"])
    for i, p in enumerate(pool):
        p["greedy32"] = arrs["greedy_ids"][i].tolist()
        p["margins"] = arrs["margins"][i].tolist()

    by_name = {p["name"]: p for p in pool}
    m16 = qo.load_bf16(model_dir)
    with phase("calibrate"):
        t0 = time.time()
        calib_files, double["calibration"] = c.double_run(
            lambda _t: build_calibration(m32, m64, m16, [by_name[n] for n in calib_names]), log=say)
        timings["calibrate"] = time.time() - t0
    del m16
    calibration = json.loads(calib_files["calibration.json"])
    e_ref = qt.e_ref_by_kind(calibration)
    pairs_obj = qt.pairs(e_ref)
    tau, tau_pair = qt.tau_logit(pairs_obj)
    theta = qt.margin_rule(tau)
    say(f"tolerances: tau_logit {tau} ({tau_pair}); gate margin rule {theta}")

    verified = {}

    def verify(p, steps):
        if p["name"] not in verified:
            with c.Torch64pPatches():
                verified[p["name"]] = c.greedy_manual_detail(m64, p["ids"], STEPS)
        d = verified[p["name"]]
        ok = d["ids"][:steps] == p["greedy32"][:steps] and min(d["margins"][:steps]) >= theta
        say(f"select: {p['name']} ({p['length']} tokens, {steps} steps) torch64p {'confirms' if ok else 'does not confirm'}")
        return ok

    with phase("select"):
        t0 = time.time()
        gate, gate_rule = select_gate(pool, theta, verify, params)
        timings["select"] = time.time() - t0
    taps_names = taps_prompts(gate)
    say(f"gate prompts: {[(p['name'], p['length'], p['gated_steps']) for p in gate]}; taps {taps_names}")

    staging = None if args.check or keep_data else args.data_out.rstrip("/\\") + ".staging"
    if staging and os.path.isdir(staging):
        shutil.rmtree(staging)
    with phase("gate"):
        t0 = time.time()
        gate_results, gate_sink, double["gate"] = double_run_stream(
            produce_gate(m32, m64, gate, taps_names, inv, tables, eos_ids, tau), staging, "gate")
        timings["gate"] = time.time() - t0
    with phase("long"):
        t0 = time.time()
        long_results, long_sink, double["long"] = double_run_stream(produce_long(m32, m64, longs), staging, "long")
        timings["long"] = time.time() - t0
    del m32, m64

    # ----- assemble ---------------------------------------------------------------------------------------------------
    greedy_section = {
        "steps": STEPS,
        "margin": "top-1 minus top-2 logit of the reference's decision row, in float64 (common.greedy_manual_detail)",
        "tau_logit": tau, "tau_logit_pair": tau_pair,
        "tau_logit_rule": "the largest logits maxAbs bound over every pair: a candidate within it moves the top-2 margin by "
                          "at most 2 * tau_logit",
        "min_margin": theta, "min_margin_rule": f"max({qt.MARGIN_FLOOR}, {qt.MARGIN_FACTOR} * tau_logit)",
        "max_prompt_tokens": qc.MAX_TOKENS,
        "ids_identical": "a candidate's greedy ids equal the reference's over each gate prompt's gated steps (EOS ignored)",
        "prefix_fallback_min_steps": qt.PREFIX_MIN_STEPS,
        "teacher_forced_min_margin": qt.TEACHER_FORCED_FACTOR * tau,
        "teacher_forced_rule": "on each tf_* sequence (prompt + torch32 greedy ids[0:63]) a candidate's argmax equals the "
                               "reference's at every row whose reference margin is at least 2 * tau_logit",
        "gate_prompts": [p["name"] for p in gate],
        "prompts": {p["name"]: {"gated_steps": p["gated_steps"], "fallback": p["fallback"], "tokens": p["length"],
                                "category": p["category"]} for p in gate},
    }
    calib_ref = {"file": f"{OUT_REL}/calibration.json", "prompts": calib_names,
                 "e_ref_rule": "the largest value over the calibration prompts and layers, per tap kind and metric",
                 "where": {f"{k}|{m}": v for (k, m), v in sorted(e_ref.items())},
                 "k_and_floors": "the dev plan's values (section 5), except the CPU32 relL2 floor: 3e-5 instead of 1e-5, "
                                 "because on this set CpuRef F32 measured up to 1.21e-5 from torch64p and 1.02e-5 from "
                                 "torch32 on the residual stream (L27.out of multilingual-002, cpuref_probe run "
                                 "20261018-091611-7cb55719; the earlier --quick smoke probe 20261018-051907-5db5512c "
                                 "measured 1.40e-5 from torch32 on csharp-000, not a g1 taps prompt), above the plan's "
                                 "1e-5 (ADR G2-0001)",
                 "fixed_pairs": "cpu64-vs-torch64p: relL2 1e-10 on every tap kind. vulkan-vs-d3d12: relL2 1e-4 on every "
                                "tap kind and max |delta logit| 1e-2 on logits only; activations have no maxAbs bound, "
                                "because the residual stream reaches 1e3 and more (ADR G2-0001, item 5)",
                 "adr": "docs/decisions/G2-0001-tolerance-calibration.md"}
    tol_doc = qt.document(SET_NAME, pairs_obj, greedy_section, calib_ref)

    pool_doc = {
        "schema": "lrm.qwen-pool/1", "set": SET_NAME, "seed": qc.SEED,
        "rule": "candidate k of a category draws from DetRng(seed, streamOf(\"qwen3-0.6b-g1/pool/<category>/<k>\")) "
                f"(tools/goldens/qwen/qwen_corpus.py) until a new text of {qc.MIN_TOKENS}..{qc.MAX_TOKENS} tokens comes up",
        "quotas": params.quotas, "margin_rule": theta, "steps": STEPS, "greedy": "torch32 manual loop, EOS ignored",
        "file": f"{OUT_REL}/pool.safetensors",
        "candidates": [{"name": p["name"], "category": p["category"], "k": p["k"], "source": p["source"], "attempts": p["attempts"],
                        "tokens": p["length"], "text_sha256": p["text_sha256"], "min_margin": min(p["margins"]),
                        "min_margin_step": int(np.argmin(p["margins"])), "prefix_at_rule": prefix_len(p["margins"], theta),
                        "distinct_tokens": distinct(p), "eligible": eligible(p, theta),
                        "calibration": p["name"] in calib_names, "gate": any(g["name"] == p["name"] for g in gate)}
                       for p in pool],
    }
    passing = {cat: sum(1 for x in pool_doc["candidates"] if x["category"] == cat and x["eligible"]) for cat in qc.CATEGORIES}
    pool_doc["eligible_per_category"] = passing

    repo_files = {"pool.safetensors": pool_files["pool.safetensors"], "calibration.json": calib_files["calibration.json"],
                  "tolerances.json": c.json_bytes(tol_doc), "pool.json": c.json_bytes(pool_doc), **gate_sink.repo}
    entries = {**{n: c.file_entry(f"{OUT_REL}/{n}", b, n.split(".")[0]) for n, b in repo_files.items()},
               **{n: e for n, e in gate_sink.entries.items() if e["location"] == "data"},
               **long_sink.entries}
    entries_sorted = sorted(entries.values(), key=lambda e: e["path"])
    dbl_all = {"threads": max(d["threads"] for d in double.values()), "identical": all(d["identical"] for d in double.values()),
               "digest": c.sha256_bytes("".join(f"{k}:{double[k]['digest']}\n" for k in sorted(double)).encode()),
               "attempts": [dict(a, stage=k) for k in sorted(double) for a in double[k]["attempts"]],
               "stages": double,
               "scope": "pool and calibration: common.double_run over their files; gate and long: every file hashed in two "
                        "runs, the agreement checks (generate(), no-cache) only in the first"}
    env = c.env_snapshot(dbl_all["threads"])
    patches = c.Torch64pPatches().record()
    prompts_rec = []
    for p in gate:
        prompts_rec.append({"name": p["name"], "role": "gate", "category": p["category"], "length": p["length"], "ids": p["ids"],
                            "dumps": "taps" if p["name"] in taps_names else "logits", "source": p["source"],
                            "text_sha256": p["text_sha256"], "gated_steps": p["gated_steps"], "fallback": p["fallback"]})
    for p in longs:
        prompts_rec.append({"name": p["name"], "role": "long", "category": p["category"], "length": p["length"], "ids": p["ids"],
                            "dumps": "long", "source": p["source"], "text_sha256": p["text_sha256"]})
    e_ref_summary = {f"{k}|{m}": v for (k, m), v in sorted(e_ref.items())}
    man = {
        "schema": c.GOLDEN_SCHEMA, "set": qc.SET, "version": qc.VERSION, "kind": "full-model",
        "about": "Qwen3-0.6B goldens for Gate 1 (G2): torch32 (the spec) and torch64p (the truth) prefill logits, taps for "
                 "two short prompts, 64-step greedy ids and margins, teacher-forced summaries, long-prompt logits, T_spec "
                 "0..8191, the seeded prompt pool with its margins, the calibration and tolerances.json. Large files live "
                 "in LRM_DATA/golden/qwen3-0.6b/g1. Consumers: M2b (gate1-cpu), X1b, I1, I2, I3; tolerances: every "
                 "model-level check (ToleranceSpec).",
        "generator": c.generator_record([os.path.join(c.TOOLS_DIR, *f.split("/")) for f in TOOL_FILES],
                                        "pwsh tools/goldens/run.ps1 qwen/make_qwen.py"),
        "params": params.record(),
        "env": env, "guards": guards, "model": model_rec,
        "config": {k: gen_cfg.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")},
        "oracles": oracle_records(patches),
        "rope": {"theta": 1000000, "head_dim": 128, "max_pos": MAX_POS,
                 "inv_freq_bits": [f"{int(b):08x}" for b in inv], "inv_freq_entries": int(len(inv)),
                 "inv_freq_layout": "head_dim/2 = 64 entries for head_dim 128; HF's full 128-wide row is cat(row, row)",
                 "inv_freq_source": "transformers Qwen3RotaryEmbedding.compute_default_rope_parameters (float32)",
                 "t_spec": "freq = f32(f32(pos) * inv_freq); cos, sin = f32(correctly rounded f64 cos, sin of freq); [max_pos, head_dim/2]",
                 "file": f"{DATA_REL}/rope.safetensors", **rope_rec,
                 "vectors": "inv_freq_sha256 and t_spec_sha256 equal the ReferenceOracles vectors"},
        "taps": {"names": c.activation_names(28), "methods": c.tap_methods_record(), "tap.method": "wrap",
                 "wrapper": "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb (q_rope, k_rope)",
                 "shapes": "[T, features], row-major, HF head order", "prompts": taps_names,
                 "cache_check": {n: {o: gate_results[n][f"cache_check_{o}"] for o in ("torch32", "torch64p")}
                                 for n in taps_names if f"cache_check_torch32" in gate_results[n]}},
        "eos": {"ignored": True, "manual_loop": "never checks EOS",
                "generate": "common.greedy_generate: GenerationConfig(do_sample=False, max_new_tokens=64, eos_token_id=None) "
                            "with model.generation_config replaced by GenerationConfig() for the call; min_new_tokens never used",
                "eos_token_id": eos_ids},
        "threads": dbl_all["threads"], "double_run": dbl_all, "tie_probe": ties, "selftests": selftests,
        "corpus": qc.corpus_record(corpus),
        "pool": {"file": f"{OUT_REL}/pool.json", "tensors": f"{OUT_REL}/pool.safetensors", "size": len(pool), "seed": qc.SEED,
                 "eligible_per_category": passing, "eligible": sum(passing.values())},
        "calibration": {"file": f"{OUT_REL}/calibration.json", "prompts": calib_names,
                        "rule": f"per category in order {list(qc.CATEGORIES)}: the first candidate below {qc.SHORT_TOKENS} tokens and "
                                f"the first at or above, then the longest candidates of the pool, to {params.calibration_size}; "
                                "chosen before any model runs",
                        "bf16": "a bf16 model's distance to torch64p, recorded for Phase 2 calibration only"},
        "gate": {**gate_rule, "margin_rule": theta, "tau_logit": tau, "tau_logit_pair": tau_pair,
                 "prompts": [p["name"] for p in gate], "taps_prompts": taps_names,
                 "taps_rule": "the two shortest gate prompts of different categories"},
        "long": {"prompts": [p["name"] for p in longs], "rows": "logits at the last row of every 64-token block and the last 64 "
                 "rows (logits.rows); final_norm.out and the per-row summaries for every row", "results": long_results},
        "prompts": prompts_rec,
        "greedy": {"steps": STEPS, "min_margin_rule": theta, "prompts": {n: {k: v for k, v in r.items() if k not in ("holdout",)}
                                                                         for n, r in gate_results.items()}},
        "tolerances": {"file": f"{OUT_REL}/tolerances.json", "tau_logit": tau, "margin_rule": theta},
        "measurements": {"e_ref_by_kind": e_ref_summary,
                         "e_ref_relL2_logits_calibration": {p["name"]: p["e_ref"]["logits"] for p in calibration["prompts"]},
                         "holdout": {**{n: r["holdout"]["logits"] for n, r in gate_results.items()},
                                     **{n: r["holdout"]["logits"] for n, r in long_results.items()}},
                         "bf16_logits": {p["name"]: p["bf16"]["logits"] for p in calibration["prompts"]}},
        "files": entries_sorted,
    }
    repo_files["manifest.json"] = c.json_bytes(man)

    # ----- telemetry --------------------------------------------------------------------------------------------------
    for prm in calibration["prompts"]:
        for tap, e in prm["e_ref"]["relL2"].items():
            log.compare(True, "relL2", e, None, pair="torch32-vs-torch64p", tap=tap, prompt=prm["name"], layer=qt.layer_of(tap),
                        kind="measurement")
        for m in ("maxAbs", "kl"):
            log.compare(True, m, prm["e_ref"]["logits"][m], None, pair="torch32-vs-torch64p", tap="logits", prompt=prm["name"],
                        layer=-1, kind="measurement")
        for tap, e in prm["bf16"]["relL2"].items():
            log.compare(True, "relL2", e, None, pair="bf16-vs-torch64p", tap=tap, prompt=prm["name"], layer=qt.layer_of(tap),
                        kind="measurement")
    for name, r in list(gate_results.items()) + list(long_results.items()):
        for m, v in r["holdout"]["logits"].items():
            log.compare(True, m, v, None, pair="torch32-vs-torch64p", tap="logits", prompt=name, layer=-1, kind="measurement")
        for tap, e in r["holdout"].get("relL2", {}).items():
            if tap != "logits":
                log.compare(True, "relL2", e, None, pair="torch32-vs-torch64p", tap=tap, prompt=name, layer=qt.layer_of(tap),
                            kind="measurement")
    for name, r in sorted(gate_results.items()):
        for oracle in ("torch32", "torch64p"):
            g = r[oracle]
            log.gauge("goldens.greedy.min_margin", g["min_margin"], "", prompt=name, oracle=oracle)
            log.gauge("goldens.greedy.min_margin_f32_ulps", g["min_margin_f32_ulps"], "ulp", prompt=name, oracle=oracle)
            log.check("goldens.greedy.f32_argmax_agrees", g["f32_argmax_agrees_every_step"], prompt=name, oracle=oracle)
            log.check("goldens.greedy.manual_equals_no_cache", g["manual_equals_no_cache"], prompt=name, oracle=oracle)
            log.check("goldens.greedy.manual_equals_generate", g["manual_equals_generate_where_f32_agrees"], prompt=name,
                      oracle=oracle, steps=g["generate_compared_steps"])
        log.check("goldens.greedy.torch64p_equals_torch32", r["torch64p_equals_torch32"], prompt=name, steps=r["gated_steps"])
        log.check("goldens.teacher_forced.oracles_agree", r["teacher_forced"]["oracles_agree_where_margin_ge_2tau"], prompt=name)
    for x in pool_doc["candidates"]:
        log.gauge("goldens.pool.min_margin", x["min_margin"], "", prompt=x["name"], category=x["category"], tokens=x["tokens"],
                  eligible=x["eligible"], rule=theta)
    for cat, n in passing.items():
        log.gauge("goldens.pool.eligible", n, "count", category=cat, of=params.quotas[cat])
    for pair, p in sorted(pairs_obj.items()):
        rule = p.get("rule")
        for kind, ms in sorted({**p.get("taps", {}), **({"(default)": p["default"]} if "default" in p else {})}.items()):
            for m, b in sorted(ms.items()):
                log.gauge("goldens.tolerance.bound", b["bound"], "", pair=pair, tap=kind, metric=m,
                          e_ref=b.get("e_ref", float("nan")), k=rule["k"] if rule else 0,
                          floor=rule["floors"][m] if rule else b["bound"])
    log.gauge("goldens.gate.margin_rule", theta, "", tau_logit=tau, tau_pair=tau_pair, prompts=len(gate))
    for k, v in sorted(timings.items()):
        log.gauge("goldens.wall_seconds", v, "s", stage=k)
    total = sum(len(b) for b in repo_files.values())
    for n, b in sorted(repo_files.items()):
        log.gauge("goldens.file_bytes", len(b), "B", file=f"{OUT_REL}/{n}")
    log.gauge("goldens.set_bytes", total, "B", set=qc.SET, budget=SET_BUDGETS[qc.SET])
    data_total = sum(e["bytes"] for e in entries_sorted if e["location"] == "data")
    log.gauge("goldens.data_bytes", data_total, "B", set=qc.SET)
    for k, d in sorted(double.items()):
        log.check("goldens.double_run_identical", d["identical"], set=qc.SET, threads=d["threads"], stage=k)
    data_files = sum(1 for e in entries_sorted if e["location"] == "data")
    say(f"{len(repo_files)} committed files, {total} bytes; {data_files} data files, {data_total} bytes; "
        f"gate {[p['name'] for p in gate]}; wall {time.time() - t_start:.0f} s")

    if args.check:
        problems = c.compare_tree(args.out, repo_files)
        want = {e["path"]: e["sha256"] for e in committed_manifest["files"] if e["location"] == "data"}
        have = {e["path"]: e["sha256"] for e in entries_sorted if e["location"] == "data"}
        problems += [f"data file differs: {p}" for p in sorted(set(want) | set(have)) if want.get(p) != have.get(p)]
        log.check("goldens.regen_identical", not problems, set=qc.SET)
        log.close("failed" if problems else "ok")
        if problems:
            print("CHECK FAILED: regenerated bytes differ")
            for p in problems:
                print("  ", p)
            return 1
        print("check: regenerated bytes are identical to", args.out, "and the data hashes in its manifest")
        return 0
    if keep_data:
        problems = existing_data_problems(args.data_out, entries_sorted)
        log.check("goldens.data_unchanged", not problems, set=qc.SET, files=data_files)
        if problems:
            log.close("failed")
            print("REFUSED: the data folder already holds different files; nothing was written:", args.data_out)
            for p in problems:
                print("  ", p)
            print("A changed set is a new version g<N+1>; --force replaces the folder (a smoke folder, never a published set).")
            return 1
        c.write_tree(args.out, repo_files)
        log.close()
        print("wrote", args.out, "; the", data_files, "data files in", args.data_out, "are byte-identical and untouched",
              "| run log:", log.path)
        return 0
    c.write_tree(args.out, repo_files)
    replace_folder(staging, args.data_out)
    log.close()
    print("wrote", args.out, "and", args.data_out, "| run log:", log.path)
    return 0


def data_present(folder: str) -> bool:
    """True when the folder exists and holds anything."""
    if not os.path.isdir(folder):
        return False
    with os.scandir(folder) as it:
        return any(True for _ in it)


def existing_data_problems(folder: str, entries: list[dict]) -> list[str]:
    """How a data folder differs from the data files a run produced (manifest entries with location "data"): files
    missing from the folder, files the run did not produce, and files whose SHA-256 differs. Empty when identical."""
    want = {}
    for e in entries:
        if e["location"] != "data":
            continue
        if not e["path"].startswith(DATA_REL + "/"):
            raise c.GoldenError(f"data file {e['path']} is not under {DATA_REL}")
        want[e["path"][len(DATA_REL) + 1:]] = e["sha256"]
    have = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            have[os.path.relpath(path, folder).replace(os.sep, "/")] = path
    problems = [f"missing from the folder: {n}" for n in sorted(set(want) - set(have))]
    problems += [f"not produced by this run: {n}" for n in sorted(set(have) - set(want))]
    problems += [f"differs: {n}" for n in sorted(set(want) & set(have)) if c.sha256_file(have[n]) != want[n]]
    return problems


def replace_folder(staging: str, folder: str) -> None:
    """Moves the staging folder into place. The old folder is renamed aside first, so a file another process holds
    open makes the rename fail with the old folder intact instead of half deleted."""
    os.makedirs(os.path.dirname(os.path.abspath(folder)), exist_ok=True)
    if not os.path.isdir(folder):
        os.replace(staging, folder)
        return
    aside = folder.rstrip("/\\") + ".replaced"
    if os.path.isdir(aside):
        shutil.rmtree(aside)
    os.replace(folder, aside)
    os.replace(staging, folder)
    shutil.rmtree(aside)


def check_model_and_tokenizer(model_dir: str) -> dict:
    from tokenizers import Tokenizer

    rec = qo.check_model(model_dir)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    rec["tokenizer"] = {"library": "tokenizers 0.23.3", "load": "Tokenizer.from_file(tokenizer.json)",
                        "encode": "encode(text, add_special_tokens=False).ids (TokenizerApi: no BOS)",
                        "sha256": rec["files"]["tokenizer.json"]}
    rec["_tokenizer"] = tok
    return rec


if __name__ == "__main__":
    sys.exit(main())
