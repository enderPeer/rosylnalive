"""Self-tests of the qwen3-0.6b/g1 tooling that need no model run (G2):

    pwsh tools/goldens/run.ps1 qwen/test_qwen.py

- the ToleranceSpec calibration vectors (docs/contracts/tolerance-spec.md) and the tolerances' structure;
- the corpus, the seeded pool and the fixed prompt rules: deterministic, in quota, in length, distinct;
- the exact top-k order (value, then index, ties included), the prefix rule and the gate selection with its fallback,
  on synthetic margins.
It needs the pinned tokenizer under LRM_DATA, but not the weights' forward pass.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common as c  # noqa: E402  (runs the process guards first)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import make_qwen as mk  # noqa: E402
import qwen_corpus as qc  # noqa: E402
import qwen_oracles as qo  # noqa: E402
import qwen_tolerances as qt  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAILURES.append(what)


def test_tolerances() -> None:
    check(not qt.self_check(), f"calibration vectors {qt.self_check()}")
    fake = {"prompts": [
        {"name": "a", "e_ref": {"relL2": {n: 1e-6 * (1 + i % 7) for i, n in enumerate(c.activation_names(28))},
                                "logits": {"relL2": 2e-6, "maxAbs": 1e-4, "kl": 3e-10}}},
        {"name": "b", "e_ref": {"relL2": {n: 2e-6 for n in c.activation_names(28)}, "logits": {"relL2": 3e-6, "maxAbs": 5e-3, "kl": 1e-10}}},
    ]}
    e = qt.e_ref_by_kind(fake)
    check(e[("logits", "maxAbs")] == {"e_ref": 5e-3, "prompt": "b", "layer": -1}, "e_ref takes the largest value and says where")
    check(e[("embed.out", "relL2")]["e_ref"] == 2e-6, "a model-level kind's e_ref")
    pairs = qt.pairs(e)
    for name, rule in qt.RULED_PAIRS.items():
        p = pairs[name]
        ok = all(b["bound"] == qt.calibrate(rule["k"], rule["floors"][m], b["e_ref"]) for t in p["taps"].values() for m, b in t.items())
        check(ok, f"{name}: every bound equals clamp(k * e_ref, floor, 10 * floor)")
        check(sorted(p["taps"]) == sorted(qt.kinds()) and sorted(p["taps"]["logits"]) == ["kl", "maxAbs", "relL2"],
              f"{name}: every tap kind, and three metrics for logits")
    check(pairs["cpu32-vs-torch32"]["taps"]["logits"]["maxAbs"]["bound"] == 3 * 5e-3, "a bound between floor and ceiling is k * e_ref")
    check(pairs["gpu-vs-torch64p"]["taps"]["logits"]["maxAbs"]["bound"] == 9 * 5e-3, "the GPU k")
    tau, who = qt.tau_logit(pairs)
    check(tau == 10 * 5e-3 and who == "gpu-vs-cpu32", f"tau_logit is the largest logits maxAbs bound ({tau}, {who})")
    check(qt.margin_rule(0.01) == 0.05 and qt.margin_rule(0.05) == 0.2, "margin rule max(0.05, 4 tau)")
    check(len(qt.kinds()) == 21, "21 tap kinds")


def test_pool(tok) -> None:
    corpus = qc.load_corpus()
    a = qc.build_pool(tok, corpus)
    b = qc.build_pool(tok, corpus)
    check([p["ids"] for p in a] == [p["ids"] for p in b], "the pool is deterministic")
    check(len(a) == qc.POOL_SIZE >= 200, f"{len(a)} candidates (at least 200)")
    check(all(qc.MIN_TOKENS <= p["length"] <= qc.MAX_TOKENS for p in a), "every candidate is 8..900 tokens")
    check(len({p["text"] for p in a}) == len(a), "candidate texts are distinct")
    check(all(sum(p["category"] == k for p in a) == n for k, n in qc.POOL_QUOTAS.items()), "category quotas")
    check(all(p["ids"] == qc.encode(tok, p["text"]) for p in a), "ids are the tokenizer's")
    chat = [p for p in a if p["category"] == "chatml"]
    check(all(p["text"].startswith("<|im_start|>") and p["text"].rstrip().endswith(("assistant", "</think>")) for p in chat),
          "ChatML prompts open an assistant turn")
    check(all(151644 in p["ids"] for p in chat), "ChatML prompts hold the <|im_start|> token")
    cal = qc.calibration_subset(a)
    check(cal == qc.calibration_subset(b) and len(cal) == qc.CALIBRATION_SIZE == len(set(cal)), f"calibration subset {cal}")
    check({n.split("-")[0] for n in cal} == set(qc.CATEGORIES), "the calibration covers every category")
    longs = qc.long_prompts(tok, corpus)
    check([p["length"] for p in longs][0] >= 1000 and all(p["length"] <= 4000 for p in longs), f"long prompts {[p['length'] for p in longs]}")


def test_top_k() -> None:
    row = np.array([1.0, 5.0, 3.0, 5.0, 3.0, 3.0, -1.0, 3.0], dtype=np.float32)
    check(qo._top_k_row(row, 4).tolist() == [1, 3, 2, 4], "top-k orders by value, then index, through a tie at the boundary")
    s = qo.row_summaries(torch.from_numpy(np.stack([row, row[::-1].copy()])), k=3)
    check(s["argmax"].tolist() == [1, 4] and s["top_ids"].tolist() == [[1, 3, 2], [4, 6, 0]], "row summaries")
    check(s["margin"].tolist() == [0.0, 0.0], "a tie has margin 0")
    lse = float(torch.logsumexp(torch.from_numpy(row).double(), 0))
    check(s["lse"][0] == lse, "logsumexp in float64")


def test_selection() -> None:
    check(mk.prefix_len([0.2] * 64, 0.05) == 64 and mk.prefix_len([0.2] * 20 + [0.01] + [0.2] * 43, 0.05) == 20, "prefix length")
    params = mk.Params(False)

    def cand(cat, k, length, low_at=None, repeats=False):
        m = [0.3] * 64
        if low_at is not None:
            m[low_at] = 0.001
        ids = [198] * 64 if repeats else list(range(64))
        return {"name": f"{cat}-{k:03d}", "category": cat, "k": k, "length": length, "margins": m, "greedy32": ids}

    pool = [cand("csharp", 0, 50), cand("csharp", 1, 400), cand("csharp", 2, 100), cand("license", 0, 60, 3),
            cand("license", 1, 950), cand("license", 2, 300), cand("design", 0, 200), cand("multilingual", 0, 40, 10),
            cand("chatml", 0, 120), cand("chatml", 1, 500), cand("chatml", 2, 80, 30), cand("design", 1, 90, 40)]
    gate, rule = mk.select_gate(pool, 0.05, lambda p, s: True, params)
    names = [p["name"] for p in gate]
    check(names[:4] == ["csharp-000", "csharp-001", "license-002", "design-000"], f"first and longest per category {names}")
    check("license-001" not in names, "over 900 tokens is never a gate prompt")
    check(rule["fallback_used"] and len(gate) == 8, f"the prefix fallback fills to 8 ({names})")
    fb = [p for p in gate if p["fallback"]]
    check([p["name"] for p in fb] == ["chatml-002", "design-001"] and [p["gated_steps"] for p in fb] == [30, 40],
          "fallback prompts gate on the prefix before their first low-margin step, longest first")
    gate3, _ = mk.select_gate([cand("csharp", 9, 30, repeats=True)] + pool, 0.05, lambda p, s: True, params)
    check("csharp-009" not in [p["name"] for p in gate3] and not mk.eligible(cand("csharp", 9, 30, repeats=True), 0.05),
          f"a degenerate continuation (fewer than {mk.MIN_DISTINCT} distinct tokens) is never a gate prompt")
    gate2, _ = mk.select_gate(pool, 0.05, lambda p, s: p["name"] != "csharp-000", params)
    check([p["name"] for p in gate2 if p["category"] == "csharp"] == ["csharp-001", "csharp-002"],
          "a candidate torch64p does not confirm is replaced by the next by k")
    check(mk.taps_prompts(gate) == ["csharp-000", "chatml-002"], "taps: the two shortest gate prompts of different categories")


def main() -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(c.data_root(), "models", "Qwen3-0.6B", "tokenizer.json"))
    test_tolerances()
    test_top_k()
    test_selection()
    test_pool(tok)
    print("test_qwen:", "all passed" if not FAILURES else f"{len(FAILURES)} FAILED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
