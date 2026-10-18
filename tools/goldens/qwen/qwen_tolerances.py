"""The ToleranceSpec numbers of goldens/qwen3-0.6b/g1 (workstream G2; the types and the loader are O2's,
docs/contracts/tolerance-spec.md).

    e_ref(kind, metric) = the largest metric(torch32, torch64p) over the calibration prompts and, for per-layer taps,
                          over the 28 layers; torch64p is the reference
    bound               = clamp(k * e_ref, floor, 10 * floor) = min(max(k * e_ref, floor), 10 * floor), in float64

The bound is computed with the same IEEE operations as ToleranceSpec.calibrate, so the loader's exact check holds.
relL2 is bounded for every tap kind; maxAbs (max |delta logit|) and kl only for logits, which is what the floors are
for. The gate's margin rule is max(0.05, 4 * tau_logit), with tau_logit the largest logits maxAbs bound of any pair
whose greedy ids Gate 1 compares.

This module is pure (no torch): `pwsh tools/goldens/run.ps1 qwen/test_qwen.py` checks it against the contract vectors.
"""

from __future__ import annotations

import common as c

SCHEMA = "lrm.tolerances/1"

# The dev plan's starting floors (section 5), except the CPU32 relL2 floor: 3e-5 instead of 1e-5. On g1, CpuRef F32,
# whose sequential FMA order is the spec of every fp32 kernel, measured up to 1.21e-5 from torch64p and 1.02e-5 from
# torch32 on the residual stream (L27.out of multilingual-002, cpuref_probe run 20261018-091611-7cb55719; the --quick
# smoke probe 20261018-051907-5db5512c measured 1.40e-5 from torch32 on csharp-000, not a g1 taps prompt), where
# torch32's own e_ref is about 2e-6 (ADR G2-0001, tools/goldens/qwen/cpuref_probe.fsx). The rule holds one floor per
# pair and metric, so every CPU32 relL2 bound, the logits' included, gets this floor; per-kind floors would be an O2
# schema change (ADR G2-0001, Alternatives).
CPU_FLOORS = {"relL2": 3e-05, "maxAbs": 0.002, "kl": 1e-07}
GPU_FLOORS = {"relL2": 0.0001, "maxAbs": 0.01, "kl": 1e-06}

# Calibrated pairs: the dev plan's k (section 5), unchanged; gpu-vs-cpu32 takes gpu-vs-torch32's rule (ADR G2-0001).
RULED_PAIRS = {
    "cpu32-vs-torch32": {"k": 3, "floors": CPU_FLOORS},
    "cpu32-vs-torch64p": {"k": 2, "floors": CPU_FLOORS},
    "gpu-vs-cpu32": {"k": 10, "floors": GPU_FLOORS},
    "gpu-vs-torch32": {"k": 10, "floors": GPU_FLOORS},
    "gpu-vs-torch64p": {"k": 9, "floors": GPU_FLOORS},
}

# Fixed bounds (no rule): CpuRef F64 against the truth on the full model, and the two GPU backends against each other.
# vulkan-vs-d3d12 bounds max |delta| on logits only, where the contract's sample has it in `default`: the plan's 1e-2 is
# a max |delta logit| bound, and residual-stream activations reach 1e3 and more, so TryBound(vulkan-vs-d3d12, <an
# activation kind>, maxAbs) is None on purpose (ADR G2-0001, item 5).
FIXED_PAIRS = {
    "cpu64-vs-torch64p": {"default": {"relL2": {"bound": 1e-10}}},
    "vulkan-vs-d3d12": {"default": {"relL2": {"bound": 0.0001}}, "taps": {"logits": {"maxAbs": {"bound": 0.01}}}},
}

LOGIT_METRICS = ("relL2", "maxAbs", "kl")
MARGIN_FLOOR = 0.05
MARGIN_FACTOR = 4
TEACHER_FORCED_FACTOR = 2
PREFIX_MIN_STEPS = 16
GREEDY_STEPS = 64


def calibrate(k: float, floor: float, e_ref: float) -> float:
    """ToleranceSpec.calibrate: clamp(k * e_ref, floor, 10 * floor), in the same float64 operations."""
    return min(max(k * e_ref, floor), 10.0 * floor)


def kinds() -> list[str]:
    """Every tap kind (ActivationNames.kind) in canonical order."""
    return ["embed.out", *c.LAYER_TAPS, "final_norm.out", "logits"]


def kind_of(tap: str) -> str:
    return tap.split(".", 1)[1] if tap.startswith("L") else tap


def layer_of(tap: str) -> int:
    return int(tap[1:tap.index(".")]) if tap.startswith("L") else -1


def e_ref_by_kind(calibration: dict) -> dict:
    """The largest e_ref per tap kind and metric over the calibration prompts (and layers), with where it occurred."""
    best: dict = {}
    for prompt in calibration["prompts"]:
        name = prompt["name"]
        for tap, e in prompt["e_ref"]["relL2"].items():
            key = (kind_of(tap), "relL2")
            if key not in best or e > best[key]["e_ref"]:
                best[key] = {"e_ref": e, "prompt": name, "layer": layer_of(tap)}
        for metric in ("maxAbs", "kl"):
            e = prompt["e_ref"]["logits"][metric]
            key = ("logits", metric)
            if key not in best or e > best[key]["e_ref"]:
                best[key] = {"e_ref": e, "prompt": name, "layer": -1}
    missing = [k for k in kinds() if (k, "relL2") not in best]
    if missing:
        raise c.GoldenError(f"calibration has no e_ref for {missing}")
    return best


def pairs(e_ref: dict) -> dict:
    """The `pairs` object of tolerances.json: every ruled pair with a bound per tap kind (relL2) and for logits
    (relL2, maxAbs, kl), each carrying its e_ref; then the fixed pairs."""
    out = {}
    for pair, rule in RULED_PAIRS.items():
        taps = {}
        for kind in kinds():
            metrics = LOGIT_METRICS if kind == "logits" else ("relL2",)
            taps[kind] = {}
            for m in metrics:
                e = e_ref[(kind, m)]["e_ref"]
                taps[kind][m] = {"bound": calibrate(rule["k"], rule["floors"][m], e), "e_ref": e}
        out[pair] = {"rule": {"k": rule["k"], "floors": dict(rule["floors"])}, "taps": taps}
    out.update({p: v for p, v in FIXED_PAIRS.items()})
    return out


def tau_logit(pairs_obj: dict) -> tuple[float, str]:
    """The largest logits maxAbs bound over every pair (each pair's greedy ids are compared in Gate 1)."""
    best, who = 0.0, ""
    for pair, p in sorted(pairs_obj.items()):
        b = p.get("taps", {}).get("logits", {}).get("maxAbs", p.get("default", {}).get("maxAbs"))
        if b is not None and b["bound"] > best:
            best, who = b["bound"], pair
    if not who:
        raise c.GoldenError("no pair bounds the logits' maxAbs")
    return best, who


def margin_rule(tau: float) -> float:
    return max(MARGIN_FLOOR, MARGIN_FACTOR * tau)


def kind_flags(e_ref: dict) -> list[dict]:
    """Where a calibrated bound is not k * e_ref: at the floor (k * e_ref below it) or at the ceiling (k * e_ref above
    10 * floor, so the bound is tighter than the rule asks)."""
    out = []
    for pair, rule in RULED_PAIRS.items():
        for (kind, m), v in sorted(e_ref.items()):
            raw = rule["k"] * v["e_ref"]
            floor = rule["floors"][m]
            state = "floor" if raw < floor else "ceiling" if raw > 10.0 * floor else "k*e_ref"
            out.append({"pair": pair, "tap": kind, "metric": m, "state": state, "headroom": floor * 10.0 / raw if raw else None})
    return out


def document(set_name: str, pairs_obj: dict, greedy: dict, calibration_ref: dict) -> dict:
    return {
        "schema": SCHEMA,
        "set": set_name,
        "about": "ToleranceSpec numbers for Qwen3-0.6B (G2). e_ref is the largest relL2 (and, for logits, max |delta logit| "
                 "and the largest per-row KL) between torch32 and torch64p over the calibration prompts and layers; "
                 "bound = clamp(k * e_ref, floor, 10 * floor). calibration.json holds every measured value.",
        "calibration": calibration_ref,
        "pairs": pairs_obj,
        "greedy": greedy,
    }


def self_check() -> list[str]:
    """The ToleranceSpec contract's calibration vectors, computed here."""
    vectors = [((3, 1e-5, 1.6e-7), 1e-5), ((3, 1e-5, 2e-5), 6.000000000000001e-05), ((3, 1e-5, 1e-3), 1e-4),
               ((9, 1e-4, 5e-5), 0.00045000000000000004)]
    return [f"calibrate{args} = {calibrate(*args)!r}, expected {want!r}" for args, want in vectors if calibrate(*args) != want]
