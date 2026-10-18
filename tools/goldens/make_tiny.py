"""Builds goldens/tiny/g1: the tiny Qwen3 model and its goldens (GoldenSet contract, docs/contracts/golden-set.md).

    pwsh tools/goldens/run.ps1 make_tiny.py            # regenerate goldens/tiny/g1 in place
    pwsh tools/goldens/run.ps1 make_tiny.py --check    # regenerate in memory and compare with the committed bytes

Outputs, all committed (each file at most 2 MiB):
- model/: config.json (real Qwen3 schema), model.safetensors (bf16, no lm_head), REVISION (local:tiny-g1), SHA256SUMS;
- fwd_f32.<prompt>.safetensors and fwd_f64p.<prompt>.safetensors: input_ids and the ActivationNames taps of torch32
  and torch64p (p2 keeps only logits);
- rope.safetensors: torch's inv_freq bits and the T_spec cos/sin tables for positions 0..511;
- greedy.safetensors: 64 greedy ids and top-2 margins per prompt and oracle (EOS ignored);
- grads_f32.safetensors, grads_f64p.safetensors: gradients of the mean next-token cross-entropy on p0;
- adamw_f32.safetensors: one torch AdamW step (foreach=False) from the bf16 weights with the torch32 gradients;
- manifest.json (schema lrm.golden/1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402  (runs the process guards first)
import numpy as np  # noqa: E402

import selftest  # noqa: E402
import tiny_model as tm  # noqa: E402

OUT_REL = f"goldens/{tm.SET}/{tm.VERSION}"
TOOL_FILES = ["common.py", "make_tiny.py", "selftest.py", "tiny_model.py"]


def _st_file(arrays: dict, meta: dict) -> bytes:
    """A golden tensor file; every tensor starts at a multiple of its element size (GoldenSet alignment rule)."""
    tensors = {n: c.st(a) for n, a in arrays.items()}
    offset = 0
    for n in sorted(tensors):
        if offset % c._ST_SIZES[tensors[n].dtype]:
            raise c.GoldenError(f"tensor {n} would start at a misaligned offset {offset}")
        offset += len(tensors[n].data)
    return c.safetensors_bytes(tensors, meta)


def _meta(**kw) -> dict:
    return {"lrm.golden": f"{tm.SET}/{tm.VERSION}", **{k: str(v) for k, v in kw.items()}}


def _write_model(d: str, files: dict) -> None:
    for n, b in files.items():
        c.write_bytes(os.path.join(d, n), b)


def select_prompts(model_dir: str) -> tuple[list[dict], int]:
    """Chooses the prompts (MIN_MARGIN rule) and the tiny eos_token_id (EOS_RULE) with torch32. Neither depends on
    the configured EOS id, so this runs on the model written with a placeholder EOS."""
    m32 = c.load_model(model_dir, "torch32")
    prompts = []
    for name, length, dumps in tm.PROMPTS:
        k, ids, greedy = 0, tm.candidate_ids(name, 0, length), None
        if name in tm.GREEDY_PROMPTS:
            for k in range(tm.MAX_CANDIDATES):
                ids = tm.candidate_ids(name, k, length)
                greedy, margins = c.greedy_manual(m32, ids, tm.GREEDY_STEPS)
                if min(margins) >= tm.MIN_MARGIN:
                    break
            else:
                raise c.GoldenError(f"no {name} candidate reaches margin {tm.MIN_MARGIN}")
        prompts.append({"name": name, "length": length, "dumps": dumps, "candidate": k, "ids": ids, "_greedy": greedy})
    p0 = prompts[0]["_greedy"]
    bos = tm.CONFIG_BASE["bos_token_id"]
    eos = next((t for s, t in enumerate(p0) if s >= tm.EOS_MIN_STEP and t not in p0[:s] and t != bos), None)
    if eos is None:
        raise c.GoldenError("EOS_RULE found no token in p0's greedy output")
    for p in prompts:
        p.pop("_greedy")
    return prompts, eos


def _f32_record(det: dict, gen_info: dict) -> dict:
    """generate() takes its argmax over float32-cast logits, the manual loop over the model dtype. These fields say
    whether that can matter: in torch64p, manual = generate() is guaranteed only where the float32 argmax agrees."""
    return {"generate_argmax_dtype": gen_info["argmax_dtype"], "f32_argmax_agrees_every_step": all(det["f32_argmax_agrees"]),
            "min_margin_f32_ulps": min(det["margin_f32_ulps"])}


def build(model_dir: str, prompts: list[dict], inv_bits: np.ndarray, tables) -> dict:
    """Every golden file except the model and the manifest, plus __results__.json (the measured facts the manifest
    reports). Deterministic: double_run calls it twice and compares the bytes."""
    files, res = {}, {"prompts": {}, "greedy": {}, "e_ref": {}, "grads": {}}
    by_name = {p["name"]: p for p in prompts}
    eos = json.loads(open(os.path.join(model_dir, "config.json"), encoding="utf-8").read())["eos_token_id"]

    # torch32: the spec, unpatched (the tap wrapper is the one allowed addition)
    c.assert_unpatched()
    m32 = c.load_model(model_dir, "torch32")
    fwd32 = {}
    for p in prompts:
        taps, chk = c.forward_taps(m32, p["ids"])
        fwd32[p["name"]] = taps
        keep = taps if p["dumps"] == "taps" else {"logits": taps["logits"]}
        arrays = {"input_ids": np.array(p["ids"], dtype=np.int64), **keep}
        files[f"fwd_f32.{p['name']}.safetensors"] = _st_file(arrays, _meta(oracle="torch32", prompt=p["name"]))
        res["prompts"][p["name"]] = {"cache_check_torch32": chk}
    greedy = {}
    for name in tm.GREEDY_PROMPTS:
        ids = by_name[name]["ids"]
        det = c.greedy_manual_detail(m32, ids, tm.GREEDY_STEPS)
        manual, margins = det["ids"], det["margins"]
        gen, gen_info = c.greedy_generate(m32, ids, tm.GREEDY_STEPS)
        nocache = c.greedy_no_cache(m32, ids, tm.GREEDY_STEPS)
        if not (manual == gen == nocache) or len(manual) != tm.GREEDY_STEPS:
            raise c.GoldenError(f"torch32 greedy {name}: manual, generate() and no-cache recompute disagree")
        leak = c.greedy_generate_without_neutral_config(m32, ids, tm.GREEDY_STEPS)
        greedy[(name, "torch32")] = (manual, margins)
        res["greedy"][name] = {"torch32": {
            "manual_equals_generate": True, "manual_equals_no_cache": True, "min_margin": min(margins),
            "distinct_tokens": len(set(manual)), "eos_first_step": manual.index(eos) if eos in manual else None,
            "generate_effective_config": gen_info["effective"], **_f32_record(det, gen_info),
            "control_without_neutral_model_config": {"new_tokens": len(leak), "stops_at_eos": len(leak) < tm.GREEDY_STEPS and leak[-1] == eos,
                                                     "eos_token_id_leaked": gen_info["eos_token_id_without_neutral_model_config"]}}}
    gp = by_name[tm.GRAD_PROMPT]["ids"]
    grads32, loss32 = c.cross_entropy_grads(m32, gp)
    master = {n: c.bf16_widen(b) for n, b in tm.weights_bf16().items()}
    adamw = c.adamw_step(master, grads32)
    del m32

    # torch64p: the truth, with the three recorded patches
    with c.Torch64pPatches():
        m64 = c.load_model(model_dir, "torch64p", tables)
        for p in prompts:
            taps, chk = c.forward_taps(m64, p["ids"])
            keep = taps if p["dumps"] == "taps" else {"logits": taps["logits"]}
            arrays = {"input_ids": np.array(p["ids"], dtype=np.int64), **keep}
            files[f"fwd_f64p.{p['name']}.safetensors"] = _st_file(arrays, _meta(oracle="torch64p", prompt=p["name"]))
            res["prompts"][p["name"]]["cache_check_torch64p"] = chk
            res["e_ref"][p["name"]] = {n: c.rel_l2(fwd32[p["name"]][n], taps[n]) for n in keep}
            if p["name"] == "p0":
                res["selftest_tiny_fp64_forward"] = selftest.test_tiny_f64(taps, p["ids"], tables)
        for name in tm.GREEDY_PROMPTS:
            ids = by_name[name]["ids"]
            det = c.greedy_manual_detail(m64, ids, tm.GREEDY_STEPS)
            manual, margins = det["ids"], det["margins"]
            gen, gen_info = c.greedy_generate(m64, ids, tm.GREEDY_STEPS)
            nocache = c.greedy_no_cache(m64, ids, tm.GREEDY_STEPS)
            if not (manual == gen == nocache):
                raise c.GoldenError(f"torch64p greedy {name}: manual, generate() and no-cache recompute disagree")
            if manual != greedy[(name, "torch32")][0]:
                raise c.GoldenError(f"greedy {name}: torch32 and torch64p choose different tokens")
            greedy[(name, "torch64p")] = (manual, margins)
            res["greedy"][name]["torch64p"] = {"manual_equals_generate": True, "manual_equals_no_cache": True,
                                              "min_margin": min(margins), "equals_torch32": True,
                                              "generate_effective_config": gen_info["effective"], **_f32_record(det, gen_info)}
        grads64, loss64 = c.cross_entropy_grads(m64, gp)
        del m64

    files["rope.safetensors"] = _st_file({"inv_freq.bits": inv_bits.astype(np.uint32), "t_spec.cos": tables[0], "t_spec.sin": tables[1]},
                                         _meta(theta=tm.CONFIG_BASE["rope_theta"], head_dim=tm.HEAD_DIM, max_pos=tm.MAX_POS))
    g_arrays = {}
    for (name, oracle), (ids, margins) in greedy.items():
        g_arrays[f"{name}.ids.{oracle}"] = np.array(ids, dtype=np.int64)
        g_arrays[f"{name}.margin.{oracle}"] = np.array(margins, dtype=np.float64)
    files["greedy.safetensors"] = _st_file(g_arrays, _meta(steps=tm.GREEDY_STEPS, eos="ignored"))
    files["grads_f32.safetensors"] = _st_file({"loss": np.array([loss32], dtype=np.float32), **grads32},
                                              _meta(oracle="torch32", prompt=tm.GRAD_PROMPT, loss="mean next-token cross-entropy"))
    files["grads_f64p.safetensors"] = _st_file({"loss": np.array([loss64], dtype=np.float64), **grads64},
                                               _meta(oracle="torch64p", prompt=tm.GRAD_PROMPT, loss="mean next-token cross-entropy"))
    files["adamw_f32.safetensors"] = _st_file(adamw, _meta(oracle="torch32", optimizer="torch.optim.AdamW foreach=False", steps=1))
    res["grads"] = {"loss_torch32": loss32, "loss_torch64p": loss64,
                    "e_ref": {n: c.rel_l2(grads32[n], grads64[n]) for n in sorted(grads32)}}
    files["__results__.json"] = c.json_bytes(res)
    return files


def manifest(model_files: dict, golden: dict, results: dict, prompts: list[dict], eos: int, env: dict, guards: dict,
             patches: list, inv_bits, tstats, selftests: dict, dbl: dict, ties: dict) -> dict:
    e_ref_p0 = results["e_ref"]["p0"]
    return {
        "schema": c.GOLDEN_SCHEMA,
        "set": tm.SET,
        "version": tm.VERSION,
        "kind": "tiny-model",
        "about": "Tiny Qwen3 (2 layers, hidden 64, GQA 4/2, head_dim 32, vocab 256) with torch32 and torch64p goldens. "
                 "Consumers: O2 (CpuRef per op), M2a (forward, greedy), X1 and C2 (tiny-model CI paths), Phase 2 (grads, AdamW).",
        "generator": c.generator_record([os.path.join(c.TOOLS_DIR, f) for f in TOOL_FILES], "pwsh tools/goldens/run.ps1 make_tiny.py"),
        "env": env,
        "guards": guards,
        "model": {
            "path": f"{OUT_REL}/model",
            "revision": tm.REVISION,
            "files": {n: c.sha256_bytes(b) for n, b in sorted(model_files.items())},
            "weights": {"dtype": "BF16", "tensors": len(tm.weight_specs()), "lm_head": "absent: tied to model.embed_tokens.weight",
                        "rng": "DetRng(seed, streamOf(\"tiny-g1/<name>\")): Irwin-Hall(12) normal z; matrix scale*z, "
                               "norm scale*(1+z/4); float32 then Bf16.narrowRne",
                        "seed": tm.SEED, "scales": {n: [k, s] for n, (_, k, s) in sorted(tm.weight_specs().items())}},
            "eos_rule": f"eos_token_id is the first token of p0's torch32 greedy output first emitted at step >= {tm.EOS_MIN_STEP} "
                        "(and not bos_token_id), so a decoder that stops at EOS cannot match the 64-step golden",
            "eos_token_id": eos,
        },
        "oracles": {
            "torch32": {"role": "spec", "dtype": "float32", "attn_implementation": "eager", "load": "from_pretrained(dtype=float32)", "patches": []},
            "torch64p": {"role": "truth", "dtype": "float64", "attn_implementation": "eager",
                         "load": "from_pretrained(dtype=float32), then .double()", "patches": patches},
        },
        "rope": {
            "theta": tm.CONFIG_BASE["rope_theta"], "head_dim": tm.HEAD_DIM, "max_pos": tm.MAX_POS,
            "inv_freq_bits": [f"{int(b):08x}" for b in inv_bits],
            "inv_freq_source": "transformers Qwen3RotaryEmbedding.compute_default_rope_parameters (float32)",
            "t_spec": "freq = f32(f32(pos) * inv_freq); cos, sin = f32(correctly rounded f64 cos, sin of freq); [max_pos, head_dim/2]",
            "t_spec_stats": tstats,
        },
        "taps": {
            "names": c.activation_names(tm.LAYERS),
            "methods": c.tap_methods_record(),
            "tap.method": "wrap",
            "wrapper": "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb (q_rope, k_rope)",
            "shapes": "[T, features], row-major, HF head order",
            "cache_check": {p: v for p, v in sorted(results["prompts"].items())},
        },
        "eos": {
            "ignored": True,
            "manual_loop": "never checks EOS",
            "generate": "GenerationConfig(do_sample=False, max_new_tokens=64, eos_token_id=None) with model.generation_config "
                        "replaced by GenerationConfig() for the call; min_new_tokens never used",
            "eos_token_id": eos,
        },
        "threads": dbl["threads"],
        "double_run": dbl,
        "tie_probe": ties,
        "selftests": {**selftests, "tiny_fp64_forward": results["selftest_tiny_fp64_forward"]},
        "prompts": [{"name": p["name"], "length": p["length"], "ids": p["ids"], "dumps": p["dumps"], "candidate": p["candidate"]} for p in prompts],
        "prompt_rule": f"candidate k = 0, 1, ... of NextInt(1, {tm.VOCAB}) draws from DetRng({tm.SEED}, streamOf(\"tiny-g1/prompt/<name>/<k>\")); "
                       f"greedy prompts take the first k whose torch32 top-2 margin is >= {tm.MIN_MARGIN} at all {tm.GREEDY_STEPS} steps",
        "greedy": {"steps": tm.GREEDY_STEPS, "min_margin_rule": tm.MIN_MARGIN, "prompts": results["greedy"]},
        "grads": {"prompt": tm.GRAD_PROMPT, "loss": "mean over t < T-1 of cross_entropy(logits[t], ids[t+1]) in the model dtype",
                  "loss_torch32": results["grads"]["loss_torch32"], "loss_torch64p": results["grads"]["loss_torch64p"]},
        "adamw": {"hparams": c.ADAMW_HPARAMS, "steps": 1, "from": "bf16 weights widened to float32", "grads": "grads_f32.safetensors",
                  "options": "foreach=False, fused=False, amsgrad=False, maximize=False", "names": "param/<name>, exp_avg/<name>, exp_avg_sq/<name>"},
        "measurements": {
            "e_ref_relL2_torch32_vs_torch64p": {p: v for p, v in sorted(results["e_ref"].items())},
            "e_ref_max_p0": max(e_ref_p0.values()),
            "grads_relL2_torch32_vs_torch64p": results["grads"]["e_ref"],
        },
        "files": [c.file_entry(f"{OUT_REL}/{n}", golden[n], _role(n)) for n in sorted(golden)],
    }


def _role(name: str) -> str:
    return "model" if name.startswith("model/") else name.split(".")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    ap.add_argument("--out", default=os.path.join(c.REPO_ROOT, *OUT_REL.split("/")), help="output directory")
    ap.add_argument("--no-telemetry", action="store_true", help="do not write a run log under LRM_DATA/runs")
    args = ap.parse_args()

    log = c.RunLog("make_tiny", enabled=not args.no_telemetry)
    c.setup_torch(8)
    guards = c.check_effective_guards()
    with tempfile.TemporaryDirectory(prefix="lrm-o1-tiny-") as tmp:
        _write_model(tmp, tm.model_files(eos_token_id=1))  # placeholder EOS: selection does not depend on it
        with log.span("goldens.select_prompts", set="tiny"):
            prompts, eos = select_prompts(tmp)
        model_files = tm.model_files(eos_token_id=eos)
        _write_model(tmp, model_files)
        m32 = c.load_model(tmp, "torch32")
        inv_bits = c.torch_inv_freq_bits(m32.config)
        del m32
        cos_h, sin_h, tinfo = c.t_spec(inv_bits, tm.MAX_POS)
        tables = (cos_h, sin_h)
        with log.span("goldens.selftests", set="tiny"):
            selftests = selftest.run_all(tmp, inv_bits, tables, os.path.join(c.data_root(), "models", "Qwen3-0.6B"))
        for k, v in sorted(selftests.items()):
            if "bound" in v:
                log.compare(v["passed"], v["metric"], v["measured"], v["bound"], check=k, pair="oracle-selftest")
            else:
                log.check("goldens.selftest", v["passed"], check=k)
        failed = [k for k, v in selftests.items() if not v["passed"]]
        if failed:
            raise c.GoldenError(f"oracle self-tests failed: {failed}")
        print("self-tests passed:", ", ".join(sorted(selftests)))
        with log.span("goldens.double_run", set="tiny"):
            golden, dbl = c.double_run(lambda _threads: build(tmp, prompts, inv_bits, tables))
        env = c.env_snapshot(dbl["threads"])
        ties = c.tie_probe()
        if not ties["lowest_index_on_ties"]:
            raise c.GoldenError("torch.argmax does not return the lowest index on ties")

    results = json.loads(golden.pop("__results__.json"))
    if not results["selftest_tiny_fp64_forward"]["passed"]:
        raise c.GoldenError(f"tiny fp64 forward self-test failed: {results['selftest_tiny_fp64_forward']}")
    log.compare(True, results["selftest_tiny_fp64_forward"]["metric"], results["selftest_tiny_fp64_forward"]["measured"],
                results["selftest_tiny_fp64_forward"]["bound"], check="tiny_fp64_forward", pair="oracle-selftest")
    golden.update({f"model/{n}": b for n, b in model_files.items()})
    tstats = {"resolved_with_mpmath": tinfo["stats"], "crosscheck": selftests["rope_t_spec"]["crosscheck"]}
    patches = c.Torch64pPatches().record()
    man = manifest(model_files, golden, results, prompts, eos, env, guards, patches, inv_bits, tstats, selftests, dbl, ties)
    files = dict(golden)
    files["manifest.json"] = c.json_bytes(man)

    for p, taps in sorted(results["e_ref"].items()):
        for n, e in sorted(taps.items()):
            layer = int(n[1:n.index(".")]) if n.startswith("L") else -1
            log.compare(True, "relL2", e, None, pair="torch32-vs-torch64p", tap=n, prompt=p, layer=layer, kind="measurement")
    for n, e in sorted(results["grads"]["e_ref"].items()):
        log.compare(True, "relL2", e, None, pair="torch32-vs-torch64p", tensor=n, kind="gradient")
    for name, per in sorted(results["greedy"].items()):
        for oracle, g in sorted(per.items()):
            log.gauge("goldens.greedy.min_margin", g["min_margin"], "", prompt=name, oracle=oracle)
            log.gauge("goldens.greedy.min_margin_f32_ulps", g["min_margin_f32_ulps"], "ulp", prompt=name, oracle=oracle)
            log.check("goldens.greedy.f32_argmax_agrees", g["f32_argmax_agrees_every_step"], prompt=name, oracle=oracle)
    total = sum(len(b) for b in files.values())
    for n, b in sorted(files.items()):
        log.gauge("goldens.file_bytes", len(b), "B", file=f"{OUT_REL}/{n}")
    log.gauge("goldens.set_bytes", total, "B", set="tiny")
    mins = {f"{n}/{o}": round(g["min_margin"], 4) for n, per in results["greedy"].items() for o, g in per.items()}
    print(f"{len(files)} files, {total} bytes; prompts {[(p['name'], p['candidate']) for p in prompts]}; eos {eos}; "
          f"min margins {mins}; max e_ref(p0) {man['measurements']['e_ref_max_p0']:.3e}; double run at {dbl['threads']} threads identical")

    if args.check:
        problems = c.compare_tree(args.out, files)
        log.check("goldens.regen_identical", not problems, set="tiny")
        log.close()
        if problems:
            print("CHECK FAILED: regenerated bytes differ from", args.out)
            for p in problems:
                print("  ", p)
            return 1
        print("check: regenerated bytes are identical to", args.out)
        return 0
    c.write_tree(args.out, files)
    log.close()
    print("wrote", args.out, "| run log:", log.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
