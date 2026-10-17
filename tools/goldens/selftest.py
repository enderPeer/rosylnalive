"""Oracle self-tests (ReferenceOracles contract, docs/contracts/reference-oracles.md).

Each test compares an oracle with an independent implementation and returns a record with the measured error and
its bound. make_tiny.py runs them all and writes the records into the tiny manifest; run this file directly to see
them without generating anything:

    pwsh tools/goldens/run.ps1 selftest.py

The required self-tests are the patched fp64 RMSNorm against a hand-written fp64 RMSNorm (relL2 <= 1e-15) and one
patched attention layer against a hand-written fp64 attention (relL2 <= 1e-13). Each also runs a negative control:
the unpatched fp64 module must fail the same bound, which shows the patch is what makes torch64p fp64.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402  (runs the process guards first)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tiny_model as tm  # noqa: E402

RMSNORM_BOUND = 1e-15
ATTENTION_BOUND = 1e-13
TINY_F64_BOUND = 1e-13


def _rec(passed: bool, metric: str, measured: float, bound: float, **extra) -> dict:
    return {"passed": bool(passed), "metric": metric, "measured": float(measured), "bound": float(bound), **extra}


# ---------------------------------------------------------------------------------------------------------------------
# Hand-written fp64 references (numpy only; no torch arithmetic)
# ---------------------------------------------------------------------------------------------------------------------


def np_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return w * (x / np.sqrt(ms + eps))


def np_rope(z: np.ndarray, cos_h: np.ndarray, sin_h: np.ndarray) -> np.ndarray:
    """NeoX rotation of z [T, heads, head_dim] with half-width tables [T, head_dim/2]."""
    half = z.shape[-1] // 2
    cc = cos_h[:, None, :].astype(np.float64)
    ss = sin_h[:, None, :].astype(np.float64)
    z1, z2 = z[..., :half], z[..., half:]
    return np.concatenate([z1 * cc - z2 * ss, z2 * cc + z1 * ss], axis=-1)


def np_attention(x: np.ndarray, w: dict, prefix: str, cos_h, sin_h, cfg: dict, taps: dict | None = None, layer: int = 0) -> np.ndarray:
    """One Qwen3 attention block in float64: projections, per-head q/k RMSNorm, RoPE, causal GQA softmax, o_proj."""
    t_len = x.shape[0]
    nh, nkv, hd, eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    q = x @ w[prefix + "q_proj.weight"].T
    k = x @ w[prefix + "k_proj.weight"].T
    v = x @ w[prefix + "v_proj.weight"].T
    qn = np_rmsnorm(q.reshape(t_len, nh, hd), w[prefix + "q_norm.weight"], eps)
    kn = np_rmsnorm(k.reshape(t_len, nkv, hd), w[prefix + "k_norm.weight"], eps)
    qr = np_rope(qn, cos_h[:t_len], sin_h[:t_len])
    kr = np_rope(kn, cos_h[:t_len], sin_h[:t_len])
    vv = v.reshape(t_len, nkv, hd)
    group = nh // nkv
    scale = hd**-0.5
    out = np.zeros((t_len, nh, hd))
    for h in range(nh):
        kh, vh = kr[:, h // group], vv[:, h // group]
        scores = (qr[:, h] @ kh.T) * scale
        for t in range(t_len):
            row = scores[t, : t + 1]
            e = np.exp(row - row.max())
            out[t, h] = (e / e.sum()) @ vh[: t + 1]
    o_in = out.reshape(t_len, nh * hd)
    o = o_in @ w[prefix + "o_proj.weight"].T
    if taps is not None:
        p = f"L{layer}.attn."
        taps.update({p + "q_proj.out": q, p + "k_proj.out": k, p + "v_proj.out": v, p + "q_norm.out": qn.reshape(t_len, -1),
                     p + "k_norm.out": kn.reshape(t_len, -1), p + "q_rope": qr.reshape(t_len, -1), p + "k_rope": kr.reshape(t_len, -1),
                     p + "v_cache": v, p + "o_proj.in": o_in, p + "o_proj.out": o})
    return o


def np_forward(ids: list[int], w: dict, cos_h, sin_h, cfg: dict) -> dict:
    """The whole Qwen3 forward in float64 with every ActivationNames tap."""
    eps = cfg["rms_norm_eps"]
    taps = {}
    x = w["model.embed_tokens.weight"][ids]
    taps["embed.out"] = x
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = np_rmsnorm(x, w[p + "input_layernorm.weight"], eps)
        taps[f"L{i}.input_layernorm.out"] = h
        o = np_attention(h, w, p + "self_attn.", cos_h, sin_h, cfg, taps, i)
        mid = x + o
        taps[f"L{i}.resid_mid"] = mid
        h2 = np_rmsnorm(mid, w[p + "post_attention_layernorm.weight"], eps)
        taps[f"L{i}.post_attention_layernorm.out"] = h2
        g = h2 @ w[p + "mlp.gate_proj.weight"].T
        u = h2 @ w[p + "mlp.up_proj.weight"].T
        d_in = (g / (1.0 + np.exp(-g))) * u
        d = d_in @ w[p + "mlp.down_proj.weight"].T
        taps.update({f"L{i}.mlp.gate.out": g, f"L{i}.mlp.up.out": u, f"L{i}.mlp.down.in": d_in, f"L{i}.mlp.down.out": d})
        x = mid + d
        taps[f"L{i}.out"] = x
    f = np_rmsnorm(x, w["model.norm.weight"], eps)
    taps["final_norm.out"] = f
    taps["logits"] = f @ w["model.embed_tokens.weight"].T
    return taps


def tiny_weights_f64() -> dict:
    return {n: c.bf16_widen(b).astype(np.float64) for n, b in tm.weights_bf16().items()}


# ---------------------------------------------------------------------------------------------------------------------
# The self-tests
# ---------------------------------------------------------------------------------------------------------------------


def test_rng_kats() -> dict:
    """DetRng/Philox4x32-10 and streamOf against the LrmCore contract vectors."""
    ok = c.philox4x32((0, 0, 0, 0), (0, 0)) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    ok &= c.philox4x32((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2) == (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)
    ok &= c.philox4x32((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)) == (
        0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)
    ok &= c.stream_of("test") == 0x73C5A94594114DD9
    words = [int(x) for x in c.DetRng(42, 7).words(8)]
    ok &= words == [0x67EE6F2C, 0xE55410CC, 0x6C7ECA35, 0x557398D3, 0xE5DDE940, 0x600F6196, 0x8FCDF8F1, 0x2C8ED839]
    blk = c._philox_blocks(0x299F31D0A4093822, 0x0370734413198A2E, 0x85A308D3243F6A88, 1)[0]
    ok &= [int(x) for x in blk] == [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]
    ok &= (int(c.DetRng(0, 0).words(1)[0]) >> 8) == 6694888
    a = c.DetRng(5, 9, 3).words(10)
    b = c.DetRng(5, 9).words(13)[3:]
    ok &= bool(np.array_equal(a, b))
    return {"passed": bool(ok), "what": "Philox4x32-10, DetRng words and positions, streamOf (LrmCore vectors)"}


def test_bf16_rne() -> dict:
    """bf16 narrowing against the LrmCore vector table and against torch's own float32 -> bfloat16 conversion."""
    vec = [(0x00000000, 0x0000), (0x80000000, 0x8000), (0x3F800000, 0x3F80), (0x3F807FFF, 0x3F80), (0x3F808000, 0x3F80),
           (0x3F808001, 0x3F81), (0x3F818000, 0x3F82), (0xBF818000, 0xBF82), (0x40490FDB, 0x4049), (0x7F7F7FFF, 0x7F7F),
           (0x7F7F8000, 0x7F80), (0x7F7FFFFF, 0x7F80), (0x00000001, 0x0000), (0x00008000, 0x0000), (0x00008001, 0x0001),
           (0x00018000, 0x0002), (0x007FFFFF, 0x0080)]
    src = np.array([v for v, _ in vec], dtype=np.uint32).view(np.float32)
    with np.errstate(over="ignore"):
        got = [int(x) for x in c.bf16_narrow_rne(src)]
    ok = got == [b for _, b in vec]
    words = c.DetRng(3, c.stream_of("o1/bf16-probe")).words(1 << 20)
    f = words.view(np.float32)
    f = f[np.isfinite(f)]
    with np.errstate(over="ignore"):
        mine = c.bf16_narrow_rne(f)
    theirs = torch.from_numpy(f.copy()).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    finite = np.isfinite(c.bf16_widen(theirs))
    agree = bool(np.array_equal(mine[finite], theirs[finite]))
    return {"passed": bool(ok and agree), "vectors": len(vec), "random_patterns": int(finite.sum()),
            "what": "bf16 RNE: LrmCore vectors, and agreement with torch .to(bfloat16) on random float32 patterns"}


def test_safetensors(real_model_dir: str | None) -> dict:
    """The deterministic writer round-trips through our reader and the safetensors library, and rebuilds the real
    checkpoint's header byte for byte (the same convention: __metadata__ first, sorted names, space padding)."""
    from safetensors.numpy import load as st_load

    r = c.DetRng(4, c.stream_of("o1/st-probe"))
    arrays = {"b.f32": r.normal_ih12(12).astype(np.float32).reshape(3, 4), "a.f64": r.normal_ih12(5),
              "c.i32": np.array([1, -2, 3], dtype=np.int32), "d.u32": np.array([7, 2**32 - 1], dtype=np.uint32)}
    blob = c.safetensors_bytes({n: c.st(a) for n, a in arrays.items()}, {"k": "v"})
    mine, meta = c.read_safetensors(blob)
    lib = st_load(blob)
    ok = meta == {"k": "v"} and all(np.array_equal(mine[n], arrays[n]) and np.array_equal(lib[n], arrays[n]) for n in arrays)
    ok &= c.safetensors_bytes({n: c.st(a) for n, a in arrays.items()}, {"k": "v"}) == blob
    rec = {"round_trip": bool(ok)}
    if real_model_dir and os.path.exists(os.path.join(real_model_dir, "model.safetensors")):
        import json
        import struct

        with open(os.path.join(real_model_dir, "model.safetensors"), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            raw = f.read(n)
        hdr = json.loads(raw)
        meta_r = hdr.pop("__metadata__")
        entries = {k: (v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0]) for k, v in hdr.items()}
        rebuilt = c.safetensors_header(entries, meta_r)
        rec["real_header_rebuilt"] = rebuilt == struct.pack("<Q", n) + raw
        rec["real_entries"] = len(entries)
        rec["real_data_offset"] = 8 + n
        ok &= rec["real_header_rebuilt"]
    rec["passed"] = bool(ok)
    rec["what"] = "safetensors writer: round trip (own reader and safetensors 0.8.0) and the Qwen3-0.6B header rebuilt"
    return rec


def test_rmsnorm() -> dict:
    """Patched fp64 Qwen3RMSNorm against np_rmsnorm, relL2 <= 1e-15 per width, with the unpatched negative control."""
    import transformers.models.qwen3.modeling_qwen3 as mq

    r = c.DetRng(5, c.stream_of("o1/rmsnorm-selftest"))
    worst, worst_unpatched, widths = 0.0, None, {}
    for n in (32, 64, 128, 1024):
        x = r.normal_ih12(16 * n).reshape(16, n)
        x[0] *= 1e-3
        x[1] *= 1e3
        w = 1.0 + 0.25 * r.normal_ih12(n)
        ref = np_rmsnorm(x, w, 1e-6)
        mod = mq.Qwen3RMSNorm(n, eps=1e-6).double()
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(w))
            unpatched = mod(torch.from_numpy(x)).numpy()
            with c.Torch64pPatches():
                patched = mod(torch.from_numpy(x)).numpy()
        e, eu = c.rel_l2(patched, ref), c.rel_l2(unpatched, ref)
        widths[str(n)] = {"patched_relL2": e, "unpatched_relL2": eu}
        worst = max(worst, e)
        worst_unpatched = eu if worst_unpatched is None else min(worst_unpatched, eu)
    passed = worst <= RMSNORM_BOUND and worst_unpatched > RMSNORM_BOUND
    return _rec(passed, "relL2", worst, RMSNORM_BOUND, widths=widths, unpatched_min_relL2=worst_unpatched,
                negative_control_fails=bool(worst_unpatched > RMSNORM_BOUND),
                what="patched fp64 Qwen3RMSNorm vs hand-written fp64 RMSNorm (rows at 1e-3, 1 and 1e3 scale)")


def _causal_mask(model, x):
    import transformers.models.qwen3.modeling_qwen3 as mq

    pos = torch.arange(x.shape[1])[None]
    mask = mq.create_causal_mask(config=model.config, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos)
    if mask is None:
        raise c.GoldenError("create_causal_mask returned None for eager attention")
    return pos, mask


def test_attention(model_dir: str, t_spec_tables) -> dict:
    """Layer 0 attention of the tiny torch64p model against np_attention, relL2 <= 1e-13 (T = 13 and T = 70), with
    the unpatched fp64 model as negative control."""
    w = tiny_weights_f64()
    cos_h, sin_h = t_spec_tables
    r = c.DetRng(6, c.stream_of("o1/attention-selftest"))
    res, worst, unpatched_min = {}, 0.0, None
    with c.Torch64pPatches():
        m64 = c.load_model(model_dir, "torch64p", t_spec_tables)
        for t_len in (13, 70):
            x = r.normal_ih12(t_len * tm.H).reshape(t_len, tm.H)
            xt = torch.from_numpy(x)[None]
            pos, mask = _causal_mask(m64, xt)
            with torch.no_grad():
                pe = m64.model.rotary_emb(xt, pos)
                out = m64.model.layers[0].self_attn(hidden_states=xt, position_embeddings=pe, attention_mask=mask)[0][0].numpy()
            ref = np_attention(x, w, "model.layers.0.self_attn.", cos_h, sin_h, tm.CONFIG_BASE)
            res[str(t_len)] = {"patched_relL2": c.rel_l2(out, ref)}
            worst = max(worst, res[str(t_len)]["patched_relL2"])
            res[str(t_len)]["x"] = x
    m32 = c.load_model(model_dir, "torch32").double()  # unpatched fp64: what a plain .double() run computes
    for t_len in (13, 70):
        x = res[str(t_len)].pop("x")
        xt = torch.from_numpy(x)[None]
        pos, mask = _causal_mask(m32, xt)
        with torch.no_grad():
            pe = m32.model.rotary_emb(xt, pos)
            out = m32.model.layers[0].self_attn(hidden_states=xt, position_embeddings=pe, attention_mask=mask)[0][0].numpy()
        ref = np_attention(x, w, "model.layers.0.self_attn.", cos_h, sin_h, tm.CONFIG_BASE)
        eu = c.rel_l2(out, ref)
        res[str(t_len)]["unpatched_relL2"] = eu
        unpatched_min = eu if unpatched_min is None else min(unpatched_min, eu)
    passed = worst <= ATTENTION_BOUND and unpatched_min > ATTENTION_BOUND
    return _rec(passed, "relL2", worst, ATTENTION_BOUND, lengths=res, unpatched_min_relL2=unpatched_min,
                negative_control_fails=bool(unpatched_min > ATTENTION_BOUND),
                what="patched fp64 attention layer (tiny layer 0) vs hand-written fp64 attention")


def test_rope(model_dir: str, inv_bits, t_spec_tables) -> dict:
    """The patched rotary embedding returns T_spec widened, bit for bit; T_spec equals a full mpmath recomputation;
    and the cross-check counts against torch, numpy and the C runtime."""
    cos_h, sin_h = t_spec_tables
    cross = c.t_spec_crosscheck(inv_bits, cos_h.shape[0], cos_h, sin_h, full_mpmath=True)
    with c.Torch64pPatches():
        m64 = c.load_model(model_dir, "torch64p", t_spec_tables)
        pos = torch.arange(cos_h.shape[0])[None]
        with torch.no_grad():
            cos, sin = m64.model.rotary_emb(torch.zeros(1, dtype=torch.float64), pos)
    want_c = np.concatenate([cos_h, cos_h], axis=-1).astype(np.float64)
    want_s = np.concatenate([sin_h, sin_h], axis=-1).astype(np.float64)
    exact = bool(np.array_equal(cos[0].numpy(), want_c) and np.array_equal(sin[0].numpy(), want_s))
    full_ok = cross["cos"]["full_mpmath_mismatches"] == 0 and cross["sin"]["full_mpmath_mismatches"] == 0
    return {"passed": bool(exact and full_ok), "patched_rotary_equals_t_spec": exact, "crosscheck": cross,
            "what": "patched Qwen3RotaryEmbedding returns T_spec widened; T_spec equals full mpmath; libm cross-check"}


def test_tiny_f64(fwd64: dict, prompt: list[int], t_spec_tables) -> dict:
    """Every torch64p tap of a prompt against the hand-written numpy fp64 forward, relL2 <= 1e-13 per tap."""
    cos_h, sin_h = t_spec_tables
    ref = np_forward(prompt, tiny_weights_f64(), cos_h, sin_h, tm.CONFIG_BASE)
    per = {n: c.rel_l2(fwd64[n], ref[n]) for n in c.activation_names(tm.LAYERS)}
    worst_name = max(per, key=lambda n: per[n])
    return _rec(per[worst_name] <= TINY_F64_BOUND, "relL2", per[worst_name], TINY_F64_BOUND, worst_tap=worst_name,
                what="torch64p tiny-model taps vs a hand-written numpy fp64 forward")


def run_all(model_dir: str, inv_bits, t_spec_tables, real_model_dir: str | None) -> dict:
    return {
        "rng_kats": test_rng_kats(),
        "bf16_rne": test_bf16_rne(),
        "safetensors": test_safetensors(real_model_dir),
        "rmsnorm_fp64": test_rmsnorm(),
        "attention_fp64": test_attention(model_dir, t_spec_tables),
        "rope_t_spec": test_rope(model_dir, inv_bits, t_spec_tables),
    }


def main() -> int:
    import json

    c.setup_torch(8)
    c.check_effective_guards()
    with tempfile.TemporaryDirectory(prefix="lrm-o1-") as d:
        for name, data in tm.model_files(eos_token_id=1).items():
            c.write_bytes(os.path.join(d, name), data)
        m32 = c.load_model(d, "torch32")
        inv = c.torch_inv_freq_bits(m32.config)
        cos_h, sin_h, _ = c.t_spec(inv, tm.MAX_POS)
        res = run_all(d, inv, (cos_h, sin_h), os.path.join(c.data_root(), "models", "Qwen3-0.6B"))
        with c.Torch64pPatches():
            m64 = c.load_model(d, "torch64p", (cos_h, sin_h))
            p0 = tm.candidate_ids("p0", 0, 13)
            taps, _ = c.forward_taps(m64, p0)
        res["tiny_fp64_forward"] = test_tiny_f64(taps, p0, (cos_h, sin_h))
    print(json.dumps(res, indent=2, sort_keys=True, default=str))
    failed = [k for k, v in res.items() if not v["passed"]]
    print("self-tests:", "all passed" if not failed else "FAILED " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
