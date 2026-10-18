"""Oracle helpers for the full Qwen3-0.6B goldens (workstream G2), on top of tools/goldens/common.py (O1).

common.py is the ReferenceOracles implementation and is never modified here. This module adds what the full model
needs on top of it:

- check_model: the pinned revision, SHA256SUMS, the toolchain pins, and lm_head.weight equal to embed_tokens;
- load_bf16: a bf16 model for calibration only (it is never an oracle);
- forward_logits and forward_taps_t: one batch-1 prefill; the taps go through common.TapRecorder with the same checks
  as common.forward_taps, but stay torch tensors, so a 900-token fp64 prefill is not copied into numpy twice;
- row_summaries: per row, argmax (lowest index on ties), the top-2 margin in float64, logsumexp in float64 and the
  top-k ids and values ordered by value, then index;
- logits_metrics: relL2 and max |delta logit| in float64 over row blocks, and the largest per-row
  KL(P_reference || P_candidate).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct

import common as c
import numpy as np
import torch
import transformers
import transformers.models.qwen3.modeling_qwen3 as mq

TOP_K = 64
ROW_CHUNK = 128


def _sha256sums(path: str) -> dict:
    out = {}
    with open(path, encoding="ascii") as f:
        for line in f:
            digest, name = line.rstrip("\n").split(" *", 1)
            out[name] = digest
    return out


def check_model(model_dir: str) -> dict:
    """The pinned checkpoint: REVISION equals tools/bootstrap/toolchain.json, every SHA256SUMS line matches its file,
    the toolchain's model and tokenizer hashes equal those lines, the config ties the embeddings, and lm_head.weight
    (the checkpoint stores it) holds exactly the bytes of model.embed_tokens.weight. Returns the manifest record."""
    pin = c.toolchain()["model"]
    with open(os.path.join(model_dir, "REVISION"), encoding="ascii") as f:
        revision = f.read().strip()
    if revision != pin["revision"]:
        raise c.GoldenError(f"model REVISION {revision} is not the pinned {pin['revision']}")
    sums = _sha256sums(os.path.join(model_dir, "SHA256SUMS"))
    files = {}
    for name in sorted(sums):
        actual = c.sha256_file(os.path.join(model_dir, name))
        if actual != sums[name]:
            raise c.GoldenError(f"{name}: SHA-256 {actual} differs from SHA256SUMS {sums[name]}")
        files[name] = actual
    listed = sorted(n for n in os.listdir(model_dir) if n != "SHA256SUMS")
    if listed != sorted(sums):
        raise c.GoldenError(f"SHA256SUMS does not cover exactly the model folder: {listed} vs {sorted(sums)}")
    for name, digest in pin["sha256"].items():
        if files.get(name) != digest:
            raise c.GoldenError(f"{name}: SHA-256 differs from the toolchain pin")
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("tie_word_embeddings") is not True:
        raise c.GoldenError("config.json does not tie the word embeddings")
    path = os.path.join(model_dir, "model.safetensors")
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        emb, head = header["model.embed_tokens.weight"], header.get("lm_head.weight")
        if head is None:
            tied = "absent: lm_head is the embedding"
        else:
            if head["dtype"] != emb["dtype"] or head["shape"] != emb["shape"]:
                raise c.GoldenError("lm_head.weight and model.embed_tokens.weight differ in dtype or shape")
            digests = []
            for e in (emb, head):
                b, end = e["data_offsets"]
                f.seek(base + b)
                h = hashlib.sha256()
                left = end - b
                while left:
                    chunk = f.read(min(left, 1 << 24))
                    h.update(chunk)
                    left -= len(chunk)
                digests.append(h.hexdigest())
            if digests[0] != digests[1]:
                raise c.GoldenError("lm_head.weight differs from model.embed_tokens.weight")
            tied = f"present and byte-identical to model.embed_tokens.weight (SHA-256 {digests[0]})"
    return {
        "path": "LRM_DATA/" + os.path.relpath(model_dir, c.data_root()).replace(os.sep, "/"),
        "id": pin["id"],
        "revision": revision,
        "files": files,
        "lm_head": tied,
        "tensors": len([k for k in header if k != "__metadata__"]),
        "dtype": "BF16 (widened exactly to float32 by from_pretrained(dtype=float32))",
    }


def load_bf16(model_dir: str):
    """Qwen3ForCausalLM in bfloat16 with eager attention, unpatched: calibration only, never an oracle."""
    c.assert_unpatched()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
    model.eval()
    if type(model) is not mq.Qwen3ForCausalLM or model.config._attn_implementation != "eager":
        raise c.GoldenError("bf16 calibration model: unexpected class or attention implementation")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise c.GoldenError("bf16 calibration model: lm_head is not tied")
    return model


def forward_logits(model, ids: list[int]) -> torch.Tensor:
    """One batch-1 prefill with use_cache=True; the logits [T, vocab] in the model's dtype."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=True)
    return out.logits[0]


def forward_taps_t(model, ids: list[int]) -> tuple[dict, dict]:
    """common.forward_taps with the taps left as torch tensors [T, features] in the model's dtype: the same recorder
    (module hooks, pre-hooks and the one apply_rotary_pos_emb wrapper) and the same checks (one wrapper call per layer,
    the ActivationNames set, logits equal to the output, k_rope and v_cache equal to the DynamicCache)."""
    x = torch.tensor([ids], dtype=torch.long)
    nl = model.config.num_hidden_layers
    with torch.no_grad(), c.TapRecorder(model) as rec:
        out = model(input_ids=x, use_cache=True)
        taps = rec.taps
        if rec.wrap_calls != nl:
            raise c.GoldenError(f"apply_rotary_pos_emb wrapper ran {rec.wrap_calls} times for {nl} layers")
    names = c.activation_names(nl)
    if sorted(taps) != sorted(names):
        raise c.GoldenError(f"tap set mismatch: missing {sorted(set(names) - set(taps))}, extra {sorted(set(taps) - set(names))}")
    if not torch.equal(taps["logits"], out.logits[0]):
        raise c.GoldenError("logits tap differs from the model output")
    cache = out.past_key_values
    for i in range(nl):
        layer = cache.layers[i]
        k = layer.keys[0].transpose(0, 1).reshape(len(ids), -1)
        v = layer.values[0].transpose(0, 1).reshape(len(ids), -1)
        if not (torch.equal(k, taps[f"L{i}.attn.k_rope"]) and torch.equal(v, taps[f"L{i}.attn.v_cache"])):
            raise c.GoldenError(f"layer {i}: k_rope or v_cache differs from the DynamicCache contents")
    return {n: taps[n] for n in names}, {"k_rope_equals_cache": True, "v_cache_equals_cache": True, "layers": nl}


def as_numpy(t: torch.Tensor) -> np.ndarray:
    """A float32 view for float32 tensors, float64 for float64; bf16 is widened exactly to float32."""
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def _top_k_row(row: np.ndarray, k: int) -> np.ndarray:
    """The first k indices of the row ordered by value descending, then index ascending (exact, ties included)."""
    part = np.argpartition(-row, k - 1)[:k]
    kth = row[part].min()
    above = np.flatnonzero(row > kth)
    equal = np.flatnonzero(row == kth)[: k - len(above)]
    idx = np.concatenate([above, equal])
    return idx[np.lexsort((idx, -row[idx]))]


def row_summaries(logits: torch.Tensor, k: int = TOP_K) -> dict:
    """Per row of [T, vocab] logits: argmax (torch.argmax, the lowest index on ties: common.tie_probe), the top-2
    margin in float64, logsumexp of the row widened to float64, and the top-k ids (I64) and values (the logits' dtype)
    ordered by value, then index."""
    if torch.isnan(logits).any():
        raise c.GoldenError("NaN logits")
    t_len = logits.shape[0]
    arr = as_numpy(logits)
    argmax = torch.argmax(logits, dim=-1).numpy().astype(np.int64)
    lse = np.empty(t_len, dtype=np.float64)
    top_ids = np.empty((t_len, k), dtype=np.int64)
    top_vals = np.empty((t_len, k), dtype=arr.dtype)
    for s in range(0, t_len, ROW_CHUNK):
        lse[s:s + ROW_CHUNK] = torch.logsumexp(logits[s:s + ROW_CHUNK].double(), dim=-1).numpy()
    for r in range(t_len):
        idx = _top_k_row(arr[r], k)
        top_ids[r] = idx
        top_vals[r] = arr[r][idx]
    if not np.array_equal(top_ids[:, 0], argmax):
        raise c.GoldenError("top-1 differs from argmax")
    margin = top_vals[:, 0].astype(np.float64) - top_vals[:, 1].astype(np.float64)
    return {"argmax": argmax, "margin": margin, "lse": lse, "top_ids": top_ids, "top_values": top_vals}


def kl_rows(reference: torch.Tensor, candidate: torch.Tensor) -> np.ndarray:
    """KL(P_reference || P_candidate) per row, in float64 (ToleranceSpec metric kl takes the largest)."""
    out = np.empty(reference.shape[0], dtype=np.float64)
    for s in range(0, reference.shape[0], ROW_CHUNK):
        lp = torch.log_softmax(reference[s:s + ROW_CHUNK].double(), dim=-1)
        lq = torch.log_softmax(candidate[s:s + ROW_CHUNK].double(), dim=-1)
        out[s:s + ROW_CHUNK] = (lp.exp() * (lp - lq)).sum(dim=-1).numpy()
    return out


def logits_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """relL2, max |delta logit| and the largest per-row KL(P_reference || P_candidate), in float64. The sums of
    squares run over blocks of ROW_CHUNK rows (numpy dot within a block, blocks in order), so a 4,000-token prompt's
    logits are never widened whole."""
    a, b = as_numpy(candidate), as_numpy(reference)
    num = den = 0.0
    worst = 0.0
    for s in range(0, a.shape[0], ROW_CHUNK):
        d = a[s:s + ROW_CHUNK].astype(np.float64).reshape(-1) - b[s:s + ROW_CHUNK].astype(np.float64).reshape(-1)
        r = b[s:s + ROW_CHUNK].astype(np.float64).reshape(-1)
        num += float(np.dot(d, d))
        den += float(np.dot(r, r))
        worst = max(worst, float(np.max(np.abs(d))))
    rel = (0.0 if num == 0.0 else math.inf) if den == 0.0 else math.sqrt(num) / math.sqrt(den)
    return {"relL2": rel, "maxAbs": worst, "kl": float(kl_rows(reference, candidate).max())}
