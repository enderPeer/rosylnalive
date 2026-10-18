"""Shared golden tooling for LRM (workstream O1).

This module implements the ReferenceOracles contract (docs/contracts/reference-oracles.md):

- torch32, the spec: unpatched transformers 5.20.1 Qwen3 in fp32 with eager attention;
- torch64p, the truth: the same model in .double() with exactly three recorded patches;
- T_spec, the RoPE table both torch64p and CpuRef use, built from torch's own fp32 inv_freq bits;
- the guards (python -I, offline, no hub kernels, deterministic algorithms, 8 threads with a 1-thread fallback);
- ActivationNames taps through module hooks and pre-hooks plus one wrapper on apply_rotary_pos_emb;
- the greedy rules (EOS ignored everywhere), the argmax tie probe, double runs;
- the GoldenSet writers (docs/contracts/golden-set.md): deterministic safetensors, manifests, SHA256SUMS;
- a run-log emitter that follows the TelemetryFacade record shape, so measurements reach T1's reports.

Import this module before anything from the Hugging Face stack: importing it runs the process guards. Every tool
runs as `python -I` from LRM_DATA/venv-golden (tools/goldens/run.ps1 does this), so tools put their own directory
on sys.path explicitly.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import secrets
import struct
import sys
import textwrap
import time

# ---------------------------------------------------------------------------------------------------------------------
# Process guards. These run at import, before torch, transformers, tokenizers or huggingface_hub can be imported.
# ---------------------------------------------------------------------------------------------------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))

# The offline and kernel switches every golden run needs. A tool sets them when they are absent and refuses to run
# when one is set to anything else.
REQUIRED_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "USE_HUB_KERNELS": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

_GUARDED_MODULES = ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors")


class GoldenError(RuntimeError):
    """A guard, self-test or golden check failed. Golden tools stop on the first one."""


def _guard_process() -> None:
    if not sys.flags.isolated:
        raise SystemExit("golden tools must run as `python -I` (see tools/goldens/run.ps1)")
    for key, want in REQUIRED_ENV.items():
        have = os.environ.get(key)
        if have is None:
            os.environ[key] = want
        elif have != want:
            raise SystemExit(f"{key}={have!r}: golden tools require {key}={want}")
    early = [m for m in _GUARDED_MODULES if m in sys.modules]
    if early:
        raise SystemExit(f"imported before tools/goldens/common.py ran its guards: {', '.join(early)}")
    if importlib.util.find_spec("kernels") is not None:
        raise SystemExit("the `kernels` package is importable; the golden venv must not have it (hub kernels)")


_guard_process()

import numpy as np  # noqa: E402  (after the guards on purpose)
import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.models.qwen3.modeling_qwen3 as mq  # noqa: E402
from torch import nn  # noqa: E402

PINNED_VERSIONS = {"torch": "2.14.1+cpu", "transformers": "5.20.1", "tokenizers": "0.23.3", "safetensors": "0.8.0"}

# Kept so tools can prove that torch32 runs unpatched: every patch and the tap wrapper restore these.
_ORIG = {
    "Qwen3RMSNorm.forward": mq.Qwen3RMSNorm.forward,
    "eager_attention_forward": mq.eager_attention_forward,
    "Qwen3RotaryEmbedding.forward": mq.Qwen3RotaryEmbedding.forward,
    "apply_rotary_pos_emb": mq.apply_rotary_pos_emb,
}


def check_effective_guards() -> dict:
    """Verifies the guards took effect inside the libraries, not just in the environment. Returns the record."""
    import huggingface_hub.constants as hfc
    from transformers.integrations import hub_kernels

    rec = {
        "python_isolated": bool(sys.flags.isolated),
        "env": {k: os.environ.get(k) for k in sorted(REQUIRED_ENV)},
        "hf_hub_offline": bool(hfc.HF_HUB_OFFLINE),
        "hub_is_offline_mode": bool(hfc.is_offline_mode()),
        "hub_kernels_enabled": bool(hub_kernels._kernels_enabled),
        "kernels_importable": importlib.util.find_spec("kernels") is not None,
    }
    if not (rec["hf_hub_offline"] and rec["hub_is_offline_mode"]):
        raise GoldenError(f"offline mode did not take effect: {rec}")
    if rec["hub_kernels_enabled"] or rec["kernels_importable"]:
        raise GoldenError(f"hub kernels are not disabled: {rec}")
    # The eager path must reach the module-level function our patch replaces; nothing may register "eager".
    if "eager" in mq.ALL_ATTENTION_FUNCTIONS:
        raise GoldenError("ALL_ATTENTION_FUNCTIONS registers 'eager'; the attention patch would be bypassed")
    return rec


def setup_torch(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if torch.get_num_threads() != threads:
        raise GoldenError(f"torch.set_num_threads({threads}) gave {torch.get_num_threads()}")


# ---------------------------------------------------------------------------------------------------------------------
# Hashing, text and JSON
# ---------------------------------------------------------------------------------------------------------------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_source(path: str) -> str:
    """SHA-256 of a text source with CRLF folded to LF, so a checkout's line endings never change a manifest."""
    with open(path, "rb") as f:
        return sha256_bytes(f.read().replace(b"\r\n", b"\n"))


def repo_rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), REPO_ROOT).replace(os.sep, "/")


def json_bytes(obj, indent: int | None = 2) -> bytes:
    """The one JSON form golden files use: sorted keys, ASCII only, shortest round-trip floats, LF, final newline."""
    if indent is None:
        text = json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    else:
        text = json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=indent)
    return (text + "\n").encode("ascii")


def write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def data_root() -> str:
    """LRM_DATA when set and not blank, else the dataRoot pinned in tools/bootstrap/toolchain.json (LrmPaths rule)."""
    env = os.environ.get("LRM_DATA", "").strip()
    if env:
        return os.path.abspath(env)
    with open(os.path.join(REPO_ROOT, "tools", "bootstrap", "toolchain.json"), encoding="utf-8") as f:
        return os.path.abspath(json.load(f)["dataRoot"])


def toolchain() -> dict:
    with open(os.path.join(REPO_ROOT, "tools", "bootstrap", "toolchain.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------------------------------------------------


def _lock_versions() -> dict:
    out = {}
    with open(os.path.join(TOOLS_DIR, "requirements.lock"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name, ver = line.split("==")
                out[name.lower().replace("_", "-")] = ver
    return out


def venv_matches_lock() -> dict:
    """Every distribution in the venv equals the lock, and nothing else is installed."""
    lock = _lock_versions()
    have = {}
    for d in importlib.metadata.distributions():
        name = (d.metadata["Name"] or "").lower().replace("_", "-")
        if name:
            have[name] = d.version
    missing = sorted(k for k in lock if k not in have)
    extra = sorted(k for k in have if k not in lock)
    differ = sorted(k for k in lock if k in have and have[k] != lock[k])
    return {"matches": not (missing or extra or differ), "missing": missing, "extra": extra, "differ": differ}


def env_snapshot(threads: int) -> dict:
    """What produced a golden: software versions, library sources that define the oracle, CPU path and threads."""
    import safetensors
    import tokenizers

    lock = venv_matches_lock()
    if not lock["matches"]:
        raise GoldenError(f"venv-golden does not match tools/goldens/requirements.lock: {lock}")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
    }
    for k, v in PINNED_VERSIONS.items():
        if versions[k] != v:
            raise GoldenError(f"{k} {versions[k]} is not the pinned {v}")
    tdir = os.path.dirname(transformers.__file__)
    sources = {
        f"transformers/{p}": sha256_source(os.path.join(tdir, *p.split("/")))
        for p in (
            "models/qwen3/modeling_qwen3.py",
            "models/qwen3/configuration_qwen3.py",
            "modeling_rope_utils.py",
            "masking_utils.py",
            "cache_utils.py",
            "generation/utils.py",
        )
    }
    return {
        "versions": versions,
        "venv_lock": {"file": "tools/goldens/requirements.lock", "sha256": sha256_source(os.path.join(TOOLS_DIR, "requirements.lock")), "matches": True},
        "library_sources": sources,
        "torch_config_sha256": sha256_bytes(torch.__config__.show().encode("utf-8")),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkl": torch.backends.mkl.is_available(),
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "threads": threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


# ---------------------------------------------------------------------------------------------------------------------
# DetRng: Philox4x32-10 exactly as Lrm.Core defines it (LrmCore contract), so F# can regenerate anything drawn here.
# ---------------------------------------------------------------------------------------------------------------------

_M32 = 0xFFFFFFFF
_PH_M0, _PH_M1, _PH_W0, _PH_W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85


def philox4x32(counter, key, rounds: int = 10):
    """One Philox4x32 block for scalar Python ints (Random123 philox4x32_R)."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for r in range(rounds):
        if r:
            k0 = (k0 + _PH_W0) & _M32
            k1 = (k1 + _PH_W1) & _M32
        p0 = _PH_M0 * c0
        p1 = _PH_M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & _M32, p1 & _M32, ((p0 >> 32) ^ c3 ^ k1) & _M32, p0 & _M32
    return c0, c1, c2, c3


def _philox_blocks(seed: int, stream: int, first_block: int, count: int) -> np.ndarray:
    """Philox4x32-10 blocks first_block .. first_block+count-1 as a uint32 array of shape [count, 4]."""
    b = np.arange(first_block, first_block + count, dtype=np.uint64)
    m32 = np.uint64(_M32)
    c0 = b & m32
    c1 = b >> np.uint64(32)
    c2 = np.full(count, stream & _M32, dtype=np.uint64)
    c3 = np.full(count, (stream >> 32) & _M32, dtype=np.uint64)
    k0, k1 = seed & _M32, (seed >> 32) & _M32
    for r in range(10):
        if r:
            k0 = (k0 + _PH_W0) & _M32
            k1 = (k1 + _PH_W1) & _M32
        p0 = c0 * np.uint64(_PH_M0)
        p1 = c2 * np.uint64(_PH_M1)
        n0 = (p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0)
        n2 = (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1)
        c0, c1, c2, c3 = n0, p1 & m32, n2, p0 & m32
    return np.stack([c0, c1, c2, c3], axis=1).astype(np.uint32)


def stream_of(name: str) -> int:
    """DetRng.streamOf: the first 8 bytes, little-endian, of Canonical.hash "lrm.rng.stream/1" over the name."""
    utf8 = name.encode("utf-8")
    digest = hashlib.sha256(b"lrm.rng.stream/1\x00" + struct.pack("<I", len(utf8)) + utf8).digest()
    return struct.unpack("<Q", digest[:8])[0]


class DetRng:
    """DetRng(seed, stream[, position]) from Lrm.Core: draw i is word i mod 4 of Philox block i / 4."""

    def __init__(self, seed: int, stream: int, position: int = 0):
        self.seed, self.stream, self.position = seed & ((1 << 64) - 1), stream & ((1 << 64) - 1), position

    def words(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros(0, dtype=np.uint32)
        first, last = self.position // 4, (self.position + n - 1) // 4
        flat = _philox_blocks(self.seed, self.stream, first, last - first + 1).reshape(-1)
        off = self.position - 4 * first
        self.position += n
        return flat[off : off + n].copy()

    def next_u32(self) -> int:
        return int(self.words(1)[0])

    def next_below(self, bound: int) -> int:
        """Unbiased rejection sampling: redraw while word < (2^32 - bound) mod bound, then word mod bound."""
        if not 0 < bound <= 1 << 32:
            raise ValueError("bound out of range")
        threshold = ((1 << 32) - bound) % bound
        while True:
            w = self.next_u32()
            if w >= threshold:
                return w % bound

    def next_int(self, lo: int, hi: int) -> int:
        return lo + self.next_below(hi - lo)

    def choice(self, seq):
        return seq[self.next_below(len(seq))]

    def f32_uniform(self, n: int) -> np.ndarray:
        """NextF32 draws as float64: (word >> 8) * 2^-24, in [0, 1)."""
        return (self.words(n) >> np.uint32(8)).astype(np.float64) * (2.0**-24)

    def normal_ih12(self, n: int) -> np.ndarray:
        """Irwin-Hall approximation of N(0, 1): the sum of 12 consecutive NextF32 draws minus 6, exact in float64."""
        u = self.f32_uniform(12 * n).reshape(n, 12)
        acc = np.zeros(n, dtype=np.float64)
        for j in range(12):  # fixed order; every partial sum is exact (at most 28 significant bits)
            acc = acc + u[:, j]
        return acc - 6.0


# ---------------------------------------------------------------------------------------------------------------------
# bf16 and float bits
# ---------------------------------------------------------------------------------------------------------------------


def bf16_narrow_rne(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits, round to nearest even (Bf16.narrowRne). Inputs must be finite here."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise GoldenError("bf16_narrow_rne: non-finite input")
    u = x.view(np.uint32).astype(np.uint64)
    r = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    return r.astype(np.uint16)


def bf16_widen(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


# ---------------------------------------------------------------------------------------------------------------------
# safetensors: a deterministic writer (sorted names, header padded with spaces to 8 bytes) and a strict reader.
# ---------------------------------------------------------------------------------------------------------------------

_NP_TO_ST = {
    np.dtype("<f4"): "F32",
    np.dtype("<f8"): "F64",
    np.dtype("<i4"): "I32",
    np.dtype("<i8"): "I64",
    np.dtype("<u4"): "U32",
    np.dtype("u1"): "U8",
}
_ST_TO_NP = {v: k for k, v in _NP_TO_ST.items()}
_ST_TO_NP["BF16"] = np.dtype("<u2")
_ST_SIZES = {"F32": 4, "F64": 8, "I32": 4, "I64": 8, "U32": 4, "U8": 1, "BF16": 2}


class StTensor:
    """One safetensors entry: dtype name, shape and the raw little-endian bytes."""

    __slots__ = ("dtype", "shape", "data")

    def __init__(self, dtype: str, shape, data: bytes):
        n = 1
        for d in shape:
            n *= int(d)
        if len(data) != n * _ST_SIZES[dtype]:
            raise GoldenError(f"safetensors entry: {len(data)} bytes for {dtype}{list(shape)}")
        self.dtype, self.shape, self.data = dtype, [int(d) for d in shape], data


def st(arr: np.ndarray) -> StTensor:
    """An entry from a numpy array of float32, float64, int32, int64, uint32 or uint8."""
    a = np.ascontiguousarray(arr)
    dt = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
    a = a.astype(dt, copy=False)
    name = _NP_TO_ST.get(np.dtype(dt.str))
    if name is None:
        raise GoldenError(f"no safetensors dtype for {a.dtype}")
    return StTensor(name, a.shape, a.tobytes())


def st_bf16(bits: np.ndarray) -> StTensor:
    a = np.ascontiguousarray(bits, dtype="<u2")
    return StTensor("BF16", a.shape, a.tobytes())


def safetensors_header(entries: dict, metadata: dict | None = None) -> bytes:
    """The 8-byte length plus the padded header for {name: (dtype, shape, nbytes)}: __metadata__ first, then tensors
    sorted by name (code point order); compact JSON; spaces pad the header so the data starts at a multiple of 8."""
    parts, offset = [], 0
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise GoldenError("safetensors metadata must map strings to strings")
        parts.append('"__metadata__":' + json.dumps(dict(sorted(metadata.items())), separators=(",", ":"), ensure_ascii=True))
    for name in sorted(entries):
        dtype, shape, nbytes = entries[name]
        entry = {"dtype": dtype, "shape": [int(d) for d in shape], "data_offsets": [offset, offset + nbytes]}
        parts.append(json.dumps(name, ensure_ascii=True) + ":" + json.dumps(entry, separators=(",", ":")))
        offset += nbytes
    header = ("{" + ",".join(parts) + "}").encode("ascii")
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header


def safetensors_bytes(tensors: dict, metadata: dict | None = None) -> bytes:
    """Deterministic safetensors (see safetensors_header); the data follows in the same sorted name order."""
    head = safetensors_header({n: (t.dtype, t.shape, len(t.data)) for n, t in tensors.items()}, metadata)
    return head + b"".join(tensors[n].data for n in sorted(tensors))


def read_safetensors(path_or_bytes) -> tuple[dict, dict]:
    """Strict reader: validates the header, the dtypes, contiguous in-range offsets. Returns (arrays, metadata)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
    (n,) = struct.unpack("<Q", raw[:8])
    if n % 8 or 8 + n > len(raw):
        raise GoldenError("safetensors: bad header length")
    header = json.loads(raw[8 : 8 + n].decode("ascii").rstrip(" "))
    meta = header.pop("__metadata__", {}) or {}
    data = raw[8 + n :]
    arrays, spans = {}, []
    for name, e in header.items():
        dt = e["dtype"]
        b, end = e["data_offsets"]
        count = 1
        for d in e["shape"]:
            count *= d
        if dt not in _ST_TO_NP or end - b != count * _ST_SIZES[dt] or not 0 <= b <= end <= len(data):
            raise GoldenError(f"safetensors: bad entry {name}")
        spans.append((b, end))
        arrays[name] = np.frombuffer(data[b:end], dtype=_ST_TO_NP[dt]).reshape(e["shape"]).copy()
        if dt == "BF16":
            arrays[name] = arrays[name].view(np.uint16)
    spans.sort()
    pos = 0
    for b, end in spans:
        if b != pos:
            raise GoldenError("safetensors: data has holes or overlaps")
        pos = end
    if pos != len(data):
        raise GoldenError("safetensors: trailing data")
    return arrays, meta


# ---------------------------------------------------------------------------------------------------------------------
# ActivationNames
# ---------------------------------------------------------------------------------------------------------------------

LAYER_TAPS = (
    "input_layernorm.out",
    "attn.q_proj.out",
    "attn.k_proj.out",
    "attn.v_proj.out",
    "attn.q_norm.out",
    "attn.k_norm.out",
    "attn.q_rope",
    "attn.k_rope",
    "attn.v_cache",
    "attn.o_proj.in",
    "attn.o_proj.out",
    "resid_mid",
    "post_attention_layernorm.out",
    "mlp.gate.out",
    "mlp.up.out",
    "mlp.down.in",
    "mlp.down.out",
    "out",
)

# How each tap is captured: (method, HF module path relative to model.model.layers[i], or the wrapped function).
TAP_METHODS = {
    "embed.out": ("hook", "model.embed_tokens"),
    "input_layernorm.out": ("hook", "input_layernorm"),
    "attn.q_proj.out": ("hook", "self_attn.q_proj"),
    "attn.k_proj.out": ("hook", "self_attn.k_proj"),
    "attn.v_proj.out": ("hook", "self_attn.v_proj"),
    "attn.q_norm.out": ("hook", "self_attn.q_norm"),
    "attn.k_norm.out": ("hook", "self_attn.k_norm"),
    "attn.q_rope": ("wrap", "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb[0]"),
    "attn.k_rope": ("wrap", "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb[1]"),
    "attn.v_cache": ("hook", "self_attn.v_proj"),
    "attn.o_proj.in": ("prehook", "self_attn.o_proj"),
    "attn.o_proj.out": ("hook", "self_attn.o_proj"),
    "resid_mid": ("prehook", "post_attention_layernorm"),
    "post_attention_layernorm.out": ("hook", "post_attention_layernorm"),
    "mlp.gate.out": ("hook", "mlp.gate_proj"),
    "mlp.up.out": ("hook", "mlp.up_proj"),
    "mlp.down.in": ("prehook", "mlp.down_proj"),
    "mlp.down.out": ("hook", "mlp.down_proj"),
    "out": ("hook", "<layer>"),
    "final_norm.out": ("hook", "model.norm"),
    "logits": ("hook", "lm_head"),
}


def activation_names(num_layers: int) -> list[str]:
    """Every tap name in canonical order: embed.out, then L{i}.<tap> per layer, then final_norm.out and logits."""
    names = ["embed.out"]
    for i in range(num_layers):
        names += [f"L{i}.{t}" for t in LAYER_TAPS]
    return names + ["final_norm.out", "logits"]


def tap_methods_record() -> dict:
    out = {}
    for key, (method, where) in TAP_METHODS.items():
        name = key if key in ("embed.out", "final_norm.out", "logits") else "L{i}." + key
        if method == "wrap":
            out[name] = {"method": "wrap", "function": where}
        else:
            module = where if key in ("embed.out", "final_norm.out", "logits") else (
                "model.layers.{i}" if where == "<layer>" else "model.layers.{i}." + where)
            out[name] = {"method": method, "module": module}
    return out


class TapRecorder:
    """Captures ActivationNames taps for one batch-1 forward: module hooks and pre-hooks, plus one wrapper on
    modeling_qwen3.apply_rotary_pos_emb that records q_rope and k_rope. A pre-hook on each self_attn tells the wrapper
    which layer is calling. Use as a context manager; it restores the original function and removes every hook."""

    def __init__(self, model):
        self.model = model
        self.cfg = model.config
        self.taps: dict[str, torch.Tensor] = {}
        self.enabled = True
        self._layer = None
        self._handles = []
        self.wrap_calls = 0

    def _put(self, name: str, t: torch.Tensor) -> None:
        if not self.enabled:
            return
        if t.dim() < 2 or t.shape[0] != 1:
            raise GoldenError(f"tap {name}: expected a batch-1 tensor, got {tuple(t.shape)}")
        if name in self.taps:
            raise GoldenError(f"tap {name} recorded twice in one forward")
        self.taps[name] = t.detach().reshape(t.shape[1], -1).clone() if t.dim() == 3 else t.detach().clone()

    def _hook(self, name, reshape_heads=False):
        def fn(_m, _args, out):
            o = out[0] if isinstance(out, tuple) else out
            if reshape_heads:  # [1, T, heads, head_dim] -> [1, T, heads * head_dim]
                o = o.reshape(o.shape[0], o.shape[1], -1)
            self._put(name, o)

        return fn

    def _prehook(self, name):
        def fn(_m, args):
            self._put(name, args[0])

        return fn

    def __enter__(self):
        if mq.apply_rotary_pos_emb is not _ORIG["apply_rotary_pos_emb"]:
            raise GoldenError("apply_rotary_pos_emb is already wrapped")
        inner = self.model.model
        h = self._handles
        h.append(inner.embed_tokens.register_forward_hook(self._hook("embed.out")))
        for i, layer in enumerate(inner.layers):
            p = f"L{i}."
            att, mlp = layer.self_attn, layer.mlp
            h.append(layer.input_layernorm.register_forward_hook(self._hook(p + "input_layernorm.out")))
            h.append(att.register_forward_pre_hook(self._enter_layer(i)))
            h.append(att.register_forward_hook(self._leave_layer(i)))
            h.append(att.q_proj.register_forward_hook(self._hook(p + "attn.q_proj.out")))
            h.append(att.k_proj.register_forward_hook(self._hook(p + "attn.k_proj.out")))
            h.append(att.v_proj.register_forward_hook(self._hook(p + "attn.v_proj.out")))
            h.append(att.v_proj.register_forward_hook(self._hook(p + "attn.v_cache")))
            h.append(att.q_norm.register_forward_hook(self._hook(p + "attn.q_norm.out", True)))
            h.append(att.k_norm.register_forward_hook(self._hook(p + "attn.k_norm.out", True)))
            h.append(att.o_proj.register_forward_pre_hook(self._prehook(p + "attn.o_proj.in")))
            h.append(att.o_proj.register_forward_hook(self._hook(p + "attn.o_proj.out")))
            h.append(layer.post_attention_layernorm.register_forward_pre_hook(self._prehook(p + "resid_mid")))
            h.append(layer.post_attention_layernorm.register_forward_hook(self._hook(p + "post_attention_layernorm.out")))
            h.append(mlp.gate_proj.register_forward_hook(self._hook(p + "mlp.gate.out")))
            h.append(mlp.up_proj.register_forward_hook(self._hook(p + "mlp.up.out")))
            h.append(mlp.down_proj.register_forward_pre_hook(self._prehook(p + "mlp.down.in")))
            h.append(mlp.down_proj.register_forward_hook(self._hook(p + "mlp.down.out")))
            h.append(layer.register_forward_hook(self._hook(p + "out")))
        h.append(inner.norm.register_forward_hook(self._hook("final_norm.out")))
        h.append(self.model.lm_head.register_forward_hook(self._hook("logits")))
        orig = _ORIG["apply_rotary_pos_emb"]
        rec = self

        def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):  # the one function wrapper (tap.method wrap)
            q_embed, k_embed = orig(q, k, cos, sin, *args, **kwargs)
            if rec.enabled:
                if rec._layer is None:
                    raise GoldenError("apply_rotary_pos_emb called outside a self_attn forward")
                rec.wrap_calls += 1
                # [1, heads, T, head_dim] -> [1, T, heads * head_dim], HF head order
                rec._put(f"L{rec._layer}.attn.q_rope", q_embed.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1))
                rec._put(f"L{rec._layer}.attn.k_rope", k_embed.transpose(1, 2).reshape(k.shape[0], k.shape[2], -1))
            return q_embed, k_embed

        mq.apply_rotary_pos_emb = apply_rotary_pos_emb
        return self

    def _enter_layer(self, i):
        def fn(_m, _args, _kwargs=None):
            self._layer = i

        return fn

    def _leave_layer(self, i):
        def fn(_m, _args, _out):
            self._layer = None

        return fn

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        mq.apply_rotary_pos_emb = _ORIG["apply_rotary_pos_emb"]
        return False

    def reset(self):
        self.taps = {}
        self.wrap_calls = 0


# ---------------------------------------------------------------------------------------------------------------------
# RoPE: torch's inv_freq bits and T_spec
# ---------------------------------------------------------------------------------------------------------------------


def torch_inv_freq_bits(config) -> np.ndarray:
    """inv_freq exactly as transformers computes it for this config (fp32), as u32 bits."""
    inv, scaling = mq.Qwen3RotaryEmbedding.compute_default_rope_parameters(config)
    if inv.dtype != torch.float32 or scaling != 1.0:
        raise GoldenError(f"unexpected default RoPE parameters: {inv.dtype}, scaling {scaling}")
    return inv.numpy().view(np.uint32).copy()


def _cr_f32_of_f64(fn_name: str, x: float, mp) -> np.float32:
    """f32(round-to-nearest f64 of the exact cos/sin(x)), computed with mpmath at 160 bits."""
    v = getattr(mp, fn_name)(mp.mpf(x))
    return np.float32(float(v))  # mpf -> float rounds to nearest; float -> float32 rounds to nearest even


def t_spec(inv_freq_bits: np.ndarray, max_pos: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """T_spec for positions 0 .. max_pos-1, as [max_pos, head_dim/2] float32 cos and sin tables.

    freq = f32(f32(pos) * inv_freq) in float32 arithmetic; cos and sin = f32(f64 cos/sin(freq)), where the f64 value is
    the correctly rounded one. The fast path uses torch's float64 cos/sin and resolves every entry whose float32
    rounding is not certain within +-2 f64 ulps with mpmath, so the result does not depend on any libm."""
    import mpmath

    mp = mpmath.mp.clone() if hasattr(mpmath.mp, "clone") else mpmath.mp
    mp.prec = 160
    inv = np.asarray(inv_freq_bits, dtype=np.uint32).view(np.float32)
    pos = np.arange(max_pos, dtype=np.float32)
    freq = (pos[:, None] * inv[None, :]).astype(np.float32)  # one float32 multiply per entry
    f64 = freq.astype(np.float64)
    out, stats = {}, {}
    for name, tfn in (("cos", torch.cos), ("sin", torch.sin)):
        c = tfn(torch.from_numpy(f64)).numpy()
        r = c.astype(np.float32)
        lo = np.nextafter(np.nextafter(c, -np.inf), -np.inf).astype(np.float32)
        hi = np.nextafter(np.nextafter(c, np.inf), np.inf).astype(np.float32)
        unsure = np.argwhere((lo != r) | (hi != r))
        for p, j in unsure:
            r[p, j] = _cr_f32_of_f64(name, float(f64[p, j]), mp)
        out[name] = r
        stats[name] = {"resolved_with_mpmath": int(len(unsure))}
    return out["cos"], out["sin"], {"freq_f32": freq, "stats": stats}


def t_spec_crosscheck(inv_freq_bits: np.ndarray, max_pos: int, cos: np.ndarray, sin: np.ndarray, full_mpmath: bool) -> dict:
    """How other float64 cos/sin sources round to float32 against T_spec: torch, numpy and the C runtime (math.cos,
    the same UCRT that .NET's Math.Cos calls on Windows). Optionally recomputes every entry with mpmath."""
    import mpmath

    inv = np.asarray(inv_freq_bits, dtype=np.uint32).view(np.float32)
    freq = (np.arange(max_pos, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    f64 = freq.astype(np.float64)
    res = {}
    for name, ref, tfn, nfn, mfn in (("cos", cos, torch.cos, np.cos, math.cos), ("sin", sin, torch.sin, np.sin, math.sin)):
        t = tfn(torch.from_numpy(f64)).numpy().astype(np.float32)
        n = nfn(f64).astype(np.float32)
        crt = np.array([[mfn(v) for v in row] for row in f64.tolist()], dtype=np.float64).astype(np.float32)
        torch32 = tfn(torch.from_numpy(freq)).numpy()  # what unpatched torch32 uses: fp32 cos of the fp32 angle
        rec = {
            "torch_f64_mismatches": int(np.sum(f32_bits(t) != f32_bits(ref))),
            "numpy_f64_mismatches": int(np.sum(f32_bits(n) != f32_bits(ref))),
            "crt_f64_mismatches": int(np.sum(f32_bits(crt) != f32_bits(ref))),
            "torch32_f32_mismatches": int(np.sum(f32_bits(torch32) != f32_bits(ref))),
            "entries": int(ref.size),
        }
        if full_mpmath:
            mp = mpmath.mp
            mp.prec = 160
            full = np.array([[np.float32(float(getattr(mp, name)(mp.mpf(v)))) for v in row] for row in f64.tolist()], dtype=np.float32)
            rec["full_mpmath_mismatches"] = int(np.sum(f32_bits(full) != f32_bits(ref)))
        res[name] = rec
    return res


# ---------------------------------------------------------------------------------------------------------------------
# The oracles: loading, and the three torch64p patches
# ---------------------------------------------------------------------------------------------------------------------


def _source_edit(obj, old: str, new: str) -> tuple:
    """Rebuilds a function from its own source with exactly one substitution, so the patch is provably that edit."""
    src = textwrap.dedent(inspect.getsource(obj))
    if src.count(old) != 1:
        raise GoldenError(f"patch target {obj.__qualname__}: expected exactly one {old!r}")
    patched = src.replace(old, new)
    ns: dict = {}
    exec(compile(patched, f"<torch64p patch of {obj.__qualname__}>", "exec"), mq.__dict__, ns)
    fn = ns[obj.__name__]
    diff = "".join(difflib.unified_diff(src.splitlines(True), patched.splitlines(True), "original", "patched", n=1))
    return fn, src, patched, diff


def _rotary_forward_t_spec(self, x, position_ids):
    """torch64p RoPE: returns T_spec widened to x.dtype. The table is pinned on the module (lrm_t_spec)."""
    tab = getattr(self, "lrm_t_spec", None)
    if tab is None:
        raise GoldenError("torch64p rotary embedding has no pinned T_spec table")
    cos_h, sin_h = tab  # float32 [max_pos, head_dim/2]
    if int(position_ids.max()) >= cos_h.shape[0] or int(position_ids.min()) < 0:
        raise GoldenError("position outside the pinned T_spec table")
    c = cos_h[position_ids]
    s = sin_h[position_ids]
    cos = torch.cat((c, c), dim=-1)
    sin = torch.cat((s, s), dim=-1)
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Torch64pPatches:
    """The three recorded torch64p patches, as a context manager that restores the originals:

    1. Qwen3RMSNorm.forward stays in the input dtype (the float32 upcast line is removed);
    2. eager_attention_forward runs softmax in query.dtype instead of float32;
    3. Qwen3RotaryEmbedding.forward returns T_spec widened to the input dtype.
    """

    def __init__(self):
        self.rms, self.rms_src, self.rms_new, self.rms_diff = _source_edit(
            _ORIG["Qwen3RMSNorm.forward"], "    hidden_states = hidden_states.to(torch.float32)\n", "")
        self.att, self.att_src, self.att_new, self.att_diff = _source_edit(
            _ORIG["eager_attention_forward"], "dim=-1, dtype=torch.float32)", "dim=-1, dtype=query.dtype)")
        self.rope_src = textwrap.dedent(inspect.getsource(_rotary_forward_t_spec))
        self.active = False

    def __enter__(self):
        if self.active or mq.Qwen3RMSNorm.forward is not _ORIG["Qwen3RMSNorm.forward"]:
            raise GoldenError("torch64p patches are already active")
        mq.Qwen3RMSNorm.forward = self.rms
        mq.eager_attention_forward = self.att
        mq.Qwen3RotaryEmbedding.forward = _rotary_forward_t_spec
        self.active = True
        return self

    def __exit__(self, *exc):
        mq.Qwen3RMSNorm.forward = _ORIG["Qwen3RMSNorm.forward"]
        mq.eager_attention_forward = _ORIG["eager_attention_forward"]
        mq.Qwen3RotaryEmbedding.forward = _ORIG["Qwen3RotaryEmbedding.forward"]
        self.active = False
        return False

    def record(self) -> list:
        return [
            {
                "id": "rmsnorm-input-dtype",
                "target": "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
                "kind": "source-edit",
                "what": "Qwen3RMSNorm.forward stays in the input dtype: the float32 upcast line is removed.",
                "original_sha256": sha256_bytes(self.rms_src.encode()),
                "patched_sha256": sha256_bytes(self.rms_new.encode()),
                "diff": self.rms_diff,
            },
            {
                "id": "softmax-query-dtype",
                "target": "transformers.models.qwen3.modeling_qwen3.eager_attention_forward",
                "kind": "source-edit",
                "what": "eager_attention_forward runs softmax in query.dtype instead of float32.",
                "original_sha256": sha256_bytes(self.att_src.encode()),
                "patched_sha256": sha256_bytes(self.att_new.encode()),
                "diff": self.att_diff,
            },
            {
                "id": "rope-t-spec",
                "target": "transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding.forward",
                "kind": "replace",
                "what": "Qwen3RotaryEmbedding.forward returns T_spec (torch fp32 inv_freq bits, f32 angles, "
                "f32 of the correctly rounded f64 cos/sin) widened to the input dtype.",
                "original_sha256": sha256_bytes(textwrap.dedent(inspect.getsource(_ORIG["Qwen3RotaryEmbedding.forward"])).encode()),
                "patched_sha256": sha256_bytes(self.rope_src.encode()),
                "source": self.rope_src,
            },
        ]


def assert_unpatched() -> None:
    """torch32 runs only on unpatched modeling code (the tap wrapper is the one allowed exception)."""
    for key, fn in (("Qwen3RMSNorm.forward", mq.Qwen3RMSNorm.forward), ("eager_attention_forward", mq.eager_attention_forward),
                    ("Qwen3RotaryEmbedding.forward", mq.Qwen3RotaryEmbedding.forward)):
        if fn is not _ORIG[key]:
            raise GoldenError(f"torch32 must run unpatched, but {key} is patched")


def load_model(model_dir: str, oracle: str, t_spec_tables=None):
    """Loads Qwen3ForCausalLM as torch32 (fp32, eager) or torch64p (the fp32 load, then .double(), with T_spec pinned
    on the rotary embedding). Asserts the class, the attention implementation, the tied lm_head and the RoPE type."""
    if oracle not in ("torch32", "torch64p"):
        raise ValueError(oracle)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager", local_files_only=True)
    model.eval()
    if type(model) is not mq.Qwen3ForCausalLM:
        raise GoldenError(f"expected Qwen3ForCausalLM, got {type(model).__name__}")
    if model.config._attn_implementation != "eager":
        raise GoldenError(f"attention implementation is {model.config._attn_implementation}, not eager")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise GoldenError("lm_head is not tied to embed_tokens")
    rot = model.model.rotary_emb
    if rot.rope_type != "default" or rot.attention_scaling != 1.0:
        raise GoldenError(f"unexpected RoPE: {rot.rope_type}, scaling {rot.attention_scaling}")
    if not all(type(layer) is mq.Qwen3DecoderLayer for layer in model.model.layers):
        raise GoldenError("unexpected decoder layer class")
    bits = torch_inv_freq_bits(model.config)
    if not np.array_equal(rot.inv_freq.numpy().view(np.uint32), bits):
        raise GoldenError("the model's inv_freq buffer differs from compute_default_rope_parameters")
    if oracle == "torch64p":
        model = model.double()
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            raise GoldenError("lm_head untied by .double()")
        if t_spec_tables is None:
            raise GoldenError("torch64p needs the T_spec tables")
        cos_h, sin_h = t_spec_tables
        model.model.rotary_emb.lrm_t_spec = (torch.from_numpy(np.ascontiguousarray(cos_h)), torch.from_numpy(np.ascontiguousarray(sin_h)))
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Forward with taps, greedy, the tie probe, gradients and AdamW
# ---------------------------------------------------------------------------------------------------------------------


def forward_taps(model, ids: list[int]) -> tuple[dict, dict]:
    """One batch-1 prefill with use_cache=True. Returns the taps as numpy arrays [T, features] in the model dtype,
    and the cache check: k_rope and v_cache must equal the DynamicCache contents bit for bit."""
    x = torch.tensor([ids], dtype=torch.long)
    nl = model.config.num_hidden_layers
    with torch.no_grad(), TapRecorder(model) as rec:
        out = model(input_ids=x, use_cache=True)
        taps = rec.taps
        if rec.wrap_calls != nl:
            raise GoldenError(f"apply_rotary_pos_emb wrapper ran {rec.wrap_calls} times for {nl} layers")
    names = activation_names(nl)
    if sorted(taps) != sorted(names):
        missing = sorted(set(names) - set(taps))
        extra = sorted(set(taps) - set(names))
        raise GoldenError(f"tap set mismatch: missing {missing}, extra {extra}")
    if not torch.equal(taps["logits"], out.logits[0]):
        raise GoldenError("logits tap differs from the model output")
    cache = out.past_key_values
    kv_ok = True
    for i in range(nl):
        layer = cache.layers[i]
        k = layer.keys[0].transpose(0, 1).reshape(len(ids), -1)  # [nKV, T, hd] -> [T, nKV*hd]
        v = layer.values[0].transpose(0, 1).reshape(len(ids), -1)
        kv_ok &= bool(torch.equal(k, taps[f"L{i}.attn.k_rope"])) and bool(torch.equal(v, taps[f"L{i}.attn.v_cache"]))
    if not kv_ok:
        raise GoldenError("k_rope or v_cache differs from the DynamicCache contents")
    arrays = {n: taps[n].numpy().copy() for n in names}
    return arrays, {"k_rope_equals_cache": True, "v_cache_equals_cache": True, "layers": nl}


# transformers 5.20.1 generation/utils.py, GenerationMixin._sample: the last logits row is cast to float32 before the
# logits processors and torch.argmax. greedy_generate asserts that the pinned source still does this.
GENERATE_F32_CAST = "next_token_logits = outputs.logits[:, -1].to(copy=True, dtype=torch.float32, device=input_ids.device)"


def _argmax_margin(logits_row: torch.Tensor) -> tuple[int, float]:
    if torch.isnan(logits_row).any():
        raise GoldenError("NaN logits")
    idx = int(torch.argmax(logits_row))
    top = torch.topk(logits_row.double(), 2).values
    return idx, float(top[0] - top[1])


def _f32_view(logits_row: torch.Tensor, idx: int, margin: float) -> tuple[bool, float]:
    """What generate() sees: it casts the last logits row to float32 before its argmax (GENERATE_F32_CAST). Returns
    whether the float32-rounded row has the same argmax as the row itself, and the top-2 margin in float32 ulps of the
    top logit. For a float32 row the cast is the identity, so the first value is always true."""
    row32 = logits_row.to(torch.float32)
    agrees = int(torch.argmax(row32)) == idx
    ulp = float(np.spacing(np.float32(abs(float(row32[idx])))))
    return agrees, margin / ulp


def greedy_manual_detail(model, ids: list[int], steps: int) -> dict:
    """greedy_manual plus, per step, what a float32 argmax over the same row picks (_f32_view): returns `ids`,
    `margins`, `f32_argmax_agrees` and `margin_f32_ulps`. In torch64p, generate() can agree with the manual loop at
    a step only where f32_argmax_agrees holds (ReferenceOracles, Greedy decoding and EOS)."""
    out_ids, margins, agrees, ulps = [], [], [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=True)
        past = out.past_key_values
        for s in range(steps):
            row = out.logits[0, -1]
            nxt, m = _argmax_margin(row)
            a, u = _f32_view(row, nxt, m)
            out_ids.append(nxt)
            margins.append(m)
            agrees.append(a)
            ulps.append(u)
            if s + 1 < steps:
                out = model(input_ids=torch.tensor([[nxt]], dtype=torch.long), past_key_values=past, use_cache=True)
                past = out.past_key_values
    return {"ids": out_ids, "margins": margins, "f32_argmax_agrees": agrees, "margin_f32_ulps": ulps}


def greedy_manual(model, ids: list[int], steps: int) -> tuple[list[int], list[float]]:
    """The canonical schedule: one prefill, then single-token decode steps with the cache. EOS is never checked.
    Each step takes torch.argmax of the last row in the model's dtype; the margin is top-1 minus top-2 in float64."""
    d = greedy_manual_detail(model, ids, steps)
    return d["ids"], d["margins"]


def greedy_no_cache(model, ids: list[int], steps: int) -> list[int]:
    """Recomputes every step from scratch over the whole sequence, without a cache."""
    seq = list(ids)
    with torch.no_grad():
        for _ in range(steps):
            logits = model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False).logits[0, -1]
            seq.append(_argmax_margin(logits)[0])
    return seq[len(ids):]


def greedy_generate(model, ids: list[int], steps: int) -> tuple[list[int], dict]:
    """generate() with a fresh GenerationConfig(do_sample=False, max_new_tokens=steps, eos_token_id=None).

    transformers 5.20.1 fills every None field of the passed config from model.generation_config, which carries the
    checkpoint's eos_token_id, so eos_token_id=None alone does not disable EOS. The model's generation config is
    therefore replaced by a neutral GenerationConfig() for the call and restored afterwards; the effective config is
    asserted to have no EOS. min_new_tokens is never used (it masks the EOS logits).

    generate() takes its argmax over float32 (GENERATE_F32_CAST, asserted against the pinned source). In torch64p it
    therefore picks from float32-rounded logits; the canonical torch64p ids are the manual loop's, and
    greedy_manual_detail records the steps where the two could differ."""
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    if GENERATE_F32_CAST not in inspect.getsource(GenerationMixin._sample):
        raise GoldenError("GenerationMixin._sample no longer casts the logits row to float32; review the torch64p greedy rule")
    gc = GenerationConfig(do_sample=False, max_new_tokens=steps, eos_token_id=None)
    saved = model.generation_config
    leaked = copy_effective(model, gc)
    model.generation_config = GenerationConfig()
    try:
        effective = copy_effective(model, gc)
        if effective["eos_token_id"] is not None or effective["do_sample"] or effective.get("min_new_tokens"):
            raise GoldenError(f"generate() would not run EOS-free greedy: {effective}")
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)
    finally:
        model.generation_config = saved
    new = out[0, len(ids):].tolist()
    return new, {"effective": effective, "eos_token_id_without_neutral_model_config": leaked["eos_token_id"],
                 "argmax_dtype": "float32"}


def greedy_generate_without_neutral_config(model, ids: list[int], steps: int) -> list[int]:
    """Negative control: the same GenerationConfig, but with the checkpoint's own model.generation_config left in
    place. In transformers 5.20.1 the checkpoint's eos_token_id leaks in and generation stops at EOS."""
    from transformers import GenerationConfig

    gc = GenerationConfig(do_sample=False, max_new_tokens=steps, eos_token_id=None)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)
    return out[0, len(ids):].tolist()


def copy_effective(model, gc) -> dict:
    eff, _ = model._prepare_generation_config(gc)
    keep = ("do_sample", "eos_token_id", "max_new_tokens", "min_new_tokens", "num_beams", "pad_token_id",
            "repetition_penalty", "temperature", "top_k", "top_p")
    return {k: getattr(eff, k, None) for k in keep}


def tie_probe() -> dict:
    """torch.argmax must return the lowest index among equal maxima (CpuReferenceSemantics argmax rule), for small and
    vocabulary-sized rows, in float32 and float64, at the current thread count."""
    cases, ok = 0, True
    rng = DetRng(0x7E1E, stream_of("o1/tie-probe"))
    for n in (2, 7, 256, 4097, 151936):
        for dt in (torch.float32, torch.float64):
            for _ in range(4):
                row = torch.from_numpy(rng.normal_ih12(n)).to(dt)
                a = rng.next_below(n)
                b = rng.next_below(n)
                peak = row.max() + 1
                row[a] = peak
                row[b] = peak
                want = min(a, b)
                got = int(torch.argmax(row))
                ok &= got == want
                cases += 1
    return {"lowest_index_on_ties": bool(ok), "cases": cases}


def cross_entropy_grads(model, ids: list[int]) -> tuple[dict, float]:
    """Gradients of the mean next-token cross-entropy over the prompt, in the model's own dtype.

    loss = mean over t < T-1 of CE(logits[t], ids[t+1]). HF's built-in loss upcasts logits to float32, which would put
    a float32 step into torch64p, so the loss is computed here with F.cross_entropy in the model dtype instead."""
    model.zero_grad(set_to_none=True)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.enable_grad():
        logits = model(input_ids=x, use_cache=False).logits[0]
        loss = nn.functional.cross_entropy(logits[:-1], x[0, 1:], reduction="mean")
        loss.backward()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            raise GoldenError(f"no gradient for {name}")
        grads[name] = p.grad.detach().numpy().copy()
    model.zero_grad(set_to_none=True)
    return grads, float(loss.detach())


ADAMW_HPARAMS = {"lr": 1e-3, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}


def adamw_step(params: dict, grads: dict) -> dict:
    """One torch.optim.AdamW step (foreach=False, fused=False) from zero state on float32 master weights."""
    ts = {n: torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(params[n], dtype=np.float32)).clone()) for n in sorted(params)}
    opt = torch.optim.AdamW([ts[n] for n in sorted(ts)], lr=ADAMW_HPARAMS["lr"], betas=tuple(ADAMW_HPARAMS["betas"]),
                            eps=ADAMW_HPARAMS["eps"], weight_decay=ADAMW_HPARAMS["weight_decay"], amsgrad=False,
                            maximize=False, foreach=False, capturable=False, differentiable=False, fused=False)
    for n in sorted(ts):
        ts[n].grad = torch.from_numpy(np.ascontiguousarray(grads[n], dtype=np.float32)).clone()
    opt.step()
    out = {}
    for n in sorted(ts):
        stt = opt.state[ts[n]]
        out[f"param/{n}"] = ts[n].detach().numpy().copy()
        out[f"exp_avg/{n}"] = stt["exp_avg"].numpy().copy()
        out[f"exp_avg_sq/{n}"] = stt["exp_avg_sq"].numpy().copy()
        if float(stt["step"]) != 1.0:
            raise GoldenError("AdamW step count is not 1")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------------------------------------


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||b|| in float64 (b is the reference); 0 when both are zero."""
    a64 = np.asarray(a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.sqrt(np.dot(b64, b64)))
    num = float(np.sqrt(np.dot(a64 - b64, a64 - b64)))
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))) if np.size(a) else 0.0


# ---------------------------------------------------------------------------------------------------------------------
# Double runs
# ---------------------------------------------------------------------------------------------------------------------


def digest_of_files(files: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\x00" + hashlib.sha256(files[name]).digest())
    return h.hexdigest()


def double_run(build, threads_first: int = 8, log=print) -> tuple[dict, dict]:
    """Runs build(threads) twice with 8 threads and compares the output digests; if they differ, falls back to one
    thread and requires that pair to agree. Returns the second run's outputs and the record for the manifest."""
    attempts = []
    for threads in (threads_first, 1):
        setup_torch(threads)
        a = build(threads)
        setup_torch(threads)
        b = build(threads)
        da, db = digest_of_files(a), digest_of_files(b)
        attempts.append({"threads": threads, "digest_a": da, "digest_b": db, "identical": da == db})
        log(f"double run at {threads} thread(s): {'identical' if da == db else 'DIFFERENT'} ({da[:16]})")
        if da == db:
            return b, {"threads": threads, "identical": True, "digest": db, "attempts": attempts}
    raise GoldenError(f"double runs differ even at 1 thread: {attempts}")


# ---------------------------------------------------------------------------------------------------------------------
# GoldenSet manifests
# ---------------------------------------------------------------------------------------------------------------------

MAX_FILE_BYTES = 2 * 1024 * 1024
GOLDEN_SCHEMA = "lrm.golden/1"


def generator_record(tool_files: list[str], command: str) -> dict:
    return {
        "command": command,
        "files": {repo_rel(p): sha256_source(p) for p in sorted(tool_files)},
    }


def file_entry(rel_path: str, data: bytes, role: str, location: str = "repo") -> dict:
    if location == "repo" and len(data) > MAX_FILE_BYTES:
        raise GoldenError(f"{rel_path} is {len(data)} bytes; committed golden files are at most {MAX_FILE_BYTES}")
    return {"path": rel_path, "location": location, "bytes": len(data), "sha256": sha256_bytes(data), "role": role}


def sha256sums_bytes(files: dict) -> bytes:
    """SHA256SUMS in sha256sum binary-mode form, sorted by name: '<hex> *<name>' per line."""
    return "".join(f"{sha256_bytes(files[n])} *{n}\n" for n in sorted(files)).encode("ascii")


def write_tree(root: str, files: dict) -> None:
    """Writes {relative path: bytes} under root, replacing it entirely."""
    import shutil

    if os.path.isdir(root):
        shutil.rmtree(root)
    for rel, data in files.items():
        write_bytes(os.path.join(root, *rel.split("/")), data)


# Manifest fields that describe the generating machine and data root rather than the goldens. They make --check a
# check for the same machine and LRM_DATA (GoldenSet contract, Verification); compare_tree names them when they
# are the only differences.
MACHINE_FIELDS = ("env.", "double_run.", "threads", "selftests.safetensors.real_")


def json_diff_paths(old, new, path: str = "") -> list[tuple[str, object, object]]:
    """The JSON paths (dotted keys, [i] for list items) whose values differ, with the old and new values. A key present
    on one side only is reported with the value <absent> on the other."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            p = f"{path}.{k}" if path else k
            if k not in old or k not in new:
                out.append((p, old.get(k, "<absent>"), new.get(k, "<absent>")))
            else:
                out.extend(json_diff_paths(old[k], new[k], p))
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in json_diff_paths(a, b, f"{path}[{i}]")]
    return [] if old == new and type(old) is type(new) else [(path, old, new)]


def _short(v, n: int = 80) -> str:
    s = json.dumps(v, sort_keys=True, ensure_ascii=True)
    return s if len(s) <= n else s[: n - 3] + "..."


def compare_tree(root: str, files: dict, max_paths: int = 40) -> list[str]:
    """Differences between an on-disk golden tree and freshly generated bytes. For a JSON file that differs, the
    following lines name every JSON path whose value changed (committed -> regenerated), and say when all of them are
    machine or data-root fields (MACHINE_FIELDS)."""
    problems = []
    have = set()
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            have.add(os.path.relpath(os.path.join(dirpath, n), root).replace(os.sep, "/"))
    for rel in sorted(set(files) | have):
        if rel not in files:
            problems.append(f"extra file on disk: {rel}")
        elif rel not in have:
            problems.append(f"missing on disk: {rel}")
        else:
            with open(os.path.join(root, *rel.split("/")), "rb") as f:
                committed = f.read()
            if committed == files[rel]:
                continue
            problems.append(f"bytes differ: {rel}")
            if rel.endswith(".json"):
                try:
                    diffs = json_diff_paths(json.loads(committed), json.loads(files[rel]))
                except ValueError:
                    problems.append(f"  {rel}: the committed file is not valid JSON")
                    continue
                for p, a, b in diffs[:max_paths]:
                    problems.append(f"  {rel} {p or '<root>'}: {_short(a)} -> {_short(b)}")
                if len(diffs) > max_paths:
                    problems.append(f"  {rel}: {len(diffs) - max_paths} more paths differ")
                if not diffs:
                    problems.append(f"  {rel}: same JSON values, different bytes (formatting)")
                elif all(p.startswith(MACHINE_FIELDS) for p, _, _ in diffs):
                    problems.append(f"  {rel}: every difference is a machine or data-root field "
                                    f"({', '.join(MACHINE_FIELDS)}); --check is a same-machine, same-LRM_DATA check")
    return problems


# ---------------------------------------------------------------------------------------------------------------------
# Run log: TelemetryFacade records as RunLog-style JSONL (keys v, t, k, n, d, x, u, node, seq, a) under LRM_DATA/runs.
# ---------------------------------------------------------------------------------------------------------------------


class RunLog:
    """A writer of T1's RunLog contract (docs/contracts/run-log.md, schema lrm.run/1) for the Python tools, which cannot
    call the F# facade. It writes LRM_DATA/runs/<runId>/manifest.json and events.n0.jsonl: a run.node header, the
    records (TelemetryFacade naming rules; golden.compare per ComparatorsAndGoldenReader), and a run.end trailer, so
    `lrm report <runId>` renders them (the parity template by default). Timestamps go only to the run log, never into
    goldens. close(status) finishes the run; a run that ends without close is finished as aborted at exit."""

    KINDS = ("span", "counter", "gauge", "histogram", "event", "gpu", "meter", "check")

    def __init__(self, tool: str, enabled: bool = True, component: str = "goldens", kind: str = "parity",
                 title: str | None = None, params: dict | None = None):
        self.enabled = enabled
        self.seq = 0
        self.span_id = 0
        self.stack = []
        self.base = [("component", component), ("tool", tool)]
        self.path = None
        self.status = None
        if enabled:
            if not self._valid_safe(kind):
                raise GoldenError(f"run kind {kind!r} is not a safe name")
            now = time.time_ns()
            self.run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now / 1e9)) + "-" + f"{secrets.randbits(32):08x}"
            self.dir = os.path.join(data_root(), "runs", self.run_id)
            os.makedirs(self.dir, exist_ok=False)
            self.manifest = {
                "schema": "lrm.run/1",
                "runId": self.run_id,
                "kind": kind,
                "title": title or f"{tool} (tools/goldens)",
                "status": "running",
                "createdUtc": self._iso(now),
                "endedUtc": None,
                "node": "n0",
                "synthetic": False,
                "gitSha": self._git_sha(),
                "executionClass": None,
                "driver": None,
                "params": dict(sorted({"tool": tool, **{k: str(v) for k, v in (params or {}).items()}}.items())),
            }
            self._write_manifest()
            self.path = os.path.join(self.dir, "events.n0.jsonl")
            self._f = open(self.path, "x", encoding="utf-8", newline="\n")
            self._line("event", "run.node", 0.0, "", {"role": "creator"}, t=now)
            import atexit

            atexit.register(self.close, "aborted")  # a run that stops on an error still leaves a readable log

    @staticmethod
    def _valid(name: str) -> bool:
        import re

        return bool(re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z0-9_]+)*", name)) and len(name) <= 128

    @staticmethod
    def _valid_safe(name: str) -> bool:
        import re

        return bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name or "")) and not name.endswith(".")

    @staticmethod
    def _iso(ns: int) -> str:
        ms = (ns // 1_000_000) % 1000
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ns // 1_000_000_000)) + f".{ms:03d}Z"

    @staticmethod
    def _git_sha():
        import subprocess

        try:
            r = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
            sha = r.stdout.strip()
            return sha if r.returncode == 0 and len(sha) == 40 else None
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _text(s: str) -> str:
        """A JSON string as the contract writes it: UTF-8, short escapes, other controls as lowercase \\u00xx, and
        U+2028 and U+2029 escaped."""
        return json.dumps(s, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    @staticmethod
    def _r(f: float) -> str:
        """The shortest round-trip text of a finite float, with an upper-case exponent as .NET's R format writes it."""
        return repr(float(f)).replace("e", "E")

    def _value(self, v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if math.isnan(v):
                return '{"$f":"NaN"}'
            if math.isinf(v):
                return '{"$f":"Infinity"}' if v > 0 else '{"$f":"-Infinity"}'
            return self._r(v)  # always holds '.' or 'E', so it reads back as a float
        return self._text(str(v))

    def _write_manifest(self):
        text = json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n"
        tmp = os.path.join(self.dir, "manifest.json.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, os.path.join(self.dir, "manifest.json"))

    def _line(self, kind, name, value, unit, attrs, duration=0, t=None, span=0, parent=0):
        if math.isnan(value):
            x = '"NaN"'
        elif math.isinf(value):
            x = '"Infinity"' if value > 0 else '"-Infinity"'
        else:
            x = self._r(value)
            x = x[:-2] if x.endswith(".0") else x
        a = {}
        if span:
            a["$span"] = span
        if parent:
            a["$parent"] = parent
        for k, v in attrs.items():
            a[k] = v  # a repeated key keeps its first position and takes its last value
        body = ",".join(f"{self._text(k)}:{self._value(v)}" for k, v in a.items())
        line = (f'{{"v":1,"t":{t if t is not None else time.time_ns()},"k":"{kind}","n":{self._text(name)},'
                f'"d":{int(duration)},"x":{x},"u":{self._text(unit)},"node":"n0","seq":{self.seq},"a":{{{body}}}}}')
        self._f.write(line + "\n")
        self._f.flush()
        self.seq += 1

    def _write(self, kind, name, value, unit, attrs, duration=0, t=None, span=0):
        if not self.enabled or self.status is not None:
            return
        if kind not in self.KINDS:
            raise GoldenError(f"telemetry kind {kind!r} is not valid")
        if not self._valid(name):
            raise GoldenError(f"telemetry name {name!r} is not valid")
        merged = {}
        for k, v in self.base + list(attrs.items()):
            if not self._valid(k):
                raise GoldenError(f"telemetry key {k!r} is not valid")
            merged[k] = v
        parent = self.stack[-1] if self.stack else 0
        self._line(kind, name, float(value), unit, merged, duration=duration, t=t, span=span, parent=parent)

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        self.span_id += 1
        sid = self.span_id
        parent = self.stack[-1] if self.stack else 0
        t0 = time.time_ns()
        c0 = time.perf_counter_ns()
        self.stack.append(sid)
        try:
            yield
        finally:
            self.stack.pop()
            d = time.perf_counter_ns() - c0
            if self.enabled and self.status is None:
                merged = dict(self.base)
                merged.update(attrs)
                self._line("span", name, float(d), "ns", merged, duration=d, t=t0, span=sid, parent=parent)

    def gauge(self, name, value, unit="", **attrs):
        self._write("gauge", name, float(value), unit, attrs)

    def counter(self, name, delta, unit="count", **attrs):
        self._write("counter", name, float(delta), unit, attrs)

    def check(self, name, passed, **attrs):
        self._write("check", name, 1.0 if passed else 0.0, "", attrs)

    def compare(self, passed: bool, metric: str, measured: float, bound: float | None, **attrs):
        """One golden.compare Check (ComparatorsAndGoldenReader): metric, measured and, when there is one, bound and
        its direction first, then the caller's keys. A measurement (an e_ref) has no bound."""
        extra = {"metric": metric, "measured": float(measured)}
        if bound is not None:
            extra["bound"] = float(bound)
            extra["direction"] = "le"
        self.check("golden.compare", passed, **extra, **attrs)

    def close(self, status: str = "ok"):
        """Writes the run.end trailer and the final manifest (status ok, failed or aborted). Later calls do nothing."""
        if not self.enabled or self.status is not None:
            return
        if status not in ("ok", "failed", "aborted"):
            raise GoldenError(f"run status {status!r} is not ok, failed or aborted")
        self.status = status
        now = time.time_ns()
        records = self.seq  # the trailer's own seq: every record before it, the header included
        self._line("event", "run.end", 0.0, "", {"status": status, "records": records, "dropped": 0, "invalid": 0}, t=now)
        self._f.close()
        self.manifest["status"] = status
        self.manifest["endedUtc"] = self._iso(now)
        self._write_manifest()
