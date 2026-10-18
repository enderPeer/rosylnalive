"""Independent known-answer vectors for the StagePipeline contract (docs/contracts/stage-pipeline.md).

Offline tool, standard library only (hashlib, struct). It re-implements, from the contract text alone, the canonical
encoding pieces, the job id, the stage args, the payloads and the chain digests, and prints the values the F# tests
check. Run it with any Python 3.10+ in isolated mode:

    python -I tests/Lrm.Serving.Tests/vectors/serving_vectors.py
"""

import hashlib
import struct


def sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def tagged(tag: str, payload: bytes) -> bytes:
    """Canonical.hash: SHA-256(ASCII(tag) || 0x00 || payload)."""
    return sha(tag.encode("ascii") + b"\x00" + payload)


def u8(v): return struct.pack("<B", v)
def u16(v): return struct.pack("<H", v)
def u32(v): return struct.pack("<I", v)
def i32(v): return struct.pack("<i", v)
def f32(v): return struct.pack("<f", v)
def boolean(v): return b"\x01" if v else b"\x00"
def bstr(b: bytes) -> bytes: return u32(len(b)) + b
def text(s: str) -> bytes: return bstr(s.encode("utf-8"))
def opt(v): return b"\x00" if v is None else b"\x01" + v
def seq(items): return u32(len(items)) + b"".join(items)


ZERO = bytes(32)
JOB_BYTES = bytes(range(1, 33))  # the bytes 01 02 ... 20


def blob_id(data: bytes) -> bytes:
    return tagged("lrm.blob/1", data)


def xclass_id(backend, device, driver, shaders, arch) -> bytes:
    return tagged("lrm.xclass/1", text(backend) + text(device) + text(driver) + opt(shaders) + text(arch))


# The fake executor's class (ClusterProtocol), a host-independent class for the vectors.
FAKE_CLASS = xclass_id("cpu-f32", "fake", "lrm.fake/1", None, "any")


def model_id(sums):
    """lrm.serving.model/1: the (name, SHA-256) pairs sorted by the name's UTF-8 bytes."""
    items = sorted(sums, key=lambda p: p[0].encode("utf-8"))
    return tagged("lrm.serving.model/1", seq([text(n) + h for n, h in items]))


def job_id(model, cls, layers, hidden, vocab, boundaries, prompts, steps, stop, eos):
    payload = (model + cls + u32(layers) + u32(hidden) + u32(vocab) + seq([u32(b) for b in boundaries])
               + seq([seq([i32(i) for i in p]) for p in prompts]) + u32(steps) + boolean(stop)
               + seq([i32(e) for e in eos]))
    return tagged("lrm.serving.job/1", payload)


def stage_args(model, layers, first, last, pos0, rows, head, dtype_code):
    return model + u32(layers) + u32(first) + u32(last) + u32(pos0) + u32(rows) + boolean(head) + u8(dtype_code)


def ids_payload(ids):
    return b"".join(i32(i) for i in ids)


def head_payload(token, logits):
    return u32(token) + b"".join(f32(x) for x in logits)


def unit_id(kind, kind_version, job, cls, lane, stage, step, prev, inp, args) -> bytes:
    payload = (text(kind) + u16(kind_version) + job + cls + u32(lane) + u32(stage) + u32(step) + prev + inp
               + bstr(args))
    return tagged("lrm.unit/1", payload)


def initial(job, lane, stage):
    return tagged("lrm.serving.init/1", job + u32(lane) + u32(stage))


def kv_rows(first, count, pos0, rows, dtype_code, per_layer):
    """per_layer: [(K row bytes, V row bytes)] for each layer in order."""
    payload = u32(first) + u32(count) + u32(pos0) + u32(rows) + u32(dtype_code)
    for k, v in per_layer:
        payload += k + v
    return tagged("lrm.serving.kvrows/1", payload)


def state_after(digest, step, input_id, output_id, kv):
    return tagged("lrm.serving.state/1", digest + u32(step) + input_id + output_id + kv)


def logits_digest(lanes):
    return tagged("lrm.serving.logits/1", seq([u32(i) + seq([bstr(o) for o in outs]) for i, outs in enumerate(lanes)]))


def main():
    sums = [("model.safetensors", bytes([0x11]) * 32), ("REVISION", bytes([0x22]) * 32),
            ("config.json", bytes([0x33]) * 32)]
    print("M1 model id         ", model_id(sums).hex())

    job = job_id(JOB_BYTES, FAKE_CLASS, 2, 64, 256, [1], [[1, 2, 3], [4]], 4, False, [7])
    print("J1 job id           ", job.hex())
    job_eos = job_id(JOB_BYTES, FAKE_CLASS, 2, 64, 256, [1], [[1, 2, 3], [4]], 4, True, [7])
    print("J2 job id (EOS)     ", job_eos.hex())

    a1 = stage_args(JOB_BYTES, 2, 0, 0, 0, 3, False, 1)
    print("A1 args (stage 0)   ", a1.hex(), len(a1))
    a2 = stage_args(JOB_BYTES, 2, 1, 1, 3, 1, True, 1)
    print("A2 args (stage 1)   ", a2.hex(), len(a2))

    p1 = ids_payload([1, -1])
    print("P1 ids [1, -1]      ", p1.hex())
    p2 = head_payload(7, [0.5])
    print("P2 head 7, [0.5]    ", p2.hex())

    u1 = unit_id("fwd.stage", 1, job, FAKE_CLASS, 0, 0, 0, ZERO, blob_id(ids_payload([1, 2, 3])), a1)
    print("U1 unit id          ", u1.hex())

    d0 = initial(job, 1, 1)
    print("D0 initial (1, 1)   ", d0.hex())
    kv = kv_rows(1, 1, 2, 1, 1, [(f32(1.5) + f32(-2.0), f32(0.25) + f32(8.0))])
    print("K1 kv rows          ", kv.hex())
    d1 = state_after(d0, 3, blob_id(b"abc"), blob_id(b""), kv)
    print("D1 state after      ", d1.hex())

    lg = logits_digest([[head_payload(5, [1.0, 2.0])], []])
    print("L1 logits digest    ", lg.hex())


if __name__ == "__main__":
    main()
