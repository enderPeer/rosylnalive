"""Independent known-answer vectors for the ClusterProtocol contract (docs/contracts/cluster.md).

Offline tool, standard library only (hashlib, struct). It re-implements, from the contract text alone, the parts of
the canonical encoding, the frame codec, WorkUnit ids and the fake pipeline that the vectors need, and prints the
values the F# tests check. Run it with any Python 3.10+ in isolated mode:

    python -I tests/Lrm.Cluster.Tests/vectors/cluster_vectors.py
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
def u64(v): return struct.pack("<Q", v)
def bstr(b: bytes) -> bytes: return u32(len(b)) + b
def text(s: str) -> bytes: return bstr(s.encode("utf-8"))
def opt(v): return b"\x00" if v is None else b"\x01" + v
def seq(items): return u32(len(items)) + b"".join(items)


ZERO = bytes(32)


def blob_id(data: bytes) -> bytes:
    return tagged("lrm.blob/1", data)


def xclass_id(backend, device, driver, shaders, arch) -> bytes:
    return tagged("lrm.xclass/1", text(backend) + text(device) + text(driver) + opt(shaders) + text(arch))


FAKE_CLASS = xclass_id("cpu-f32", "fake", "lrm.fake/1", None, "any")


def unit_payload(kind, kind_version, job, cls, lane, stage, step, prev, inp, args) -> bytes:
    return (text(kind) + u16(kind_version) + job + cls + u32(lane) + u32(stage) + u32(step) + prev + inp
            + bstr(args))


def unit_id(*fields) -> bytes:
    return tagged("lrm.unit/1", unit_payload(*fields))


def frame(kind, flags, epoch, sender, seqno, unit, payload) -> bytes:
    header = (b"LRM1" + u16(1) + u16(kind) + u32(flags) + u32(0) + u64(epoch) + u64(sender) + u64(seqno)
              + u32(len(payload)) + u32(0) + unit)
    assert len(header) == 80
    body = header + payload
    return body + sha(body)


# ---- fake pipeline -------------------------------------------------------------------------------

def expand(seed: bytes, length: int) -> bytes:
    out = b""
    block = 0
    while len(out) < length:
        out += tagged("lrm.fake.expand/1", seed + u32(block))
        block += 1
    return out[:length]


def fake_job_id(seed, lanes, steps, stages, prompt_bytes, hidden_bytes) -> bytes:
    return tagged("lrm.fake.job/1", u64(seed) + u32(lanes) + u32(steps) + u32(stages) + u32(prompt_bytes)
                  + u32(hidden_bytes))


def fake_reference(seed, lanes, steps, stages, prompt_bytes, hidden_bytes):
    job = fake_job_id(seed, lanes, steps, stages, prompt_bytes, hidden_bytes)
    digest = {}
    head = {}
    accepted = []
    tokens = [[] for _ in range(lanes)]
    for lane in range(lanes):
        data = expand(tagged("lrm.fake.prompt/1", job + u32(lane)), prompt_bytes)
        for step in range(steps):
            for stage in range(stages):
                key = (lane, stage)
                if key not in digest:
                    digest[key] = tagged("lrm.fake.init/1", job + u32(lane) + u32(stage))
                    head[key] = ZERO
                out_len = 4 if stage == stages - 1 else hidden_bytes
                args = u32(out_len)
                inp = blob_id(data)
                uid = unit_id("fwd.stage", 1, job, FAKE_CLASS, lane, stage, step, head[key], inp, args)
                h = tagged("lrm.fake.out/1", job + u32(lane) + u32(stage) + u32(step) + digest[key] + bstr(data))
                out = expand(h, out_len)
                digest[key] = tagged("lrm.fake.state/1", digest[key] + u32(step) + inp + blob_id(out))
                head[key] = uid
                accepted.append((lane, step, stage, uid, blob_id(out), digest[key]))
                data = out
            tokens[lane].append(struct.unpack("<I", data)[0])
    accepted.sort(key=lambda a: (a[0], a[1], a[2]))
    run = tagged("lrm.cluster.run/1", job + seq([u32(l) + u32(t) + u32(s) + i + o + d
                                                 for (l, t, s, i, o, d) in accepted]))
    chains = sorted(head.keys())
    replicas = tagged("lrm.cluster.replicas/1", seq([u32(l) + u32(s) + head[(l, s)] + digest[(l, s)]
                                                     for (l, s) in chains]))
    return job, run, replicas, tokens, len(accepted)


def main():
    print("fake class id      ", FAKE_CLASS.hex())

    # Frame F1: Lease from the coordinator, epoch 7, seq 42, DurationMs 1000, LeaseEcho 41.
    f1_payload = u32(1000) + u64(41)
    f1 = frame(4, 0, 7, 0, 42, ZERO, f1_payload)
    print("F1 payload         ", f1_payload.hex())
    print("F1 frame           ", f1.hex())
    print("F1 trailer         ", f1[-32:].hex())

    # Units U1 (step 0) and U2 (step 1) of one chain.
    job = bytes(range(1, 33))
    in1 = blob_id(b"abc")
    in2 = blob_id(b"")
    args = u32(4)
    u1 = ("fwd.stage", 1, job, FAKE_CLASS, 1, 2, 0, ZERO, in1, args)
    u1_id = unit_id(*u1)
    u2 = ("fwd.stage", 1, job, FAKE_CLASS, 1, 2, 1, u1_id, in2, args)
    u2_id = unit_id(*u2)
    print("U1 payload         ", unit_payload(*u1).hex())
    print("U1 id              ", u1_id.hex())
    print("U2 payload         ", unit_payload(*u2).hex())
    print("U2 id              ", u2_id.hex())

    # Frame F2: Assign of U1 with input "abc", epoch 3, seq 9, Retransmit flag.
    f2_payload = unit_payload(*u1) + bstr(b"abc")
    f2 = frame(6, 1, 3, 0, 9, u1_id, f2_payload)
    print("F2 frame           ", f2.hex())
    print("F2 trailer         ", f2[-32:].hex())

    for name, params in [("small", (1, 2, 3, 3, 16, 64)), ("standard", (1, 3, 8, 3, 64, 256))]:
        job, run, replicas, tokens, n = fake_reference(*params)
        print(f"fake {name:8} job    ", job.hex())
        print(f"fake {name:8} run    ", run.hex())
        print(f"fake {name:8} repl   ", replicas.hex())
        print(f"fake {name:8} units  ", n)
        print(f"fake {name:8} tokens ", tokens)


if __name__ == "__main__":
    main()
