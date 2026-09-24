#!/usr/bin/env python3
"""Rewrite a GLM-5.3-Flash Q4_K GGUF in place: BF16 KDA q/k/v and the output
head become Q8_0.

Why: glm53_quantize.py's q4 recipe already specifies Q8_0 for the
`linear_attention` and `output` roles, but the GGUF published at
antirez/glm-5.3-flash-gguf predates that recipe and ships them as BF16. Those
tensors are ~54%% of the bytes read per decoded token, and ds4's
kda_inputs_q8_bf16 fast path requires kda_q/k/v to be Q8_0, so on the published
file that path cannot engage.

In place is safe: every converted tensor shrinks (BF16 2 bytes/weight -> Q8_0
1.0625), nothing else moves, and the GGUF header keeps its byte length because
the tensor type is a fixed-width uint32. Walking front to back, the write
cursor always trails the read cursor. The file is truncated at the end.

Deliberately NOT converted, because ds4 fast paths require BF16:
  blk.*.kda_output                      HC-expand epilogue
  kda_f_a / f_b / g_a / g_b / beta      KDA gate trio
  token_embd                            row lookup; no per-token bandwidth

Usage:
    python3 glm53_kda_q8_inplace.py MODEL.gguf --dry-run    # inspect the plan
    python3 glm53_kda_q8_inplace.py MODEL.gguf              # rewrite in place

Then run ds4-server with DS4_GLM_ENABLE_KDA_Q8_INPUTS=1.

DESTRUCTIVE and non-resumable. A crash mid-write leaves an unusable file and
you re-download. There is no undo.
"""
import argparse, os, struct, sys, time
import numpy as np

BF16, Q8_0 = 30, 8
# ggml type -> (name, bytes per block, elements per block)
TYPE = {0:('F32',4,1), 1:('F16',2,1), 8:('Q8_0',34,32), 10:('Q2_K',84,256),
        12:('Q4_K',144,256), 14:('Q6_K',210,256), 30:('BF16',2,1)}
CHUNK_BLOCKS = 2 * 1024 * 1024


def nbytes(ty, n):
    _, blk, per = TYPE[ty]
    if n % per:
        raise ValueError(f"{n} elements not divisible by {per}")
    return n // per * blk


def should_convert(name, ty):
    if ty != BF16:
        return False
    if name == "output.weight":
        return True
    if not name.startswith("blk."):
        return False
    return name.rsplit('.', 1)[0].split('.')[-1] in ("kda_q", "kda_k", "kda_v")


def parse(path):
    f = open(path, 'rb')
    if f.read(4) != b'GGUF':
        sys.exit("not a GGUF file")
    struct.unpack('<I', f.read(4))
    nt, = struct.unpack('<Q', f.read(8))
    nkv, = struct.unpack('<Q', f.read(8))

    def rstr():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n)

    def skip(t):
        if t == 8: rstr()
        elif t in (0, 1, 7): f.read(1)
        elif t in (2, 3, 4, 5, 6): f.read(4)
        elif t in (10, 11, 12): f.read(8)
        elif t == 9:
            at, = struct.unpack('<I', f.read(4))
            n, = struct.unpack('<Q', f.read(8))
            for _ in range(n): skip(at)
        else:
            raise ValueError(f"unknown gguf value type {t}")

    align = 32
    for _ in range(nkv):
        k = rstr()
        t, = struct.unpack('<I', f.read(4))
        if k == b'general.alignment':
            align, = struct.unpack('<I', f.read(4))
        else:
            skip(t)

    infos = []
    for _ in range(nt):
        name = rstr()
        nd, = struct.unpack('<I', f.read(4))
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        ty_off = f.tell()
        ty, = struct.unpack('<I', f.read(4))
        off, = struct.unpack('<Q', f.read(8))
        n = 1
        for d in dims: n *= d
        infos.append(dict(name=name.decode(), ty=ty, off=off, n=n,
                          ty_off=ty_off, off_off=ty_off + 4))
    hdr_end = f.tell()
    f.close()
    return dict(nt=nt, align=align, infos=infos,
                data_start=(hdr_end + align - 1) // align * align)


def plan(path):
    g = parse(path)
    a = g['align']
    cur = 0
    out = []
    for t in g['infos']:
        conv = should_convert(t['name'], t['ty'])
        new_ty = Q8_0 if conv else t['ty']
        old_b = nbytes(t['ty'], t['n'])
        new_b = nbytes(new_ty, t['n'])
        cur = (cur + a - 1) // a * a
        out.append(dict(t, conv=conv, new_ty=new_ty, new_off=cur,
                        old_b=old_b, new_b=new_b))
        cur += new_b
    return g, out, cur


def q8_0(x):
    """Byte-identical to ds4q_quantize_q8_0 in gguf-tools/quants.c."""
    x = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    y = x * inv[:, None]
    # C roundf() is round-half-away-from-zero; np.round is banker's rounding.
    q = np.trunc(y + np.copysign(np.float32(0.5), y)).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g, p, data_bytes = plan(args.gguf)
    ds = g['data_start']
    new_size = ds + data_bytes
    cur_size = os.path.getsize(args.gguf)
    conv = [t for t in p if t['conv']]
    GiB = 1 << 30

    worst = min(t['off'] - t['new_off'] for t in p)
    print(f"  file          {cur_size/GiB:.2f} GiB, {g['nt']} tensors")
    print(f"  converting    {len(conv)} tensors -> Q8_0")
    groups = {}
    for t in conv:
        k = (t['name'].rsplit('.', 1)[0].split('.')[-1]
             if t['name'].startswith('blk.') else t['name'])
        a = groups.setdefault(k, [0, 0, 0])
        a[0] += 1; a[1] += t['old_b']; a[2] += t['new_b']
    for k, (c, o, n) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"    {k:16} n={c:<4} {o/GiB:7.2f} -> {n/GiB:7.2f} GiB")
    print(f"  new size      {new_size/GiB:.2f} GiB (saves {(cur_size-new_size)/GiB:.2f} GiB)")
    print(f"  safety        min(read-write) = {worst} bytes" +
          ("  OK" if worst >= 0 else "  UNSAFE"))
    if worst < 0:
        sys.exit("refusing: writes would overtake reads")
    if args.dry_run:
        print("  dry run, nothing written")
        return
    if not conv:
        print("  nothing to convert; already Q8_0?")
        return

    fd = os.open(args.gguf, os.O_RDWR)
    t0 = time.time(); done = 0
    total = sum(t['old_b'] for t in p)
    try:
        for i, t in enumerate(p):
            r, w = ds + t['off'], ds + t['new_off']
            if t['conv']:
                blocks = t['n'] // 32
                for b0 in range(0, blocks, CHUNK_BLOCKS):
                    nb = min(CHUNK_BLOCKS, blocks - b0)
                    raw = os.pread(fd, nb * 32 * 2, r + b0 * 32 * 2)
                    os.pwrite(fd, q8_0(bf16_to_f32(raw)).tobytes(), w + b0 * 34)
            elif w != r:
                rem, o = t['old_b'], 0
                while rem:
                    c = min(rem, 256 << 20)
                    os.pwrite(fd, os.pread(fd, c, r + o), w + o)
                    o += c; rem -= c
            done += t['old_b']
            if i % 100 == 0 or i == len(p) - 1:
                el = time.time() - t0
                print(f"  {i+1}/{len(p)}  {done/GiB:7.1f}/{total/GiB:.1f} GiB  "
                      f"{done/GiB/max(el,0.01):5.2f} GiB/s", flush=True)
        for t in p:
            os.pwrite(fd, struct.pack('<I', t['new_ty']), t['ty_off'])
            os.pwrite(fd, struct.pack('<Q', t['new_off']), t['off_off'])
        os.ftruncate(fd, new_size)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"  done in {time.time()-t0:.0f}s -> {os.path.getsize(args.gguf)/GiB:.2f} GiB")


if __name__ == "__main__":
    main()
