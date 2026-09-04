#!/usr/bin/env python3
"""embed_worker.py — long-lived Chronos embedding worker (model loaded ONCE).

A persistent subprocess that keeps the Chronos backbone in memory and answers
embedding requests, so callers pay the ~45 MB model load a single time per
session instead of once per request. Torch lives only here — isolated from the
xgboost in the parent (they segfault together on macOS).

Wire protocol on the binary channel (repeated until stdin EOF):
    parent → worker   [8-byte big-endian length][ .npy of float32 (N, CTX) ]   log-close windows
    worker → parent   [8-byte big-endian length][ .npy of float32 (N, 256) ]   mean-pooled embeddings

Produces identical results to futures_foundation.foundation.embed_bars
(same pipe.embed(...).mean(1)).
"""
import io
import os
import struct
import sys

import numpy as np

_BATCH = 64


def _read_array(stream):
    hdr = _read_exact(stream, 8)
    if hdr is None:
        return None
    (n,) = struct.unpack(">Q", hdr)
    body = _read_exact(stream, n)
    if body is None:
        return None
    return np.load(io.BytesIO(body))


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _write_array(stream, arr):
    bio = io.BytesIO()
    np.save(bio, np.asarray(arr, dtype=np.float32))
    data = bio.getvalue()
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()


def main():
    # Keep a PRIVATE binary channel on the real stdout, then redirect fd 1 →
    # stderr so any library chatter can't corrupt the protocol.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    inb = sys.stdin.buffer

    import torch
    from chronos import BaseChronosPipeline
    try:
        from futures_foundation import foundation
        src = foundation.active_source()
    except Exception:
        src = os.environ.get("CHRONOS_FT_CKPT") or "amazon/chronos-bolt-tiny"
    print(f"[embed_worker] loading Chronos once: {src}", file=sys.stderr, flush=True)
    pipe = BaseChronosPipeline.from_pretrained(src, device_map="cpu",
                                               dtype=torch.float32)
    print("[embed_worker] ready", file=sys.stderr, flush=True)

    while True:
        windows = _read_array(inb)
        if windows is None:                       # stdin closed → shut down
            break
        X = np.asarray(windows, dtype=np.float32)
        chunks = []
        with torch.no_grad():
            for s in range(0, len(X), _BATCH):
                emb, _ = pipe.embed(torch.tensor(X[s:s + _BATCH]))
                chunks.append(emb.mean(1).cpu().numpy())
        _write_array(out, np.concatenate(chunks).astype(np.float32))


if __name__ == "__main__":
    main()
