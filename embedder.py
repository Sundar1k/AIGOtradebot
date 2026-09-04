#!/usr/bin/env python3
"""embedder.py — client for the warm Chronos embedding worker.

Drop-in for futures_foundation.foundation.embed_bars, but routed through a
persistent worker (embed_worker.py) that loads the model once per session. Builds
the log-close context windows here (cheap numpy) and ships them to the worker.
Falls back to the one-shot library path if the worker can't start.

The worker is a fresh subprocess (spawned, not forked), so it never inherits the
parent's xgboost — torch stays isolated.
"""
import atexit
import io
import os
import struct
import subprocess
import sys
import threading

import numpy as np

from config import CTX

D_MODEL = 256
_HERE = os.path.dirname(os.path.abspath(__file__))
_proc = None
_lock = threading.Lock()


def _start():
    global _proc
    _proc = subprocess.Popen(
        [sys.executable, os.path.join(_HERE, "embed_worker.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)   # stderr → inherits parent's
    atexit.register(shutdown)


def shutdown():
    global _proc
    if _proc and _proc.poll() is None:
        try:
            _proc.stdin.close()
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise RuntimeError("embed worker closed the pipe")
        buf.extend(chunk)
    return bytes(buf)


def _roundtrip(windows):
    bio = io.BytesIO()
    np.save(bio, windows.astype(np.float32))
    data = bio.getvalue()
    _proc.stdin.write(struct.pack(">Q", len(data)))
    _proc.stdin.write(data)
    _proc.stdin.flush()
    (n,) = struct.unpack(">Q", _read_exact(_proc.stdout, 8))
    return np.load(io.BytesIO(_read_exact(_proc.stdout, n)))


def embed_bars(close, indices, ctx: int = CTX):
    """Mean-pooled Chronos embeddings (len(indices), 256) for the log-close
    windows ending at each index — same as foundation.embed_bars, warm."""
    c = np.asarray(close, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros((0, D_MODEL), np.float32)
    lp = np.log(c)
    windows = np.stack([lp[i - ctx + 1:i + 1] for i in idx]).astype(np.float32)
    with _lock:
        try:
            if _proc is None or _proc.poll() is not None:
                _start()
            return _roundtrip(windows)
        except Exception as e:                      # worker trouble → one-shot path
            from futures_foundation import foundation
            sys.stderr.write(f"[embedder] warm worker failed ({e}); "
                             f"falling back to embed_bars\n")
            shutdown()
            return foundation.embed_bars(close, indices, ctx=ctx)
