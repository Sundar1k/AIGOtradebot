#!/usr/bin/env python3
"""veto_server.py — GPU sidecar for the fine-tuned 7B veto layer.

The autotrade supervisor runs with CUDA_VISIBLE_DEVICES= (CPU-pinned so it
never fights GPU training), so the fine-tuned model CANNOT load in-process.
This service loads the 4-bit base + LoRA adapter ONCE (lazy singleton, ~3-4
min on a GTX 1070) and serves veto decisions over localhost HTTP:

    POST /decide  {"text": "ES 3m. RSI 45, ... Score +2."}
      -> {"action": "BUY|SELL|NO TRADE", "reason": "...", "ok": true}
    GET  /health  -> {"loaded": bool, "vram_mib": n}

Runs under systemd user unit veto.service with the GPU venv
(finrl-x-venv, HF_HUB_OFFLINE=1). Supervisor's veto_fn POSTs here and
fail-closed: if this service is down, no entries (watchdog alerts).
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import llm_veto  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
_model_loaded = False
_model_lock = time.time()
# torch model.generate is NOT thread-safe: ThreadingHTTPServer can hand two
# requests to two threads and a concurrent forward pass corrupts the model /
# segfaults the server (observed: RemoteDisconnected + connection refused
# under backtest load). ALL inference must serialize through this lock.
_infer_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # silence per-request noise
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                    # client gave up (timeout) — no traceback spam

    def do_GET(self):
        if self.path == "/health":
            import llm_veto
            try:
                stats = llm_veto.cache_stats()
            except Exception:
                stats = {}
            self._json(200, {"loaded": _model_loaded, **stats})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global _model_loaded
        if self.path not in ("/decide", "/score", "/decide_batch", "/score_batch",
                             "/score_batch_v2"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            t0 = time.time()
            with _infer_lock:                  # serialize all model inference
                if self.path == "/decide_batch":
                    texts = [str(t) for t in payload.get("texts", []) if str(t).strip()]
                    if not texts:
                        self._json(400, {"error": "empty texts"})
                        return
                    # v2 (2026-08-21): TRUE GPU batching — one generate()
                    # call instead of the serialized loop (~2-3x faster).
                    out = llm_veto.decide_batch(texts)
                    _model_loaded = True
                    d = {"results": out, "n": len(out)}
                elif self.path == "/score_batch":
                    texts = [str(t) for t in payload.get("texts", []) if str(t).strip()]
                    if not texts:
                        self._json(400, {"error": "empty texts"})
                        return
                    quals = [llm_veto.quality(t) for t in texts]
                    d = {"qualities": quals, "n": len(quals)}
                elif self.path == "/score_batch_v2":
                    # TRUE GPU batching (one generate per batch) — same scores
                    # as /score_batch, ~10-20x throughput. Selection-validator
                    # dataset scoring (2026-08-30); live veto path untouched.
                    texts = [str(t) for t in payload.get("texts", []) if str(t).strip()]
                    if not texts:
                        self._json(400, {"error": "empty texts"})
                        return
                    quals = llm_veto.quality_batch(texts)
                    d = {"qualities": quals, "n": len(quals)}
                elif self.path == "/score":
                    text = str(payload.get("text", ""))
                    if not text.strip():
                        self._json(400, {"error": "empty text"})
                        return
                    q = llm_veto.quality(text)
                    d = {"quality": q}
                else:
                    text = str(payload.get("text", ""))
                    if not text.strip():
                        self._json(400, {"error": "empty text"})
                        return
                    d = llm_veto.decide(text)
                    _model_loaded = True
            d["infer_ms"] = int((time.time() - t0) * 1000)
            self._json(200, d)
        except Exception as e:
            # CUDA OOM resilience (2026-08-23): an OOM used to kill the whole
            # server (8 crash-loops on 2026-08-23 17:04). Recover instead:
            # drop the fragmented CUDA cache and keep serving; the next call
            # re-attempts and torch re-allocates on the freed blocks.
            msg = str(e)
            if "OutOfMemoryError" in msg or "CUDA out of memory" in msg:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self._json(500, {"error": msg})


def main():
    print(f"[veto_server] loading model (~3 min on GTX 1070)…", flush=True)
    t0 = time.time()
    llm_veto._load()          # preload so the first RTH bar is instant
    global _model_loaded
    _model_loaded = True
    print(f"[veto_server] model ready in {time.time()-t0:.0f}s — "
          f"listening http://{HOST}:{PORT}", flush=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
