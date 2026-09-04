#!/usr/bin/env python3
"""ai_context.py — thin client for the AI-context sidecar (ai_worker.py).

The trading bot's venv never loads YOLO/TTM — it asks the sidecar (a
separate user service, finrl-x-venv, CUDA_VISIBLE_DEVICES=0) over HTTP.
Advisory only: any failure returns "" and the veto sees the trained format
exactly as before.

Endpoint: GET http://127.0.0.1:8767/context?symbol=NQ → {"line": " ..."}
"""
import os
import urllib.request
import urllib.parse
import json

WORKER_URL = os.environ.get("AUTOTRADE_AI_URL", "http://127.0.0.1:8767/context")
TIMEOUT = 5.0


def ai_context_line(bars, symbol: str) -> str:
    """Advisory line for the veto context. Returns "" on any failure."""
    try:
        url = f"{WORKER_URL}?symbol={urllib.parse.quote(symbol)}"
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
        return data.get("line", "")
    except Exception:
        return ""


if __name__ == "__main__":
    import sys
    print("AI context:", repr(ai_context_line(None, "NQ")))
