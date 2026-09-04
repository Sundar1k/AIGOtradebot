#!/usr/bin/env python3
"""ai_worker.py — YOLO + TTM advisory context sidecar for the algo bot.

Holds YOLO (ft2 best.pt) + TTM (granite-timeseries-ttm-r2) in memory and
answers context requests over localhost HTTP, so the trading bot's venv never
needs ultralytics/tsfm and the models load ONCE per session. Mirrors
veto.service (separate user service, finrl-x-venv, CUDA_VISIBLE_DEVICES=0)
and veto_server.py (stdlib http.server — no new deps).

Endpoints:
  GET /context?symbol=NQ  → {"line": " YOLO: ... . TTM 5h fc +0.02%."}
  GET /health             → {"ok": true, "models": true}

The worker reads its own history (data/*_3min.csv + /tmp/live_loop_data
extension) — the bot's 500-bar window is far too short for TTM's 512-bar
context. Point-in-time: forecast = next 20 bars AFTER the last available bar.

Advisory only — never gates entries. Fail-open: any error → {"line": ""}.
"""
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "granite-tsfm"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "yolo-patterns"))

from ultralytics import YOLO
from tsfm_public.toolkit.get_model import get_model

DATA_DIR = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data")
LIVE_DIR = "/tmp/live_loop_data"
TTM = "ibm-granite/granite-timeseries-ttm-r2"
CTX = 512
PRED = 20

YOLO_MODEL = None
TTM_MODEL = None


def load_models():
    global YOLO_MODEL, TTM_MODEL
    if YOLO_MODEL is None:
        print("[ai_worker] loading YOLO...", flush=True)
        YOLO_MODEL = YOLO(os.path.join(os.path.expanduser("~"), "yolo-patterns/runs/ft2/weights/best.pt"))
    if TTM_MODEL is None:
        print("[ai_worker] loading TTM...", flush=True)
        TTM_MODEL = get_model(
            TTM, context_length=CTX, prediction_length=PRED,
            freq_prefix_tuning=False, freq=None,
            prefer_l1_loss=False, prefer_longer_context=True)
        TTM_MODEL.eval()
    print("[ai_worker] models ready", flush=True)


def full_history(symbol):
    """5-year data file extended with live bars; 15-min OHLC dataframe."""
    hist = pd.read_csv(f"{DATA_DIR}/{symbol}_3min.csv")
    hist = hist.rename(columns={"datetime": "time"})
    hist["time"] = pd.to_datetime(hist["time"], utc=True)
    live_path = f"{LIVE_DIR}/{symbol}.csv"
    if os.path.exists(live_path):
        live = pd.read_csv(live_path)
        live["time"] = pd.to_datetime(live["time"], utc=True)
        cols = [c for c in ("time", "open", "high", "low", "close")
                if c in live.columns]
        hist = pd.concat([hist[cols], live[cols]]
                         ).drop_duplicates("time").sort_values("time")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    r15 = hist.set_index("time")[["open", "high", "low", "close"]].resample(
        "15min").agg(agg).dropna()
    return r15


def build_context(symbol):
    """Returns the advisory line for symbol. Raises on failure (fail-open)."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gen_dataset import render

    r15 = full_history(symbol)

    # --- YOLO: render last 200 15-min bars in bot chart style, detect ---
    w = r15.tail(200)
    png = f"/tmp/ai_ctx_{symbol}.png"
    fig, ax = render(w["open"].to_numpy(float), w["high"].to_numpy(float),
                     w["low"].to_numpy(float), w["close"].to_numpy(float), png)
    plt.close(fig)
    res = YOLO_MODEL.predict([png], device=0, conf=0.25, batch=1,
                             verbose=False)[0]
    pat, conf = "", 0.0
    if res.boxes is not None and len(res.boxes):
        pat = YOLO_MODEL.names[int(res.boxes.cls[0].item())]
        conf = float(res.boxes.conf[0].item())

    # --- TTM: zero-shot 20-bar forecast on last 512 15-min closes ---
    fc = None
    if len(r15) >= CTX:
        x = torch.tensor(r15["close"].to_numpy(float)[-CTX:],
                         dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            out = TTM_MODEL(past_values=x, freq_token=torch.ones(1),
                            return_loss=False, return_dict=True)
        pred = out.prediction_outputs[0, :, 0].numpy()
        fc = float(pred[-1] / float(r15["close"].iloc[-1]) - 1)

    parts = []
    if pat:
        parts.append(f"YOLO: {pat} ({conf:.2f})")
    if fc is not None:
        parts.append(f"TTM 5h fc {fc:+.2%}")
    return (" " + ". ".join(parts) + ".") if parts else ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            self._send({"ok": True,
                        "models": YOLO_MODEL is not None and TTM_MODEL is not None})
            return
        if u.path == "/context":
            symbol = parse_qs(u.query).get("symbol", ["NQ"])[0]
            try:
                line = build_context(symbol)
                self._send({"line": line, "symbol": symbol})
            except Exception as e:
                print(f"[ai_worker] error {symbol}: {e}", flush=True)
                self._send({"line": "", "symbol": symbol, "error": str(e)})
            return
        self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    load_models()
    print("[ai_worker] serving on :8767", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8767), Handler).serve_forever()
