#!/usr/bin/env python3
"""edge_monitor.py — rolling edge monitor (two-tier) for the autotrader.

Answers one question every time a trade closes or a signal is graded:
  "Is my recent realized edge statistically gone — am I losing money now?"

Tier 1 — signal-quality (fast, no trade sample needed): tracks the CLEAR-RATE
  (fraction of graded signals with proba >= CLEAR_FLOOR) over a trailing window.
  A collapsing clear-rate is the grader itself flagging a low-edge market BEFORE
  losses accumulate (August 2026: 6% clear-rate vs a normal ~30%).

Tier 2 — realized-edge (slow, hard evidence): rolling window of the last N
  closed trades' R-multiples, bootstrapped (10000 draws, fixed seed) to test
  whether the recent edge is statistically gone (mean R < 0).

State machine: normal -> watch -> halt -> (fresh window recovers) -> resume.

ADVISORY-FIRST: AUTOTRADE_EDGE_MONITOR=advisory (default) logs + exposes state
  but does NOT gate entries. =enforce will gate (wired in supervisor.py). =off
  disables. Never panic-closes open positions — it only pulls NEW exposure.

Point-in-time only: uses completed trades + already-graded signals, never future
data, and NEVER uses the 68% good-month backtest as its expectation.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections import deque
from typing import Optional

import numpy as np

STATE_FILE = os.path.join(os.path.expanduser("~"), ".autotrade_edge_monitor.json")

MODE = os.environ.get("AUTOTRADE_EDGE_MONITOR", "advisory").lower()  # off|advisory|enforce

# ── Tier 2 (realized edge) tunables ────────────────────────────────────────
WINDOW = int(os.environ.get("EDGE_WINDOW", "15"))               # trades in window
BOOT_DRAWS = int(os.environ.get("EDGE_BOOT_DRAWS", "10000"))
BOOT_SEED = int(os.environ.get("EDGE_BOOT_SEED", "42"))
B_HEALTHY = float(os.environ.get("EDGE_B_HEALTHY", "0.50"))     # de-risk line (mean R)
WATCH_P = float(os.environ.get("EDGE_WATCH_P", "0.70"))         # P(meanR < B_healthy) > this -> watch
HALT_P = float(os.environ.get("EDGE_HALT_P", "0.90"))           # P(meanR < 0) > this -> halt
HALT_WR = float(os.environ.get("EDGE_HALT_WR", "0.30"))         # WR below this AND meanR<0 -> halt
COOLDOWN_H = float(os.environ.get("EDGE_COOLDOWN_H", "24"))     # halt duration before auto-resume

# ── Tier 1 (signal clear-rate) tunables ────────────────────────────────────
CLEAR_FLOOR = float(os.environ.get("EDGE_CLEAR_FLOOR", "0.35"))       # fixed floor (not evolve floor)
CLEAR_DAYS = float(os.environ.get("EDGE_CLEAR_DAYS", "3.0"))          # trailing signal window (days)
CLEAR_MIN = int(os.environ.get("EDGE_CLEAR_MIN", "20"))               # min signals to judge
CLEAR_BASELINE = float(os.environ.get("EDGE_CLEAR_BASELINE", "0.30"))  # normal clear-rate (measured)
CLEAR_WATCH_FRAC = float(os.environ.get("EDGE_CLEAR_WATCH", "0.50"))   # watch if < frac of baseline


def _epoch(ts: Optional[str]) -> float:
    """ISO timestamp -> unix seconds (wall clock if None)."""
    if not ts:
        return time.time()
    try:
        return dt.datetime.fromisoformat(ts).timestamp()
    except Exception:
        return time.time()


def bootstrap_p_lt(r_values, threshold: float, draws: int = BOOT_DRAWS,
                   seed: int = BOOT_SEED) -> float:
    """One-sided bootstrap: P(bootstrap mean R < threshold) over `r_values`."""
    r = np.asarray(list(r_values), dtype=float)
    n = r.size
    if n < 2:
        return 0.5
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(draws, n), replace=True).mean(axis=1)
    return float((means < threshold).mean())


class EdgeMonitor:
    def __init__(self, state_file: Optional[str] = STATE_FILE, quiet: bool = False):
        self.state_file = state_file if (state_file and not quiet) else None
        self.quiet = quiet
        self.state = "normal"                     # normal | watch | halt
        self.trades = deque(maxlen=200)           # window-relevant trades (cleared on halt)
        self._total = 0                           # lifetime trade count (for status)
        self.signals = deque(maxlen=2000)         # (proba, ts) history for Tier 1
        self.halted_at = None                     # ISO ts of the last halt
        self.last_change = 0.0
        self.clear_rate = None                    # latest Tier 1 measurement
        self.p_lt0 = None                         # latest P(meanR < 0)
        self.p_lt_healthy = None                  # latest P(meanR < B_healthy)
        self.window_mean_r = None
        self.window_wr = None
        self.window_n = 0
        self._load()

    # ── persistence ────────────────────────────────────────────────────────
    def _load(self):
        if not self.state_file:
            return
        try:
            with open(self.state_file) as f:
                d = json.load(f)
            self.state = d.get("state", "normal")
            self.trades = deque(d.get("trades", []), maxlen=200)
            self._total = int(d.get("total", len(self.trades)))
            self.signals = deque([(s["proba"], s["ts"]) for s in d.get("signals", [])],
                                 maxlen=2000)
            self.halted_at = d.get("halted_at")
            self.last_change = float(d.get("last_change", 0.0))
            self.clear_rate = d.get("clear_rate")
            self.p_lt0 = d.get("p_lt0")
            self.p_lt_healthy = d.get("p_lt_healthy")
            self.window_mean_r = d.get("window_mean_r")
            self.window_wr = d.get("window_wr")
            self.window_n = int(d.get("window_n", 0))
            if self.state not in ("normal", "watch", "halt"):
                self.state = "normal"
        except Exception:
            pass

    def _save(self):
        if not self.state_file:
            return
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "state": self.state, "halted_at": self.halted_at,
                "last_change": self.last_change, "total": self._total,
                "trades": [{"r": t["r"], "ts": t.get("ts", "")}
                           for t in list(self.trades)[-100:]],
                "signals": [{"proba": p, "ts": ts} for p, ts in list(self.signals)[-1000:]],
                "clear_rate": self.clear_rate,
                "p_lt0": self.p_lt0, "p_lt_healthy": self.p_lt_healthy,
                "window_mean_r": self.window_mean_r, "window_wr": self.window_wr,
                "window_n": self.window_n,
            }, f)
        os.replace(tmp, self.state_file)

    def _log(self, text: str):
        if not self.quiet:
            print(f"🛡 EDGE-MONITOR: {text}", flush=True)

    # ── Tier 1: signal clear-rate ─────────────────────────────────────────
    def record_signal(self, proba: float, ts: Optional[str] = None):
        """Called for every graded signal (before floor/ceiling filtering)."""
        if MODE == "off":
            return
        ts = ts or dt.datetime.now(dt.timezone.utc).isoformat()
        self.signals.append((float(proba), ts))
        self._refresh_clear_rate(ts)
        self._save()

    def _refresh_clear_rate(self, now_ts: str):
        if not self.signals:
            self.clear_rate = None
            return
        cutoff = _epoch(now_ts) - CLEAR_DAYS * 86400
        recent = [p for p, ts in self.signals if _epoch(ts) >= cutoff]
        if len(recent) < CLEAR_MIN:
            self.clear_rate = None          # not enough signal data yet
            return
        self.clear_rate = sum(1 for p in recent if p >= CLEAR_FLOOR) / len(recent)

    # ── Tier 2: realized edge ─────────────────────────────────────────────
    def record(self, trade: dict):
        """Called on every closed trade. trade: {r, strategy, side, ts, ...}."""
        if MODE == "off":
            return
        ts = trade.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat()
        self._total += 1
        self.trades.append({"r": float(trade["r"]), "ts": ts})
        self._evaluate(ts)
        self._save()

    def _window(self) -> list:
        """Trades to judge: trailing WINDOW (post-halt resume uses a fresh slice)."""
        return list(self.trades)[-WINDOW:]

    def _evaluate(self, now_ts: str):
        now = _epoch(now_ts)

        # HALT -> auto-resume after cooldown. A sticky halt (wait for a fresh
        # window to recover) is wrong twice: (1) in enforce mode no trades close
        # while halted, so a fresh window can never form -> permanent deadlock;
        # (2) it blocks the recovery leg of a V-shaped month (Aug 2026 lost the
        # first week, then recovered — a sticky halt would lock in the dip and
        # miss +7R of recovery). So: pause COOLDOWN_H, then resume to TEST.
        if self.state == "halt":
            if self.halted_at and now - _epoch(self.halted_at) < COOLDOWN_H * 3600:
                return                             # still in cooldown — stay halted
            self._set_state("normal", "cooldown elapsed — resuming to test recovery",
                            now_ts)
            # fall through: the fresh window (cleared on halt) is judged normally;
            # if it is still losing, the normal branch re-halts.

        win = self._window()
        n = len(win)
        if n == 0:
            self.window_n = 0
            return
        rs = [t["r"] for t in win]
        wr = sum(1 for r in rs if r > 0) / n
        avg = sum(rs) / n
        self.window_mean_r = round(avg, 3)
        self.window_wr = round(wr, 3)
        self.window_n = n
        if n < WINDOW:
            return                                   # warm-up: show stats, don't judge
        self.p_lt0 = bootstrap_p_lt(rs, 0.0)
        self.p_lt_healthy = bootstrap_p_lt(rs, B_HEALTHY)

        losing = (self.p_lt0 > HALT_P) or (wr < HALT_WR and avg < 0)
        if losing:
            self._set_state("halt", f"realized edge gone — {n} trades WR {wr:.0%} "
                            f"avg {avg:+.2f}R, P(meanR<0)={self.p_lt0:.2f}", now_ts)
        elif self.p_lt_healthy > WATCH_P:
            self._set_state("watch", f"edge degrading — {n} trades WR {wr:.0%} "
                            f"avg {avg:+.2f}R, P(meanR<{B_HEALTHY:.1f})={self.p_lt_healthy:.2f}",
                            now_ts)

        # Tier 1 can only escalate to WATCH (never halt) — the grader's own
        # low-confidence signal, logged before losses materialize.
        if self.state == "normal" and self.clear_rate is not None:
            if self.clear_rate < CLEAR_WATCH_FRAC * CLEAR_BASELINE:
                self._set_state("watch", f"signal clear-rate collapsed to "
                                f"{self.clear_rate:.0%} (baseline {CLEAR_BASELINE:.0%})", now_ts)

    def _set_state(self, new: str, why: str, now_ts: Optional[str] = None):
        if new == self.state:
            return
        if new == "halt":
            self.halted_at = now_ts or dt.datetime.now(dt.timezone.utc).isoformat()
            self.trades.clear()          # reset the window — fresh window starts now
        if new == "normal":
            self.halted_at = None
        self.state = new
        self.last_change = _epoch(now_ts)
        self._log(f"{self.state.upper()}: {why}  [mode={MODE}]")

    # ── read side ─────────────────────────────────────────────────────────
    def blocks_entries(self) -> bool:
        """True when the gate should actually stop new entries (enforce only)."""
        return MODE == "enforce" and self.state == "halt"

    def tick(self, now_ts: Optional[str] = None):
        """Called each bar by the supervisor: auto-resume after the cooldown
        elapses (needed in enforce mode, where no trades close while halted so
        record() never fires to drive the transition)."""
        if MODE == "off":
            return
        now_ts = now_ts or dt.datetime.now(dt.timezone.utc).isoformat()
        if self.state == "halt" and self.halted_at:
            if _epoch(now_ts) - _epoch(self.halted_at) >= COOLDOWN_H * 3600:
                self._set_state("normal", "cooldown elapsed — resuming to test recovery",
                                now_ts)
                self._save()

    def is_blocked_at(self, ts: Optional[str] = None) -> bool:
        """Would the gate block a NEW entry at time ts? (enforce semantics —
        used by the validation replay and the future enforce wiring.)"""
        if MODE == "off" or self.state != "halt" or not self.halted_at:
            return False
        ts = ts or dt.datetime.now(dt.timezone.utc).isoformat()
        return _epoch(ts) - _epoch(self.halted_at) < COOLDOWN_H * 3600

    def status(self) -> dict:
        return {
            "mode": MODE,
            "state": self.state,
            "blocks_entries": self.blocks_entries(),
            "window_n": self.window_n,
            "window_wr": self.window_wr,
            "window_mean_r": self.window_mean_r,
            "p_meanr_lt_0": self.p_lt0,
            "p_meanr_lt_healthy": self.p_lt_healthy,
            "clear_rate": self.clear_rate,
            "halted_at": self.halted_at,
            "total_trades": self._total,
        }


if __name__ == "__main__":
    # self-test: a losing streak should trip halt; a winning streak should resume.
    e = EdgeMonitor(state_file=None, quiet=True)
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(18):
        ts = (t0 + dt.timedelta(minutes=i)).isoformat()
        e.record({"r": -1.0, "strategy": "ema", "side": "SHORT", "ts": ts})
    print("after 18 losses:", {k: v for k, v in e.status().items()
                               if k in ("state", "window_n", "window_mean_r", "p_meanr_lt_0")})
    assert e.state == "halt", f"expected halt, got {e.state}"
    # recover: winning trades AFTER the halt, past the cooldown
    t1 = t0 + dt.timedelta(days=2)
    for i in range(8):
        ts = (t1 + dt.timedelta(minutes=i)).isoformat()
        e.record({"r": 2.0, "strategy": "ema", "side": "LONG", "ts": ts})
    print("after recovery:", {k: v for k, v in e.status().items()
                              if k in ("state", "window_n", "window_mean_r")})
    print("self-test OK")
