#!/usr/bin/env python3
"""evolve.py — self-evolution engine for the autotrader (SAFE version).

The bot learns from its OWN closed trades and adapts its entry strictness:

  * every closed trade is recorded (R multiple, strategy, side, time)
  * a rolling window (last N trades) tracks win rate + avg R
  * when recent performance is BAD  -> floor tightens (fewer, higher-conviction
    entries) up to a hard cap
  * when recent performance is GOOD -> floor eases back toward the baseline,
    never below it
  * stance changes are persisted (~/.autotrade_evolve.json, restart-safe) and
    alerted to Telegram

GUARANTEES (why this is safe for a funded eval):
  - only touches the NEXT entry's confidence floor — never an open position,
    never the strategy rules, never the breakers ($400/$1500 stay absolute)
  - hard bounds: floor in [baseline, baseline + 0.30]
  - the veto layer is untouched: even a "loosened" bot still needs LLM approval
  - a restart can't silently loosen: stance is read back from disk
"""
import datetime as dt
import json
import os
import time
from collections import deque

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(os.path.expanduser("~"), ".autotrade_evolve.json")
TG_CHAT = "5882586287"
TG_ENV = os.path.join(os.path.expanduser("~"), "hermes-restore/hermes-home/.env")

# Tunables (env-overridable)
WINDOW = int(os.environ.get("EVOLVE_WINDOW", "30"))       # trades in the window
                                                          # 20 -> 30 (2026-08-19): 20
                                                          # trades on stdR~1.4 is a
                                                          # random walk — adaptation was
                                                          # chasing noise.
BAD_WINRATE = float(os.environ.get("EVOLVE_BAD_WINRATE", "0.40"))
GOOD_WINRATE = float(os.environ.get("EVOLVE_GOOD_WINRATE", "0.55"))
FLOOR_STEP = float(os.environ.get("EVOLVE_FLOOR_STEP", "0.05"))   # per adjustment
FLOOR_CAP = float(os.environ.get("EVOLVE_FLOOR_CAP", "0.30"))     # max raise above baseline
COOLDOWN_MIN = int(os.environ.get("EVOLVE_COOLDOWN_MIN", "60"))   # min between stance changes
# Statistical-honesty gate (2026-08-19): only move the floor when the window's
# avg R is beyond ~1 standard error from zero (stdR ~1.4 per trade typical for
# this book). A -0.3R average on 30 trades is noise; -0.5R+ is signal. This
# stops the evolver from tightening (or easing) on random streaks — the exact
# failure that pushed the floor to 0.43 on 9 contaminated trades.
EVOLVE_STD_R = float(os.environ.get("EVOLVE_STD_R", "1.4"))      # per-trade R std
EVOLVE_MIN_SE = float(os.environ.get("EVOLVE_MIN_SE", "1.0"))    # |avgR| must exceed N SEs
EVOLVE_MIN_BAND = float(os.environ.get("EVOLVE_MIN_BAND", "0.08"))  # floor never within
                                                                    # this of the ceiling
                                                                    # (band would be too
                                                                    # narrow to trade)
# Confidence-ceiling adaptation (learned from misses 2026-08-17): the model's
# overconfident band (proba >= ~0.50) has negative expectancy in recent
# regimes. CEIL_BASE is the default ceiling (config.PROBA_CEIL), CEIL_FLOOR
# the lowest we'll ever take (0.40 — below that we'd rather not trade than
# chase), CEIL_HIGH the loose bound (1.0 = effectively disabled).
CEIL_BASE = float(os.environ.get("EVOLVE_CEIL_BASE", "0.50"))
CEIL_FLOOR = float(os.environ.get("EVOLVE_CEIL_FLOOR", "0.40"))
CEIL_HIGH = float(os.environ.get("EVOLVE_CEIL_HIGH", "1.0"))
CEIL_BAND = float(os.environ.get("EVOLVE_CEIL_BAND", "0.50"))   # the "high" band
CEIL_BAD_WR = float(os.environ.get("EVOLVE_CEIL_BAD_WR", "0.40"))  # band WR below this -> tighten
CEIL_GOOD_WR = float(os.environ.get("EVOLVE_CEIL_GOOD_WR", "0.55"))  # band WR above this -> ease
CEIL_MIN_TRADES = int(os.environ.get("EVOLVE_CEIL_MIN_TRADES", "8"))  # min band trades to judge


def _tg_token() -> str:
    try:
        for line in open(TG_ENV):
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _alert(text: str):
    import requests
    token = _tg_token()
    if not token:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": TG_CHAT, "text": text,
                            "disable_notification": False}, timeout=15)
    except Exception:
        pass


def _min_avg_r(n: int) -> float:
    """The |avgR| threshold below which a window's performance is noise.

    SE = EVOLVE_STD_R / sqrt(n); a window average within ±EVOLVE_MIN_SE*SE
    of zero is statistically indistinguishable from a coin flip at this
    book's variance and MUST NOT move the floor."""
    if n <= 0:
        return 0.0
    return EVOLVE_MIN_SE * EVOLVE_STD_R / (n ** 0.5)


class Evolver:
    def __init__(self, baseline_floor: float, state_file: str = STATE_FILE):
        self.baseline = baseline_floor
        self.state_file = state_file
        self.trades = deque(maxlen=200)          # full history (bounded)
        self.floor = baseline_floor
        self.ceil = CEIL_BASE                    # confidence ceiling
        self.stance = "normal"
        self.last_change = 0.0
        self.last_ceil_change = 0.0              # separate cooldown for the ceiling
        self._load()

    # ---- persistence -------------------------------------------------
    def _load(self):
        try:
            with open(self.state_file) as f:
                d = json.load(f)
            self.trades = deque(d.get("trades", []), maxlen=200)
            self.floor = float(d.get("floor", self.baseline))
            self.ceil = float(d.get("ceil", CEIL_BASE))
            self.stance = d.get("stance", "normal")
            self.last_change = float(d.get("last_change", 0.0))
            self.last_ceil_change = float(d.get("last_ceil_change", 0.0))
            # never allow a corrupt/over-loose floor to persist
            max_floor = max(self.baseline,
                            min(self.baseline + FLOOR_CAP,
                                self.ceil - EVOLVE_MIN_BAND))
            if not (self.baseline <= self.floor <= max_floor + 1e-9):
                self.floor = max_floor
                self.stance = "normal"
            # ceiling sanity: keep inside [CEIL_FLOOR, CEIL_HIGH]
            if not (CEIL_FLOOR <= self.ceil <= CEIL_HIGH + 1e-9):
                self.ceil = CEIL_BASE
        except Exception:
            pass

    def _save(self):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"trades": list(self.trades)[-100:], "floor": self.floor,
                       "ceil": self.ceil, "stance": self.stance,
                       "last_change": self.last_change,
                       "last_ceil_change": self.last_ceil_change}, f)
        os.replace(tmp, self.state_file)

    # ---- trade recording --------------------------------------------
    def record(self, trade: dict):
        """trade: {r, strategy, side, entry, exit, ts} — called on every close."""
        self.trades.append(trade)
        self._save()
        self._maybe_adapt()

    # ---- adaptation ---------------------------------------------------
    def _maybe_adapt(self):
        if len(self.trades) < 5:                 # need a little data first
            return
        recent = list(self.trades)[-WINDOW:]
        wins = sum(1 for t in recent if t["r"] > 0)
        wr = wins / len(recent)
        avg_r = sum(t["r"] for t in recent) / len(recent)
        need = _min_avg_r(len(recent))   # noise floor: |avgR| below this = no move
        now = time.time()

        # Floor adaptation (its OWN cooldown — the ceiling branch below must
        # not be starved when the floor just moved).
        if now - self.last_change >= COOLDOWN_MIN * 60:
            if wr < BAD_WINRATE and avg_r < -need:
                new_floor = min(self.floor + FLOOR_STEP,
                                self.baseline + FLOOR_CAP,
                                self.ceil - EVOLVE_MIN_BAND)
                if new_floor > self.floor + 1e-9:
                    self.floor = new_floor
                    self.stance = "cautious" if new_floor < self.baseline + FLOOR_CAP else "tight"
                    self.last_change = now
                    self._save()
                    _alert(f"🧠 EVOLVE: tightening — last {len(recent)} trades "
                           f"WR {wr:.0%}, avg {avg_r:+.2f}R (beyond ±{need:.2f}R "
                           f"noise floor). Entry floor {new_floor:.2f}. Veto stays ON.")
            elif wr > GOOD_WINRATE and avg_r > need:
                new_floor = max(self.floor - FLOOR_STEP, self.baseline)
                if new_floor < self.floor - 1e-9:
                    self.floor = new_floor
                    self.stance = "normal" if new_floor <= self.baseline + 1e-9 else "cautious"
                    self.last_change = now
                    self._save()
                    _alert(f"🧠 EVOLVE: easing — last {len(recent)} trades "
                           f"WR {wr:.0%}, avg {avg_r:+.2f}R. Entry floor "
                           f"{new_floor:.2f}.")

        # Confidence-ceiling learning (from misses): watch the HIGH-proba band
        # (proba >= CEIL_BAND). If it keeps losing, tighten the ceiling — the
        # model's overconfident signals are the ones that lose. If it recovers
        # (regime like 2021 where high proba won), ease back toward CEIL_HIGH.
        # Uses its OWN cooldown so the floor's adaptation doesn't starve it.
        band = [t for t in recent if t.get("proba", 0.0) >= CEIL_BAND]
        if len(band) >= CEIL_MIN_TRADES and now - self.last_ceil_change >= COOLDOWN_MIN * 60:
            band_wins = sum(1 for t in band if t["r"] > 0)
            band_wr = band_wins / len(band)
            if band_wr < CEIL_BAD_WR:
                new_ceil = max(self.ceil - FLOOR_STEP, CEIL_FLOOR)
                if new_ceil < self.ceil - 1e-9:
                    self.ceil = new_ceil
                    self.last_ceil_change = now
                    self._save()
                    _alert(f"🎯 EVOLVE CEIL: high-proba band (≥{CEIL_BAND:.2f}) "
                           f"WR {band_wr:.0%} ({len(band)} trades) — confidence "
                           f"ceiling {new_ceil:.2f} (overconfident signals losing)")
            elif band_wr > CEIL_GOOD_WR:
                new_ceil = min(self.ceil + FLOOR_STEP, CEIL_HIGH)
                if new_ceil > self.ceil + 1e-9:
                    self.ceil = new_ceil
                    self.last_ceil_change = now
                    self._save()
                    _alert(f"🎯 EVOLVE CEIL: high-proba band (≥{CEIL_BAND:.2f}) "
                           f"WR {band_wr:.0%} ({len(band)} trades) — ceiling "
                           f"eased to {new_ceil:.2f}")

    # ---- read side -----------------------------------------------------
    def current_floor(self) -> float:
        return self.floor

    def winrate_by_quality(self, min_q: int = 1) -> dict:
        """Empirical win rate of closed trades grouped by quality score band.

        Returns {band: {"n": n, "wins": w, "winrate": r}} for bands
        (q>=8), (6<=q<=7), (4<=q<=5), (q<=3) plus overall. This is the
        honest "70% win rate" measurement: we only claim a score band is
        good once its real outcomes prove it.
        """
        bands = {"8-10": [], "6-7": [], "4-5": [], "1-3": []}
        for t in self.trades:
            q = int(t.get("quality", 0))
            if q >= 8:
                bands["8-10"].append(t)
            elif q >= 6:
                bands["6-7"].append(t)
            elif q >= 4:
                bands["4-5"].append(t)
            elif q >= 1:
                bands["1-3"].append(t)
        out = {}
        for k, v in bands.items():
            n = len(v)
            if n:
                wins = sum(1 for t in v if t["r"] > 0)
                out[k] = {"n": n, "wins": wins,
                          "winrate": round(wins / n, 3)}
        return out

    def status(self) -> dict:
        recent = list(self.trades)[-WINDOW:] if self.trades else []
        wins = sum(1 for t in recent if t["r"] > 0) if recent else 0
        band = [t for t in recent if t.get("proba", 0.0) >= CEIL_BAND]
        band_wr = None
        if len(band) >= 3:
            band_wr = round(sum(1 for t in band if t["r"] > 0) / len(band), 3)
        return {"floor": round(self.floor, 3),
                "baseline": round(self.baseline, 3),
                "ceil": round(self.ceil, 3),
                "stance": self.stance,
                "window_trades": len(recent),
                "window_winrate": round(wins / len(recent), 3) if recent else None,
                "high_band_trades": len(band),
                "high_band_winrate": band_wr,
                "total_trades": len(self.trades),
                "by_quality": self.winrate_by_quality()}


if __name__ == "__main__":
    # quick self-test: simulate a losing streak then a winning streak
    e = Evolver(0.35, state_file="/tmp/evolve_test.json")
    import shutil
    shutil.rmtree("/tmp/evolve_test.json", ignore_errors=True)
    for i in range(10):
        e.record({"r": -1.0, "strategy": "ema", "side": "LONG",
                  "entry": 100, "exit": 99, "ts": dt.datetime.now().isoformat()})
    print("after 10 losses:", e.status())
    for i in range(10):
        e.record({"r": 1.5, "strategy": "ema", "side": "LONG",
                  "entry": 100, "exit": 101, "ts": dt.datetime.now().isoformat()})
    print("after 10 wins:", e.status())
