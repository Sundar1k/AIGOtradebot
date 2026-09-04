#!/usr/bin/env python3
"""supervisor.py — always-on autonomous wrapper for the algoTraderBot live loop.

Runs the same per-bar logic as bot.py (BotContext + handle_bar) but adds the
guardrails a self-trading agent needs:

  * RTH gate        — entries only 09:30-15:55 America/New_York, weekdays
  * Circuit breaker — real-account daily loss / trailing drawdown limits;
                      on breach: close everything, halt until the next ET day,
                      alert Telegram
  * Heartbeat       — JSON state file for the watchdog (freshness + halt state)
  * Restart-safe    — start_balance persists across restarts so a restart can't
                      bypass the daily-loss breaker
  * --once          — smoke-test mode: one immediate iteration, no bar wait

Env overrides: AUTOTRADE_DAILY_LOSS (default 400), AUTOTRADE_TRAILING_DD (1500).
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
# signals.py/telegram.py are now local (migrated from topstep-bot 2026-08-21)

import requests
import config
from broker import make_broker
from bot import BotContext, handle_bar
import evolve
import edge_monitor
from concurrent.futures import ThreadPoolExecutor

ET = ZoneInfo("America/New_York")
DAILY_LOSS_LIMIT = float(os.environ.get("AUTOTRADE_DAILY_LOSS", "400"))
TRAILING_DD_LIMIT = float(os.environ.get("AUTOTRADE_TRAILING_DD", "1500"))
# Topstep consistency rule (payout eligibility): best day < 40% of total
# profit. Breaching it doesn't suspend the account but blocks payouts, so we
# alert approaching and halt entries for the day on breach (protects the
# payout path). AUTOTRADE_CONSISTENCY_HALT=0 disables the halt (alert only).
CONSISTENCY_LIMIT = float(os.environ.get("AUTOTRADE_CONSISTENCY_LIMIT", "0.40"))
CONSISTENCY_WARN = float(os.environ.get("AUTOTRADE_CONSISTENCY_WARN", "0.35"))
CONSISTENCY_HALT = os.environ.get("AUTOTRADE_CONSISTENCY_HALT", "1") == "1"
PROFIT_TARGET = float(os.environ.get("AUTOTRADE_PROFIT_TARGET", "500"))
PROFIT_BAND = float(os.environ.get("AUTOTRADE_PROFIT_BAND", "150"))
PROFIT_WINDOW_H = float(os.environ.get("AUTOTRADE_PROFIT_WINDOW_H", "25"))
VETO_URL = os.environ.get("AUTOTRADE_VETO_URL", "http://127.0.0.1:8765/decide")
VETO_FAIL_OPEN = os.environ.get("AUTOTRADE_VETO_FAIL_OPEN", "0") == "1"
# VETO_MODE: "gate" (LLM action hard-gates entries — original) | "advisory"
# (LLM action is logged + reflected but NEVER blocks an entry).
# Demoted to advisory 2026-08-22 after a blind-OOS audit (NQ W-window) showed the
# veto blocked 13 winners vs 3 losers — net-negative on selection.
# Code default = advisory (2026-08-30, constitution VI): a manual run or a lost
# service env must never silently re-enable the proven-negative gate. The
# service env may still force "gate" explicitly if ever wanted again.
VETO_MODE = os.environ.get("AUTOTRADE_VETO_MODE", "advisory")
# ENTRY WINDOW (2026-08-31, spec-kit cycle 3 — GO): new entries only
# 09:30-12:00 ET (America/New_York wall time, DST-correct). Position
# MANAGEMENT continues 24h. Verdict: OOS +0.455R/PF 1.93 vs +0.027/1.03
# all-day, P=1.0 (specs/time-window/verdict.md). Env-gated so it is
# reversible: AUTOTRADE_ENTRY_WINDOW=1 enables (set in the service env per
# user approval 2026-08-31); anything else = no gate (pre-wiring behavior).
ENTRY_WINDOW = os.environ.get("AUTOTRADE_ENTRY_WINDOW", "0") == "1"
ENTRY_WIN_OPEN = dt.time(9, 30)
ENTRY_WIN_CLOSE = dt.time(12, 0)


def entry_gate(ts=None) -> bool:
    """True = new entries allowed. Management is NEVER gated (handle_bar
    applies this only to the detect/enter path). Fail-open by construction:
    any error returns True (never blocks trading). `ts` is the signal-bar
    timestamp (used by backtests); live ignores it and uses wall clock."""
    if not ENTRY_WINDOW:
        return True
    try:
        if ts is None:
            now = now_et()
        else:
            now = pd.Timestamp(ts).tz_convert(ET)
        return ENTRY_WIN_OPEN <= now.time() < ENTRY_WIN_CLOSE
    except Exception:
        return True
STATE_FILE = os.path.join(os.path.expanduser("~"), ".autotrade_state")
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(15, 55)
# Trading-hours policy: AUTOTRADE_HOURS=rth (weekday US day session only) or
# all (every CME futures session: Sun 18:00 ET → Fri 17:00 ET, minus the
# daily 17:00-18:00 ET maintenance break). User chose "all" 2026-08-17.
HOURS_MODE = os.environ.get("AUTOTRADE_HOURS", "all").lower()
TG_CHAT = "5882586287"
TG_ENV = os.path.join(os.path.expanduser("~"), "hermes-restore/hermes-home/.env")


def _tg_token() -> str:
    try:
        for line in open(TG_ENV):
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def alert(text: str):
    token = _tg_token()
    if not token:
        print("[supervisor] no telegram token — alert skipped", flush=True)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": TG_CHAT, "text": text,
                            "disable_notification": False}, timeout=15)
    except Exception as e:
        print(f"[supervisor] tg alert failed: {e}", flush=True)


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def in_rth(now: dt.datetime) -> bool:
    """Is the market tradeable RIGHT NOW?

    HOURS_MODE=all: CME futures sessions — Sunday 18:00 ET through Friday
    17:00 ET, excluding the daily 17:00-18:00 ET maintenance break.
    HOURS_MODE=rth: legacy weekday 09:30-15:55 ET gate.
    """
    if HOURS_MODE == "rth":
        return now.weekday() < 5 and RTH_OPEN <= now.time() < RTH_CLOSE

    wd = now.weekday()
    t = now.time()
    if wd == 6:                     # Sunday: opens 18:00 ET
        return t >= dt.time(18, 0)
    if wd == 4:                     # Friday: closes 17:00 ET
        return t < dt.time(17, 0)
    if wd < 4:                      # Mon-Thu: 24h minus 17:00-18:00 break
        return not (dt.time(17, 0) <= t < dt.time(18, 0))
    return False                    # Saturday: closed


def build_state_line(bars, symbol: str) -> str:
    """The veto's market-state text — EXACTLY the v1 training format.
    Shared by veto_fn (live signals) and the prefetch thread (cache warm).
    bars: broker df with time/open/high/low/close/volume columns."""
    import signals as botsignals
    df = bars.rename(columns={"time": "datetime"})
    sc = botsignals.compute_scores(df)
    row = sc.iloc[-1]
    ema_f = df["close"].ewm(span=10, adjust=False).mean().iloc[-1]
    ema_s = df["close"].ewm(span=30, adjust=False).mean().iloc[-1]
    ema_dir = "above" if ema_f > ema_s else "below"
    kk, dd = botsignals.stochastic(df)
    stoch = float(kk.iloc[-1])
    stoch_trend = "rising" if kk.iloc[-1] >= dd.iloc[-1] else "falling"
    score = int(row["score"])
    sym = config.base_symbol(symbol)
    line = (f"{sym} 3m. RSI {row['rsi']:.0f}, EMA10 {ema_dir} EMA30, "
            f"stochastic {stoch:.0f} {stoch_trend}, "
            f"ATR {row['atr']:.0f}. Score {score:+.0f}.")
    # Advisory AI context (YOLO pattern + TTM 5h forecast) — appended ONLY
    # when enabled. Never gates entries; never alters the trained prefix.
    if os.environ.get("AUTOTRADE_AI_CONTEXT", "0") == "1":
        try:
            import ai_context
            line += ai_context.ai_context_line(bars, symbol)
        except Exception:
            pass   # advisory only — any failure keeps the trained format
    return line


def make_veto_fn():
    """Build the LLM confirm/veto filter for BotContext.veto_fn.

    Returns fn(s, sig, bars) -> (ok: bool, why: str). Builds the market-state
    text in the EXACT language the 7B model was fine-tuned on
    (finetune/gen_dataset.py: "SYM 3m. RSI x, EMA10 above/below EMA30,
    stochastic y rising/falling, ATR z. Score s."), POSTs it to the GPU
    sidecar (veto_server.py, localhost:8765), and agrees only when the
    model's action matches the engine's side.

    Two gates:
      1. EVENT BLACKOUT — no entries inside high-impact macro windows (FOMC,
         CPI, NFP…). Deterministic, never relies on the LLM.
      2. LLM quality gate — the fine-tuned model rates the setup 1-10 in a
         SEPARATE short call (decide() prompt must stay in-distribution);
         entries below AUTOTRADE_QUALITY_MIN are blocked.

    Fail-closed: if the sidecar is unreachable, entries are BLOCKED
    (AUTOTRADE_VETO_FAIL_OPEN=1 flips that — trade without the veto).
    """
    import signals as botsignals
    import market_context
    import regime

    quality_min = int(os.environ.get("AUTOTRADE_QUALITY_MIN", "6"))
    # Regime gate (Lo 2004 AMH): trade only in the regimes the model was
    # trained in. strict = block in panic/high-vol (patterns don't transfer);
    # off = trade any regime (pre-gate behavior).
    regime_gate = os.environ.get("AUTOTRADE_REGIME_GATE", "strict")
    blocked_regimes = set(os.environ.get(
        "AUTOTRADE_REGIME_BLOCK", "panic").split(","))

    def _regime_block():
        """Return (blocked: bool, why: str) from the regime state file."""
        if regime_gate == "off":
            return False, ""
        try:
            with open(regime.STATE_FILE) as f:
                st = json.load(f)
            if st.get("regime") in blocked_regimes and st.get("gate") == "on":
                return True, (f"REGIME GATE — market {st['regime']} "
                              f"(prob {st.get('prob')}, vol "
                              f"{st.get('ann_vol_pct')}%) — patterns don't "
                              f"transfer; idling")
        except Exception:
            pass
        return False, ""

    def veto_fn(s, sig, bars):
        try:
            # HARD news gate: no entries inside a high-impact event window
            # (FOMC, CPI, NFP…). Deterministic — never relies on the LLM.
            ev = market_context.event_blackout()
            if ev is not None:
                return False, (f"EVENT BLACKOUT — {ev['title']} at "
                               f"{ev['time'].strftime('%H:%M')} ET")

            # REGIME GATE: if the HMM says panic/high-vol, block entries.
            blk, why = _regime_block()
            if blk:
                return False, why

            # CHOP GATE (2026-08-23): ATR14/ATR100 vol-expansion ratio.
            # Walk-forward validated on 281 ledger trades (66% vs 58% WR,
            # +21% avgR). Deterministic, point-in-time, fail-open on errors.
            try:
                import chop_gate
                cblk, cwhy = chop_gate.should_block(bars)
                if cblk:
                    return False, f"CHOP GATE — {cwhy}"
            except Exception as e:
                print(f"⚠️ chop gate failed: {e}", flush=True)

            # CANDLE-CONFLICT GATE (#6): if enabled by the attribution agent
            # (validated rule), block signals that oppose the 30-min candle
            # pattern. Ledger evidence (2026-08-17): conflict signals lose
            # ~2.2x more often than aligned ones.
            if os.environ.get("AUTOTRADE_BLOCK_CONFLICT", "0") == "1":
                try:
                    import candle_patterns
                    pats = candle_patterns.pattern_at_time(
                        bars, bars["time"].iloc[-1])
                    pdir = candle_patterns.pattern_direction(pats)
                    if pdir != 0 and pdir != sig.direction:
                        return False, (f"CANDLE CONFLICT — pattern "
                                       f"{'+'.join(pats)} opposes "
                                       f"{'LONG' if sig.direction > 0 else 'SHORT'}")
                except Exception as e:
                    print(f"⚠️ conflict gate failed: {e}", flush=True)

            # Same features + functions the dataset was mined with (signals.py).
            # bars columns: time/open/high/low/close/volume (broker.get_bars).
            # NOTE (2026-08-17): v1 basic features only — the rich-feature
            # quality model (volume/hour/EMA50/range) failed its holdout eval
            # and was reverted. Keep this state line byte-identical to v1's
            # training format.
            state = build_state_line(bars, config.SYMBOL)

            # FAST VETO (distilled student) — reject-only pre-filter. If the
            # XGBoost student is confident the 7B would say NO TRADE, block now
            # and skip the GPU call. Never approves on its own (the 7B still
            # confirms every entry). Env-gated AUTOTRADE_FAST_VETO=1; on any
            # error it falls through to the 7B (fail-open to the authority).
            if os.environ.get("AUTOTRADE_FAST_VETO", "0") == "1":
                try:
                    import fast_veto
                    fr = fast_veto.decide_fast(state)
                    if fr.get("confident_reject"):
                        return False, (f"FAST VETO (student) — P(NO TRADE)="
                                       f"{fr['p_no_trade']:.2f}, skipping GPU")
                except Exception as e:
                    print(f"⚠️ fast veto failed (falling through to 7B): {e}",
                          flush=True)

            r = requests.post(VETO_URL, json={"text": state}, timeout=120)
            r.raise_for_status()
            d = r.json()
            action = d.get("action", "NO TRADE")
            want = "BUY" if sig.direction > 0 else "SELL"
            ok = action == want
            why = f"{action} | {d.get('reason', '')} | {d.get('infer_ms', 0)}ms"

            # LIVE DISTILL CAPTURE — append (state -> teacher action) so the
            # student can be retrained on the TRUE production signal
            # distribution (not just historical bars). Env-gated
            # AUTOTRADE_VETO_CAPTURE=1; append-only, never touches the decision.
            if os.environ.get("AUTOTRADE_VETO_CAPTURE", "0") == "1":
                try:
                    with open(os.path.join(os.path.expanduser("~"), ".autotrade_veto_labels.jsonl"), "a") as cf:
                        cf.write(json.dumps({"state": state, "action": action,
                                             "want": want, "ok": ok,
                                             "t": int(time.time())}) + "\n")
                except Exception as e:
                    print(f"⚠️ veto capture failed: {e}", flush=True)


            # AGREEMENT REPORT (#4 — the honest multi-voice debate): count how
            # many independent signals agree with the entry. The veto is ONE
            # voice (single prompt, deterministic — a literal debate would be
            # theater). The other voices: candle pattern direction, score sign.
            # Logged for the ledger/reflection; never gates by itself.
            try:
                import candle_patterns
                pats = candle_patterns.pattern_at_time(
                    bars, bars["time"].iloc[-1])
                pdir = candle_patterns.pattern_direction(pats)
                agree = 1 if ok else 0                    # veto voice
                if pdir != 0:
                    agree += 1 if pdir == sig.direction else 0
                s._veto_agreement = agree
                s._veto_patterns = pats
                why += f" | voices {agree}/2 (veto+pattern)"
            except Exception as e:
                print(f"⚠️ agreement calc failed: {e}", flush=True)

            # Quality gate — the v2 model was trained to emit action + quality
            # together, but the live adapter is v1 (never emits quality), so
            # fall back to the SEPARATE /score endpoint (short prompt, keeps the
            # decide prompt in its training distribution). Stored on ctx for the
            # evolution engine (win rate by score band). AUTOTRADE_QUALITY_MIN
            # enforces; 0 = collect-only until the score's predictive power is
            # validated (honest-eval-first, 2026-08-19).
            s._veto_reason = d.get("reason", "")       # reflection memory (#5)
            if ok:
                q = int(d.get("quality", 0) or 0)
                if q == 0:
                    try:
                        rq = requests.post(
                            VETO_URL.rsplit("/", 1)[0] + "/score",
                            json={"text": state}, timeout=60)
                        q = int((rq.json() or {}).get("quality", 0) or 0)
                    except Exception:
                        q = 0
                s._veto_quality = q
                if q and q < quality_min:
                    return False, f"quality {q}/10 < {quality_min} — setup too weak"
                why += f" | quality {q}/10"
            if VETO_MODE == "advisory":
                # LLM veto DEMOTED to log-only: the action/agreement/quality are
                # still logged above (ledger + reflection), but the LLM's verdict
                # never gates the entry. Blackout/regime/conflict/quality gates
                # above still block. Re-enable with AUTOTRADE_VETO_MODE=gate.
                return True, why + " | [advisory]"
            return ok, why
        except Exception as e:
            if VETO_FAIL_OPEN:
                return True, f"⚠️ veto down, fail-open ({e})"
            return False, f"veto service unavailable — entry blocked ({e})"

    return veto_fn


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


# Module-level fetch pool (created once in main; reused every beat — the
# per-beat ThreadPoolExecutor was rebuilt ~80x/session for no reason).
_FETCH_POOL = None


def save_state(st: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE_FILE)


# ---- Topstep consistency rule (best day < 40% of total profit) ----------
def consistency_status(st: dict) -> dict:
    """Compute the consistency ratio from the state's per-day PnL history.

    st["pnl_history"] = [{date, pnl}, ...] of completed ET days plus
    st["today_pnl"] for the current day. Returns {ratio, best, total, n_days}.
    ratio = best_day_pnl / total_pnl (only when total > 0).
    """
    hist = st.get("pnl_history") or []
    today = float(st.get("today_pnl", 0.0))
    days = [float(h.get("pnl", 0.0)) for h in hist] + [today]
    total = sum(days)
    best = max(days) if days else 0.0
    ratio = (best / total) if total > 1e-9 else 0.0
    return {"ratio": round(ratio, 3), "best": round(best, 2),
            "total": round(total, 2), "n_days": len(days)}


def check_consistency(st: dict) -> str:
    """Alert + halt logic. Returns a reason string if entries must halt for
    the day (best-day breach), else ''."""
    cs = consistency_status(st)
    if cs["total"] <= 0 or cs["n_days"] < 2:
        return ""                      # no profit yet / too few days to judge
    ratio, best, total = cs["ratio"], cs["best"], cs["total"]

    if ratio >= CONSISTENCY_LIMIT:
        msg = (f"📊 CONSISTENCY BREACH — best day ${best:,.0f} is "
               f"{100*ratio:.0f}% of ${total:,.0f} total profit (limit "
               f"{100*CONSISTENCY_LIMIT:.0f}%)")
        if CONSISTENCY_HALT and not st.get("consistency_halted"):
            st["consistency_halted"] = True
            alert(f"🚫 {msg} — entries halted for the day to protect payouts")
            return "consistency rule breached — entries halted for the day"
        if not st.get("consistency_alerted"):
            st["consistency_alerted"] = True
            alert(f"📊 {msg} — payout eligibility at risk")
    elif ratio >= CONSISTENCY_WARN and not st.get("consistency_warned"):
        st["consistency_warned"] = True
        alert(f"⚠️ Consistency approaching limit: best day ${best:,.0f} = "
              f"{100*ratio:.0f}% of ${total:,.0f} (warn at "
              f"{100*CONSISTENCY_WARN:.0f}%, limit {100*CONSISTENCY_LIMIT:.0f}%)")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="one immediate iteration (smoke test), then exit")
    args = ap.parse_args()

    print(f"=== AUTOTRADE supervisor {dt.datetime.now().isoformat()} "
          f"loss_limit=${DAILY_LOSS_LIMIT:.0f} dd_limit=${TRAILING_DD_LIMIT:.0f} "
          f"profit_target=${PROFIT_TARGET:.0f}+-${PROFIT_BAND:.0f}/{PROFIT_WINDOW_H:.0f}h ===",
          flush=True)

    client = make_broker()
    client.authenticate()
    acct = client.pick_account(config.ACCOUNT)

    # Multi-symbol: one BotContext per traded symbol (models are shared —
    # strategies are timeframe-based; config.SYMBOL is swapped per context
    # so grading + the veto prompt see the right instrument).
    ctxs = []
    for sym in config.TRADE_SYMBOLS:
        contract = client.get_active_contract(sym)
        # Fail-closed tick resolution: a missing/invalid tickSize must never
        # reach the order path (2026-08-18: a guessed 1.0 placed a GC stop
        # 10x too wide — -$914). Refuse to run if unresolvable.
        try:
            tick_size = float(contract["tickSize"])
            tick_value = float(contract["tickValue"])
        except (KeyError, TypeError, ValueError):
            _specs = client.get_contract_specs(sym) or (0.0, 0.0)
            tick_size, tick_value = float(_specs[0]), float(_specs[1])
        if not tick_size or tick_size <= 0 or not tick_value or tick_value <= 0:
            raise RuntimeError(
                f"cannot resolve tick spec for {sym!r} "
                f"(tickSize={tick_size!r} tickValue={tick_value!r}) — "
                f"refusing to run (fail-closed)")
        # Per-symbol roster: base lanes + orb only where the screen proved it
        # (NQ/ES/GC 54-60% PF 2.4+; YM/RTY excluded — negative/thin).
        from strategies import make_strategies as _mk
        _active = list(config.ACTIVE_STRATEGIES)
        if sym in config.ORB_SYMBOLS and "orb" not in _active:
            _active.append("orb")
        c = BotContext(client, acct["id"], contract["id"], tick_size, tick_value,
                       strategies=_mk(_active))
        c.symbol = sym
        c.veto_fn = make_veto_fn()
        c.entry_gate = entry_gate          # 09:30-12:00 ET window (cycle-3 GO)
        c.evolve_floor = config.PROBA_FLOOR
        c.on_trade_close = None        # wired after evolver creation
        ctxs.append(c)
        print(f"  🛰 {sym:3s} -> {contract['id']} (tick {tick_size:g}) "
              f"[{'+'.join(s.name for s in c.strategies)}]", flush=True)

    evolver = evolve.Evolver(baseline_floor=config.PROBA_FLOOR)
    edge = edge_monitor.EdgeMonitor()            # rolling edge monitor (advisory-first)
    for c in ctxs:
        c.on_trade_close = evolver.record
        c.evolve_floor = evolver.current_floor()
        c.evolve_ceil = evolver.ceil
        c.on_signal = edge.record_signal          # Tier 1 clear-rate
        if config.PROBA_CEIL >= 1.0:      # ceiling disabled in config -> follow the evolver
            config.PROBA_CEIL = evolver.ceil

    # Reflection memory (#5): every closed trade's veto reason + outcome
    # appended to the lessons file. Wrapped around evolver.record so both run.
    import reflection as reflection_mem
    _orig_record = evolver.record

    def _record_with_reflection(trade: dict):
        _orig_record(trade)
        reflection_mem.record(trade)
        edge.record(trade)                       # Tier 2 realized-edge
        # GAME: score the closed trade (lives/XP/streak/gold). Advisory —
        # never raises into the trading path.
        try:
            import game as game_mod
            ev = game_mod.record_trade(
                trade.get("r", 0.0),
                grade=trade.get("game_grade", "") or "",
                gold_delta=trade.get("pnl"))
            print(f"🎮 {ev['event']}: R={ev['r']:+.2f} win={ev['win']} "
                  f"grade={ev['grade']} XP={ev['xp']} lives={ev['lives']} "
                  f"gold=${ev['gold']:.0f} streak={ev['streak']}", flush=True)
            # victory / defeat checks against today's gold
            if ev["gold"] >= 1000:
                game_mod.record_victory()
                print("🎮🏆 VICTORY — $1,000 day banked! (+10 XP)", flush=True)
            elif ev["gold"] <= -400 or ev["lives"] <= 0:
                game_mod.record_defeat(
                    "gold<=-400" if ev["gold"] <= -400 else "lives=0")
                print("🎮💀 DEFEAT — day lost. Breaker will handle the halt.",
                      flush=True)
        except Exception as e:
            print(f"⚠️ game scoring error (non-fatal): {e}", flush=True)

    evolver.record = _record_with_reflection
    for c in ctxs:
        c.on_trade_close = evolver.record

    # Warm the news/event caches now so the first live signal isn't delayed
    # by network fetches inside the veto path.
    try:
        import market_context
        for sym in config.TRADE_SYMBOLS:
            market_context.context_line(sym)
        print("📰 market context primed", flush=True)
    except Exception as e:
        print(f"⚠️ market context warm-up failed (non-fatal): {e}", flush=True)
    names = "+".join(s.name for s in ctxs[0].strategies)
    global _FETCH_POOL
    _FETCH_POOL = ThreadPoolExecutor(max_workers=max(4, len(ctxs)))
    print(f"✅ {acct['name']} | symbols={','.join(c.symbol for c in ctxs)} | "
          f"{config.TIMEFRAME_MIN}-min | [{names}] | conf≥{config.PROBA_FLOOR} "
          f"≤{config.PROBA_CEIL} | "
          f"entry-window: {'09:30-12:00 ET [ON]' if ENTRY_WINDOW else '24h [OFF]'} | "
          f"exit: {ctxs[0].exit_mode} | "
          f"pattern: {config.PATTERN_STOP_MULT}xATR cd{config.PATTERN_COOLDOWN_SIGNALS} | "
          f"veto: 7B fine-tuned LLM ({VETO_MODE}, "
          f"{'fail-open' if VETO_FAIL_OPEN else 'fail-closed'}) | "
          f"evolve: floor {evolver.current_floor():.2f} ceil {evolver.ceil:.2f} "
          f"({evolver.stance}) | edge-monitor: {edge_monitor.MODE}",
          flush=True)

    st = load_state()
    today = now_et().date().isoformat()
    if st.get("date") != today:
        st = {"date": today, "start_balance": None, "at_peak": None,
              "halted": False, "reason": "", "trades": 0, "last_beat": "",
              "pnl_history": st.get("pnl_history") or [],
              "today_pnl": 0.0, "consistency_halted": False,
              "profit_window_start": st.get("profit_window_start"),
              "profit_window_start_balance": st.get("profit_window_start_balance"),
              "profit_halted": st.get("profit_halted", False),
              "profit_halt_reason": st.get("profit_halt_reason", ""),
              "profit_zone_alerted": st.get("profit_zone_alerted", False)}
    if st.get("start_balance") is None:
        # account started at $50k; the all-time peak can never be below that
        st["start_balance"] = float(acct["balance"])
        st["at_peak"] = max(float(acct["balance"]), 50_000.0)
        save_state(st)

    # Restore in-flight entry bookkeeping (persisted each beat) so a restart
    # mid-trade still records the exit into the evolution ledger. Only restore
    # when the position is ACTUALLY open — a record whose trade already closed
    # at the broker (process died before booking the exit, e.g. the 2026-08-19
    # account lock) is a GHOST: it would make the per-symbol scan break after
    # the first symbol and block the rest of the book. Drop ghosts.
    for c in ctxs:
        lt = (st.get("last_trades") or {}).get(c.symbol)
        if isinstance(lt, dict) and lt.get("entry") is not None:
            try:
                pos = client.open_position(c.account_id, c.contract_id)
            except Exception:
                pos = None
            if pos is None:
                print(f"  ↻ dropped stale in-flight bookkeeping for {c.symbol} "
                      f"(position already closed at the broker)", flush=True)
                continue
            lt = dict(lt)
            nm = lt.get("strategy")
            lt["strategy"] = next((s for s in c.strategies
                                   if s.name == nm), None)
            c.last_trade = lt
            print(f"  ↻ restored in-flight bookkeeping for {c.symbol}", flush=True)

    # Restart-safety markers: bars <= the persisted last-seen time must never
    # re-fire after a restart (a veto-blocked or already-traded signal was the
    # 2026-08-18 GC -$914 incident). Restore them now.
    for c in ctxs:
        ts = (st.get("seen_signals") or {}).get(c.symbol)
        if ts:
            try:
                c.min_bar_time = pd.Timestamp(ts)
                print(f"  ↻ seen-signal marker {c.symbol}: bars ≤ {ts} "
                      f"will not re-fire", flush=True)
            except Exception:
                pass

    trade_state = None
    while True:
        if not args.once:
            period = config.TIMEFRAME_MIN * 60
            time.sleep(period - (time.time() % period) + 2)
        now = now_et()
        today = now.date().isoformat()

        # daily reset (ET day boundary): daily-loss baseline resets, but the
        # all-time peak (suspension-relevant) NEVER resets.
        if st.get("date") != today:
            # close out yesterday's PnL into the consistency history
            prev_pnl = float(st.get("today_pnl", 0.0))
            if abs(prev_pnl) > 1e-9:
                hist = st.get("pnl_history") or []
                hist.append({"date": st["date"], "pnl": round(prev_pnl, 2)})
                st["pnl_history"] = hist[-60:]     # keep last 60 days
            st["date"] = today
            st["start_balance"] = float(acct["balance"])
            st["halted"] = False
            st["reason"] = ""
            st["trades"] = 0
            st["last_beat"] = ""
            st["today_pnl"] = 0.0
            st["consistency_halted"] = False
            st["consistency_warned"] = False
            st["consistency_alerted"] = False
            print(f"🔄 new ET day {today} — breaker reset", flush=True)
            alert(f"🔄 AUTOTRADE new day {today} — breaker reset")

        try:
            acct = client.pick_account(config.ACCOUNT)
            bal = float(acct["balance"])
            st["at_peak"] = max(st.get("at_peak", bal), bal)
            daily_loss = st["start_balance"] - bal
            dd = st["at_peak"] - bal

            # circuit breaker checks (real account numbers). Topstep 50K kills
            # at -$2,000/day and -$2,000 from all-time peak; our limits sit
            # well inside both so suspension is unreachable.
            if not st.get("halted"):
                reason = ""
                if daily_loss >= DAILY_LOSS_LIMIT:
                    reason = f"daily loss ${daily_loss:,.0f} >= ${DAILY_LOSS_LIMIT:,.0f}"
                elif dd >= TRAILING_DD_LIMIT:
                    reason = f"trailing DD ${dd:,.0f} from all-time peak ${st['at_peak']:,.0f} >= ${TRAILING_DD_LIMIT:,.0f}"
                if reason:
                    st["halted"] = True
                    st["reason"] = reason
                    try:
                        # close EVERYTHING across all symbols
                        for c in ctxs:
                            pos = client.open_position(c.account_id, c.contract_id)
                            if pos:
                                client.close_position(c.account_id, c.contract_id)
                                print(f"🛑 breaker: closed {c.symbol} {pos}", flush=True)
                            stray = client.cancel_orders(c.account_id, c.contract_id)
                            if stray:
                                print(f"🛑 breaker: cancelled {stray} stray {c.symbol} orders",
                                      flush=True)
                    except Exception as e:
                        print(f"⚠️ breaker close failed: {e}", flush=True)
                    alert(f"🚨 AUTOTRADE HALTED — {reason}. All flat. "
                          f"Resumes next ET day. Balance ${bal:,.2f}")
                    print(f"🚨 HALTED: {reason}", flush=True)

            else:
                # Recovery: a trailing-DD halt clears the moment the account
                # climbs back above the wall (danger passed). A daily-loss halt
                # stays until the ET-day reset (discipline).
                if ("trailing DD" in str(st.get("reason", ""))
                        and daily_loss < DAILY_LOSS_LIMIT
                        and dd < TRAILING_DD_LIMIT):
                    st["halted"] = False
                    st["reason"] = ""
                    print("🔄 breaker cleared (DD wall) — resuming", flush=True)
                    alert("🔄 AUTOTRADE resumed — trailing-DD wall cleared")

            # Profit target (rolling window): bank PROFIT_TARGET (+/- BAND) within
            # PROFIT_WINDOW_H hours, then close everything and pause entries so the
            # win is locked in (user goal: make $500 ±$150 in 25h).
            win_start = st.get("profit_window_start")
            win_expired = (
                win_start is None or
                (now - dt.datetime.fromisoformat(win_start)).total_seconds()
                >= PROFIT_WINDOW_H * 3600
            )
            if win_expired:
                st["profit_window_start"] = now.isoformat()
                st["profit_window_start_balance"] = bal
                st["profit_halted"] = False
                st["profit_zone_alerted"] = False
                print(f"🎯 profit window ({PROFIT_WINDOW_H:.0f}h) from ${bal:,.2f} "
                      f"— target +${PROFIT_TARGET:,.0f} (±${PROFIT_BAND:,.0f})",
                      flush=True)
            win_pnl = bal - float(st["profit_window_start_balance"])
            st["profit_window_pnl"] = round(win_pnl, 2)
            if not st.get("profit_halted") and win_pnl >= PROFIT_TARGET:
                st["profit_halted"] = True
                st["profit_halt_reason"] = (
                    f"profit target banked: +${win_pnl:,.2f} "
                    f"(goal ${PROFIT_TARGET:,.0f} ±${PROFIT_BAND:,.0f} "
                    f"in {PROFIT_WINDOW_H:.0f}h)")
                try:
                    for c in ctxs:
                        pos = client.open_position(c.account_id, c.contract_id)
                        if pos:
                            client.close_position(c.account_id, c.contract_id)
                            print(f"🎯 profit-lock: closed {c.symbol} {pos}",
                                  flush=True)
                        stray = client.cancel_orders(c.account_id, c.contract_id)
                        if stray:
                            print(f"🎯 profit-lock: cancelled {stray} stray "
                                  f"{c.symbol} orders", flush=True)
                except Exception as e:
                    print(f"⚠️ profit-lock close failed: {e}", flush=True)
                alert(f"🎯 PROFIT TARGET BANKED — +${win_pnl:,.2f} "
                      f"(goal ${PROFIT_TARGET:,.0f} ±${PROFIT_BAND:,.0f} in "
                      f"{PROFIT_WINDOW_H:.0f}h). Closed everything — locking it in. "
                      f"Balance ${bal:,.2f}")
                print(f"🎯 PROFIT TARGET BANKED: +${win_pnl:,.2f} — locking in",
                      flush=True)
            elif (not st.get("profit_halted")
                    and win_pnl >= PROFIT_TARGET - PROFIT_BAND
                    and not st.get("profit_zone_alerted")):
                st["profit_zone_alerted"] = True
                alert(f"🎯 profit zone reached: +${win_pnl:,.2f} of "
                      f"${PROFIT_TARGET:,.0f} target (band ±${PROFIT_BAND:,.0f})")
                print(f"🎯 profit zone: +${win_pnl:,.2f} — "
                      f"${max(PROFIT_TARGET - win_pnl, 0):,.2f} to target",
                      flush=True)

            # heartbeat (always, even halted)
            st["last_beat"] = dt.datetime.now().isoformat()
            st["balance"] = bal
            st["today_pnl"] = round(bal - st["start_balance"], 2)
            st["evolve"] = evolver.status()      # floor/stance for the watchdog
            st["edge_monitor"] = edge.status()   # edge-monitor state (advisory/enforce)
            st["consistency"] = consistency_status(st)
            # Persist per-symbol entry bookkeeping so a restart or external
            # close mid-trade can never lose the trade from the learner's
            # ledger (2026-08-18: NQ win + ES win were never recorded because
            # last_trade was memory-only and the process restarted).
            st["last_trades"] = {
                c.symbol: {k: (v.name if k == "strategy" and v else v)
                           for k, v in c.last_trade.items()}
                for c in ctxs if getattr(c, "last_trade", None)
            }
            # Restart-safety: newest bar each symbol has consumed. Restored on
            # startup as min_bar_time so no already-processed bar can re-fire.
            st["seen_signals"] = {
                c.symbol: str(c.last_bar_seen)
                for c in ctxs if getattr(c, "last_bar_seen", None) is not None
            }
            save_state(st)

            if st.get("halted") or st.get("profit_halted"):
                h_reason = st.get("reason") or st.get("profit_halt_reason", "")
                print(f"⛔ halted ({h_reason}) — skipping bar", flush=True)
                trade_state = None
                if args.once:
                    break
                continue

            # Consistency gate: best-day breach → no entries today (payout path)
            cons_reason = check_consistency(st)
            if cons_reason and st.get("consistency_halted"):
                print(f"📊 {cons_reason} — skipping bar", flush=True)
                trade_state = None
                if args.once:
                    break
                continue

            # RTH gate: outside session → idle (no entries, no manages)
            if not in_rth(now):
                print(f"⏸ outside RTH ({now.strftime('%H:%M')}) — idling", flush=True)
                trade_state = None
                if args.once:
                    break
                continue

            # Per-symbol scan: MAX_OPEN_POSITIONS cap across the whole book
            # (TopstepX eval = 1 — see config.MAX_OPEN_POSITIONS). When the
            # book is at cap, only the owning symbol manages; the others are
            # skipped for entries.
            book_pos = client.any_open_position(ctxs[0].account_id)
            if book_pos is not None:
                book_contract = book_pos.get("contractId")
            else:
                book_contract = None

            # Foreign-position watch (2026-08-19): a position on a contract the
            # bot doesn't trade = a manual/other-session trade on the eval
            # account. It holds bot entries (one-position cap) and bleeds the
            # same DD wall — alert once per day so it can't ride along silently.
            known_contracts = {c.contract_id for c in ctxs}
            if (book_contract is not None
                    and book_contract not in known_contracts
                    and st.get("foreign_pos_alerted") != today):
                st["foreign_pos_alerted"] = today
                alert(f"👁 FOREIGN POSITION on the eval account: "
                      f"{book_contract} — a manual or other-session trade is "
                      f"open. Bot entries are on hold (one-position cap).")
                print(f"👁 foreign position {book_contract} — manual trade? "
                      f"entries held", flush=True)

            # PARALLEL bar fetches (network is the slow part; requests.Session
            # is thread-safe). Decisions below stay SERIAL: config.SYMBOL is a
            # module global that grading/veto read, and the one-position rule
            # needs a single serial decision point. Pool is module-level.
            def _fetch(c):
                try:
                    bars = client.get_bars(c.contract_id, config.TIMEFRAME_MIN)
                    return c, bars
                except Exception as e:
                    print(f"⚠️ fetch {c.symbol} failed: {e}", flush=True)
                    return c, None

            fetched = {}
            for c, bars in _FETCH_POOL.map(_fetch, ctxs):
                fetched[c.symbol] = bars

            # VETO PREFETCH (feature #3): warm the veto cache with the CURRENT
            # state of every symbol, in the background. When a signal fires on
            # a state we've already seen, the veto answer is INSTANT (cache
            # hit, ~0.01s) instead of a 17-32s GPU call. Fire-and-forget: if
            # the prefetch is still running when a real signal needs the veto,
            # the real call just waits on the same lock (no worse than before).
            try:
                if os.environ.get("AUTOTRADE_VETO_PREFETCH", "1") == "1":
                    import threading
                    def _prefetch():
                        try:
                            states = []
                            for c in ctxs:
                                bars = fetched.get(c.symbol)
                                if bars is None or len(bars) < 30:
                                    continue
                                states.append(build_state_line(bars, c.symbol))
                            if states:
                                requests.post(
                                    "http://127.0.0.1:8765/decide_batch",
                                    json={"texts": states}, timeout=60)
                                # v2 (2026-08-21): ALSO warm the quality cache —
                                # /score was always cold before, adding ~8s to
                                # every entry. Both answers now hit on repeat.
                                requests.post(
                                    "http://127.0.0.1:8765/score_batch",
                                    json={"texts": states}, timeout=60)
                        except Exception:
                            pass          # prefetch is best-effort only
                    threading.Thread(target=_prefetch, daemon=True).start()
            except Exception as e:
                print(f"⚠️ veto prefetch failed: {e}", flush=True)

            edge.tick()          # edge monitor: auto-resume after cooldown (enforce mode)

            for c in ctxs:
                config.SYMBOL = c.symbol          # grading + veto see this instrument
                c.evolve_floor = evolver.current_floor()
                c.evolve_ceil = evolver.ceil
                c.edge_gate = edge                # hard gate when MODE == enforce

                # Book cap (config.MAX_OPEN_POSITIONS — TopstepX eval: 1):
                # a position is open somewhere, so only its owner manages;
                # other symbols skip entries this cycle.
                if (config.MAX_OPEN_POSITIONS <= 1
                        and book_contract is not None
                        and book_contract != c.contract_id):
                    continue

                bars = fetched.get(c.symbol)
                if bars is None or len(bars) < config.CTX + 30:
                    continue
                # Live candlestick patterns (30-min view) — logged each cycle,
                # observation-only: never gates until the ledger proves an edge.
                try:
                    import candle_patterns
                    for t, pats in candle_patterns.detect_patterns(bars):
                        if pats:
                            print(f"🕯 {c.symbol} 30m {t} "
                                  f"{' + '.join(pats)}", flush=True)
                except Exception as e:
                    print(f"⚠️ pattern detect {c.symbol} failed: {e}", flush=True)
                trade_state = handle_bar(c, bars, trade_state)
                # fixed-RR mode returns None even after an entry — break on
                # the entry bookkeeping too, or the next symbol could attempt
                # a second entry in the same cycle (2026-08-18: the platform's
                # one-position rule rejected it, but it must be explicit).
                if trade_state is not None or c.last_trade is not None:
                    break                        # entered — done for this cycle

        except Exception as e:
            print(f"⚠️ loop error: {type(e).__name__}: {e}", flush=True)
            alert(f"⚠️ AUTOTRADE error: {type(e).__name__}: {str(e)[:120]}")

        if args.once:
            break

    print("=== supervisor exit ===", flush=True)


if __name__ == "__main__":
    main()
