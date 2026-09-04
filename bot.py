#!/usr/bin/env python3
"""
bot.py — multi-strategy TopstepX AI bot with a PPO trailing exit.

Each bar:  every active strategy detects its mechanical entry  →  grades it with
its own Chronos+XGBoost model  →  the best graded signal (proba ≥ floor) is
taken  →  the PPO policy trails the stop until exit.

    detect (SuperTrend flip / EMA cross)  →  model grades  →  enter  →  PPO trail

Strategies and exit behaviour are configured in config.py / .env. Run:

    pip install -r requirements.txt
    cp .env.example .env          # then fill in your TopstepX credentials
    python bot.py                 # live (places LIVE orders)
    python bot.py --backtest --symbol NQ --start 2026-01-01 --end 2026-03-01
    python bot.py --retrain-exit  # retrain the PPO trailing exit

⚠️  EDUCATIONAL — live mode places LIVE orders. Run it on a practice/evaluation
    account first. NQ 3-min is the models' training scope.
"""
import datetime as dt
from datetime import timedelta
import os
import time

import config
from ppo_exit import exit_manager as ex
import strategies as strat
from broker import SIDE, make_broker
from logsetup import get_logger

log = get_logger()


def ensure_exit_policy():
    """The PPO exit policy for the active timeframe, training one if it's missing.

    With USE_PPO_EXIT on and no policy for this timeframe, FLAG it and run
    train_ppo_exit for that timeframe — in a SUBPROCESS, so torch/SB3 never load
    next to xgboost in the trading process (they segfault together on macOS).
    Returns the policy path, or None if PPO exit is off or the train produced
    nothing (the bot then falls back to the fixed-RR bracket exit)."""
    if not config.USE_PPO_EXIT:
        return None
    path = config.policy_path()
    if os.path.exists(path):
        return path
    log.warning("⚠️  no PPO exit policy for %d-min (%s missing) — training one now "
                "(one-time per timeframe; runs train_ppo_exit)…",
                config.TIMEFRAME_MIN, os.path.basename(path))
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "ppo_exit.train_ppo_exit",
                        "--timeframe", str(config.TIMEFRAME_MIN)], cwd=config.HERE)
    if r.returncode != 0 or not os.path.exists(path):
        log.warning("⚠️  could not train a %d-min PPO policy — falling back to the "
                    "fixed %sR exit", config.TIMEFRAME_MIN, config.RR)
        return None
    log.info("✅ trained PPO exit for %d-min → %s",
             config.TIMEFRAME_MIN, os.path.basename(path))
    return path


class BotContext:
    """Everything a bar needs: the broker (live or simulated), the active
    strategies, the PPO policy, and the trade identifiers. Shared by the live
    loop and the backtester so they run identical per-bar logic."""

    def __init__(self, client, account_id, contract_id, tick_size,
                 tick_value=0.0, log_candles=True, strategies=None):
        from ppo_exit import trail_exit_env as tee  # numpy-only PPO policy loader
        self.client = client
        self.account_id = account_id
        self.contract_id = contract_id
        self.tick_size = tick_size
        self.tick_value = tick_value      # $ per tick per contract (for risk sizing)
        self.log_candles = log_candles
        self.tee = tee
        # Per-symbol strategy roster (supervisor passes symbol-specific lists;
        # default = the global ACTIVE_STRATEGIES set).
        self.strategies = (strat.make_strategies() if strategies is None
                           else strategies)
        self.policy = None
        pol = ensure_exit_policy()
        if pol:
            self.policy = tee.NumpyMlpPolicy.load(pol)
        self.trailing = bool(self.policy) and config.USE_TRAILING_STOP
        # Optional (s, sig, bars) -> (bool, reason) confirm/veto filter before
        # any entry order. None = no filter. Set by supervisor.py for the LLM
        # veto layer; backtests leave it unset.
        self.veto_fn = None
        # Self-evolution hooks (set by supervisor.py; None in backtests):
        #  - evolve_floor: dynamic entry-confidence floor from the Evolver
        #    (None -> config.PROBA_FLOOR)
        #  - evolve_ceil: dynamic confidence ceiling from the Evolver
        #    (None -> config.PROBA_CEIL)
        #  - on_trade_close: callable(trade_dict) called whenever a trade
        #    closes, with the realized R — feeds the evolution engine
        #  - last_trade: entry bookkeeping so a silent broker bracket fill
        #    (fixed-RR mode) can still be recorded accurately
        self.evolve_floor = None
        self.evolve_ceil = None
        self.on_trade_close = None
        self.on_signal = None       # callable(proba) per graded signal (edge monitor Tier 1)
        # Selection-validator hook (2026-08-30, spec-kit selection-validation):
        # callable(s, sig, floor, ceil, bars) fired for EVERY graded signal
        # (before jump/take filtering) — observation-only, default None.
        # Feeds the selection_validator dataset. Never affects decisions.
        self.on_graded_signal = None
        # Edge-monitor hard gate (2026-08-23): set by supervisor to the live
        # EdgeMonitor instance. When AUTOTRADE_EDGE_MONITOR=enforce and the
        # monitor's bootstrap says the edge has collapsed, blocks_entries()
        # stops new entries (managing open positions still proceeds).
        self.edge_gate = None
        self.last_trade = None
        # Restart-safety (2026-08-19): min_bar_time is the newest bar this
        # process may still DETECT on — any bar <= it was already processed in
        # a previous run. Persisted in supervisor state so a restart can never
        # re-fire a signal that was already veto-blocked or already traded
        # (2026-08-18: a veto-blocked GC signal re-fired after a restart and
        # took the -$914 tick-bug loss). None = no marker (fresh start).
        self.min_bar_time = None
        # last_bar_seen: newest bar processed this run (persisted per beat).
        self.last_bar_seen = None
        # Entry-window gate (2026-08-31, spec-kit time-window GO): callable()
        # -> bool, True = entries allowed. Set by supervisor; None = no gate
        # (backtests unchanged). MANAGEMENT always runs; only NEW entries are
        # gated. The spec-kit verdict (09:30-12:00 ET window, OOS P=1.0) is
        # wired here. Env AUTOTRADE_ENTRY_WINDOW=1 + service env decide.
        self.entry_gate = None
        # Pattern-lane cooldown: skip the next N pattern signals after a -1R
        # stop (config.PATTERN_COOLDOWN_SIGNALS). In-memory only — a restart
        # resets it, which is acceptable (seen-signal marker covers replays).
        self.pattern_skip_signals = 0

    @property
    def exit_mode(self):
        return ("PPO native-trail" if self.trailing else
                "PPO stop-reprice" if self.policy else f"fixed {config.RR}R")

    @property
    def sizing_mode(self):
        if config.RISK_PER_TRADE and self.tick_value:
            return f"risk ${config.RISK_PER_TRADE:g}/trade (≤{config.MAX_CONTRACTS})"
        return f"fixed {config.SIZE}"


def position_size(ctx: BotContext, stop_ticks: int) -> int:
    """Contracts for a trade: risk-based when RISK_PER_TRADE > 0 (size from the
    stop distance), else the fixed SIZE — capped at MAX_CONTRACTS.

        size = min(MAX_CONTRACTS,
                   risk_sizing and stop_ticks
                       ? max(1, floor(RISK_PER_TRADE / (|stop_ticks| × tick_value)))
                       : SIZE)
    """
    if config.RISK_PER_TRADE and ctx.tick_value and stop_ticks:
        per_contract = abs(stop_ticks) * ctx.tick_value
        n = max(1, int(config.RISK_PER_TRADE // per_contract))
    else:
        n = config.SIZE
    return min(config.MAX_CONTRACTS, n)


def handle_bar(ctx: BotContext, bars, trade_state):
    """Run one bar of bot logic: if in a trade, trail the stop; otherwise detect
    + grade across strategies and enter the best signal. Returns the updated
    trade_state. Identical for live trading and backtesting."""
    c = ctx.client
    stamp = bars["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    # Restart-safety: remember the newest bar we've consumed. The supervisor
    # persists this per beat so a restart skips bars already processed
    # (prevents re-firing veto-blocked or already-traded signals). The marker
    # advances in-process too, so the SAME bar can never be re-detected on a
    # later cycle (2026-08-18: the 12:30 candle was entered once and then
    # re-evaluated 3 min later in the same process). The detection gate below
    # compares against _pre_seen — the marker from BEFORE this call — so the
    # current bar is always eligible exactly once.
    _pre_seen = getattr(ctx, "min_bar_time", None)
    ctx.last_bar_seen = bars["time"].iloc[-1]
    if _pre_seen is None or bars["time"].iloc[-1] > _pre_seen:
        ctx.min_bar_time = bars["time"].iloc[-1]
    if ctx.log_candles:
        last = bars.iloc[-1]
        log.info("candle %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%s", stamp,
                 last["open"], last["high"], last["low"], last["close"],
                 last.get("volume", "?"))

    pos = c.open_position(ctx.account_id, ctx.contract_id)
    just_exited = False        # set when a flat-detected exit was logged this bar
    if pos:
        # In a trade — let the PPO policy trail the stop. Without a policy the
        # attached fixed bracket manages the exit itself.
        if ctx.policy is None:
            return trade_state
        if trade_state is None:
            trade_state = ex.reconstruct_state(
                c, ctx.account_id, ctx.contract_id, pos, ctx.strategies[0])
        if trade_state:
            prev = trade_state
            trade_state = ex.manage_trail(ctx.tee, ctx.policy, c, ctx.account_id,
                                          ctx.contract_id, ctx.tick_size, bars,
                                          trade_state, ctx.trailing)
            if trade_state is None and ctx.on_trade_close is not None:
                # manage_trail closed at market (trailed-SL cross or max-hold):
                # realized R from the bar close vs entry.
                sign = prev["sign"]
                cur = float(bars["close"].iloc[-1])
                r = sign * (cur - prev["entry"]) / prev["risk"]
                s = prev.get("strategy")
                if r <= -1.0 and s is not None and s.name == "pattern":
                    ctx.pattern_skip_signals = config.PATTERN_COOLDOWN_SIGNALS
                ctx.on_trade_close({
                    "r": r, "strategy": s.name if s else "?",
                    "side": "LONG" if sign > 0 else "SHORT",
                    "entry": prev["entry"], "exit": cur,
                    "quality": getattr(s, "_veto_quality", 0),
                    "veto_reason": getattr(s, "_veto_reason", ""),
                    "symbol": getattr(ctx, "symbol", "?"),
                    "game_grade": prev.get("game_grade", ""),
                    "pnl": r * prev["risk"] * ctx.tick_value * prev.get("size", 1),
                    "exit_kind": "trail",
                    "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                })
            if trade_state is None:
                ctx.last_trade = None
        return trade_state

    # Flat. If we were holding a trade (trade_state set) but the broker shows no
    # position and we didn't close it ourselves, the RESTING protective stop filled
    # at the broker. That exit is otherwise SILENT (manage_trail never ran), so log
    # it — inferred from the stop level it rested at — then clear state.
    if trade_state is not None:
        # 2026-08-19 phantom-exit fix: cancel EVERY resting order for the
        # contract BEFORE booking the exit. A leftover sibling (separate
        # stop/target from the Position-Brackets fallback) could otherwise
        # fill after this stop — into a naked position the bot no longer
        # tracks, or as a phantom second exit booked off bar extremes.
        c.cancel_orders(ctx.account_id, ctx.contract_id)
        px, r = ex.stop_fill_exit(trade_state)
        s = trade_state.get("strategy")
        log.info("🛑 EXIT %s [%s] %s | broker stop filled @ %.2f | %+.2fR | %d bars",
                 stamp, s.name if s else "?",
                 "LONG" if trade_state["sign"] > 0 else "SHORT",
                 px, r, trade_state.get("bars_held", 0))
        if r <= -1.0 and s is not None and s.name == "pattern":
            ctx.pattern_skip_signals = config.PATTERN_COOLDOWN_SIGNALS
        if ctx.on_trade_close is not None:
            ctx.on_trade_close({
                "r": r, "strategy": s.name if s else "?",
                "side": "LONG" if trade_state["sign"] > 0 else "SHORT",
                "entry": trade_state["entry"], "exit": px,
                "quality": getattr(s, "_veto_quality", 0),
                "veto_reason": getattr(s, "_veto_reason", ""),
                "proba": trade_state.get("proba", 0.0),
                "symbol": getattr(ctx, "symbol", "?"),
                "game_grade": trade_state.get("game_grade", ""),
                "pnl": r * trade_state["risk"] * ctx.tick_value * trade_state.get("size", 1),
                "exit_kind": "stop",
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
        trade_state = None
        ctx.last_trade = None
        just_exited = True
    elif ctx.policy is None and ctx.last_trade is not None:
        # Fixed-RR mode (no policy -> no trade_state): the broker bracket filled
        # silently (target OR stop). Infer which from the bar extremes, exactly
        # like SimBroker.process_exits: stop wins if both touched. Cancel any
        # resting siblings FIRST (see phantom-exit fix above).
        c.cancel_orders(ctx.account_id, ctx.contract_id)
        lt = ctx.last_trade
        sign = lt["sign"]
        hi, lo = float(bars["high"].iloc[-1]), float(bars["low"].iloc[-1])
        hit_stop = (lo <= lt["stop"]) if sign > 0 else (hi >= lt["stop"])
        hit_tgt = (hi >= lt["target"]) if sign > 0 else (lo <= lt["target"])
        if hit_stop:
            px, r, kind = lt["stop"], sign * (lt["stop"] - lt["entry"]) / lt["risk"], "stop"
        elif hit_tgt:
            px, r, kind = lt["target"], sign * (lt["target"] - lt["entry"]) / lt["risk"], "target"
        else:                       # filled on a gap — approximate at the close
            px = float(bars["close"].iloc[-1])
            r = sign * (px - lt["entry"]) / lt["risk"]
            kind = "gap"
        s = lt.get("strategy")
        log.info("🛑 EXIT %s [%s] %s | bracket %s filled @ %.2f | %+.2fR",
                 stamp, s.name if s else "?", "LONG" if sign > 0 else "SHORT",
                 kind, px, r)
        if r <= -1.0 and s is not None and s.name == "pattern":
            ctx.pattern_skip_signals = config.PATTERN_COOLDOWN_SIGNALS
        if ctx.on_trade_close is not None:
            ctx.on_trade_close({
                "r": r, "strategy": s.name if s else "?",
                "side": "LONG" if sign > 0 else "SHORT",
                "entry": lt["entry"], "exit": px,
                "quality": getattr(s, "_veto_quality", 0),
                "veto_reason": getattr(s, "_veto_reason", ""),
                "proba": lt.get("proba", 0.0),
                "symbol": getattr(ctx, "symbol", "?"),
                "game_grade": lt.get("game_grade", ""),
                "pnl": r * lt["risk"] * ctx.tick_value * lt.get("size", 1),
                "exit_kind": kind,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
        ctx.last_trade = None
        just_exited = True

    # Reconcile: a flat account should have NO resting orders, so cancel any strays
    # (e.g. a stop bracket orphaned by a market close, a missed exit, or a manual
    # order). Left alone, a stray could fill into an unmanaged naked position the
    # bot never opened and never trails. Skipped when an exit was just booked —
    # that path already cancelled resting siblings before booking.
    if not just_exited:
        stray = c.cancel_orders(ctx.account_id, ctx.contract_id)
        if stray:
            log.warning("🧹 %s  reconcile: cancelled %d stray order(s) while flat",
                        stamp, stray)
    if just_exited:
        return None          # never re-detect on the bar that just closed the trade

    # Restart-safety gate: if the newest bar was already processed by a
    # previous run (marker set), do NOT detect — position management above
    # still runs, but no new entries can re-fire from history. Uses _pre_seen
    # (the marker from before this call) so the current bar is never blocked
    # on its first pass.
    if _pre_seen is not None and bars["time"].iloc[-1] <= _pre_seen:
        return None

    # ENTRY-WINDOW GATE (2026-08-31, spec-kit time-window GO): management
    # above always runs; only NEW entries are gated. Observation-only when
    # unset (default None — backtests and any ctx without a gate are
    # unchanged). The gate is the supervisor's, built from America/New_York
    # wall time (DST-correct per the cycle-3 DST lesson).
    if getattr(ctx, "entry_gate", None) is not None:
        try:
            if not ctx.entry_gate(bars["time"].iloc[-1]):
                log.info("⏱ entry-window gate: outside 09:30-12:00 ET — "
                         "no new entries (management continues)")
                return None
        except Exception:
            pass                       # gate failure = allow (never blocks trading)

    # Detect across strategies (cheap), then grade. Strategies that fire on this
    # bar share one Chronos embedding (same context) — computed once.
    fired = [(s, sig) for s in ctx.strategies if (sig := s.detect(bars))]
    candidates = []
    floor = ctx.evolve_floor if ctx.evolve_floor is not None else config.PROBA_FLOOR
    ceil = (ctx.evolve_ceil if ctx.evolve_ceil is not None
            else config.PROBA_CEIL)                 # confidence ceiling (1.0 = off)
    if fired:
        emb = strat.embed_context(bars, len(bars) - 1)   # one Chronos pass per bar
        for s, sig in fired:
            sig.proba, sig.r_hat = s.grade(bars, sig, emb=emb)
            if ctx.on_signal is not None:
                ctx.on_signal(sig.proba)          # Tier 1 clear-rate (before any filter)
            if getattr(ctx, "on_graded_signal", None) is not None:
                try:
                    ctx.on_graded_signal(s, sig, floor, ceil, bars)
                except Exception:
                    pass                          # observation-only — never affects the loop
            side_txt = "LONG" if sig.direction > 0 else "SHORT"
            # Jump filter (factor-zoo paper): skip signals on/after a jump bar —
            # |close-close| move >> ATR. Point-in-time, selection-only.
            if strat.recent_jump(bars, len(bars) - 1):
                log.info("signal %s [%s] %s | proba=%.3f | JUMP — skipped",
                         stamp, s.name, side_txt, sig.proba)
                continue
            take = (sig.proba >= floor) and (sig.proba <= ceil)
            log.info("signal %s [%s] %s | proba=%.3f r_hat=%.2f | %s", stamp,
                     s.name, side_txt, sig.proba, sig.r_hat,
                     "TAKE" if take else
                     (f"skip (>{ceil:.2f} ceil)" if sig.proba > ceil
                      else f"skip (<{floor:.2f})"))
            if take:
                # Pattern-lane cooldown (2026-08-19): skip the next N pattern
                # signals after a -1R stop — the lane re-fires every 30m close
                # and losing-streak re-entries were the top live bleed.
                if (s.name == "pattern"
                        and getattr(ctx, "pattern_skip_signals", 0) > 0
                        and sig.direction < 0):
                    ctx.pattern_skip_signals -= 1
                    log.info("signal %s [%s] %s | proba=%.3f | COOLDOWN — "
                             "skipped (%d pattern signal(s) left)",
                             stamp, s.name, side_txt, sig.proba,
                             ctx.pattern_skip_signals)
                    continue
                candidates.append((s, sig))

    if not candidates:
        return None
    s, sig = max(candidates, key=lambda c_: c_[1].proba)   # highest proba wins

    # GAME: grade the chosen setup (advisory — logged + stored on the trade
    # so record_trade can credit the grade later). Never blocks.
    try:
        import game as game_mod
        _aligned = bool(getattr(sig, "pattern_dir", 0)
                        and sig.pattern_dir == sig.direction)
        _g = game_mod.grade_setup(
            sig.proba, floor, aligned=_aligned, regime="trending",
            chop_ok=True, r_hat=sig.r_hat,
            et_hour=dt.datetime.now(dt.timezone(timedelta(hours=-4))).hour,
            symbol=getattr(ctx, "symbol", "?"),
            first_attempt=(getattr(ctx, "symbol", "?")
                           not in game_mod.load_state()["symbols_traded_today"]))
        sig.game_grade = _g["grade"]
        log.info("🎮 setup %s [%s] score=%d grade=%s mode=%s "
                 "(advisory)", stamp, s.name, _g["score"], _g["grade"],
                 _g["mode"])
    except Exception as e:
        log.warning("game grading failed (non-fatal): %s", e)

    stop_ticks = max(1, round(sig.risk / ctx.tick_size))
    size = position_size(ctx, stop_ticks)
    side = SIDE["BUY"] if sig.direction > 0 else SIDE["SELL"]
    side_txt = "LONG" if sig.direction > 0 else "SHORT"

    # Edge-monitor hard gate (enforce mode only): statistical edge-collapse
    # halt stops NEW entries; open-position management below still runs.
    if ctx.edge_gate is not None and ctx.edge_gate.blocks_entries():
        log.info("⏸ EDGE-HALT %s — edge monitor enforce: entries blocked", stamp)
        return trade_state

    # LLM veto layer (set by supervisor): confirm or block the entry.
    if ctx.veto_fn is not None:
        ok, why = ctx.veto_fn(s, sig, bars)
        if not ok:
            log.info("🚫 VETO %s %s [%s] | %s", stamp, side_txt, s.name, why)
            return None

    if ctx.policy is not None:
        trade_state = {"sign": sig.direction, "entry": sig.entry,
                       "risk": sig.risk, "stop": sig.stop, "bars_held": 0,
                       "mfe": 0.0, "peak_R": 0.0, "trail_ticks": stop_ticks,
                       "strategy": s, "proba": sig.proba,
                       "game_grade": getattr(sig, "game_grade", ""),
                       "symbol": getattr(ctx, "symbol", "?")}
        ctx.last_trade = dict(trade_state)
        ctx.last_trade["target"] = sig.entry + sig.direction * config.RR * sig.risk
        if ctx.trailing:
            c.place_market_with_trail(ctx.account_id, ctx.contract_id,
                                      side=side, size=size, trail_ticks=stop_ticks)
            log.info("🎯 ENTER %s %s [%s] %d | native trail %dt | PPO (proba %.3f)",
                     stamp, side_txt, s.name, size, stop_ticks, sig.proba)
        else:
            c.place_market_with_stop(ctx.account_id, ctx.contract_id,
                                     side=side, size=size, stop_ticks=stop_ticks,
                                     tick_size=ctx.tick_size)
            log.info("🎯 ENTER %s %s [%s] %d | stop %dt | PPO reprice (proba %.3f)",
                     stamp, side_txt, s.name, size, stop_ticks, sig.proba)
    else:
        target_ticks = max(1, round(config.RR * sig.risk / ctx.tick_size))
        # fixed-RR mode: no trade_state, so keep entry bookkeeping BEFORE the
        # order call — if the broker fill lands but the fallback raises after
        # (slow fill, rejected protective order), the next bar's silent-fill
        # detection still accounts the position instead of orphaning it.
        ctx.last_trade = {"sign": sig.direction, "entry": sig.entry,
                          "risk": sig.risk, "stop": sig.stop, "bars_held": 0,
                          "strategy": s, "proba": sig.proba,
                          "game_grade": getattr(sig, "game_grade", ""),
                          "symbol": getattr(ctx, "symbol", "?"),
                          "target": sig.entry + sig.direction * config.RR * sig.risk}
        c.place_market_with_brackets(ctx.account_id, ctx.contract_id,
                                     side=side, size=size,
                                     stop_ticks=stop_ticks, target_ticks=target_ticks,
                                     tick_size=ctx.tick_size)
        log.info("🎯 ENTER %s %s [%s] %d | stop %dt | target %dt (%sR)",
                 stamp, side_txt, s.name, size, stop_ticks, target_ticks, config.RR)
    return trade_state


def run():
    """Live trading loop against the configured broker."""
    client = make_broker()
    client.authenticate()
    acct = client.pick_account(config.ACCOUNT)
    contract = client.get_active_contract(config.SYMBOL)
    # tick size / value come straight from the broker contract — never hardcoded.
    tick_size = float(contract["tickSize"])
    tick_value = float(contract["tickValue"])
    ctx = BotContext(client, acct["id"], contract["id"], tick_size, tick_value)
    names = "+".join(s.name for s in ctx.strategies)
    log.info("✅ %s | %s | %d-min | [%s] | conf≥%.2f | exit: %s | size: %s",
             acct["name"], ctx.contract_id, config.TIMEFRAME_MIN, names,
             config.PROBA_FLOOR, ctx.exit_mode, ctx.sizing_mode)
    log.info("▶ running — Ctrl-C to stop")

    trade_state = None
    rolled_on = dt.datetime.now(dt.timezone.utc).date()
    while True:
        # wait for the next bar close (+2s so the API has published it)
        period = config.TIMEFRAME_MIN * 60
        time.sleep(period - (time.time() % period) + 2)
        try:
            # Follow the roll: the broker API is the source of truth for the
            # front month. Re-resolve once a day while flat so a long-running
            # session moves to the new front contract (and its clean warmup
            # history) without a restart.
            today = dt.datetime.now(dt.timezone.utc).date()
            if today != rolled_on and trade_state is None \
                    and client.open_position(ctx.account_id, ctx.contract_id) is None:
                rolled_on = today
                front = client.get_active_contract(config.SYMBOL)
                if front["id"] != ctx.contract_id:
                    ctx.contract_id = front["id"]
                    ctx.tick_size = float(front["tickSize"])
                    ctx.tick_value = float(front["tickValue"])
                    log.info("🔄 rolled to front contract %s (tick %g, $%g/tick)",
                             front.get("name", front["id"]),
                             ctx.tick_size, ctx.tick_value)

            bars = client.get_bars(ctx.contract_id, config.TIMEFRAME_MIN)
            if len(bars) < config.CTX + 30:    # need >=128 closes + warmup
                continue
            trade_state = handle_bar(ctx, bars, trade_state)
        except Exception as e:        # keep the loop alive on transient errors
            log.warning("⚠️  %s", e)


def _retrain_exit(quick: bool, timesteps: int):
    """Retrain the PPO trailing-exit policy (delegates to train_ppo_exit)."""
    import sys
    from ppo_exit import train_ppo_exit
    sys.argv = (["train_ppo_exit.py", "--timeframe", str(config.TIMEFRAME_MIN)]
                + (["--quick"] if quick else ["--timesteps", str(timesteps)]))
    log.info("retraining PPO exit for %d-min (%s)…", config.TIMEFRAME_MIN,
             "quick" if quick else f"{timesteps} steps")
    train_ppo_exit.main()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="multi-strategy AI futures bot")
    ap.add_argument("--backtest", action="store_true",
                    help="simulate over a local CSV (no API calls)")
    ap.add_argument("--symbol", default=config.SYMBOL,
                    help="backtest symbol (uses data/<symbol>_<timeframe>min.csv)")
    ap.add_argument("--timeframe", type=int, default=None, metavar="MIN",
                    help="bar interval in minutes (default %d). NOTE: the entry "
                         "models and PPO exit are trained on 3-min bars, so other "
                         "values are out of distribution." % config.TIMEFRAME_MIN)
    ap.add_argument("--strategy", nargs="+", metavar="NAME",
                    choices=list(strat.REGISTRY),
                    help="strategies to run: %(choices)s "
                         "(overrides config.ACTIVE_STRATEGIES)")
    ap.add_argument("--start", help="backtest start date (YYYY-MM-DD, inclusive)")
    ap.add_argument("--end", help="backtest end date (YYYY-MM-DD, exclusive)")
    ap.add_argument("--size", type=int,
                    help="fixed contracts per trade (overrides config.SIZE)")
    ap.add_argument("--risk", type=float,
                    help="$ risk per trade; sizes contracts from the stop "
                         "(overrides config.RISK_PER_TRADE). Use instead of --size")
    ap.add_argument("--max-contracts", type=int,
                    help="cap on risk-sized contracts (overrides config.MAX_CONTRACTS)")
    ap.add_argument("--proba-floor", type=float,
                    help="minimum entry confidence (proba) to take a trade, 0–1 "
                         "(overrides config.PROBA_FLOOR)")
    ap.add_argument("--retrain-exit", action="store_true",
                    help="retrain the PPO trailing-exit policy, then exit")
    ap.add_argument("--quick", action="store_true",
                    help="with --retrain-exit: fast smoke train")
    ap.add_argument("--timesteps", type=int, default=600_000,
                    help="with --retrain-exit: PPO training steps")
    args = ap.parse_args()

    if args.size is not None and args.risk is not None:
        raise SystemExit("use either --size or --risk, not both")
    if args.timeframe is not None:
        if args.timeframe < 1:
            raise SystemExit("--timeframe must be >= 1 (minutes)")
        config.TIMEFRAME_MIN = args.timeframe
        config.apply_exit_config()       # load this timeframe's exit shaping
    if args.strategy:
        config.ACTIVE_STRATEGIES = args.strategy
    if args.proba_floor is not None:
        if not 0.0 <= args.proba_floor <= 1.0:
            raise SystemExit("--proba-floor must be between 0 and 1")
        config.PROBA_FLOOR = args.proba_floor
    if args.max_contracts is not None:
        config.MAX_CONTRACTS = args.max_contracts
    if args.size is not None:
        if args.size < 1:
            raise SystemExit("--size must be >= 1")
        config.SIZE = args.size
        config.RISK_PER_TRADE = 0.0      # explicit fixed size disables risk sizing
    if args.risk is not None:
        if args.risk <= 0:
            raise SystemExit("--risk must be > 0")
        config.RISK_PER_TRADE = args.risk

    if args.retrain_exit:
        _retrain_exit(args.quick, args.timesteps)
        raise SystemExit(0)

    if args.backtest:
        import backtest
        backtest.run_backtest(symbol=args.symbol, start=args.start, end=args.end)
        raise SystemExit(0)

    if not config.TOPSTEPX_USERNAME or not config.TOPSTEPX_API_KEY:
        raise SystemExit("missing credentials — copy .env.example to .env and "
                         "set TOPSTEPX_USERNAME / TOPSTEPX_API_KEY")
    run()
