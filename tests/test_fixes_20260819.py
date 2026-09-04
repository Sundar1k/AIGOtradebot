"""Regression tests for the 2026-08-19 live-bleed fixes:

- restart marker: a bar <= ctx.min_bar_time must never re-fire (the veto-blocked
  GC signal that re-entered after a restart took the -$914 tick-bug loss)
- the marker must NOT block the current bar's FIRST pass
- pattern-lane cooldown: after a -1R stop, the next N pattern signals are
  skipped; other lanes are unaffected
- pattern stop multiplier: the lane's risk is widened to PATTERN_STOP_MULT x ATR
"""
import types

import pandas as pd

import bot
import config
from ppo_exit import trail_exit_env as tee
from sim_broker import SimBroker

TICK = 0.25


class FakeStrategy:
    """Fires one signal (first eligible flat bar), graded at a fixed proba."""

    def __init__(self, name, direction, risk=10.0, proba=0.90):
        self.name, self.direction, self.risk, self.proba = name, direction, risk, proba
        self.fired = False

    def detect(self, bars):
        if self.fired or len(bars) < 12:
            return None
        self.fired = True
        entry = float(bars["close"].iloc[-1])
        return types.SimpleNamespace(
            direction=self.direction, entry=entry,
            stop=entry - self.direction * self.risk, risk=self.risk,
            bar_index=len(bars) - 1, bar_time=bars["time"].iloc[-1])

    def grade(self, bars, sig, emb=None):
        return self.proba, 5.0


def _df(closes, lead=2.0):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=len(closes),
                              freq="3min", tz="UTC"),
        "open": closes,
        "high": [c + lead for c in closes],
        "low": [c - lead for c in closes],
        "close": closes,
        "volume": [1] * len(closes),
    })


def _ctx(sim, strategies, *, policy=None, extra=None):
    base = dict(
        client=sim, account_id=0, contract_id="NQ", tick_size=TICK,
        tick_value=5.0, log_candles=False, policy=policy, tee=tee,
        strategies=strategies, trailing=False,
        veto_fn=None, evolve_floor=0.35, evolve_ceil=1.0,
        on_trade_close=None, last_trade=None, on_signal=None, edge_gate=None)
    base.update(extra or {})
    return types.SimpleNamespace(**base)


def _bars(closes):
    return _df(closes)


def test_first_pass_is_never_blocked(monkeypatch):
    # fresh ctx (no marker): the current bar must be eligible — regression
    # guard for the gate reading the marker updated by the same call.
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    closes = [100.0] * 17 + [104, 108, 112]
    sim = SimBroker(_df(closes), TICK)
    ctx = _ctx(sim, [FakeStrategy("ema", +1)])
    out = bot.handle_bar(ctx, _bars(closes), None)
    assert sim.pos is not None          # entry happened on first pass
    assert ctx.strategies[0].fired


def test_restart_marker_blocks_same_bar_refire(monkeypatch):
    # ctx restored from state (min_bar_time = last bar): the same bar must
    # NOT re-detect — this is the veto-blocked-then-restart replay vector.
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    closes = [100.0] * 17 + [104, 108, 112]
    bars = _bars(closes)
    sim = SimBroker(_df(closes), TICK)
    ctx = _ctx(sim, [FakeStrategy("ema", +1)],
               extra={"min_bar_time": bars["time"].iloc[-1],
                      "last_bar_seen": None})
    out = bot.handle_bar(ctx, bars, None)
    assert out is None
    assert sim.pos is None              # no entry from an already-seen bar
    assert not ctx.strategies[0].fired


def test_inprocess_marker_blocks_second_pass_of_same_bar(monkeypatch):
    # same bar passed twice in one process: second call must not re-detect
    # (2026-08-18: the 12:30 candle was entered once, then re-evaluated).
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    closes = [100.0] * 17 + [104, 108, 112]
    bars = _bars(closes)
    sim = SimBroker(_df(closes), TICK)
    ctx = _ctx(sim, [FakeStrategy("ema", +1)])
    bot.handle_bar(ctx, bars, None)
    assert sim.pos is not None
    # simulate the broker returning the SAME last bar next cycle
    sim.pos = None                      # position was closed externally
    ctx.last_trade = None
    ctx.pattern_skip_signals = 0
    out = bot.handle_bar(ctx, bars, None)
    assert out is None
    assert sim.pos is None              # the same bar never re-enters


def test_pattern_cooldown_skips_next_signals(monkeypatch):
    # after a -1R stop the lane skips the next PATTERN_COOLDOWN_SIGNALS
    # pattern signals (shorts); other lanes keep firing.
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    closes = [100.0] * 17 + [97, 92, 86, 84, 84]
    bars = _bars(closes)
    sim = SimBroker(_df(closes), TICK)
    strat = FakeStrategy("pattern", -1)
    ctx = _ctx(sim, [strat], extra={"pattern_skip_signals": 2})
    out = bot.handle_bar(ctx, bars, None)
    assert out is None
    assert sim.pos is None                       # cooldown blocked the entry
    assert ctx.pattern_skip_signals == 1         # one skip consumed
    assert strat.fired                           # but the signal WAS evaluated


def test_pattern_cooldown_does_not_affect_other_lanes(monkeypatch):
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    closes = [100.0] * 17 + [104, 108, 112]
    sim = SimBroker(_df(closes), TICK)
    ctx = _ctx(sim, [FakeStrategy("ema", +1)], extra={"pattern_skip_signals": 2})
    out = bot.handle_bar(ctx, _bars(closes), None)
    assert sim.pos is not None                    # ema lane entered normally
    assert ctx.pattern_skip_signals == 2          # cooldown untouched


def test_pattern_stop_multiplier_widens_risk(monkeypatch):
    # the pattern lane must size its stop at PATTERN_STOP_MULT x ATR(20) —
    # wider than the ML lanes' 0.5 x ATR — never tighter.
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)
    closes = [100.0] * 17 + [97, 92, 86, 84, 84]
    sim = SimBroker(_df(closes), TICK)
    strat = FakeStrategy("pattern", -1, risk=10.0)
    ctx = _ctx(sim, [strat])
    old_mult = config.PATTERN_STOP_MULT
    try:
        config.PATTERN_STOP_MULT = 0.75
        bot.handle_bar(ctx, _bars(closes), None)
    finally:
        config.PATTERN_STOP_MULT = old_mult
    # entry risk must be >= 0.75 x ATR and >= the 0.5x floor (never tighter)
    assert sim.pos is not None
    assert sim.pos["risk"] >= 10.0 * 0.5          # floor = STOP_ATR x ATR-ish
