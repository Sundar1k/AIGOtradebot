"""Position sizing: fixed vs risk-based, with the MAX_CONTRACTS cap and a 1-lot
floor. size = min(MAX, RISK / (stop_ticks × tick_value))."""
import types

import config
import bot


def _ctx(tick_value):
    return types.SimpleNamespace(tick_value=tick_value)


def test_fixed_size_when_no_risk():
    config.RISK_PER_TRADE = 0.0
    config.SIZE = 3
    config.MAX_CONTRACTS = 10
    assert bot.position_size(_ctx(5.0), stop_ticks=40) == 3


def test_risk_based_size():
    config.RISK_PER_TRADE = 500.0
    config.MAX_CONTRACTS = 20
    # 500 / (40 ticks × $5/tick = $200 risk/contract) = 2 contracts
    assert bot.position_size(_ctx(5.0), stop_ticks=40) == 2


def test_risk_based_capped_at_max():
    config.RISK_PER_TRADE = 10_000.0
    config.MAX_CONTRACTS = 5
    assert bot.position_size(_ctx(5.0), stop_ticks=40) == 5     # would be 50 → capped


def test_risk_based_floor_is_one():
    config.RISK_PER_TRADE = 50.0
    config.MAX_CONTRACTS = 10
    # 50 / $200 = 0.25 → floored to 1 (never trade zero on a valid signal)
    assert bot.position_size(_ctx(5.0), stop_ticks=40) == 1


def test_falls_back_to_fixed_without_tick_value():
    config.RISK_PER_TRADE = 500.0
    config.SIZE = 2
    config.MAX_CONTRACTS = 10
    assert bot.position_size(_ctx(0.0), stop_ticks=40) == 2     # no $/tick → fixed
