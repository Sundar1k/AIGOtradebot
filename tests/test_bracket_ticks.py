"""Bracket-tick signing — the order-rejection fix.

TopstepX wants SL/TP ticks SIGNED relative to the fill: a long's stop sits below
(negative), a short's above (positive); TP is the mirror. Magnitude is clamped to
the 4-tick broker minimum. (Bug: we sent positive for both → "Invalid stop loss
ticks (57). Ticks should be less than zero when longing.")
"""
from broker import SIDE, MIN_BRACKET_TICKS, _stop_bracket_ticks, _target_bracket_ticks

LONG, SHORT = SIDE["BUY"], SIDE["SELL"]


def test_long_stop_is_negative():
    assert _stop_bracket_ticks(LONG, 57) == -57      # the exact rejected case


def test_short_stop_is_positive():
    assert _stop_bracket_ticks(SHORT, 57) == 57


def test_long_target_is_positive():
    assert _target_bracket_ticks(LONG, 114) == 114


def test_short_target_is_negative():
    assert _target_bracket_ticks(SHORT, 114) == -114


def test_min_tick_clamp_both_sides():
    assert _stop_bracket_ticks(LONG, 2) == -MIN_BRACKET_TICKS    # too-tight long
    assert _stop_bracket_ticks(SHORT, 0) == MIN_BRACKET_TICKS    # zero short
    assert _target_bracket_ticks(LONG, 1) == MIN_BRACKET_TICKS


def test_magnitude_uses_abs():
    # a negative tick count in must not flip the sign convention
    assert _stop_bracket_ticks(LONG, -57) == -57
    assert _stop_bracket_ticks(SHORT, -57) == 57
