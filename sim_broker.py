#!/usr/bin/env python3
"""sim_broker.py — a CSV-driven broker that mimics TopstepXClient for backtests.

It implements the order/position methods the bot calls, but fills against
historical bars instead of the API:

  * market entries fill at the signal bar's close
  * a protective stop / trailing stop / fixed-target are simulated bar-by-bar
  * trailing stops follow the best price natively (ratchet-only); the PPO can
    tighten the follow distance via modify_trail_price
  * every closed trade is recorded for the backtest summary

Conservative fills: if a bar's range touches both stop and target, the stop is
assumed first. Open positions at end-of-data are closed at the last close.
"""
from dataclasses import dataclass

from broker_base import SIDE, OrderRouter


@dataclass
class Trade:
    strategy: str
    direction: int          # +1 long / -1 short
    entry_time: object
    entry: float
    exit_time: object
    exit: float
    risk: float
    r: float                # realized R-multiple (NET of slippage + commission)
    mfe_r: float            # max favorable excursion, in R (best the trade ever showed)
    bars_held: int
    reason: str             # "stop" | "target" | "eod"
    slippage_cost: float = 0.0   # adverse slippage drag, in R (entry + exit)
    commission: float = 0.0      # commission drag, in R (entry + exit)


class SimBroker(OrderRouter):
    """An OrderRouter that fills against a bars DataFrame — the backtest broker.

    Cost model (opt-in, so low-level unit tests keep gross, exact-R fills):
      * slip_ticks          — adverse fill per side in ticks (entry + exit)
      * commission_per_side — $ per contract per side (needs tick_value to
                              convert to R via the trade's dollar risk)
    When enabled, every realized R is reported NET of both.
    """

    def __init__(self, df, tick_size: float, *, tick_value: float = 0.0,
                 slip_ticks: float = 0.0, commission_per_side: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.tick = tick_size
        self.tick_value = tick_value
        self.slip_ticks = slip_ticks
        self.commission_per_side = commission_per_side
        self.cursor = 0
        self.pos = None        # open-position dict, or None
        self.trades = []

    # ── backtest driver hooks ──────────────────────────────────────────
    def set_bar(self, i: int):
        self.cursor = i

    def process_exits(self):
        """Test the current bar against the working stop/target, close if hit,
        else advance a native trailing stop. Called once per bar before the
        bot acts."""
        if self.pos is None:
            return
        p, i = self.pos, self.cursor
        bar = self.df.iloc[i]
        sign, hi, lo = p["sign"], bar["high"], bar["low"]

        # track max favorable excursion (the best price seen this bar)
        p["best"] = max(p["best"], hi) if sign > 0 else min(p["best"], lo)

        hit_stop = (lo <= p["stop"]) if sign > 0 else (hi >= p["stop"])
        if hit_stop:
            self._close(p["stop"], i, "stop")
            return
        if p.get("target") is not None:
            hit_tp = (hi >= p["target"]) if sign > 0 else (lo <= p["target"])
            if hit_tp:
                self._close(p["target"], i, "target")
                return

        p["bars_held"] += 1
        if p.get("trailing"):          # native trailing stop ratchets to best price
            dist = p["trail_ticks"] * self.tick
            p["stop"] = (max(p["stop"], p["best"] - dist) if sign > 0
                         else min(p["stop"], p["best"] + dist))

    def close_open(self):
        """Force-close any open position at the last bar's close (end of data)."""
        if self.pos is not None:
            self._close(float(self.df.iloc[self.cursor]["close"]),
                        self.cursor, "eod")

    def tag_strategy(self, name: str):
        if self.pos is not None and self.pos.get("strategy") is None:
            self.pos["strategy"] = name

    def _cost_r(self, risk: float):
        """(slippage_R, commission_R) for a round trip, in R. Zero when the cost
        model is off (default)."""
        slip_r = comm_r = 0.0
        if risk > 0 and self.slip_ticks > 0 and self.tick > 0:
            slip_r = (2.0 * self.slip_ticks * self.tick) / risk
        if (risk > 0 and self.commission_per_side > 0
                and self.tick_value > 0 and self.tick > 0):
            usd_per_point = self.tick_value / self.tick
            comm_r = (2.0 * self.commission_per_side) / (risk * usd_per_point)
        return slip_r, comm_r

    def _close(self, price, i, reason):
        p = self.pos
        gross_r = p["sign"] * (price - p["entry"]) / p["risk"]
        mfe_r = p["sign"] * (p["best"] - p["entry"]) / p["risk"]
        slip_r, comm_r = self._cost_r(p["risk"])
        self.trades.append(Trade(
            strategy=p.get("strategy") or "?", direction=p["sign"],
            entry_time=self.df.iloc[p["entry_idx"]]["time"], entry=p["entry"],
            exit_time=self.df.iloc[i]["time"], exit=price, risk=p["risk"],
            r=float(gross_r - slip_r - comm_r), mfe_r=float(mfe_r),
            bars_held=p["bars_held"], reason=reason,
            slippage_cost=float(slip_r), commission=float(comm_r)))
        self.pos = None

    # ── TopstepXClient-compatible surface ──────────────────────────────
    def open_position(self, account_id, contract_id):
        if self.pos is None:
            return None
        return {"contractId": contract_id, "size": self.pos["size"],
                "averagePrice": self.pos["entry"],
                "type": 1 if self.pos["sign"] > 0 else 2}

    def _enter(self, side, size, stop_ticks, *, trailing, target_ticks=None):
        sign = 1 if side == SIDE["BUY"] else -1
        entry = float(self.df.iloc[self.cursor]["close"])
        risk = stop_ticks * self.tick
        self.pos = {
            "sign": sign, "size": size, "entry": entry, "entry_idx": self.cursor,
            "risk": risk, "stop": entry - sign * risk, "bars_held": 0,
            "best": entry, "trailing": trailing, "trail_ticks": stop_ticks,
            "target": (entry + sign * target_ticks * self.tick
                       if target_ticks else None),
            "strategy": None,
        }

    def place_market_with_stop(self, account_id, contract_id, *, side, size,
                               stop_ticks, tick_size=None):
        self._enter(side, size, stop_ticks, trailing=False)
        return {"success": True}

    def place_market_with_trail(self, account_id, contract_id, *, side, size, trail_ticks):
        self._enter(side, size, trail_ticks, trailing=True)
        return {"success": True}

    def place_market_with_brackets(self, account_id, contract_id, *, side, size,
                                   stop_ticks, target_ticks, tick_size=None):
        self._enter(side, size, stop_ticks, trailing=False, target_ticks=target_ticks)
        return {"success": True}

    def working_stop_order(self, account_id, contract_id):
        if self.pos is None:
            return None
        return {"id": 1, "stopPrice": self.pos["stop"],
                "type": 5 if self.pos["trailing"] else 4}

    def modify_stop_price(self, account_id, order_id, stop_price):
        if self.pos is not None:
            self.pos["stop"] = stop_price
        return {"success": True}

    def modify_trail_price(self, account_id, order_id, trail_price):
        if self.pos is not None:
            self.pos["trail_ticks"] = max(1, round(trail_price / self.tick))
        return {"success": True}

    def cancel_order(self, account_id, order_id):
        return {"success": True}      # no resting broker orders to orphan in the sim

    def cancel_orders(self, account_id, contract_id):
        return 0                      # the sim has no resting broker orders to sweep

    def close_position(self, account_id, contract_id, price=None):
        # Market-close: fill at `price` (the enforced trailed-SL level) if given,
        # else the current bar's close. Records the trade with reason "trail".
        if self.pos is not None:
            fill = price if price is not None else float(self.df.iloc[self.cursor]["close"])
            self._close(fill, self.cursor, "trail")
        return {"success": True}
