#!/usr/bin/env python3
"""live_roundtrip_test.py — PROVE the live order path end-to-end.

The bot has NEVER placed a real order (zero ENTER lines in its logs — the
2-week backtest used SimBroker). Per the audit discipline: log/journal
entries are NOT a track record; the exchange is the only ground truth.

This test places ONE micro-contract (MNQ = $2/point, ~$0.50 risk), verifies
the fill + bracket attachment via the exchange API, then closes and verifies
flat. If any step fails, the bot is NOT live-ready.

Usage: ./.venv/bin/python live_roundtrip_test.py
"""
import sys
import time

import config
from broker import make_broker, SIDE

MICRO = "MNQ"          # micro = 1/10 point value, tiny $ risk


def main():
    c = make_broker()
    c.authenticate()
    acct = c.pick_account(config.ACCOUNT)
    aid = acct["id"]
    contract = c.get_active_contract(MICRO)
    cid = contract["id"]
    print(f"account {acct['name']} | {MICRO} -> {cid}", flush=True)

    steps = []
    def step(name, ok, detail=""):
        steps.append(ok)
        print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""), flush=True)

    # 0. pre: flat
    pos = c.any_open_position(aid)
    step("pre-check flat", pos is None, f"pos={pos}")

    # 1. place market order with 2R bracket (4-tick min clamps apply)
    try:
        r = c.place_market_with_brackets(
            aid, cid, side=SIDE["BUY"], size=1, stop_ticks=8, target_ticks=16)
        step("place market+2R bracket", bool(r.get("success")), str(r.get("orderId", r))[:80])
    except Exception as e:
        step("place market+2R bracket", False, str(e)[:120])
        print("\n❌ LIVE ORDER PATH FAILED — bot is NOT live-ready", flush=True)
        return 1
    time.sleep(2)

    # 2. verify position opened
    pos = c.any_open_position(aid)
    step("position opened", pos is not None, f"size={pos.get('size') if pos else '?'}")

    # 3. verify bracket orders attached (stop + target)
    try:
        ords = c._post("/Order/searchOpen", {"accountId": aid}).get("orders", [])
        bracket = [o for o in ords if o.get("contractId") == cid]
        step("bracket orders attached", len(bracket) >= 2,
             f"{len(bracket)} open orders on {MICRO}")
    except Exception as e:
        step("bracket orders attached", False, str(e)[:100])

    # 4. close the position (market) — exercises the exit path
    try:
        r = c.close_position(aid, cid)
        step("close position", bool(r.get("success")), str(r.get("orderId", r))[:80])
    except Exception as e:
        step("close position", False, str(e)[:120])
    time.sleep(2)

    # 5. verify flat + no stray orders (the supervisor's reconcile behavior)
    pos = c.any_open_position(aid)
    step("flat after close", pos is None)
    try:
        ords = c._post("/Order/searchOpen", {"accountId": aid}).get("orders", [])
        stray = [o for o in ords if o.get("contractId") == cid]
        step("no stray orders", len(stray) == 0, f"{len(stray)} remaining")
    except Exception as e:
        step("no stray orders", False, str(e)[:100])

    # 6. balance sanity (should be ~unchanged, micro risk only)
    acct2 = c.pick_account(config.ACCOUNT)
    bal = float(acct2.get("balance", 0))
    step("balance readable", bal > 0, f"${bal:,.2f}")

    ok = all(steps)
    print("=" * 60, flush=True)
    print(f"LIVE ROUND-TRIP: {'✅ ALL PASS — bot is live-ready' if ok else '❌ FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
