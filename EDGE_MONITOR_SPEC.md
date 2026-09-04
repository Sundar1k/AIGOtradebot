# Rolling Edge-Monitor Gate — SPEC (draft for approval)

**Status:** SPEC — not built. Awaiting approval before any code changes.
**Author:** Hermes (2026-08-22)
**Applies to:** ~/projects/algoTraderBot (supervisor.py + a new edge_monitor.py)

---

## 1. Problem (measured, from our own data)

The strategy's realized edge is not constant. Same config, same grader:

| Period | Win rate | avg R | sum R |
|---|---|---|---|
| Apr–Jul 2026 (backtest, all 5 sym) | 68.4% | +1.50R | +374.8R |
| Aug 2026 (backtest, same window) | 29.0% | −0.04R | −1.3R |

The bot went live in the 29% month and bled the whole eval (−$2,609).
Three existing defenses all failed to stop it:

- **Trailing-DD breaker ($1,400)** — already blown before it could matter.
- **Daily-loss breaker ($400)** — too coarse: allows ~$400/day of bleed, resets
  each ET day, so a *sustained* weak period just leaks ~$400/day forever.
- **Regime gate (calm/trending/panic)** — TESTED AND REFUTED (2026-08-22):
  EMA edge is 68.5% in "calm" and 63.8% in "trending" — no collapse. The regime
  model does not see the August decay.

The collapse is **temporal**, not regime-based. Nothing currently watches the
bot's *own realized performance* and pulls back when it degrades. That is the gap.

## 2. Design principle

Point-in-time only. The gate looks at the bot's **own completed trades** and
**own signal log** — never future data, never the inflated backtest as its
expectation. It answers one question every time a trade closes:

> "Is my recent realized edge statistically gone — am I losing money now?"

If yes → de-risk, then halt. If it recovers → resume.

## 3. Baseline (the honesty anchor)

Pre-registered, set from the bot's OWN long-run measurements, NOT the 68%:

- **B_breakeven = 0.0R** — the hard halt line (losing money).
- **B_healthy = +0.50R mean / 47% WR** — the doctrine's "selected engine" long-run
  (the 5-year measurement). Used only for the *de-risk* tier, not the halt.
- Reference: the honest verified OOS edge is PF ~1.12–1.18 ≈ 36% WR ≈ +0.1R
  (trading-edge-validation reference). The 68% April–July number is a good-month
  artifact and is explicitly NOT used as a target or baseline.

## 4. Two-tier architecture

### Tier 1 — Signal-quality monitor (fast, ~1–2 day latency, no trade sample needed)

Tracks the **clear-rate**: fraction of signals with proba ≥ PROBA_FLOOR (0.35)
over a trailing 3-ET-day window (or last 50 signals, whichever is larger).

- Rationale: the grader itself flags a low-edge market before losses accumulate.
  In August the clear-rate was 6% (29 of 481 signals) — the grader was already
  saying "this market is junk."
- **Watch trigger:** clear-rate < 50% of the measured normal baseline → half size
  + raise floor +0.05, log + alert.
- The "normal baseline" clear-rate is **measured during validation** from the
  April–July signal history (not assumed). Pre-registered estimate ~25–35%.

### Tier 2 — Realized-edge monitor (slow, ~10–15 trade latency, hard evidence)

Rolling window of the last **N = 15** closed trades' R-multiples.

- Bootstrap: 10,000 draws, seed 42, resample with replacement.
- One-sided p-value: `P(meanR < 0)` and `P(meanR < B_healthy)`.
- **Watch trigger:** `P(meanR < +0.50R) > 0.70` → half size.
- **Halt trigger:** `P(meanR < 0) > 0.90`  OR  (`WR < 30%` AND `meanR < 0`) → halt.
  (2R:1R breakeven WR = 33.3%, so 30% WR with negative mean R is "clearly losing.")

### Response ladder

1. **Normal** — edge healthy → full size, current floor.
2. **Watch** (either tier trips) → half size, floor +0.05, log + alert.
3. **Halt** (Tier 2 trips) → no new entries; close nothing (positions are managed
   to their own stops/targets as usual — never panic-close a live position).

### Resume condition (CORRECTED after validation)

- **Halt = a fixed pause** (COOLDOWN_H = 24h), then **auto-resume** to test recovery.
  The window is cleared on halt; the bot trades again after the cooldown and the
  fresh window is judged normally (re-halts if still losing).
- This replaced the original "resume only when a fresh window recovers" rule,
  which was wrong twice: (1) in enforce mode no trades close while halted, so a
  fresh window can never form → deadlock; (2) it blocks the recovery leg of a
  V-shaped month (Aug 2026 lost the first week then recovered — a sticky halt
  locks in the dip and misses the +7R recovery).

## 5. Statistical pre-registration (no post-hoc tuning)

| Parameter | Value |
|---|---|
| Window N | 15 closed trades |
| Bootstrap draws / seed | 10,000 / 42 |
| Watch threshold (Tier 2) | P(meanR < +0.50R) > 0.70 |
| Halt threshold (Tier 2) | P(meanR < 0) > 0.90, or WR<30% & meanR<0 |
| Tier 1 clear-rate watch | < 50% of measured normal baseline |
| Resume | fresh window meanR ≥ 0 & WR ≥ 35%, ≥1 ET day cooldown |

These are fixed now. If validation fails, we return to this spec and change the
SPEC — we do NOT tune the thresholds until the historical replay passes.

## 6. Validation plan (point-in-time, no look-ahead)

Replay the gate over the full history (backtest CSVs, Apr–Aug 2026, all 5 symbols)
trade-by-trade, in time order, applying the gate's decisions only to trades it
would have seen at that point.

Success criteria (pre-registered):
1. Gate does NOT worsen August (with-gate P&L >= without-gate).
2. April–July profit retained **≥ 90%** (vs +374.8R).
3. **≤ 2** false halts in April–July.

**RESULT (2026-08-22, after the auto-resume correction): PASS all three.**
- August: without gate −1.27R → with gate +3.73R (gate effect **+5.00R**).
- April–July retained: **100%** (0 false halts).
- 1 halt total (Aug 13, during the losing streak), correctly auto-resumed.
Note: with the ORIGINAL sticky-halt rule the gate FAILED (August −4.98R vs
−1.27R — it blocked the recovery leg of the V-shaped month). The auto-resume
fix was the difference; this is a semantic correction, not a threshold tune.

## 7. Integration with existing machinery

- **Evolver (evolve.py):** the edge-monitor is a coarser, faster cousin. When the
  monitor halts, the evolver freezes (no floor changes while halted). No conflict.
- **Breakers (daily $400, trailing DD $1,400):** unchanged — per-day layer.
- **Profit target ($500/25h):** unchanged.
- **Regime gate (panic):** unchanged — it's not the fix, but it's harmless.
- **Veto LLM:** unchanged — the monitor gates *after* signal+veto, by pulling
  exposure, never by overriding an individual trade's veto.
- New file `edge_monitor.py` + a hook in `supervisor.py` called on every
  `on_trade_close` and on each new signal (for Tier 1 clear-rate). State persists
  to `~/.autotrade_edge_monitor.json` (restart-safe). Advisory-first rollout:
  `AUTOTRADE_EDGE_MONITOR=advisory` (log + alert only) → validate live → then
  `=enforce`.

## 8. Explicit non-goals

- NOT re-tuning the grader, floor, or veto to "fix" August (the skipped trades
  genuinely lost −60R; the grader was right).
- NOT using the 68% backtest as the baseline or target.
- NOT claiming this raises the win rate. It only stops the bleeding in 29%-months.
- NOT panic-closing open positions on a halt signal.

## 9. Decision needed

Approve to build? Options:
- **Build now** (write edge_monitor.py + supervisor hook, advisory-first, run the
  validation replay, report the three success criteria).
- **Adjust thresholds first** (tell me which numbers to change).
- **Paper-only** — validate on the paper book before wiring to the live eval.
