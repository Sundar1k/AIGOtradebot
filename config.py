#!/usr/bin/env python3
"""config.py — all bot settings in one place.

Credentials can also be supplied via the TOPSTEPX_USERNAME / TOPSTEPX_API_KEY /
TOPSTEPX_ACCOUNT environment variables (those win over the values here).
"""
import os

try:
    from dotenv import load_dotenv
except ImportError:                       # dotenv optional (not needed for backtest)
    def load_dotenv(*a, **k):
        return False

HERE = os.path.dirname(os.path.abspath(__file__))

# ── credentials ────────────────────────────────────────────────────────
# Stored in a .env file (gitignored) — copy .env.example to .env and fill in.
#   TOPSTEPX_USERNAME=...   TOPSTEPX_API_KEY=...   TOPSTEPX_ACCOUNT=...
# Real environment variables still win over .env.
load_dotenv(os.path.join(HERE, ".env"))
TOPSTEPX_USERNAME = os.environ.get("TOPSTEPX_USERNAME", "")
TOPSTEPX_API_KEY  = os.environ.get("TOPSTEPX_API_KEY", "")
ACCOUNT = os.environ.get("TOPSTEPX_ACCOUNT", "")   # "" = first tradable account

# ── broker ─────────────────────────────────────────────────────────────
BROKER = "topstepx"        # which BrokerClient to use (see broker.make_broker)

# ── market / sizing ────────────────────────────────────────────────────
API_BASE = "https://api.topstepx.com/api"
SYMBOL = "NQ"
TIMEFRAME_MIN = 3                # bar interval in minutes (CLI: --timeframe)
TRAINED_TIMEFRAME_MIN = 3        # the interval the models/PPO were trained on; other
#                                 timeframes run but are out of distribution
SIZE = 1

# Micro contracts trade the SAME bars as their full-size parent (so the models
# apply directly) at 1/10 the point value. Map each micro → its parent.
MICRO_PARENT = {"MNQ": "NQ", "MES": "ES", "M2K": "RTY", "MYM": "YM",
                "MGC": "GC", "MCL": "CL"}


def base_symbol(symbol: str) -> str:
    """The full-size parent for a micro (MNQ→NQ), else the symbol itself. Used
    for model feature derivation and choosing the data/<sym>_3min.csv file."""
    return MICRO_PARENT.get(symbol, symbol)


# Tick size and tick value are NOT hard-coded — they come from the broker API
# (/Contract/search → tickSize, tickValue) for both live and backtests.

# Full-size tickers the shipped entry models were trained on. Micros map to these
# via base_symbol(), so they're in-distribution too.
TRAINED_SYMBOLS = ("NQ", "ES", "RTY", "YM", "GC")
# Live trading universe — the symbols the supervisor scans each cycle.
# Must be a subset of TRAINED_SYMBOLS (models + veto know these). The user
# chose all 5 (2026-08-17). Env AUTOTRADE_SYMBOLS="NQ,ES" overrides.
TRADE_SYMBOLS = tuple(
    s.strip().upper()
    for s in os.environ.get("AUTOTRADE_SYMBOLS", ",".join(TRAINED_SYMBOLS)).split(",")
    if s.strip()
)

# Position sizing — use either a fixed SIZE or RISK_PER_TRADE (not both).
#   SIZE           fixed contracts per trade.
#   RISK_PER_TRADE if > 0, size from the stop instead:
#                  contracts = floor(RISK_PER_TRADE / (stop_ticks × tick_value)),
#                  clamped to [1, MAX_CONTRACTS]. 0 = use fixed SIZE.
RISK_PER_TRADE = 0.0       # $ risked per trade (0 = off)
MAX_CONTRACTS = 10         # cap on risk-sized contracts

# Book-wide position cap — the TopstepX eval subscription allows only ONE
# open position at a time across ALL symbols (platform-enforced; a second
# entry gets rejected by the broker). The supervisor keeps entries under
# this: while the book is at cap, only the owning symbol manages its
# position, the others skip entries. NOTE: the gate surfaces the book via
# any_open_position() (first position only), so values > 1 need a counting
# implementation — 1 is the only value the TopstepX eval can use.
MAX_OPEN_POSITIONS = 1     # max simultaneous positions across the whole book

# ── strategy selection ─────────────────────────────────────────────────
# Which strategies run. One name = single strategy; list both to run them
# together — when both fire on the same bar, the higher-proba signal is taken.
# Pattern lane PAUSED 2026-08-19 (live A/B): 10 live [pattern] trades were
# 2W/8L (-3.18R), ALL at proba 0.400 — the bottom of the confidence band —
# while its 2-month backtest edge is marginal (+4.4R plain, +8.1R with
# cooldown). The validated engine is ema + orb. Re-enable with one line
# after 2 weeks of clean NQ-only + cooldown data proves (or disproves) it.
ACTIVE_STRATEGIES = ["ema"]              # any of: supertrend, ema, keltner, bos, pattern, orb, gann
# gann REMOVED 2026-08-22: no 3-min bundle (models/gann_chronos.joblib missing;
# only the 15-min exists) → grade() returns the constant (0.40, 0.0) → proba
# gate is a formality (every flip passes at exactly 0.40) → both live losses
# today were gann (2x -1R, -$602.56, daily-loss halt). The new startup
# validation (base.py) also refuses to start with a bundle-less active
# strategy. Re-enable only once models/gann_chronos.joblib is trained.
                                                 # 2026-08-21: +gann activated — Phase 4 winner
                                                 # (OOS: -0.097R -> -0.040R, PF 0.94, 55k trades)
# Per-symbol EXTRA strategies (2026-08-18 screen, Apr-Jun window): orb is
# quality on NQ/ES/GC (54-60% WR, PF 2.4-3.0) but a LOSER on YM (32.4%, PF
# 0.96) and thin on RTY (47.2%, PF 1.79) — enable it only where it wins.
ORB_SYMBOLS = {"NQ", "ES", "GC"}

# Pattern lane symbol universe (2026-08-19, after live bleed): the [pattern]
# lane's ONLY backtested-positive symbol is NQ (PF 1.22, avgR +0.14, n=232).
# ES is a backtested NET LOSER (PF 0.84, avgR -0.09, n=293) and RTY/YM/GC have
# zero pattern backtest rows — live 08-18/19 night: 4 of 5 signals on RTY/ES,
# 3 of those lost -1R each. Lane now fires only on symbols with a proven edge.
PATTERN_SYMBOLS = {"NQ"}
# ⛔ DOCTRINE — PREDICT LESS, SELECT BETTER (see DOCTRINE.md)
# 70% direction winrate is unreachable on this data (needs r≈0.59;
# best measured r≈0.27). The edge is selection + 2R payoff (47.2% WR /
# +0.58R / PF 2.11). Do NOT add "min winrate" knobs or raise model
# pressure chasing direction. Sanctioned levers: quality-band gating
# (AUTOTRADE_QUALITY_MIN), vol/regime gates, validated flow changes.
PROBA_FLOOR = 0.40          # enter only when a strategy grades its signal >= this
                            # 2026-09-01 USER OVERRIDE: raised 0.35 -> 0.40 per the
                            # selection-evolution candidate (0.40/0.65/chop2.0).
                            # NOTE: that candidate was INCONCLUSIVE (140<150
                            # hold-out trades) per the pre-registered rule; this
                            # config is user-mandated, NOT validated. Revert to
                            # 0.35 if live results diverge from the +0.42R hold-out.
# Confidence CEILING: skip signals with proba ABOVE this. The model's
# overconfident band (proba >= ~0.50) has been toxic for years (2022: 7% WR,
# 2023: 5% WR at proba >= 0.60; recent 2026 window: 38.5%/16.7% at >= 0.50)
# while 0.35-0.50 is the sweet spot (WR 56-60%). Learned from the miss
# analysis (2026-08-17, 5y NQ backtest): ceiling 0.50 -> WR 52.9->55.6%,
# expectancy +0.577R -> +0.655R, better in every year. 1.0 = disabled.
# 2026-09-01 USER OVERRIDE: ceiling raised to 0.65 via the service env
# (AUTOTRADE_PROBA_CEIL=0.65) with the selection-evolution candidate — the
# higher-proba zone is NOT toxic inside the 09:30-12:00 window (all-day
# toxicity only). User-mandated, NOT validated (INCONCLUSIVE 140<150).
PROBA_CEIL = float(os.environ.get("AUTOTRADE_PROBA_CEIL", "0.65"))

# Pattern lane stop/cooldown (2026-08-19 live-bleed fixes):
#   PATTERN_STOP_MULT — the pattern lane trades 30m patterns; the base ML stop
#   (STOP_ATR × ATR20 of the 3-min bars) was suspected of being too tight for
#   30m candles. Grid backtest (NQ, Apr-Jun 2026, 6 cells) DISPROVED that:
#   0.75x and 1.0x multipliers did NOT help (pattern sumR +6.8 / -5.7 vs +4.4
#   at 0.5x; book PF best at 0.5x). The live bleed came from non-NQ symbols
#   (now excluded via PATTERN_SYMBOLS) + the GC tick bug + sample noise, NOT
#   stop width. 0.5 stays — matching the trained trade definition.
#   PATTERN_COOLDOWN_SIGNALS — after a -1R (or worse) stop, skip the next N
#   pattern signals. VALIDATED (NQ Apr-Jun 2026): book sumR +121.4 -> +128.1,
#   PF 1.55 -> 1.80, avgR +0.312 -> +0.421 at mult 0.5; pattern lane itself
#   +4.4R -> +8.1R. Combined with PPO exit (USE_PPO_EXIT): +180.5R / PF 1.81.
PATTERN_STOP_MULT = float(os.environ.get("AUTOTRADE_PATTERN_STOP_MULT", "0.5"))
PATTERN_COOLDOWN_SIGNALS = int(os.environ.get("AUTOTRADE_PATTERN_COOLDOWN", "2"))

# ── shared trade definition (matches how the models scored trades) ─────
ATR_P     = 20              # ATR period for the protective stop
STOP_ATR  = 0.5            # stop = STOP_ATR * ATR(ATR_P) from entry
ADX_P     = 14
ADX_SLOPE = 5              # bars for the adx-slope feature
MIN_GAP   = 20             # (reference) min bars between signals
CTX       = 128            # Chronos context window

# ── backtest / paper cost model (makes the sim HONEST — live fills carry these) ─
# SLIPPAGE_TICKS: adverse fill per side (entry + exit) in TICKS. A market order
#   can never fill at the exact signal-bar close; 1 tick is the minimum adverse
#   move. The backtest/paper SimBroker subtracts 2×this (round trip) from every
#   trade's realized R. 0 = disable (gross fills, used by low-level unit tests).
SLIPPAGE_TICKS = int(os.environ.get("AUTOTRADE_SLIPPAGE_TICKS", "1"))
# COMMISSION_PER_SIDE_USD: broker commission per contract per SIDE ($). Set to
#   the real all-in TopstepX micro rate to charge 2×this per round trip. 0 = off.
COMMISSION_PER_SIDE_USD = float(os.environ.get("AUTOTRADE_COMMISSION_PER_SIDE_USD", "0.0"))
# JUMP_ATR_MULT: jump filter — skip a signal when the current bar (or one of the
#   last 2) is a "jump": a |close-to-close| move larger than JUMP_ATR_MULT ×
#   ATR(ATR_P). Large jump moves are theoretically unpredictable (Aleti /
#   Bollerslev / Siggaard, "Intraday Factor Zoo", drop them outright), so
#   skipping them is a SELECTION filter, not a direction bet. 0 = off.
JUMP_ATR_MULT = float(os.environ.get("AUTOTRADE_JUMP_ATR_MULT", "0.0"))

# SuperTrend strategy params
ST_PERIOD, ST_MULT = 10, 3.0
# EMA-cross strategy params
EMA_FAST, EMA_SLOW = 9, 20
SLOW_SLOPE_K = 5
ADX_GATE = 18.0            # only fire EMA crosses when ADX >= this (trend gate)
# Keltner-channel breakout strategy params
KC_LEN, KC_MULT, KC_ATR_P = 20, 1.5, 20
KC_ADX_THRESH = 20.0      # only fire Keltner breakouts when ADX >= this
KC_MID_SLOPE_K = 5
# Break-of-structure strategy params
SWING_K = 2               # fractal half-width for confirmed swings
# Opening-range-breakout strategy params
ORB_BARS = 5              # bars in the opening range (5 × 3-min = 15 min)
ORB_OPEN_MIN = 9 * 60 + 30   # session open = 09:30 in ORB_TZ (minutes from midnight)
ORB_TZ = "America/New_York"
ORB_ADX_GATE = 18.0       # only fire ORB breakouts when ADX >= this
ORB_CLOSE_MIN = 16 * 60   # stop firing ORB breakouts at 16:00 ET — the opening
#   range stays mathematically "active" until midnight ET, but a 09:30 range is
#   stale by the evening; gate entries to the RTH session [~09:45, 16:00) ET so
#   the bot doesn't take low-quality overnight breakouts on the morning range.

# ── models ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(HERE, "models")
FFM_COLUMNS_PATH = os.path.join(MODELS_DIR, "ffm_feature_columns.json")

# ── exit ───────────────────────────────────────────────────────────────
# PPO trailing exit. A comment here once claimed PPO exit was "validated OFF
# (over-tightening)" — that was WRONG for this window: grid backtest (NQ,
# Apr-Jun 2026, same params) shows PPO ON = +180.5R / PF 1.81 vs fixed-2R
# +128.1R / PF 1.80 (pattern lane +27.8R vs +8.1R; ema +152.7R vs +120.0R).
# The live "over-tightening" was the 3-min ATR stop + non-NQ symbols, not the
# trail. PPO ON is the validated setting (2026-08-19). Fixed 2R bracket (RR
# below) remains the fallback when no policy file exists.
USE_PPO_EXIT = True
# False (default) = PPO trailing: the policy reprices the stop each bar via
#   /Order/modify — this is what the policy is trained for, so the trail is fully
#   policy-driven (can loosen or tighten the trail distance).
# True = broker-native trailing stop that the PPO can only *tighten* — simpler
#   intra-bar protection, but the policy can't widen, so it mostly sits idle.
USE_TRAILING_STOP = False
POLICY_PATH = os.path.join(HERE, "ppo_exit", "policies", "ppo_trail_exit.npz")
RR = 2.0                    # fixed-R take-profit fallback (no PPO policy)


def policy_path():
    """The PPO policy for the active timeframe: ppo_trail_exit.npz at the trained
    3-min default, ppo_trail_exit_<tf>min.npz otherwise (e.g. ..._1min.npz). The
    exit is retrained per timeframe — bar geometry (ATR, MAX_HOLD bars) differs."""
    base, ext = os.path.splitext(POLICY_PATH)
    return POLICY_PATH if TIMEFRAME_MIN == TRAINED_TIMEFRAME_MIN \
        else f"{base}_{TIMEFRAME_MIN}min{ext}"

# Trailing exit shape:
#   ACTIVATE_R — hold the initial stop (1R = STOP_ATR×ATR) until the trade's peak
#                reaches this many R; only then start trailing. Lets winners run
#                through early pullbacks before we protect.
#   GIVEBACK_R — once trailing, the stop never sits more than this many R below the
#                running peak (the PPO may trail tighter, never looser).
# e.g. ACTIVATE_R=2, GIVEBACK_R=0.75: risk 1R until +2R, then lock in ≥ +1.25R and
# ride, giving back at most 0.75R from the best point.
ACTIVATE_R = 2.0
GIVEBACK_R = 0.75

# Asymmetric PPO-exit reward (research item #11, DRL Multipair paper): penalize
# losing mark-to-market moves more than we reward winning ones, so the policy is
# more loss-averse. 1.0 = symmetric (current). >1 = loss-averse.
PPO_LOSS_PENALTY = float(os.environ.get("AUTOTRADE_PPO_LOSS_PENALTY", "1.0"))

# Per-timeframe exit shaping. The best ACTIVATE_R / GIVEBACK_R / STOP_ATR differ by
# timeframe (1-min vs 3-min), so they live in exit_configs.json keyed by minutes and
# are applied for the active timeframe. These knobs are read at runtime by BOTH the
# live exit (exit_manager) and the training sim (trail_exit_env), so training=live.
# Tune with optimize_exit.py; after changing a timeframe's config, retrain its policy
# (train_ppo_exit --timeframe N) so the PPO matches.
EXIT_CONFIGS_PATH = os.path.join(HERE, "ppo_exit", "exit_configs.json")


def apply_exit_config(tf=None):
    """Apply exit_configs.json[<tf>] to ACTIVATE_R / GIVEBACK_R / STOP_ATR. No-op
    (keeps the module defaults) if the file or the timeframe key is missing. Returns
    the applied dict, or None."""
    global ACTIVATE_R, GIVEBACK_R, STOP_ATR
    import json
    tf = TIMEFRAME_MIN if tf is None else tf
    try:
        with open(EXIT_CONFIGS_PATH) as f:
            cfg = json.load(f).get(str(tf))
    except (FileNotFoundError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    if "ACTIVATE_R" in cfg:
        ACTIVATE_R = float(cfg["ACTIVATE_R"])
    if "GIVEBACK_R" in cfg:
        GIVEBACK_R = float(cfg["GIVEBACK_R"])
    if "STOP_ATR" in cfg:
        STOP_ATR = float(cfg["STOP_ATR"])
    return {"ACTIVATE_R": ACTIVATE_R, "GIVEBACK_R": GIVEBACK_R, "STOP_ATR": STOP_ATR}


apply_exit_config()             # apply the default timeframe's saved config at import
