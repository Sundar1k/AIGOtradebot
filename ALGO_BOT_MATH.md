# algoTraderBot — Complete Math Reference
(Generated 2026-08-28 from graph + source. All 10 layers in execution order.)

## Constants (config.py)
```
TIMEFRAME = 3 min
SYMBOLS   = {NQ, ES, RTY, YM, GC}
RISK_PCT  = 0.005..0.02          (per-trade risk fraction)
P_MIN     = 0.35                  (signal probability gate)
K_STOP    = 1.5..3.0              (stop in ATR units)
K_TARGET  = 2 · K_STOP            (≥ 2:1 reward:risk)
K_TRAIL   = 1.0..2.0              (trail in ATR units)
MAX_BARS  = 20                    (time stop = 1h on 3m)
EOD_ET    = 16:55                 (flatten at close)
P_NO_THRESH = 0.95                (fast-veto reject-only threshold)
MAX_POSITIONS = 1..N
MIN_WR    = 0.40                  (edge monitor block threshold)
K_block   = 3..5                  (consecutive epochs to block)
HB_TIMEOUT = 60s                  (heartbeat watchdog)
```

---

## Layer 1 — Indicators
```
gain_i     = max(C_i − C_{i−1}, 0)
loss_i     = max(C_{i−1} − C_i, 0)
avg_gain_t = (avg_gain_{t−1}·13 + gain_t) / 14
avg_loss_t = (avg_loss_{t−1}·13 + loss_t) / 14
RS_t       = avg_gain_t / max(avg_loss_t, ε)
RSI_t      = 100 − 100 / (1 + RS_t)

k_n        = 2 / (n + 1)
EMA_n(t)   = k_n·C_t + (1 − k_n)·EMA_n(t−1)
side_t     = 1[EMA10_t ≥ EMA30_t]

hh_t       = max(H_{t−13:t})
ll_t       = min(L_{t−13:t})
K_t        = 100·(C_t − ll_t) / max(hh_t − ll_t, ε)
dir_K_t    = 1[K_t ≥ K_{t−1}]

TR_t       = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|)
ATR_t      = (ATR_{t−1}·19 + TR_t) / 20

x_t        = [onehot(sym)∈ℝ^5, RSI_t, side_t, K_t, dir_K_t, ATR_t, Score_t] ∈ ℝ^12
```

## Layer 2 — Signal probability
```
Score_t    = sign(EMA10_t − EMA30_t) · intensity
proba_t    = f_XGB(x_t) ∈ [0, 1]

intent_t   = 0                                       if proba_t < P_MIN
           = sign(EMA10_t − EMA30_t) ∈ {−1, +1}      if proba_t ≥ P_MIN
```

## Layer 3 — Veto (env-gated, currently OFF)
```
veto_7B_t     = LLM.approve(x_t) ∈ {0, 1}                   # 120s timeout
p_no_t        = XGB_student.p_no(x_t) ∈ [0, 1]
pre_filter_t  = 1[p_no_t ≥ P_NO_THRESH]                     # reject-only

effective_veto_t = 0                                          (default OFF)
                 = 1 − pre_filter_t                          if AUTOTRADE_FAST_VETO=1
                 = 1 − veto_7B_t                              if AUTOTRADE_VETO_MODE=gate
```

## Layer 4 — Position sizing
```
risk_$_t        = equity_t · RISK_PCT
stop_distance_t = K_STOP · ATR_t
contracts_raw_t = ⌊ risk_$_t / (stop_distance_t · point_value) ⌋

atr_pct_t       = percentile_rank(ATR_t, window=252)
floor_t         = floor_low + (floor_high − floor_low)·atr_pct_t

contracts_t     = max(contracts_raw_t, floor_t) · 1[contracts_raw_t ≥ 1]
```

## Layer 5 — Entry order
```
entry_t    = C_t
stop_t     = C_t − intent_t · K_STOP · ATR_t
target_t   = C_t + intent_t · K_TARGET · ATR_t
qty_t      = contracts_t

A_t = intent_t · contracts_t · 1[
          proba_t ≥ P_MIN
        ∧ effective_veto_t = 1
        ∧ ¬blocked_t
        ∧ |open_positions| < MAX_POSITIONS
      ]
```

## Layer 6 — In-trade management
```
bars_held = t − entry_time
state_t   = (intent, bars_held, unrealized_R_t)

new_stop_t = best(stop_t, C_t − intent·K_TRAIL·ATR_t)        if intent = +1
             worst(stop_t, C_t + intent·K_TRAIL·ATR_t)       if intent = −1
stop_t    := new_stop_t

exit_t = 1[
            C_t crossed stop_t
          ∨ C_t crossed target_t
          ∨ bars_held ≥ MAX_BARS
          ∨ now_et() ≥ EOD_ET
        ]
```

## Layer 7 — P&L in R-multiples
```
slip_R_t  = 0.025·qty·tick_value / risk_$_t
comm_R_t  = 0.02·qty·round_trip_cost / risk_$_t
gross_R_t = (exit_t − entry_t)·intent_t·qty·point_value / risk_$_t
net_R_t   = gross_R_t − slip_R_t − comm_R_t

cost_per_trade_R = 𝔼[slip_R + comm_R] ≈ 0.07R
```

## Layer 8 — Expectancy
```
W_t       = #{i ≤ t : net_R_i > 0}
L_t       = #{i ≤ t : net_R_i < 0}
N_t       = W_t + L_t
winrate_t = W_t / max(N_t, 1)

avg_win_R_t  = mean(net_R_i | i ≤ t, net_R_i > 0)
avg_loss_R_t = mean(net_R_i | i ≤ t, net_R_i < 0)            # negative

E_t = winrate_t · avg_win_R_t − (1 − winrate_t)·|avg_loss_R_t| − cost_per_trade_R
```

## Layer 9 — Edge monitor (watchdog)
```
rolling_W_t  = #{i ∈ [t−49, t] : net_R_i > 0}
rolling_L_t  = #{i ∈ [t−49, t] : net_R_i < 0}
rolling_WR_t = rolling_W_t / max(rolling_W_t + rolling_L_t, 1)

blocked_t = 1[
               rolling_WR_t < MIN_WR
             ∧ #{consecutive epochs with rolling_WR_t < MIN_WR} ≥ K_block
            ]

# process-level
alive_t    = 1[heartbeat_received in last HB_TIMEOUT]
restart_t  = 1[¬alive_t ∧ ¬alive_{t−1}]
guardian_t = 1[¬alive_t]                                    # close all
```

## Layer 10 — Edge validity (PBO / half-life)
```
log(|E_t| + ε) = a − b·t
half_life      = ln(2) / b
edge_alive_t   = 1[half_life > 30 days ∧ b ≥ 0]

PBO = #{symmetric CV paths where best IS config loses OOS} / total_paths
PBO < 0.25 → trustworthy.  Current: 0.14
```

---

## Composite action (one line)
```
A_t = sign(EMA10_t − EMA30_t) · max(floor_t, ⌊equity_t·RISK_PCT / (K_STOP·ATR_t·point_value)⌋)
      · 1[proba_t ≥ P_MIN] · 1[veto_t = 1] · 1[¬blocked_t] · 1[|open| < MAX_POSITIONS]
```

## Composite P&L (one line)
```
P&L_t = (exit_t − entry_t)·intent_t·qty·point_value
       − 0.025·qty·tick_value
       − 0.02·qty·round_trip_cost
```

## Composite edge (one line)
```
E_t = winrate_t·avg_win_R_t − (1 − winrate_t)·|avg_loss_R_t| − 0.07
```

## Three numbers decide if the bot stays live
```
1. E_t > 0                       — edge positive in this regime
2. half_life > 30 days           — edge not decaying faster than you can re-fit
3. PBO < 0.25                    — backtest not overfit (yours: 0.14)
```

## Empirical state (2026-08-28)
```
Apr-Jul:  E = 0.68·2.0 − 0.32·1.0 − 0.07 = +0.97R per trade    ← edge alive
Aug:      E = 0.29·2.0 − 0.71·1.0 − 0.07 = −0.20R per trade    ← bleeding
Veto:     OFF (net-negative on NQ blind audit: 13W/3L)
Fast-veto: env-gated, hot-reloadable, 97.3% precision at P(NO)≥0.95
SML:      6 direction attempts DEAD, 1 volatility DEAD, 1 distillation DEAD
PBO:      0.14 ✓
```

## What the math says is actually important
```
The edge lives in Layer 2 (proba_t ≥ 0.35).
Every ML layer above it tried to filter and failed on holdout.
The discipline lives in Layers 9-10 (E_t, half_life, PBO).
The safety lives in the heartbeat watchdog + guardian.
Three independent "don't trade if the math says no" gates.
```
