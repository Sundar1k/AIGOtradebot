# AlgoTraderBot — Framework Digest (v2, updated 2026-08-17 20:10 AEST)
Audit: codebase-audit skill · Repo: ~/projects/algoTraderBot · Services: autotrade.service + veto.service
Companion dirs: ~/topstep-bot (signals.py, watchdog.py, doctor.py, telegram.py) · ~/.hermes/scripts (cron wrappers)
v2 deltas vs v1: parallel fetch, candle-pattern detector, conflict tags, veto cache + prefetch + batch, reflection memory, multi-voice agreement, gated conflict-block rule, xgboost re-save, doctor upgrades.

---

## 1. OVERVIEW

| Component | Instrument/Data | Timeframe | Cadence | Entrypoint | Live/Paper |
|---|---|---|---|---|---|
| autotrade supervisor | NQ, ES, RTY, YM, GC (micros trade parent bars) | 3-min | every 3 min, 24/5 (CME hours) | supervisor.py | LIVE TopstepX eval sim ($50K, id 26251526) |
| veto sidecar | — | — | on-demand + prefetch | veto_server.py (veto.service) | GPU (CUDA 0), fail-closed, decision cache |
| missed-trade learner | same 5 symbols | 3-min replay | every 3h (:35) | missed_trades.py via cron | paper (simulates brackets on history) |
| attribution agent | same 5 symbols | 3-min replay | Sat 18:30 | attribution.py via cron | applies rules to live config |
| candle pattern detector | live 3-min → 30-min agg | 30-min | every scan cycle | candle_patterns.py (imported by supervisor + learner) | observation + ledger tags |
| regime detector | NQ daily returns | daily | every 6h | regime.py --update via cron | writes .autotrade_regime.json |
| walk-forward retrain | 5-symbol history | — | Sat 18:00 | finetune/walkforward.py | trains veto adapter |

Infra: systemd user units (autotrade.service CPUQuota=50%, CPU-pinned; veto.service GPU), Hermes cron (11 jobs), Telegram alerts, TopstepX REST API.

---

## 2. DECISION PIPELINE (per 3-min bar, per symbol; entry path)

```
fetch bars (5 symbols PARALLEL via ThreadPoolExecutor; decisions serial)
  → [P] VETO PREFETCH (background thread, fire-and-forget): builds state line for
        each symbol, POSTs /decide_batch → warms the decision cache so a signal
        on a seen state gets an INSTANT veto (0.01s vs 17-32s)
  → [G1] position check: open position on symbol? → trail/exit, no new entry
  → [G2] book-level: one position MAX across book (Topstep rule)
  → [G3] strategy detect: EMA(9/20) cross AND ADX(14) ≥ 18.0 → Signal
  → [G4] grade: Chronos embed + XGBoost → proba, r_hat
  → [G5] confidence window: 0.35 ≤ proba ≤ 0.50 (floor/ceil)
  → [G6] veto_fn (ordered):
       G6a EVENT BLACKOUT (news calendar, deterministic)
       G6b REGIME GATE (HMM file, block panic)
       G6c CANDLE-CONFLICT GATE — ONLY if AUTOTRADE_BLOCK_CONFLICT=1 (validated
            rule, currently OFF): block when 30-min pattern opposes signal dir
       G6d LLM VETO — POST state to :8765/decide (cache-first) → action==side
       G6e QUALITY GATE (min 0 = inert)
       G6f AGREEMENT REPORT — voices N/2 (veto + candle pattern); logged, never gates
  → [G7] position size → market order + bracket (stop 0.5×ATR, target 2R)
```

| Gate | Input | Condition | PASS | BLOCK | Config key (file) |
|---|---|---|---|---|---|
| P prefetch | current state text | always (bg) | cache warm | — | AUTOTRADE_VETO_PREFETCH=1 (env) |
| G1 position | broker | open pos | manage/trail | — | — |
| G2 book cap | broker | one pos/book | own symbol | skip others | — |
| G3 detect | bars | EMA9×20 cross + ADX≥18 | Signal | wait | EMA_FAST/SLOW, ADX_GATE (config.py) |
| G4 grade | embed+hand | — | proba, r_hat | — | CTX=128 |
| G5 confidence | proba | 0.35≤p≤0.50 | candidate | skip | PROBA_FLOOR/CEIL (config.py) |
| G6a blackout | events cache | in window | block | block | market_context.* |
| G6b regime | regime json | panic | block | block | AUTOTRADE_REGIME_GATE |
| G6c conflict | 30m pattern vs dir | conflict & enabled | block | block | AUTOTRADE_BLOCK_CONFLICT (env, .env) |
| G6d LLM veto | :8765/decide | action==side | pass | block (fail-closed) | AUTOTRADE_VETO_FAIL_OPEN=0 |
| G6e quality | veto resp | q≥min | pass | block | AUTOTRADE_QUALITY_MIN=0 |
| G6f agreement | veto+pattern | N/A (log only) | log voices | — | — |
| G7 size | stop_ticks | — | enter | — | RISK_PER_TRADE=0 → SIZE=1 |

⚠️ G6c is wired but INACTIVE until the attribution agent validates and sets AUTOTRADE_BLOCK_CONFLICT=1 (ledger: 27 conflict samples vs 30 minimum — evidence strong: 18.5% WR vs 41.1% aligned).

---

## 3. LIFECYCLE (entry → exit)

- Entry: market order at signal-bar close, size 1 contract (risk sizing off).
- Stop: 0.5 × ATR(20) from entry. Target: 2.0R. Fixed 2R bracket (PPO disabled).
- Exit paths: broker stop (−1R), broker target (+2R), gap close, circuit breaker (daily loss ≥$400 / DD ≥$1500 → close ALL + cancel strays), reconcile stray-cancel.
- on_trade_close → evolver.record (R, strategy, side, quality) → adapts floor/ceil → ALSO reflection.record (writes lessons file, feature #5).

---

## 4. CONFIG TRUTH-TABLE (live path)

| Key | Value | Defined in | Read by live? |
|---|---|---|---|
| BROKER | "topstepx" | config.py | yes |
| TIMEFRAME_MIN | 3 | config.py | yes |
| TRADE_SYMBOLS | NQ,ES,RTY,YM,GC | .env AUTOTRADE_SYMBOLS | yes |
| PROBA_FLOOR | 0.35 | config.py | yes |
| PROBA_CEIL | 0.50 | config.py | yes |
| ACTIVE_STRATEGIES | ["ema"] | config.py | yes |
| ADX_GATE | 18.0 | config.py | yes |
| STOP_ATR / ATR_P | 0.5 / 20 | config.py (+exit_configs.json) | yes |
| RR | 2.0 | config.py | yes |
| SIZE / MAX_CONTRACTS | 1 / 10 | config.py | yes |
| USE_PPO_EXIT / TRAILING | False/False | config.py | yes (fixed 2R) |
| ACTIVATE_R / GIVEBACK_R | 2.0 / 0.75 | config.py | NO — PPO off, decorative |
| AUTOTRADE_DAILY_LOSS / TRAILING_DD | 400 / 1500 | systemd unit env | yes |
| AUTOTRADE_HOURS | all | systemd unit env | yes |
| AUTOTRADE_QUALITY_MIN | 0 | systemd unit env | yes (inert) |
| AUTOTRADE_VETO_FAIL_OPEN | 0 | supervisor default | yes (fail-closed) |
| AUTOTRADE_REGIME_GATE/BLOCK | strict/panic | supervisor default | yes |
| AUTOTRADE_PROTECTED_SYMBOLS | NQ,YM | .env | yes (attribution) |
| AUTOTRADE_VETO_PREFETCH | 1 | supervisor default | yes (feature #3) |
| AUTOTRADE_BLOCK_CONFLICT | 0 (off) | .env (set by attribution when validated) | yes (G6c, feature #6) |
| AUTOTRADE_LEARN_* | defaults | env | yes (missed_trades) |
| AUTOTRADE_ATTRIB_APPLY | 1 | env | yes (attribution) |
| non-ema strategy params (ST_PERIOD, KC_*, ORB_*, SWING_K) | various | config.py | NO — dead (only "ema" active) |

---

## 5. STATE & DATA CONTRACTS

| File | Schema | Writer | Reader | Freq |
|---|---|---|---|---|
| .autotrade_state | {date, start_balance, at_peak, halted, reason, trades, last_beat, pnl_history[], today_pnl, consistency_halted, balance, evolve{floor,baseline,ceil,stance,...}, consistency{...}} | supervisor | supervisor, watchdog, doctor, learner, attribution | 3-min |
| .autotrade_regime.json | {regime, prob, gate, symbol, ann_vol_pct, ts, state_means} | regime.py cron | supervisor (G6b) | 6h |
| .autotrade_events.json | [{title, importance, time, type}] — 7,087 entries, 140 high-impact | market_context (cached 12h) | supervisor (G6a) | refresh |
| .autotrade_missed.json | {records[{symbol,time,dir,proba,r_hat,r,kind,patterns,pattern_dir,conflict,ts}], updated, last_change, changes[]} | missed_trades.py | learner, attribution | 3h |
| .autotrade_rules.json | {applied[{ts,rule,target,action,evidence,validation,applied,restart_ok}]} | attribution.py | attribution | Sat |
| .autotrade_rules.md | human-readable rule change log | attribution.py | user | Sat |
| .autotrade_lessons.md | reflection entries: ## ts — SYM SIDE ±R (WIN/LOSS) + entry/exit + veto reason | reflection.py (via on_trade_close) | user, attribution (future) | per closed trade |
| log/bot.log | text log: candles, 🕯 patterns, signals, ENTER/EXIT/VETO, voices | bot logger | doctor, debugging | every bar |

---

## 6. EXTERNAL SERVICES

| Provider | Endpoint | Purpose | Timeout | Failure behavior |
|---|---|---|---|---|
| TopstepX | api.topstepx.com/api | auth, bars, orders, positions | 30s | BLOCKS (skip symbol / loop error + alert) |
| veto sidecar | 127.0.0.1:8765 /decide /decide_batch /score /health | LLM veto + cache | 30s | BLOCKS entries (fail-closed) |
| Fed calendar | federalreserve.gov/json/calendar.json | FOMC events | 20s | PASS-THROUGH → [] (Fed events vanish; doctor checks) |
| Google News RSS | news.google.com/rss/search | headlines | 20s | PASS-THROUGH → "" (informational) |
| Telegram | api.telegram.org | alerts | 15s | PASS-THROUGH (alerts lost silently) |

Veto cache: in-process LRU (256 states) in llm_veto.py, keyed by exact state text, safe by greedy determinism (do_sample=False). Stats on /health (hits/misses/size/hit_rate). Measured: cold 32s → cached 0.011s.

---

## 7. LEARNING MODULES (interfaces)

- `evolve.Evolver` — record(trade) adapts floor/ceil from closed-trade WR/avgR; current_floor()/ceil; status(). BLOCKS: feeds G5 thresholds.
- `missed_trades.py` — replay(symbol, days) detects+grades+simulates every signal incl. pattern tags; analyze(records, floor) band stats; apply_floor(new) lowers floor when just-missed band proves profitable (n≥20, WR≥55%, avgR>0, 48h cooldown). Stores: ledger json.
- `attribution.py` — detect_symbol_drops / detect_floor_raise / detect_ceil_tighten / detect_conflict_block; validate() chrono 67/33 split; apply_rule() writes config/.env + restarts. BLOCKS: drops symbols, raises floor, tightens ceil, enables conflict-block — each gated OOS. Stores: rules.json/md.
- `candle_patterns.py` — detect_patterns(bars) → [(time, [patterns])]; pattern_at_time(bars, ts) for signal tagging; pattern_direction(pats) → +1/-1/0. Pure numpy/pandas, no model. 13 patterns incl. volume-confirmed reversals.
- `reflection.py` — record(trade) appends veto reason + outcome to lessons file; latest(n). Stores: .autotrade_lessons.md.
- `regime.py` — current_regime(symbol) → {regime, prob, gate}. BLOCKS: G6b. Stores: regime json.
- `market_context.py` — event_blackout(), get_headlines(), context_line(). BLOCKS: G6a. Stores: events cache.

---

## 8. MODELS

| Artifact | Type | Features | Output | Retrain |
|---|---|---|---|---|
| models/ema_cross_chronos.joblib (+5 others) | XGBHead + XGBRiskHead bundle | 76 FFM cols (ffm_feature_columns.json, exact order) + 5 hand features → 337-dim | proba, r_hat | walk-forward weekly |
| finetune/output8b | Qwen2.5-7B 4-bit NF4 + LoRA | state text (v1 format, byte-fixed) | BUY/SELL/NO TRADE + reason | walk-forward weekly |
| base | ~/qwen-dl | — | — | — |

⚠️ RE-SAVED 2026-08-17: all 6 bundles re-serialized in current xgboost format (re-save_models.py, bit-identical predictions verified). No feature_names_in_ on XGBHead → positional inference; FFM order pinned by json. r_hat head: verified NOISE (corr +0.026) — informational only.

---

## 9. SCHEDULE (Hermes cron)

| Job | Schedule | Script | Purpose |
|---|---|---|---|
| autotrade-watchdog | */10 * * * * | autotrade-watchdog.py → topstep-bot/watchdog.py | self-heal: restart dead/stale autotrade+veto; alert |
| autotrade-missed-learner | 35 */3 * * * | autotrade-missed.py → missed_trades.py | simulate skipped trades (incl. patterns/conflict); maybe lower floor |
| autotrade-doctor | 5 9 * * * | autotrade-doctor.py → topstep-bot/doctor.py | silent-failure checks (quality gate, adapter, regime, events incl. FED, signals.py, candles) |
| autotrade-attribution | 30 18 * * 6 | autotrade-attribution.py → attribution.py | propose+validate+apply rules (symbols/floor/ceil/conflict) |
| regime-update | 0 */6 * * * | (agent) regime.py --update | HMM regime refresh |
| walkforward-weekly | 0 18 * * 6 | walkforward-weekly.sh | veto retrain (GPU-gated) |
| topstep-bot-scan | */5 0-5,23 * * 1-5 | scan.sh | paper scanner (separate system) |
| topstep-bot-eod | 5 6 * * 2-6 | eod.sh | daily summary |
| hf-7b-download-watch | every 5m | hf_dl_watch.sh | STALE (download done) — harmless token burn |

---

## 10. INTERNAL ARCHITECTURE (module dependency graph)

```
supervisor.py ──► bot.py (handle_bar, BotContext)
   │                ├─► strategies/ema_cross → base.Strategy (detect/grade)
   │                │     └─► embedder.py ──► embed_worker.py (Chronos subprocess, thread-locked)
   │                │     └─► indicators.py, futures_foundation (FFM 76)
   │                ├─► broker.py (TopstepXClient, thread-safe parallel fetches)
   │                ├─► evolve.py (floor/ceil) + reflection.py (lessons) via on_trade_close
   │                └─► llm_veto via veto_server.py (:8765, _infer_lock, LRU cache, /decide_batch)
   ├─► market_context.py ──► signals.py (in ~/topstep-bot ⚠️ cross-repo)
   ├─► candle_patterns.py (live 🕯 logging; conflict gate G6c; agreement voices)
   ├─► regime.py (.autotrade_regime.json)
   └─► bot + broker ──► TopstepX REST

aux: missed_trades.py / attribution.py (cron) — read ledger/config, may write config/.env + restart
     topstep-bot/{watchdog,doctor}.py (cron) — health + self-heal
     re-save_models.py — one-off xgboost format migration (backups in /tmp/model_backup_*)
```

---

## 11. KNOWN DIVERGENCES & RISKS

1. **r_hat is noise** (verified, corr +0.026) — informational only; never gate on it.
2. **0.30-0.35 band loses** (-0.23R avg) — floor at 0.35 by data + user decision.
3. **NQ/YM protected** (user choice) — attribution skips them; ES/RTY/GC droppable.
4. **Model positional inference** — no feature_names_in_; FFM order pinned by json; re-saved to current xgboost format (compat risk removed).
5. **Quality gate** — must stay 0 while v1 veto never emits quality; doctor checks daily.
6. **llm_veto default adapter** = output8b (fixed); systemd unit also pins it — keep in sync.
7. **Veto latency** — 17-32s cold; cache+prefetch+batch mitigate; _infer_lock serializes GPU. Firefox disk thrash can starve model load (observed; niced).
8. **Conflict rule** — 27 samples vs 30 min; gate refuses to act yet (disciplined). Evidence: conflict 18.5% WR vs aligned 41.1%. Will auto-propose when threshold crossed.
9. **Backtest vs live** — SimBroker conservative fills; no veto in backtest; live-only filters. Expectancy differs.
10. **News calendar silent degradation** — Fed fetch failure → [] (doctor checks FED presence now).
11. **hf-7b-download-watch stale** — token burn only.
12. **Telegram alerts swallow errors** — advisory only.
13. **CPUQuota=50%** — parallel fetch helps; grading bursts could lag cycle under load.
14. **STOP_ATR override** — exit_configs.json matches config (0.5) — verified no divergence.
15. **Ledger merge upgrade** — replay now upgrades existing records with pattern tags in place (293 upgraded) — keeps tags current without duplication.

---

## SINGLE POINT OF TRUTH (10 lines)

- Live entry path: supervisor.py → bot.handle_bar → gates (strategies/base.py + supervisor.veto_fn). Everything else is advisory/post-hoc.
- **PROBA_FLOOR/CEIL in config.py** — the only filter between signal and veto; keep in sync across config.py, .env, and .autotrade_state → evolve.floor/ceil (evolver's copy, clamped to [baseline, baseline+0.30]).
- Symbols: .env AUTOTRADE_SYMBOLS (overrides config); PROTECTED list in .env; restart service after changes.
- Veto adapter: llm_veto.py default AND veto.service Environment must both point to output8b.
- Quality gate: AUTOTRADE_QUALITY_MIN must stay 0 (v1 model never emits quality).
- signals.py must stay in ~/topstep-bot (supervisor imports it; doctor checks).
- Model feature order: pinned by models/ffm_feature_columns.json (76) + strategies/ema_cross hand features — never reorder without retraining.
- Veto cache: safe only because decode is greedy (do_sample=False) — if sampling is ever enabled, disable the cache.
- Conflict block (G6c) is dormant until attribution validates and sets AUTOTRADE_BLOCK_CONFLICT=1 — the ledger (27 conflict samples) decides, never the eyeball.
- Watchdog restarts, doctor catches rot, learner tunes floor, attribution tunes rules — all read .autotrade_state/ledger; none replaces the supervisor's judgment.
