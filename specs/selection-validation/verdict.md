# Verdict: Selection Validator (quality-band) — FINAL

Generated: 2026-08-30 (autonomous run) | OOS boundary: 2026-08-01 (pre-registered, FR-007) | dataset rows: 98,363

## Decision: **KILL — quality-band lever archived (all q)**

Pre-registered rule (spec.md SC-003): GO iff N>=150 OOS, P(ΔavgR>0) > 0.95 AND PF_cand >= PF_base; KILL decisive when evidence is against.

## OOS slice (Aug 1-25 2026 — the only numbers that decide)

| selector | closed | WR | avgR | sumR | PF | P(ΔavgR>0) | verdict |
|---|---|---|---|---|---|---|---|
| quality>=5 | 99 | 0.1313 | -1.2444 | -123.1969 | 0.170 | 0.442 | KILL |
| quality>=6 | 77 | 0.0909 | -1.5428 | -118.7958 | 0.094 | 0.055 | KILL |
| quality>=7 | 71 | 0.0986 | -1.5219 | -108.0522 | 0.102 | 0.075 | KILL |
| baseline | 482 | 0.1266 | -1.2181 | -587.1236 | 0.187 | — | — |

## Train slice (calibration only — no decision power)

- baseline: n=10845 WR=0.4381 avgR=0.5022 PF=2.01
- candidate q=5: n=3399 avgR=0.3793 PF=1.71 | q=6: n=2602 avgR=0.3651 PF=1.68 | q=7: n=2269 avgR=0.3238 PF=1.59

## Interpretation (numbers, then words)

- The harness is honest: baseline reproduces the doctrine edge on history (avgR +0.50, PF 2.01 vs doctrine +0.58 / PF 2.11; small gap = full 5.3y window + replay gate-stack divergence, documented).
- August 2026 OOS: the funnel itself collapsed (WR 12.7%, avgR -1.22R, PF 0.19) — the regime wipeout the doctrine warned about, at scale.
- The quality band made selection WORSE in every configuration: PF 0.17/0.09/0.10 vs 0.19 baseline; P(ΔavgR>0) = 0.44/0.06/0.08. At q=6/7 the filter is anti-predictive — the 7B rated losing setups higher.
- The 7B veto quality ratings carry NO out-of-sample selection power. The lever is archived per pre-registration.

## Files
- dataset: selection_validator/data/signals_*.jsonl (98,363 rows, 11,370 funnel-taken, leak-checked, deduped)
- harness: selection_validator/{dataset,selectors,harness,verdict,quality_score,dedup_signals,replay}.py + tests/ (14 tests green)
- replay: 5 symbols x 5.3y through the EXACT live loop (bot.handle_bar); 11,327 trades, outcomes matched by entry bar (100%)

