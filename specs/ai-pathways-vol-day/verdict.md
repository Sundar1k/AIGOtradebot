# AI Pathways vol-day claim — tested on algoTraderBot's OWN data (2026-09-02)
Source video: "Claude Tests The Most Misunderstood Concept In Trading" (WPLkK4M9aDY)
His claim: vol does NOT change whether you trade; mean R holds up on high-VIX days (huge winners
offset losers); win rate barely moves; vol only speeds resolution; sizing (smaller on high-vol)
cuts equity swings ~42% at slight total cost.

## Test on our data (point-in-time, NQ only, 2021-04 -> 2026-08)
Method: signals_NQ.jsonl realized trades (n=2,680), vol proxy = trailing 200-bar mean |dclose|/close
at signal ts. Tercile buckets by vol. Script: ~/reference/ai-pathways/vol_day_test.py.

| tercile | n    | meanR  | medianR | winrate | sumR   | PF   | exit stop/trail |
|---------|------|--------|---------|---------|--------|------|-----------------|
| low     |  894 | +0.847 | +0.312  | 50.3%   | +757.1 | 3.30 | 48.8% / 51.2%   |
| med     |  893 | +0.811 | +0.000  | 46.5%   | +723.8 | 3.25 | 53.1% / 46.9%   |
| high    |  893 | +0.683 | +0.000  | 41.0%   | +610.1 | 2.70 | 58.3% / 41.7%   |

MeanR high-low: -0.164 | MedianR high-low: -0.312.
Resolution DOES speed up on high-vol (stops 58.3% vs 48.8%) — confirms his "vol executes your
plan faster", but for OUR funnel it executes via the STOP more often.

Sizing sim (50% size on high tercile): totalR +2091 -> +1786 (-15%), daily std 3.80R -> 3.16R
(-17% swings), max daily loss unchanged (-21.7R).

## Verdict
1. His general claim does NOT transfer to this funnel: high-vol days are lower-expectancy here on
   mean, median, winrate AND PF (monotonic). Our selection strategies are not vol-neutral.
2. Vol-scaled sizing (his C1): REJECTED for default — costs 15% total R to save 17% swing.
   Would only make sense as an explicit capital-preservation mode, user-decided, never default.
3. High-vol day downweighting/gate: the numbers (WR 41% / PF 2.70 on high tercile) suggest a
   real effect, but our regime-halt history says coarse vol halts over-throttle (64% removal).
   Any gate experiment must be pre-registered with a removal cap (like the regime-halt spec).
4. No live config change made. This was an exploratory audit on the settled dataset.
