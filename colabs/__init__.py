#!/usr/bin/env python3
"""colabs/__init__.py — reconstructed Chronos labelers for AlgoTraderBot.

The original colabs package (which trained the shipped *_chronos.joblib
bundles) is not on disk. These classes reimplement the StrategyLabeler
protocol (futures_foundation.pipeline.strategy.StrategyLabeler /
BaseChronosStrategy) using the CURRENT strategy code in strategies/ and the
settle logic from lane_test.py, so walk-forward training is reproducible.

VERIFICATION GATE (2026-08-22): labeler.build() over the full 3-min span
must reproduce the shipped bundle metadata — ema: n_train_signals 85,158,
label_dist [59894, 25264]; supertrend_1min: 313,892 / [217650, 96242].
Run: python -m colabs.verify  (prints counts per labeler vs metadata).
"""
