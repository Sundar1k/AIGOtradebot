# NOTICE — Upstream Attribution

This repository is a heavily modified and extended evolution of
[algoTraderBot](https://github.com/johnamcruz/algoTraderBot) by John Cruz,
distributed under the MIT License. Of the 736 files in the evolved tree,
692 were reworked or written anew for this project; the files listed
below are the ones that remain unmodified upstream content.

## Upstream MIT components (unmodified files)

The following files in this repository were copied unmodified from the
upstream MIT-licensed project and remain licensed under the MIT License:

```
broker_base.py
embed_worker.py
embedder.py
indicators.py
models/ffm_feature_columns.json
models/supertrend_chronos_1min.joblib
ppo_exit/__init__.py
ppo_exit/exit_configs.json
ppo_exit/optimize_exit.py
ppo_exit/policies/ppo_trail_exit.npz
ppo_exit/policies/ppo_trail_exit_1min.npz
ppo_exit/policies/ppo_trail_exit_1min_sb3.zip
ppo_exit/policies/ppo_trail_exit_sb3.zip
ppo_exit/precompute_proba.py
ppo_exit/proba_cache_1min.npz
ppo_exit/trail_exit_env.py
ppo_exit/train_ppo_exit.py
requirements.txt
strategies/base.py
strategies/bos.py
strategies/cisd_ote.py
strategies/cisd_ote_detect.py
strategies/ema_cross.py
strategies/keltner.py
strategies/orb.py
strategies/supertrend.py
tests/conftest.py
tests/test_bracket_ticks.py
tests/test_broker_contract.py
tests/test_cisd_ote.py
tests/test_close_cancels_brackets.py
tests/test_config_micros.py
tests/test_detect_signal.py
tests/test_ensure_policy.py
tests/test_exit_configs.py
tests/test_indicators.py
tests/test_optimize_exit.py
tests/test_orb_gate.py
tests/test_position_size.py
tests/test_sim_broker.py
tests/test_strategy_triggers.py
tests/test_train_live_parity.py
```

The MIT License text as received from the upstream project:

```
MIT License

Copyright (c) 2026 John Cruz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Licensing of the remainder

All other files — every strategy modification, the selection funnel,
selection_validator harness, the spec-kit experiment records under
specs/, the veto/LLM integration, entry-window gating, edge monitor,
attribution agent, and all documentation not listed above — are the
original work of **Sundar1k**, the author of this project, and are licensed
under the terms in LICENSE (ATPL-1.0): attribution required, plus a 5%
commission on Net Trading Profits payable to the Bitcoin address in
COMMISSION.md when the software is used to generate trading income.

Modified files derived from upstream code (e.g. bot.py, broker.py,
config.py, sim_broker.py, strategies/* modifications) are Sundar1k's work
under ATPL-1.0; the upstream MIT permission for the underlying ideas and
code is preserved for any user who prefers to rely on the MIT terms for
the upstream portions.
