"""Auto-train the PPO exit when the active timeframe has no policy. If the user
runs a timeframe (e.g. --timeframe 1) without a trained/cached exit, the bot flags
it and runs train_ppo_exit for that timeframe — as a subprocess so torch/SB3 never
load next to xgboost. If PPO is off or the train produces nothing, it falls back to
the fixed-RR exit (returns None)."""
import subprocess
import types

import bot
import config


def test_none_when_ppo_disabled():
    config.USE_PPO_EXIT = False
    assert bot.ensure_exit_policy() is None


def test_returns_existing_policy_without_training(tmp_path, monkeypatch):
    config.USE_PPO_EXIT = True
    p = tmp_path / "ppo_trail_exit.npz"
    p.write_bytes(b"policy")
    monkeypatch.setattr(config, "policy_path", lambda: str(p))
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0))
    assert bot.ensure_exit_policy() == str(p)
    assert calls == []                                   # already trained → no retrain


def test_trains_when_missing(tmp_path, monkeypatch):
    config.USE_PPO_EXIT = True
    config.TIMEFRAME_MIN = 5                              # no 5-min policy shipped
    p = tmp_path / "ppo_trail_exit_5min.npz"
    monkeypatch.setattr(config, "policy_path", lambda: str(p))

    def fake_run(cmd, *a, **k):
        assert "--timeframe" in cmd and "5" in cmd        # trains for the active tf
        p.write_bytes(b"trained")                         # the retrain produces the policy
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert bot.ensure_exit_policy() == str(p)
    assert p.exists()


def test_falls_back_when_train_fails(tmp_path, monkeypatch):
    config.USE_PPO_EXIT = True
    config.TIMEFRAME_MIN = 5
    p = tmp_path / "ppo_trail_exit_5min.npz"             # never created
    monkeypatch.setattr(config, "policy_path", lambda: str(p))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=1))
    assert bot.ensure_exit_policy() is None              # → fixed-RR fallback
