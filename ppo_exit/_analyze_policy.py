import numpy as np
import sys

# Feature order in the obs vector (trail_exit_env.py _obs / exit_manager.py exit_obs):
FEATURES = ["unreal (unrealized R)", "mfe (max fav. excursion)",
            "stop_dist (R)", "atr_R (volatility)", "bars_norm (time)",
            "mom (3-bar momentum)"]
TRAIL_MULTS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.5]

for path in ["ppo_exit/policies/ppo_trail_exit.npz",
             "ppo_exit/policies/ppo_trail_exit_1min.npz"]:
    d = np.load(path)
    n = int(d["n_layers"])
    layers = [(d[f"w{k}"], d[f"b{k}"]) for k in range(n)]
    print(f"\n{'='*70}\n{path}  ({n} linear layers)")
    for k, (w, b) in enumerate(layers):
        print(f"  layer {k}: {w.shape}  (out x in)  |W| mean={np.abs(w).mean():.3f}")
    w0, b0 = layers[0]
    hidden = w0.shape[0]

    # 1) Raw first-layer |weight| per feature (unbiased: weights are in
    #    normalized units and inputs are clipped to the same ±10 scale).
    per_feat = np.abs(w0).mean(axis=0)
    print("\n  first-layer mean |W| per feature (raw):")
    for f, v in zip(FEATURES, per_feat):
        print(f"    {f:26s} {v:.3f}")

    # 2) Sensitivity: mean |dlogit_i / dx_j| over actions and hidden units.
    #    d(logits)/dx = W_last * diag-ish... use full chain:
    #    logits = W2 * relu(W1 x + b1) + b2  (assuming 1 hidden layer);
    #    for deeper nets, compose numerically via finite differences instead.
    if n == 2:
        w1, b1 = layers[1]
        # analytic: J = w1.T @ (w0 * (h > 0))  for each hidden unit h
        sens = np.zeros(len(FEATURES))
        # integrate over a grid of representative obs points
        rng = np.random.default_rng(0)
        for _ in range(2000):
            x = rng.uniform(-3, 3, size=len(FEATURES))  # typical operating range
            h = np.maximum(0, w0 @ x + b0)
            dlogits = w1.T * (h > 0)              # (n_actions, hidden)
            # jacobian per feature: sum over hidden of |dlogits/dx_j|
            J = np.abs(dlogits @ w0).sum(axis=0)  # (features,)
            sens += J
        sens /= 2000
        print("\n  analytic |dlogits/dx| sensitivity per feature (2000-point grid):")
    else:
        # numeric finite-difference sensitivity over the grid
        # NOTE: live inference (NumpyMlpPolicy.action) uses Tanh — SB3
        # MlpPolicy default — so the forward pass here must match.
        def _logits(x):
            h = x
            for (w, b) in layers[:-1]:
                h = np.tanh(w @ h + b)
            wl, bl = layers[-1]
            return wl @ h + bl
        sens = np.zeros(len(FEATURES))
        rng = np.random.default_rng(0)
        for _ in range(2000):
            x = rng.uniform(-3, 3, size=len(FEATURES))
            base = _logits(x)
            for j in range(len(FEATURES)):
                xp = x.copy(); xp[j] += 0.05
                sens[j] += np.abs(_logits(xp) - base).sum() / 0.05
        sens /= 2000
        print("\n  numeric |dlogits/dx| sensitivity per feature (2000-point grid):")
    for f, v in zip(FEATURES, sens):
        bar = "#" * int(round(30 * v / sens.max()))
        print(f"    {f:26s} {v:8.3f}  {bar}")
    total = sens.sum()
    print("\n  relative importance (%):")
    for f, v in zip(FEATURES, sens):
        print(f"    {f:26s} {100*v/total:5.1f}%")

    # 3) Action bias: what does the policy pick when obs is neutral (all zero)?
    x = np.zeros(len(FEATURES))
    h = x
    for (w, b) in layers[:-1]:
        h = np.maximum(0, w @ h + b)
    wl, bl = layers[-1]
    logits = wl @ h + bl
    print("\n  logits at neutral obs (all 0):")
    for m, l in zip(TRAIL_MULTS, logits):
        print(f"    trail {m:4.1f}x ATR  logit {l:7.3f}")
    print(f"    -> argmax: {TRAIL_MULTS[int(np.argmax(logits))]}x ATR")
