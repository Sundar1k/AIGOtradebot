import numpy as np

TRAIL_MULTS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.5]
FEATURES = ["unreal", "mfe", "stop_dist", "atr_R", "bars_norm", "mom"]

def load(path):
    d = np.load(path)
    n = int(d["n_layers"])
    return [(d[f"w{k}"], d[f"b{k}"]) for k in range(n)]

def action(layers, obs):
    h = np.asarray(obs, dtype=np.float32)
    for (w, b) in layers[:-1]:
        h = np.tanh(w @ h + b)      # SB3 MlpPolicy default = Tanh (live inference)
    wl, bl = layers[-1]
    return int(np.argmax(wl @ h + bl))

def probe(layers, label, sweep_idx, sweep_vals, fixed, other_unreal=None):
    print(f"\n--- {label} ---")
    for v in sweep_vals:
        obs = list(fixed)
        obs[sweep_idx] = v
        a = action(layers, obs)
        print(f"  {FEATURES[sweep_idx]}={v:5.2f}  -> trail {TRAIL_MULTS[a]:4.1f}x ATR")

for path, name in [("ppo_exit/policies/ppo_trail_exit.npz", "3-min (live)"),
                   ("ppo_exit/policies/ppo_trail_exit_1min.npz", "1-min")]:
    L = load(path)
    print(f"\n{'#'*60}\n{name}: {path}")
    # typical mid-trade state: +1.5R unrealized, stop 0.5R away, normal vol,
    # 20% of max hold, mild momentum
    base = [1.5, 1.5, 0.5, 1.0, 0.2, 0.1]
    probe(L, "sweep MFE (trade runs deeper) @ unreal=+1.5R",
          1, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], base)
    probe(L, "sweep unrealized R (profit now) @ MFE=+1.5R",
          0, [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], base)
    probe(L, "sweep volatility atr_R (0.5 = quiet, 3 = wild)",
          3, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], base)
    probe(L, "sweep time-in-trade bars_norm (0..1 = 80 bars)",
          4, [0.05, 0.2, 0.4, 0.6, 0.8, 1.0], base)
    probe(L, "sweep momentum mom (R per 3 bars)",
          5, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], base)
