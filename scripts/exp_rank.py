"""Validate the month-rank gain: variants of cohort ranking + a different CV seed."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, yaml
from signalscope import models
from signalscope.cv import run_cv
from signalscope.io import ROOT, load_config, path, read_signals, seed_everything

cfg = load_config(); seed_everything(cfg["seed"])
X = models.feature_matrix(pd.read_parquet(path(cfg, "processed") / "features_train.parquet"))
s = read_signals(cfg, "train"); y = s["eskalatsiya"].to_numpy()
P = yaml.safe_load((ROOT / "configs/tuned/lgb.yaml").read_text())["params"]
imp = pd.read_csv(ROOT / "artifacts/feature_importance.csv")["feature"].tolist()


def ranked(Xd, cols, key, suffix, replace=False):
    r = Xd[cols].groupby(key).rank(pct=True)
    if replace:
        out = Xd.copy(); out[cols] = r.values; return out
    return pd.concat([Xd, r.add_suffix(suffix)], axis=1)


mo = s["signal_sanasi"].dt.to_period("M").astype(str).values
qt = s["signal_sanasi"].dt.to_period("Q").astype(str).values
wk = s["signal_sanasi"].dt.to_period("W").astype(str).values
rnd = np.random.default_rng(0).integers(0, 24, len(s))   # placebo: random groups of month size
V = {"m_top40": ranked(X, imp[:40], mo, "_mr"), "m_top100": ranked(X, imp[:100], mo, "_mr"),
     "q_top40": ranked(X, imp[:40], qt, "_qr"), "w_top40": ranked(X, imp[:40], wk, "_wr"),
     "placebo_top40": ranked(X, imp[:40], rnd, "_pr"), "global_top40": ranked(X, imp[:40], np.zeros(len(s)), "_gr")}
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
for name in ["base"] + list(V):
    Xd = X if name == "base" else V[name]; t0 = time.time()
    res = run_cv(Xd, y, models.lgbm(P, 16), 5, 3, seed)
    rec = {"name": f"rank_{name}_cvseed{seed}", **res.summary(), "runtime_s": round(time.time() - t0, 1)}
    open(ROOT / "artifacts/experiments.jsonl", "a").write(json.dumps(rec) + "\n")
    print(name, rec["oof_auc_mean_over_repeats"], flush=True)
