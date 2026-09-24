"""More auto-research steps: LightGBM DART, in-fold top-N feature selection, month-rank features."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, lightgbm as lgb, yaml
from signalscope import models
from signalscope.cv import run_cv
from signalscope.io import ROOT, load_config, path, read_signals, seed_everything

cfg = load_config(); seed_everything(cfg["seed"])
X = models.feature_matrix(pd.read_parquet(path(cfg, "processed") / "features_train.parquet"))
s = read_signals(cfg, "train"); y = s["eskalatsiya"].to_numpy()
P = yaml.safe_load((ROOT / "configs/tuned/lgb.yaml").read_text())["params"]


def select_topn(n, params):
    """Rank features by gain of a quick model fitted on the TRAIN fold only, keep top n, refit."""
    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        q = lgb.LGBMClassifier(**{"objective": "binary", "verbose": -1, "n_jobs": 16, "deterministic": True,
                                  "random_state": seed, **params}).fit(X_tr, y_tr)
        keep = X_tr.columns[np.argsort(-q.booster_.feature_importance("gain"))[:n]]
        return models.lgbm(params, 16)(X_tr[keep], y_tr, X_va[keep], y_va, None, seed)
    return fp


def with_month_rank(Xd):
    """AMEX-style: percentile rank of each feature within the alert's signal month (label-free, uses train+test
    in production; here train only)."""
    m = s["signal_sanasi"].dt.to_period("M").astype(str).values
    top = pd.read_csv(ROOT / "artifacts/feature_importance.csv")["feature"].head(40).tolist()
    r = Xd[top].groupby(m).rank(pct=True).add_suffix("_mrank")
    return pd.concat([Xd, r], axis=1)


dart = {**P, "boosting_type": "dart", "n_estimators": 1200, "learning_rate": 0.03, "drop_rate": 0.1,
        "skip_drop": 0.5}
EXPS = {"dart": (X, models.lgbm(dart, 16)),
        "sel150": (X, select_topn(150, P)), "sel300": (X, select_topn(300, P)),
        "month_rank": (with_month_rank(X), models.lgbm(P, 16)),
        "ref_fixed": (X, models.lgbm(P, 16))}
for name in (sys.argv[1:] or EXPS):
    Xd, fp = EXPS[name]; t0 = time.time()
    res = run_cv(Xd, y, fp, 5, 3, cfg["seed"])
    np.save(ROOT / "artifacts/cache" / f"oof_{name}.npy", res.oof)
    rec = {"name": name, **res.summary(), "runtime_s": round(time.time() - t0, 1)}
    open(ROOT / "artifacts/experiments.jsonl", "a").write(json.dumps(rec) + "\n")
    print(name, rec["oof_auc_mean_over_repeats"], rec["fold_auc_std"], rec["runtime_s"], flush=True)
