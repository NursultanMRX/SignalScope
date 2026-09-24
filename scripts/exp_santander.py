"""Santander-inspired experiments: (a) within-class column-shuffle augmentation + LGBM, (b) modified Naive Bayes,
(c) 2-leaf (additive) LightGBM. All inside the same CV folds; results -> experiments.jsonl."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from signalscope import models
from signalscope.cv import run_cv
from signalscope.io import ROOT, load_config, path, read_signals, seed_everything

cfg = load_config(); seed_everything(cfg["seed"])
X = models.feature_matrix(pd.read_parquet(path(cfg, "processed") / "features_train.parquet"))
y = read_signals(cfg, "train")["eskalatsiya"].to_numpy()


def augment(x, yy, rng, t=2):
    """Shuffle each column independently within a class (Santander 'LGB 2 leaves + augment')."""
    parts_x, parts_y = [x], [yy]
    for cls, reps in [(1, t), (0, t // 2)]:
        base = x[yy == cls]
        for _ in range(reps):
            x1 = base.copy()
            for c in range(x1.shape[1]):
                x1[:, c] = x1[rng.permutation(len(x1)), c]
            parts_x.append(x1); parts_y.append(np.full(len(x1), cls))
    return np.vstack(parts_x), np.concatenate(parts_y)


def lgb_aug(params, n_aug=3):
    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        rng = np.random.default_rng(seed)
        pv = np.zeros(len(X_va))
        for i in range(n_aug):
            xa, ya = augment(X_tr.to_numpy(), y_tr, rng)
            m = lgb.LGBMClassifier(**{"objective": "binary", "verbose": -1, "n_jobs": 16, "deterministic": True,
                                      "random_state": seed + i, **params})
            m.fit(xa, ya)
            pv += m.predict_proba(X_va.to_numpy())[:, 1]
        return pv / n_aug, None, {}
    return fp


def naive_bayes(n_bins=20, prior_strength=200):
    """Modified NB: per feature, smoothed P(y=1 | quantile bin) fitted on the train fold; sum of log-odds lifts."""
    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        p0 = y_tr.mean(); lo0 = np.log(p0 / (1 - p0))
        score = np.zeros(len(X_va))
        for c in X_tr.columns:
            v = X_tr[c].to_numpy(); w = X_va[c].to_numpy()
            ok = ~np.isnan(v)
            if ok.sum() < 1000 or np.nanstd(v) == 0:
                continue
            edges = np.unique(np.nanquantile(v[ok], np.linspace(0, 1, n_bins + 1))[1:-1])
            bt = np.where(np.isnan(v), -1, np.searchsorted(edges, v))
            bv = np.where(np.isnan(w), -1, np.searchsorted(edges, w))
            df = pd.DataFrame({"b": bt, "y": y_tr}).groupby("b")["y"].agg(["sum", "count"])
            p = (df["sum"] + prior_strength * p0) / (df["count"] + prior_strength)
            lift = np.log(p / (1 - p)) - lo0
            score += pd.Series(bv).map(lift).fillna(0).to_numpy()
        return score, None, {}
    return fp


EXPS = {"nb_20bins": naive_bayes(20, 200), "nb_10bins": naive_bayes(10, 400),
        "lgb_2leaves": models.lgbm({"n_estimators": 8000, "learning_rate": 0.02, "num_leaves": 2,
                                    "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1,
                                    "colsample_bytree": 0.3}, 16, inner_es=True, es_rounds=500),
        "lgb_aug_santander": lgb_aug({"n_estimators": 1500, "learning_rate": 0.01, "num_leaves": 13,
                                      "min_child_samples": 80, "subsample": 0.4, "subsample_freq": 5,
                                      "colsample_bytree": 0.05})}
for name in (sys.argv[1:] or EXPS):
    t0 = time.time()
    res = run_cv(X, y, EXPS[name], 5, 2, cfg["seed"])
    np.save(ROOT / "artifacts" / "cache" / f"oof_{name}.npy", res.oof)
    rec = {"name": name, **res.summary(), "runtime_s": round(time.time() - t0, 1)}
    open(ROOT / "artifacts" / "experiments.jsonl", "a").write(json.dumps(rec) + "\n")
    print(name, rec["oof_auc_mean_over_repeats"], rec["fold_auc_std"], rec["runtime_s"], flush=True)
