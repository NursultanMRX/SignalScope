"""Optuna tuning (seeded TPE + median pruning) for one model family -> configs/tuned/<model>.yaml.

Objective = OOF AUC of 5-fold CV (1 repeat) with early stopping on an inner split of each training fold.
The tuned number of rounds is the mean best iteration, so final CV/refit use fixed rounds (no early stopping).
Usage: python scripts/tune.py lgb|xgb|cat
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from signalscope import models  # noqa: E402
from signalscope.cv import splits  # noqa: E402
from signalscope.io import ROOT, load_config, path, read_signals, seed_everything  # noqa: E402


def space(trial: optuna.Trial, kind: str) -> dict:
    if kind == "lgb":
        return {"n_estimators": 8000,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 3, 31, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.03, 0.5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 2.0, 4.8])}
    if kind == "xgb":
        return {"n_estimators": 8000,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.03, 0.5, log=True),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 0.1, log=True),
                "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 2.0, 4.8])}
    return {"iterations": 6000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
            "rsm": trial.suggest_float("rsm", 0.05, 0.5, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": 128}


def main() -> None:
    kind = sys.argv[1]
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    X = models.feature_matrix(pd.read_parquet(path(cfg, "processed") / "features_train.parquet"))
    y = read_signals(cfg, "train")["eskalatsiya"].to_numpy()
    sp = splits(y, cfg["cv"]["n_splits"], 1, seed)
    nj = {"lgb": 16, "xgb": 4, "cat": 12}[kind]

    def objective(trial: optuna.Trial) -> float:
        p = space(trial, kind)
        if kind == "lgb":
            fp = models.lgbm(p, nj, inner_es=True)
        elif kind == "xgb":
            fp = models.xgb(p, nj, device="cuda", inner_es=True)
        else:
            fp = models.catboost(p, nj, device="CPU", inner_es=True)
        oof, its = np.zeros(len(y)), []
        for i, (tr, va) in enumerate(sp):
            pv, _, info = fp(X.iloc[tr], y[tr], X.iloc[va], y[va], None, seed + i)
            oof[va] = pv
            its.append(info["best_iter"])
            trial.report(float(roc_auc_score(y[va], pv)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("rounds", int(np.mean(its)))
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2))
    t0 = time.time()
    study.optimize(objective, n_trials=cfg["tuning"][f"{kind}_trials"], timeout=cfg["tuning"]["timeout_s"])
    best = study.best_trial
    params = space(optuna.trial.FixedTrial(best.params), kind)
    rounds_key = "iterations" if kind == "cat" else "n_estimators"
    params[rounds_key] = int(best.user_attrs["rounds"])
    out = ROOT / "configs" / "tuned" / f"{kind}.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump({"oof_auc_tuning": round(best.value, 5), "n_trials": len(study.trials),
                                   "runtime_s": round(time.time() - t0), "params": params}, sort_keys=True),
                   encoding="utf-8")
    study.trials_dataframe().to_csv(ROOT / "artifacts" / f"optuna_{kind}.csv", index=False)
    print(kind, "best", best.value, params)


if __name__ == "__main__":
    main()
