"""Phases 4-6: repeated-CV every model with frozen tuned params, fit blend weights, refit on all data, predict test.

Outputs (artifacts/): oof_<model>.parquet, test_<model>.parquet, cv_report.json, feature_importance.csv
Usage: python scripts/run_train.py [model ...]     (default: config final.models)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from signalscope import models  # noqa: E402
from signalscope.cv import run_cv, splits  # noqa: E402
from signalscope.ensemble import fit_weights, foldwise_check, to_ranks  # noqa: E402
from signalscope.io import ROOT, load_config, path, read_signals, read_tx, seed_everything  # noqa: E402


def tuned(kind: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "tuned" / f"{kind}.yaml").read_text(encoding="utf-8"))["params"]


def factory(kind: str, cfg: dict, scale_rounds: float = 1.0, seq=None, seq_test=None):
    nj = cfg["n_jobs"]
    if kind == "lgb":
        p = tuned("lgb")
        p["n_estimators"] = int(p["n_estimators"] * scale_rounds)
        return models.lgbm(p, nj)
    if kind == "xgb":
        p = tuned("xgb")
        p["n_estimators"] = int(p["n_estimators"] * scale_rounds)
        return models.xgb(p, nj, device="cpu")   # CPU hist: bit-reproducible
    if kind == "cat":
        p = tuned("cat")
        p["iterations"] = int(p["iterations"] * scale_rounds)
        return models.catboost(p, nj, device="CPU")
    if kind == "lr":
        return models.logreg()
    if kind == "seq":
        from signalscope.seqnn import seqnn
        return seqnn(seq, seq_test)
    raise ValueError(kind)


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    kinds = sys.argv[1:] or cfg["final"]["models"]
    art = path(cfg, "artifacts")
    ftr = pd.read_parquet(path(cfg, "processed") / "features_train.parquet")
    fte = pd.read_parquet(path(cfg, "processed") / "features_test.parquet")
    s_tr, s_te = read_signals(cfg, "train"), read_signals(cfg, "test")
    assert (ftr.signal_id.values == s_tr.signal_id.values).all() and (fte.signal_id.values == s_te.signal_id.values).all()
    y = s_tr["eskalatsiya"].to_numpy()
    X, Xt = models.feature_matrix(ftr), models.feature_matrix(fte)
    assert "signal_id" not in X.columns
    seq = seq_test = None
    if "seq" in kinds:
        from signalscope.seqnn import build_sequences
        L = cfg.get("seqnn", {}).get("max_len", 1024)
        seq = build_sequences(s_tr, read_tx(cfg, "train"), L, cfg["feature_set"]["burst_seconds"])
        seq_test = build_sequences(s_te, read_tx(cfg, "test"), L, cfg["feature_set"]["burst_seconds"])
    report_path = art / "cv_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"models": {}}
    n_splits, n_rep = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]

    for kind in kinds:
        t0 = time.time()
        Xk, Xtk = (pd.DataFrame({"_row": np.arange(len(y))}), pd.DataFrame({"_row": np.arange(len(Xt))})) \
            if kind == "seq" else (X, Xt)
        reps = 1 if kind == "seq" else n_rep
        res = run_cv(Xk, y, factory(kind, cfg, seq=seq, seq_test=seq_test), n_splits, reps, seed)
        # final: refit on ALL training rows, averaged over seeds
        preds = []
        for sd in cfg["final"]["seeds"]:
            if kind == "seq":
                fp = factory(kind, cfg, seq=seq, seq_test=seq_test)
                _, pt, _ = fp(Xk, y, Xk.iloc[:10], y[:10], Xtk, sd)
            else:
                fp = factory(kind, cfg, scale_rounds=cfg["final"]["refit_rounds_factor"])
                _, pt, info = fp(X, y, X.iloc[:10], y[:10], Xt, sd)
                if kind == "lgb" and sd == cfg["final"]["seeds"][0]:
                    pd.DataFrame({"feature": X.columns, "gain": info["gain"]}).sort_values(
                        "gain", ascending=False).to_csv(art / "feature_importance.csv", index=False)
            preds.append(pt)
            if kind == "lr":
                break   # deterministic, seed-independent
        test_pred = np.mean(preds, axis=0)
        pd.DataFrame({"signal_id": s_tr.signal_id, "oof": res.oof}).to_parquet(art / f"oof_{kind}.parquet", index=False)
        pd.DataFrame({"signal_id": s_te.signal_id, "pred": test_pred}).to_parquet(art / f"test_{kind}.parquet",
                                                                                   index=False)
        report["models"][kind] = {**res.summary(), "oof_auc_of_mean_pred": round(float(roc_auc_score(y, res.oof)), 5),
                                  "repeats": reps, "runtime_s": round(time.time() - t0, 1)}
        print(kind, report["models"][kind], flush=True)
        report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # blend on every model that has OOF predictions
    avail = sorted(k for k in report["models"] if (art / f"oof_{k}.parquet").exists())
    ranks = to_ranks({k: pd.read_parquet(art / f"oof_{k}.parquet")["oof"].to_numpy() for k in avail})
    w = fit_weights(ranks, y)
    fw = foldwise_check(ranks, y, splits(y, n_splits, 1, seed + 100))
    blend_oof = sum(w[k] * ranks[k] for k in w)
    report["blend"] = {"weights": w, "oof_auc": round(float(roc_auc_score(y, blend_oof)), 5), "foldwise": fw}
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("blend", report["blend"])


if __name__ == "__main__":
    main()
