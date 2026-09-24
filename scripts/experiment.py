"""Auto-research loop step: build (or load cached) features, run a fast CV, append to artifacts/experiments.jsonl.

Examples:
  python scripts/experiment.py --name base --families agg,win,time,pt,lastk,hist,burst
  python scripts/experiment.py --name no_burst --families agg,win,time,pt,lastk,hist
  python scripts/experiment.py --name drop_lastk --exclude "^last\\d"
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signalscope import models  # noqa: E402
from signalscope.cv import run_cv  # noqa: E402
from signalscope.features import build_features, fit_reference  # noqa: E402
from signalscope.io import ROOT, load_config, read_signals, read_tx, seed_everything  # noqa: E402

DETREND = "--detrend" in sys.argv
if DETREND:
    sys.argv.remove("--detrend")
FAST_LGB = {"n_estimators": 5000, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 80,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "reg_lambda": 10.0}


def cached_features(cfg: dict, families: tuple[str, ...], burst_seconds: float, calendar: bool) -> pd.DataFrame:
    code = "".join(q.read_text(encoding="utf-8") for q in sorted((ROOT / "src/signalscope/features").glob("*.py")))
    key = hashlib.md5(json.dumps([families, burst_seconds, calendar, cfg["features"], code, DETREND]).encode()).hexdigest()[:10]
    p = ROOT / "artifacts" / "cache" / f"feat_train_{key}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    s, tx = read_signals(cfg, "train"), read_tx(cfg, "train")
    ref = fit_reference(s, tx, burst_seconds, detrend=DETREND)
    f = build_features(s, tx, ref, cfg["features"]["windows_days"], cfg["features"]["last_k"],
                       burst_seconds, calendar, families)
    p.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(p)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--families", default="agg,win,time,pt,lastk,hist,burst")
    ap.add_argument("--burst-seconds", type=float, default=300.0)
    ap.add_argument("--calendar", action="store_true")
    ap.add_argument("--exclude", default=None, help="regex of feature columns to drop")
    ap.add_argument("--include", default=None, help="regex of feature columns to keep")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--params", default=None, help="json overrides for LightGBM params")
    ap.add_argument("--model", default="lgb", choices=["lgb", "lr"])
    a = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg["seed"])
    t0 = time.time()
    fam = tuple(a.families.split(","))
    f = cached_features(cfg, fam, a.burst_seconds, a.calendar)
    y = read_signals(cfg, "train")["eskalatsiya"].to_numpy()
    X = models.feature_matrix(f)
    if a.include:
        X = X[[c for c in X.columns if re.search(a.include, c)]]
    if a.exclude:
        X = X[[c for c in X.columns if not re.search(a.exclude, c)]]
    params = {**FAST_LGB, **(json.loads(a.params) if a.params else {})}
    fixed = params.pop("_fixed", False)
    fp = (models.lgbm(params, cfg["n_jobs"], inner_es=not fixed) if a.model == "lgb" else models.logreg())
    res = run_cv(X, y, fp, cfg["cv"]["n_splits"], a.repeats, cfg["seed"])
    rec = {"name": a.name, "detrend": DETREND, "families": a.families, "burst_seconds": a.burst_seconds, "calendar": a.calendar,
           "exclude": a.exclude, "include": a.include, "model": a.model, "params": params if a.model == "lgb" else {},
           "n_features": X.shape[1], **res.summary(),
           "best_iter_mean": float(np.mean([i.get("best_iter", 0) for i in res.infos])),
           "runtime_s": round(time.time() - t0, 1)}
    if a.model == "lgb":
        gain = np.mean([i["gain"] for i in res.infos], axis=0)
        top = pd.Series(gain, X.columns).sort_values(ascending=False).head(15)
        rec["top_features"] = [k for k in top.index]
    with open(ROOT / "artifacts" / "experiments.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
