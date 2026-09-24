"""Phase 7: blend test predictions, write outputs/team_<TEAM_ID>.csv, validate, record SHA256 + run manifest."""
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from signalscope.io import load_config, path, raw_path, read_signals  # noqa: E402
from signalscope.validate_submission import validate  # noqa: E402


def choose(report: dict) -> dict[str, float]:
    """Use the fitted blend only if it beats the best single model on held-out folds; else best single model."""
    fw = report["blend"]["foldwise"]
    singles = {k[len("single_"):]: v for k, v in fw.items() if k.startswith("single_")}
    best = max(singles, key=singles.get)
    if fw["fitted"] >= singles[best]:
        return {k: v for k, v in report["blend"]["weights"].items() if v > 0}
    return {best: 1.0}


def main() -> None:
    cfg = load_config()
    art, out = path(cfg, "artifacts"), path(cfg, "outputs")
    report = json.loads((art / "cv_report.json").read_text())
    w = choose(report)
    s_te = read_signals(cfg, "test")
    s_tr = read_signals(cfg, "train")
    score = np.zeros(len(s_te))
    oof = np.zeros(len(s_tr))
    for k, wk in w.items():
        te = pd.read_parquet(art / f"test_{k}.parquet")
        assert (te.signal_id.values == s_te.signal_id.values).all()
        score += wk * rankdata(te["pred"].to_numpy()) / len(te)
        oof += wk * rankdata(pd.read_parquet(art / f"oof_{k}.parquet")["oof"].to_numpy()) / len(oof)
    # raw-score drift check (before rank mapping): test predictions should look like OOF predictions
    from scipy.stats import ks_2samp
    raw_ks = {k: round(float(ks_2samp(pd.read_parquet(art / f"oof_{k}.parquet")["oof"],
                                     pd.read_parquet(art / f"test_{k}.parquet")["pred"]).statistic), 4) for k in w}
    print("raw OOF vs test KS:", raw_ks)
    assert max(raw_ks.values()) < 0.1, "test score distribution differs from OOF distribution"
    final = rankdata(score, method="average") / (len(score) + 1)   # monotone -> (0,1), keeps ranking
    sub = pd.DataFrame({"signal_id": s_te["signal_id"].astype(str), "ehtimollik": final})
    f = out / f"team_{cfg['team_id']}.csv"
    sub.to_csv(f, index=False, float_format=cfg["submission"]["float_format"], encoding="utf-8", lineterminator="\n")
    rep = validate(f, raw_path(cfg, "test_signals"), cfg["team_id"], oof=oof)
    # 8-decimal round trip must not change ranking
    back = pd.read_csv(f, dtype={"signal_id": str})["ehtimollik"].to_numpy()
    assert (rankdata(back) == rankdata(final)).all(), "rounding changed the ranking"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    pkgs = ["numpy", "pandas", "polars", "pyarrow", "scikit-learn", "lightgbm", "xgboost", "catboost", "scipy",
            "optuna", "torch"]
    manifest = {"submission": f.relative_to(f.parents[1]).as_posix(), "sha256": sha, "weights": w,
                "validator": rep, "raw_oof_vs_test_ks": raw_ks, "seed": cfg["seed"], "final_seeds": cfg["final"]["seeds"], "git_commit": commit,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {p: version(p) for p in pkgs}}
    (art / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", f, "sha256", sha, "weights", w)


if __name__ == "__main__":
    main()
