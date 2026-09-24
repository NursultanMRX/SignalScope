"""Generate notebooks/team_<TEAM_ID>_reproducible.ipynb: a self-contained notebook that re-creates the submission
from the five raw files. The feature/CV/model/validator code is INLINED from src/ (so notebook and pipeline share the
exact same logic) and tuned hyper-parameters are frozen as literals."""
import json
import re
import sys
from pathlib import Path

import nbformat as nbf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signalscope.io import load_config  # noqa: E402

MODULES = ["features/aggregates.py", "features/windows.py", "features/temporal.py", "features/sequence.py",
           "features/__init__.py", "cv.py", "models.py", "ensemble.py", "validate_submission.py"]


def clean(src: str) -> str:
    """Drop package-relative imports and __main__ blocks so modules can live in one namespace."""
    out, skip = [], False
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('if __name__ == "__main__":'):
            break
        if re.match(r"from (\.|signalscope)", line) or line.startswith("from __future__"):
            # swallow multi-line parenthesised imports
            while "(" in line and ")" not in line:
                i += 1
                line = lines[i]
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip() + "\n"


def main() -> None:
    cfg = load_config()
    tid = cfg["team_id"]
    tuned = {k: yaml.safe_load((ROOT / "configs" / "tuned" / f"{k}.yaml").read_text())["params"]
             for k in ("lgb", "xgb", "cat")}
    report = json.loads((ROOT / "artifacts" / "cv_report.json").read_text())
    manifest = json.loads((ROOT / "artifacts" / "run_manifest.json").read_text())
    weights = manifest["weights"]
    fs = cfg["feature_set"]
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: C.append(nbf.v4.new_code_cell(s))  # noqa: E731

    md(f"""# SignalScope UZ — reproducible submission notebook (team `{tid}`)

**Task.** Estimate the probability that each alert (`signal_id`) in the hidden test set is escalated
(`eskalatsiya = 1`), from the customer's transaction history. Metric: ROC-AUC.

**Approach.** One feature row per alert, built only from transactions strictly *before* the alert date
(aggregates per type × direction, 1–90-day windows, amount histograms, last-20 transactions, the pre-alert burst).
Tuned LightGBM / XGBoost / CatBoost / logistic regression are evaluated with repeated stratified 5-fold CV
(5×3, seed 42). The final prediction uses the model set chosen by the fold-wise blend check
(**{" + ".join(weights)}**), refit on all training rows with {len(cfg["final"]["seeds"])} seeds, rank-mapped to (0, 1).

**Runtime.** Full run ≈ 4 min (measured 225 s) on a 32-thread CPU, no GPU needed; `FAST_MODE=True` ≈ 1 min.
**Hardware used.** Windows 11, 32 logical CPU cores, 32 GB RAM (GPU present but not used here).

**Inputs.** Only the five provided files in `DATA_DIR`. **Output.** `OUTPUT_DIR/team_{tid}.csv` + an inline validator.""")

    code(f"""# ---- configuration -------------------------------------------------------------------------------
from pathlib import Path
DATA_DIR = Path("../data/raw") if Path("../data/raw").exists() else Path("data/raw")
# ^ put train_signals.csv, test_signals.csv, train_transactions.parquet, test_transactions.parquet,
#   sample_submission.csv here (paths are relative; no absolute paths needed)
OUTPUT_DIR = Path("outputs_notebook")
TEAM_ID = "{tid}"
SEED = {cfg["seed"]}
N_JOBS = {cfg["n_jobs"]}                  # fixed thread count -> deterministic GBDT results
FAST_MODE = False               # True = quick smoke run (1 CV repeat, LightGBM only, 1 seed)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)""")

    code("""# ---- environment ---------------------------------------------------------------------------------
# pip install numpy==2.2.6 pandas==2.3.3 polars==1.34.0 pyarrow==21.0.0 scipy==1.16.2 scikit-learn==1.7.2 \\
#             lightgbm==4.6.0 xgboost==3.0.5 catboost==1.2.8 matplotlib==3.10.6
import os, random, time, warnings, hashlib
os.environ["PYTHONHASHSEED"] = str(SEED)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, polars as pl
import lightgbm, xgboost, catboost, sklearn, scipy
random.seed(SEED); np.random.seed(SEED)
for m in (np, pd, pl, lightgbm, xgboost, catboost, sklearn, scipy):
    print(f"{m.__name__:12s} {m.__version__}")
T0 = time.time()""")

    md("## 1. Load data and audit")
    code("""train_sig = pd.read_csv(DATA_DIR / "train_signals.csv", dtype={"signal_id": str}, parse_dates=["signal_sanasi"])
test_sig = pd.read_csv(DATA_DIR / "test_signals.csv", dtype={"signal_id": str}, parse_dates=["signal_sanasi"])
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype={"signal_id": str})
tx_train = pl.read_parquet(DATA_DIR / "train_transactions.parquet")
tx_test = pl.read_parquet(DATA_DIR / "test_transactions.parquet")
print("signals", train_sig.shape, test_sig.shape, "| tx", tx_train.shape, tx_test.shape)
print("escalation rate", round(train_sig.eskalatsiya.mean(), 4))
print("dup ids", train_sig.signal_id.duplicated().sum(), test_sig.signal_id.duplicated().sum(),
      "| nulls", int(train_sig.isna().sum().sum()), int(tx_train.null_count().sum_horizontal()[0]))
print("date range train", train_sig.signal_sanasi.min().date(), train_sig.signal_sanasi.max().date(),
      "| test", test_sig.signal_sanasi.min().date(), test_sig.signal_sanasi.max().date())
assert list(sample.columns) == ["signal_id", "ehtimollik"] and set(sample.signal_id) == set(test_sig.signal_id)
for name, tx, s in [("train", tx_train, train_sig), ("test", tx_test, test_sig)]:
    j = tx.join(pl.from_pandas(s[["signal_id", "signal_sanasi"]]).with_columns(pl.col("signal_sanasi").cast(pl.Datetime("us"))), on="signal_id")
    print(name, "transactions at/after the alert date (excluded from features):",
          j.filter(pl.col("tranzaksiya_vaqti") >= pl.col("signal_sanasi")).height)""")

    md("## 2. Compact EDA")
    code("""import matplotlib.pyplot as plt
t = tx_train.join(pl.from_pandas(train_sig[["signal_id", "eskalatsiya"]]), on="signal_id")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
train_sig.eskalatsiya.value_counts().sort_index().plot.bar(ax=ax[0], color=["#2a78d6", "#eb6834"]); ax[0].set_title("class balance")
n = t.group_by(["signal_id", "eskalatsiya"]).len().to_pandas()
for y, c in [(0, "#2a78d6"), (1, "#eb6834")]:
    ax[1].hist(n.loc[n.eskalatsiya == y, "len"], bins=60, alpha=.5, density=True, color=c, label=f"y={y}")
ax[1].set_title("transactions per alert"); ax[1].legend()
b = t.filter(pl.col("tranzaksiya_turi") == "bank_otkazmasi").to_pandas()
for y, c in [(0, "#2a78d6"), (1, "#eb6834")]:
    ax[2].hist(b.loc[b.eskalatsiya == y, "miqdor_indeksi"], bins=80, alpha=.5, density=True, color=c, label=f"y={y}")
ax[2].set_title("bank transfer amount index"); ax[2].legend(); plt.tight_layout(); plt.show()
del t, n, b""")

    md("## 3. Feature engineering (inlined from `src/signalscope/features`, identical to the pipeline)")
    for mod in MODULES[:5]:
        code(f"# ---- {mod} ----\n" + clean((ROOT / "src" / "signalscope" / mod).read_text(encoding="utf-8")))
    code(f"""FAMILIES = {tuple(fs["families"])!r}
WINDOWS, LAST_K, BURST_S = {cfg["features"]["windows_days"]!r}, {cfg["features"]["last_k"]}, {fs["burst_seconds"]}
t0 = time.time()
ref = fit_reference(train_sig, tx_train, BURST_S)                       # train-only reference statistics
kw = dict(ref=ref, windows=WINDOWS, last_k=LAST_K, burst_seconds=BURST_S, calendar={fs["calendar"]}, families=FAMILIES)
F_train = build_features(train_sig, tx_train, **kw)
F_test = build_features(test_sig, tx_test, **kw)
assert list(F_train.columns) == list(F_test.columns)
X, X_test = F_train.drop(columns="signal_id"), F_test.drop(columns="signal_id")   # signal_id is NOT a feature
y = train_sig["eskalatsiya"].to_numpy()
print("features", X.shape, X_test.shape, f"{{time.time() - t0:.1f}}s")""")

    md("## 4. Validation scheme and models (inlined `cv.py`, `models.py`, `ensemble.py`)")
    for mod in MODULES[5:8]:
        code(f"# ---- {mod} ----\n" + clean((ROOT / "src" / "signalscope" / mod).read_text(encoding="utf-8")))
    code(f"""TUNED = {json.dumps(tuned, indent=1)}   # frozen Optuna results (configs/tuned/*.yaml)
REFIT_FACTOR = {cfg["final"]["refit_rounds_factor"]}
FINAL_SEEDS = {cfg["final"]["seeds"]!r}
FINAL_MODELS = {list(weights)!r}           # chosen by the fold-wise blend check in the pipeline
FINAL_WEIGHTS = {weights!r}

def factory(kind, scale=1.0):
    p = dict(TUNED.get(kind, {{}}))
    if kind in ("lgb", "xgb"): p["n_estimators"] = int(p["n_estimators"] * scale)
    if kind == "cat": p["iterations"] = int(p["iterations"] * scale)
    return {{"lgb": lambda: lgbm(p, N_JOBS), "xgb": lambda: xgb(p, N_JOBS, device="cpu"),
            "cat": lambda: catboost(p, N_JOBS, device="CPU"), "lr": lambda: logreg()}}[kind]()""")

    code(f"""# Repeated stratified K-fold: train/test cover the same dates and adversarial AUC ~0.5 -> random split is valid.
cv_models = ["lgb"] if FAST_MODE else ["lgb", "xgb", "cat", "lr"]
repeats = 1 if FAST_MODE else 3
cv_res = {{}}
for k in cv_models:
    t0 = time.time()
    r = run_cv(X, y, factory(k), 5, repeats, SEED)
    cv_res[k] = r
    print(f"{{k:4s}} fold AUC {{np.mean(r.fold_auc):.4f}} ± {{np.std(r.fold_auc):.4f}} | OOF AUC {{roc_auc_score(y, r.oof):.4f}} | {{time.time() - t0:.0f}}s")
if len(cv_res) > 1:
    ranks = to_ranks({{k: v.oof for k, v in cv_res.items()}})
    print("fold-wise blend check:", foldwise_check(ranks, y, splits(y, 5, 1, SEED + 100)))
# pipeline reference numbers (artifacts/cv_report.json) for comparison:
print("pipeline:", {json.dumps({k: v["oof_auc_of_mean_pred"] for k, v in report["models"].items()})})""")

    md("## 5. Final fit on all training data, prediction, submission file")
    code("""from scipy.stats import rankdata
seeds = FINAL_SEEDS[:1] if FAST_MODE else FINAL_SEEDS
models_used = ["lgb"] if FAST_MODE else FINAL_MODELS
score = np.zeros(len(X_test)); oof_blend = np.zeros(len(X))
for k in models_used:
    preds = []
    for sd in seeds:
        _, pt, _ = factory(k, REFIT_FACTOR)(X, y, X.iloc[:10], y[:10], X_test, sd)
        preds.append(pt)
        if k == "lr": break
    w = FINAL_WEIGHTS.get(k, 1.0) if not FAST_MODE else 1.0
    score += w * rankdata(np.mean(preds, axis=0)) / len(X_test)
    oof_blend += w * rankdata(cv_res[k].oof) / len(X)
final = rankdata(score, method="average") / (len(score) + 1)          # monotone map to (0,1)
sub = pd.DataFrame({"signal_id": test_sig["signal_id"].astype(str), "ehtimollik": final})
out_file = OUTPUT_DIR / f"team_{TEAM_ID}.csv"
sub.to_csv(out_file, index=False, float_format="%.8f", encoding="utf-8", lineterminator="\\n")
print(out_file, "sha256", hashlib.sha256(out_file.read_bytes()).hexdigest())
print("OOF AUC of the final model set:", round(roc_auc_score(y, oof_blend), 5))""")

    md("## 6. Submission validator (inlined `validate_submission.py`)")
    code("# ---- validate_submission.py ----\n" + clean((ROOT / "src" / "signalscope" / "validate_submission.py")
                                                    .read_text(encoding="utf-8")))
    code("""try:
    rep = validate(out_file, DATA_DIR / "test_signals.csv", TEAM_ID, oof=oof_blend)
    back = pd.read_csv(out_file, dtype={"signal_id": str})["ehtimollik"].to_numpy()
    assert (rankdata(back) == rankdata(final)).all(), "8-decimal rounding changed the ranking"
    print("PASS")
except AssertionError as e:
    print("FAIL:", e); raise
print(f"total runtime {time.time() - T0:.0f}s")""")

    nb["cells"] = C
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    out = ROOT / cfg["paths"]["notebooks"] / f"team_{tid}_reproducible.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
