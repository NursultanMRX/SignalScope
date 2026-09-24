"""Data audit: schema, integrity, leakage and train/test shift checks.

Produces a JSON-serialisable dict. Every number reported in docs/FINDINGS.md §Audit comes from here.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _desc(s: pd.Series) -> dict:
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).round(4)
    return {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
            **{f"q{int(k * 100)}": float(v) for k, v in q.items()}}


def tx_with_age(tx: pl.DataFrame, sig: pd.DataFrame) -> pl.DataFrame:
    """Join signal date and compute age_days = signal_sanasi - tranzaksiya_vaqti (in days)."""
    s = pl.from_pandas(sig[["signal_id", "signal_sanasi"]]).with_columns(
        pl.col("signal_sanasi").cast(pl.Datetime("us")))
    return tx.join(s, on="signal_id", how="inner").with_columns(
        ((pl.col("signal_sanasi") - pl.col("tranzaksiya_vaqti")).dt.total_seconds() / 86400).alias("age_days"))


def simple_aggregates(txa: pl.DataFrame) -> pd.DataFrame:
    """Small per-signal aggregate set used only for adversarial validation."""
    t = txa.filter(pl.col("age_days") > 0)
    agg = t.group_by("signal_id").agg(
        pl.len().alias("n_tx"),
        pl.col("miqdor_indeksi").mean().alias("amt_mean"),
        pl.col("miqdor_indeksi").std().alias("amt_std"),
        pl.col("miqdor_indeksi").max().alias("amt_max"),
        (pl.col("kirim_chiqim") == "kirim").mean().alias("in_share"),
        (pl.col("tranzaksiya_turi") == "naqd").mean().alias("naqd_share"),
        (pl.col("tranzaksiya_turi") == "xalqaro").mean().alias("xalqaro_share"),
        (pl.col("tranzaksiya_turi") == "karta").mean().alias("karta_share"),
        pl.col("age_days").max().alias("first_age"),
        pl.col("age_days").min().alias("last_age"),
        (pl.col("age_days") <= 7).sum().alias("n_7d"),
        (pl.col("age_days") <= 30).sum().alias("n_30d"),
        pl.col("tranzaksiya_vaqti").dt.hour().mean().alias("hour_mean"),
    )
    return agg.to_pandas()


def adversarial_auc(train_feat: pd.DataFrame, test_feat: pd.DataFrame, seed: int, n_jobs: int) -> dict:
    X = pd.concat([train_feat, test_feat], ignore_index=True).drop(columns=["signal_id"])
    y = np.r_[np.zeros(len(train_feat)), np.ones(len(test_feat))]
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    imp = np.zeros(X.shape[1])
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed,
                               n_jobs=n_jobs, deterministic=True, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += m.feature_importances_
    top = pd.Series(imp, index=X.columns).sort_values(ascending=False).head(5)
    return {"auc": round(float(roc_auc_score(y, oof)), 4), "top_features": top.round(0).to_dict()}


def run_audit(cfg: dict, sig_tr: pd.DataFrame, sig_te: pd.DataFrame, tx_tr: pl.DataFrame, tx_te: pl.DataFrame,
              sample: pd.DataFrame) -> dict:
    out: dict = {}
    # --- signals
    for name, s in [("train", sig_tr), ("test", sig_te)]:
        out[f"signals_{name}"] = {
            "rows": len(s), "columns": list(s.columns),
            "nulls": s.isna().sum().to_dict(),
            "dup_signal_id": int(s["signal_id"].duplicated().sum()),
            "id_pattern_ok": bool(s["signal_id"].str.fullmatch(r"SG_\d{6}").all()),
            "date_min": str(s["signal_sanasi"].min().date()), "date_max": str(s["signal_sanasi"].max().date()),
        }
    out["train_test_id_overlap"] = int(len(set(sig_tr.signal_id) & set(sig_te.signal_id)))
    out["sample_submission"] = {
        "columns": list(sample.columns), "rows": len(sample),
        "same_ids_as_test": bool(set(sample.signal_id) == set(sig_te.signal_id)),
        "same_order_as_test": bool((sample.signal_id.values == sig_te.signal_id.values).all()),
    }
    # --- target
    y = sig_tr["eskalatsiya"]
    out["target"] = {"values": sorted(y.unique().tolist()), "rate": round(float(y.mean()), 5),
                     "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    by_m = sig_tr.groupby(sig_tr.signal_sanasi.dt.to_period("M"))["eskalatsiya"].agg(["mean", "size"])
    out["target"]["by_month"] = {str(k): [round(float(r["mean"]), 4), int(r["size"])] for k, r in by_m.iterrows()}
    ct = pd.crosstab(sig_tr.signal_sanasi.dt.to_period("M"), y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    out["target"]["month_chi2_p"] = float(p)
    wk = pd.crosstab(sig_tr.signal_sanasi.dt.weekday, y)
    out["target"]["weekday_chi2_p"] = float(stats.chi2_contingency(wk)[1])
    out["target"]["signal_id_order_auc"] = round(float(roc_auc_score(y, sig_tr.signal_id.str[3:].astype(int))), 4)
    out["target"]["signal_date_auc"] = round(float(roc_auc_score(y, sig_tr.signal_sanasi.astype("int64"))), 4)
    # --- transactions
    for name, tx, s in [("train", tx_tr, sig_tr), ("test", tx_te, sig_te)]:
        pdx = tx.to_pandas()
        txa = tx_with_age(tx, s)
        per = txa.group_by("signal_id").agg(pl.len().alias("n"),
                                            pl.col("age_days").max().alias("span")).to_pandas()
        post = txa.filter(pl.col("age_days") <= 0)
        a = pdx["miqdor_indeksi"]
        out[f"tx_{name}"] = {
            "rows": len(pdx), "dtypes": {c: str(t) for c, t in pdx.dtypes.items()},
            "nulls": pdx.isna().sum().to_dict(),
            "dup_rows_exact": int(pdx.duplicated().sum()),
            "ts_min": str(pdx.tranzaksiya_vaqti.min()), "ts_max": str(pdx.tranzaksiya_vaqti.max()),
            "tz": str(getattr(pdx.tranzaksiya_vaqti.dt, "tz", None)),
            "orphan_tx_rows": int((~pdx.signal_id.isin(s.signal_id)).sum()),
            "signals_without_tx": int((~s.signal_id.isin(pdx.signal_id)).sum()),
            "post_signal_rows": post.height,
            "post_signal_signals": post["signal_id"].n_unique(),
            "post_signal_max_hours_after": round(float(-post["age_days"].min() * 24), 2) if post.height else 0.0,
            "at_or_after_midnight_exact": int((txa["age_days"] == 0).sum()),
            "age_days_max": round(float(txa["age_days"].max()), 3),
            "kirim_chiqim": pdx.kirim_chiqim.value_counts().to_dict(),
            "tranzaksiya_turi": pdx.tranzaksiya_turi.value_counts().to_dict(),
            "miqdor": {**_desc(a), "skew": round(float(a.skew()), 4), "kurt": round(float(a.kurt()), 4),
                       "n_negative": int((a < 0).sum()), "n_zero": int((a == 0).sum()),
                       "n_gt_4": int((a > 4).sum()), "n_unique": int(a.nunique())},
            "tx_per_signal": _desc(per["n"].astype(float)),
            "history_span_days": _desc(per["span"]),
            "hour_dist": pdx.tranzaksiya_vaqti.dt.hour.value_counts().sort_index().to_dict(),
        }
    cats_tr = set(out["tx_train"]["tranzaksiya_turi"]) | set(out["tx_train"]["kirim_chiqim"])
    cats_te = set(out["tx_test"]["tranzaksiya_turi"]) | set(out["tx_test"]["kirim_chiqim"])
    out["unseen_categories_in_test"] = sorted(cats_te - cats_tr)
    # train vs test KS on per-signal counts
    f_tr = simple_aggregates(tx_with_age(tx_tr, sig_tr))
    f_te = simple_aggregates(tx_with_age(tx_te, sig_te))
    out["drift_ks"] = {c: round(float(stats.ks_2samp(f_tr[c].dropna(), f_te[c].dropna()).pvalue), 4)
                       for c in f_tr.columns if c != "signal_id"}
    out["adversarial_validation"] = adversarial_auc(f_tr, f_te, cfg["seed"], cfg["n_jobs"])
    # signal date as adversarial feature as well
    ds = pd.concat([sig_tr.signal_sanasi, sig_te.signal_sanasi]).astype("int64")
    out["adversarial_signal_date_auc"] = round(float(roc_auc_score(
        np.r_[np.zeros(len(sig_tr)), np.ones(len(sig_te))], ds)), 4)
    return out
