"""Model factories. Each returns a `fit_predict` closure compatible with `cv.run_cv`.

Early stopping never looks at the validation fold: when `inner_es` is set, 10% of the *training* fold is held
out for early stopping; otherwise a fixed number of boosting rounds (from config) is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _inner(X, y, seed, frac=0.1):
    return train_test_split(X, y, test_size=frac, stratify=y, random_state=seed)


def lgbm(params: dict, n_jobs: int, inner_es: bool = False, es_rounds: int = 200):
    import lightgbm as lgb

    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        p = {"objective": "binary", "verbose": -1, "n_jobs": n_jobs, "deterministic": True,
             "force_col_wise": True, "random_state": seed, **params}
        m = lgb.LGBMClassifier(**p)
        info = {}
        if inner_es:
            a, b, ya, yb = _inner(X_tr, y_tr, seed)
            m.fit(a, ya, eval_set=[(b, yb)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
            info["best_iter"] = int(m.best_iteration_)
        else:
            m.fit(X_tr, y_tr)
        pv = m.predict_proba(X_va)[:, 1]
        pt = None if X_te is None else m.predict_proba(X_te)[:, 1]
        info["gain"] = m.booster_.feature_importance("gain")
        return pv, pt, info
    return fp


def xgb(params: dict, n_jobs: int, device: str = "cpu", inner_es: bool = False, es_rounds: int = 200):
    import xgboost as xg

    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        p = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "device": device,
             "n_jobs": n_jobs, "random_state": seed, **params}
        info = {}
        if inner_es:
            p["early_stopping_rounds"] = es_rounds
            m = xg.XGBClassifier(**p)
            a, b, ya, yb = _inner(X_tr, y_tr, seed)
            m.fit(a, ya, eval_set=[(b, yb)], verbose=False)
            info["best_iter"] = int(m.best_iteration)
        else:
            m = xg.XGBClassifier(**p)
            m.fit(X_tr, y_tr, verbose=False)
        pv = m.predict_proba(X_va)[:, 1]
        pt = None if X_te is None else m.predict_proba(X_te)[:, 1]
        return pv, pt, info
    return fp


def catboost(params: dict, n_jobs: int, device: str = "CPU", inner_es: bool = False, es_rounds: int = 300):
    from catboost import CatBoostClassifier

    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        p = {"loss_function": "Logloss", "eval_metric": "AUC", "random_seed": seed, "verbose": 0,
             "thread_count": n_jobs, "task_type": device, "allow_writing_files": False, **params}
        info = {}
        m = CatBoostClassifier(**p)
        if inner_es:
            a, b, ya, yb = _inner(X_tr, y_tr, seed)
            m.fit(a, ya, eval_set=(b, yb), early_stopping_rounds=es_rounds, use_best_model=True)
            info["best_iter"] = int(m.get_best_iteration())
        else:
            m.fit(X_tr, y_tr)
        pv = m.predict_proba(X_va)[:, 1]
        pt = None if X_te is None else m.predict_proba(X_te)[:, 1]
        return pv, pt, info
    return fp


def logreg(C: float = 0.05):
    """Median impute + quantile-normal scaling + L2 logistic regression; all fitted on the train fold only."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        pipe = make_pipeline(SimpleImputer(strategy="median"),
                             QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed),
                             LogisticRegression(C=C, max_iter=3000))
        keep = X_tr.columns[X_tr.notna().any()]
        pipe.fit(X_tr[keep], y_tr)
        pv = pipe.predict_proba(X_va[keep])[:, 1]
        pt = None if X_te is None else pipe.predict_proba(X_te[keep])[:, 1]
        return pv, pt, {}
    return fp


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the id column; everything else is a numeric feature."""
    return df.drop(columns=[c for c in ("signal_id",) if c in df.columns])


def rank01(x: np.ndarray) -> np.ndarray:
    """Monotone map to (0,1) without ties created: average rank / (n+1)."""
    from scipy.stats import rankdata
    return rankdata(x, method="average") / (len(x) + 1)
