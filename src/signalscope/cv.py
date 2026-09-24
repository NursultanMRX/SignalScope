"""Single CV entry point used by every model: repeated stratified K-fold + OOF bookkeeping.

Split choice (see docs/DECISIONS.md): train and test cover the same date range and adversarial validation AUC
is ~0.50, so the test set is a random sample of the same distribution -> RepeatedStratifiedKFold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

# fit_predict(X_tr, y_tr, X_va, y_va, X_te, seed) -> (pred_va, pred_te, info)
FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame | None, int],
                      tuple[np.ndarray, np.ndarray | None, dict]]


@dataclass
class CVResult:
    oof: np.ndarray                       # mean OOF prediction over repeats
    test: np.ndarray | None               # mean test prediction over all fold models
    fold_auc: list[float]
    repeat_oof_auc: list[float]
    infos: list[dict] = field(default_factory=list)

    @property
    def oof_auc(self) -> float:
        return float(np.mean(self.repeat_oof_auc))

    def summary(self) -> dict:
        return {"fold_auc_mean": round(float(np.mean(self.fold_auc)), 5),
                "fold_auc_std": round(float(np.std(self.fold_auc)), 5),
                "oof_auc_mean_over_repeats": round(self.oof_auc, 5),
                "oof_auc_of_mean_pred": None,
                "n_folds": len(self.fold_auc)}


def splits(y: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), y))


def run_cv(X: pd.DataFrame, y: np.ndarray, fit_predict: FitPredict, n_splits: int = 5, n_repeats: int = 3,
           seed: int = 42, X_test: pd.DataFrame | None = None, verbose: bool = False) -> CVResult:
    """Run repeated stratified CV. Any preprocessing must live inside `fit_predict` (fit on train fold only)."""
    n = len(y)
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    test_sum = None if X_test is None else np.zeros(len(X_test))
    fold_auc, rep_auc, infos = [], [], []
    rep_oof = np.zeros(n)
    for i, (tr, va) in enumerate(splits(y, n_splits, n_repeats, seed)):
        p_va, p_te, info = fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], X_test, seed + i)
        rep_oof[va] = p_va
        oof_sum[va] += p_va
        oof_cnt[va] += 1
        fold_auc.append(float(roc_auc_score(y[va], p_va)))
        infos.append(info)
        if p_te is not None and test_sum is not None:
            test_sum += p_te
        if verbose:
            print(f"fold {i}: auc={fold_auc[-1]:.5f} {info}")
        if (i + 1) % n_splits == 0:
            rep_auc.append(float(roc_auc_score(y, rep_oof)))
            rep_oof = np.zeros(n)
    k = len(fold_auc)
    return CVResult(oof=oof_sum / np.maximum(oof_cnt, 1),
                    test=None if test_sum is None else test_sum / k,
                    fold_auc=fold_auc, repeat_oof_auc=rep_auc, infos=infos)
