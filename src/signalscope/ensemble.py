"""Rank blending with non-negative simplex weights fitted on OOF predictions, plus a fold-wise overfit check."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def to_ranks(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: rankdata(v) / len(v) for k, v in preds.items()}


def blend(ranks: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    return sum(w[k] * ranks[k] for k in w)


def fit_weights(ranks: dict[str, np.ndarray], y: np.ndarray, l2: float = 0.01) -> dict[str, float]:
    """Maximise AUC of the rank blend over the simplex (softmax parametrisation, Nelder-Mead), with a small
    L2 pull towards equal weights to avoid overfitting the OOF set."""
    keys = sorted(ranks)
    k = len(keys)
    if k == 1:
        return {keys[0]: 1.0}
    M = np.column_stack([ranks[c] for c in keys])

    def weights(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def loss(z):
        w = weights(z)
        return -roc_auc_score(y, M @ w) + l2 * np.sum((w - 1 / k) ** 2)

    best = None
    for start in [np.zeros(k)] + [np.eye(k)[i] * 2 for i in range(k)]:
        r = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    w = weights(best.x)
    return {c: float(round(v, 4)) for c, v in zip(keys, w)}


def foldwise_check(ranks: dict[str, np.ndarray], y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
                   ) -> dict:
    """Fit weights on K-1 folds of OOF, score the held-out fold; compare with equal weights and best single."""
    rows = []
    for tr, va in folds:
        w = fit_weights({k: v[tr] for k, v in ranks.items()}, y[tr])
        eq = {k: 1 / len(ranks) for k in ranks}
        rows.append({"fitted": roc_auc_score(y[va], blend({k: v[va] for k, v in ranks.items()}, w)),
                     "equal": roc_auc_score(y[va], blend({k: v[va] for k, v in ranks.items()}, eq)),
                     **{f"single_{k}": roc_auc_score(y[va], v[va]) for k, v in ranks.items()}})
    keys = rows[0].keys()
    return {k: round(float(np.mean([r[k] for r in rows])), 5) for k in keys}
