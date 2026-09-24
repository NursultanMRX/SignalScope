"""Sequence model over raw pre-signal transactions (GRU + attention pooling), trained inside CV folds.

Motivation: AMEX 1st place and the EBES event-sequence benchmark (arXiv 2410.03399) both find that a
sequence network adds diversity to aggregate-feature GBDTs. Used only as a blend member.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import polars as pl

from .features.aggregates import TYPES, prepare_tx

N_FEAT = 13


def build_sequences(signals: pd.DataFrame, tx: pl.DataFrame, max_len: int, burst_seconds: float) -> np.ndarray:
    """Return float32 array [n_signals, max_len, N_FEAT], most recent transactions at the end, left-padded with 0.

    Channel 0 is a 'real token' mask (1 for a transaction, 0 for padding).
    """
    t = prepare_tx(tx, signals)
    t = t.with_columns(
        (pl.col("tranzaksiya_vaqti").diff().over("signal_id").dt.total_seconds() / 3600.0).fill_null(0.0)
        .alias("gap_h"),
        (pl.col("age_days") * 86400.0 <= burst_seconds).cast(pl.Float32).alias("is_burst"),
        pl.col("tranzaksiya_vaqti").rank("ordinal", descending=True).over("signal_id").alias("rk"),
    ).filter(pl.col("rk") <= max_len)
    hr = pl.col("hour").cast(pl.Float64) * (2 * np.pi / 24)
    feats = t.select(
        "signal_id", "rk",
        pl.lit(1.0).alias("mask"),
        *[(pl.col("tranzaksiya_turi") == ty).cast(pl.Float32).alias(ty) for ty in TYPES],
        pl.col("is_in").cast(pl.Float32),
        pl.col("amt").cast(pl.Float32),
        (pl.col("age_days").log1p() / 5.2).cast(pl.Float32).alias("lage"),
        (pl.col("gap_h").clip(0).log1p() / 5.0).cast(pl.Float32).alias("lgap"),
        hr.sin().cast(pl.Float32).alias("hs"), hr.cos().cast(pl.Float32).alias("hc"),
        ((pl.col("wday") >= 5).cast(pl.Float32)).alias("we"),
        pl.col("is_burst"),
    )
    idx = pd.Series(np.arange(len(signals)), index=signals["signal_id"].values)
    row = idx.loc[feats["signal_id"].to_numpy()].to_numpy()
    pos = max_len - feats["rk"].to_numpy()          # rk=1 (most recent) -> last position
    arr = np.zeros((len(signals), max_len, N_FEAT), dtype=np.float32)
    arr[row, pos] = feats.drop(["signal_id", "rk"]).to_numpy().astype(np.float32)
    return arr


def _net(hidden: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.GELU())
            self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.att = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

        def forward(self, x):
            m = x[..., 0:1]
            h, _ = self.gru(self.inp(x))
            a = self.att(h).masked_fill(m == 0, -1e4)
            w = torch.softmax(a, dim=1)
            att = (w * h).sum(1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(torch.cat([att, mean], 1)).squeeze(-1)
    return Net()


def seqnn(seq: np.ndarray, seq_test: np.ndarray | None, epochs: int = 12, hidden: int = 64, lr: float = 2e-3,
          batch: int = 256, device: str = "cuda"):
    """fit_predict factory for cv.run_cv. X frames only carry the row positions (column '_row')."""
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def predict(model, arr):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), 1024):
                out.append(torch.sigmoid(model(torch.from_numpy(arr[i:i + 1024]).to(device))).float().cpu().numpy())
        return np.concatenate(out)

    def fp(X_tr, y_tr, X_va, y_va, X_te, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        r_tr = X_tr["_row"].to_numpy()
        a, b = train_test_split(np.arange(len(r_tr)), test_size=0.1, stratify=y_tr, random_state=seed)
        xa, ya, xb, yb = seq[r_tr[a]], y_tr[a].astype(np.float32), seq[r_tr[b]], y_tr[b]
        model = _net(hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(a) / batch)))
        lossf = torch.nn.BCEWithLogitsLoss()
        g = np.random.default_rng(seed)
        best, best_state, best_ep = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            perm = g.permutation(len(a))
            for i in range(0, len(perm), batch):
                j = perm[i:i + batch]
                xb_t = torch.from_numpy(xa[j]).to(device)
                yb_t = torch.from_numpy(ya[j]).to(device)
                opt.zero_grad()
                loss = lossf(model(xb_t), yb_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            auc = roc_auc_score(yb, predict(model, xb))
            if auc > best:
                best, best_ep = auc, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        pv = predict(model, seq[X_va["_row"].to_numpy()])
        pt = None if (X_te is None or seq_test is None) else predict(model, seq_test)
        return pv, pt, {"best_epoch": best_ep, "inner_auc": round(best, 4)}
    return fp
