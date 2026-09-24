"""Feature builder: one row per signal_id, computed only from transactions strictly before signal_sanasi."""
from __future__ import annotations

import pandas as pd
import polars as pl

from .aggregates import (aggregate_features, amount_hist_features, burst_features, burst_relative_features,
                         global_thresholds, prepare_tx, type_amount_edges, weekly_baseline)
from .sequence import last_k_features, passthrough_features, transition_features
from .temporal import temporal_features
from .windows import shift_features, window_features

__all__ = ["build_features", "prepare_tx", "fit_reference", "split_burst"]


def split_burst(t: pl.DataFrame, burst_seconds: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split prepared tx into (history, burst). Burst = the last `burst_seconds` before the signal date.

    EDA: ~40 transactions per alert are stamped 23:57-23:59 the evening before the signal date (a timestamp
    artefact of the synthetic generator); they are kept (they are pre-signal) but summarised separately so they
    do not distort history statistics.
    """
    is_b = pl.col("age_days") * 86400.0 <= burst_seconds
    return t.filter(~is_b), t.filter(is_b)


def fit_reference(signals_train: pd.DataFrame, tx_train: pl.DataFrame, burst_seconds: float,
                  detrend: bool = False) -> dict:
    """Global reference statistics fitted on TRAIN transactions only (reused as-is for test)."""
    base = weekly_baseline(tx_train) if detrend else None
    t, _ = split_burst(prepare_tx(tx_train, signals_train, base), burst_seconds)
    return {"thresholds": global_thresholds(t), "edges": type_amount_edges(t), "baseline": base}


def build_features(signals: pd.DataFrame, tx: pl.DataFrame, ref: dict, windows: list[int], last_k: int,
                   burst_seconds: float = 300.0, calendar: bool = False, families: tuple[str, ...] | None = None
                   ) -> pd.DataFrame:
    """Build the model matrix. Same code path for train and test; output row order = `signals` order.

    `ref` must come from `fit_reference` on TRAIN so test never influences features.
    The returned frame has `signal_id` plus float feature columns only (signal_id is NOT a feature).
    """
    fam = families or ("agg", "win", "time", "pt", "lastk", "hist", "burst", "shift")
    t_all = prepare_tx(tx, signals, ref.get("baseline"))
    t, tb = split_burst(t_all, burst_seconds)
    parts = []
    if "agg" in fam:
        parts.append(aggregate_features(t, ref["thresholds"]))
    if "win" in fam:
        parts.append(window_features(t, windows))
    if "trans" in fam:
        parts.append(transition_features(t))
    if "shift" in fam:
        parts.append(shift_features(t))
    if "trans" in fam:
        parts.append(transition_features(t))
    if "shift" in fam:
        parts.append(shift_features(t))
    if "time" in fam:
        parts.append(temporal_features(t))
    if "pt" in fam:
        parts.append(passthrough_features(t))
    if "lastk" in fam:
        parts.append(last_k_features(t, last_k))
    if "hist" in fam:
        parts.append(amount_hist_features(t, ref["edges"]))
    if "burst" in fam:
        parts.append(burst_features(tb))
    f = pl.from_pandas(signals[["signal_id"]])
    for p in parts:
        f = f.join(p, on="signal_id", how="left")
    if "agg" in fam and "burst" in fam:
        f = burst_relative_features(f)
    if calendar:
        sd = pl.from_pandas(signals[["signal_id", "signal_sanasi"]])
        f = f.join(sd.select("signal_id", pl.col("signal_sanasi").dt.month().alias("sig_month"),
                             pl.col("signal_sanasi").dt.weekday().alias("sig_wday"),
                             pl.col("signal_sanasi").dt.day().alias("sig_mday"),
                             ((pl.col("signal_sanasi") - pl.datetime(2025, 1, 1)).dt.total_days()).alias("sig_days")),
                   on="signal_id", how="left")
    df = f.to_pandas()
    num = [c for c in df.columns if c != "signal_id"]
    df[num] = df[num].astype("float64")
    # signals with no transactions in a segment: counts are 0, statistics stay NaN (GBDT handles NaN)
    cnt = [c for c in num if c.endswith("_n") or c in ("n_tx", "active_days")]
    df[cnt] = df[cnt].fillna(0.0)
    return df.reset_index(drop=True)
