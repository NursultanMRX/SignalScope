"""Base preparation + whole-history aggregate features (volume, amount, direction, type).

All functions take the *prepared* transaction frame from `prepare_tx`, which already contains only
transactions strictly before the signal date (age_days > 0).
"""
from __future__ import annotations

import pandas as pd
import polars as pl

TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
DIRS = ["kirim", "chiqim"]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def prepare_tx(tx: pl.DataFrame, signals: pd.DataFrame, baseline: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join signal date, compute age and helper columns, and DROP every transaction at/after the signal date.

    signal_sanasi is date-only, so the alert moment is taken as 00:00:00 of that date; a transaction is
    usable only if tranzaksiya_vaqti < signal_sanasi (age_days > 0).
    """
    s = pl.from_pandas(signals[["signal_id", "signal_sanasi"]]).with_columns(
        pl.col("signal_sanasi").cast(pl.Datetime("us")))
    t = (tx.select("signal_id", "tranzaksiya_vaqti", "kirim_chiqim", "tranzaksiya_turi", "miqdor_indeksi")
         .join(s, on="signal_id", how="inner")
         .with_columns(((pl.col("signal_sanasi") - pl.col("tranzaksiya_vaqti")).dt.total_seconds() / 86400.0)
                       .alias("age_days"))
         .filter(pl.col("age_days") > 0))
    if baseline is not None:
        # de-trend: amount relative to the calendar-week x type median of all TRAIN customers (label-free)
        t = (t.with_columns(pl.col("tranzaksiya_vaqti").dt.truncate("1w").alias("_wk"))
             .join(baseline, on=["_wk", "tranzaksiya_turi"], how="left")
             .with_columns((pl.col("miqdor_indeksi") - pl.col("_base").fill_null(0.0)).alias("miqdor_indeksi"))
             .drop(["_wk", "_base"]))
    t = t.with_columns(
        pl.col("miqdor_indeksi").alias("amt"),
        pl.col("miqdor_indeksi").exp().alias("amt_exp"),
        (pl.col("kirim_chiqim") == "kirim").cast(pl.Int8).alias("is_in"),
        pl.col("age_days").floor().cast(pl.Int32).alias("day"),
        pl.col("tranzaksiya_vaqti").dt.hour().alias("hour"),
        (pl.col("tranzaksiya_vaqti").dt.weekday() - 1).alias("wday"),  # 0 = Monday
    ).with_columns(
        (pl.col("amt_exp") * (2 * pl.col("is_in") - 1)).alias("flow"),
        ((pl.col("hour") < 6)).cast(pl.Int8).alias("is_night"),
        (pl.col("wday") >= 5).cast(pl.Int8).alias("is_weekend"),
    )
    # deterministic order: by signal, time, then remaining columns as tie-breakers
    return t.sort(["signal_id", "tranzaksiya_vaqti", "kirim_chiqim", "tranzaksiya_turi", "amt"])


def _amount_block(col: str, prefix: str, cond: pl.Expr | None = None) -> list[pl.Expr]:
    c = pl.col(col) if cond is None else pl.col(col).filter(cond)
    return [
        c.len().alias(f"{prefix}_n"),
        c.sum().alias(f"{prefix}_sum"),
        c.mean().alias(f"{prefix}_mean"),
        c.std().alias(f"{prefix}_std"),
        c.max().alias(f"{prefix}_max"),
        c.min().alias(f"{prefix}_min"),
        c.median().alias(f"{prefix}_med"),
    ]


def aggregate_features(t: pl.DataFrame, thresholds: dict[str, float]) -> pl.DataFrame:
    """Whole-history aggregates. `thresholds` holds global amount quantiles computed on TRAIN only."""
    a = pl.col("amt")
    e: list[pl.Expr] = [
        pl.len().alias("n_tx"),
        pl.col("day").n_unique().alias("active_days"),
        pl.col("age_days").max().alias("first_age"),
        pl.col("age_days").min().alias("last_age"),
        a.mean().alias("amt_mean"), a.std().alias("amt_std"), a.min().alias("amt_min"), a.max().alias("amt_max"),
        a.skew().alias("amt_skew"), a.kurtosis().alias("amt_kurt"),
        pl.col("amt_exp").sum().alias("amtx_sum"), pl.col("amt_exp").mean().alias("amtx_mean"),
        pl.col("flow").sum().alias("net_flow"),
        (a > thresholds["p90"]).mean().alias("share_gt_p90"),
        (a > thresholds["p99"]).mean().alias("share_gt_p99"),
        (a > thresholds["p99"]).sum().alias("n_gt_p99"),
        (a < thresholds["p10"]).mean().alias("share_lt_p10"),
        pl.col("is_in").mean().alias("in_share"),
    ]
    e += [a.quantile(q, "linear").alias(f"amt_q{int(q * 100)}") for q in QUANTILES]
    for d in DIRS:
        cond = pl.col("kirim_chiqim") == d
        e += _amount_block("amt", f"{d}", cond)
        e.append(pl.col("amt_exp").filter(cond).sum().alias(f"{d}_amtx_sum"))
        e.append(pl.col("amt_exp").filter(cond).max().alias(f"{d}_amtx_max"))
    for ty in TYPES:
        cond = pl.col("tranzaksiya_turi") == ty
        e += _amount_block("amt", f"{ty}", cond)
        e.append(pl.col("amt_exp").filter(cond).sum().alias(f"{ty}_amtx_sum"))
        e.append(cond.mean().alias(f"{ty}_share"))
        e.append(pl.col("day").filter(cond).n_unique().alias(f"{ty}_days"))
        e.append(pl.col("age_days").filter(cond).min().alias(f"{ty}_last_age"))
        for d in DIRS:
            c2 = cond & (pl.col("kirim_chiqim") == d)
            e.append(c2.sum().alias(f"{ty}_{d}_n"))
            e.append(pl.col("amt_exp").filter(c2).sum().alias(f"{ty}_{d}_amtx_sum"))
            e.append(pl.col("amt").filter(c2).mean().alias(f"{ty}_{d}_mean"))
            e.append(pl.col("amt").filter(c2).max().alias(f"{ty}_{d}_max"))
    f = t.group_by("signal_id").agg(e)
    f = f.with_columns(
        (pl.col("first_age") - pl.col("last_age")).alias("span_days"),
        (pl.col("n_tx") / pl.col("active_days")).alias("tx_per_active_day"),
        (pl.col("active_days") / pl.col("first_age").ceil()).alias("active_day_ratio"),
        (pl.col("amt_max") - pl.col("amt_q50")).alias("amt_max_minus_med"),
        (pl.col("kirim_n") / (pl.col("chiqim_n") + 1)).alias("in_out_n_ratio"),
        (pl.col("kirim_amtx_sum") / (pl.col("chiqim_amtx_sum") + 1e-3)).alias("in_out_amtx_ratio"),
        (pl.col("chiqim_amtx_sum") / (pl.col("kirim_amtx_sum") + 1e-3)).alias("out_in_amtx_ratio"),
        (pl.col("kirim_max") - pl.col("chiqim_max")).alias("max_in_minus_max_out"),
        (pl.col("net_flow") / (pl.col("amtx_sum") + 1e-3)).alias("net_flow_norm"),
        (pl.col("naqd_share") + pl.col("xalqaro_share")).alias("cash_intl_share"),
    )
    shares = [pl.col(f"{ty}_share") for ty in TYPES]
    ent = pl.sum_horizontal([pl.when(s > 0).then(-s * s.log()).otherwise(0.0) for s in shares])
    f = f.with_columns(ent.alias("type_entropy"),
                       pl.sum_horizontal([(s > 0).cast(pl.Int8) for s in shares]).alias("type_nunique"))
    return f


def global_thresholds(t_train: pl.DataFrame) -> dict[str, float]:
    """Global amount quantiles from TRAIN transactions (pre-signal), reused unchanged for test."""
    q = t_train.select([pl.col("amt").quantile(p, "linear").alias(k)
                        for k, p in {"p10": 0.1, "p90": 0.9, "p99": 0.99}.items()])
    return {k: float(v) for k, v in q.row(0, named=True).items()}


def type_amount_edges(t_train: pl.DataFrame, n_bins: int = 8) -> dict[str, list[float]]:
    """Per-type amount bin edges (quantiles of TRAIN pre-signal transactions)."""
    qs = [i / n_bins for i in range(1, n_bins)]
    out = {}
    for ty in TYPES:
        a = t_train.filter(pl.col("tranzaksiya_turi") == ty)["amt"]
        out[ty] = [float(a.quantile(q, "linear")) for q in qs]
    return out


def amount_hist_features(t: pl.DataFrame, edges: dict[str, list[float]]) -> pl.DataFrame:
    """Share of each type's transactions falling in each (train-quantile) amount bin, per direction too.

    Captures shifts in the *shape* of the amount distribution (EDA: escalated alerts have relatively more
    small bank transfers and fewer large ones), which mean/max statistics only partly see.
    """
    e: list[pl.Expr] = []
    for ty in TYPES:
        cond = pl.col("tranzaksiya_turi") == ty
        ed = [-1e9] + edges[ty] + [1e9]
        for i in range(len(ed) - 1):
            b = (pl.col("amt") > ed[i]) & (pl.col("amt") <= ed[i + 1])
            e.append((cond & b).sum().alias(f"{ty}_bin{i}_n"))
            for d in DIRS:
                e.append((cond & b & (pl.col("kirim_chiqim") == d)).sum().alias(f"{ty}_{d}_bin{i}_n"))
    f = t.group_by("signal_id").agg(e)
    tot = t.group_by("signal_id").agg([(pl.col("tranzaksiya_turi") == ty).sum().alias(f"_{ty}_tot") for ty in TYPES])
    f = f.join(tot, on="signal_id")
    r = []
    for ty in TYPES:
        for i in range(len(edges[ty]) + 1):
            r.append((pl.col(f"{ty}_bin{i}_n") / pl.col(f"_{ty}_tot")).alias(f"{ty}_bin{i}_frac"))
            for d in DIRS:
                r.append((pl.col(f"{ty}_{d}_bin{i}_n") / pl.col(f"_{ty}_tot")).alias(f"{ty}_{d}_bin{i}_frac"))
    return f.with_columns(r).drop([c for c in f.columns if c.startswith("_")])


def burst_features(tb: pl.DataFrame) -> pl.DataFrame:
    """Aggregates over the pre-signal 'burst' segment (last few minutes before the signal date)."""
    a = pl.col("amt")
    e = [pl.len().alias("b_n"), a.mean().alias("b_amt_mean"), a.std().alias("b_amt_std"), a.max().alias("b_amt_max"),
         a.min().alias("b_amt_min"), a.median().alias("b_amt_med"), pl.col("amt_exp").sum().alias("b_amtx_sum"),
         pl.col("flow").sum().alias("b_net_flow"), pl.col("is_in").mean().alias("b_in_share")]
    for ty in TYPES:
        c = pl.col("tranzaksiya_turi") == ty
        e += [c.sum().alias(f"b_{ty}_n"), a.filter(c).mean().alias(f"b_{ty}_mean"),
              a.filter(c).max().alias(f"b_{ty}_max")]
        for d in DIRS:
            c2 = c & (pl.col("kirim_chiqim") == d)
            e += [c2.sum().alias(f"b_{ty}_{d}_n"), a.filter(c2).mean().alias(f"b_{ty}_{d}_mean")]
    return tb.group_by("signal_id").agg(e)


def burst_relative_features(f: pl.DataFrame) -> pl.DataFrame:
    """Burst segment compared with the customer's own history (needs aggregate + burst + window columns)."""
    r = [
        (pl.col("b_amt_mean") - pl.col("amt_mean")).alias("br_amt_mean_diff"),
        (pl.col("b_amt_max") - pl.col("amt_max")).alias("br_amt_max_diff"),
        ((pl.col("b_amt_mean") - pl.col("amt_mean")) / (pl.col("amt_std") + 1e-3)).alias("br_amt_mean_z"),
        (pl.col("b_n") / (pl.col("n_tx") / pl.col("first_age").ceil() + 1e-3)).alias("br_n_vs_daily_rate"),
        (pl.col("b_n") / (pl.col("n_tx") + 1)).alias("br_n_vs_hist_n"),
        (pl.col("b_in_share") - pl.col("in_share")).alias("br_in_share_diff"),
    ]
    for ty in TYPES:
        r.append((pl.col(f"b_{ty}_n") / (pl.col("b_n") + 1e-9) - pl.col(f"{ty}_share")).alias(f"br_{ty}_share_diff"))
        r.append((pl.col(f"b_{ty}_mean") - pl.col(f"{ty}_mean")).alias(f"br_{ty}_mean_diff"))
    return f.with_columns(r)


def weekly_baseline(tx_train: pl.DataFrame) -> pl.DataFrame:
    """Calendar-week x type median of miqdor_indeksi over all TRAIN transactions (no labels involved)."""
    return (tx_train.with_columns(pl.col("tranzaksiya_vaqti").dt.truncate("1w").alias("_wk"))
            .group_by(["_wk", "tranzaksiya_turi"]).agg(pl.col("miqdor_indeksi").median().alias("_base")))
