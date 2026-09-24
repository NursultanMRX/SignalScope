"""Recency windows: activity in the last W days before the signal, and its ratio to earlier activity."""
from __future__ import annotations

import polars as pl

from .aggregates import TYPES


def window_features(t: pl.DataFrame, windows: list[int]) -> pl.DataFrame:
    age = pl.col("age_days")
    e: list[pl.Expr] = []
    for w in windows:
        cur = age <= w
        prev = (age > w) & (age <= 2 * w)
        e += [
            cur.sum().alias(f"w{w}_n"),
            prev.sum().alias(f"w{w}_prev_n"),
            pl.col("amt_exp").filter(cur).sum().alias(f"w{w}_amtx_sum"),
            pl.col("amt_exp").filter(prev).sum().alias(f"w{w}_prev_amtx_sum"),
            pl.col("amt").filter(cur).mean().alias(f"w{w}_amt_mean"),
            pl.col("amt").filter(cur).max().alias(f"w{w}_amt_max"),
            pl.col("flow").filter(cur).sum().alias(f"w{w}_net_flow"),
            pl.col("is_in").filter(cur).mean().alias(f"w{w}_in_share"),
            (cur & (pl.col("kirim_chiqim") == "chiqim")).sum().alias(f"w{w}_out_n"),
            pl.col("amt_exp").filter(cur & (pl.col("kirim_chiqim") == "chiqim")).sum().alias(f"w{w}_out_amtx_sum"),
            pl.col("day").filter(cur).n_unique().alias(f"w{w}_active_days"),
        ]
        for ty in TYPES:
            e.append((cur & (pl.col("tranzaksiya_turi") == ty)).sum().alias(f"w{w}_{ty}_n"))
    f = t.group_by("signal_id").agg(e)
    # ratios need the whole-history rate: rate = n_tx / observed history length (days)
    base = t.group_by("signal_id").agg(pl.len().alias("_n"), pl.col("age_days").max().ceil().alias("_hist"),
                                       pl.col("amt_exp").sum().alias("_ax"))
    f = f.join(base, on="signal_id")
    r = []
    for w in windows:
        r += [
            (pl.col(f"w{w}_n") / (pl.col(f"w{w}_prev_n") + 1)).alias(f"w{w}_vs_prev_n"),
            (pl.col(f"w{w}_amtx_sum") / (pl.col(f"w{w}_prev_amtx_sum") + 1)).alias(f"w{w}_vs_prev_amtx"),
            ((pl.col(f"w{w}_n") / w) / (pl.col("_n") / pl.col("_hist"))).alias(f"w{w}_rate_vs_hist"),
            ((pl.col(f"w{w}_amtx_sum") / w) / (pl.col("_ax") / pl.col("_hist"))).alias(f"w{w}_amtx_rate_vs_hist"),
            (pl.col(f"w{w}_n") / pl.col("_n")).alias(f"w{w}_share_n"),
            (pl.col(f"w{w}_out_n") / (pl.col(f"w{w}_n") + 1)).alias(f"w{w}_out_frac"),
        ]
        for ty in ("naqd", "xalqaro"):
            r.append((pl.col(f"w{w}_{ty}_n") / (pl.col(f"w{w}_n") + 1)).alias(f"w{w}_{ty}_frac"))
    # acceleration: short-window rate relative to the next longer window
    for a, b in zip(windows[:-1], windows[1:]):
        r.append(((pl.col(f"w{a}_n") / a) / (pl.col(f"w{b}_n") / b + 1e-3)).alias(f"accel_{a}_{b}"))
    return f.with_columns(r).drop(["_n", "_hist", "_ax"])


def shift_features(t: pl.DataFrame, windows: tuple[int, ...] = (7, 14, 30, 60)) -> pl.DataFrame:
    """Behaviour change: amount level / direction / type mix in the last W days minus the level before W."""
    age = pl.col("age_days")
    e: list[pl.Expr] = []
    for w in windows:
        cur, old = age <= w, age > w
        e += [(pl.col("amt").filter(cur).mean() - pl.col("amt").filter(old).mean()).alias(f"sh{w}_amt_mean"),
              (pl.col("amt").filter(cur).std() - pl.col("amt").filter(old).std()).alias(f"sh{w}_amt_std"),
              (pl.col("is_in").filter(cur).mean() - pl.col("is_in").filter(old).mean()).alias(f"sh{w}_in_share")]
        for ty in ("karta", "bank_otkazmasi", "naqd"):
            c = pl.col("tranzaksiya_turi") == ty
            e.append((pl.col("amt").filter(cur & c).mean() - pl.col("amt").filter(old & c).mean())
                     .alias(f"sh{w}_{ty}_amt_mean"))
            e.append((c.filter(cur).mean() - c.filter(old).mean()).alias(f"sh{w}_{ty}_share"))
    return t.group_by("signal_id").agg(e)
