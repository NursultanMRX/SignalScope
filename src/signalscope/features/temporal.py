"""Time-of-day / weekday patterns, inter-arrival gaps, daily burstiness and trends."""
from __future__ import annotations

import polars as pl

HOUR_BINS = [(0, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]


def temporal_features(t: pl.DataFrame, trend_windows: tuple[int, ...] = (30, 60, 90, 180)) -> pl.DataFrame:
    h = pl.col("hour")
    e: list[pl.Expr] = [
        pl.col("is_night").mean().alias("night_share"),
        pl.col("is_weekend").mean().alias("weekend_share"),
        h.mean().alias("hour_mean"), h.std().alias("hour_std"),
        pl.col("amt_exp").filter(pl.col("is_night") == 1).sum().alias("night_amtx_sum"),
        ((pl.col("is_night") == 1) & (pl.col("kirim_chiqim") == "chiqim")).sum().alias("night_out_n"),
        pl.col("tranzaksiya_vaqti").dt.minute().eq(0).mean().alias("round_minute_share"),
    ]
    e += [((h >= a) & (h < b)).mean().alias(f"hour_{a}_{b}_share") for a, b in HOUR_BINS]
    e += [(pl.col("wday") == d).mean().alias(f"wday{d}_share") for d in range(7)]
    # exponentially weighted activity (recency-weighted counts / amounts)
    for hl in (3, 7, 30):
        wgt = (-pl.col("age_days") * (0.6931471805599453 / hl)).exp()
        e.append(wgt.sum().alias(f"ewm{hl}_n"))
        e.append((wgt * pl.col("amt_exp")).sum().alias(f"ewm{hl}_amtx"))
        e.append((wgt * pl.col("flow")).sum().alias(f"ewm{hl}_flow"))
    # linear trend of daily count / amount over the last W days (zeros included), via closed-form sums.
    # x = day index counted towards the signal (x = W-1-day), so positive slope = activity increasing.
    for w in trend_windows:
        cur = pl.col("day") < w
        x = (w - 1 - pl.col("day")).cast(pl.Float64)
        xm, xv = (w - 1) / 2.0, (w * w - 1) / 12.0
        e.append((((x.filter(cur).sum() / w) - xm * (cur.sum() / w)) / xv).alias(f"trend{w}_n"))
        e.append((((x * pl.col("amt_exp")).filter(cur).sum() / w
                   - xm * (pl.col("amt_exp").filter(cur).sum() / w)) / xv).alias(f"trend{w}_amtx"))
    f = t.group_by("signal_id").agg(e)

    # inter-arrival gaps (hours) within each signal's history
    g = (t.select("signal_id", "tranzaksiya_vaqti", "is_in", "amt")
         .with_columns((pl.col("tranzaksiya_vaqti").diff().over("signal_id").dt.total_seconds() / 3600.0)
                       .alias("gap_h")))
    gap = pl.col("gap_h").drop_nulls()
    gf = g.group_by("signal_id").agg(
        gap.mean().alias("gap_mean"), gap.std().alias("gap_std"), gap.median().alias("gap_med"),
        gap.min().alias("gap_min"), gap.max().alias("gap_max"),
        gap.quantile(0.1, "linear").alias("gap_q10"), gap.quantile(0.9, "linear").alias("gap_q90"),
        (gap < 1).mean().alias("gap_lt1h_share"), (gap < (1 / 12)).mean().alias("gap_lt5m_share"),
        (gap > 24 * 7).sum().alias("gap_gt7d_n"),
        gap.log1p().mean().alias("gap_logmean"), gap.log1p().std().alias("gap_logstd"),
    ).with_columns((pl.col("gap_std") / pl.col("gap_mean")).alias("gap_cv"))
    # dormancy then reactivation: longest gap and how recently the activity resumed after it
    g2 = g.with_columns(pl.col("gap_h").fill_null(0).alias("gap_h"))
    idx = (g2.group_by("signal_id")
           .agg(pl.col("gap_h").max().alias("_mg"),
                pl.col("tranzaksiya_vaqti").sort_by("gap_h").last().alias("_resume_ts"),
                pl.col("tranzaksiya_vaqti").max().alias("_last_ts")))
    gf = gf.join(idx, on="signal_id").with_columns(
        ((pl.col("_last_ts") - pl.col("_resume_ts")).dt.total_seconds() / 86400.0).alias("days_active_after_max_gap"),
        ((pl.col("_mg") > 24 * 30)).cast(pl.Int8).alias("dormant_reactivated_30d"),
    ).drop(["_mg", "_resume_ts", "_last_ts"])

    # daily count distribution with zero-days included (history = observed span in whole days)
    d = t.group_by(["signal_id", "day"]).agg(pl.len().alias("c"), pl.col("amt_exp").sum().alias("ax"))
    hist = t.group_by("signal_id").agg(pl.col("day").max().add(1).alias("H"))
    dd = (d.join(hist, on="signal_id").group_by("signal_id")
          .agg(pl.col("c").sum().alias("_s"), (pl.col("c") ** 2).sum().alias("_s2"),
               pl.col("c").max().alias("daily_n_max"), pl.col("ax").max().alias("daily_amtx_max"),
               pl.col("ax").sum().alias("_ax"), pl.col("H").first().alias("_H"),
               pl.col("day").sort_by("c").last().alias("busiest_day_age"))
          .with_columns((pl.col("_s") / pl.col("_H")).alias("daily_n_mean"))
          .with_columns(((pl.col("_s2") / pl.col("_H")) - pl.col("daily_n_mean") ** 2).clip(0).sqrt()
                        .alias("daily_n_std"))
          .with_columns(((pl.col("daily_n_max") - pl.col("daily_n_mean")) / (pl.col("daily_n_std") + 1e-6))
                        .alias("burst_z"),
                        (pl.col("daily_amtx_max") / (pl.col("_ax") / pl.col("_H") + 1e-6)).alias("burst_amtx_ratio"))
          .drop(["_s", "_s2", "_ax", "_H"]))
    return f.join(gf, on="signal_id", how="left").join(dd, on="signal_id", how="left")
