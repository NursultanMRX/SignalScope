"""Order-aware features: pass-through (money in, then quickly out) and the last-k transactions flattened."""
from __future__ import annotations

import polars as pl

TYPE_CODE = {"karta": 0, "bank_otkazmasi": 1, "naqd": 2, "xalqaro": 3}


def passthrough_features(t: pl.DataFrame) -> pl.DataFrame:
    """For every outgoing tx, look back at the most recent incoming tx of the same signal (as-of join)."""
    cols = ["signal_id", "tranzaksiya_vaqti", "amt", "amt_exp", "tranzaksiya_turi"]
    outs = t.filter(pl.col("is_in") == 0).select(cols).sort("tranzaksiya_vaqti")
    ins = (t.filter(pl.col("is_in") == 1).select(cols)
           .rename({"tranzaksiya_vaqti": "in_ts", "amt": "in_amt", "amt_exp": "in_amtx", "tranzaksiya_turi": "in_type"})
           .with_columns(pl.col("in_ts").alias("_k")).sort("_k"))
    j = outs.with_columns(pl.col("tranzaksiya_vaqti").alias("_k")).join_asof(
        ins, on="_k", by="signal_id", strategy="backward")
    j = j.with_columns(((pl.col("tranzaksiya_vaqti") - pl.col("in_ts")).dt.total_seconds() / 3600.0).alias("lag_h"),
                       (pl.col("amt") - pl.col("in_amt")).abs().alias("amt_diff"))
    lag = pl.col("lag_h")
    e = [
        pl.len().alias("pt_out_n"),
        lag.median().alias("pt_lag_med"), lag.mean().alias("pt_lag_mean"), lag.min().alias("pt_lag_min"),
        (lag < 1).sum().alias("pt_1h_n"), (lag < 24).sum().alias("pt_24h_n"),
        ((lag < 24) & (pl.col("amt_diff") < 0.25)).sum().alias("pt_24h_similar_n"),
        ((lag < 72) & (pl.col("amt_diff") < 0.25)).sum().alias("pt_72h_similar_n"),
        ((lag < 24) & (pl.col("amt") > 1.0)).sum().alias("pt_24h_large_n"),
        pl.col("amt_diff").filter(lag < 24).mean().alias("pt_24h_amtdiff_mean"),
        (pl.col("amt_exp").filter(lag < 24).sum()).alias("pt_24h_out_amtx"),
        ((lag < 24) & (pl.col("in_type") == "xalqaro")).sum().alias("pt_24h_from_intl_n"),
        ((lag < 24) & (pl.col("tranzaksiya_turi").is_in(["naqd", "xalqaro"]))).sum().alias("pt_24h_to_cash_intl_n"),
    ]
    f = j.group_by("signal_id").agg(e).with_columns(
        (pl.col("pt_24h_n") / pl.col("pt_out_n")).alias("pt_24h_frac"),
        (pl.col("pt_24h_similar_n") / pl.col("pt_out_n")).alias("pt_24h_similar_frac"),
    )
    return f


def last_k_features(t: pl.DataFrame, k: int) -> pl.DataFrame:
    """Last k transactions before the signal, flattened (rank 0 = most recent)."""
    r = (t.select("signal_id", "tranzaksiya_vaqti", "age_days", "amt", "is_in", "tranzaksiya_turi")
         .with_columns(pl.col("tranzaksiya_vaqti").rank("ordinal", descending=True).over("signal_id")
                       .sub(1).alias("rk"))
         .filter(pl.col("rk") < k)
         .with_columns(pl.col("tranzaksiya_turi").replace_strict(TYPE_CODE, return_dtype=pl.Int8).alias("ty")))
    wide = None
    for c, name in [("amt", "amt"), ("age_days", "age"), ("is_in", "in"), ("ty", "type")]:
        p = r.pivot(on="rk", index="signal_id", values=c, aggregate_function="first", sort_columns=True)
        p = p.rename({str(i): f"last{i}_{name}" for i in range(k) if str(i) in p.columns})
        wide = p if wide is None else wide.join(p, on="signal_id", how="left")
    # make the column set independent of data (k columns per block even if nobody has k tx)
    missing = [f"last{i}_{n}" for n in ("amt", "age", "in", "type") for i in range(k)
               if f"last{i}_{n}" not in wide.columns]
    if missing:
        wide = wide.with_columns([pl.lit(None, dtype=pl.Float64).alias(m) for m in missing])
    ordered = ["signal_id"] + [f"last{i}_{n}" for n in ("amt", "age", "in", "type") for i in range(k)]
    lastk = r.group_by("signal_id").agg(pl.col("amt").mean().alias(f"last{k}_amt_mean"),
                                        pl.col("amt").max().alias(f"last{k}_amt_max"),
                                        pl.col("is_in").mean().alias(f"last{k}_in_share"),
                                        pl.col("age_days").max().alias(f"last{k}_span_days"),
                                        (pl.col("ty") >= 2).mean().alias(f"last{k}_cash_intl_share"))
    return wide.select(ordered).cast({c: pl.Float64 for c in ordered[1:]}).join(lastk, on="signal_id", how="left")


def transition_features(t: pl.DataFrame) -> pl.DataFrame:
    """Order-aware features suggested by AmEx/AML write-ups: direction/type transitions, consecutive amount deltas,
    last-vs-history contrasts and short-horizon velocity (max transactions in rolling 5m/1h/6h/24h)."""
    s = t.select("signal_id", "tranzaksiya_vaqti", "amt", "is_in", "tranzaksiya_turi").with_columns(
        pl.col("is_in").shift(1).over("signal_id").alias("p_in"),
        pl.col("tranzaksiya_turi").shift(1).over("signal_id").alias("p_ty"),
        pl.col("amt").diff().over("signal_id").alias("d_amt"),
        pl.col("amt").rank("average").over("signal_id").alias("amt_rank"),
        pl.len().over("signal_id").alias("_n"),
    )
    e: list[pl.Expr] = [
        (pl.col("is_in") != pl.col("p_in")).mean().alias("tr_dir_switch_rate"),
        (pl.col("tranzaksiya_turi") != pl.col("p_ty")).mean().alias("tr_type_switch_rate"),
        ((pl.col("p_in") == 1) & (pl.col("is_in") == 0)).mean().alias("tr_in_to_out_rate"),
        ((pl.col("p_in") == 0) & (pl.col("is_in") == 1)).mean().alias("tr_out_to_in_rate"),
        pl.col("d_amt").abs().mean().alias("tr_damt_abs_mean"), pl.col("d_amt").std().alias("tr_damt_std"),
        (pl.col("d_amt").abs() < 0.05).mean().alias("tr_damt_similar_share"),
        pl.col("amt").last().alias("tr_last_amt"),
        (pl.col("amt").last() - pl.col("amt").mean()).alias("tr_last_minus_mean"),
        ((pl.col("amt").last() - pl.col("amt").mean()) / (pl.col("amt").std() + 1e-3)).alias("tr_last_z"),
        (pl.col("amt").tail(5).mean() - pl.col("amt").mean()).alias("tr_last5_minus_mean"),
        (pl.col("amt").tail(20).mean() - pl.col("amt").mean()).alias("tr_last20_minus_mean"),
        (pl.col("amt_rank").last() / pl.col("_n").last()).alias("tr_last_amt_pct_rank"),
        (pl.col("amt").first() - pl.col("amt").last()).alias("tr_first_minus_last"),
    ]
    types = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
    for a in types:
        for b in types:
            if a != b:
                e.append(((pl.col("p_ty") == a) & (pl.col("tranzaksiya_turi") == b)).mean().alias(f"tr_{a}_to_{b}"))
    f = s.group_by("signal_id").agg(e)
    # velocity: max number of transactions inside any rolling window of length h (per signal)
    ts = t.select("signal_id", "tranzaksiya_vaqti").with_columns(pl.lit(1).alias("one"))
    for name, per in [("5m", "5m"), ("1h", "1h"), ("6h", "6h"), ("24h", "24h")]:
        r = (ts.rolling(index_column="tranzaksiya_vaqti", period=per, group_by="signal_id")
             .agg(pl.col("one").sum().alias("c")).group_by("signal_id").agg(pl.col("c").max().alias(f"vel_max_{name}")))
        f = f.join(r, on="signal_id", how="left")
    return f
