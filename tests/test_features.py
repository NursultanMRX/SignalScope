"""Feature builder guarantees: no post-signal information, train/test parity, determinism, no id feature."""
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl


def test_future_transactions_do_not_change_features(sample, build):
    s, tx = sample
    base, ref = build(s, tx)
    # inject extreme transactions AT and AFTER each alert date; they must be ignored completely
    sd = pl.from_pandas(s[["signal_id", "signal_sanasi"]]).with_columns(pl.col("signal_sanasi").cast(pl.Datetime("us")))
    fut = pl.concat([
        sd.select("signal_id", (pl.col("signal_sanasi") + timedelta(hours=h)).alias("tranzaksiya_vaqti"))
        for h in (0, 1, 30, 24 * 20)
    ]).with_columns(pl.lit("chiqim").alias("kirim_chiqim"), pl.lit("xalqaro").alias("tranzaksiya_turi"),
                    pl.lit(6.5).alias("miqdor_indeksi"))
    tx2 = pl.concat([tx, fut.select(tx.columns).cast(tx.schema)])
    leaked, _ = build(s, tx2, ref)
    pd.testing.assert_frame_equal(base, leaked)


def test_train_and_test_code_paths_are_identical(sample, build):
    s, tx = sample
    a, ref = build(s, tx)
    # same rows passed as if they were the test split (shuffled order) must give the same features per signal
    s_shuf = s.sample(frac=1.0, random_state=0).reset_index(drop=True)
    b, _ = build(s_shuf.drop(columns="eskalatsiya"), tx, ref)
    b = b.set_index("signal_id").loc[a["signal_id"]].reset_index()
    pd.testing.assert_frame_equal(a, b)


def test_features_are_deterministic(sample, build):
    s, tx = sample
    a, _ = build(s, tx)
    b, _ = build(s, tx.sample(fraction=1.0, shuffle=True, seed=1))   # input row order must not matter
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_signal_id_not_a_feature_and_output_shape(sample, build):
    s, tx = sample
    f, _ = build(s, tx)
    assert f.columns[0] == "signal_id"
    assert (f["signal_id"].values == s["signal_id"].values).all()
    assert all(np.issubdtype(t, np.floating) for t in f.dtypes.iloc[1:])
    assert not any("signal_id" in c for c in f.columns[1:])
