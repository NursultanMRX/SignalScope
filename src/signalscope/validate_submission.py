"""Submission validator (checks a-g of the competition rules). Usage: python -m signalscope.validate_submission <csv>"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def validate(csv_path: str | Path, test_signals_path: str | Path, team_id: str,
             oof: np.ndarray | None = None, verbose: bool = True) -> dict:
    p = Path(csv_path)
    report: dict = {}
    # g) raw text checks
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "file has a UTF-8 BOM"
    text = raw.decode("utf-8")
    lines = text.splitlines()
    assert lines[0] == "signal_id,ehtimollik", f"first line is {lines[0]!r}"
    assert '"' not in text, "quoted fields found"
    # b) filename
    assert re.fullmatch(r"team_.+\.csv", p.name), f"bad filename {p.name}"
    assert p.name == f"team_{team_id}.csv", f"filename {p.name} != team_{team_id}.csv"
    # a/e) header, columns, no index
    df = pd.read_csv(p, dtype={"signal_id": str})
    assert list(df.columns) == ["signal_id", "ehtimollik"], f"columns {list(df.columns)}"
    assert all(len(line.split(",")) == 2 for line in lines), "a row does not have exactly 2 fields"
    # c) ids
    test = pd.read_csv(test_signals_path, dtype={"signal_id": str})
    assert df["signal_id"].notna().all() and (df["signal_id"].str.len() > 0).all(), "blank ids"
    assert not df["signal_id"].duplicated().any(), "duplicate ids"
    assert len(df) == len(test), f"rows {len(df)} != test {len(test)}"
    assert set(df["signal_id"]) == set(test["signal_id"]), "id set differs from test_signals.csv"
    pat = re.compile(r"SG_\d{6}")
    assert df["signal_id"].map(lambda s: bool(pat.fullmatch(s))).all(), "id format differs from test file"
    # d) values
    v = pd.to_numeric(df["ehtimollik"], errors="raise").to_numpy(dtype=float)
    assert np.isfinite(v).all(), "NaN/inf predictions"
    assert (v >= 0).all() and (v <= 1).all(), "predictions outside [0,1]"
    # f) not constant, few ties
    uniq = len(np.unique(v)) / len(v)
    assert uniq >= 0.95, f"unique-value ratio {uniq:.3f} < 0.95"
    q = np.quantile(v, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    report.update(rows=len(df), unique_ratio=round(uniq, 5), mean=round(float(v.mean()), 5),
                  quantiles=[round(float(x), 5) for x in q])
    if oof is not None:
        from scipy.stats import ks_2samp
        from scipy.stats import rankdata
        # compare shapes after rank-normalising both (the submission is rank-mapped)
        ks = ks_2samp(rankdata(oof) / (len(oof) + 1), v)
        report["ks_vs_oof_rank"] = round(float(ks.statistic), 4)
        assert ks.statistic < 0.05, "test prediction distribution differs from OOF distribution"
    report["status"] = "PASS"
    if verbose:
        print("VALIDATION PASS", report)
    return report


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from signalscope.io import load_config, raw_path

    cfg = load_config()
    validate(sys.argv[1], raw_path(cfg, "test_signals"), cfg["team_id"])
