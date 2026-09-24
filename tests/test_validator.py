"""Submission validator must accept a correct file and reject each rule violation."""
import numpy as np
import pandas as pd
import pytest

from signalscope.io import raw_path
from signalscope.validate_submission import validate


@pytest.fixture()
def good(tmp_path, cfg):
    test = pd.read_csv(raw_path(cfg, "test_signals"), dtype={"signal_id": str})
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"signal_id": test.signal_id, "ehtimollik": rng.random(len(test))})
    p = tmp_path / f"team_{cfg['team_id']}.csv"
    df.to_csv(p, index=False, float_format="%.8f", lineterminator="\n")
    return p, df


def run(p, cfg):
    return validate(p, raw_path(cfg, "test_signals"), cfg["team_id"], verbose=False)


def test_valid_file_passes(good, cfg):
    assert run(good[0], cfg)["status"] == "PASS"


@pytest.mark.parametrize("breakage", ["dup", "missing", "nan", "range", "constant", "rounded", "index", "header",
                                      "bom", "name"])
def test_violations_fail(good, cfg, tmp_path, breakage):
    p, df = good
    df = df.copy()
    kw = dict(index=False, float_format="%.8f", lineterminator="\n")
    if breakage == "dup":
        df.iloc[1, 0] = df.iloc[0, 0]
    elif breakage == "missing":
        df = df.iloc[1:]
    elif breakage == "nan":
        df.iloc[3, 1] = np.nan
    elif breakage == "range":
        df.iloc[3, 1] = 1.5
    elif breakage == "constant":
        df["ehtimollik"] = 0.5
    elif breakage == "rounded":
        df["ehtimollik"] = df["ehtimollik"].round(1)
    elif breakage == "index":
        kw["index"] = True
    elif breakage == "header":
        df = df.rename(columns={"ehtimollik": "prob"})
    if breakage == "name":
        p = tmp_path / "submission.csv"
    df.to_csv(p, **kw)
    if breakage == "bom":
        p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    with pytest.raises((AssertionError, ValueError)):
        run(p, cfg)
