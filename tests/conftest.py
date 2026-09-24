import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signalscope.features import build_features, fit_reference  # noqa: E402
from signalscope.io import load_config, read_signals, read_tx  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sample(cfg):
    """300 training alerts and their transactions: small enough for fast tests, real data shape."""
    s = read_signals(cfg, "train").head(300).reset_index(drop=True)
    tx = read_tx(cfg, "train").filter(pl.col("signal_id").is_in(s["signal_id"].tolist()))
    return s, tx


@pytest.fixture(scope="session")
def build(cfg):
    fs = cfg["feature_set"]

    def _build(s, tx, ref=None):
        ref = ref or fit_reference(s, tx, fs["burst_seconds"])
        return build_features(s, tx, ref, cfg["features"]["windows_days"], cfg["features"]["last_k"],
                              fs["burst_seconds"], fs["calendar"], tuple(fs["families"])), ref
    return _build
