"""Config and raw-data loading helpers. All paths are relative to the repo root."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/config.yaml (or a given path) as a dict."""
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if present) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def raw_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"]["raw"] / cfg["files"][key]


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_signals(cfg: dict, split: str) -> pd.DataFrame:
    """Read train/test signals; signal_id is always a string, date parsed to datetime64."""
    df = pd.read_csv(raw_path(cfg, f"{split}_signals"), dtype={"signal_id": str})
    df["signal_sanasi"] = pd.to_datetime(df["signal_sanasi"], format="%Y-%m-%d")
    return df


def read_tx(cfg: dict, split: str) -> pl.DataFrame:
    """Read raw transactions as a polars frame."""
    return pl.read_parquet(raw_path(cfg, f"{split}_tx"))
