"""Compare the notebook-generated CSV with the pipeline CSV (ehtimollik must match within 1e-6)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.io import ROOT, load_config  # noqa: E402

cfg = load_config()
name = f"team_{cfg['team_id']}.csv"
a = pd.read_csv(ROOT / "outputs" / name, dtype={"signal_id": str})
b = pd.read_csv(ROOT / "notebooks" / "outputs_notebook" / name, dtype={"signal_id": str})
assert (a.signal_id.values == b.signal_id.values).all(), "row order / ids differ"
d = float(np.abs(a.ehtimollik.values - b.ehtimollik.values).max())
ha = hashlib.sha256((ROOT / "outputs" / name).read_bytes()).hexdigest()
hb = hashlib.sha256((ROOT / "notebooks" / "outputs_notebook" / name).read_bytes()).hexdigest()
print(f"max |pipeline - notebook| = {d:.2e}; identical bytes: {ha == hb}")
assert d <= 1e-6, "notebook and pipeline predictions differ"
