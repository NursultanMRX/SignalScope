"""Experiment: GRU sequence model, 5-fold CV, logs to experiments.jsonl and saves OOF for blend checks."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from signalscope.cv import run_cv
from signalscope.io import ROOT, load_config, read_signals, read_tx, seed_everything
from signalscope.seqnn import build_sequences, seqnn

cfg = load_config(); seed_everything(cfg["seed"])
max_len = int(sys.argv[1]) if len(sys.argv) > 1 else 256
s = read_signals(cfg, "train"); y = s["eskalatsiya"].to_numpy()
t0 = time.time(); seq = build_sequences(s, read_tx(cfg, "train"), max_len, 300.0); print("seq", seq.shape, time.time() - t0)
X = pd.DataFrame({"_row": np.arange(len(s))})
res = run_cv(X, y, seqnn(seq, None), 5, 1, cfg["seed"], verbose=True)
rec = {"name": f"seqnn_gru_L{max_len}", **res.summary(), "runtime_s": round(time.time() - t0, 1)}
np.save(ROOT / "artifacts" / "cache" / f"oof_seqnn_L{max_len}.npy", res.oof)
open(ROOT / "artifacts" / "experiments.jsonl", "a").write(json.dumps(rec) + "\n"); print(rec)
