"""Phase 1: data audit -> artifacts/audit.json (+ console summary)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from signalscope.audit import run_audit  # noqa: E402
from signalscope.io import load_config, path, raw_path, read_signals, read_tx, seed_everything  # noqa: E402


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    sig_tr, sig_te = read_signals(cfg, "train"), read_signals(cfg, "test")
    tx_tr, tx_te = read_tx(cfg, "train"), read_tx(cfg, "test")
    sample = pd.read_csv(raw_path(cfg, "sample_submission"), dtype={"signal_id": str})
    res = run_audit(cfg, sig_tr, sig_te, tx_tr, tx_te, sample)
    out = path(cfg, "artifacts") / "audit.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("tx_")}, indent=1, default=str)[:4000])
    for s in ("train", "test"):
        t = res[f"tx_{s}"]
        print(s, {k: t[k] for k in ("rows", "dup_rows_exact", "orphan_tx_rows", "signals_without_tx",
                                     "post_signal_rows", "post_signal_signals", "post_signal_max_hours_after",
                                     "at_or_after_midnight_exact", "ts_min", "ts_max")})
    print("saved", out)


if __name__ == "__main__":
    main()
