"""Phase 3: build features for train and test with one code path -> data/processed/features_{train,test}.parquet."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signalscope.features import build_features, fit_reference  # noqa: E402
from signalscope.io import load_config, path, read_signals, read_tx, seed_everything  # noqa: E402


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    fs = cfg["feature_set"]
    s_tr, tx_tr = read_signals(cfg, "train"), read_tx(cfg, "train")
    ref = fit_reference(s_tr, tx_tr, fs["burst_seconds"])
    out = path(cfg, "processed")
    (path(cfg, "artifacts") / "feature_reference.json").write_text(json.dumps(ref, indent=1), encoding="utf-8")
    kw = dict(ref=ref, windows=cfg["features"]["windows_days"], last_k=cfg["features"]["last_k"],
              burst_seconds=fs["burst_seconds"], calendar=fs["calendar"], families=tuple(fs["families"]))
    f_tr = build_features(s_tr, tx_tr, **kw)
    s_te, tx_te = read_signals(cfg, "test"), read_tx(cfg, "test")
    f_te = build_features(s_te, tx_te, **kw)
    assert list(f_tr.columns) == list(f_te.columns), "train/test feature columns differ"
    f_tr.to_parquet(out / "features_train.parquet", index=False)
    f_te.to_parquet(out / "features_test.parquet", index=False)
    print("features", f_tr.shape, f_te.shape)


if __name__ == "__main__":
    main()
