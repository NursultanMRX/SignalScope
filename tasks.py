"""Cross-platform task runner (Windows has no `make`; the Makefile just calls this).

Usage: uv run python tasks.py <stage> [<stage> ...]
Stages: audit eda features tune train predict notebook site test test-all package all
`all` = audit features eda train predict notebook site  (tuning is NOT re-run: tuned params live in configs/tuned/)
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable]
ENV = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}


def run(*args: str, cwd: Path = ROOT) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=cwd, env=ENV, check=True)


def team_id() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from signalscope.io import load_config
    return load_config()["team_id"]


def stage(name: str) -> None:
    s = ROOT / "scripts"
    if name == "audit":
        run(*PY, str(s / "run_audit.py"))
    elif name == "features":
        run(*PY, str(s / "run_features.py"))
    elif name == "eda":
        run(*PY, str(s / "run_eda.py"))
    elif name == "tune":
        for k in ("lgb", "xgb", "cat"):
            run(*PY, str(s / "tune.py"), k)
    elif name == "train":
        run(*PY, str(s / "run_train.py"))
    elif name == "predict":
        run(*PY, str(s / "run_predict.py"))
    elif name == "notebook":
        run(*PY, str(s / "build_notebook.py"))
        run(*PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=3600", f"team_{team_id()}_reproducible.ipynb", cwd=ROOT / "notebooks")
        run(*PY, str(s / "compare_notebook.py"))
    elif name == "site":
        run(*PY, str(s / "build_site.py"))
    elif name == "test":
        run(*PY, "-m", "pytest", "-q", "-m", "not slow")
    elif name == "test-all":
        run(*PY, "-m", "pytest", "-q")
    elif name == "package":
        tid = team_id()
        out = ROOT / "final_submission"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        shutil.copy(ROOT / "outputs" / f"team_{tid}.csv", out)
        shutil.copy(ROOT / "notebooks" / f"team_{tid}_reproducible.ipynb", out)
        url = (ROOT / "SITE_URL.txt").read_text().strip() if (ROOT / "SITE_URL.txt").exists() else "NOT DEPLOYED YET"
        (out / "SITE_URL.txt").write_text(url + "\n")
        req = subprocess.run(["uv", "export", "--no-hashes", "--no-dev", "--no-emit-project"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        (out / "requirements.txt").write_text(req)
        for p in sorted(out.iterdir()):
            print(p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    elif name == "all":
        for st in ("audit", "features", "eda", "train", "predict", "notebook", "site"):
            stage(st)
    else:
        raise SystemExit(f"unknown stage {name!r}\n{__doc__}")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["all"]:
        stage(a)
