"""Execute the reproducible notebook in FAST_MODE on a copy and check its CSV passes the validator."""
import json
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from signalscope.io import raw_path
from signalscope.validate_submission import validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_notebook_fast_mode(cfg, tmp_path):
    src = ROOT / cfg["paths"]["notebooks"] / f"team_{cfg['team_id']}_reproducible.ipynb"
    nb = nbformat.read(src, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code" and "FAST_MODE = False" in c.source:
            c.source = c.source.replace("FAST_MODE = False", "FAST_MODE = True")
            c.source = c.source.replace('Path("outputs_notebook")', f'Path(r"{tmp_path / "out"}")')
    work = tmp_path / "notebooks"
    work.mkdir()
    shutil.copytree(ROOT / "data" / "raw", tmp_path / "data" / "raw")
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    out = tmp_path / "out" / f"team_{cfg['team_id']}.csv"
    assert validate(out, raw_path(cfg, "test_signals"), cfg["team_id"], verbose=False)["status"] == "PASS"
    text = json.dumps(nb.dict())
    assert "PASS" in text and "FAIL:" not in text
