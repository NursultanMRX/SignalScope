# SignalScope UZ: alert escalation prediction

Hackathon project: estimate the probability that each alert in the hidden test set is escalated by a specialist,
from the customer's transaction history (synthetic data). Metric: ROC-AUC.

| Deliverable | Where |
|---|---|
| Submission | `outputs/team_<TEAM_ID>.csv` (currently `team_SSUZ7K.csv`) |
| Public EDA website | see `SITE_URL.txt` (Vercel) |
| Reproducible notebook | `notebooks/team_<TEAM_ID>_reproducible.ipynb` |
| Packaged deliverables | `final_submission/` (`uv run python tasks.py package`) |

## Results
Repeated stratified 5-fold CV (5 folds × 3 repeats, seed 42):

| Model | Fold AUC mean ± std | OOF AUC |
|---|---|---|
| **LightGBM (tuned, final)** | **0.6479 ± 0.0125** | **0.6500** |
| XGBoost (tuned) | 0.6477 ± 0.0123 | 0.6495 |
| CatBoost (tuned) | 0.6409 ± 0.0114 | 0.6460 |
| Logistic regression | 0.6128 ± 0.0126 | 0.6169 |
| Rank blend of all four | – | 0.6501 (not better fold-wise → not used) |

The final model is LightGBM refit on all training data with 5 seeds. For context, other teams' public repositories
on the same data report 0.563–0.630. Everything we tried, including what failed, is in `docs/DECISIONS.md`.

## Reproduce
Requirements: [uv](https://docs.astral.sh/uv/), Python 3.11 (uv installs it), ~16 GB RAM. No GPU needed.

```bash
uv sync                                   # pinned environment from uv.lock
# put the 5 provided files into data/raw/:
#   train_signals.csv test_signals.csv train_transactions.parquet test_transactions.parquet sample_submission.csv
uv run python tasks.py all                # audit -> features -> eda -> train -> predict -> notebook -> site
uv run python tasks.py test               # leakage / parity / determinism / validator tests
uv run python tasks.py package            # final_submission/ with the 3 deliverables + requirements.txt
```

`make <stage>` does the same on Linux/macOS. Tuned hyper-parameters are frozen in `configs/tuned/`; re-tuning is a
separate stage (`tasks.py tune`, ~40 min). Runtime of `all` on a 32-thread CPU: roughly 25 minutes
(features 10 s, EDA 3 min, CV + refit ~9 min, notebook ~12 min).

Changing the team id: edit `team_id` in `configs/config.yaml`, then `uv run python tasks.py predict notebook site package`.

## Website deployment (Vercel)
The site is plain static HTML prebuilt into `site/` by `uv run python tasks.py site` and committed. The repo-root
`vercel.json` tells Vercel to skip any install/build (no Python runtime) and serve `site/` as the output directory;
`.vercelignore` keeps the upload to `site/` only. Pushing to `main` redeploys production. After rebuilding the site,
commit `site/` so the deployment picks it up.

## How it works
1. **Audit** (`src/signalscope/audit.py`): schema, duplicates, orphans, post-alert transactions, adversarial validation.
2. **Features** (`src/signalscope/features/`): one row per alert from transactions strictly before `signal_sanasi`
   00:00. Families: per-type × direction amount statistics, 1–90-day windows, amount histograms (train-fitted edges),
   last 20 transactions, and the 3-minute pre-alert burst. 626 features.
3. **CV** (`src/signalscope/cv.py`): repeated stratified K-fold; train and test share the same date range and
   adversarial AUC is 0.502, so a random split is appropriate. All preprocessing is fitted inside the training fold.
4. **Models** (`src/signalscope/models.py`, `scripts/tune.py`): LightGBM / XGBoost / CatBoost tuned with Optuna,
   plus logistic regression. **Ensemble** (`src/signalscope/ensemble.py`): rank blend kept only if it wins fold-wise.
5. **Submission** (`scripts/run_predict.py`, `src/signalscope/validate_submission.py`): rank-mapped to (0, 1),
   8 decimals, validated (header, ids, range, ties, BOM, filename), SHA256 in `artifacts/run_manifest.json`.

## Folder map
```
configs/        config.yaml (seeds, paths, CV, features) + tuned/ (frozen Optuna params)
src/signalscope io, audit, features/, cv, models, ensemble, seqnn, viz, validate_submission
scripts/        run_audit, run_features, run_eda, tune, run_train, run_predict, build_notebook, build_site, experiment
notebooks/      team_<TEAM_ID>_reproducible.ipynb (self-contained, executed)
artifacts/      audit.json, eda/, cv_report.json, oof/test predictions, experiments.jsonl, run_manifest.json
site/ site_src/ static website (output) and its template/CSS/JS
docs/           DECISIONS.md, FINDINGS.md, RESEARCH.md
tests/          leakage, parity, determinism, validator, notebook tests
```

## Limitations
- The signal is weak: the best single feature has AUC 0.574 and the model 0.650. We found no legitimate way to go
  much higher (see `docs/DECISIONS.md`, "Target of 0.8–0.9 AUC").
- No counterparties or customer ids, so no graph features. `miqdor_indeksi` is a standardised index, not money.
- The 3-minute pre-alert burst looks like a generator artefact; it is used because it is pre-alert by timestamp.
- `TEAM_ID` is still the placeholder `SSUZ7K`.
