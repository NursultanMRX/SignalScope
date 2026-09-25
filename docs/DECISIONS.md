# Decisions log

Each entry: decision, evidence, alternatives considered. Numbers come from `artifacts/audit.json`,
`artifacts/eda/stats.json`, `artifacts/cv_report.json` and `artifacts/experiments.jsonl`.

## Project setup
- **Repo root is `F:\AlertLens-UZ`** (not a nested `signalscope-uz/` folder): the workspace already was the project
  folder; layout otherwise follows the brief.
- **uv + Python 3.11** (team request). Dependencies pinned in `pyproject.toml`, resolved in `uv.lock`;
  `requirements.txt` for the final package is exported from the lock.
- **`tasks.py` task runner + thin Makefile**: Windows has no `make`; `make <stage>` and
  `uv run python tasks.py <stage>` are equivalent.
- **TEAM_ID `040817EA`** (official) in `configs/config.yaml`; it replaced the placeholder `SSUZ7K`. The id only names the
  files (the predictions do not depend on it), so the CSV and notebook were renamed, not re-generated; the CSV bytes
  and SHA256 are unchanged.
- **kaleido 1.1.0** for PNG export: kaleido 0.2.1 hangs on Windows and 0.1.x is incompatible with plotly 6.
- **Skills.** Used: ml-experiment-scaffold, data:validate-data, dataviz, modern-web-guidance, find-skills,
  autoresearch (installed from `github/awesome-copilot`, its keep/revert loop is implemented by
  `scripts/experiment.py` + `artifacts/experiments.jsonl`). The raw sample file was named
  `sample_submission (3).csv`; copied to `data/raw/sample_submission.csv` (raw files are otherwise untouched).

## Data and leakage
- **Alert moment = 00:00 of `signal_sanasi`.** The date has no time part. Transactions with
  `tranzaksiya_vaqti >= signal_sanasi` are dropped before any feature: 3,934 train rows (2,620 alerts) and 1,456 test
  rows (1,163 alerts); 3,094 of the train rows are stamped exactly 00:00:00 on the alert date. Their presence is not
  predictive (AUC 0.502), so the strict rule costs nothing.
- **`signal_id` leakage check:** ID order vs target AUC = 0.4985. Not used as a feature. File row order vs target
  AUC = 0.508 (not used either).
- **3-minute pre-alert burst.** 7.9% of pre-alert transactions are stamped 23:57–23:59 on the evening before the alert date
  (98.9% of alerts, median 31 each). They are pre-alert by timestamp, so they are allowed; we summarise them as a
  separate feature family (`b_*`, `br_*`) instead of mixing them into history statistics. Gain +0.007 OOF AUC; each
  burst statistic alone has AUC ≤ 0.53, so it is not a disguised label leak.
- **No customer linkage.** Shared timestamps between different alerts occur at chance level (607k vs ~537k expected
  by chance) and no (timestamp, amount) pair is shared across alerts, so there is no "same customer" key to exploit.
- **Train/test split is random**: identical date ranges (2025-01-01…2026-12-31), adversarial validation AUC 0.502,
  signal-date adversarial AUC 0.500.

## Validation
- **Repeated stratified 5-fold (5×3, seed 42)**, chosen because of the random split above. A time-based split would
  only throw away data. Early stopping during tuning uses a 10% split *inside the training fold*; final CV runs use
  fixed tuned rounds, so the CV procedure equals the refit procedure.
- **Blend rule:** keep a fitted blend only if, with weights fitted on 4 folds and scored on the 5th, it beats the best
  single model. Result: fitted blend 0.64987 vs LightGBM 0.65002 → final = LightGBM alone (5 seeds, refit on all
  training rows with rounds × 1.1).

## Features
- Families kept: aggregates, recency windows, last-20 transactions, amount histograms, burst (626 features).
- Families dropped by ablation (5×2 CV, LightGBM): time-of-day/gaps (−0.0003), pass-through (−0.0019),
  behaviour shift (−0.0018), transitions/velocity/deltas (−0.0010), calendar (±0), weekly amount de-trending (−0.012).
- Global amount thresholds and per-type histogram edges are fitted on **train** transactions only and reused for test.

## Models tried (auto-research ledger)
| Experiment | OOF AUC | Kept |
|---|---|---|
| LightGBM tuned (Optuna, 71 trials) | 0.6500 | yes (final) |
| XGBoost tuned (55 trials) | 0.6495 | in blend check |
| CatBoost tuned (30 trials) | 0.6460 | in blend check |
| Logistic regression (quantile-normal + L2) | 0.6169 | in blend check |
| GRU on last 256 / 1024 transactions | 0.564 / 0.568 | no (lowers blend) |
| Transaction-level multiple-instance model | 0.555 | no |
| Naive Bayes / 2-leaf LGBM (Santander ideas; column-shuffle augmentation written but not run) | 0.560 / 0.637 | no |
| LightGBM DART | 0.607 | no |
| In-fold top-N feature selection (150/300) | 0.6478 / 0.6476 | no (neutral) |
| Duplicated rank copies of top-40 features | +0.002 to +0.003 | no: a placebo (random groups) gives the same gain, so it is a sampling artefact, not signal |

## Target of 0.8–0.9 AUC
The team asked to push towards 0.8–0.9. Every lever above was tested; the best honest CV is 0.650. Univariate best is
0.574; public repositories of other teams on the same data report 0.563–0.630. We found no legitimate source of
signal that could reach 0.8, and we did not use any leak.

## Website
- Static site in `site/` (plain HTML/CSS/JS, Plotly.js 3.1.1 from jsDelivr with SRI, PNG fallback per chart).
- Uzbek default, English toggle; light/dark follows the system with a 2-state override (modern-web-guidance
  `dark-mode` guide); below-the-fold sections use `content-visibility: auto` + `contain-intrinsic-size`.
- Only aggregated numbers are published; the build fails if a `SG_\d{6}` id appears in the output.
- Deployment: Vercel, static only (root `vercel.json`: no install/build, `outputDirectory: site`). URL in `SITE_URL.txt`.
- Restructured for the VisionX presentation (team name from `team_name` in config) into 8 numbered sections:
  overview with KPI tiles → dataset → EDA → key insights → feature engineering → model → performance → conclusion.
  Six key EDA charts carry a one/two-sentence takeaway above them; the other six sit in a collapsed "More charts".
  Five insights are phrased as question → answer → statistic. The conclusion answers three questions (behaviours,
  how the model uses them, limitations).
- Model section states that validation is split at alert level (transactions are never split on their own) and that
  the site, the CSV and the notebook use the same model; the CSV's SHA256 prefix is printed next to it.
- Performance adds a model-comparison chart (fold AUC ± std from `cv_report.json`). The ROC curve (`f16_roc`) is built
  from `artifacts/oof_*.parquet` + train labels when those exist locally; only 201 curve points per model are
  committed. Without them the chart is omitted rather than filled with placeholder numbers.
- The site rebuilds without raw data: feature count is cached in `artifacts/eda/site_meta.json`, and PNG fallbacks
  keep the committed file when kaleido has no Chrome.
- Second pass for the CTO review: the final model and its validation ROC-AUC sit in a banner at the top of the page;
  the feature-engineering section shows how many of the 626 features the final model actually uses, per group;
  key insights cut to 4; the performance section states that every number is a validation (OOF) result and shows
  the hidden-test ROC-AUC as "scored by organizers" instead of mixing it with validation.
