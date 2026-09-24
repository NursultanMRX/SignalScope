# Research notes: techniques from structurally similar competitions

Problem shape: one labelled entity (alert) -> many timestamped transactions -> binary target, ROC-AUC.
The "auto-research" skill requested by the team is not installed; research was done with web search and
recorded here. Each idea is mapped to the feature family or modelling step that implements it.

| Source | Technique | Where used here |
|---|---|---|
| AMEX Default Prediction, 1st place (LightGBM + GRU hybrid) | Flat aggregations (mean/std/min/max/last), last-minus-first and ratio features, rank features | `features/aggregates.py`, `features/windows.py` (window vs previous window ratios) |
| AMEX 1st place | GRU over raw sequences, blended with GBDT by rank | Optional sequence model (P5.6), rank blend (`ensemble.py`) |
| AMEX top solutions (summary) | XGBoost + LightGBM (incl. DART) + CatBoost ensemble, Optuna tuning, careful CV | `models.py`, `ensemble.py` |
| AMEX 15th place | Knowledge distillation NN -> LGBM | Not used (time budget); noted as next step |
| Elo Merchant / Home Credit | Time-windowed aggregates and recency ("days since last", activity in last N days) | `features/windows.py`, `features/temporal.py` |
| IEEE-CIS Fraud | Frequency encoding, careful train/test drift checks (adversarial validation) | Audit adversarial validation; frequency of repeated amounts |
| AML literature (Feedzai alert triage; rule-based AML typologies) | Cash ratio, foreign (cross-border) share, structuring (many just-below-threshold amounts), layering / pass-through (money in then quickly out), unusual time between transactions | `features/aggregates.py` (naqd/xalqaro shares), `features/sequence.py` (pass-through, gaps), structuring counts |

Sources:
- https://www.kaggle.com/competitions/amex-default-prediction/writeups/lucky-shake-1st-solution-update-github-code
- https://deepwiki.com/mlcontests/Kaggle-American-Express-Default-Prediction-1st-solution
- https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-08-30-AMEX-Kaggle-Summary/
- https://arxiv.org/pdf/2112.07508 (AML alert optimization with ML)
- https://link.springer.com/article/10.1007/s11227-023-05708-z (false-positive alert suppression in AML)
