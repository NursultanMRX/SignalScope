# SignalScope UZ: project rules

Alert-escalation prediction (ROC-AUC) + public EDA website + reproducible notebook for the financial-monitoring hackathon.

## Hard rules (from the competition brief)
- No hidden labels, no external copies of target labels, no attempts to obtain organizer-only data.
- `signal_id` (and its numeric order) is never a model feature. It may be checked for correlation only to report leakage.
- Only information available at alert time: transactions at/after `signal_sanasi` (00:00) are excluded from features.
- All random seeds fixed; dependency versions pinned (`pyproject.toml` + `uv.lock`).
- Predictions are real numbers in [0, 1], never hard labels. Submission: `outputs/team_<TEAM_ID>.csv`,
  columns `signal_id,ehtimollik`, `float_format="%.8f"`, no index, UTF-8 without BOM.
- Never publish raw transactions, per-customer rows or signal ids on the website.

## Working conventions
- Environment: `uv sync`; run everything via `uv run python tasks.py <stage>` (or `make <stage>`).
- `configs/config.yaml` is the single source of truth (team_id, seeds, paths, CV, feature families).
- Tuned hyper-parameters are frozen in `configs/tuned/*.yaml`; `tasks.py all` does not re-tune.
- Every experiment is logged to `artifacts/experiments.jsonl`; keep an idea only if CV improves.
- Decisions go to `docs/DECISIONS.md`, numbers/claims to `docs/FINDINGS.md`. Site claims must trace to FINDINGS.
