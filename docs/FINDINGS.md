# Findings

All numbers are produced by the scripts and stored in `artifacts/` (`audit.json`, `eda/stats.json`,
`cv_report.json`, `experiments.jsonl`, `run_manifest.json`). The website only shows numbers listed here.

## 1. Data audit (`scripts/run_audit.py` → `artifacts/audit.json`)
| Check | Train | Test |
|---|---|---|
| Signals | 14,000 | 6,000 |
| Transactions | 6,987,663 | 3,027,575 |
| Nulls / duplicate signal ids / exact duplicate tx rows | 0 / 0 / 0 | 0 / 0 / 0 |
| Signals without tx / orphan tx | 0 / 0 | 0 / 0 |
| Alert date range | 2025-01-01 … 2026-12-31 | 2025-01-01 … 2026-12-31 |
| Transaction range | 2024-07-05 … 2026-12-31 | 2024-07-05 … 2026-12-31 |
| Timezone | naive (no tz) | naive |
| Tx at/after alert date (excluded) | 3,934 rows, 2,620 alerts | 1,456 rows, 1,163 alerts |
| Tx per alert (mean / median / max) | 499 / 461 / 2,279 | 505 / 461 / 1,985 |
| History span | median 179.5 days (max 180) | same |
| Categories | in/out 76/24%, karta 54%, bank 39%, naqd 6%, xalqaro 0.5% | same; no unseen categories |
| `miqdor_indeksi` | mean −0.13, sd 0.98, range −2.91 … 6.69, skew 0.63, 60% negative | same |

- Target: 17.18% escalated (2,405 / 14,000). Monthly rate 13.9%–21.2%, χ² p = 0.094 (no significant time effect);
  weekday χ² p = 0.79.
- `signal_id` order vs target AUC 0.4985; signal date vs target AUC 0.508 → no leakage through ids or dates.
- Adversarial validation (LightGBM on 13 aggregates, 5-fold): **AUC 0.502** → train and test are exchangeable.
- Calendar drift inside the history: median `miqdor_indeksi` falls over calendar time for every type
  (bank transfers +0.26 in 2024-07 → −0.39 in 2026-12). It is the same in train and test; de-trending it hurt CV.

## 2. EDA figures (`scripts/run_eda.py` → `artifacts/eda/`)
Statistics: Mann-Whitney U (escalated vs dismissed) with Cliff's δ = 2·AUC − 1; χ² for categorical mixes.

1. **Target (f01).** 17.2% positive; month-to-month variation is noise (χ² p = 0.094) → no calendar features.
2. **History size (f02).** Escalated alerts have slightly more transactions: median 453 vs 417, δ = 0.063,
   p = 1.3e-6. Small effect.
3. **Burst (f04).** 7.9% of pre-alert rows lie in the last 3 minutes before the alert date (23:57–23:59),
   98.9% of alerts, median 31 per alert. Burst size alone: δ = 0.025, AUC 0.513. Generator artefact, kept as features.
4. **Direction (f05).** Incoming share δ = −0.025, normalised net flow δ = −0.020: direction alone carries little.
5. **Types (f06).** Type mix is the same for both classes (transaction-level Cramér's V = 0.0035); type × direction
   lift within ±5%.
6. **Amount ratio (f07).** Strongest pattern: escalated alerts have relatively more small and fewer large bank
   transfers. Per-alert mean transfer amount median 0.064 vs 0.206, δ = −0.131, p < 1e-20. Cards weaker
   (δ = −0.040, p = 0.002), cash none (δ = −0.002, p = 0.87). No bunching below a threshold (smurfing test, polynomial
   fit residuals |z| < 6 away from the type minimum).
7. **Amount by type (f08).** Amount level strongly depends on type (xalqaro > naqd > bank > karta) → per-type stats.
8. **Time of day (f10).** Hours nearly uniform (synthetic); the 23:00 peak is the burst; weekdays identical.
9. **Univariate AUC (f11).** Best single feature `bank_otkazmasi_chiqim_mean` AUC 0.574 (as 1 − 0.426); only 26
   features exceed 0.55. Strong features are highly correlated.
10. **Drift (f12).** Max KS over top-25 features 0.022; adversarial AUC 0.502.
11. **Event time (f03).** Daily activity is ~6% higher for escalated alerts across the whole 180 days (volume ratio
    1.062) with the same curve shape; no pre-alert spike. Activity peaks ~132 days before the alert and falls towards it.
12. **Windows (f09).** Window rate ÷ history rate for 1–90 days: largest |δ| = 0.021 → recency ratios barely differ.
13. **Ablation (f13).** ΔOOF AUC when a family is removed: aggregates +0.0262, burst +0.0068, last-20 tx +0.0060,
    windows +0.0021, amount histogram +0.0014, time patterns −0.0003, pass-through −0.0019.
14. **Importance (f14).** Top features: `karta_min`, `naqd_kirim_mean`, `kirim_min`, `bank_otkazmasi_chiqim_mean`,
    `bank_otkazmasi_std`, `naqd_kirim_max`, `amt_min`, `bank_otkazmasi_bin3_frac`, `bank_otkazmasi_mean`,
    `bank_otkazmasi_min`.

## 3. Results (`artifacts/cv_report.json`, repeated stratified 5-fold × 3, seed 42)
| Model | Fold AUC mean ± std | OOF AUC (mean of repeats) |
|---|---|---|
| LightGBM (tuned) | 0.6479 ± 0.0125 | 0.6500 |
| XGBoost (tuned) | 0.6477 ± 0.0123 | 0.6495 |
| CatBoost (tuned) | 0.6409 ± 0.0114 | 0.6460 |
| Logistic regression | 0.6128 ± 0.0126 | 0.6169 |
| Rank blend (fitted weights) | – | 0.6501 (fold-wise check 0.6499 < LGBM 0.6500) |

**Final submission:** LightGBM, 5 seeds, refit on all training rows. OOF AUC 0.650.

All of the above are **validation** numbers (out-of-fold on the 14,000 labelled training alerts). The hidden-test
ROC-AUC is computed by the organizers only and is not known to us; the website labels the two separately.

**Features used by the final model** (`artifacts/feature_importance.csv`, gain of the full-train refit):
626 features built and passed to LightGBM, 463 with gain > 0 (163 never used for a split). Top 20 features carry
35% of total gain, top 100 carry 70%.

| Group | Built | Used (gain > 0) | Share of gain |
|---|---|---|---|
| Aggregates by type × direction | 130 | 118 | 46% |
| Amount histogram (`*_bin#_*`) | 192 | 128 | 28% |
| 1–90-day windows (`w#_*`, `accel_*`) | 167 | 119 | 11% |
| 3-minute burst (`b_*`, `br_*`) | 51 | 40 | 9% |
| Last 20 transactions (`last*`) | 86 | 58 | 7% |

**Submission and notebook check (static, 2026-09-25):** `outputs/team_040817EA.csv` has the right header, 6,000 unique
`SG_######` ids, values in [0, 1] with 8 decimals, LF line endings, no BOM or quotes; SHA256 `96e74eef…` matches
`run_manifest.json`. The executed notebook has cells run in order 1–18 with no errors, reports the same CV table,
writes a CSV with the same SHA256 and ends with `VALIDATION PASS`.

## 4. Verification checklist (Phase 10)
Filled in at the end of the run; see the section below.
