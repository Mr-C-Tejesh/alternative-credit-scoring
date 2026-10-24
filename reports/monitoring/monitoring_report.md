# Batch Inference Monitoring & Data-Drift Diagnostic Report

**Execution Mode**: `default_demonstration`  
**Reference Dataset**: `data/synthetic_applicants.csv (training partition: 80%)` (8,000 rows)  
**Incoming Batch**: `data/synthetic_applicants.csv (held-out test partition: 20%)` (2,000 rows)  
**Binning Methodology**: Reference-derived quantiles with infinite tails (10 bins, smoothing $\epsilon = 0.0001$)  
**Configured Thresholds**: PSI Warning $\ge 0.10$, PSI Critical $\ge 0.25$, Missingness Warning $\ge 5.0$ pp, Missingness Critical $\ge 15.0$ pp  

---

## Critical Methodological Limitations & Governance Notice

> [!CAUTION]
> **SYNTHETIC AND EDUCATIONAL PURPOSE ONLY**  
> All records, fitted models, probability predictions, and drift measurements in this report are strictly simulated for machine learning engineering and observability demonstration. This analysis does NOT evaluate real-world borrowers, actual financial behavior, creditworthiness, financial inclusion, or regulatory compliance.

> [!IMPORTANT]
> **MONITORING AND OBSERVABILITY BOUNDARIES**  
> 1. **Statistical drift is not equivalent to performance degradation**: A measurable shift in feature distribution or predicted probabilities indicates input divergence from the reference baseline, but does not prove model failure or increased error rates.
> 2. **Unlabeled production reality**: Without ground-truth outcome labels on incoming batches, true predictive accuracy, ROC-AUC, Brier score, and probability calibration cannot be computed.
> 3. **Absence of drift does not guarantee safety**: Low PSI values confirm distribution stability relative to reference quantiles, but do not protect against unmonitored covariates, non-stationary target relationships, or adversarial inputs.
> 4. **Heuristic thresholds**: The PSI warning (0.10) and critical (0.25) thresholds are operational rules of thumb for this engineering demonstration, NOT universal statistical laws or regulatory mandates.
> 5. **No automated credit policy**: Drift warnings are diagnostic signals for engineering investigation. They must never trigger automated credit denials, pricing changes, or unverified model weight retraining.

---

## Executive Summary

| Diagnostic Domain | Total Monitored | Status OK | Status WARNING | Status CRITICAL | Status INSUFFICIENT DATA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Monitored Predictor Features** | 14 | **14** | 0 | 0 | 0 |
| **Model Score Distributions** | 3 | **3** | 0 | 0 | 0 |

In the default demonstration, the incoming batch represents the held-out test partition (2,000 rows) evaluated against the reference training partition (8,000 rows) under the identical seed-2026 stratified split. As expected for an independent, identically distributed partition from the same synthetic generator, all 14 features and all 3 model score distributions exhibit near-zero drift and remain well within operational tolerance.

---

## Input Validation & Schema Compliance

The input batch and reference data passed all pre-scoring integrity checks:
- **Feature Frame Completeness**: All 14 expected predictor features (`ALL_FEATURES`) were present with correct numeric types and zero infinite values.
- **Target Separation**: The target variable (`default_within_12_months`) and synthetic applicant identifiers were strictly excluded from predictor matrices.
- **Label Independence**: Model feature extraction and drift diagnostics were conducted entirely without requiring or referencing target labels in the incoming batch.
- **Missingness Contract**: Traditional predictors (excluding `bureau_score`) contained zero missing values. Alternative features adhered to permitted missingness bounds.
- **Domain Constraints**: All non-null predictor values strictly complied with documented `DOMAIN_BOUNDS`.
- **DataFrame Immutability**: Source inputs and row ordering were strictly preserved without in-place mutation.

---

## Feature Distribution Drift Table

| Feature Name | Feature Group | Ref Missing | Batch Missing | Missingness Δ | Non-Missing PSI | KS Statistic | KS p-value | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `age_years` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0075 | 0.0265 | 0.2089 | **OK** |
| `monthly_income_inr` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0081 | 0.0246 | 0.2837 | **OK** |
| `employment_tenure_years` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0040 | 0.0195 | 0.5729 | **OK** |
| `debt_to_income_ratio` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0038 | 0.0230 | 0.3625 | **OK** |
| `existing_loan_count` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0025 | 0.0116 | 0.9811 | **OK** |
| `credit_utilization_ratio` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0044 | 0.0295 | 0.1220 | **OK** |
| `prior_delinquencies_24m` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0021 | 0.0095 | 0.9986 | **OK** |
| `bureau_history_available` | traditional | 0.0% | 0.0% | +0.00 pp | 0.0000 | 0.0016 | 1.0000 | **OK** |
| `bureau_score` | traditional | 46.1% | 46.3% | +0.16 pp | 0.0047 | 0.0288 | 0.4651 | **OK** |
| `utility_payment_timeliness` | alternative | 7.2% | 7.0% | -0.22 pp | 0.0091 | 0.0301 | 0.1310 | **OK** |
| `rental_payment_timeliness` | alternative | 11.5% | 13.0% | +1.51 pp | 0.0090 | 0.0194 | 0.6579 | **OK** |
| `digital_payment_regularity` | alternative | 4.8% | 5.0% | +0.25 pp | 0.0034 | 0.0158 | 0.8348 | **OK** |
| `cashflow_stability_index` | alternative | 8.4% | 7.4% | -0.94 pp | 0.0036 | 0.0213 | 0.5031 | **OK** |
| `alternative_history_months` | alternative | 0.0% | 0.0% | +0.00 pp | 0.0025 | 0.0176 | 0.6988 | **OK** |

*Note: Missingness change is evaluated independently in percentage points (pp). PSI and two-sample KS statistics are computed exclusively on non-missing numeric values. When non-missing observations are absent or below the minimum sample threshold (n < 5), PSI and KS statistics are marked as N/A.*

---

## Model Score Distribution Shifts

Predicted default probabilities ($P(\text{default} = 1)$) were generated across the three saved pipelines using their respective feature contracts:

| Model Pipeline | Reference Mean | Incoming Mean | Mean Shift | Abs Mean Shift | Score PSI | Score Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traditional baseline** | 0.2751 | 0.2660 | -0.0091 | 0.0091 | 0.0073 | **OK** |
| **Hybrid (traditional + alternative)** | 0.2751 | 0.2669 | -0.0082 | 0.0082 | 0.0066 | **OK** |
| **Hybrid (HistGradientBoosting)** | 0.2752 | 0.2664 | -0.0087 | 0.0087 | 0.0057 | **OK** |

### Score Percentile Summaries

| Model & Partition | Min | P05 | P25 | P50 (Median) | P75 | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traditional baseline (Ref)** | 0.0004 | 0.0271 | 0.1050 | 0.2202 | 0.4008 | 0.7128 | 0.9926 |
| **Traditional baseline (Batch)** | 0.0039 | 0.0263 | 0.1081 | 0.2107 | 0.3803 | 0.6917 | 0.9810 |
| **Hybrid (traditional + alternative) (Ref)** | 0.0001 | 0.0029 | 0.0261 | 0.1285 | 0.4762 | 0.9201 | 0.9998 |
| **Hybrid (traditional + alternative) (Batch)** | 0.0001 | 0.0024 | 0.0236 | 0.1179 | 0.4507 | 0.9129 | 0.9998 |
| **Hybrid (HistGradientBoosting) (Ref)** | 0.0006 | 0.0048 | 0.0260 | 0.1086 | 0.4856 | 0.9284 | 0.9989 |
| **Hybrid (HistGradientBoosting) (Batch)** | 0.0006 | 0.0043 | 0.0244 | 0.1043 | 0.4545 | 0.9205 | 0.9976 |

*Diagnostic Interpretation: Score shifts reflect changes in the model's output probability distribution. In the absence of incoming labels, a shift in mean predicted risk indicates a change in average model output, but does not provide evidence of calibration drift or degraded ranking discrimination.*

> [!NOTE]
> **Methodological Limitation (Training-Reference Score Comparison)**:
> In the default demonstration, the reference feature set is the training partition used to fit the model pipelines, while the incoming batch consists of held-out test observations. Consequently, the reference score distribution represents in-sample model predictions, whereas the incoming batch score distribution represents held-out (out-of-sample) predictions. Any observed score-distribution differences reflect both finite-sample variation and in-sample versus out-of-sample prediction behavior; they must NOT be attributed solely to population drift.

---

## Prioritized Warnings & Diagnostic Evidence

No operational warnings or critical alerts triggered. All 14 monitored features and all 3 model score distributions remain well within the configured operational warning thresholds (PSI < 0.10, |ΔMissingness| < 5.0 pp).

---

## Observations vs. Potential Explanations

1. **Observed Finding**: Largest feature PSI in the demonstration run is `utility_payment_timeliness` (PSI = 0.00906).
   *Hypothesis (Not Proven Fact)*: In an i.i.d. random split of finite synthetic data, minor differences across deciles represent standard finite-sample sampling variation rather than structural population drift.
2. **Observed Finding**: Largest absolute mean score change is `Traditional baseline` (mean change = -0.0091, score PSI = 0.00732).
   *Hypothesis (Not Proven Fact)*: The slight difference in mean predicted probability between the 8,000-row training set and 2,000-row test set reflects finite-sample variance across test applicants and in-sample versus out-of-sample prediction behavior; score PSI remains well below the 0.10 threshold.
3. **Observed Finding**: Alternative feature missingness rates in incoming batch closely match reference missingness rates (all differences within $\pm 1.5$ percentage points).
   *Hypothesis (Not Proven Fact)*: The missing-at-random synthetic data generation mechanism is uniform across the entire synthetic dataset.

---

## Reproduction Command

To reproduce this batch monitoring analysis deterministically:

```bash
.venv/bin/python monitoring.py
```

To run with custom caller-supplied batches:

```bash
.venv/bin/python monitoring.py --reference-path path/to/reference.csv --batch-path path/to/incoming_batch.csv
```

### Reproducibility Contract
- Executing `monitoring.py` with identical configuration on identical datasets produces byte-for-byte identical `feature_drift.csv`, `score_monitoring.csv`, and `monitoring_report.md`.
- `monitoring_metadata.json` is identical across repeated runs except for `generated_at_utc`, which intentionally records runtime provenance.
