# Phase 07: Bootstrap Uncertainty Quantification & Paired Model Comparison Report

> **Notice: Synthetic Educational Prototype**
> This report analyzes uncertainty across fixed model artifacts evaluated on synthetic applicant records.
> All data, feature values, bureau scores, and target default labels are mathematically generated.
> These results do NOT represent real-world credit risk, real lending policy, or credit-invisible fairness.

## 1. Executive Summary & Objective

In credit risk model evaluation, point estimates computed on a single held-out test split can obscure sampling variability.
To quantify this variability rigorously without retraining or altering production model artifacts, we perform non-parametric
paired bootstrap resampling (1,000 replicates, 95% two-sided percentile intervals,
random seed 2027).

### Key Principles
1. **Fixed Model Conditioning**: The bootstrap procedure resamples held-out test observations conditioning on existing fitted models.
   It measures sampling variance of evaluation metrics on test sets of size $N$, NOT retraining variance across new training sets.
2. **Strict Paired Resampling**: Every pairwise comparison between models evaluates predictions on the **exact same resampled observation indices**
   for every bootstrap replicate. This accounts for observation-level covariance between model predictions rather than incorrectly assuming independent samples.
3. **Zero Data Leakage / Artifact Preservation**: No model was refitted, re-tuned, or modified. Existing models and datasets are purely evaluated.

## 2. Experimental Setup & Cohort Definitions

- **Dataset Source**: `data/synthetic_applicants.csv` (generation seed 2026)
- **Held-out Test Partition**: 20% stratified test split (2,000 rows, split random state 2026)
- **Target Column**: `default_within_12_months`

### Cohorts Evaluated
| Cohort | Definition | Sample Size ($N$) | Positives (Defaults) | Negatives | Synthetic Default Rate |
|---|---|---|---|---|---|
| `overall_test_set` | All held-out synthetic test records | 2,000 | 550 | 1,450 | 0.2750 |
| `no_bureau_history` | Held-out synthetic test records with bureau_history_available == 0 | 926 | 259 | 667 | 0.2797 |

> [!NOTE]
> The `no_bureau_history` cohort is a synthetic simulation-defined subgroup (`bureau_history_available == 0`).
> It does not represent real-world credit-invisible populations, and this analysis must not be construed as a fairness or compliance assessment.

## 3. Absolute Model Metric Confidence Intervals

Point estimates and 95% two-sided percentile confidence intervals $[CI_{lower}, CI_{upper}]$ across all models and cohorts:

### Cohort: `overall_test_set`

| Model | Metric | Point Estimate | 95% CI Lower | 95% CI Upper | Replicates Valid |
|---|---|---|---|---|---|
| Traditional baseline | `roc_auc` | 0.8071 | 0.7846 | 0.8284 | 1000/1000 |
| Traditional baseline | `gini_coefficient` | 0.6142 | 0.5692 | 0.6569 | 1000/1000 |
| Traditional baseline | `ks_statistic` | 0.4569 | 0.4300 | 0.5107 | 1000/1000 |
| Traditional baseline | `average_precision` | 0.6100 | 0.5695 | 0.6572 | 1000/1000 |
| Traditional baseline | `brier_score` | 0.1519 | 0.1428 | 0.1608 | 1000/1000 |
| Traditional baseline | `log_loss` | 0.4610 | 0.4381 | 0.4845 | 1000/1000 |
| Traditional baseline | `ece_10_bins` | 0.0212 | 0.0179 | 0.0427 | 1000/1000 |
| Hybrid (traditional + alternative) | `roc_auc` | 0.9098 | 0.8948 | 0.9237 | 1000/1000 |
| Hybrid (traditional + alternative) | `gini_coefficient` | 0.8196 | 0.7896 | 0.8475 | 1000/1000 |
| Hybrid (traditional + alternative) | `ks_statistic` | 0.6667 | 0.6353 | 0.7076 | 1000/1000 |
| Hybrid (traditional + alternative) | `average_precision` | 0.8126 | 0.7843 | 0.8399 | 1000/1000 |
| Hybrid (traditional + alternative) | `brier_score` | 0.1033 | 0.0947 | 0.1115 | 1000/1000 |
| Hybrid (traditional + alternative) | `log_loss` | 0.3298 | 0.3038 | 0.3552 | 1000/1000 |
| Hybrid (traditional + alternative) | `ece_10_bins` | 0.0111 | 0.0118 | 0.0312 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `roc_auc` | 0.9063 | 0.8920 | 0.9200 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `gini_coefficient` | 0.8126 | 0.7840 | 0.8401 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `ks_statistic` | 0.6680 | 0.6349 | 0.7058 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `average_precision` | 0.7968 | 0.7661 | 0.8273 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `brier_score` | 0.1060 | 0.0965 | 0.1152 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `log_loss` | 0.3386 | 0.3112 | 0.3656 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `ece_10_bins` | 0.0180 | 0.0156 | 0.0365 | 1000/1000 |

### Cohort: `no_bureau_history`

| Model | Metric | Point Estimate | 95% CI Lower | 95% CI Upper | Replicates Valid |
|---|---|---|---|---|---|
| Traditional baseline | `roc_auc` | 0.7698 | 0.7341 | 0.8005 | 1000/1000 |
| Traditional baseline | `gini_coefficient` | 0.5396 | 0.4683 | 0.6011 | 1000/1000 |
| Traditional baseline | `ks_statistic` | 0.4095 | 0.3614 | 0.4798 | 1000/1000 |
| Traditional baseline | `average_precision` | 0.5680 | 0.5075 | 0.6297 | 1000/1000 |
| Traditional baseline | `brier_score` | 0.1643 | 0.1518 | 0.1786 | 1000/1000 |
| Traditional baseline | `log_loss` | 0.4954 | 0.4648 | 0.5302 | 1000/1000 |
| Traditional baseline | `ece_10_bins` | 0.0240 | 0.0199 | 0.0579 | 1000/1000 |
| Hybrid (traditional + alternative) | `roc_auc` | 0.9109 | 0.8921 | 0.9309 | 1000/1000 |
| Hybrid (traditional + alternative) | `gini_coefficient` | 0.8218 | 0.7843 | 0.8619 | 1000/1000 |
| Hybrid (traditional + alternative) | `ks_statistic` | 0.6668 | 0.6268 | 0.7252 | 1000/1000 |
| Hybrid (traditional + alternative) | `average_precision` | 0.8208 | 0.7791 | 0.8615 | 1000/1000 |
| Hybrid (traditional + alternative) | `brier_score` | 0.1041 | 0.0917 | 0.1165 | 1000/1000 |
| Hybrid (traditional + alternative) | `log_loss` | 0.3308 | 0.2958 | 0.3656 | 1000/1000 |
| Hybrid (traditional + alternative) | `ece_10_bins` | 0.0222 | 0.0201 | 0.0488 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `roc_auc` | 0.9017 | 0.8822 | 0.9217 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `gini_coefficient` | 0.8034 | 0.7644 | 0.8434 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `ks_statistic` | 0.6543 | 0.6120 | 0.7086 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `average_precision` | 0.7943 | 0.7474 | 0.8391 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `brier_score` | 0.1097 | 0.0965 | 0.1229 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `log_loss` | 0.3484 | 0.3098 | 0.3873 | 1000/1000 |
| Hybrid (HistGradientBoosting) | `ece_10_bins` | 0.0285 | 0.0208 | 0.0547 | 1000/1000 |

## 4. Paired Model Comparison & Performance Differences

For every bootstrap replicate $b$, the paired difference is evaluated on identical resampled indices:
$$\Delta_b = \text{metric}(M_A) - \text{metric}(M_B)$$

### Metric Direction Interpretation
- **Higher is better** (`roc_auc`, `gini_coefficient`, `ks_statistic`, `average_precision`): Positive $\Delta$ favors Model A.
- **Lower is better** (`brier_score`, `log_loss`, `ece_10_bins`): Negative $\Delta$ favors Model A.
- **Interval Excludes Zero (`interval_excludes_zero`)**: Descriptive indicator that the 95% percentile interval does not contain $0.0$.
  *(Note: This is a property of the empirical interval, not a p-value or proof of commercial/practical significance).* 

### Cohort: `overall_test_set`

| Model A | Model B | Metric | Direction | Point $\Delta$ | 95% CI Lower | 95% CI Upper | Excludes Zero? |
|---|---|---|---|---|---|---|---|
| Hybrid (traditional + alternative) | Traditional baseline | `roc_auc` | higher | +0.1027 | +0.0848 | +0.1206 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `gini_coefficient` | higher | +0.2054 | +0.1696 | +0.2411 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `ks_statistic` | higher | +0.2098 | +0.1613 | +0.2440 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `average_precision` | higher | +0.2026 | +0.1678 | +0.2350 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `brier_score` | lower | -0.0485 | -0.0565 | -0.0415 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `log_loss` | lower | -0.1312 | -0.1523 | -0.1133 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `ece_10_bins` | lower | -0.0101 | -0.0222 | +0.0048 | No |
| Hybrid (HistGradientBoosting) | Traditional baseline | `roc_auc` | higher | +0.0992 | +0.0806 | +0.1179 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `gini_coefficient` | higher | +0.1985 | +0.1612 | +0.2357 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `ks_statistic` | higher | +0.2112 | +0.1600 | +0.2413 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `average_precision` | higher | +0.1868 | +0.1492 | +0.2214 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `brier_score` | lower | -0.0458 | -0.0543 | -0.0386 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `log_loss` | lower | -0.1225 | -0.1451 | -0.1021 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `ece_10_bins` | lower | -0.0031 | -0.0193 | +0.0101 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `roc_auc` | higher | -0.0035 | -0.0087 | +0.0017 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `gini_coefficient` | higher | -0.0069 | -0.0174 | +0.0035 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `ks_statistic` | higher | +0.0013 | -0.0226 | +0.0232 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `average_precision` | higher | -0.0158 | -0.0277 | -0.0042 | Yes |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `brier_score` | lower | +0.0027 | -0.0003 | +0.0056 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `log_loss` | lower | +0.0088 | -0.0003 | +0.0177 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `ece_10_bins` | lower | +0.0070 | -0.0074 | +0.0169 | No |

### Cohort: `no_bureau_history`

| Model A | Model B | Metric | Direction | Point $\Delta$ | 95% CI Lower | 95% CI Upper | Excludes Zero? |
|---|---|---|---|---|---|---|---|
| Hybrid (traditional + alternative) | Traditional baseline | `roc_auc` | higher | +0.1411 | +0.1128 | +0.1734 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `gini_coefficient` | higher | +0.2822 | +0.2256 | +0.3467 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `ks_statistic` | higher | +0.2574 | +0.1951 | +0.3240 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `average_precision` | higher | +0.2528 | +0.2006 | +0.3047 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `brier_score` | lower | -0.0602 | -0.0718 | -0.0488 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `log_loss` | lower | -0.1646 | -0.1956 | -0.1366 | Yes |
| Hybrid (traditional + alternative) | Traditional baseline | `ece_10_bins` | lower | -0.0018 | -0.0243 | +0.0188 | No |
| Hybrid (HistGradientBoosting) | Traditional baseline | `roc_auc` | higher | +0.1319 | +0.1029 | +0.1644 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `gini_coefficient` | higher | +0.2638 | +0.2059 | +0.3288 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `ks_statistic` | higher | +0.2448 | +0.1706 | +0.3053 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `average_precision` | higher | +0.2264 | +0.1722 | +0.2794 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `brier_score` | lower | -0.0546 | -0.0671 | -0.0428 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `log_loss` | lower | -0.1470 | -0.1811 | -0.1160 | Yes |
| Hybrid (HistGradientBoosting) | Traditional baseline | `ece_10_bins` | lower | +0.0045 | -0.0248 | +0.0217 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `roc_auc` | higher | -0.0092 | -0.0161 | -0.0013 | Yes |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `gini_coefficient` | higher | -0.0184 | -0.0322 | -0.0026 | Yes |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `ks_statistic` | higher | -0.0125 | -0.0507 | +0.0155 | No |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `average_precision` | higher | -0.0264 | -0.0419 | -0.0103 | Yes |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `brier_score` | lower | +0.0056 | +0.0014 | +0.0094 | Yes |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `log_loss` | lower | +0.0176 | +0.0046 | +0.0288 | Yes |
| Hybrid (HistGradientBoosting) | Hybrid (traditional + alternative) | `ece_10_bins` | lower | +0.0063 | -0.0170 | +0.0215 | No |

## 5. Statistical Interpretation Guardrails & Limitations

1. **Conditioning Scope**: These intervals reflect test-sample variability conditional on the **fixed fitted models**.
   They do NOT capture variation from retraining models on fresh training splits or variation in synthetic data generation parameters.
2. **Paired vs Independent Evaluation**: Independent confidence intervals for two models may overlap even when paired differences
   consistently favor one model, because paired differences cancel out sample-wide difficulty variations across resampled test sets.
3. **Zero Inclusion / Exclusion**: An interval crossing zero does not establish that two models are equivalent. An interval excluding zero
   does not establish business significance or deployment readiness.
4. **No Fairness or Legal Inferences**: The models and labels are entirely synthetic. No assertions regarding algorithmic bias,
   disparate impact, or regulatory compliance (e.g. ECOA, FCRA, RBI guidelines) can be derived from this analysis.

## 6. Execution Environment & Dependencies

- **Python Version**: `3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 (clang-1600.0.26.6)]`
- **NumPy**: `2.5.4`
- **pandas**: `3.0.6`
- **scikit-learn**: `1.7.2`
- **matplotlib**: `3.11.2`

---
*Report generated automatically by `uncertainty_analysis.py`.*
