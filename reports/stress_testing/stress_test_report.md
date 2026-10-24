# Stress Testing & Input-Degradation Sensitivity Analysis Report

> [!IMPORTANT]
> **SYNTHETIC AND EDUCATIONAL PURPOSE ONLY**
> All data, models, and credit-risk scores are generated exclusively for machine learning pipeline
> validation and educational demonstration. The synthetic target formulation deliberately incorporates
> alternative-data features. This analysis evaluates model-input sensitivity on controlled held-out
> observations; it does **not** constitute empirical evidence of real-world underwriting accuracy,
> fairness, demographic inclusion, or lending suitability. Labels are preserved intentionally to isolate
> input sensitivity conditional on fixed records, and do not represent shifted real-world populations.

## Executive Summary

This report presents findings from **Phase 08** controlled input-degradation stress testing across **35 distinct scenarios**.
The experiment evaluated **3 serialized models** on **2,000 held-out synthetic test records**
(550 defaults, 1,450 non-defaults) across **7 standard credit evaluation metrics**
(totaling **735 evaluations**).

### Key Empirical Findings

1. **Traditional Baseline as Negative Control**: Because the traditional model uses only traditional bureau and demographic features,
   its predictions and evaluation metrics remained completely untouched (0.0000 delta and 0.0000 probability change across all 34 stress scenarios).
   This confirms that all observed degradations in hybrid models are strictly attributable to their reliance on alternative data.
2. **Total Alternative-Data Dropout as an Important Vulnerability**: Total loss of the alternative data source (`joint_missingness_rate_1.00`)
   produced the largest overall degradation across both hybrid models. The Hybrid HistGradientBoosting model experienced a ROC-AUC decline
   from **0.9063** to **0.7682** (degradation: +0.1382),
   while Hybrid Logistic Regression declined from **0.9098** to **0.7820**.
   Crucially, while both hybrid pipelines continue producing predictions due to median imputation and missingness indicators,
   their observed discrimination falls below the standalone traditional model's clean ROC-AUC (**0.8071**).
   The hybrid models do not automatically switch to the traditional baseline architecture; median imputation and existing missingness
   indicators are inference preprocessing behavior, not a model-selection fallback. Total alternative-source loss is therefore an
   important failure scenario resulting in sub-baseline performance, not proof of robust fallback performance.
3. **Bounded Sensitivity to Measurement Noise**: Injecting additive zero-mean Gaussian noise up to 20% of synthetic domain range
   caused modest discrimination declines (ROC-AUC degradation < 0.02 across all single-feature continuous noise scenarios).
   Cashflow stability and rental timeliness were the most noise-sensitive continuous alternative features.
4. **Apparent Inverted Deltas and Regularization Effects**: In 23 of 735 evaluations (3.1%), direction-adjusted degradation was slightly negative
   (e.g., ECE improving by ~0.004). This occurs because median imputation and clipping mildly pull extreme prediction probabilities toward
   the baseline default rate, occasionally reducing bin-level calibration error in overconfident tree leaves.

---

## Methodology & Experimental Design

### Evaluation Protocol
- **Dataset**: `data/synthetic_applicants.csv` (10,000 rows).
- **Held-Out Test Split**: Fraction = `0.20`, Split Seed = `2026` (stratified on `default_within_12_months`).
- **Test Cohort**: 2,000 rows (550 defaults, 1,450 non-defaults).
- **Models Evaluated**:
  1. `Traditional baseline` (Logistic Regression on traditional bureau features only).
  2. `Hybrid (traditional + alternative)` (Logistic Regression on all 14 features).
  3. `Hybrid (HistGradientBoosting)` (Gradient-boosted decision trees on all 14 features).
- **Degradation Direction Adjustment**:
  - For higher-is-better metrics (`roc_auc`, `gini_coefficient`, `ks_statistic`, `average_precision`):
    $$\text{Degradation} = \text{Metric}_{\text{clean}} - \text{Metric}_{\text{scenario}}$$
  - For lower-is-better metrics (`brier_score`, `log_loss`, `ece_10_bins`):
    $$\text{Degradation} = \text{Metric}_{\text{scenario}} - \text{Metric}_{\text{clean}}$$
  - A **positive degradation** indicates performance worsened; a **negative degradation** indicates apparent improvement.

### Scenario Matrix Summary

| Scenario Family | Count | Description | Perturbed Inputs |
| :--- | :---: | :--- | :--- |
| `clean_reference` | 1 | Original uncorrupted held-out test data | None |
| `joint_missingness` | 4 | Simultaneous MCAR dropout at 25%, 50%, 75%, 100% | All 5 alternative features |
| `individual_missingness` | 15 | Single-feature MCAR dropout at 25%, 50%, 100% | 1 alternative feature per scenario |
| `measurement_noise` | 15 | Bounded Gaussian noise (5%, 10%, 20% range / 1, 3, 6 months) | 1 alternative feature per scenario |
| **Total Scenarios** | **35** | | |

---

## Clean Reference Baseline Performance

The clean reference reproduces the unperturbed held-out evaluation verified in Phase 07:

| Model | ROC-AUC | Gini | KS Stat | Avg Precision | Brier Score | Log Loss | ECE (10 bins) | Mean Pred Prob |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Traditional baseline | 0.8071 | 0.6142 | 0.4569 | 0.6100 | 0.1519 | 0.4610 | 0.0212 | 0.2660 |
| Hybrid (traditional + alternative) | 0.9098 | 0.8196 | 0.6667 | 0.8126 | 0.1033 | 0.3298 | 0.0111 | 0.2669 |
| Hybrid (HistGradientBoosting) | 0.9063 | 0.8126 | 0.6680 | 0.7968 | 0.1060 | 0.3386 | 0.0180 | 0.2664 |

---

## Largest Observed Performance Degradations

The top 10 performance degradations ranked across all models, scenarios, and metrics by direction-adjusted degradation:

| Rank | Model | Scenario ID | Metric | Metric Direction | Clean Metric | Scenario Metric | Raw Delta | Direction-Adjusted Degradation |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_1.00` | `gini_coefficient` | higher | 0.8126 | 0.5363 | -0.2763 | **+0.2763** |
| 2 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_1.00` | `average_precision` | higher | 0.7968 | 0.5323 | -0.2645 | **+0.2645** |
| 3 | Hybrid (traditional + alternative) | `joint_missingness_rate_1.00` | `gini_coefficient` | higher | 0.8196 | 0.5640 | -0.2555 | **+0.2555** |
| 4 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_1.00` | `ks_statistic` | higher | 0.6680 | 0.4152 | -0.2529 | **+0.2529** |
| 5 | Hybrid (traditional + alternative) | `joint_missingness_rate_1.00` | `average_precision` | higher | 0.8126 | 0.5677 | -0.2449 | **+0.2449** |
| 6 | Hybrid (traditional + alternative) | `joint_missingness_rate_1.00` | `ks_statistic` | higher | 0.6667 | 0.4436 | -0.2231 | **+0.2231** |
| 7 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_0.75` | `gini_coefficient` | higher | 0.8126 | 0.6406 | -0.1721 | **+0.1721** |
| 8 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_1.00` | `log_loss` | lower | 0.3386 | 0.5034 | +0.1648 | **+0.1648** |
| 9 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_0.75` | `ks_statistic` | higher | 0.6680 | 0.5072 | -0.1608 | **+0.1608** |
| 10 | Hybrid (HistGradientBoosting) | `joint_missingness_rate_0.75` | `average_precision` | higher | 0.7968 | 0.6370 | -0.1597 | **+0.1597** |

---

## Model-by-Model Stress Evaluation

### 1. Traditional Baseline (Negative Control)
- In all 35 scenarios, the traditional model was supplied only with `TRADITIONAL_FEATURES`.
- Because stress corruptions only affect alternative features, the traditional model served as an invariant negative control.
- Maximum observed probability change: `0.0000`.
- Maximum observed metric degradation: `0.0000`.

### 2. Hybrid Logistic Regression

#### Impact of Joint Alternative Dropout
| Joint Dropout Rate | ROC-AUC | Gini | KS Stat | Brier Score | Log Loss | Mean Prob Delta | P95 Prob Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 25% | 0.8870 | 0.7739 | 0.6154 | 0.1154 | 0.3647 | 0.0446 | 0.2769 |
| 50% | 0.8612 | 0.7224 | 0.5643 | 0.1296 | 0.4017 | 0.0952 | 0.3943 |
| 75% | 0.8325 | 0.6650 | 0.5236 | 0.1430 | 0.4375 | 0.1424 | 0.4429 |
| 100% | 0.7820 | 0.5640 | 0.4436 | 0.1607 | 0.4866 | 0.1894 | 0.4778 |

### 3. Hybrid HistGradientBoosting

#### Impact of Joint Alternative Dropout
| Joint Dropout Rate | ROC-AUC | Gini | KS Stat | Brier Score | Log Loss | Mean Prob Delta | P95 Prob Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 25% | 0.8846 | 0.7691 | 0.6179 | 0.1183 | 0.3720 | 0.0430 | 0.3104 |
| 50% | 0.8541 | 0.7083 | 0.5558 | 0.1336 | 0.4103 | 0.0925 | 0.4626 |
| 75% | 0.8203 | 0.6406 | 0.5072 | 0.1489 | 0.4526 | 0.1379 | 0.5009 |
| 100% | 0.7682 | 0.5363 | 0.4152 | 0.1669 | 0.5034 | 0.1821 | 0.5343 |

---

## Prediction Instability & Distribution Shift

Summary of scenarios producing the largest individual-level prediction shifts (ranked by 95th percentile absolute probability delta):

| Model | Scenario ID | Scenario Family | Mean Absolute Prob Delta | 95th Percentile Prob Delta |
| :--- | :--- | :--- | :---: | :---: |
| Hybrid (HistGradientBoosting) | `joint_missingness_rate_1.00` | `joint_missingness` | 0.1821 | **0.5343** |
| Hybrid (HistGradientBoosting) | `joint_missingness_rate_0.75` | `joint_missingness` | 0.1379 | **0.5009** |
| Hybrid (traditional + alternative) | `joint_missingness_rate_1.00` | `joint_missingness` | 0.1894 | **0.4778** |
| Hybrid (HistGradientBoosting) | `joint_missingness_rate_0.50` | `joint_missingness` | 0.0925 | **0.4626** |
| Hybrid (traditional + alternative) | `joint_missingness_rate_0.75` | `joint_missingness` | 0.1424 | **0.4429** |
| Hybrid (HistGradientBoosting) | `noise_utility_payment_timeliness_std_20pct_range` | `measurement_noise` | 0.0895 | **0.4017** |
| Hybrid (traditional + alternative) | `joint_missingness_rate_0.50` | `joint_missingness` | 0.0952 | **0.3943** |
| Hybrid (traditional + alternative) | `noise_utility_payment_timeliness_std_20pct_range` | `measurement_noise` | 0.0820 | **0.3259** |
| Hybrid (HistGradientBoosting) | `joint_missingness_rate_0.25` | `joint_missingness` | 0.0430 | **0.3104** |
| Hybrid (HistGradientBoosting) | `missingness_utility_payment_timeliness_rate_1.00` | `individual_missingness` | 0.0647 | **0.2899** |

---

## Apparent Inverted Deltas and Regularization Effects

In a small number of evaluations (23 out of 735, or 3.1%), direction-adjusted degradation was slightly negative,
indicating that the scenario metric appeared marginally better than clean baseline. The table below lists the most prominent instances:

| Model | Scenario ID | Metric | Metric Direction | Clean Metric | Scenario Metric | Direction-Adjusted Degradation |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Hybrid (HistGradientBoosting) | `missingness_digital_payment_regularity_rate_1.00` | `ece_10_bins` | lower | 0.0180 | 0.0138 | **-0.0043** |
| Hybrid (HistGradientBoosting) | `noise_alternative_history_months_std_1_months` | `ece_10_bins` | lower | 0.0180 | 0.0152 | **-0.0028** |
| Hybrid (HistGradientBoosting) | `missingness_alternative_history_months_rate_0.50` | `ece_10_bins` | lower | 0.0180 | 0.0152 | **-0.0028** |
| Hybrid (HistGradientBoosting) | `noise_alternative_history_months_std_3_months` | `ks_statistic` | higher | 0.6680 | 0.6705 | **-0.0025** |
| Hybrid (HistGradientBoosting) | `noise_digital_payment_regularity_std_10pct_range` | `average_precision` | higher | 0.7968 | 0.7983 | **-0.0015** |
| Hybrid (HistGradientBoosting) | `missingness_alternative_history_months_rate_0.25` | `average_precision` | higher | 0.7968 | 0.7982 | **-0.0015** |
| Hybrid (HistGradientBoosting) | `missingness_rental_payment_timeliness_rate_0.50` | `ece_10_bins` | lower | 0.0180 | 0.0166 | **-0.0014** |
| Hybrid (HistGradientBoosting) | `missingness_alternative_history_months_rate_1.00` | `ece_10_bins` | lower | 0.0180 | 0.0169 | **-0.0012** |

> [!NOTE]
> **Why do these inverted deltas occur?**
> 1. **Shrinkage Toward Central Tendency**: Imputing missing values with medians dampens extreme predicted probabilities,
>    slightly reducing Expected Calibration Error (ECE) where clean predictions were slightly overconfident in extreme bins.
> 2. **Boundary Perturbation**: Small amounts of continuous noise occasionally break marginal score ties in favor of true positives,
>    producing minor non-significant fluctuations (< 0.003) in KS statistic or Average Precision.

---

## Limitations & Boundary Conditions

1. **Label-Preserving Input Corruption**: Ground-truth target labels are preserved while inputs are corrupted.
   This does not simulate real-world covariate shift where the true underlying default hazard changes.
2. **Synthetic Data Support**: Degradation rates and noise scales are defined with respect to synthetic domain bounds.
   They do not represent empirically measured sensor noise or real-world banking API outage frequencies.
3. **No Retraining**: All models remained fixed to assess direct runtime vulnerability.

---

## Reproduction Command

```bash
.venv/bin/python stress_testing.py --seed 2028
```

*Note on Reproducibility:* Repeated executions with identical configuration produce byte-identical `scenario_metrics.csv`,
`scenario_deltas.csv`, and `stress_test_report.md`. The metadata file `stress_test_metadata.json` is identical across runs
except for `generated_at_utc`, which intentionally records the runtime execution timestamp for provenance tracking.
