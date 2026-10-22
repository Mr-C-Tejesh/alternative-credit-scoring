# Experiment Summary — Synthetic Alternative Credit Scoring

**Experiment ID:** SYN-BASELINE-002
**Status:** Completed as an engineering demonstration only
**Dataset:** 10,000 generated synthetic records (`data/synthetic_applicants.csv`)
**Seed:** 2026
**Split:** Stratified random 80/20 train/test split
**Target:** `default_within_12_months` (synthetic binary outcome)
**Calibration binning:** Uniform probability partitioning, 10 bins

---

## 1. Held-out test evaluation metrics

The metrics below were evaluated on the 2,000 held-out synthetic test records (`test_metrics.csv`).

| Model | ROC-AUC ↑ | Gini ↑ | KS stat ↑ | Avg Precision ↑ | Brier score ↓ | Log loss ↓ | ECE (10 bins) ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Traditional baseline** | 0.8071 | 0.6142 | 0.4569 | 0.6100 | 0.1519 | 0.4610 | 0.0212 |
| **Hybrid Logistic Regression** | 0.9098 | 0.8196 | 0.6667 | 0.8126 | 0.1033 | 0.3298 | 0.0111 |
| **Hybrid HistGradientBoosting** | 0.9063 | 0.8126 | 0.6680 | 0.7968 | 0.1060 | 0.3386 | 0.0180 |

*Legend: ↑ Higher score is preferable (better discrimination/ranking); ↓ Lower score is preferable (better probability accuracy/calibration).*

---

## 2. No-bureau-history subgroup metrics

For the synthetic subgroup with no bureau history available (`bureau_history_available == 0`, representing thin-file applicants in the simulation; `no_bureau_history_subgroup_metrics.csv`):

| Model | ROC-AUC ↑ | Gini ↑ | KS stat ↑ | Avg Precision ↑ | Brier score ↓ | Log loss ↓ | ECE (10 bins) ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Traditional baseline** | 0.7698 | 0.5396 | 0.4095 | 0.5680 | 0.1643 | 0.4954 | 0.0240 |
| **Hybrid Logistic Regression** | 0.9109 | 0.8218 | 0.6668 | 0.8208 | 0.1041 | 0.3308 | 0.0222 |
| **Hybrid HistGradientBoosting** | 0.9017 | 0.8034 | 0.6543 | 0.7943 | 0.1097 | 0.3484 | 0.0285 |

> **Subgroup Interpretation Caution:**
> In this simulation, the subgroup represents ~46% of records by construction. However, in real-world credit risk assessment, thin-file or unbanked populations often represent smaller, non-randomly selected subgroups. Subgroup metrics must be evaluated with cautious sample-size consideration and confidence intervals, as small sample sizes can artificially inflate discrimination or introduce high variance.

---

## 3. Metric definitions and methodology

### Score orientation
In all models and evaluation routines, predictions represent the **modeled probability of borrower default within 12 months** (positive class $Y = 1$). A **higher predicted probability denotes higher credit risk**.

### Ranking & discrimination metrics (order-preserving)
1. **ROC-AUC (Receiver Operating Characteristic — Area Under Curve):**
   - *Measures:* The probability that a randomly selected defaulting borrower is assigned a higher risk probability than a randomly selected non-defaulting borrower.
   - *Direction:* Higher is better (0.5 = random chance, 1.0 = perfect ranking separation).
2. **Gini Coefficient (Somers' D):**
   - *Definition:* $\text{Gini} = 2 \times \text{ROC-AUC} - 1$.
   - *Measures:* Relative ranking discrimination power normalized between $-1.0$ and $+1.0$. In credit scoring practice, Gini expresses the area between the Lorenz/ROC curve and the diagonal line of equality.
   - *Direction:* Higher is better (0.0 = no discrimination beyond random ordering, 1.0 = perfect separation).
3. **Kolmogorov-Smirnov (KS) Statistic:**
   - *Definition:* $\text{KS} = \max_t | \text{TPR}(t) - \text{FPR}(t) |$, the maximum vertical distance between the cumulative default (True Positive Rate) and cumulative non-default (False Positive Rate) distributions across all possible probability thresholds $t$.
   - *Measures:* Maximum score separation capability between good and bad borrower distributions.
   - *Direction:* Higher is better (0.0 = distributions identical, 1.0 = distributions completely separated with no overlap).
4. **Average Precision (PR-AUC):**
   - *Measures:* Area under the Precision-Recall curve, calculating the weighted mean of precisions achieved at each threshold. Especially informative in imbalanced default datasets.
   - *Direction:* Higher is better (prevalence = baseline, 1.0 = perfect precision across all recall levels).

### Probability calibration & scoring rules (value-preserving)
5. **Brier Score:**
   - *Definition:* Mean squared error between predicted probabilities and binary outcomes: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
   - *Measures:* Strictly proper scoring rule quantifying both refinement and calibration.
   - *Direction:* Lower is better (0.0 = perfect probabilistic forecast).
6. **Log Loss (Cross-Entropy Loss):**
   - *Definition:* Negative log-likelihood of the true binary outcomes given predicted probabilities: $-\frac{1}{N} \sum_{i=1}^N [y_i \ln p_i + (1 - y_i) \ln(1 - p_i)]$.
   - *Measures:* Penalizes confident but erroneous predictions heavily.
   - *Direction:* Lower is better (0.0 = perfect probability assignments).
7. **Expected Calibration Error (ECE):**
   - *Definition:* Weighted average difference between predicted confidence (mean probability) and observed default frequency across $M = 10$ uniform probability bins:
     $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{observed\_frequency}(B_m) - \text{mean\_probability}(B_m) \right|$$
   - *Measures:* Calibration deviation / reliability. Bins with zero samples contribute 0 and do not yield NaN.
   - *Direction:* Lower is better (0.0 = perfectly calibrated probabilities).

---

## 4. Key methodological distinctions

### Discrimination vs. Calibration
- **Discrimination metrics (ROC-AUC, Gini, KS)** measure whether the model can rank high-risk applicants above low-risk applicants. Any strictly monotonic transformation of scores leaves ROC-AUC, Gini, and KS unchanged.
- **Calibration metrics (ECE, Brier score, log loss)** measure whether the numerical output accurately reflects real-world empirical default frequencies (e.g., whether applicants predicted at 10% risk default at a 10% rate). Monotonic transformations can severely degrade calibration without altering discrimination.
- Both discrimination and calibration are necessary for real-world risk underwriting: discrimination ensures correct risk ranking, while calibration ensures appropriate capital provisioning, expected loss estimation, and pricing.

### Dependence of ECE on Binning
ECE is sensitive to the number of bins ($M$) and binning strategy (uniform vs. quantile). A small number of bins can smooth over local miscalibrations, whereas a large number of bins creates sample-sparse bins with noisy frequency estimates. Therefore, ECE must always be reported alongside its bin configuration (here: `n_bins=10`, uniform interval $[0, 1]$) and supplemented by proper scoring rules (Brier score and log loss) that do not require discretization.

---

## 5. Visual artifacts generated

- `reports/metric_comparison.png`: Bar comparison of test discrimination metrics.
- `reports/roc_curves.png`: ROC curves for all three models with diagonal reference.
- `reports/precision_recall_curves.png`: Precision-recall curves displaying trade-offs across thresholds.
- `reports/calibration_curves.png`: Reliability diagrams plotting observed default rate against mean predicted probability across 10 uniform bins, including an ideal 45-degree calibration line.
- `reports/permutation_importance.png`: Feature importance assessing test AUC drop under permutation.

---

## 6. Interpretation limits & synthetic data disclaimer

1. **Synthetic data only:** The dataset was generated locally using seed 2026. The alternative-feature relationship with default was explicitly authored into the data-generation formula. The observed hybrid performance lift confirms pipeline execution and feature recovery in this synthetic scenario; it does **not** prove the empirical efficacy, validity, fairness, or legal permissibility of alternative credit scoring in real-world lending.
2. **Domain bounds:** Model validation adheres to explicit demo bounds (e.g., DTI $\in [0.0, 1.2]$, bureau score $\in [300, 900]$). Extrapolation to arbitrary borrower profiles requires deliberate, domain-specific out-of-distribution policies.
3. **No causal inference:** Permutation importance reflects statistical predictive association under the simulated data mechanism, not causal impact.
