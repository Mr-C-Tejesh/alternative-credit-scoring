# Experiment Summary — Synthetic Alternative Credit Scoring

- **Experiment ID:** SYN-BASELINE-003
- **Status:** Completed as an engineering demonstration only
- **Dataset:** 10,000 generated synthetic records (`data/synthetic_applicants.csv`)
- **Seed:** 2026
- **Split:** Stratified random 80/20 train/test split (8,000 train / 2,000 test)
- **Target:** `default_within_12_months` (synthetic binary outcome)
- **Calibration binning:** Uniform probability partitioning, 10 bins

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
In all models and evaluation routines, raw predictions represent the **modeled probability of borrower default within 12 months** (positive class $Y = 1$). A **higher predicted probability denotes higher credit risk**.

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

## 5. Illustrative credit scoring & risk bands

To provide an intuitive interface familiar to credit professionals, default probabilities are mapped to a traditional 300–850 scoring scale using the industry-standard Points to Double the Odds (PDO) formulation.

### Probability-to-Score Transformation
The transformation converts default probability $p$ into a credit score where **higher scores indicate lower default risk** (higher creditworthiness).
$$\text{Odds} = \frac{1 - p}{p} \quad (\text{good-to-bad odds})$$
$$\text{Factor} = \frac{\text{PDO}}{\ln(2)}$$
$$\text{Offset} = \text{Base Score} - \text{Factor} \cdot \ln(\text{Base Odds})$$
$$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1 - p}{p}\right) = \text{Offset} - \text{Factor} \cdot \ln\left(\frac{p}{1 - p}\right)$$

### Configuration Parameters
- **Base Score:** 600
- **Base Odds:** 50:1 good-to-bad ($p = \frac{1}{51} \approx 1.961\% \to \text{Score} = 600.0$)
- **Points to Double the Odds (PDO):** 20 (doubling good odds to 100:1 yields $+20$ points $\to 620$; halving good odds to 25:1 yields $-20$ points $\to 580$)
- **Score Bounds:** $[300, 850]$ with clamping at extreme probabilities ($p \to 0 \implies 850$, $p \to 1 \implies 300$).

### Illustrative Model Risk Bands
These bands are illustrative engineering demonstrations and do **not** represent statutory or regulatory credit ratings:
- **Lower model-estimated risk:** Modeled default probability $p < 0.10$
- **Middle model-estimated risk:** Modeled default probability $0.10 \le p < 0.20$
- **Higher model-estimated risk:** Modeled default probability $p \ge 0.20$

The illustrative credit score is a monotonic, bounded, and rounded transformation of modeled default probability (continuous raw scores at $p=0.10$ and $p=0.20$ evaluate to $\approx 550.52$ and $\approx 527.12$, respectively). Because display scores are subject to integer rounding and clamping to $[300, 850]$, scores should not be substituted for the continuous probability-band boundaries.

---

## 6. Out-of-fold threshold sensitivity analysis (Training Partition)

### Data Leakage Prevention Methodology
Evaluating classification decision thresholds on the final held-out test partition introduces optimistic bias and data snooping leakage. In this pipeline:
1. Decision thresholds ($\tau \in [0.05, 0.95]$ in increments of $0.05$) are evaluated **strictly on out-of-fold (OOF) predictions** generated across a 5-fold stratified cross-validation of the **training set (8,000 samples)**.
2. In each cross-validation fold, the preprocessing pipeline (imputation, standard scaling) is fitted solely on that fold's training slice.
3. The final 2,000 test records are strictly held out to evaluate final generalizability and are **never** used for threshold selection.

### Key Threshold Metrics (Out-of-Fold Training Partition)

The table below summarizes performance across five representative decision thresholds ($\tau \in \{0.10, 0.20, 0.30, 0.40, 0.50\}$) for all three models:

| Model | Threshold ($\tau$) | Precision | Recall (TPR) | False Positive Rate | False Negative Rate | Balanced Accuracy |
|:---|---:|---:|---:|---:|---:|---:|
| **Traditional baseline** | 0.10 | 0.3466 | 0.9600 | 0.6870 | 0.0400 | 0.6365 |
| Traditional baseline | 0.20 | 0.4252 | 0.8319 | 0.4268 | 0.1681 | 0.7025 |
| Traditional baseline | 0.30 | 0.5074 | 0.6815 | 0.2511 | 0.3185 | 0.7152 |
| Traditional baseline | 0.40 | 0.5818 | 0.5266 | 0.1436 | 0.4734 | 0.6915 |
| Traditional baseline | 0.50 | 0.6556 | 0.3848 | 0.0767 | 0.6152 | 0.6540 |
| **Hybrid Logistic Regression** | 0.10 | 0.4773 | 0.9464 | 0.3933 | 0.0536 | 0.7765 |
| Hybrid Logistic Regression | 0.20 | 0.5747 | 0.8805 | 0.2473 | 0.1195 | 0.8166 |
| Hybrid Logistic Regression | 0.30 | 0.6556 | 0.8087 | 0.1612 | 0.1913 | 0.8237 |
| Hybrid Logistic Regression | 0.40 | 0.7141 | 0.7388 | 0.1123 | 0.2612 | 0.8132 |
| Hybrid Logistic Regression | 0.50 | 0.7602 | 0.6611 | 0.0792 | 0.3389 | 0.7910 |
| **Hybrid HistGradientBoosting** | 0.10 | 0.4904 | 0.9196 | 0.3626 | 0.0804 | 0.7785 |
| Hybrid HistGradientBoosting | 0.20 | 0.5832 | 0.8537 | 0.2316 | 0.1463 | 0.8111 |
| Hybrid HistGradientBoosting | 0.30 | 0.6530 | 0.7874 | 0.1588 | 0.2126 | 0.8143 |
| Hybrid HistGradientBoosting | 0.40 | 0.7053 | 0.7297 | 0.1157 | 0.2703 | 0.8069 |
| Hybrid HistGradientBoosting | 0.50 | 0.7488 | 0.6488 | 0.0826 | 0.3512 | 0.7831 |

Full metrics across all 19 thresholds are exported in `reports/threshold_analysis.csv` and visualized in `reports/threshold_analysis.png`.

### Decision Threshold Trade-Offs & Classification Behavior
Threshold effects are evaluated purely as mathematical classification behavior on synthetic data, not as a validated lending policy:
- **Lower classification thresholds ($\tau \approx 0.10$):** Generally identify more synthetic defaults ($\text{Recall} > 94\%$), while also increasing false positives ($\text{FPR} = 36\% - 69\%$).
- **Higher classification thresholds ($\tau \approx 0.50$):** Reduce false positives ($\text{FPR} = 7\% - 8\%$), but miss a substantial portion of synthetic defaults ($\text{FNR} = 34\% - 62\%$).
- **Absence of Cost-Optimal Threshold:** Without empirical, institution-specific business-cost inputs and validated risk preferences, no cost-optimal classification threshold can be identified. In real-world credit risk operations, a threshold is chosen to minimize expected loss under an explicit cost matrix:
  $$\text{Expected Loss} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP}$$
  where $C_{\text{FN}}$ (the financial cost associated with an undetected default) and $C_{\text{FP}}$ (the cost or lost margin associated with an incorrectly flagged applicant) are unknown institutional parameters. The synthetic threshold analysis presented here illustrates model classification mechanics across thresholds; it does not recommend an approval, rejection, or credit policy cutoff.

---

## 7. Exact additive local feature explanations (Hybrid Logistic Regression)

For the hybrid Logistic Regression model, local predictions are decomposed into exact additive feature contributions in transformed-feature/log-odds space:

### Additive Log-Odds Decomposition
For an applicant's preprocessed feature vector $\mathbf{x}' \in \mathbb{R}^M$, the model's log-odds decision function is an exact linear combination of the model coefficients $\boldsymbol{\beta}$ and intercept $\beta_0$:
$$\text{decision\_function}(\mathbf{x}') = \beta_0 + \sum_{j=1}^M \beta_j x'_j$$
$$\text{contribution}_j = \beta_j x'_j$$
$$\text{Reconstructed Log-Odds} = \beta_0 + \sum_{j=1}^M \text{contribution}_j$$

### Properties & Verification
- **Numerical Reconstruction:** Tested across multiple applicants, the difference $|\text{Reconstructed} - \text{decision\_function}|$ is smaller than $10^{-14}$ (exact machine precision, well within the $10^{-5}$ tolerance).
- **Directional Interpretation:**
  - $\text{contribution}_j > 0$: Increases the model's log-odds of synthetic default (elevates default risk; decreases illustrative credit score).
  - $\text{contribution}_j < 0$: Decreases the model's log-odds of synthetic default (lowers default risk; increases illustrative credit score).
- **Model Mechanics vs. Causality:** These values represent the fitted linear model's mathematical behavior on normalized, imputed inputs, not causal effects.
- **Educational Scope:** These decompositions provide mathematical transparency into how the linear model formed its output on synthetic applicant data. They are educational explanations and do not constitute legally sufficient adverse-action reason codes or proof of regulatory compliance under credit regulations (e.g., US FCRA/ECOA or Indian DPDP/RBI guidelines).

---

## 8. Visual and tabular artifacts generated

- `reports/metric_comparison.png`: Bar comparison of test discrimination metrics.
- `reports/roc_curves.png`: ROC curves for all three models with diagonal reference.
- `reports/precision_recall_curves.png`: Precision-recall curves displaying trade-offs across thresholds.
- `reports/calibration_curves.png`: Reliability diagrams plotting observed default rate against mean predicted probability across 10 uniform bins, including an ideal 45-degree calibration line.
- `reports/permutation_importance.png`: Feature importance assessing test AUC drop under permutation.
- `reports/threshold_analysis.png`: 4-panel threshold evaluation grid showing Precision-Recall, Sensitivity-Specificity, Error Rates (FPR/FNR), and Balanced Accuracy vs. Decision Threshold.
- `reports/threshold_analysis.csv`: Complete tabular export of 19 thresholds $\times$ 3 models evaluated on training out-of-fold data.
- `reports/run_summary.json`: Machine-readable summary of dataset parameters, evaluated metrics, and threshold evaluation methodology.

---

## 9. Interpretation limits & synthetic data disclaimer

1. **Synthetic data only:** The dataset was generated locally using seed 2026. The alternative-feature relationship with default was explicitly authored into the data-generation formula. The observed hybrid performance lift confirms pipeline execution and feature recovery in this synthetic scenario; it does **not** prove the empirical efficacy, validity, fairness, or legal permissibility of alternative credit scoring in real-world lending.
2. **Domain bounds:** Model validation adheres to explicit demo bounds (e.g., DTI $\in [0.0, 1.2]$, bureau score $\in [300, 900]$). Extrapolation to arbitrary borrower profiles requires deliberate, domain-specific out-of-distribution policies.
3. **No causal inference:** Permutation importance and linear feature contributions reflect statistical predictive association under the simulated data mechanism, not causal impact.
4. **No underwriting authority:** The illustrative credit scores (300–850) and model risk bands are educational representations of model default probability. They do not convey credit approval, credit rejection, or credit bureau score authority.
