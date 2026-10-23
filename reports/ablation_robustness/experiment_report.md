# Controlled Feature Ablation & Multi-Seed Robustness Study

> [!WARNING]
> **SIMULATION BENCHMARK ONLY — NOT REAL CREDIT RISK EVIDENCE**
> This study was performed exclusively on synthetic benchmark records generated via mathematical formulas.
> In the generator (`train.py`), default probability is explicitly constructed using both traditional and
> alternative behavioral features. The observed performance degradation when removing alternative features is
> therefore **expected by construction** and demonstrates the mathematical mechanics of feature ablation.
> It does **not** provide empirical evidence that non-traditional financial data predicts real-world loan default,
> and these findings must never be cited as statutory justification for underwriting decisions.

---

## 1. Study Objective & Research Questions

This experiment evaluates the statistical robustness and sensitivity of the hybrid credit scoring pipeline
across **20 independent synthetic simulation datasets** (seeds 2026 through 2045), addressing three core questions:
1. **Seed-to-Seed Variability:** How much do discrimination and calibration metrics fluctuate due to random sampling?
2. **Controlled Feature Ablation:** What is the marginal performance degradation when removing one alternative feature at a time?
3. **Ranking Stability:** Does the relative importance of alternative features remain consistent across independent synthetic generations?

---

## 2. Experimental Methodology & Leakage Prevention

- **Simulation Seeds:** 20 independent seeds (`2026` to `2045`).
- **Sample Size:** 10,000 synthetic applicant records per seed.
- **Partitioning:** Stratified 80/20 train/test split per seed (8,000 train samples, 2,000 held-out test samples).
- **Strict Leakage Prevention:** Preprocessing transformers (median imputer and standard scaler) were fitted strictly on the training partition for each repetition. Test rows were transformed using training parameters with zero data snooping.
- **Paired Evaluation:** All 7 model scenarios per seed were trained on identical training rows and evaluated on the exact same held-out test rows.
- **Subtraction Direction:** Paired differences are defined as:
  $$\Delta \text{Metric} = \text{Metric}_{\text{scenario}} - \text{Metric}_{\text{full\_hybrid}}$$
  - For higher-is-better metrics (ROC-AUC, Gini, KS, PR-AUC): $\Delta < 0$ indicates performance degradation relative to the full hybrid model.
  - For lower-is-better metrics (Brier score, log loss, ECE): $\Delta > 0$ indicates error/calibration degradation relative to the full hybrid model.

### 2.1 Practical Negligible-Delta Classification Conventions

To prevent minor floating-point fluctuations around zero from being misclassified as meaningful degradation, paired deltas ($\Delta \text{Metric} = \text{Metric}_{\text{ablated}} - \text{Metric}_{\text{full\_hybrid}}$) are classified into three mutually exclusive categories using explicit metric-specific absolute tolerances:

| Metric | Negligible Tolerance ($\tau$) | Classification Criteria |
| :--- | :---: | :--- |
| `roc_auc` | 0.001 | Negative: $\Delta < -0.001$ · Negligible: $|\Delta| \le 0.001$ · Positive: $\Delta > 0.001$ |
| `gini_coefficient` | 0.002 | Negative: $\Delta < -0.002$ · Negligible: $|\Delta| \le 0.002$ · Positive: $\Delta > 0.002$ |
| `ks_statistic` | 0.001 | Negative: $\Delta < -0.001$ · Negligible: $|\Delta| \le 0.001$ · Positive: $\Delta > 0.001$ |
| `average_precision` | 0.001 | Negative: $\Delta < -0.001$ · Negligible: $|\Delta| \le 0.001$ · Positive: $\Delta > 0.001$ |
| `brier_score` | 0.001 | Negative: $\Delta < -0.001$ · Negligible: $|\Delta| \le 0.001$ · Positive: $\Delta > 0.001$ |
| `log_loss` | 0.001 | Negative: $\Delta < -0.001$ · Negligible: $|\Delta| \le 0.001$ · Positive: $\Delta > 0.001$ |
| `ece_10_bins` | 0.001 | Negative: $\Delta < -0.001$ · Negligible: $|\Delta| \le 0.001$ · Positive: $\Delta > 0.001$ |

> [!NOTE]
> **Reporting Convention Notice:** These thresholds are practical reporting conventions to distinguish observable performance drops from numerical noise. They are **not** formal hypothesis tests, p-values, confidence intervals, or evidence of real-world underwriting relevance.

---

## 3. Absolute Model Performance Across 20 Seeds

Descriptive summary statistics of held-out test performance across the 20 simulation seeds:

| Model Scenario | Mean ROC-AUC (± Std) | Median ROC-AUC | Mean KS Stat (± Std) | Mean PR-AUC (± Std) | Mean Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full hybrid** | 0.9031 (±0.0068) | 0.9025 | 0.6471 (±0.0121) | 0.8045 (±0.0133) | 0.1074 |
| **Hybrid without alternative history months** | 0.9016 (±0.0072) | 0.9005 | 0.6435 (±0.0164) | 0.8016 (±0.0136) | 0.1084 |
| **Hybrid without rental payment timeliness** | 0.8944 (±0.0081) | 0.8933 | 0.6288 (±0.0169) | 0.7877 (±0.0148) | 0.1122 |
| **Hybrid without utility payment timeliness** | 0.8854 (±0.0071) | 0.8834 | 0.6124 (±0.0188) | 0.7679 (±0.0131) | 0.1174 |
| **Hybrid without digital payment regularity** | 0.8929 (±0.0071) | 0.8939 | 0.6262 (±0.0153) | 0.7839 (±0.0127) | 0.1132 |
| **Hybrid without cash-flow stability index** | 0.8876 (±0.0071) | 0.8874 | 0.6169 (±0.0149) | 0.7766 (±0.0134) | 0.1159 |
| **Traditional baseline** | 0.7899 (±0.0139) | 0.7921 | 0.4398 (±0.0249) | 0.6004 (±0.0199) | 0.1564 |

---

## 4. Paired Feature Ablation Impact (Relative to Full Hybrid)

Marginal paired differences evaluated on the identical held-out test set for each of the 20 seeds:

| Ablated Feature Scenario | Mean Δ ROC-AUC (± Std) | Median Δ ROC-AUC | Negative (Δ < -0.001) | Negligible (|Δ| ≤ 0.001) | Positive (Δ > 0.001) | Mean Δ PR-AUC | Mean Δ Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid without alternative history months** | -0.0015 (±0.0013) | -0.0017 | 70% (14/20) | 20% (4/20) | 10% (2/20) | -0.0029 | +0.0009 |
| **Hybrid without rental payment timeliness** | -0.0086 (±0.0023) | -0.0085 | 100% (20/20) | 0% (0/20) | 0% (0/20) | -0.0169 | +0.0048 |
| **Hybrid without utility payment timeliness** | -0.0177 (±0.0035) | -0.0175 | 100% (20/20) | 0% (0/20) | 0% (0/20) | -0.0366 | +0.0100 |
| **Hybrid without digital payment regularity** | -0.0102 (±0.0033) | -0.0116 | 100% (20/20) | 0% (0/20) | 0% (0/20) | -0.0206 | +0.0058 |
| **Hybrid without cash-flow stability index** | -0.0155 (±0.0030) | -0.0160 | 100% (20/20) | 0% (0/20) | 0% (0/20) | -0.0280 | +0.0084 |
| **Traditional baseline** | -0.1132 (±0.0125) | -0.1110 | 100% (20/20) | 0% (0/20) | 0% (0/20) | -0.2042 | +0.0490 |

*Note: Directional categories are mutually exclusive and sum to 100% across seeds using the practical tolerance $|\Delta \text{ROC-AUC}| \le 0.001$.*

### 4.1 Feature Ranking Distribution Across Seeds (Ranks 1 to 5)

Single-feature ablations ranked by held-out $\Delta \text{ROC-AUC}$ ascending (rank 1 = most negative delta, representing the largest single-feature performance drop). `traditional_baseline` is excluded because it removes all alternative features rather than one:

| Feature Omitted | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Mean Rank | Min–Max Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **utility_payment_timeliness** | 15 (75%) | 4 (20%) | 0 (0%) | 1 (5%) | 0 (0%) | 1.35 | 1–4 |
| **cashflow_stability_index** | 5 (25%) | 12 (60%) | 3 (15%) | 0 (0%) | 0 (0%) | 1.90 | 1–3 |
| **digital_payment_regularity** | 0 (0%) | 4 (20%) | 10 (50%) | 6 (30%) | 0 (0%) | 3.10 | 2–4 |
| **rental_payment_timeliness** | 0 (0%) | 0 (0%) | 7 (35%) | 13 (65%) | 0 (0%) | 3.65 | 3–4 |
| **alternative_history_months** | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 20 (100%) | 5.00 | 5–5 |

- **Complete Hierarchy Frequency:** The exact complete ordering (`utility > cashflow > digital > rental > alt_history_months`) occurred in only **7 of 20 seeds (35.0%)**.
- **Sampling Variance & Rank Fluctuations:** While utility payment timeliness and cash-flow stability generally demonstrate the largest effects (together accounting for 100% of Rank 1 outcomes), relative rankings vary across simulation seeds due to finite-sample variability.
- **Directional Consistency vs. Relative Ranking:** A feature having a consistently negative paired delta across 100% of seeds (which holds for utility timeliness, cash-flow stability, digital regularity, and rental timeliness) confirms that omitting the feature reliably reduces model discrimination under this synthetic process. However, this does **not** mean the relative importance hierarchy among these features is rigid or deterministic across different datasets.

---

## 5. Key Findings & Robustness Observations

1. **Dominant Features vs. Ranking Permutations:**
   - Removing **utility payment timeliness** produced the single largest performance drop in 15 of 20 seeds (mean rank 1.35), while **cash-flow stability** produced the largest drop in 5 seeds and second-largest in 12 seeds (mean rank 1.90).
   - Although the Full Hybrid model outperformed the Traditional Baseline across 100% of simulation seeds, the exact complete five-feature hierarchy occurred in only **7 of 20 seeds (35.0%)**. Finite sampling variance across independent synthetic draws causes ranking permutations between closely competitive signals (e.g., utility vs. cashflow for rank 1; digital vs. rental for rank 3).
2. **Descriptive Sample Variability:**
   - Across independent synthetic generations, the Full Hybrid ROC-AUC varied within a narrow band (std ≈ 0.007–0.010), demonstrating that model training is stable given this generative specification.
   - Note: The reported ranges represent descriptive sample spread across seeds, **not** formal population confidence intervals.
3. **Calibration Impact:**
   - Removing alternative behavioral indicators systematically increased test Brier score and log loss, indicating that omitted features impair both discrimination and probability reliability under the synthetic generating process.

---

## 6. Visual Artifacts

A multi-panel figure visualizing paired delta boxplots and absolute metric comparisons is saved at:
- `reports/ablation_robustness/feature_ablation_plot.png`

---

## 7. Limitations & Governance Constraints

1. **Mathematical Artefact:** In `train.py`, target default is explicitly calculated as a logistic function with non-zero weights on alternative features. The ablation drop is a direct consequence of this formula.
2. **No Real-World Generalization:** Real borrower data exhibits non-stationary correlations, complex missingness patterns, selection bias, and macroeconomic shifts not captured by this benchmark.
3. **Prohibition on Lending Use:** Neither these metrics nor the ablated models may be used for actual credit underwriting or regulatory adverse action notices.

---

## 8. Reproducibility

To reproduce this multi-seed study from scratch:

```bash
python experiments/alternative_feature_ablation.py
```

To run a fast test with custom seeds or sample counts:

```bash
python experiments/alternative_feature_ablation.py --n-seeds 3 --n-samples 1000
```
