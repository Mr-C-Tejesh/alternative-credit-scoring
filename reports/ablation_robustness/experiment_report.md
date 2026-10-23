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

| Ablated Feature Scenario | Mean Δ ROC-AUC (± Std) | Median Δ ROC-AUC | Seeds Δ < 0 | Mean Δ PR-AUC | Mean Δ Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid without alternative history months** | -0.0015 (±0.0013) | -0.0017 | 90% (18/20) | -0.0029 | +0.0009 |
| **Hybrid without rental payment timeliness** | -0.0086 (±0.0023) | -0.0085 | 100% (20/20) | -0.0169 | +0.0048 |
| **Hybrid without utility payment timeliness** | -0.0177 (±0.0035) | -0.0175 | 100% (20/20) | -0.0366 | +0.0100 |
| **Hybrid without digital payment regularity** | -0.0102 (±0.0033) | -0.0116 | 100% (20/20) | -0.0206 | +0.0058 |
| **Hybrid without cash-flow stability index** | -0.0155 (±0.0030) | -0.0160 | 100% (20/20) | -0.0280 | +0.0084 |
| **Traditional baseline** | -0.1132 (±0.0125) | -0.1110 | 100% (20/20) | -0.2042 | +0.0490 |

---

## 5. Key Findings & Robustness Observations

1. **Consistent Feature Hierarchy Across Seeds:**
   - Removing **utility payment timeliness** or **cash-flow stability** produced the largest and most consistent drops in discrimination across all 20 seeds.
   - 100% of simulation repetitions confirmed that the Full Hybrid model outperformed the Traditional Baseline in ROC-AUC and KS statistic.
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
