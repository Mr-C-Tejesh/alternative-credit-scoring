# Alternative Credit Scoring Using Non-Traditional Data — Research Prototype

[![CI Tests](https://github.com/Mr-C-Tejesh/alternative-credit-scoring/actions/workflows/tests.yml/badge.svg)](https://github.com/Mr-C-Tejesh/alternative-credit-scoring/actions/workflows/tests.yml)

> [!WARNING]
> **SIMULATED RESEARCH DEMONSTRATION ONLY**
> This project is trained and evaluated strictly on synthetic benchmark data generated locally. It is an educational and engineering demonstration, **not** empirical evidence that these features predict real-world borrower creditworthiness. Do **not** use these models, illustrative scores, risk bands, or explanations for actual lending decisions, credit pricing, underwriting, or statutory adverse-action notices. Never enter real personal or financial data into the demo.

---

## 1. Project Overview

This repository demonstrates the end-to-end architecture of an alternative credit scoring pipeline, comparing a traditional credit bureau baseline against hybrid models that incorporate non-traditional behavioural data (utility payments, rental timeliness, digital payment regularity, and cash-flow stability).

The implementation includes:
- **Synthetic Data Generation & Data Contracts:** 10,000 deterministic records generated via `train.py` (seed `2026`) with explicit schema and domain bounds enforced by `validation.py`.
- **Baseline vs. Hybrid Pipeline Comparison:** Preprocessed inside scikit-learn pipelines (median imputation with missingness indicators, standard scaling) evaluating:
  1. *Traditional Logistic Regression Baseline* (traditional financial features only).
  2. *Hybrid Logistic Regression* (traditional + alternative financial behavior).
  3. *Hybrid HistGradientBoosting* (non-linear gradient-boosted benchmark).
- **Credit-Risk Evaluation Metrics:** ROC-AUC, Gini coefficient ($2 \times \text{ROC\_AUC} - 1$), Kolmogorov-Smirnov (KS) statistic, PR-AUC, Brier score, log loss, and Expected Calibration Error (ECE, 10 uniform bins).
- **Out-of-Fold Threshold Analysis:** Empirical classification sensitivity ($\tau \in [0.05, 0.95]$) evaluated strictly on 5-fold cross-validated out-of-fold predictions on the **training partition** (8,000 samples) to prevent data snooping leakage.
- **Illustrative PDO Credit Score Transformation:** Converts predicted default probabilities into an intuitive $300\text{--}850$ credit score using Points to Double the Odds (PDO) methodology.
- **Exact Local Additive Explanations:** Decomposes hybrid Logistic Regression predictions into exact feature contributions in log-odds space ($\Delta \text{logit}$).
- **Interactive Streamlit Demo & CLI:** End-to-end inference interfaces with runtime input contract enforcement and missingness simulation for thin-file / unbanked borrowers.

---

## 2. Environment & System Requirements

- **Python Runtime:** Python `3.10` to `3.13`. Tested and verified locally on macOS with Python `3.13.7` and automatically executed in CI with Python `3.12` on `ubuntu-latest`.
- **Dependency Constraints (`requirements.txt`):** The repository declares compatible version ranges (constraints) rather than exact pins:
  - `numpy>=2.0`
  - `pandas>=2.0`
  - `scikit-learn>=1.4,<1.8` *(trained on 1.7.x; includes a defensive deserialization shim in `inference_helper.load_saved_model()`)*
  - `matplotlib>=3.8`
  - `joblib>=1.3`
  - `streamlit>=1.35`
- **Continuous Integration Status:** Automated testing is configured in `.github/workflows/tests.yml` for pushes and pull requests to `main` on Python `3.12` (`ubuntu-latest`). Remote execution for the verified base commit (`c090464`) was confirmed passing (`success`).

---

## 3. Quick Start & Setup

### A. Clone and Create Virtual Environment

```bash
git clone https://github.com/Mr-C-Tejesh/alternative-credit-scoring.git
cd alternative-credit-scoring

# Create virtual environment
python -m venv .venv

# Activate environment:
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B. Run Automated Unit Tests

The test suite contains 151 unit and integration tests covering data validation, metric invariants, threshold sensitivity, PDO scoring, additive explainability, CLI smoke tests, and Streamlit `AppTest` execution:

```bash
python -m unittest discover -s tests -v
```

### C. Run the Interactive Streamlit Web App

Launch the local interactive application:

```bash
streamlit run app.py
```

The web interface enables:
- Adjusting applicant inputs within validated synthetic domain bounds.
- Toggling formal bureau record availability (`bureau_score = NaN`).
- Simulating unbanked or cold-start scenarios via multiselect controls for unavailable alternative data.
- Viewing estimated 12-month default probability, illustrative credit score, risk band, and top risk-increasing/decreasing factors with full decomposition tables.

### D. Run the CLI Prediction Demonstration

Run sample predictions via the command line:

```bash
# Standard applicant profile
python predict.py

# Simulate an applicant with no bureau history (thin-file / unbanked)
python predict.py --no-bureau --income 35000 --dti 0.45 --delinquencies 1

# Custom applicant inputs
python predict.py --age 40 --income 65000 --loans 2 --utilization 0.30
```

### E. (Optional) Re-train and Regenerate Artifacts

To regenerate the synthetic dataset, refit all three models, and reproduce all metrics/reports:

```bash
python train.py
```

---

## 4. Feature Groups & Supported Domain Bounds

The operational boundaries below reflect the synthetic generator distribution and the input validation contract in `validation.py`. They are **model support bounds for this synthetic benchmark**, not universal borrower limits or statutory lending rules:

| Feature Name | Type | Supported Range | Missingness Policy |
| :--- | :--- | :--- | :--- |
| `age_years` | integer | $[20, 65]$ | Prohibited |
| `monthly_income_inr` | float | $[8000.0, 250000.0]$ | Prohibited |
| `employment_tenure_years` | float | $[0.0, 35.0]$ | Prohibited |
| `debt_to_income_ratio` | float | $[0.0, 1.20]$ | Prohibited |
| `existing_loan_count` | integer | $[0, 8]$ | Prohibited |
| `credit_utilization_ratio` | float | $[0.0, 1.25]$ | Prohibited |
| `prior_delinquencies_24m` | integer | $[0, 5]$ | Prohibited |
| `bureau_history_available` | binary | $\{0, 1\}$ | Prohibited |
| `bureau_score` | float | $[300.0, 900.0]$ | Mandatory `NaN` when `bureau_history_available == 0` |
| `utility_payment_timeliness` | float | $[0.05, 1.00]$ | Permitted missing (`NaN`) |
| `rental_payment_timeliness` | float | $[0.05, 1.00]$ | Permitted missing (`NaN`) |
| `digital_payment_regularity` | float | $[0.02, 1.00]$ | Permitted missing (`NaN`) |
| `cashflow_stability_index` | float | $[0.02, 1.00]$ | Permitted missing (`NaN`) |
| `alternative_history_months` | integer | $[0, 24]$ | Permitted missing (`NaN` at inference) |

---

## 5. Model Outputs & Mathematical Methodology

### A. Estimated Default Probability ($p$)
The hybrid Logistic Regression model outputs an estimated probability of default within 12 months ($p \in [0, 1]$). Because this model was trained on simulated data, $p$ reflects synthetic patterns, not true borrower risk.

### B. Illustrative PDO Credit Score
To provide an intuitive scale familiar to credit professionals, default probabilities are mapped to a traditional $300\text{--}850$ scale using Points to Double the Odds (PDO):
$$\text{Score} = \text{base\_score} + \left(\frac{\text{PDO}}{\ln 2}\right) \ln\left(\frac{\frac{1-p}{p}}{\text{base\_odds}}\right)$$
- **Base Score:** 600 at 50:1 good-to-bad odds ($p \approx 1.96\%$).
- **PDO:** 20 points (doubling the good-to-bad odds adds 20 points; halving odds subtracts 20 points).
- **Scale Bounds:** $[300, 850]$ (clamped at extremes).
- *Notice:* The score is an educational monotonic transformation of probability. Integer score thresholds should not be substituted for probability band cutoffs.

### C. Illustrative Risk Bands
- **Lower model-estimated risk:** $p < 0.10$
- **Middle model-estimated risk:** $0.10 \le p < 0.20$
- **Higher model-estimated risk:** $p \ge 0.20$
- *Notice:* These bands do not generate approve/reject decisions. A lower modeled probability does not guarantee repayment.

### D. Exact Local Feature Explanations
For the hybrid Logistic Regression model, predictions decompose additively into log-odds space:
$$\text{logit}(p) = \text{intercept} + \sum_{j=1}^M (\text{coef}_j \cdot x'_j)$$
- **Positive contributions (+):** Increase the model's log-odds of synthetic default.
- **Negative contributions (-):** Reduce the model's log-odds of synthetic default.
- **Mathematical Nature:** Feature contributions describe the mathematical mechanics of the fitted linear decision function on standardized inputs. They are **not** real-world causal drivers and are **not** legally compliant adverse-action reason codes under credit regulations (e.g., US ECOA/FCRA or Indian DPDP/RBI guidelines).

---

## 6. Repository Architecture

```
alternative-credit-scoring/
├── .github/
│   └── workflows/
│       └── tests.yml            # CI test runner (GitHub Actions on push/PR to main)
├── data/
│   └── synthetic_applicants.csv # 10,000 generated benchmark records
├── models/
│   ├── hybrid_hist_gradient_boosting.joblib
│   ├── hybrid_logistic_regression.joblib
│   └── traditional_baseline.joblib
├── reports/
│   ├── calibration_curves.png
│   ├── metric_comparison.png
│   ├── permutation_importance.png
│   ├── threshold_analysis.png
│   ├── test_metrics.csv
│   ├── threshold_analysis.csv
│   └── experiment_summary.md    # Detailed methodology and audit report
├── tests/
│   └── test_pipeline.py         # 151 comprehensive automated unit and AppTest tests
├── app.py                       # Streamlit web demonstration
├── predict.py                   # Command-line prediction and explanation demo
├── train.py                     # Synthetic data generation and model training pipeline
├── validation.py                # Schema contracts and domain boundary validation
├── risk_scoring.py              # PDO score transformation and risk band mapping
├── explainability.py            # Exact additive log-odds decomposition for hybrid LR
├── inference_helper.py          # Shared applicant row builder and inference boundary
├── requirements.txt             # Verified runtime dependencies
├── MODEL_CARD.md                # Comprehensive model card and technical governance
├── DEPLOYMENT.md               # Future deployment and environment readiness notes
└── README.md                    # Project documentation
```

---

## 7. Limitations & Governance Notice

1. **Simulation Benchmark Only:** Features and target outcomes are artificially constructed by mathematical formulas.
2. **No Real Lending Authority:** Outputs must not be used to approve, reject, price, or rank real applicants.
3. **No Causal Interpretation:** Feature weights reflect predictive associations within the simulated data generator, not causal levers for borrower intervention.
4. **Mandatory Audits for Real-World Adaptation:** Deploying any alternative credit model requires an authorized real dataset, provable data lineage, explicit consumer consent, and thorough disparate-impact / algorithmic fairness audits under applicable financial regulations.
