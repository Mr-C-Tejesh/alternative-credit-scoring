# Model Card — Synthetic Credit-Risk Research Prototype

## Intended use
Software demonstration and student learning only: demonstrate data generation, preprocessing, baseline comparison, probability prediction, metrics, threshold sensitivity analysis, illustrative score transformation, and feature-importance/explainability tooling.

## Prohibited use
Do not use to approve, reject, price, rank or otherwise make decisions about real people or actual lending applications. Do not enter real financial or personal data.

## Data
10,000 simulated records generated locally by `train.py` using seed 2026. No real borrower records were used. Labels are sampled from a probability rule deliberately defined in the same generator. That rule explicitly uses both traditional and alternative feature examples, so a lift in hybrid performance is partly expected by construction.

## Features & Supported Domain
- **Traditional/application (9 predictors):**
  - `age_years` $\in [21, 65]$
  - `monthly_income_inr` $\in [10000, 200000]$
  - `employment_tenure_years` $\in [0.0, 40.0]$
  - `debt_to_income_ratio` $\in [0.0, 1.2]$
  - `existing_loan_count` $\in [0, 8]$
  - `credit_utilization_ratio` $\in [0.0, 1.2]$
  - `prior_delinquencies_24m` $\in [0, 6]$
  - `bureau_history_available` $\in \{0, 1\}$
  - `bureau_score` $\in [300, 900]$ (mandatory NaN when `bureau_history_available == 0`)
- **Synthetic alternative features (5 predictors):**
  - `utility_payment_timeliness` $\in [0.05, 1.0]$
  - `rental_payment_timeliness` $\in [0.05, 1.0]$
  - `digital_payment_regularity` $\in [0.05, 1.0]$
  - `cashflow_stability_index` $\in [0.05, 1.0]$
  - `alternative_history_months` $\in [6, 24]$

## Models
- Logistic Regression on traditional/application features.
- Logistic Regression on traditional plus alternative examples.
- HistGradientBoosting on the combined synthetic features.

## Validation, Metrics & Threshold Analysis
- Fixed random seed (2026); stratified 80/20 train/test split (8,000 train / 2,000 test).
- Preprocessing pipelines (median imputation, standard scaling) are fitted strictly on training data.
- Metrics evaluated: ROC-AUC, Gini coefficient, KS statistic, average precision (PR-AUC), Brier score, log loss, and Expected Calibration Error (ECE, 10 uniform bins).
- **Out-of-Fold Threshold Analysis:** Decision threshold evaluation ($\tau \in [0.05, 0.95]$) is performed strictly on 5-fold cross-validated out-of-fold predictions on the **training partition** (8,000 samples). The held-out test partition is reserved strictly for generalization evaluation and was **not** used for threshold selection, preventing data snooping leakage.

## Illustrative Score Transformation & Risk Bands
To provide an intuitive interface familiar to credit professionals, default probabilities are mapped to a traditional 300–850 scoring scale using the industry-standard Points to Double the Odds (PDO) formulation:
$$\text{Score} = \text{Offset} - \frac{\text{PDO}}{\ln 2} \ln\left(\frac{p}{1-p}\right)$$
- **Base Score:** 600 at 50:1 good-to-bad odds ($p \approx 1.96\%$)
- **PDO:** 20 points
- **Scale Bounds:** $[300, 850]$ (clamped at extremes)
- **Illustrative Model Risk Bands:**
  - *Lower model-estimated risk:* $p < 0.10$ ($\text{Score} \ge 552$)
  - *Middle model-estimated risk:* $0.10 \le p < 0.20$ ($510 \le \text{Score} < 552$)
  - *Higher model-estimated risk:* $p \ge 0.20$ ($\text{Score} < 510$)
- **Non-Regulatory Notice:** Scores and risk bands are mathematical transformations of model probabilities for educational demonstration only. They are not credit bureau scores and convey no underwriting authority.

## Local Explainability
For the hybrid Logistic Regression model, predictions provide exact additive local feature contributions decomposed in log-odds space:
$$\text{decision\_function}(\mathbf{x}') = \beta_0 + \sum_{j=1}^M \beta_j x'_j$$
The sum of local feature contributions and global intercept reconstructs the model decision function to within machine precision ($< 10^{-14}$), enabling transparent identification of the primary risk-increasing and risk-decreasing factors for any applicant.

## Limitations
- Simulated population, features, missingness and target generation.
- Synthetic no-bureau-history group is not a validated representation of credit-invisible borrowers.
- No external validation, real-world calibration, drift testing or credible financial-cost model.
- Feature importance and local feature contributions reflect predictive associations under the simulation rule, not causal impact.
- No fairness conclusions should be drawn from the current simulation.
- Thresholds and risk bands are not validated operational underwriting policies.

## Revalidation required
Use a properly authorised real dataset; document target horizon and data provenance; define feature availability at decision time; evaluate chronologically when possible; check calibration and error costs; perform justified privacy and fairness audits; and independently review the deployment context before any real-world use.
